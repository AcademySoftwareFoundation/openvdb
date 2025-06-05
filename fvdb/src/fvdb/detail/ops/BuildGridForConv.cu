// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

nanovdb::GridHandle<TorchDeviceBuffer>
buildCoarseGridFromFineGridCPU(const GridBatchImpl &fineBatchHdl,
                               const nanovdb::Coord branchingFactor) {
    using GridType  = nanovdb::ValueOnIndex;
    using IndexTree = nanovdb::NanoTree<GridType>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &fineGridHdl = fineBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(fineGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < fineGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *fineGrid = fineGridHdl.template grid<GridType>(bidx);
        if (!fineGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &fineTree = fineGrid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(fineTree); it.isValid(); it++) {
            const nanovdb::Coord coarseIjk =
                (it->first.asVec3d() / branchingFactor.asVec3d()).floor();
            proxyGridAccessor.setValue(coarseIjk, 1.0f);
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        ret.buffer().to(torch::kCPU);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

__device__ void
convIjkForGridCallback(int32_t bidx,
                       int32_t lidx,
                       int32_t vidx,
                       int32_t cidx,
                       const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> batchAcc,
                       const nanovdb::Coord &kernelSize,
                       const nanovdb::Coord &stride,
                       int kernelVolume,
                       TorchRAcc32<int32_t, 2> outIJKData,
                       TorchRAcc32<fvdb::JIdxType, 1> outIJKBIdx,
                       TorchRAcc32<bool, 1> outMask) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    if (!leaf.isActive(vidx))
        return;

    const nanovdb::Coord &srcIjk = leaf.offsetToGlobalCoord(vidx);
    const int64_t index          = ((int64_t)leaf.getValue(vidx)) - 1;
    const int64_t baseOffset     = batchAcc.voxelOffset(bidx);

    int lower[3], upper[3];
    for (int i = 0; i < 3; ++i) {
        if (kernelSize[i] % 2 == 0) {
            lower[i] = 0;
            upper[i] = kernelSize[i] - 1;
        } else {
            lower[i] = -(kernelSize[i] - 1) / 2;
            upper[i] = (kernelSize[i] - 1) / 2;
        }
    }

    int64_t count = 0;
    for (int di = lower[0]; di <= upper[0]; di += 1) {
        for (int dj = lower[1]; dj <= upper[1]; dj += 1) {
            for (int dk = lower[2]; dk <= upper[2]; dk += 1, count += 1) {
                const nanovdb::Coord dstIjk = srcIjk + nanovdb::Coord(dk, dj, di);
                if (dstIjk[0] % stride[2] != 0 || dstIjk[1] % stride[1] != 0 ||
                    dstIjk[2] % stride[0] != 0)
                    continue;
                //  The original torchsparse implementation has a weird bug that checks the
                //  coordsMin. if (dstIjk[0] < coordsMin[0] || dstIjk[1] < coordsMin[1] || dstIjk[2]
                //  < coordsMin[2])
                //      continue;
                // if (dstIjk[0] > coordsMax[0] || dstIjk[1] > coordsMax[1] || dstIjk[2] >
                // coordsMax[2])
                //     continue;

                const int64_t base  = (baseOffset + index) * kernelVolume + count;
                outIJKData[base][0] = dstIjk[0] / stride[2];
                outIJKData[base][1] = dstIjk[1] / stride[1];
                outIJKData[base][2] = dstIjk[2] / stride[0];
                outIJKBIdx[base]    = bidx;
                outMask[base]       = true;
            }
        }
    }
}

JaggedTensor
convIJKForGrid(const GridBatchImpl &batchHdl,
               const nanovdb::Coord &kernelSize,
               const nanovdb::Coord &stride) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    if (stride == nanovdb::Coord(1) || stride == kernelSize) {
        return dispatchCoarseIJKForFineGrid<torch::kCUDA>(batchHdl, nanovdb::Coord(stride));
    }

    const int32_t kernelVolume = kernelSize.x() * kernelSize.y() * kernelSize.z();

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    const torch::TensorOptions optsMask =
        torch::TensorOptions().dtype(torch::kBool).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({batchHdl.totalVoxels() * kernelVolume, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * kernelVolume}, optsBIdx);
    torch::Tensor outMask    = torch::zeros({batchHdl.totalVoxels() * kernelVolume}, optsMask);

    // For each voxel in source grid, compute possible voxels in target grid that affect them
    auto outIJKAcc = outIJK.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor32<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();
    auto outMaskAcc = outMask.packed_accessor32<bool, 1, torch::RestrictPtrTraits>();

    auto cb = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchImpl::Accessor<nanovdb::ValueOnIndex> bacc) {
        convIjkForGridCallback(bidx,
                               lidx,
                               vidx,
                               cidx,
                               bacc,
                               kernelSize,
                               stride,
                               kernelVolume,
                               outIJKAcc,
                               outIJKBIdxAcc,
                               outMaskAcc);
    };
    forEachVoxelCUDA<nanovdb::ValueOnIndex>(256, 1, batchHdl, cb);

    outIJK     = outIJK.index({outMask});
    outIJKBIdx = outIJKBIdx.index({outMask});

    return JaggedTensor::from_data_indices_and_list_ids(
        outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConv<torch::kCUDA>(const GridBatchImpl &baseGridHdl,
                                       const nanovdb::Coord &kernelSize,
                                       const nanovdb::Coord &stride) {
    JaggedTensor coords = convIJKForGrid(baseGridHdl, kernelSize, stride);
    return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridForConv<torch::kCPU>(const GridBatchImpl &baseBatchHdl,
                                      const nanovdb::Coord &kernelSize,
                                      const nanovdb::Coord &stride) {
    using GridType = nanovdb::ValueOnIndex;
    if (stride == nanovdb::Coord(1) || stride == kernelSize) {
        return buildCoarseGridFromFineGridCPU(baseBatchHdl, stride);
    }

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();
    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());

    int lower[3], upper[3];
    for (int i = 0; i < 3; i += 1) {
        if (kernelSize[i] % 2 == 0) {
            lower[i] = 0;
            upper[i] = kernelSize[i] - 1;
        } else {
            lower[i] = -(kernelSize[i] - 1) / 2;
            upper[i] = (kernelSize[i] - 1) / 2;
        }
    }

    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseGridHdl.template grid<GridType>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            const nanovdb::Coord &ijk0 = it->first;

            for (int di = lower[0]; di <= upper[0]; di += 1) {
                for (int dj = lower[1]; dj <= upper[1]; dj += 1) {
                    for (int dk = lower[2]; dk <= upper[2]; dk += 1) {
                        const nanovdb::Coord dstIjk = ijk0 + nanovdb::Coord(dk, dj, di);
                        if (dstIjk[0] % stride[2] != 0 || dstIjk[1] % stride[1] != 0 ||
                            dstIjk[2] % stride[0] != 0)
                            continue;
                        proxyGridAccessor.setValue(nanovdb::Coord(dstIjk[0] / stride[2],
                                                                  dstIjk[1] / stride[1],
                                                                  dstIjk[2] / stride[0]),
                                                   1.0f);
                    }
                }
            }
        }

        proxyGridAccessor.merge();
        auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
            *proxyGrid, 0u, false, false);
        batchHandles.push_back(std::move(ret));
    }

    if (batchHandles.size() == 1) {
        return std::move(batchHandles[0]);
    } else {
        return nanovdb::mergeGrids(batchHandles);
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
