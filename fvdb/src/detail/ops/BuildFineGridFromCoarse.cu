// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/GridBatchImpl.h>
#include <detail/ops/Ops.h>
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/Utils.h>
#include <detail/utils/cuda/ForEachCUDA.cuh>
#include <detail/utils/cuda/RAIIRawDeviceBuffer.h>
#include <detail/utils/cuda/Utils.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/csrc/api/include/torch/types.h>

#include <thrust/device_vector.h>

namespace fvdb::detail::ops {

__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord &ijk0,
           const nanovdb::CoordBBox &bbox,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    int32_t count = 0;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk                      = ijk0 + nanovdb::Coord(di, dj, dk);
                outIJK[base + count][0]  = ijk[0];
                outIJK[base + count][1]  = ijk[1];
                outIJK[base + count][2]  = ijk[2];
                outIJKBIdx[base + count] = bidx;
                count += 1;
            }
        }
    }
}

__device__ inline void
copyCoords(const fvdb::JIdxType bidx,
           const int64_t base,
           const nanovdb::Coord size,
           const nanovdb::Coord &ijk0,
           TorchRAcc64<int32_t, 2> outIJK,
           TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    return copyCoords(bidx,
                      base,
                      ijk0,
                      nanovdb::CoordBBox(nanovdb::Coord(0), size - nanovdb::Coord(1)),
                      outIJK,
                      outIJKBIdx);
}

__device__ void
fineIjkForCoarseGridVoxelCallback(int32_t bidx,
                                  int32_t lidx,
                                  int32_t vidx,
                                  int32_t cidx,
                                  const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> batchAcc,
                                  nanovdb::Coord upsamplingFactor,
                                  TorchRAcc64<int32_t, 2> outIJKData,
                                  TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const nanovdb::NanoGrid<nanovdb::ValueOnIndex> *gridPtr = batchAcc.grid(bidx);
    const typename nanovdb::NanoGrid<nanovdb::ValueOnIndex>::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset     = batchAcc.voxelOffset(bidx);
    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];
    if (leaf.isActive(vidx)) {
        const int64_t value            = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t index            = (baseOffset + value) * totalPadAmount;
        const nanovdb::Coord coarseIjk = leaf.offsetToGlobalCoord(vidx);
        const nanovdb::Coord fineIjk(coarseIjk[0] * upsamplingFactor[0],
                                     coarseIjk[1] * upsamplingFactor[1],
                                     coarseIjk[2] * upsamplingFactor[2]);
        copyCoords(bidx, index, upsamplingFactor, fineIjk, outIJKData, outIJKBIdx);
    }
}

JaggedTensor
fineIJKForCoarseGrid(const GridBatchImpl &batchHdl,
                     nanovdb::Coord upsamplingFactor,
                     const std::optional<JaggedTensor> &maybeMask) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const int64_t totalPadAmount = upsamplingFactor[0] * upsamplingFactor[1] * upsamplingFactor[2];

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    torch::Tensor outIJK     = torch::empty({batchHdl.totalVoxels() * totalPadAmount, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({batchHdl.totalVoxels() * totalPadAmount},
                                            optsBIdx); // TODO: Don't populate for single batch

    auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

    auto cb = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchImpl::Accessor<nanovdb::ValueOnIndex> bacc) {
        fineIjkForCoarseGridVoxelCallback(
            bidx, lidx, vidx, cidx, bacc, upsamplingFactor, outIJKAcc, outIJKBIdxAcc);
    };

    forEachVoxelCUDA<nanovdb::ValueOnIndex>(1024, 1, batchHdl, cb);

    // FIXME: (Francis) this uses a bunch of extra memory. Maybe we can avoid it in the future by
    // allowing for invalid values in the nanovdb GPU grid building process
    if (maybeMask.has_value()) {
        std::vector<torch::Tensor> stack;
        stack.reserve(totalPadAmount);
        for (int i = 0; i < totalPadAmount; ++i) {
            stack.push_back(maybeMask.value().jdata());
        }
        torch::Tensor mask = torch::stack(stack, 1).view({-1});
        outIJK             = outIJK.index({mask}).contiguous();
        outIJKBIdx         = outIJKBIdx.index({mask}).contiguous();
        return JaggedTensor::from_data_indices_and_list_ids(
            outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
    }

    return JaggedTensor::from_data_offsets_and_list_ids(
        outIJK, batchHdl.voxelOffsets(true) * totalPadAmount, batchHdl.jlidx(true));
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildFineGridFromCoarse<torch::kCUDA>(const GridBatchImpl &coarseBatchHdl,
                                              const nanovdb::Coord subdivisionFactor,
                                              const std::optional<JaggedTensor> &subdivMask) {
    JaggedTensor coords = fineIJKForCoarseGrid(coarseBatchHdl, subdivisionFactor, subdivMask);
    return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildFineGridFromCoarse<torch::kCPU>(const GridBatchImpl &coarseBatchHdl,
                                             const nanovdb::Coord subdivisionFactor,
                                             const std::optional<JaggedTensor> &subdivMask) {
    using GridType = nanovdb::ValueOnIndex;
    torch::Tensor subdivMaskTensor;
    if (subdivMask.has_value()) {
        subdivMaskTensor = subdivMask.value().jdata();
    } else {
        subdivMaskTensor = torch::zeros(0, torch::TensorOptions().dtype(torch::kBool));
    }

    using IndexTree = nanovdb::NanoTree<GridType>;

    const nanovdb::GridHandle<TorchDeviceBuffer> &coarseGridHdl = coarseBatchHdl.nanoGridHandle();
    const torch::TensorAccessor<bool, 1> &subdivMaskAcc = subdivMaskTensor.accessor<bool, 1>();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(coarseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < coarseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::NanoGrid<GridType> *coarseGrid = coarseGridHdl.template grid<GridType>(bidx);
        if (!coarseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        const IndexTree &coarseTree = coarseGrid->tree();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        const int64_t joffset = coarseBatchHdl.cumVoxels(bidx);
        for (auto it = ActiveVoxelIterator<GridType, -1>(coarseTree); it.isValid(); it++) {
            const nanovdb::Coord baseIjk(it->first[0] * subdivisionFactor[0],
                                         it->first[1] * subdivisionFactor[1],
                                         it->first[2] * subdivisionFactor[2]);

            if (subdivMaskAcc.size(0) > 0 && !subdivMaskAcc[it->second + joffset]) {
                continue;
            }

            for (int i = 0; i < subdivisionFactor[0]; i += 1) {
                for (int j = 0; j < subdivisionFactor[1]; j += 1) {
                    for (int k = 0; k < subdivisionFactor[2]; k += 1) {
                        const nanovdb::Coord fineIjk = baseIjk + nanovdb::Coord(i, j, k);
                        proxyGridAccessor.setValue(fineIjk, 1);
                    }
                }
            }
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

} // namespace fvdb::detail::ops
