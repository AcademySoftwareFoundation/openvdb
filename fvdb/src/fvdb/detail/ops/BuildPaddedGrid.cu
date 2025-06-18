// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

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

__device__ inline void
copyCoordsWithoutBorder(
    const typename nanovdb::DefaultReadAccessor<nanovdb::ValueOnIndex> gridAccessor,
    const fvdb::JIdxType bidx,
    const int64_t base,
    const nanovdb::Coord &ijk0,
    const nanovdb::CoordBBox &bbox,
    const TorchRAcc64<int64_t, 1> packInfoBase,
    TorchRAcc64<int32_t, 2> outIJK,
    TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    bool active = true;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk    = ijk0 + nanovdb::Coord(di, dj, dk);
                active = active && gridAccessor.isActive(ijk);
            }
        }
    }
    if (active) {
        int64_t outBase     = packInfoBase[base];
        outIJK[outBase][0]  = ijk0[0];
        outIJK[outBase][1]  = ijk0[1];
        outIJK[outBase][2]  = ijk0[2];
        outIJKBIdx[outBase] = bidx;
    }
}

__device__ inline void
countCoordsWithoutBorder(
    const typename nanovdb::DefaultReadAccessor<nanovdb::ValueOnIndex> gridAccessor,
    const fvdb::JIdxType bidx,
    const int64_t base,
    const nanovdb::Coord &ijk0,
    const nanovdb::CoordBBox &bbox,
    TorchRAcc64<int64_t, 1> outCounter) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));
    nanovdb::Coord ijk;
    bool active = true;
    for (int di = bbox.min()[0]; di <= bbox.max()[0]; di += 1) {
        for (int dj = bbox.min()[1]; dj <= bbox.max()[1]; dj += 1) {
            for (int dk = bbox.min()[2]; dk <= bbox.max()[2]; dk += 1) {
                ijk    = ijk0 + nanovdb::Coord(di, dj, dk);
                active = active && gridAccessor.isActive(ijk);
            }
        }
    }

    outCounter[base] = active ? 1 : 0;
}

__device__ void
ijkForGridVoxelCallback(int32_t bidx,
                        int32_t lidx,
                        int32_t vidx,
                        int32_t cidx,
                        const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> batchAcc,
                        const nanovdb::CoordBBox bbox,
                        TorchRAcc64<int32_t, 2> outIJKData,
                        TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const int32_t totalPadAmount = static_cast<int32_t>(bbox.volume());

    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const int64_t totalVoxels           = gridPtr->activeVoxelCount();
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value       = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t base        = (baseOffset + value) * totalPadAmount;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        copyCoords(bidx, base, ijk0, bbox, outIJKData, outIJKBIdx);
    }
}

__device__ void
ijkForGridVoxelCallbackWithoutBorder(int32_t bidx,
                                     int32_t lidx,
                                     int32_t vidx,
                                     int32_t cidx,
                                     const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> batchAcc,
                                     const nanovdb::CoordBBox bbox,
                                     const TorchRAcc64<int64_t, 1> packInfoBase,
                                     TorchRAcc64<int32_t, 2> outIJKData,
                                     TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const auto gridAccessor             = gridPtr->getAccessor();
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value       = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t base        = baseOffset + value;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        copyCoordsWithoutBorder(
            gridAccessor, bidx, base, ijk0, bbox, packInfoBase, outIJKData, outIJKBIdx);
    }
}

__device__ void
ijkForGridVoxelCallbackWithoutBorderCount(
    int32_t bidx,
    int32_t lidx,
    int32_t vidx,
    int32_t cidx,
    const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> batchAcc,
    const nanovdb::CoordBBox bbox,
    TorchRAcc64<int64_t, 1> outCounter) {
    const nanovdb::OnIndexGrid *gridPtr = batchAcc.grid(bidx);
    const auto gridAccessor             = gridPtr->getAccessor();
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        gridPtr->tree().template getFirstNode<0>()[lidx];
    const int64_t baseOffset = batchAcc.voxelOffset(bidx);

    if (leaf.isActive(vidx)) {
        const int64_t value       = ((int64_t)leaf.getValue(vidx)) - 1;
        const int64_t base        = baseOffset + value;
        const nanovdb::Coord ijk0 = leaf.offsetToGlobalCoord(vidx);
        countCoordsWithoutBorder(gridAccessor, bidx, base, ijk0, bbox, outCounter);
    }
}

JaggedTensor
paddedIJKForGrid(const GridBatchImpl &batchHdl, const nanovdb::CoordBBox &bbox) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const int32_t totalPadAmount = static_cast<int32_t>(bbox.volume());

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
        ijkForGridVoxelCallback(bidx, lidx, vidx, cidx, bacc, bbox, outIJKAcc, outIJKBIdxAcc);
    };
    forEachVoxelCUDA<nanovdb::ValueOnIndex>(1024, 1, batchHdl, cb);

    return JaggedTensor::from_data_offsets_and_list_ids(
        outIJK, batchHdl.voxelOffsets() * totalPadAmount, batchHdl.jlidx());
}

JaggedTensor
paddedIJKForGridWithoutBorder(const GridBatchImpl &batchHdl, const nanovdb::CoordBBox &bbox) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "GridBatchImpl must have a valid index");

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(batchHdl.device());
    const torch::TensorOptions optsCounter =
        torch::TensorOptions().dtype(torch::kInt64).device(batchHdl.device());

    torch::Tensor outCounter = torch::empty({batchHdl.totalVoxels()}, optsCounter);
    auto outCounterAcc       = outCounter.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();
    auto cb                  = [=] __device__(int32_t bidx,
                             int32_t lidx,
                             int32_t vidx,
                             int32_t cidx,
                             GridBatchImpl::Accessor<nanovdb::ValueOnIndex> bacc) {
        ijkForGridVoxelCallbackWithoutBorderCount(
            bidx, lidx, vidx, cidx, bacc, bbox, outCounterAcc);
    };
    forEachVoxelCUDA<nanovdb::ValueOnIndex>(512, 1, batchHdl, cb);

    torch::Tensor cumCounts    = torch::cumsum(outCounter, 0);
    int64_t numVoxels          = cumCounts[-1].item<int64_t>();
    torch::Tensor packInfoBase = cumCounts - outCounter;

    torch::Tensor outIJK = torch::empty({numVoxels, 3}, optsData);
    torch::Tensor outIJKBIdx =
        torch::empty({numVoxels}, optsBIdx); // TODO: Don't populate for single batch
    if (numVoxels == 0) {
        // TODO(ruilong): Shall we raise error? Do we support empty grid?
        return JaggedTensor::from_data_indices_and_list_ids(
            outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
    }

    auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    auto outIJKBIdxAcc =
        outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();
    auto packInfoBaseAcc = packInfoBase.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>();

    auto cb2 = [=] __device__(int32_t bidx,
                              int32_t lidx,
                              int32_t vidx,
                              int32_t cidx,
                              GridBatchImpl::Accessor<nanovdb::ValueOnIndex> bacc) {
        ijkForGridVoxelCallbackWithoutBorder(
            bidx, lidx, vidx, cidx, bacc, bbox, packInfoBaseAcc, outIJKAcc, outIJKBIdxAcc);
    };
    forEachVoxelCUDA<nanovdb::ValueOnIndex>(512, 1, batchHdl, cb2);

    return JaggedTensor::from_data_indices_and_list_ids(
        outIJK, outIJKBIdx, batchHdl.jlidx(), batchHdl.batchSize());
}

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromGridWithoutBorderCPU(const GridBatchImpl &baseBatchHdl, int BMIN, int BMAX) {
    using GridType = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseGridHdl.template grid<GridType>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }
        auto baseGridAccessor = baseGrid->getAccessor();

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0 = it->first;
            bool active         = true;
            for (int di = BMIN; di <= BMAX && active; di += 1) {
                for (int dj = BMIN; dj <= BMAX && active; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX && active; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        if (ijk != ijk0) {
                            active = active && baseGridAccessor.isActive(
                                                   ijk); // if any surrounding is off, turn it off.
                        }
                    }
                }
            }
            if (active) {
                proxyGridAccessor.setValue(ijk0, 1.0f);
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

nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromGridCPU(const GridBatchImpl &baseBatchHdl, int BMIN, int BMAX) {
    using GridType = nanovdb::ValueOnIndex;

    TORCH_CHECK(BMIN <= BMAX, "BMIN must be less than BMAX");

    const nanovdb::GridHandle<TorchDeviceBuffer> &baseGridHdl = baseBatchHdl.nanoGridHandle();

    std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
    batchHandles.reserve(baseGridHdl.gridCount());
    for (uint32_t bidx = 0; bidx < baseGridHdl.gridCount(); bidx += 1) {
        const nanovdb::OnIndexGrid *baseGrid = baseGridHdl.template grid<GridType>(bidx);
        if (!baseGrid) {
            throw std::runtime_error("Failed to get pointer to nanovdb index grid");
        }

        using ProxyGridT       = nanovdb::tools::build::Grid<float>;
        auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
        auto proxyGridAccessor = proxyGrid->getWriteAccessor();

        for (auto it = ActiveVoxelIterator(baseGrid->tree()); it.isValid(); it++) {
            nanovdb::Coord ijk0 = it->first;
            for (int di = BMIN; di <= BMAX; di += 1) {
                for (int dj = BMIN; dj <= BMAX; dj += 1) {
                    for (int dk = BMIN; dk <= BMAX; dk += 1) {
                        const nanovdb::Coord ijk = ijk0 + nanovdb::Coord(di, dj, dk);
                        proxyGridAccessor.setValue(ijk, 1.0f);
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

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kCUDA>(const GridBatchImpl &baseBatchHdl,
                                      int bmin,
                                      int bmax,
                                      bool excludeBorder) {
    JaggedTensor coords;
    if (excludeBorder) {
        coords = paddedIJKForGridWithoutBorder(
            baseBatchHdl, nanovdb::CoordBBox(nanovdb::Coord(bmin), nanovdb::Coord(bmax)));
    } else {
        coords = paddedIJKForGrid(baseBatchHdl,
                                  nanovdb::CoordBBox(nanovdb::Coord(bmin), nanovdb::Coord(bmax)));
    }
    return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildPaddedGrid<torch::kCPU>(const GridBatchImpl &baseBatchHdl,
                                     int bmin,
                                     int bmax,
                                     bool excludeBorder) {
    if (excludeBorder) {
        return buildPaddedGridFromGridWithoutBorderCPU(baseBatchHdl, bmin, bmax);
    } else {
        return buildPaddedGridFromGridCPU(baseBatchHdl, bmin, bmax);
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb
