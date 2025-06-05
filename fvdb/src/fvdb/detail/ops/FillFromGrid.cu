// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType>
__hostdev__ inline void
fillToGridVoxelCallback(
    int32_t batchIdx,
    int32_t leafIdx,
    int32_t voxelIdx,
    int32_t channelIdx,
    GridBatchImpl::Accessor<nanovdb::ValueOnIndex> fromGridHdl,
    GridBatchImpl::Accessor<nanovdb::ValueOnIndex> toGridHdl,
    torch::PackedTensorAccessor64<ScalarType, 2, torch::RestrictPtrTraits> fromFeatures,
    torch::PackedTensorAccessor64<ScalarType, 2, torch::RestrictPtrTraits> toFeatures) {
    using LeafNodeT = typename nanovdb::OnIndexGrid::LeafNodeType;

    const nanovdb::OnIndexGrid *gpuFromGrid = fromGridHdl.grid(batchIdx);
    const int64_t fromBaseOffset            = fromGridHdl.voxelOffset(batchIdx);

    const LeafNodeT &fromLeaf   = gpuFromGrid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord voxIjk = fromLeaf.offsetToGlobalCoord(voxelIdx);
    const bool isFromActive     = fromLeaf.isActive(voxelIdx);

    if (!isFromActive) {
        return;
    }

    const nanovdb::OnIndexGrid *gpuToGrid = toGridHdl.grid(batchIdx);
    const int64_t toBaseOffset            = toGridHdl.voxelOffset(batchIdx);
    const auto toGridAcc                  = gpuToGrid->getAccessor();
    if (!toGridAcc.isActive(voxIjk)) {
        return;
    }

    const int64_t toIndex   = (int64_t)toGridAcc.getValue(voxIjk) + toBaseOffset - 1;
    const int64_t fromIndex = fromLeaf.getValue(voxelIdx) + fromBaseOffset - 1;

    toFeatures[toIndex][channelIdx] = fromFeatures[fromIndex][channelIdx];
}

template <typename ScalarType>
void
fillToGridCPU(const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> &fromGridHandle,
              const GridBatchImpl::Accessor<nanovdb::ValueOnIndex> &toGridHandle,
              const torch::TensorAccessor<ScalarType, 2> fromFeatures,
              torch::TensorAccessor<ScalarType, 2> toFeatures,
              bool isContiguous) {
    for (int64_t bi = 0; bi < fromGridHandle.batchSize(); bi += 1) {
        const nanovdb::OnIndexGrid *fromGrid = fromGridHandle.grid(bi);
        const nanovdb::OnIndexGrid *toGrid   = toGridHandle.grid(bi);

        const int64_t fromBaseOffset = fromGridHandle.voxelOffset(bi);
        const int64_t toBaseOffset   = toGridHandle.voxelOffset(bi);

        for (auto it = ActiveVoxelIterator<-1>(fromGrid->tree(), fromBaseOffset); it.isValid();
             it++) {
            const nanovdb::Coord voxIjk = it->first;

            if (!toGrid->getAccessor().isActive(voxIjk)) {
                continue;
            }
            const int64_t toIndex =
                (int64_t)toGrid->getAccessor().getValue(voxIjk) + toBaseOffset - 1;
            if (isContiguous) {
                memcpy(toFeatures[toIndex].data(),
                       fromFeatures[it->second].data(),
                       fromFeatures.size(1) * sizeof(ScalarType));
            } else {
                for (int c = 0; c < toFeatures.size(1); ++c) {
                    toFeatures[toIndex][c] = fromFeatures[it->second][c];
                }
            }
        }
    }
}

template <>
void
dispatchFillFromGrid<torch::kCUDA>(const GridBatchImpl &fromGrid,
                                   const GridBatchImpl &toGrid,
                                   const torch::Tensor &fromFeatures,
                                   torch::Tensor &toFeatures) {
    AT_DISPATCH_V2(
        fromFeatures.scalar_type(),
        "fillToGrid",
        AT_WRAP([&]() {
            auto fromFeaturesAcc =
                fromFeatures.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto toFeaturesAcc =
                toFeatures.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
            auto toGridAcc = toGrid.deviceAccessor<nanovdb::ValueOnIndex>();
            auto callback  = [=] __device__(int64_t bidx,
                                           int64_t lidx,
                                           int64_t vidx,
                                           int64_t cidx,
                                           GridBatchImpl::Accessor<nanovdb::ValueOnIndex>
                                               fromGridAcc) {
                fillToGridVoxelCallback<scalar_t>(
                    bidx, lidx, vidx, cidx, fromGridAcc, toGridAcc, fromFeaturesAcc, toFeaturesAcc);
            };
            forEachVoxelCUDA<nanovdb::ValueOnIndex>(512, fromFeatures.size(1), fromGrid, callback);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template <>
void
dispatchFillFromGrid<torch::kCPU>(const GridBatchImpl &fromGrid,
                                  const GridBatchImpl &toGrid,
                                  const torch::Tensor &fromFeatures,
                                  torch::Tensor &toFeatures) {
    bool isContiguous = fromFeatures.is_contiguous() && toFeatures.is_contiguous();

    AT_DISPATCH_V2(fromFeatures.scalar_type(),
                   "fillToGrid",
                   AT_WRAP([&]() {
                       fillToGridCPU<scalar_t>(fromGrid.hostAccessor<nanovdb::ValueOnIndex>(),
                                               toGrid.hostAccessor<nanovdb::ValueOnIndex>(),
                                               fromFeatures.accessor<scalar_t, 2>(),
                                               toFeatures.accessor<scalar_t, 2>(),
                                               isContiguous);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
