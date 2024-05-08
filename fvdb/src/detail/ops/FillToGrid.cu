#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {


template <typename GridType, typename ScalarType>
__hostdev__ inline void fillToGridVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                   GridBatchImpl::Accessor<GridType> fromGridHdl,
                                                    GridBatchImpl::Accessor<GridType> toGridHdl,
                                                    torch::PackedTensorAccessor32<ScalarType, 2, torch::RestrictPtrTraits> fromFeatures,
                                                    torch::PackedTensorAccessor32<ScalarType, 2, torch::RestrictPtrTraits> toFeatures) {

    using LeafNodeT = typename nanovdb::NanoGrid<GridType>::LeafNodeType;

    const nanovdb::NanoGrid<GridType>* gpuFromGrid = fromGridHdl.grid(batchIdx);
    const int64_t fromBaseOffset = fromGridHdl.voxelOffset(batchIdx);

    const LeafNodeT& fromLeaf = gpuFromGrid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord voxIjk = fromLeaf.offsetToGlobalCoord(voxelIdx);
    const bool isFromActive = fromLeaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx);

    if (!isFromActive) {
        return;
    }

    const nanovdb::NanoGrid<GridType>* gpuToGrid = toGridHdl.grid(batchIdx);
    const int64_t toBaseOffset = toGridHdl.voxelOffset(batchIdx);
    const auto toGridAcc = gpuToGrid->getAccessor();
    if (!toGridAcc.template get<ActiveOrUnmasked<GridType>>(voxIjk)) {
        return;
    }

    const int64_t toIndex = (int64_t) toGridAcc.getValue(voxIjk) + toBaseOffset - 1;
    const int64_t fromIndex = fromLeaf.getValue(voxelIdx) + fromBaseOffset - 1;

    toFeatures[toIndex][channelIdx] = fromFeatures[fromIndex][channelIdx];
}


template <typename GridType, typename ScalarType>
void fillToGridCPU(const GridBatchImpl::Accessor<GridType>& fromGridHandle,
                   const GridBatchImpl::Accessor<GridType>& toGridHandle,
                   const  torch::TensorAccessor<ScalarType, 2> fromFeatures,
                   torch::TensorAccessor<ScalarType, 2> toFeatures,
                   bool isContiguous) {

    for (size_t bi = 0; bi < fromGridHandle.batchSize(); bi += 1) {
        const nanovdb::NanoGrid<GridType>* fromGrid = fromGridHandle.grid(bi);
        const nanovdb::NanoGrid<GridType>* toGrid = toGridHandle.grid(bi);

        const int64_t fromBaseOffset = fromGridHandle.voxelOffset(bi);
        const int64_t toBaseOffset = toGridHandle.voxelOffset(bi);

        for (auto it = ActiveVoxelIterator<GridType, -1>(fromGrid->tree(), false, fromBaseOffset); it.isValid(); it++) {
            const nanovdb::Coord voxIjk = it->first;

            if (!toGrid->getAccessor().template get<ActiveOrUnmasked<GridType>>(voxIjk)) {
                continue;
            }
            const int64_t toIndex = (int64_t) toGrid->getAccessor().getValue(voxIjk) + toBaseOffset - 1;
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
void dispatchFillToGrid<torch::kCUDA>(const GridBatchImpl& fromGrid,
                        const GridBatchImpl& toGrid,
                        const torch::Tensor& fromFeatures,
                        torch::Tensor& toFeatures) {

    FVDB_DISPATCH_GRID_TYPES(fromGrid, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(fromFeatures.scalar_type(), "fillToGrid", [&]() {
            auto fromFeaturesAcc = fromFeatures.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto toFeaturesAcc = toFeatures.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
            auto toGridAcc = toGrid.deviceAccessor<GridType>();
            auto callback = [=] __device__ (int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, GridBatchImpl::Accessor<GridType> fromGridAcc) {
                fillToGridVoxelCallback<GridType, scalar_t>(bidx, lidx, vidx, cidx, fromGridAcc, toGridAcc, fromFeaturesAcc, toFeaturesAcc);
            };
            forEachVoxelCUDA<GridType>(512, fromFeatures.size(1), fromGrid, callback);

        });
    });
}


template <>
void dispatchFillToGrid<torch::kCPU>(const GridBatchImpl& fromGrid,
                        const GridBatchImpl& toGrid,
                        const torch::Tensor& fromFeatures,
                        torch::Tensor& toFeatures) {

    bool isContiguous = fromFeatures.is_contiguous() && toFeatures.is_contiguous();

    FVDB_DISPATCH_GRID_TYPES(toGrid, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(fromFeatures.scalar_type(), "fillToGrid", [&]() {
            fillToGridCPU<GridType, scalar_t>(fromGrid.hostAccessor<GridType>(),
                                              toGrid.hostAccessor<GridType>(),
                                              fromFeatures.accessor<scalar_t, 2>(),
                                              toFeatures.accessor<scalar_t, 2>(), isContiguous);
        });
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
