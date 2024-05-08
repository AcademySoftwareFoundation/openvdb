#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Per-voxel callback for getting the batch index of each enabled voxel in a batch of grids
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void enabledGridCoordsVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
                                                        GridBatchImpl::Accessor<GridType> gridAccessor,
                                                        TorchAccessor<int64_t, 1> leafBaseOffset,
                                                        TorchAccessor<int16_t, 1> outJIdx) {
    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);
    const int64_t outIdx = leafBaseOffset[leafIdx] + leaf.template get<UnmaskedPerLeaf<GridType>>(voxelIdx);
    if (leaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx)) {
        outJIdx[outIdx] = batchIdx;
    }
}


template <c10::DeviceType DeviceTag>
inline torch::Tensor EnabledVoxelsJIdx(const GridBatchImpl& gridBatch) {
    using GridType = nanovdb::ValueOnIndexMask;

    // Compute a prefix sum of the unmasked voxels per leaf
    const torch::Tensor leafBaseOffset = countEnabledPerLeafShiftedByOne<GridType, DeviceTag>(gridBatch).cumsum(0, torch::kInt64);
    const int64_t numEnabledCoords = leafBaseOffset[-1].item<int64_t>();

    torch::Tensor outJIdx = torch::empty({numEnabledCoords}, torch::dtype(torch::kInt16).device(gridBatch.device()));

    // Get the enabled grid batch indices
    auto leafBaseOffsetAcc = tensorAccessor<DeviceTag, int64_t, 1>(leafBaseOffset);
    auto outJIdxAcc = tensorAccessor<DeviceTag, int16_t, 1>(outJIdx);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            enabledGridCoordsVoxelCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, gridAccessor, leafBaseOffsetAcc, outJIdxAcc);
        };
        forEachVoxelCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            enabledGridCoordsVoxelCallback<GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, gridAccessor, leafBaseOffsetAcc, outJIdxAcc);
        };
        forEachVoxelCPU<GridType>(1, gridBatch, cb);
    }
    return outJIdx;
}


template <c10::DeviceType DeviceTag>
torch::Tensor GridJIdx(const GridBatchImpl& gridBatch, bool ignoreDisabledVoxels) {
    return FVDB_DISPATCH_GRID_TYPES(gridBatch, [&]() {
        if (ignoreDisabledVoxels || nanovdb::util::is_same<GridType, nanovdb::ValueOnIndex>::value) {
            return ops::dispatchJIdxForJOffsets<DeviceTag>(gridBatch.voxelOffsets(ignoreDisabledVoxels), gridBatch.totalVoxels());
        } else if (nanovdb::util::is_same<GridType, nanovdb::ValueOnIndexMask>::value) {
            TORCH_CHECK(!ignoreDisabledVoxels, "This should never happen");
            return EnabledVoxelsJIdx<DeviceTag>(gridBatch);
        }
    });
}



template <>
torch::Tensor dispatchJIdxForGrid<torch::kCPU>(const GridBatchImpl& gridBatch, bool ignoreDisabledVoxels) {
    return GridJIdx<torch::kCPU>(gridBatch, ignoreDisabledVoxels);
}

template <>
torch::Tensor dispatchJIdxForGrid<torch::kCUDA>(const GridBatchImpl& gridBatch, bool ignoreDisabledVoxels) {
    return GridJIdx<torch::kCUDA>(gridBatch, ignoreDisabledVoxels);
}

} // namespace ops
} // namespace detail
} // namespace fvdb