#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"

namespace fvdb {
namespace detail {
namespace ops {


/// @brief Per-voxel callback for getting the enabled grid coordinates in a batch of grids
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void enabledGridCoordsVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
                                                       GridBatchImpl::Accessor<GridType> gridAccessor,
                                                       TorchAccessor<int64_t, 1> leafBaseOffset,
                                                       TorchAccessor<int32_t, 2> outGridCoords) {
    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);
    const int64_t outIdx = leafBaseOffset[leafIdx] + leaf.template get<UnmaskedPerLeaf<GridType>>(voxelIdx);
    if (leaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx)) {
        outGridCoords[outIdx][0] = ijk[0];
        outGridCoords[outIdx][1] = ijk[1];
        outGridCoords[outIdx][2] = ijk[2];
    }
}


/// @brief Per-voxel callback which computes the active grid coordinates for a batch of grids
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void activeGridCoordsVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
                                                      GridBatchImpl::Accessor<GridType> gridAccessor,
                                                      TorchAccessor<int32_t, 2> outGridCoords) {

    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);


    const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);
    if (leaf.isActive(voxelIdx)) {
        const int64_t idx = baseOffset + (int64_t) leaf.getValue(voxelIdx) - 1;
        outGridCoords[idx][0] = ijk[0];
        outGridCoords[idx][1] = ijk[1];
        outGridCoords[idx][2] = ijk[2];
    }
}


/// @brief Get the enabled grid coordinates for a batch of grids (ignoring disabled voxels)
/// @param gridBatch The batch of grids (must be mutable)
/// @param outGridCoords Tensor which will contain the output grid coordinates
template <c10::DeviceType DeviceTag>
void GetEnabledGridCoords(const GridBatchImpl& gridBatch, torch::Tensor& outGridCoords) {
    using GridType = nanovdb::ValueOnIndexMask;

    // Compute a prefix sum of the unmasked voxels per leaf
    const torch::Tensor leafBaseOffset = countEnabledPerLeafShiftedByOne<GridType, DeviceTag>(gridBatch).cumsum(0, torch::kInt64);

    // Get the unmasked grid coordinates
    auto leafBaseOffsetAcc = tensorAccessor<DeviceTag, int64_t, 1>(leafBaseOffset);
    auto outCoordsAcc = tensorAccessor<DeviceTag, int32_t, 2>(outGridCoords);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            enabledGridCoordsVoxelCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, gridAccessor, leafBaseOffsetAcc, outCoordsAcc);
        };
        forEachVoxelCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            enabledGridCoordsVoxelCallback<GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, gridAccessor, leafBaseOffsetAcc, outCoordsAcc);
        };
        forEachVoxelCPU<GridType>(1, gridBatch, cb);
    }
}


/// @brief Get the active grid coordinates for a batch of grids (including disabled coordinates in mutable grids)
/// @tparam GridType The type of the grid (one of ValueOnIndex, ValueOnIndexMask)
/// @param gridBatch The batch of grids
/// @param outGridCoords Tensor which will contain the output grid coordinates
template <c10::DeviceType DeviceTag, typename GridType>
void GetActiveGridCoords(const GridBatchImpl& gridBatch, torch::Tensor& outGridCoords) {
    auto outCoordsAcc = tensorAccessor<DeviceTag, int32_t, 2>(outGridCoords);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            activeGridCoordsVoxelCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, gridAccessor, outCoordsAcc);
        };
        forEachVoxelCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            activeGridCoordsVoxelCallback<GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, gridAccessor, outCoordsAcc);
        };
        forEachVoxelCPU<GridType>(1, gridBatch, cb);
    }
}


/// @brief Get the number of active (or enabled for mutable grids) ijk coordiantes in a batch of grids
/// @tparam DeviceTag Which device to run on
/// @param gridBatch The batch of grids to get the active coordinates for
/// @param ignoreDisabledVoxels If set to true, and the grid batch is mutable, also return coordinates that are disabled
/// @return A JaggedTensor or shape [B, -1, 3] of active/enabled IJK coordinates
template <c10::DeviceType DeviceTag>
JaggedTensor ActiveGridCoords(const GridBatchImpl& gridBatch, bool ignoreDisabledVoxels) {
    gridBatch.checkNonEmptyGrid();
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(gridBatch.device());
    torch::Tensor outGridCoords = torch::empty({gridBatch.totalEnabledVoxels(ignoreDisabledVoxels), 3}, opts);
    FVDB_DISPATCH_GRID_TYPES(gridBatch, [&]() {
        if (ignoreDisabledVoxels || nanovdb::util::is_same<GridType, nanovdb::ValueOnIndex>::value) {
            GetActiveGridCoords<DeviceTag, GridType>(gridBatch, outGridCoords);
        } else if (nanovdb::util::is_same<GridType, nanovdb::ValueOnIndexMask>::value) {
            TORCH_CHECK(!ignoreDisabledVoxels, "This should never happen");
            GetEnabledGridCoords<DeviceTag>(gridBatch, outGridCoords);
        }
    });
    return gridBatch.jaggedTensor(outGridCoords, ignoreDisabledVoxels);
}



template <>
JaggedTensor dispatchActiveGridCoords<torch::kCUDA>(const GridBatchImpl& gridBatch, bool ignoreMasked) {
    return ActiveGridCoords<torch::kCUDA>(gridBatch, ignoreMasked);
}

template <>
JaggedTensor dispatchActiveGridCoords<torch::kCPU>(const GridBatchImpl& gridBatch, bool ignoreMasked) {
    return ActiveGridCoords<torch::kCPU>(gridBatch, ignoreMasked);
}


} // namespace ops
} // namespace detail
} // namespace fvdb