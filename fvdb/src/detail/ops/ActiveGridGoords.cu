// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/GridBatchImpl.h>
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/ForEachCPU.h>
#include <detail/utils/cuda/ForEachCUDA.cuh>
#include <detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Per-voxel callback which computes the active grid coordinates for a batch of grids
template <template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void
activeGridCoordsVoxelCallback(int64_t batchIdx,
                              int64_t leafIdx,
                              int64_t voxelIdx,
                              GridBatchImpl::Accessor<nanovdb::ValueOnIndex> gridAccessor,
                              TorchAccessor<int32_t, 2> outGridCoords) {
    const nanovdb::OnIndexGrid *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::OnIndexGrid::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);

    const nanovdb::Coord &ijk = leaf.offsetToGlobalCoord(voxelIdx);
    if (leaf.isActive(voxelIdx)) {
        const int64_t idx     = baseOffset + (int64_t)leaf.getValue(voxelIdx) - 1;
        outGridCoords[idx][0] = ijk[0];
        outGridCoords[idx][1] = ijk[1];
        outGridCoords[idx][2] = ijk[2];
    }
}

/// @brief Get the active grid coordinates for a batch of grids (including disabled coordinates in
/// mutable grids)
/// @tparam GridType The type of the grid (one of ValueOnIndex, ValueOnIndexMask)
/// @param gridBatch The batch of grids
/// @param outGridCoords Tensor which will contain the output grid coordinates
template <c10::DeviceType DeviceTag>
void
GetActiveGridCoords(const GridBatchImpl &gridBatch, torch::Tensor &outGridCoords) {
    auto outCoordsAcc = tensorAccessor<DeviceTag, int32_t, 2>(outGridCoords);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int64_t batchIdx,
                                 int64_t leafIdx,
                                 int64_t voxelIdx,
                                 int64_t,
                                 GridBatchImpl::Accessor<nanovdb::ValueOnIndex> gridAccessor) {
            activeGridCoordsVoxelCallback<TorchRAcc32>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCoordsAcc);
        };
        forEachVoxelCUDA<nanovdb::ValueOnIndex>(1024, 1, gridBatch, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=] __device__(int64_t batchIdx,
                                 int64_t leafIdx,
                                 int64_t voxelIdx,
                                 int64_t,
                                 GridBatchImpl::Accessor<nanovdb::ValueOnIndex> gridAccessor) {
            activeGridCoordsVoxelCallback<TorchRAcc32>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCoordsAcc);
        };
        forEachVoxelPrivateUse1<nanovdb::ValueOnIndex>(1, gridBatch, cb);
    } else {
        auto cb = [=](int64_t batchIdx,
                      int64_t leafIdx,
                      int64_t voxelIdx,
                      int64_t,
                      GridBatchImpl::Accessor<nanovdb::ValueOnIndex> gridAccessor) {
            activeGridCoordsVoxelCallback<TorchAcc>(
                batchIdx, leafIdx, voxelIdx, gridAccessor, outCoordsAcc);
        };
        forEachVoxelCPU<nanovdb::ValueOnIndex>(1, gridBatch, cb);
    }
}

/// @brief Get the number of active (or enabled for mutable grids) ijk coordinates in a batch of
/// grids
/// @tparam DeviceTag Which device to run on
/// @param gridBatch The batch of grids to get the active coordinates for
/// @param ignoreDisabledVoxels If set to true, and the grid batch is mutable, also return
/// coordinates that are disabled
/// @return A JaggedTensor or shape [B, -1, 3] of active/enabled IJK coordinates
template <c10::DeviceType DeviceTag>
JaggedTensor
ActiveGridCoords(const GridBatchImpl &gridBatch) {
    gridBatch.checkNonEmptyGrid();
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(gridBatch.device());
    torch::Tensor outGridCoords = torch::empty({gridBatch.totalVoxels(), 3}, opts);
    GetActiveGridCoords<DeviceTag>(gridBatch, outGridCoords);
    return gridBatch.jaggedTensor(outGridCoords);
}

template <>
JaggedTensor
dispatchActiveGridCoords<torch::kCUDA>(const GridBatchImpl &gridBatch) {
    return ActiveGridCoords<torch::kCUDA>(gridBatch);
}

template <>
JaggedTensor
dispatchActiveGridCoords<torch::kCPU>(const GridBatchImpl &gridBatch) {
    return ActiveGridCoords<torch::kCPU>(gridBatch);
}

template <>
JaggedTensor
dispatchActiveGridCoords<torch::kPrivateUse1>(const GridBatchImpl &gridBatch) {
    return ActiveGridCoords<torch::kPrivateUse1>(gridBatch);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
