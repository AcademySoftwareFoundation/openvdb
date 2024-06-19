#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/nanovdb/CustomAccessors.h"

namespace fvdb {
namespace detail {
namespace ops {


/// @brief Per-voxel callback to compute a mask of the enabled voxels in a bounding box for a batch of grids
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void enabledGridVoxelInBoundsMaskCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
                                                             GridBatchImpl::Accessor<GridType> gridAccessor,
                                                             TorchAccessor<int64_t, 1> leafBaseOffset,
                                                             TorchAccessor<int32_t, 3> bboxes,
                                                             TorchAccessor<bool, 1> outGridBoundsMask) {
    const nanovdb::CoordBBox maskBbox(nanovdb::Coord(bboxes[batchIdx][0][0], bboxes[batchIdx][0][1], bboxes[batchIdx][0][2]),
                                      nanovdb::Coord(bboxes[batchIdx][1][0], bboxes[batchIdx][1][1], bboxes[batchIdx][1][2]));

    const nanovdb::NanoGrid<GridType> *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    if (maskBbox.hasOverlap(leaf.bbox())) {
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);
        if (leaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx) && maskBbox.isInside(ijk)) {
            const int64_t outIdx = leafBaseOffset[leafIdx] + leaf.template get<UnmaskedPerLeaf<GridType>>(voxelIdx);
            outGridBoundsMask[outIdx] = true;
        }
    }
}

/// @brief Per-voxel callback to compute a mask of the active grid voxels in a bounding box for a batch of grids
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void activeGridVoxelInBoundsMaskCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
                                                            GridBatchImpl::Accessor<GridType> gridAccessor,
                                                            TorchAccessor<int32_t, 3> bboxes,
                                                            TorchAccessor<bool, 1> outGridBoundsMask) {

    const nanovdb::CoordBBox maskBbox(nanovdb::Coord(bboxes[batchIdx][0][0], bboxes[batchIdx][0][1], bboxes[batchIdx][0][2]),
                                      nanovdb::Coord(bboxes[batchIdx][1][0], bboxes[batchIdx][1][1], bboxes[batchIdx][1][2]));

    const nanovdb::NanoGrid<GridType> *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    if (maskBbox.hasOverlap(leaf.bbox())) {
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);
        if (leaf.isActive(voxelIdx) && maskBbox.isInside(ijk)) {
            const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);
            const int64_t idx = baseOffset + (int64_t)leaf.getValue(voxelIdx) - 1;
            outGridBoundsMask[idx] = true;
        }
    }
}

/// @brief Get a boolean mask of the enabled grid voxels for a batch of grids (ignoring disabled voxels)
/// @param gridBatch The batch of grids (must be mutable)
/// @param batchBboxes The batch of bounding boxes
/// @param outGridCoords Tensor which will contain the output grid coordinates
template <c10::DeviceType DeviceTag>
void GetEnabledVoxelsInBoundsMask(const GridBatchImpl& gridBatch,
                                  torch::Tensor& batchBboxes,
                                  torch::Tensor& outGridBoundsMask) {
    using GridType = nanovdb::ValueOnIndexMask;

    // Compute a prefix sum of the unmasked voxels per leaf
    const torch::Tensor leafBaseOffset = countEnabledPerLeafShiftedByOne<GridType, DeviceTag>(gridBatch).cumsum(0, torch::kInt64);

    // Get the unmasked grid coordinates
    auto leafBaseOffsetAcc = tensorAccessor<DeviceTag, int64_t, 1>(leafBaseOffset);
    auto outMaskAcc = tensorAccessor<DeviceTag, bool, 1>(outGridBoundsMask);
    auto bboxAcc = tensorAccessor<DeviceTag, int32_t, 3>(batchBboxes);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            enabledGridVoxelInBoundsMaskCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, gridAccessor, leafBaseOffsetAcc, bboxAcc, outMaskAcc);
        };
        forEachVoxelCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=](int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            enabledGridVoxelInBoundsMaskCallback<GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, gridAccessor, leafBaseOffsetAcc, bboxAcc, outMaskAcc);
        };
        forEachVoxelCPU<GridType>(1, gridBatch, cb);
    }
}

/// @brief Get a boolean mask of the active grid voxels for a batch of grids  (including disabled coordinates in mutable grids)
/// @tparam GridType The type of the grid (one of ValueOnIndex, ValueOnIndexMask)
/// @param gridBatch The batch of grids
/// @param batchBboxes The batch of bounding boxes
/// @param outGridCoords Tensor which will contain the output grid coordinates
template <c10::DeviceType DeviceTag, typename GridType>
void GetActiveVoxelsInBoundsMask(const GridBatchImpl& gridBatch,
                                 torch::Tensor& batchBboxes,
                                 torch::Tensor& outGridBoundsMask) {
    auto outMaskAcc = tensorAccessor<DeviceTag, bool, 1>(outGridBoundsMask);
    auto bboxAcc = tensorAccessor<DeviceTag, int32_t, 3>(batchBboxes);

    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            activeGridVoxelInBoundsMaskCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, gridAccessor, bboxAcc, outMaskAcc);
        };
        forEachVoxelCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=](int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
            activeGridVoxelInBoundsMaskCallback<GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, gridAccessor, bboxAcc, outMaskAcc);
        };
        forEachVoxelCPU<GridType>(1, gridBatch, cb);
    }
}

template <c10::DeviceType DeviceTag>
JaggedTensor ActiveVoxelsInBoundsMask(const GridBatchImpl& batchHdl,
                                      const Vec3iBatch& ijkMin,
                                      const Vec3iBatch& ijkMax,
                                      bool ignoreDisabledVoxels) {

    batchHdl.checkNonEmptyGrid();

    // output storage
    auto opts = torch::TensorOptions().dtype(torch::kBool).device(batchHdl.device());
    torch::Tensor outGridBoundsMask = torch::zeros({batchHdl.totalEnabledVoxels(ignoreDisabledVoxels)}, opts);

    // bbox to tensor storage
    const std::vector<nanovdb::Coord>& bboxMins = ijkMin.value(batchHdl.batchSize(), false, "ijk_min");
    const std::vector<nanovdb::Coord>& bboxMaxs = ijkMax.value(batchHdl.batchSize(), false, "ijk_max");

    torch::Tensor batchBboxes = torch::empty({batchHdl.batchSize(), 2, 3},
                                             torch::TensorOptions().dtype(torch::kInt32).device(batchHdl.device()));

    for (size_t batchIdx = 0; batchIdx < batchHdl.batchSize(); batchIdx++) {
        for (size_t dimIdx = 0; dimIdx < 3; dimIdx++) {
            batchBboxes[batchIdx][0][dimIdx] = bboxMins[batchIdx][dimIdx];
            batchBboxes[batchIdx][1][dimIdx] = bboxMaxs[batchIdx][dimIdx];
        }
    }

    // create boolean mask of active voxels
    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        if (ignoreDisabledVoxels || nanovdb::util::is_same<GridType, nanovdb::ValueOnIndex>::value) {
            GetActiveVoxelsInBoundsMask<DeviceTag, GridType>(batchHdl, batchBboxes, outGridBoundsMask);
        } else if (nanovdb::util::is_same<GridType, nanovdb::ValueOnIndexMask>::value) {
            TORCH_CHECK(!ignoreDisabledVoxels, "This should never happen");
            GetEnabledVoxelsInBoundsMask<DeviceTag>(batchHdl, batchBboxes, outGridBoundsMask);
        }
    });

    return batchHdl.jaggedTensor(outGridBoundsMask, ignoreDisabledVoxels);
}


template <>
JaggedTensor dispatchActiveVoxelsInBoundsMask<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                          const Vec3iBatch& boundsMinIjk,
                                                          const Vec3iBatch& boundsMaxIjk,
                                                          bool ignoreDisabledVoxels) {
    return ActiveVoxelsInBoundsMask<torch::kCUDA>(batchHdl, boundsMinIjk, boundsMaxIjk, ignoreDisabledVoxels);
}

template <>
JaggedTensor dispatchActiveVoxelsInBoundsMask<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                         const Vec3iBatch& boundsMinIjk,
                                                         const Vec3iBatch& boundsMaxIjk,
                                                         bool ignoreDisabledVoxels) {
    return ActiveVoxelsInBoundsMask<torch::kCPU>(batchHdl, boundsMinIjk, boundsMaxIjk, ignoreDisabledVoxels);
}

} // namespace ops
} // namespace detail
} // namespace fvdb