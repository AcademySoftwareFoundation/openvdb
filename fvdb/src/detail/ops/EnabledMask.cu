#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

/// @brief Per-voxel callback for getting the enabled state of every voxel in a batch of grids
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void enabledMaskCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
                                            GridBatchImpl::Accessor<GridType> gridAccessor,
                                            TorchAccessor<bool, 1> outEnabledMask,
                                            bool returnDisabled) {
    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];

    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);
    const bool enabled = leaf.template get<ActiveOrUnmasked<GridType>>(voxelIdx);
    if (leaf.isActive(voxelIdx)) {
        const int64_t outIdx = baseOffset + (int64_t) leaf.getValue(voxelIdx) - 1;
        if (returnDisabled) {
            outEnabledMask[outIdx] = !enabled;
        } else {
            outEnabledMask[outIdx] = enabled;
        }
    }
}


/// @brief Return a boolean mask for every voxel in the grid batch indicating if it is enabled or not
/// @tparam DeviceTag The device to run the kernel on
/// @param gridBatch The batch of grids to get the enabled mask for
/// @param returnDisabled If true, return a mask of disabled voxels instead of enabled voxels
/// @return A boolean mask for every voxel in the grid batch indicating if it is enabled or not
template <c10::DeviceType DeviceTag>
JaggedTensor EnabledMask(const GridBatchImpl& batchHdl, bool returnDisabled) {
    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> JaggedTensor {
        batchHdl.checkNonEmptyGrid();

        torch::Tensor outMask = torch::empty({batchHdl.totalVoxels()}, torch::TensorOptions().device(batchHdl.device()).dtype(torch::kBool));
        auto outMaskAcc = tensorAccessor<DeviceTag, bool, 1>(outMask);

        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
                enabledMaskCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, gridAccessor, outMaskAcc, returnDisabled);
            };
            forEachVoxelCUDA<GridType>(1024, 1, batchHdl, cb);
        } else {
            auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, GridBatchImpl::Accessor<GridType> gridAccessor) {
                enabledMaskCallback<GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, gridAccessor, outMaskAcc, returnDisabled);
            };
            forEachVoxelCPU<GridType>(1, batchHdl, cb);
        }

        return batchHdl.jaggedTensor(outMask, true /* ignoreDisabledVoxels */);
    });
}


template <>
JaggedTensor dispatchEnabledMask<torch::kCUDA>(const GridBatchImpl& batchHdl, bool returnDisabled) {
    return EnabledMask<torch::kCUDA>(batchHdl, returnDisabled);
}

template <>
JaggedTensor dispatchEnabledMask<torch::kCPU>(const GridBatchImpl& batchHdl, bool returnDisabled) {
    return EnabledMask<torch::kCPU>(batchHdl, returnDisabled);
}


} // namespace ops
} // namespace detail
} // namespace fvdb