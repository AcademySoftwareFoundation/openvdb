// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_COUNTENABLEDPERLEAF_CUH
#define FVDB_DETAIL_OPS_COUNTENABLEDPERLEAF_CUH

#include <detail/GridBatchImpl.h>
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/ForEachCPU.h>
#include <detail/utils/cuda/ForEachCUDA.cuh>

namespace fvdb {

namespace _private {

/// @brief Per-leaf callback function for counting the number of enabled voxels in each leaf used by
///        countUnmaskedPerLeafShiftedByOne
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void
countEnabledPerLeafShiftByOneLeafCallback(
    int32_t batchIdx,
    int32_t leafIdx,
    fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor,
    TorchAccessor<int32_t, 1> outUnmaskedPerLeafAcc) {
    const nanovdb::NanoGrid<GridType> *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t numUnmasked =
        static_cast<int64_t>(leaf.template get<fvdb::TotalUnmaskedPerLeaf<GridType>>(1111));
    outUnmaskedPerLeafAcc[gridAccessor.leafOffset(batchIdx) + leafIdx + 1] = numUnmasked;
}

} // namespace _private

/// @brief Count the number of unmasked voxels in each leaf node in the grid batch on the GPU
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @param gridBatch The grid batch
/// @return A tensor of size [total_leaves + 1] containing the number of unmasked voxels in each
/// leaf (and zero in the first index)
///         i.e. [0, #unmasked_in_leaf_0, #unmasked_in_leaf_1, ... #unmasksed_in_leaf_N]
template <typename GridType, c10::DeviceType DeviceTag>
__host__ inline torch::Tensor
countEnabledPerLeafShiftedByOne(const fvdb::detail::GridBatchImpl &gridBatch) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(gridBatch.device());
    torch::Tensor unmaskedPerLeaf = torch::zeros({gridBatch.totalLeaves() + 1}, opts);

    auto outUnmaskedPerLeafAcc = tensorAccessor<DeviceTag, int32_t, 1>(unmaskedPerLeaf);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int32_t batchIdx,
                                 int32_t leafIdx,
                                 int32_t,
                                 fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
            _private::countEnabledPerLeafShiftByOneLeafCallback<GridType, TorchRAcc32>(
                batchIdx, leafIdx, gridAccessor, outUnmaskedPerLeafAcc);
        };
        forEachLeafCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=](int32_t batchIdx,
                      int32_t leafIdx,
                      int32_t,
                      fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
            _private::countEnabledPerLeafShiftByOneLeafCallback<GridType, TorchAcc>(
                batchIdx, leafIdx, gridAccessor, outUnmaskedPerLeafAcc);
        };
        forEachLeafCPU<GridType>(1, gridBatch, cb);
    }
    return unmaskedPerLeaf;
}

} // namespace fvdb

#endif // FVDB_DETAIL_OPS_COUNTENABLEDPERLEAF_CUH
