#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/nanovdb/CustomAccessors.h"


namespace fvdb {
namespace detail {
namespace ops {

/// @brief Callback that runs for every leaf in a single grid
/// @param gpuGrid The nanovdb grid to run on
/// @param li the index of the leaf to process
/// @param outUnmaskedPerLeaf the output tensor storing the number of unmasked voxels in each leaf
template <template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void countEnabledPerLeafOneGridCallback(const nanovdb::NanoGrid<nanovdb::ValueOnIndexMask>* gpuGrid,
                                                           int32_t li, int32_t ci,
                                                           TensorAccessor<int64_t, 1> outUnmaskedPerLeaf) {
    using LeafNodeT = typename nanovdb::NanoTree<nanovdb::ValueOnIndexMask>::LeafNodeType;
    const LeafNodeT& leaf = gpuGrid->tree().template getFirstNode<0>()[li];
    outUnmaskedPerLeaf[li] = (int64_t) leaf.template get<TotalUnmaskedPerLeaf<nanovdb::ValueOnIndexMask>>(1111);
}

/// @brief Callback passed to forEachLeafCUDA to count the number of unmasked voxels per leaf
/// @param batchIdx the batch index of the current leaf being processed
/// @param leafIdx the leaf index in the current grid
/// @param batchAccessor the batch grid accessor
/// @param outUnmaskedPerLeaf the output tensor storing the number of unmasked voxels in each leaf
/// @return
template <template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void countUnmaskedPerLeafCallback(int32_t batchIdx, int32_t leafIdx,
                                                    GridBatchImpl::Accessor<nanovdb::ValueOnIndexMask> batchAccessor,
                                                    TensorAccessor<int64_t, 1> outUnmaskedPerLeaf) {
    using GridType = nanovdb::ValueOnIndexMask;
    const nanovdb::NanoGrid<GridType>* grid = batchAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t numUnmasked = (int64_t) leaf.template get<TotalUnmaskedPerLeaf<GridType>>(1111);
    outUnmaskedPerLeaf[batchAccessor.leafOffset(batchIdx) + leafIdx] = numUnmasked;
}



template <c10::DeviceType DeviceTag>
int64_t CountEnabledVoxels(const GridBatchImpl& batchHdl, int batchIdx) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(batchHdl.device());
    torch::Tensor unmaskedPerLeaf;
    // auto batchAccessor = batchHdl.deviceAccessor<nanovdb::ValueOnIndexMask>();

    if (batchIdx < 0) {
        if (batchHdl.totalLeaves() == 0) {
            return 0;
        }
        // Count the number of unmasked voxels in each leaf node, then cumsum and return the last item to get the total
        unmaskedPerLeaf = torch::empty({batchHdl.totalLeaves()}, opts);
        auto unmaskedPerLeafAcc = tensorAccessor<DeviceTag, int64_t, 1>(unmaskedPerLeaf);
        if constexpr (DeviceTag == torch::kCUDA) {
            auto callback = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t, GridBatchImpl::Accessor<nanovdb::ValueOnIndexMask> batchAcc) {
                countUnmaskedPerLeafCallback<TorchRAcc32>(batchIdx, leafIdx, batchAcc, unmaskedPerLeafAcc);
            };
            forEachLeafCUDA<nanovdb::ValueOnIndexMask>(1024, 1, batchHdl, callback);
        } else {
            auto callback = [=] (int32_t batchIdx, int32_t leafIdx, int32_t, GridBatchImpl::Accessor<nanovdb::ValueOnIndexMask> batchAcc) {
                countUnmaskedPerLeafCallback<TorchAcc>(batchIdx, leafIdx, batchAcc, unmaskedPerLeafAcc);
            };
            forEachLeafCPU<nanovdb::ValueOnIndexMask>(1, batchHdl, callback);
        }
    } else {
        if (batchHdl.numLeaves(batchIdx) == 0) {
            return 0;
        }
        // Count the number of unmasked voxels in each leaf node for a single batch item
        unmaskedPerLeaf = torch::empty({batchHdl.numLeaves(batchIdx)}, opts);
        auto unmaskedPerLeafAcc = tensorAccessor<DeviceTag, int64_t, 1>(unmaskedPerLeaf);
        if constexpr (DeviceTag == torch::kCUDA) {
            auto callback = [=] __device__ (const nanovdb::NanoGrid<nanovdb::ValueOnIndexMask>* grid, int32_t leafIdx, int32_t cIdx) {
                countEnabledPerLeafOneGridCallback<TorchRAcc32>(grid, leafIdx, cIdx, unmaskedPerLeafAcc);
            };
            forEachLeafInOneGridCUDA<nanovdb::ValueOnIndexMask>(1024, 1, batchIdx, batchHdl, callback);
        } else {
            auto callback = [=] (const nanovdb::NanoGrid<nanovdb::ValueOnIndexMask>* grid, int32_t leafIdx, int32_t cIdx) {
                countEnabledPerLeafOneGridCallback<TorchAcc>(grid, leafIdx, cIdx, unmaskedPerLeafAcc);
            };
            forEachLeafInOneGridCPU<nanovdb::ValueOnIndexMask>(1, batchIdx, batchHdl, callback);
        }
    }

    return unmaskedPerLeaf.cumsum(0, torch::kInt64)[-1].cpu().item().to<int64_t>();
}



template <>
int64_t dispatchCountEnabledVoxels<torch::kCUDA>(const GridBatchImpl& batchHdl, int batchIdx) {
    return CountEnabledVoxels<torch::kCUDA>(batchHdl, batchIdx);
}

template <>
int64_t dispatchCountEnabledVoxels<torch::kCPU>(const GridBatchImpl& batchHdl, int batchIdx) {
    return CountEnabledVoxels<torch::kCPU>(batchHdl, batchIdx);
}

} // namespace ops
} // namespace detail
} // namespace fvdb