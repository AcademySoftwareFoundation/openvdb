#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {


template <typename Dtype, typename GridType, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void maxPoolVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                             GridBatchImpl::Accessor<GridType> coarseBatchAccessor,
                                             GridBatchImpl::Accessor<GridType> fineBatchAccessor,
                                             const TensorAccessor<Dtype, 2> fineData,
                                             TensorAccessor<Dtype, 2> outCoarseData,
                                             nanovdb::Coord poolingFactor,
                                             nanovdb::Coord stride) {

    const nanovdb::NanoGrid<GridType>* coarseGrid = coarseBatchAccessor.grid(batchIdx);
    const nanovdb::NanoGrid<GridType>* fineGrid = fineBatchAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& coarseLeaf = coarseGrid->tree().template getFirstNode<0>()[leafIdx];
    const auto fineGridAcc = fineGrid->getAccessor();
    const int64_t coarseBaseOffset = coarseBatchAccessor.voxelOffset(batchIdx);
    const int64_t fineBaseOffset = fineBatchAccessor.voxelOffset(batchIdx);
    const int64_t coarseVoxelIndex = coarseLeaf.getValue(voxelIdx);

    if (coarseVoxelIndex == 0) {
        return;
    }
    const nanovdb::Coord coarseIjk = coarseLeaf.offsetToGlobalCoord(voxelIdx);
    const nanovdb::Coord fineIjk0(coarseIjk[0] * stride[0],
                                  coarseIjk[1] * stride[1],
                                  coarseIjk[2] * stride[2]);
    const int64_t coarseIndex = coarseVoxelIndex - static_cast<int64_t>(1) + coarseBaseOffset;
    outCoarseData[coarseIndex][channelIdx] = -INFINITY;

    for (unsigned i = 0; i < poolingFactor[0]; i += 1) {
        for (unsigned j = 0; j < poolingFactor[1]; j += 1) {
            for (unsigned k = 0; k < poolingFactor[2]; k += 1) {
                nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                if (!fineGridAcc.template get<ActiveOrUnmasked<GridType>>(fineIjk)) {
                    continue;
                }
                const int64_t fineIndex = (int64_t) fineGridAcc.getValue(fineIjk) + fineBaseOffset - 1;
                const Dtype currentValue = outCoarseData[coarseIndex][channelIdx];
                outCoarseData[coarseIndex][channelIdx] =
                    c10::cuda::compat::max(fineData[fineIndex][channelIdx], currentValue);
            }
        }
    }
}


template <typename Dtype, typename GridType, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void maxPoolBackardVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                    GridBatchImpl::Accessor<GridType> coarseBatchAccessor,
                                                    GridBatchImpl::Accessor<GridType> fineBatchAccessor,
                                                    const TensorAccessor<Dtype, 2> fineData,
                                                    const TensorAccessor<Dtype, 2> coarseGradOut,
                                                    TensorAccessor<Dtype, 2> outFineGradIn,
                                                    nanovdb::Coord poolingFactor,
                                                    nanovdb::Coord stride) {

    const nanovdb::NanoGrid<GridType>* coarseGrid = coarseBatchAccessor.grid(batchIdx);
    const nanovdb::NanoGrid<GridType>* fineGrid = fineBatchAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& coarseLeaf = coarseGrid->tree().template getFirstNode<0>()[leafIdx];
    const auto fineGridAcc = fineGrid->getAccessor();
    const int64_t coarseBaseOffset = coarseBatchAccessor.voxelOffset(batchIdx);
    const int64_t fineBaseOffset = fineBatchAccessor.voxelOffset(batchIdx);

    int64_t leafValue = coarseLeaf.getValue(voxelIdx);
    if (leafValue == 0) {
        return;
    }
    const nanovdb::Coord coarseIjk = coarseLeaf.offsetToGlobalCoord(voxelIdx);
    const nanovdb::Coord fineIjk0(coarseIjk[0] * stride[0],
                                  coarseIjk[1] * stride[1],
                                  coarseIjk[2] * stride[2]);

    Dtype maxValue = -INFINITY;
    int64_t maxIndex = -1;

    for (unsigned i = 0; i < poolingFactor[0]; i += 1) {
        for (unsigned j = 0; j < poolingFactor[1]; j += 1) {
            for (unsigned k = 0; k < poolingFactor[2]; k += 1) {
                nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                if (!fineGridAcc.template get<ActiveOrUnmasked<GridType>>(fineIjk)) {
                    continue;
                }

                const int64_t fineIndex = (int64_t) fineGridAcc.getValue(fineIjk) + fineBaseOffset - 1;
                const Dtype fineValue = fineData[fineIndex][channelIdx];
                if (fineValue > maxValue) {
                    maxIndex = fineIndex;
                    maxValue = fineValue;
                }
            }
        }
    }

    if (maxIndex >= 0) {
        outFineGradIn[maxIndex][channelIdx] = coarseGradOut[leafValue - 1 + coarseBaseOffset][channelIdx];
    }
}




template <c10::DeviceType DeviceTag>
torch::Tensor DownsampleGridMaxPool(const GridBatchImpl& fineBatchHdl,
                                    const GridBatchImpl& coarseBatchHdl,
                                    const torch::Tensor& fineData,
                                    nanovdb::Coord poolingFactor,
                                    nanovdb::Coord stride) {
    TORCH_CHECK(fineBatchHdl.device() == coarseBatchHdl.device(), "coarse_grid and fine_grid must be on the same device");
    for (int i = 0; i < 3; i += 1) {
        TORCH_CHECK_VALUE(poolingFactor[i] > 0, "pooling_factor must be greater than 0");
        TORCH_CHECK_VALUE(stride[i] >= 0, "stride must be greater than or equal to 0");
        if (stride[i] == 0) {
            stride[i] = poolingFactor[i];
        }
    }

    coarseBatchHdl.checkNonEmptyGrid();
    fineBatchHdl.checkNonEmptyGrid();
    coarseBatchHdl.checkDevice(fineData);

    TORCH_CHECK(fineData.dim() > 1, "fine_data must have more than one dimension. i.e. have shape (num_voxels, *)");
    TORCH_CHECK(fineData.size(0) == (int64_t) fineBatchHdl.totalVoxels(), "fine_data must have the same number of voxels as fine_grid");

    int64_t numOutputValues = coarseBatchHdl.totalVoxels();
    auto opts = torch::TensorOptions().dtype(fineData.dtype()).device(fineData.device());
    torch::Tensor outCoarseData = torch::empty(spliceShape({numOutputValues}, fineData), opts);

    torch::Tensor fineDataReshape = featureCoalescedView(fineData);
    torch::Tensor outCoarseDataReshape = featureCoalescedView(outCoarseData);

    FVDB_DISPATCH_GRID_TYPES(fineBatchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(fineData.scalar_type(), "DownsampleGridMaxPool", [&]() {
            auto fineBatchAcc = gridBatchAccessor<DeviceTag, GridType>(fineBatchHdl);
            auto fineDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(fineDataReshape);
            auto outCoarseDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outCoarseDataReshape);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto maxPoolPerVoxel = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    maxPoolVoxelCallback<scalar_t, GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, channelIdx, coarseBatchAccessor, fineBatchAcc,
                                                                                fineDataAcc, outCoarseDataAcc, poolingFactor, stride);
                };
                forEachVoxelCUDA<GridType>(384, outCoarseData.size(1), coarseBatchHdl, maxPoolPerVoxel);
            } else {
                auto maxPoolPerVoxel = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    maxPoolVoxelCallback<scalar_t, GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, channelIdx, coarseBatchAccessor, fineBatchAcc,
                                                                             fineDataAcc, outCoarseDataAcc, poolingFactor, stride);
                };
                forEachVoxelCPU<GridType>(outCoarseData.size(1), coarseBatchHdl, maxPoolPerVoxel);
            }
        });
    });

    return outCoarseData;
}


template <c10::DeviceType DeviceTag>
torch::Tensor DownsampleGridMaxPoolBackward(const GridBatchImpl& coarseBatchHdl,
                                            const GridBatchImpl& fineBatchHdl,
                                            const torch::Tensor& fineData,
                                            const torch::Tensor& coarseGradOut,
                                            nanovdb::Coord poolingFactor,
                                            nanovdb::Coord stride) {

    TORCH_CHECK(fineData.is_contiguous(), "fine_data must be contiguous");
    TORCH_CHECK(coarseGradOut.is_contiguous(), "coarse_grad_out must be contiguous");
    for (int i = 0; i < 3; i += 1) {
        TORCH_CHECK_VALUE(poolingFactor[i] > 0, "pooling_factor must be greater than 0");
        TORCH_CHECK_VALUE(stride[i] >= 0, "stride must be greater than or equal to 0");
        if (stride[i] == 0) {
            stride[i] = poolingFactor[i];
        }
    }
    torch::Tensor fineDataReshape = featureCoalescedView(fineData);
    torch::Tensor coarseGradOutReshape = featureCoalescedView(coarseGradOut);
    torch::Tensor outGradInReshape = torch::zeros_like(fineDataReshape);  // [#fin

    FVDB_DISPATCH_GRID_TYPES(fineBatchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(fineData.scalar_type(), "DownsampleGridMaxPoolBackward", [&]() {

            auto fineBatchAcc = gridBatchAccessor<DeviceTag, GridType>(fineBatchHdl);
            auto fineDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(fineDataReshape);
            auto coarseGradOutAcc = tensorAccessor<DeviceTag, scalar_t, 2>(coarseGradOutReshape);
            auto outFineGradInAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGradInReshape);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    maxPoolBackardVoxelCallback<scalar_t, GridType, TorchRAcc32>(
                        batchIdx, leafIdx, voxelIdx, channelIdx,
                        coarseBatchAccessor, fineBatchAcc,
                        fineDataAcc, coarseGradOutAcc, outFineGradInAcc, poolingFactor, stride);
                };
                forEachVoxelCUDA<GridType>(384, fineData.size(1), coarseBatchHdl, cb);
            } else {
                auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    maxPoolBackardVoxelCallback<scalar_t, GridType, TorchAcc>(
                        batchIdx, leafIdx, voxelIdx, channelIdx,
                        coarseBatchAccessor, fineBatchAcc,
                        fineDataAcc, coarseGradOutAcc, outFineGradInAcc, poolingFactor, stride);
                };
                forEachVoxelCPU<GridType>(fineData.size(1), coarseBatchHdl, cb);
            }
        });
    });
    return outGradInReshape.reshape(spliceShape({fineData.size(0)}, coarseGradOut));
}


template <>
torch::Tensor dispatchDownsampleGridMaxPool<torch::kCUDA>(const GridBatchImpl& fineBatchHdl,
                                                        const GridBatchImpl& coarseBatchHdl,
                                                        const torch::Tensor& fineData,
                                                        nanovdb::Coord poolingFactor,
                                                        nanovdb::Coord stride) {
    return DownsampleGridMaxPool<torch::kCUDA>(fineBatchHdl, coarseBatchHdl, fineData, poolingFactor, stride);
}

template <>
torch::Tensor dispatchDownsampleGridMaxPool<torch::kCPU>(const GridBatchImpl& fineBatchHdl,
                                                       const GridBatchImpl& coarseBatchHdl,
                                                       const torch::Tensor& fineData,
                                                       nanovdb::Coord poolingFactor,
                                                       nanovdb::Coord stride) {
    return DownsampleGridMaxPool<torch::kCPU>(fineBatchHdl, coarseBatchHdl, fineData, poolingFactor, stride);
}


template <>
torch::Tensor dispatchDownsampleGridMaxPoolBackward<torch::kCUDA>(const GridBatchImpl& coarseBatchHdl,
                                                                const GridBatchImpl& fineBatchHdl,
                                                                const torch::Tensor& fineData,
                                                                const torch::Tensor& coarseGradOut,
                                                                nanovdb::Coord poolingFactor,
                                                                nanovdb::Coord stride) {
    return DownsampleGridMaxPoolBackward<torch::kCUDA>(coarseBatchHdl, fineBatchHdl, fineData, coarseGradOut, poolingFactor, stride);
}

template <>
torch::Tensor dispatchDownsampleGridMaxPoolBackward<torch::kCPU>(const GridBatchImpl& coarseBatchHdl,
                                                               const GridBatchImpl& fineBatchHdl,
                                                               const torch::Tensor& fineData,
                                                               const torch::Tensor& coarseGradOut,
                                                               nanovdb::Coord poolingFactor,
                                                               nanovdb::Coord stride) {
    return DownsampleGridMaxPoolBackward<torch::kCPU>(coarseBatchHdl, fineBatchHdl, fineData, coarseGradOut, poolingFactor, stride);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
