#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {


template <typename Dtype, typename GridType, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void avgPoolVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                             GridBatchImpl::Accessor<GridType> coarseBatchAccessor,
                                             GridBatchImpl::Accessor<GridType> fineBatchAccessor,
                                             const TensorAccessor<Dtype, 2> fineData,
                                             TensorAccessor<Dtype, 2> outCoarseData,
                                             nanovdb::Coord poolingFactor,
                                             nanovdb::Coord stride,
                                             Dtype avgFactor) {

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
    Dtype avgValue = static_cast<Dtype>(0.0);

    for (unsigned i = 0; i < poolingFactor[0]; i += 1) {
        for (unsigned j = 0; j < poolingFactor[1]; j += 1) {
            for (unsigned k = 0; k < poolingFactor[2]; k += 1) {
                nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                if (!fineGridAcc.template get<ActiveOrUnmasked<GridType>>(fineIjk)) {
                    continue;
                }
                const int64_t fineIndex = (int64_t) fineGridAcc.getValue(fineIjk) + fineBaseOffset - 1;
                avgValue += fineData[fineIndex][channelIdx];
            }
        }
    }

    outCoarseData[coarseIndex][channelIdx] = avgValue * avgFactor;
}


template <typename Dtype, typename GridType, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void avgPoolBackardVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                    GridBatchImpl::Accessor<GridType> coarseBatchAccessor,
                                                    GridBatchImpl::Accessor<GridType> fineBatchAccessor,
                                                    const TensorAccessor<Dtype, 2> fineData,
                                                    const TensorAccessor<Dtype, 2> coarseGradOut,
                                                    TensorAccessor<Dtype, 2> outFineGradIn,
                                                    nanovdb::Coord poolingFactor,
                                                    nanovdb::Coord stride,
                                                    Dtype avgFactor) {

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

    for (unsigned i = 0; i < poolingFactor[0]; i += 1) {
        for (unsigned j = 0; j < poolingFactor[1]; j += 1) {
            for (unsigned k = 0; k < poolingFactor[2]; k += 1) {
                nanovdb::Coord fineIjk = fineIjk0 + nanovdb::Coord(i, j, k);
                if (!fineGridAcc.template get<ActiveOrUnmasked<GridType>>(fineIjk)) {
                    continue;
                }
                const int64_t fineIndex = (int64_t) fineGridAcc.getValue(fineIjk) + fineBaseOffset - 1;
                outFineGradIn[fineIndex][channelIdx] = coarseGradOut[leafValue - 1 + coarseBaseOffset][channelIdx] * avgFactor;
            }
        }
    }

}




template <c10::DeviceType DeviceTag>
torch::Tensor DownsampleGridAvgPool(const GridBatchImpl& fineBatchHdl,
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
    torch::Tensor outCoarseData = torch::zeros(spliceShape({numOutputValues}, fineData), opts);

    torch::Tensor fineDataReshape = featureCoalescedView(fineData);
    torch::Tensor outCoarseDataReshape = featureCoalescedView(outCoarseData);
    double avgFactor = 1.0 / (poolingFactor[0] * poolingFactor[1] * poolingFactor[2]);

    FVDB_DISPATCH_GRID_TYPES(fineBatchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(fineData.scalar_type(), "DownsampleGridAvgPool", [&]() {
            auto fineBatchAcc = gridBatchAccessor<DeviceTag, GridType>(fineBatchHdl);
            auto fineDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(fineDataReshape);
            auto outCoarseDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outCoarseDataReshape);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto avgPoolPerVoxel = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    avgPoolVoxelCallback<scalar_t, GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, channelIdx, coarseBatchAccessor, fineBatchAcc,
                                                                                fineDataAcc, outCoarseDataAcc, poolingFactor, stride,
                                                                                static_cast<scalar_t>(avgFactor));
                };
                forEachVoxelCUDA<GridType>(384, outCoarseData.size(1), coarseBatchHdl, avgPoolPerVoxel);
            } else {
                auto avgPoolPerVoxel = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    avgPoolVoxelCallback<scalar_t, GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, channelIdx, coarseBatchAccessor, fineBatchAcc,
                                                                             fineDataAcc, outCoarseDataAcc, poolingFactor, stride,
                                                                             static_cast<scalar_t>(avgFactor));
                };
                forEachVoxelCPU<GridType>(outCoarseData.size(1), coarseBatchHdl, avgPoolPerVoxel);
            }
        });
    });

    return outCoarseData;
}


template <c10::DeviceType DeviceTag>
torch::Tensor DownsampleGridAvgPoolBackward(const GridBatchImpl& coarseBatchHdl,
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
    double avgFactor = 1.0 / (poolingFactor[0] * poolingFactor[1] * poolingFactor[2]);

    FVDB_DISPATCH_GRID_TYPES(fineBatchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(fineData.scalar_type(), "DownsampleGridAvgPoolBackward", [&]() {

            auto fineBatchAcc = gridBatchAccessor<DeviceTag, GridType>(fineBatchHdl);
            auto fineDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(fineDataReshape);
            auto coarseGradOutAcc = tensorAccessor<DeviceTag, scalar_t, 2>(coarseGradOutReshape);
            auto outFineGradInAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGradInReshape);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    avgPoolBackardVoxelCallback<scalar_t, GridType, TorchRAcc32>(
                        batchIdx, leafIdx, voxelIdx, channelIdx,
                        coarseBatchAccessor, fineBatchAcc,
                        fineDataAcc, coarseGradOutAcc, outFineGradInAcc, poolingFactor, stride,
                        static_cast<scalar_t>(avgFactor));
                };
                forEachVoxelCUDA<GridType>(384, fineData.size(1), coarseBatchHdl, cb);
            } else {
                auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> coarseBatchAccessor) {
                    avgPoolBackardVoxelCallback<scalar_t, GridType, TorchAcc>(
                        batchIdx, leafIdx, voxelIdx, channelIdx,
                        coarseBatchAccessor, fineBatchAcc,
                        fineDataAcc, coarseGradOutAcc, outFineGradInAcc, poolingFactor, stride,
                        static_cast<scalar_t>(avgFactor));
                };
                forEachVoxelCPU<GridType>(fineData.size(1), coarseBatchHdl, cb);
            }
        });
    });
    return outGradInReshape.reshape(spliceShape({fineData.size(0)}, coarseGradOut));
}



template <>
torch::Tensor dispatchDownsampleGridAvgPool<torch::kCUDA>(const GridBatchImpl& fineBatchHdl,
                                                          const GridBatchImpl& coarseBatchHdl,
                                                          const torch::Tensor& fineData,
                                                          nanovdb::Coord poolingFactor,
                                                          nanovdb::Coord stride) {
    return DownsampleGridAvgPool<torch::kCUDA>(fineBatchHdl, coarseBatchHdl, fineData, poolingFactor, stride);
}

template <>
torch::Tensor dispatchDownsampleGridAvgPool<torch::kCPU>(const GridBatchImpl& fineBatchHdl,
                                                         const GridBatchImpl& coarseBatchHdl,
                                                         const torch::Tensor& fineData,
                                                         nanovdb::Coord poolingFactor,
                                                         nanovdb::Coord stride) {
    return DownsampleGridAvgPool<torch::kCPU>(fineBatchHdl, coarseBatchHdl, fineData, poolingFactor, stride);
}


template <>
torch::Tensor dispatchDownsampleGridAvgPoolBackward<torch::kCUDA>(const GridBatchImpl& coarseBatchHdl,
                                                                  const GridBatchImpl& fineBatchHdl,
                                                                  const torch::Tensor& fineData,
                                                                  const torch::Tensor& coarseGradOut,
                                                                  nanovdb::Coord poolingFactor,
                                                                  nanovdb::Coord stride) {
    return DownsampleGridAvgPoolBackward<torch::kCUDA>(coarseBatchHdl, fineBatchHdl, fineData, coarseGradOut, poolingFactor, stride);
}

template <>
torch::Tensor dispatchDownsampleGridAvgPoolBackward<torch::kCPU>(const GridBatchImpl& coarseBatchHdl,
                                                                 const GridBatchImpl& fineBatchHdl,
                                                                 const torch::Tensor& fineData,
                                                                 const torch::Tensor& coarseGradOut,
                                                                 nanovdb::Coord poolingFactor,
                                                                 nanovdb::Coord stride) {
    return DownsampleGridAvgPoolBackward<torch::kCPU>(coarseBatchHdl, fineBatchHdl, fineData, coarseGradOut, poolingFactor, stride);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
