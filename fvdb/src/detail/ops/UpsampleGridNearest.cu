#include <c10/cuda/CUDAException.h>
#include <THC/THCAtomics.cuh>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename Dtype, typename GridType, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void upsampleNearestVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                     GridBatchImpl::Accessor<GridType> coarseBatchAccessor,
                                                     GridBatchImpl::Accessor<GridType> fineBatchAccessor,
                                                     const TensorAccessor<Dtype, 2> coarseData, // [B*N, C]
                                                     TensorAccessor<Dtype, 2> outFineData,      // [B*N, C]
                                                     nanovdb::Coord upsamplingFactor) {
    const nanovdb::NanoGrid<GridType>* coarseGrid = coarseBatchAccessor.grid(batchIdx);
    const nanovdb::NanoGrid<GridType>* fineGrid = fineBatchAccessor.grid(batchIdx);

    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& fineLeaf = fineGrid->tree().template getFirstNode<0>()[leafIdx];

    const auto coarseGridAcc = coarseGrid->getAccessor();
    const int64_t fineBaseOffset = fineBatchAccessor.voxelOffset(batchIdx);
    const int64_t coarseBaseOffset = coarseBatchAccessor.voxelOffset(batchIdx);

    const int64_t fineVoxelIndex = fineLeaf.getValue(voxelIdx);

    const nanovdb::Coord fineIjk = fineLeaf.offsetToGlobalCoord(voxelIdx);
    const nanovdb::Coord coarseIjk = nanovdb::math::Vec3<Dtype>((Dtype) fineIjk[0] / (Dtype) upsamplingFactor[0],
                                                          (Dtype) fineIjk[1] / (Dtype) upsamplingFactor[1],
                                                          (Dtype) fineIjk[2] / (Dtype) upsamplingFactor[2]).floor();
    const int64_t coarseIndex = coarseGridAcc.getValue(coarseIjk) - static_cast<int64_t>(1) + coarseBaseOffset;
    const int64_t fineIndex = fineVoxelIndex - 1 + fineBaseOffset;

    if (fineVoxelIndex > 0 && coarseGridAcc.template get<ActiveOrUnmasked<GridType>>(coarseIjk)) {
        outFineData[fineIndex][channelIdx] = coarseData[coarseIndex][channelIdx];
    }
}


template <typename Dtype, typename GridType, c10::DeviceType DeviceTag, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void upsampleNearestBackwardsVoxelCallback(int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx,
                                                              GridBatchImpl::Accessor<GridType> coarseBatchAccessor,
                                                              GridBatchImpl::Accessor<GridType> fineBatchAccessor,
                                                              const TensorAccessor<Dtype, 2> fineData, // [B*N, C]
                                                              TensorAccessor<Dtype, 2> outCoarseData,  // [B*N, C]
                                                              nanovdb::Coord upsamplingFactor) {
    const nanovdb::NanoGrid<GridType>* coarseGrid = coarseBatchAccessor.grid(batchIdx);
    const nanovdb::NanoGrid<GridType>* fineGrid = fineBatchAccessor.grid(batchIdx);

    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& fineLeaf = fineGrid->tree().template getFirstNode<0>()[leafIdx];

    const auto coarseGridAcc = coarseGrid->getAccessor();
    const int64_t fineBaseOffset = fineBatchAccessor.voxelOffset(batchIdx);
    const int64_t coarseBaseOffset = coarseBatchAccessor.voxelOffset(batchIdx);

    const int64_t fineVoxelIndex = fineLeaf.getValue(voxelIdx);

    const nanovdb::Coord fineIjk = fineLeaf.offsetToGlobalCoord(voxelIdx);
    const nanovdb::Coord coarseIjk = nanovdb::math::Vec3<Dtype>((Dtype) fineIjk[0] / (Dtype) upsamplingFactor[0],
                                                          (Dtype) fineIjk[1] / (Dtype) upsamplingFactor[1],
                                                          (Dtype) fineIjk[2] / (Dtype) upsamplingFactor[2]).floor();
    const int64_t coarseIndex = coarseGridAcc.getValue(coarseIjk) - static_cast<int64_t>(1) + coarseBaseOffset;
    const int64_t fineIndex = fineVoxelIndex - 1 + fineBaseOffset;

    if (fineVoxelIndex > 0 && coarseGridAcc.template get<ActiveOrUnmasked<GridType>>(coarseIjk)) {
        if constexpr (DeviceTag == torch::kCUDA) {
            gpuAtomicAddNoReturn(&outCoarseData[coarseIndex][channelIdx], fineData[fineIndex][channelIdx]);
        } else {
            // FIXME: (@fwilliams) Atomics
            outCoarseData[coarseIndex][channelIdx] += fineData[fineIndex][channelIdx];
        }
    }
}


template <c10::DeviceType DeviceTag>
torch::Tensor UpsampleGridNearest(const GridBatchImpl& coarseBatchAccessor,
                                  const GridBatchImpl& fineBatchAccessor,
                                  const torch::Tensor& coarseData,
                                  nanovdb::Coord upsamplingFactor) {
    coarseBatchAccessor.checkNonEmptyGrid();
    fineBatchAccessor.checkNonEmptyGrid();

    for (int i = 0; i < 3; i += 1) {
        TORCH_CHECK(upsamplingFactor[i] > 0, "upsampling_factor must be greater than 0");
    }
    TORCH_CHECK(coarseData.dim() > 1,
                "coarse_data must have more than one dimension. i.e. have shape (num_voxels, *)");
    TORCH_CHECK(coarseData.size(0) == (int64_t) coarseBatchAccessor.totalVoxels(),
                "coarse_data must have the same number of voxels as coarse_grid");

    const int64_t numOutputValues = fineBatchAccessor.totalVoxels();
    auto opts = torch::TensorOptions().dtype(coarseData.dtype()).device(coarseData.device());
    torch::Tensor outFineData = torch::zeros(spliceShape({numOutputValues}, coarseData), opts);

    torch::Tensor coarseDataReshape = featureCoalescedView(coarseData);
    torch::Tensor outFineDataReshape = featureCoalescedView(outFineData);
    TORCH_CHECK(outFineDataReshape.is_contiguous(), "out_fine_data must be contiguous. This should never happen");

    FVDB_DISPATCH_GRID_TYPES(coarseBatchAccessor, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(coarseData.scalar_type(), "UpsampleGridNearest", [&]() {
            auto coarseBatchAcc = gridBatchAccessor<DeviceTag, GridType>(coarseBatchAccessor);
            auto coarseDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(coarseDataReshape);
            auto outFineDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outFineDataReshape);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto callback = [=] __device__ (
                    int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> fineBatchAccessor) {
                        upsampleNearestVoxelCallback<scalar_t, GridType, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, channelIdx,
                                                                        coarseBatchAcc, fineBatchAccessor,
                                                                        coarseDataAcc, outFineDataAcc, upsamplingFactor);
                    };
                forEachVoxelCUDA<GridType>(640, outFineData.size(1), fineBatchAccessor, callback);
            } else {
                auto callback = [=] (
                    int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> fineBatchAccessor) {
                        upsampleNearestVoxelCallback<scalar_t, GridType, TorchAcc>(batchIdx, leafIdx, voxelIdx, channelIdx,
                                                                         coarseBatchAcc, fineBatchAccessor,
                                                                         coarseDataAcc, outFineDataAcc, upsamplingFactor);
                    };
                forEachVoxelCPU<GridType>(outFineData.size(1), fineBatchAccessor, callback);
            }
        });
    });

    return outFineData;
}


template <c10::DeviceType DeviceTag>
torch::Tensor UpsampleGridNearestBackward(const GridBatchImpl& fineBatchAccessor,
                                          const GridBatchImpl& coarseBatchAccessor,
                                          const torch::Tensor& gradOut,
                                          const torch::Tensor& coarseData,
                                          nanovdb::Coord upsamplingFactor) {

    for (int i = 0; i < 3; i += 1) {
        TORCH_CHECK(upsamplingFactor[i] > 0, "upsampling_factor must be greater than 0");
    }

    torch::Tensor coarseDataReshape = featureCoalescedView(coarseData);
    torch::Tensor gradOutReshape = featureCoalescedView(gradOut);
    torch::Tensor outGradInReshape = torch::zeros_like(coarseDataReshape);

    FVDB_DISPATCH_GRID_TYPES(coarseBatchAccessor, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradOut.scalar_type(), "UpsampleGridNearestBackward", [&]() {
            auto coarseBatchAcc = gridBatchAccessor<DeviceTag, GridType>(coarseBatchAccessor);
            auto gradOutAcc = tensorAccessor<DeviceTag, scalar_t, 2>(gradOutReshape);
            auto outCoarseDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGradInReshape);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto callback = [=] __device__ (
                    int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> fineBatchAccessor) {
                        upsampleNearestBackwardsVoxelCallback<scalar_t, GridType, DeviceTag, TorchRAcc32>(batchIdx, leafIdx, voxelIdx, channelIdx,
                                                                                coarseBatchAcc, fineBatchAccessor,
                                                                                gradOutAcc, outCoarseDataAcc, upsamplingFactor);
                    };
                forEachVoxelCUDA<GridType>(640, outGradInReshape.size(1), fineBatchAccessor, callback);
            } else {
                auto callback = [=] (
                    int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t channelIdx, GridBatchImpl::Accessor<GridType> fineBatchAccessor) {
                        upsampleNearestBackwardsVoxelCallback<scalar_t, GridType, DeviceTag, TorchAcc>(batchIdx, leafIdx, voxelIdx, channelIdx,
                                                                                coarseBatchAcc, fineBatchAccessor,
                                                                                gradOutAcc, outCoarseDataAcc, upsamplingFactor);
                    };
                forEachVoxelCPU<GridType>(outGradInReshape.size(1), fineBatchAccessor, callback);
            }
        });
    });
    torch::Tensor outGradIn = outGradInReshape.reshape(spliceShape({coarseData.size(0)}, gradOut));
    TORCH_CHECK(outGradIn.is_contiguous(), "out_grad_in must be contiguous. This should never happen");
    return outGradIn;
}




template <>
torch::Tensor dispatchUpsampleGridNearest<torch::kCUDA>(const GridBatchImpl& coarseBatchAccessor,
                                                      const GridBatchImpl& fineBatchAccessor,
                                                      const torch::Tensor& coarseData,
                                                      nanovdb::Coord upsamplingFactor) {
    return UpsampleGridNearest<torch::kCUDA>(coarseBatchAccessor, fineBatchAccessor, coarseData, upsamplingFactor);
}


template <>
torch::Tensor dispatchUpsampleGridNearest<torch::kCPU>(const GridBatchImpl& coarseBatchAccessor,
                                                     const GridBatchImpl& fineBatchAccessor,
                                                     const torch::Tensor& coarseData,
                                                     nanovdb::Coord upsamplingFactor) {
    return UpsampleGridNearest<torch::kCPU>(coarseBatchAccessor, fineBatchAccessor, coarseData, upsamplingFactor);
}

template <>
torch::Tensor dispatchUpsampleGridNearestBackward<torch::kCUDA>(const GridBatchImpl& fineBatchAccessor,
                                                              const GridBatchImpl& coarseBatchAccessor,
                                                              const torch::Tensor& gradOut,
                                                              const torch::Tensor& coarseData,
                                                              nanovdb::Coord upsamplingFactor) {
    return UpsampleGridNearestBackward<torch::kCUDA>(fineBatchAccessor, coarseBatchAccessor, gradOut, coarseData, upsamplingFactor);
}

template <>
torch::Tensor dispatchUpsampleGridNearestBackward<torch::kCPU>(const GridBatchImpl& fineBatchAccessor,
                                                             const GridBatchImpl& coarseBatchAccessor,
                                                             const torch::Tensor& gradOut,
                                                             const torch::Tensor& coarseData,
                                                             nanovdb::Coord upsamplingFactor) {
    return UpsampleGridNearestBackward<torch::kCPU>(fineBatchAccessor, coarseBatchAccessor, gradOut, coarseData, upsamplingFactor);
}

} // namespace ops
} // namespace detail
} // namespace fvdb

