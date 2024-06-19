#include <ATen/OpMathType.h>
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/BezierInterpolationWithGradIterator.h"


namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType DeviceTag, typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void sampleBezierWithGradBackwardCallback(int32_t bidx, int32_t eidx, int32_t cidx,
                                                      JaggedAccessor<ScalarType, 2> points,
                                                      TensorAccessor<ScalarType, 2> gradOutFeatures,
                                                      TensorAccessor<ScalarType, 3> gradOutGradFeatures,
                                                      BatchGridAccessor<GridType> batchAccessor,
                                                      TensorAccessor<ScalarType, 2> outGridData) {
    using MathType = at::opmath_type<ScalarType>;

    auto pointsData = points.data();

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    auto transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->getAccessor();

    const nanovdb::math::Vec3<MathType> xyz = transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                                                        static_cast<MathType>(pointsData[eidx][1]),
                                                        static_cast<MathType>(pointsData[eidx][2]));
    auto gradTransform = transform.template applyGrad<MathType>(xyz);

    for (auto it = BezierInterpolationWithGradIterator<MathType>(xyz); it.isValid(); ++it) {
        if (gridAcc.template get<ActiveOrUnmasked<GridType>>(it->first)) {
            const int64_t indexIjk = gridAcc.getValue(it->first) - 1 + baseOffset;
            MathType addValue = it->second[0] * gradOutFeatures[eidx][cidx];
            #pragma unroll
            for (int dim = 0; dim < 3; ++dim) {
                addValue += it->second[dim + 1] * gradOutGradFeatures[eidx][cidx][dim] * gradTransform[dim];
            }
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cidx], static_cast<ScalarType>(addValue));
            } else {
                // FIXME: (@fwilliams) make this thread safe
                outGridData[indexIjk][cidx] += static_cast<ScalarType>(addValue);
            }
        }
    }
}

template <c10::DeviceType DeviceTag>
torch::Tensor SampleGridBezierWithGradBackward(const GridBatchImpl& batchHdl,
                                               const JaggedTensor& points,
                                               const torch::Tensor& gradOutFeatures,
                                               const torch::Tensor& gradOutGradFeatures,
                                               const torch::Tensor& data) {
    torch::Tensor dataReshape = featureCoalescedView(data);  // [N, -1]

    // FIXME handle more dimensions
    torch::Tensor outGrad = torch::zeros_like(dataReshape);  // [N, -1]
    auto outShape = spliceShape({outGrad.size(0)}, data, 1);  // [B*M, *]

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "SampleGridBezierWithGradBackward", ([&] {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto gradOutFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(gradOutFeatures);
            auto gradOutGradFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 3>(gradOutGradFeatures);
            auto outGridDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGrad);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pts) {
                    sampleBezierWithGradBackwardCallback<DeviceTag, scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, cidx, pts, gradOutFeaturesAcc, gradOutGradFeaturesAcc, batchAcc, outGridDataAcc);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(256, outGrad.size(1), points, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
                    sampleBezierWithGradBackwardCallback<DeviceTag, scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, cidx, pts, gradOutFeaturesAcc, gradOutGradFeaturesAcc, batchAcc, outGridDataAcc);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(outGrad.size(1), points, cb);
            }

        }));
    });
    return outGrad.reshape(outShape);
}


template <>
torch::Tensor dispatchSampleGridBezierWithGradBackward<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                   const JaggedTensor& points,
                                                                   const torch::Tensor& gradOutFeatures,
                                                                   const torch::Tensor& gradOutGradFeatures,
                                                                   const torch::Tensor& data) {
    return SampleGridBezierWithGradBackward<torch::kCUDA>(batchHdl, points, gradOutFeatures, gradOutGradFeatures, data);
}

template <>
torch::Tensor dispatchSampleGridBezierWithGradBackward<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                                  const JaggedTensor& points,
                                                                  const torch::Tensor& gradOutFeatures,
                                                                  const torch::Tensor& gradOutGradFeatures,
                                                                  const torch::Tensor& data) {
    return SampleGridBezierWithGradBackward<torch::kCPU>(batchHdl, points, gradOutFeatures, gradOutGradFeatures, data);
}

} // namespace ops
} // namespace detail
} // namespace fvdb