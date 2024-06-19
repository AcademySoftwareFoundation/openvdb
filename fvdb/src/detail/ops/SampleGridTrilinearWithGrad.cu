#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/TrilinearInterpolationWithGradIterator.h"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void sampleTrilinearWithGradCallback(int32_t bidx, int32_t eidx, int32_t cidx,
                                                 JaggedAccessor<ScalarType, 2> points,
                                                 TensorAccessor<ScalarType, 2> gridData,
                                                 BatchGridAccessor<GridType> batchAccessor,
                                                 TensorAccessor<ScalarType, 2> outFeatures,
                                                 TensorAccessor<ScalarType, 3> outGradFeatures) {
    using MathType = at::opmath_type<ScalarType>;

    const auto& pointsData = points.data();

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const VoxelCoordTransform& transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::math::Vec3<MathType> xyz = transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                                                        static_cast<MathType>(pointsData[eidx][1]),
                                                        static_cast<MathType>(pointsData[eidx][2]));

    auto gradTransform = transform.template applyGrad<MathType>(xyz);

    #pragma unroll
    for (auto it = TrilinearInterpolationWithGradIterator<MathType>(xyz); it.isValid(); ++it) {
        const nanovdb::math::Vec4<MathType> wXYZ = it->second;
        const nanovdb::Coord ijk = it->first;
        if (gridAcc.template get<ActiveOrUnmasked<GridType>>(ijk)) {
            const int64_t indexIjk = gridAcc.getValue(ijk) - 1 + baseOffset;
            outFeatures[eidx][cidx] += wXYZ[0] * gridData[indexIjk][cidx];
            #pragma unroll
            for (int dim = 0; dim < 3; ++ dim) {
                outGradFeatures[eidx][cidx][dim] += wXYZ[dim + 1] * gridData[indexIjk][cidx] * gradTransform[dim];
            }
        }
    }
}


template <c10::DeviceType DeviceTag>
std::vector<torch::Tensor> SampleGridTrilinearWithGrad(const GridBatchImpl& batchHdl,
                                                       const JaggedTensor& points,
                                                       const torch::Tensor& gridData) {
    auto opts = torch::TensorOptions().dtype(gridData.dtype()).device(gridData.device()).requires_grad(gridData.requires_grad());
    torch::Tensor gridDataReshape = featureCoalescedView(gridData);                           // [B*N, -1]
    torch::Tensor outFeatures = torch::zeros({points.size(0), gridDataReshape.size(1)}, opts);  // [B*M, -1]
    torch::Tensor outGradFeatures = torch::zeros({points.size(0), gridDataReshape.size(1), 3}, opts);  // [B*M, -1, 3]
    std::vector<int64_t> outShape = spliceShape({points.size(0)}, gridData, 1);  // [B*M, *]
    std::vector<int64_t> outGradShape = outShape;
    outGradShape.push_back(3);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "SampleGridTrilinearWithGrad", ([&] {
            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto gridDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
            auto outFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);
            auto outGradFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 3>(outGradFeatures);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pts) {
                    sampleTrilinearWithGradCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, outGradFeaturesAcc);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(256, gridData.size(1), points, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
                    sampleTrilinearWithGradCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc, outGradFeaturesAcc);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(gridData.size(1), points, cb);
            }
        }));
    });

    return {
        outFeatures.reshape(outShape),
        outGradFeatures.reshape(outGradShape)
    };
}


template <>
std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                           const JaggedTensor& points,
                                                                           const torch::Tensor& gridData) {
    return SampleGridTrilinearWithGrad<torch::kCUDA>(batchHdl, points, gridData);
}

template <>
std::vector<torch::Tensor> dispatchSampleGridTrilinearWithGrad<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                                          const JaggedTensor& points,
                                                                          const torch::Tensor& gridData) {
    return SampleGridTrilinearWithGrad<torch::kCPU>(batchHdl, points, gridData);
}

} // namespace ops
} // namespace detail
} // namespace fvdb