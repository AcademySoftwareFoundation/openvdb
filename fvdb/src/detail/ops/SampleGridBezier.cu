#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/BezierInterpolationIterator.h"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void sampleBezierCallback(int32_t bidx, int32_t eidx, int32_t cidx,
                                      JaggedAccessor<ScalarType, 2> points,
                                      TensorAccessor<ScalarType, 2> gridData,
                                      BatchGridAccessor<GridType> batchAccessor,
                                      TensorAccessor<ScalarType, 2> outFeatures) {
    using MathType = at::opmath_type<ScalarType>;

    const auto& pointsData = points.data();
    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const VoxelCoordTransform& transform = batchAccessor.primalTransform(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);

    auto gridAcc = gpuGrid->tree().getAccessor();

    const nanovdb::math::Vec3<MathType> xyz = transform.apply(static_cast<MathType>(pointsData[eidx][0]),
                                                        static_cast<MathType>(pointsData[eidx][1]),
                                                        static_cast<MathType>(pointsData[eidx][2]));

    for (auto it = BezierInterpolationIterator<MathType>(xyz); it.isValid(); ++it) {
        const nanovdb::Coord ijk = it->first;
        const bool isActive = gridAcc.template get<ActiveOrUnmasked<GridType>>(ijk);
        const MathType wBezier = it->second;
        const int64_t indexIjk = gridAcc.getValue(ijk) - 1 + baseOffset;
        if (isActive) {
            outFeatures[eidx][cidx] += wBezier * gridData[indexIjk][cidx];
        }
    }
}


template <c10::DeviceType DeviceTag>
std::vector<torch::Tensor> SampleGridBezier(const GridBatchImpl& batchHdl,
                                            const JaggedTensor& points,
                                            const torch::Tensor& gridData) {

    auto opts = torch::TensorOptions().dtype(gridData.dtype()).device(gridData.device()).requires_grad(gridData.requires_grad());
    torch::Tensor gridDataReshape = featureCoalescedView(gridData);                   // [N, -1]
    torch::Tensor outFeatures = torch::zeros({points.size(0), gridDataReshape.size(1)}, opts);  // [B*M, -1]
    auto outShape = spliceShape({points.size(0)}, gridData, 1);  // [B*M, *]

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "SampleGridBezier", ([&] {
            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto gridDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(gridDataReshape);
            auto outFeaturesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outFeatures);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> pts) {
                    sampleBezierCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(256, gridDataReshape.size(1), points, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> pts) {
                    sampleBezierCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, cidx, pts, gridDataAcc, batchAcc, outFeaturesAcc);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(gridDataReshape.size(1), points, cb);
            }
        }));
    });

    return {outFeatures.reshape(outShape)};
}



template <>
std::vector<torch::Tensor> dispatchSampleGridBezier<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                const JaggedTensor& points,
                                                                const torch::Tensor& gridData) {
    return SampleGridBezier<torch::kCUDA>(batchHdl, points, gridData);
}

template <>
std::vector<torch::Tensor> dispatchSampleGridBezier<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                               const JaggedTensor& points,
                                                               const torch::Tensor& gridData) {
    return SampleGridBezier<torch::kCPU>(batchHdl, points, gridData);
}

} // namespace ops
} // namespace detail
} // namespace fvdb

