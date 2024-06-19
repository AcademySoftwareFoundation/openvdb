#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/BezierInterpolationIterator.h"


namespace fvdb {
namespace detail {
namespace ops {

template <c10::DeviceType DeviceTag, typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void splatIntoGridBezierCallback(int32_t bidx, int32_t eidx, int32_t cidx,
                                             JaggedAccessor<ScalarType, 2> points,
                                             TensorAccessor<ScalarType, 2> pointsData,
                                             BatchGridAccessor<GridType> batchAccessor,
                                             TensorAccessor<at::opmath_type<ScalarType>, 2> outGridData) {
    using MathType = at::opmath_type<ScalarType>;

    const auto pointCoords = points.data();
    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    auto gridAcc = gpuGrid->getAccessor();
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);
    const VoxelCoordTransform& transform = batchAccessor.primalTransform(bidx);

    const nanovdb::math::Vec3<MathType> xyz = transform.apply(
        static_cast<MathType>(pointCoords[eidx][0]),
        static_cast<MathType>(pointCoords[eidx][1]),
        static_cast<MathType>(pointCoords[eidx][2])
    );

    #pragma unroll
    for (auto it = BezierInterpolationIterator<MathType>(xyz); it.isValid(); ++it) {
        if (gridAcc.template get<ActiveOrUnmasked<GridType>>(it->first)) {
            const int64_t indexIjk = (int64_t) gridAcc.getValue(it->first) - 1 + baseOffset;
            const MathType addValue = it->second * static_cast<MathType>(pointsData[eidx][cidx]);
            if constexpr (DeviceTag == torch::kCUDA) {
                gpuAtomicAddNoReturn(&outGridData[indexIjk][cidx], addValue);
            } else {
                // FIXME: (@fwilliams) Make me threadsafe
                outGridData[indexIjk][cidx] += addValue;
            }
        }
    }
}


template <c10::DeviceType DeviceTag>
torch::Tensor SplatIntoGridBezier(const GridBatchImpl& batchHdl,
                                  const JaggedTensor& points,
                                  const torch::Tensor& pointsData) {

    int64_t numOutputValues = batchHdl.totalVoxels();
    auto opts = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outGridData = torch::zeros(spliceShape({numOutputValues}, pointsData, 1), opts); // [N, *]

    torch::Tensor pointsDataReshape = featureCoalescedView(pointsData);                     // [B*M, -1]
    torch::Tensor outGridDataReshape = featureCoalescedView(outGridData);        // [N, -1]

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "SplatIntoGridBezier", ([&] {

            torch::Tensor _outGridData;
            if (points.scalar_type() == at::kHalf) {
                _outGridData = torch::zeros_like(outGridDataReshape, outGridData.options().dtype(torch::kFloat32));
            } else {
                _outGridData = outGridDataReshape;
            }

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto pointsDataAcc = tensorAccessor<DeviceTag, scalar_t, 2>(pointsData);
            auto outGridDataAcc = tensorAccessor<DeviceTag, at::opmath_type<scalar_t>, 2>(_outGridData);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    splatIntoGridBezierCallback<DeviceTag, scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, cidx, ptsA, pointsDataAcc, batchAcc, outGridDataAcc);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(256, pointsData.size(1), points, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    splatIntoGridBezierCallback<DeviceTag, scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, cidx, ptsA, pointsDataAcc, batchAcc, outGridDataAcc);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(pointsData.size(1), points, cb);
            }

            if (points.scalar_type() == at::kHalf) {
                outGridData.copy_(_outGridData);
            }
        }));
    });

    return outGridData;
}




template <>
torch::Tensor dispatchSplatIntoGridBezier<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                      const JaggedTensor& points,
                                                      const torch::Tensor& pointsData) {
    return SplatIntoGridBezier<torch::kCUDA>(batchHdl, points, pointsData);
}


template <>
torch::Tensor dispatchSplatIntoGridBezier<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                     const JaggedTensor& points,
                                                     const torch::Tensor& pointsData) {
    return SplatIntoGridBezier<torch::kCPU>(batchHdl, points, pointsData);
}


} // namespace ops
} // namespace detail
} // namespace fvdb

