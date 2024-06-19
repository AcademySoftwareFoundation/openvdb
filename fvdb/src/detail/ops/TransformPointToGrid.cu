#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void transformPointsToGridCallback(int32_t bidx, int32_t eidx,
                                                      JaggedAccessor<ScalarType, 2> pts,
                                                      TensorAccessor<ScalarType, 2> outPts,
                                                      BatchGridAccessor<GridType> batchAccessor,
                                                      bool primal) {
    const auto tx = primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);

    const auto pt = pts.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci = tx.apply(pt[0], pt[1], pt[2]);
    outPts[eidx][0] = wci[0];
    outPts[eidx][1] = wci[1];
    outPts[eidx][2] = wci[2];
}

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void transformPointsToGridBackwardCallback(int32_t bidx, int32_t eidx,
                                                              JaggedAccessor<ScalarType, 2> gradOut,
                                                              TensorAccessor<ScalarType, 2> outGradIn,
                                                              BatchGridAccessor<GridType> batchAccessor,
                                                              bool primal) {
    const auto tx = primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);
    const auto gradOutI = gradOut.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci = tx.applyGrad(gradOutI[0], gradOutI[1], gradOutI[2]);
    outGradIn[eidx][0] = wci[0] * gradOutI[0];
    outGradIn[eidx][1] = wci[1] * gradOutI[1];
    outGradIn[eidx][2] = wci[2] * gradOutI[2];
}


template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void invTransformPointsToGridCallback(int32_t bidx, int32_t eidx,
                                                         JaggedAccessor<ScalarType, 2> pts,
                                                         TensorAccessor<ScalarType, 2> outPts,
                                                         BatchGridAccessor<GridType> batchAccessor,
                                                         bool primal) {
    const auto tx = primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);
    const auto pt = pts.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci = tx.applyInv(pt[0], pt[1], pt[2]);
    outPts[eidx][0] = wci[0];
    outPts[eidx][1] = wci[1];
    outPts[eidx][2] = wci[2];
}


template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void invTransformPointsToGridBackwardCallback(int32_t bidx, int32_t eidx,
                                                                 JaggedAccessor<ScalarType, 2> gradOut,
                                                                 TensorAccessor<ScalarType, 2> outGradIn,
                                                                 BatchGridAccessor<GridType> batchAccessor,
                                                                 bool primal) {
    const auto tx = primal ? batchAccessor.primalTransform(bidx) : batchAccessor.dualTransform(bidx);
    const auto gradOutI = gradOut.data()[eidx];
    const nanovdb::math::Vec3<ScalarType> wci = tx.applyInvGrad(gradOutI[0], gradOutI[1], gradOutI[2]);
    outGradIn[eidx][0] = wci[0] * gradOutI[0];
    outGradIn[eidx][1] = wci[1] * gradOutI[1];
    outGradIn[eidx][2] = wci[2] * gradOutI[2];
}



template <c10::DeviceType DeviceTag>
torch::Tensor TransformPointsToGrid(const GridBatchImpl& batchHdl,
                                    const JaggedTensor& points,
                                    bool isPrimal) {
    batchHdl.checkDevice(points);
    TORCH_CHECK_VALUE(points.dim() == 2, "points must have shape [B*N, 3]");
    TORCH_CHECK_VALUE(points.size(-1) == 3, "points must have shape [B*N, 3]");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    points.check_valid();

    auto opts = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outCoords = torch::empty({points.size(0), points.size(1)}, opts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "pointsInGrid", [&]() {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outCoordsAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outCoords);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    transformPointsToGridCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(512, 1, points, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    transformPointsToGridCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, points, cb);
            }
        });
    });
    return outCoords;
}


template <c10::DeviceType DeviceTag>
torch::Tensor InvTransformPointsToGrid(const GridBatchImpl& batchHdl,
                                       const JaggedTensor& points,
                                       bool isPrimal) {
    batchHdl.checkDevice(points);
    TORCH_CHECK_VALUE(points.dim() == 2, "points must have shape [B*N, 3]");
    TORCH_CHECK_VALUE(points.size(-1) == 3, "points must have shape [B*N, 3]");
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    points.check_valid();

    auto opts = torch::TensorOptions().dtype(points.dtype()).device(points.device());
    torch::Tensor outCoords = torch::empty({points.size(0), points.size(1)}, opts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "pointsInGrid", [&]() {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outCoordsAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outCoords);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    invTransformPointsToGridCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(512, 1, points, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    invTransformPointsToGridCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, ptsA, outCoordsAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, points, cb);
            }
        });
    });
    return outCoords;
}


template <c10::DeviceType DeviceTag>
torch::Tensor TransformPointsToGridBackward(const GridBatchImpl& batchHdl,
                                            const JaggedTensor& gradOut,
                                            bool isPrimal) {
    torch::Tensor outGradIn = torch::empty_like(gradOut.jdata());

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradOut.scalar_type(), "pointsInGrid", [&]() {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outGradInAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGradIn);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    transformPointsToGridBackwardCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(512, 1, gradOut, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    transformPointsToGridBackwardCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, gradOut, cb);
            }
        });
    });
    return outGradIn;
}


template <c10::DeviceType DeviceTag>
torch::Tensor InvTransformPointsToGridBackward(const GridBatchImpl& batchHdl,
                                               const JaggedTensor& gradOut,
                                               bool isPrimal) {
    torch::Tensor outGradIn = torch::empty_like(gradOut.jdata());

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradOut.scalar_type(), "pointsInGrid", [&]() {

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outGradInAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outGradIn);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    invTransformPointsToGridBackwardCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(512, 1, gradOut, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    invTransformPointsToGridBackwardCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(bidx, eidx, ptsA, outGradInAcc, batchAcc, isPrimal);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, gradOut, cb);
            }
        });
    });
    return outGradIn;
}




template <>
torch::Tensor dispatchTransformPointsToGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                        const JaggedTensor& points,
                                                        bool isPrimal) {
    return TransformPointsToGrid<torch::kCUDA>(batchHdl, points, isPrimal);
}


template <>
torch::Tensor dispatchInvTransformPointsToGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                           const JaggedTensor& points,
                                                           bool isPrimal) {
   return InvTransformPointsToGrid<torch::kCUDA>(batchHdl, points, isPrimal);
}


template <>
torch::Tensor dispatchTransformPointsToGridBackward<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                const JaggedTensor& gradOut,
                                                                bool isPrimal) {
    return TransformPointsToGridBackward<torch::kCUDA>(batchHdl, gradOut, isPrimal);
}


template <>
torch::Tensor dispatchInvTransformPointsToGridBackward<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                   const JaggedTensor& gradOut,
                                                                   bool isPrimal) {
    return InvTransformPointsToGridBackward<torch::kCUDA>(batchHdl, gradOut, isPrimal);
}







template <>
torch::Tensor dispatchTransformPointsToGrid<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                       const JaggedTensor& points,
                                                       bool isPrimal) {
    return TransformPointsToGrid<torch::kCPU>(batchHdl, points, isPrimal);
}


template <>
torch::Tensor dispatchInvTransformPointsToGrid<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                          const JaggedTensor& points,
                                                          bool isPrimal) {
   return InvTransformPointsToGrid<torch::kCPU>(batchHdl, points, isPrimal);
}


template <>
torch::Tensor dispatchTransformPointsToGridBackward<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                               const JaggedTensor& gradOut,
                                                               bool isPrimal) {
    return TransformPointsToGridBackward<torch::kCPU>(batchHdl, gradOut, isPrimal);
}


template <>
torch::Tensor dispatchInvTransformPointsToGridBackward<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                                  const JaggedTensor& gradOut,
                                                                  bool isPrimal) {
    return InvTransformPointsToGridBackward<torch::kCPU>(batchHdl, gradOut, isPrimal);
}


} // namespace ops
} // namespace detail
} // namespace fvdb