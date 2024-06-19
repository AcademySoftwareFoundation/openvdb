#include <c10/cuda/CUDAException.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/OpMathType.h>

#include "detail/utils/cuda/Utils.cuh"
#include "Types.h"


namespace fvdb {
namespace detail {
namespace ops {


template <typename ScalarType, bool IsTouch, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void cubesInGridCallback(int32_t bidx, int32_t eidx,
                                            JaggedAccessor<ScalarType, 2> points,
                                            TensorAccessor<bool, 1> outMask,
                                            BatchGridAccessor<GridType> batchAccessor,
                                            nanovdb::math::Vec3<at::opmath_type<ScalarType>> deltaStart,
                                            nanovdb::math::Vec3<at::opmath_type<ScalarType>> deltaEnd,
                                            bool ignoreMasked) {

    using MathType = at::opmath_type<ScalarType>;

    const auto* gpuGrid = batchAccessor.grid(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    const VoxelCoordTransform& transform = batchAccessor.primalTransform(bidx);

    const auto pointCoord = points.data()[eidx];
    const nanovdb::math::Vec3<MathType> xyz_s = transform.apply(static_cast<MathType>(pointCoord[0]) + deltaStart[0],
                                                          static_cast<MathType>(pointCoord[1]) + deltaStart[1],
                                                          static_cast<MathType>(pointCoord[2]) + deltaStart[2]);
    const nanovdb::math::Vec3<MathType> xyz_e = transform.apply(static_cast<MathType>(pointCoord[0]) + deltaEnd[0],
                                                          static_cast<MathType>(pointCoord[1]) + deltaEnd[1],
                                                          static_cast<MathType>(pointCoord[2]) + deltaEnd[2]);
    const nanovdb::Coord s = xyz_s.round();
    const nanovdb::Coord e = xyz_e.round();

    bool mask = !IsTouch;
    for (int vx = s[0]; vx <= e[0]; ++vx) {
        for (int vy = s[1]; vy <= e[1]; ++vy) {
            for (int vz = s[2]; vz <= e[2]; ++vz) {
                const nanovdb::Coord vox{vx, vy, vz};
                const bool isActive = ignoreMasked ? primalAcc.isActive(vox) : primalAcc.template get<ActiveOrUnmasked<GridType>>(vox);
                // IsTouch -> isActive? True : mask
                // !IsTouch -> notActive? False : mask
                mask = (IsTouch == isActive) ? IsTouch : mask;
            }
        }
    }
    outMask[eidx] = mask;
}


template <c10::DeviceType DeviceTag, bool IsTouch>
JaggedTensor CubesInGrid(const GridBatchImpl& batchHdl,
                               const JaggedTensor& cubeCenters,
                               const Vec3dOrScalar& padMinTensor,
                               const Vec3dOrScalar& padMaxTensor,
                               bool ignoreDisabledVoxels) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(cubeCenters);
    TORCH_CHECK_TYPE(cubeCenters.is_floating_point(), "cubeCenters must have a floating point type");
    TORCH_CHECK(cubeCenters.dim() == 2, std::string("Expected cubeCenters to have 2 dimensions (shape (n, 3)) but got ") +
                                        std::to_string(cubeCenters.dim()) + " dimensions");
    TORCH_CHECK(cubeCenters.size(0) > 0, "Empty tensor (cubeCenters)");
    TORCH_CHECK(cubeCenters.size(1) == 3,
                "Expected 3 dimensional cubeCenters but got cubeCenters.shape[1] = " +
                std::to_string(cubeCenters.size(1)));

    const nanovdb::Vec3d padMin = padMinTensor.value();
    const nanovdb::Vec3d padMax = padMaxTensor.value();

    auto opts = torch::TensorOptions().dtype(torch::kBool).device(cubeCenters.device());
    torch::Tensor outMask = torch::empty({cubeCenters.size(0)}, opts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(cubeCenters.scalar_type(), "CubesInGrid", [&]() {
            using opmath_t = at::opmath_type<scalar_t>;

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outMaskAccessor = tensorAccessor<DeviceTag, bool, 1>(outMask);
            nanovdb::math::Vec3<opmath_t> dstart(padMin);
            nanovdb::math::Vec3<opmath_t> dend(padMax);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                    cubesInGridCallback<scalar_t, IsTouch, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, ptsA, outMaskAccessor, batchAcc, dstart, dend, ignoreDisabledVoxels);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(512, 1, cubeCenters, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
                    cubesInGridCallback<scalar_t, IsTouch, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, ptsA, outMaskAccessor, batchAcc, dstart, dend, ignoreDisabledVoxels);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, cubeCenters, cb);
            }
        });
    });

    return cubeCenters.jagged_like(outMask);
}


template <>
JaggedTensor dispatchCubesInGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                             const JaggedTensor& cubeCenters,
                                             const Vec3dOrScalar& padMin,
                                             const Vec3dOrScalar& padMax,
                                             bool ignoreDisabledVoxels) {
    return CubesInGrid<torch::kCUDA, false>(batchHdl, cubeCenters, padMin, padMax, ignoreDisabledVoxels);
}


template <>
JaggedTensor dispatchCubesInGrid<torch::kCPU>(const GridBatchImpl& batchHdl,
                                            const JaggedTensor& cubeCenters,
                                            const Vec3dOrScalar& padMin,
                                            const Vec3dOrScalar& padMax,
                                            bool ignoreDisabledVoxels) {
    return CubesInGrid<torch::kCPU, false>(batchHdl, cubeCenters, padMin, padMax, ignoreDisabledVoxels);
}

template <>
JaggedTensor dispatchCubesIntersectGrid<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                             const JaggedTensor& cubeCenters,
                                             const Vec3dOrScalar& padMin,
                                             const Vec3dOrScalar& padMax,
                                             bool ignoreDisabledVoxels) {
    return CubesInGrid<torch::kCUDA, true>(batchHdl, cubeCenters, padMin, padMax, ignoreDisabledVoxels);
}


template <>
JaggedTensor dispatchCubesIntersectGrid<torch::kCPU>(const GridBatchImpl& batchHdl,
                                            const JaggedTensor& cubeCenters,
                                            const Vec3dOrScalar& padMin,
                                            const Vec3dOrScalar& padMax,
                                            bool ignoreDisabledVoxels) {
    return CubesInGrid<torch::kCPU, true>(batchHdl, cubeCenters, padMin, padMax, ignoreDisabledVoxels);
}


} // namespace ops
} // namespace detail
} // namespace fvdb
