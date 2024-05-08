#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"

namespace fvdb {
namespace detail {
namespace ops {

static const int INVALID_SIGN = 10;

template <typename T>
static __forceinline__ __hostdev__ int sgn(const T& val) {
    if (val != val) {
        return INVALID_SIGN;
    }
    return (T(0) < val) - (val < T(0));
}


template <typename ScalarT, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void rayImplicitCallback(int32_t bidx, int32_t eidx,
                                            JaggedAccessor<ScalarT, 2> raysO,
                                            JaggedAccessor<ScalarT, 2> raysD,
                                            JaggedAccessor<ScalarT, 1> gridScalarsJ,
                                            BatchGridAccessor<GridType> batchAcc,
                                            TensorAccessor<ScalarT, 1> outTimes, ScalarT eps) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAcc.grid(bidx);
    const auto gridAcc = gpuGrid->getAccessor();

    const VoxelCoordTransform transform = batchAcc.dualTransform(bidx);
    const nanovdb::CoordBBox dualBbox = batchAcc.dualBbox(bidx);

    const auto rayO = raysO.data()[eidx];
    const auto rayD = raysD.data()[eidx];
    auto gridScalars = gridScalarsJ.data();
    nanovdb::math::Ray<ScalarT> rayVox = transform.applyToRay(
        rayO[0], rayO[1], rayO[2],
        rayD[0], rayD[1], rayD[2]
    );
    if (!rayVox.clip(dualBbox)) {
        outTimes[eidx] = -1.0;
        return;
    }

    int scalarSign = INVALID_SIGN; // Initialize to dummy value to check if first intersection
    ScalarT lastScalar = 0.0;
    ScalarT lastTime = rayVox.t0();
    bool found = false;

    for (auto it = HDDAVoxelIterator<decltype(gridAcc), ScalarT>(rayVox, gridAcc); it.isValid(); it++) {
        const ScalarT t0 = it->second.t0, t1 = it->second.t1;
        const ScalarT deltaT = t1 - t0;
        const nanovdb::Coord& ijk = it->first;
        if (deltaT < eps) {
            continue;
        }

        const int64_t voxelIndex = gridAcc.getValue(ijk) - 1;
        const ScalarT voxelValue = gridScalars[voxelIndex];
        const ScalarT voxelTime = 0.5 * (t0 + t1);
        const int voxelSign = sgn(voxelValue);

        // Francis: This might be faster than the if below since it doesn't require a branch
        // const bool hit = (scalarSign != INVALID_SIGN) && (voxelSign != INVALID_SIGN) && (scalarSign != voxelSign);
        // const ScalarT s0 = lastScalar;
        // const ScalarT s1 = voxelValue;
        // const ScalarT lam = s1 / (s1 - s0); // Linearly interpolate values along rays
        // const ScalarT hitTime = lam * lastTime + (1.0 - lam) * voxelTime;
        // outTimes[eidx] = (hit && !found) ? hitTime : outTimes[eidx];
        // found = found || hit;

        // sign change from a valid value to a valid value, then return
        if (scalarSign != INVALID_SIGN && voxelSign != INVALID_SIGN && scalarSign != voxelSign) { // sign change, return
            const ScalarT s0 = lastScalar;
            const ScalarT s1 = voxelValue;
            const ScalarT lam = s1 / (s1 - s0); // Linearly interpolate values along rays
            outTimes[eidx] = lam * lastTime + (1.0 - lam) * voxelTime;
            found = true;
            break;
        }
        scalarSign = voxelSign;
        lastScalar = voxelValue;
        lastTime = voxelTime;

    }
    if (!found) {
        outTimes[eidx] = -1.0;
    }
}





template <c10::DeviceType DeviceTag>
JaggedTensor RayImplicitIntersection(const GridBatchImpl& batchHdl,
                                           const JaggedTensor& rayO,
                                           const JaggedTensor& rayD,
                                           const JaggedTensor& gridScalars,
                                           float eps) {

    batchHdl.checkDevice(rayO);
    batchHdl.checkDevice(rayD);
    batchHdl.checkDevice(gridScalars);
    TORCH_CHECK_TYPE(rayO.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayD.is_floating_point(), "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(gridScalars.is_floating_point(), "gridScalars must have a floating point type");

    TORCH_CHECK_TYPE(rayO.dtype() == rayD.dtype(), "all tensors must have the same type");
    TORCH_CHECK_TYPE(rayD.dtype() == gridScalars.dtype(), "all tensors must have the same type");

    TORCH_CHECK(rayO.dim() == 2, std::string("Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ") +
                                 std::to_string(rayO.dim()) + " dimensions");
    TORCH_CHECK(rayD.dim() == 2, std::string("Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ") +
                                 std::to_string(rayD.dim()) + " dimensions");

    TORCH_CHECK(rayD.size(0) == rayO.size(0), std::string("Expected ray_origins and ray_directions to have the same shape "));
    TORCH_CHECK(rayO.size(1) == 3, std::string("Expected ray_origins to have shape (n, 3)"));
    TORCH_CHECK(rayD.size(1) == 3, std::string("Expected ray_directions to have shape (n, 3)"));

    TORCH_CHECK(gridScalars.dim() == 1, std::string("Expected grid_scalars to have 1 dimension (shape (num_voxels,)) but got ") +
                                        std::to_string(gridScalars.dim()) + " dimensions");
    TORCH_CHECK(gridScalars.size(0) == batchHdl.totalVoxels(), std::string("Expected one scalar per voxel but got ") +
                                                               std::to_string(gridScalars.size(0)) + " scalars and there are " +
                                                               std::to_string(batchHdl.totalVoxels()) + " voxels.");

    auto optsF = torch::TensorOptions().dtype(rayO.dtype()).device(rayO.device());
    torch::Tensor outTimes = torch::empty({rayO.size(0)}, optsF);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES(rayO.scalar_type(), "RayImplicitIntersection", [&]() {

            int64_t numThreads = 256 + 128;
            if constexpr (nanovdb::util::is_same<scalar_t, double>::value) {
                numThreads = 256;
            }

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto rayDAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayD);
            auto gridScalarsAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(gridScalars);
            auto outTimesAcc = tensorAccessor<DeviceTag, scalar_t, 1>(outTimes);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> rOA) {
                    rayImplicitCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, rOA, rayDAcc, gridScalarsAcc, batchAcc, outTimesAcc, eps);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayO, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                    rayImplicitCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, rOA, rayDAcc, gridScalarsAcc, batchAcc, outTimesAcc, eps);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayO, cb);
            }
        });
    });

    return rayO.jagged_like(outTimes);
}


template <>
JaggedTensor dispatchRayImplicitIntersection<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                         const JaggedTensor& rayOrigins,
                                                         const JaggedTensor& rayDirections,
                                                         const JaggedTensor& gridScalars,
                                                         float eps) {
    return RayImplicitIntersection<torch::kCUDA>(batchHdl, rayOrigins, rayDirections, gridScalars, eps);
}


template <>
JaggedTensor dispatchRayImplicitIntersection<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                        const JaggedTensor& rayOrigins,
                                                        const JaggedTensor& rayDirections,
                                                        const JaggedTensor& gridScalars,
                                                        float eps) {
    return RayImplicitIntersection<torch::kCPU>(batchHdl, rayOrigins, rayDirections, gridScalars, eps);
}

} // namespace ops
} // namespace detail
} // namespace fvdb







