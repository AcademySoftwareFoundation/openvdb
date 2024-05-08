#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType>
__hostdev__ float _calcDt(ScalarType t, ScalarType coneAngle, ScalarType minStepSize,
                           const ScalarType maxStepSize) {
    return nanovdb::math::Clamp(t * coneAngle, minStepSize, maxStepSize);
}


template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void countSamplesPerRayCallback(int32_t bidx, int32_t eidx,
                                            const JaggedAccessor<ScalarType, 2> rayO,     // [B*M, 3]
                                            const JaggedAccessor<ScalarType, 2> rayD,     // [B*M, 3]
                                            const JaggedAccessor<ScalarType, 1> tMin,     // [B*M,]
                                            const JaggedAccessor<ScalarType, 1> tMax,     // [B*M]
                                            TensorAccessor<int32_t, 1> outRayCounts,      // [B*M]
                                            BatchGridAccessor<GridType> batchAccessor,
                                            ScalarType minStepSize,
                                            ScalarType coneAngle,
                                            bool includeEndpointSegments) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);

    VoxelCoordTransform transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox dualBbox = batchAccessor.dualBbox(bidx);
    auto gridAcc = gpuGrid->getAccessor();

    const auto& rayOi = rayO.data()[eidx];
    const auto& rayDi = rayD.data()[eidx];
    const ScalarType tMini = tMin.data()[eidx];
    const ScalarType tMaxi = tMax.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(
        rayOi[0], rayOi[1], rayOi[2], rayDi[0], rayDi[1], rayDi[2], tMini, tMaxi
    );

    if (!rayVox.clip(dualBbox)) {
        outRayCounts[eidx] = 0;
        return;
    }

    // Count samples along ray
    int32_t numSamples = 0;

    ScalarType maxStepSize = static_cast<ScalarType>(1e10);

    // Count samples
    ScalarType t0 = tMini;
    ScalarType stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
    ScalarType t1;

    // For each contiguous segment of voxels
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, false); it.isValid(); ++it) {

        if (includeEndpointSegments) {
            // Step t0 consistently until it intersects the voxel (t0 is out of the voxel)
            ScalarType distToVox = it->t0 - t0;
            t0 += c10::cuda::compat::floor(distToVox / stepSize) * stepSize;
            t1 = t0 + stepSize;

            if (t0 > it->t1) {
                // A single step would take us past the end of the segment,
                // so we only record one step here.
                numSamples += 1;
                continue;
            }

            if ((t0 - it->t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between t0 and the start of the segment,
                // so we record it as a sample.
                numSamples += 1;
            }

            while (t1 < it->t1) {
                numSamples += 1;
                t0 = t1;
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                t1 += stepSize;
            }

            if ((it->t1 - t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between the end of the segment and t0,
                // so we record it as a sample.
                numSamples += 1;
            }
        } else {
            // Step t0 consistently until it intersects the voxel (tmid is in the voxel)
            ScalarType distToVox = it->t0 - t0;
            t0 = t0 + c10::cuda::compat::floor(distToVox / stepSize + 0.5f) * stepSize;
            stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
            t1 = t0 + stepSize;
            while ((t0 + t1) * 0.5 < it->t1 && (t0 + t1) * 0.5 >= it->t0) {
                numSamples += 1;
                t0 = t1;
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                t1 += stepSize;
            }
        }
    }
    outRayCounts[eidx] = numSamples;
}


template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void generateRaySamplesCallback(int32_t bidx, int32_t eidx,
                                            const JaggedAccessor<ScalarType, 2> rayO,     // [B*M, 3]
                                            const JaggedAccessor<ScalarType, 2> rayD,     // [B*M, 3]
                                            const JaggedAccessor<ScalarType, 1> tMin,     // [B*M,]
                                            const JaggedAccessor<ScalarType, 1> tMax,     // [B*M]
                                            const TensorAccessor<int32_t, 2> packedInfo,  // [B*M, 2]
                                            TensorAccessor<ScalarType, 2> outRayTimes,    // [B*M*S, 2]
                                            TensorAccessor<int32_t, 1> outRayIndices,     // [B*M*S]
                                            BatchGridAccessor<GridType> batchAccessor,
                                            ScalarType minStepSize,
                                            ScalarType coneAngle,
                                            bool includeEndpointSegments) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);

    VoxelCoordTransform transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox dualBbox = batchAccessor.dualBbox(bidx);
    auto gridAcc = gpuGrid->getAccessor();

    const auto& rayOi = rayO.data()[eidx];
    const auto& rayDi = rayD.data()[eidx];
    const ScalarType tMini = tMin.data()[eidx];
    const ScalarType tMaxi = tMax.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(
        rayOi[0], rayOi[1], rayOi[2], rayDi[0], rayDi[1], rayDi[2], tMini, tMaxi
    );

    if (!rayVox.clip(dualBbox)) {
        return;
    }

    // Count samples along ray
    int32_t numSamples = 0;

    ScalarType maxStepSize = static_cast<ScalarType>(1e10);

    // Track ray sample and region of space which it occupies
    ScalarType t0 = tMini;
    ScalarType stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
    ScalarType t1;

    const int32_t rayStartIdx = packedInfo[eidx][0];

    // For each contiguous segment of voxels
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, false); it.isValid(); ++it) {

        if (includeEndpointSegments) {
            // Step t0 consistently until it intersects the voxel
            ScalarType distToVox = it->t0 - t0;
            t0 += c10::cuda::compat::ceil(distToVox / stepSize) * stepSize;
            t1 = t0 + stepSize;

            if (t0 > it->t1) {
                // A single step would take us past the end of the segment,
                // so we only record one step here.
                outRayTimes[rayStartIdx + numSamples][0] = it->t0;
                outRayTimes[rayStartIdx + numSamples][1] = it->t1;
                outRayIndices[rayStartIdx + numSamples] = eidx;
                numSamples += 1;
                continue;
            }

            if ((t0 - it->t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between t0 and the start of the segment,
                // so we record it as a sample.
                outRayTimes[rayStartIdx + numSamples][0] = it->t0;
                outRayTimes[rayStartIdx + numSamples][1] = t0;
                outRayIndices[rayStartIdx + numSamples] = eidx;
                numSamples += 1;
            }

            while (t1 < it->t1) {
                outRayTimes[rayStartIdx + numSamples][0] = t0;
                outRayTimes[rayStartIdx + numSamples][1] = t1;
                outRayIndices[rayStartIdx + numSamples] = eidx;
                numSamples += 1;
                t0 = t1;
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                t1 += stepSize;
            }

            if ((it->t1 - t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between the end of the segment and t0,
                // so we record it as a sample.
                outRayTimes[rayStartIdx + numSamples][0] = t0;
                outRayTimes[rayStartIdx + numSamples][1] = it->t1;
                outRayIndices[rayStartIdx + numSamples] = eidx;
                numSamples += 1;
            }
        } else {
            // Step t0 consistently until it intersects the voxel (tmid is in the voxel)
            ScalarType distToVox = it->t0 - t0;
            t0 = t0 + c10::cuda::compat::floor(distToVox / stepSize + 0.5f) * stepSize;
            stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
            t1 = t0 + stepSize;
            while ((t0 + t1) * 0.5 < it->t1 && (t0 + t1) * 0.5 >= it->t0) {
                outRayTimes[rayStartIdx + numSamples][0] = t0;
                outRayTimes[rayStartIdx + numSamples][1] = t1;
                outRayIndices[rayStartIdx + numSamples] = eidx;
                numSamples += 1;
                t0 = t1;
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                t1 += stepSize;
            }
        }
    }
}

template <c10::DeviceType DeviceTag>
std::vector<JaggedTensor> UniformRaySamples(const GridBatchImpl& batchHdl,
                                                  const JaggedTensor& rayOrigins,
                                                  const JaggedTensor& rayDirections,
                                                  const JaggedTensor& tMin,
                                                  const JaggedTensor& tMax,
                                                  const double minStepSize,
                                                  const double coneAngle,
                                                  const bool includeEndpointSegments) {

    batchHdl.checkDevice(rayOrigins);
    batchHdl.checkDevice(rayDirections);
    batchHdl.checkDevice(tMin);
    batchHdl.checkDevice(tMax);
    TORCH_CHECK_TYPE(rayOrigins.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayDirections.is_floating_point(), "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(tMin.is_floating_point(), "tmin must have a floating point type");
    TORCH_CHECK_TYPE(tMax.is_floating_point(), "tmax must have a floating point type");

    TORCH_CHECK_TYPE(rayOrigins.dtype() == rayDirections.dtype(), "all tensors must have the same type");
    TORCH_CHECK_TYPE(tMin.dtype() == tMin.dtype(), "all tensors must have the same type");
    TORCH_CHECK_TYPE(tMin.dtype() == rayOrigins.dtype(), "all tensors must have the same type");

    TORCH_CHECK(rayOrigins.dim() == 2, std::string("Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ") +
                                 std::to_string(rayOrigins.dim()) + " dimensions");
    TORCH_CHECK(rayDirections.dim() == 2, std::string("Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ") +
                                 std::to_string(rayDirections.dim()) + " dimensions");
    TORCH_CHECK(tMin.dim() == 1, std::string("Expected tmin to have 1 dimension (shape (n,)) but got ") +
                                 std::to_string(tMin.dim()) + " dimensions");
    TORCH_CHECK(tMax.dim() == 1, std::string("Expected tmin to have 1 dimension (shape (n,)) but got ") +
                                 std::to_string(tMax.dim()) + " dimensions");
    TORCH_CHECK(rayOrigins.size(0) == tMin.size(0),
                "ray_origins and tmin must have the same size in dimension 0 but got " +
                std::to_string(rayOrigins.size(0)) + " and " + std::to_string(tMin.size(0)));
    TORCH_CHECK(rayOrigins.size(0) == tMax.size(0),
                "ray_origins and tmin must have the same size in dimension 0 but got " +
                std::to_string(rayOrigins.size(0)) + " and " + std::to_string(tMin.size(0)));
    TORCH_CHECK(rayOrigins.size(0) == rayDirections.size(0),
                "ray_origins and ray_directions must have the same size in dimension 0 but got " +
                std::to_string(rayOrigins.size(0)) + " and " + std::to_string(rayDirections.size(0)));
    TORCH_CHECK(minStepSize > 0.0, "minStepSize must be positive");
    TORCH_CHECK(coneAngle >= 0.0, "coneAngle must be none negitive");


    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> std::vector<JaggedTensor> {
        return AT_DISPATCH_FLOATING_TYPES_AND_HALF(rayOrigins.scalar_type(), "UniformRaySamples", [&]() -> std::vector<JaggedTensor> {
            int64_t numThreads = 256 + 128;
            if constexpr (nanovdb::util::is_same<scalar_t, double>::value) {
                numThreads = 256;
            }
            auto optsF = torch::TensorOptions().dtype(rayOrigins.dtype()).device(rayOrigins.device());
            auto optsI = torch::TensorOptions().dtype(torch::kInt32).device(rayOrigins.device());
            auto optsI16 = torch::TensorOptions().dtype(torch::kInt16).device(rayOrigins.device());

            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto rayOriginsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayOrigins);
            auto rayDirectionsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);

            auto tMinAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(tMin);
            auto tMaxAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(tMax);

            // Count number of segments along each ray
            torch::Tensor rayCounts = torch::zeros({rayOrigins.size(0)}, optsI);  // [B*M]
            auto outCountsAcc = tensorAccessor<DeviceTag, int32_t, 1>(rayCounts);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> rOA) {
                    countSamplesPerRayCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, rOA, rayDirectionsAcc, tMinAcc, tMaxAcc, outCountsAcc, batchAcc, minStepSize, coneAngle, includeEndpointSegments);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                    countSamplesPerRayCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, rOA, rayDirectionsAcc, tMinAcc, tMaxAcc, outCountsAcc, batchAcc, minStepSize, coneAngle, includeEndpointSegments);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb);
            }

            // Compute pack offsets
            torch::Tensor cumRayCount = rayCounts.cumsum(0, torch::kInt32);  // [B*M]
            torch::Tensor outPackInfo = torch::stack({cumRayCount - rayCounts, rayCounts}, 1);  // [B*M, 2]
            const int32_t totalSamples = cumRayCount[cumRayCount.size(0) - 1].item<int32_t>();

            // Allocate output tensors
            torch::Tensor outRayTimes = torch::zeros({totalSamples, 2}, optsF);   // [B*M*S, 2]
            torch::Tensor outBIdx = rayOrigins.jagged_dim() == 0 ? torch::empty({0}, optsI16) : torch::zeros({totalSamples}, optsI16); // [B*M*S]
            torch::Tensor outRIdx = torch::zeros({totalSamples}, optsI); // [B*M*S]

            // Compute output voxels and times
            auto packInfoAcc = tensorAccessor<DeviceTag, int32_t, 2>(outPackInfo);
            auto outBIdxAcc = tensorAccessor<DeviceTag, int16_t, 1>(outBIdx);

            auto outRayTimesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outRayTimes);
            auto outRayIndicesAcc = tensorAccessor<DeviceTag, int32_t, 1>(outRIdx);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> roA) {
                    generateRaySamplesCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, roA, rayDirectionsAcc, tMinAcc, tMaxAcc, packInfoAcc, outRayTimesAcc, outRayIndicesAcc,
                        batchAcc, minStepSize, coneAngle, includeEndpointSegments);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> roA) {
                    generateRaySamplesCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, roA, rayDirectionsAcc, tMinAcc, tMaxAcc, packInfoAcc, outRayTimesAcc, outRayIndicesAcc, batchAcc,
                        minStepSize, coneAngle, includeEndpointSegments);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb);
            }

            // FIXME: outBIdx does not seem correct for batches...
            return {
                rayOrigins.jagged_like(outPackInfo),
                JaggedTensor::from_data_and_jidx(outRIdx, outBIdx, batchHdl.batchSize()),
                JaggedTensor::from_data_and_jidx(outRayTimes, outBIdx, batchHdl.batchSize()),
            };
        });
    });
}



template <>
std::vector<JaggedTensor> dispatchUniformRaySamples<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                const JaggedTensor& rayO,
                                                                const JaggedTensor& rayD,
                                                                const JaggedTensor& tMin,
                                                                const JaggedTensor& tMax,
                                                                const double minStepSize,
                                                                const double coneAngle,
                                                                const bool includeEndSegments) {
    return UniformRaySamples<torch::kCUDA>(batchHdl, rayO, rayD, tMin, tMax, minStepSize, coneAngle, includeEndSegments);
}
template <>
std::vector<JaggedTensor> dispatchUniformRaySamples<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                               const JaggedTensor& rayO,
                                                               const JaggedTensor& rayD,
                                                               const JaggedTensor& tMin,
                                                               const JaggedTensor& tMax,
                                                               const double minStepSize,
                                                               const double coneAngle,
                                                               const bool includeEndSegments) {
    return UniformRaySamples<torch::kCPU>(batchHdl, rayO, rayD, tMin, tMax, minStepSize, coneAngle, includeEndSegments);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
