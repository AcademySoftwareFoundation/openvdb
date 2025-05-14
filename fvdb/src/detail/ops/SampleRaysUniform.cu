// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/cuda/Utils.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType>
__hostdev__ float
_calcDt(ScalarType t, ScalarType coneAngle, ScalarType minStepSize, const ScalarType maxStepSize) {
    return nanovdb::math::Clamp(t * coneAngle, minStepSize, maxStepSize);
}

template <typename ScalarType, typename GridType,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
countSamplesPerRayCallback(int32_t bidx, int32_t eidx,
                           const JaggedAccessor<ScalarType, 2> rayO,         // [B*M, 3]
                           const JaggedAccessor<ScalarType, 2> rayD,         // [B*M, 3]
                           const JaggedAccessor<ScalarType, 1> tMin,         // [B*M,]
                           const JaggedAccessor<ScalarType, 1> tMax,         // [B*M]
                           TensorAccessor<int32_t, 1>          outRayCounts, // [B*M]
                           BatchGridAccessor<GridType> batchAccessor, ScalarType minStepSize,
                           ScalarType coneAngle, bool includeEndpointSegments, ScalarType eps) {
    const nanovdb::NanoGrid<GridType> *gpuGrid = batchAccessor.grid(bidx);

    VoxelCoordTransform transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox  dualBbox  = batchAccessor.dualBbox(bidx);
    auto                gridAcc   = gpuGrid->getAccessor();

    const auto                    &rayOi  = rayO.data()[eidx];
    const auto                    &rayDi  = rayD.data()[eidx];
    const ScalarType               tMini  = tMin.data()[eidx];
    const ScalarType               tMaxi  = tMax.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(
        rayOi[0], rayOi[1], rayOi[2], rayDi[0], rayDi[1], rayDi[2], tMini, tMaxi);

    if (!rayVox.clip(dualBbox)) {
        outRayCounts[eidx + 1] = 0;
        return;
    }

    // Count samples along ray
    int32_t numSamples = 0;

    ScalarType maxStepSize = static_cast<ScalarType>(1e10);

    // Count samples
    ScalarType t0       = tMini;
    ScalarType stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
    ScalarType t1;

    // For each contiguous segment of voxels
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, false);
         it.isValid(); ++it) {
        const ScalarType deltaT = it->t1 - it->t0;
        if (deltaT < eps) {
            continue;
        }

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
                t0       = t1;
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
            t0       = t0 + c10::cuda::compat::floor(distToVox / stepSize + 0.5f) * stepSize;
            stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
            t1       = t0 + stepSize;
            while ((t0 + t1) * 0.5 < it->t1 && (t0 + t1) * 0.5 >= it->t0) {
                numSamples += 1;
                t0       = t1;
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                t1 += stepSize;
            }
        }
    }
    outRayCounts[eidx + 1] = numSamples;
}

template <typename ScalarType, typename GridType,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
generateRaySamplesCallback(int32_t bidx, int32_t rayIdx,
                           const JaggedAccessor<ScalarType, 2>         rayO,        // [B*M, 3]
                           const JaggedAccessor<ScalarType, 2>         rayD,        // [B*M, 3]
                           const JaggedAccessor<ScalarType, 1>         tMin,        // [B*M,]
                           const JaggedAccessor<ScalarType, 1>         tMax,        // [B*M]
                           const TensorAccessor<fvdb::JOffsetsType, 1> outJOffsets, // [B*M, 2]
                           TensorAccessor<fvdb::JIdxType, 1>           outJIdx,     // [B*M, 2]
                           TensorAccessor<fvdb::JLIdxType, 2>          outJLIdx,    // [B*M, 2]
                           TensorAccessor<ScalarType, 2>               outRayTimes, // [B*M*S, 2]
                           BatchGridAccessor<GridType> batchAccessor, ScalarType minStepSize,
                           ScalarType coneAngle, bool includeEndpointSegments, bool returnMidpoint,
                           ScalarType eps) {
    const nanovdb::NanoGrid<GridType> *gpuGrid = batchAccessor.grid(bidx);

    VoxelCoordTransform transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox  dualBbox  = batchAccessor.dualBbox(bidx);
    auto                gridAcc   = gpuGrid->getAccessor();

    const auto                    &rayOi  = rayO.data()[rayIdx];
    const auto                    &rayDi  = rayD.data()[rayIdx];
    const ScalarType               tMini  = tMin.data()[rayIdx];
    const ScalarType               tMaxi  = tMax.data()[rayIdx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(
        rayOi[0], rayOi[1], rayOi[2], rayDi[0], rayDi[1], rayDi[2], tMini, tMaxi);

    if (outJLIdx.size(0) > 0) {
        const fvdb::JLIdxType batchStartIdx = rayO.offsetStart(bidx);
        outJLIdx[rayIdx][0]                 = bidx;
        outJLIdx[rayIdx][1]                 = rayIdx - batchStartIdx;
    }

    if (!rayVox.clip(dualBbox)) {
        return;
    }

    // Count samples along ray
    fvdb::JOffsetsType numSamples = 0;

    ScalarType maxStepSize = static_cast<ScalarType>(1e10);

    // Track ray sample and region of space which it occupies
    ScalarType t0       = tMini;
    ScalarType stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
    ScalarType t1;

    const fvdb::JOffsetsType rayStartIdx = outJOffsets[rayIdx];

    // For each contiguous segment of voxels
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, false);
         it.isValid(); ++it) {
        const ScalarType deltaT = it->t1 - it->t0;
        if (deltaT < eps) {
            continue;
        }

        if (includeEndpointSegments) {
            // Step t0 consistently until it intersects the voxel
            ScalarType distToVox = it->t0 - t0;
            t0 += c10::cuda::compat::ceil(distToVox / stepSize) * stepSize;
            t1 = t0 + stepSize;

            if (t0 > it->t1) {
                // A single step would take us past the end of the segment,
                // so we only record one step here.
                if (returnMidpoint) {
                    outRayTimes[rayStartIdx + numSamples][0] = (it->t0 + it->t1) / ScalarType(2.0);
                } else {
                    outRayTimes[rayStartIdx + numSamples][0] = it->t0;
                    outRayTimes[rayStartIdx + numSamples][1] = it->t1;
                }
                outJIdx[rayStartIdx + numSamples] = rayIdx;
                numSamples += 1;
                continue;
            }

            if ((t0 - it->t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between t0 and the start of the segment,
                // so we record it as a sample.
                if (returnMidpoint) {
                    outRayTimes[rayStartIdx + numSamples][0] = (it->t0 + t0) / ScalarType(2.0);
                } else {
                    outRayTimes[rayStartIdx + numSamples][0] = it->t0;
                    outRayTimes[rayStartIdx + numSamples][1] = t0;
                }
                outJIdx[rayStartIdx + numSamples] = rayIdx;
                numSamples += 1;
            }

            while (t1 < it->t1) {
                if (returnMidpoint) {
                    outRayTimes[rayStartIdx + numSamples][0] = (t0 + t1) / ScalarType(2.0);
                } else {
                    outRayTimes[rayStartIdx + numSamples][0] = t0;
                    outRayTimes[rayStartIdx + numSamples][1] = t1;
                }
                outJIdx[rayStartIdx + numSamples] = rayIdx;
                numSamples += 1;
                t0       = t1;
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                t1 += stepSize;
            }

            if ((it->t1 - t0) > nanovdb::math::Delta<ScalarType>::value()) {
                // There exists a gap between the end of the segment and t0,
                // so we record it as a sample.
                if (returnMidpoint) {
                    outRayTimes[rayStartIdx + numSamples][0] = (t0 + it->t1) / ScalarType(2.0);
                } else {
                    outRayTimes[rayStartIdx + numSamples][0] = t0;
                    outRayTimes[rayStartIdx + numSamples][1] = it->t1;
                }
                outJIdx[rayStartIdx + numSamples] = rayIdx;
                numSamples += 1;
            }
        } else {
            // Step t0 consistently until it intersects the voxel (tmid is in the voxel)
            ScalarType distToVox = it->t0 - t0;
            t0       = t0 + c10::cuda::compat::floor(distToVox / stepSize + 0.5f) * stepSize;
            stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
            t1       = t0 + stepSize;
            while ((t0 + t1) * 0.5 < it->t1 && (t0 + t1) * 0.5 >= it->t0) {
                if (returnMidpoint) {
                    outRayTimes[rayStartIdx + numSamples][0] = (t0 + t1) / ScalarType(2.0);
                } else {
                    outRayTimes[rayStartIdx + numSamples][0] = t0;
                    outRayTimes[rayStartIdx + numSamples][1] = t1;
                }
                outJIdx[rayStartIdx + numSamples] = rayIdx;
                numSamples += 1;
                t0       = t1;
                stepSize = _calcDt(t0, coneAngle, minStepSize, maxStepSize);
                t1 += stepSize;
            }
        }
    }
}

template <c10::DeviceType DeviceTag>
JaggedTensor
UniformRaySamples(const GridBatchImpl &batchHdl, const JaggedTensor &rayOrigins,
                  const JaggedTensor &rayDirections, const JaggedTensor &tMin,
                  const JaggedTensor &tMax, const double minStepSize, const double coneAngle,
                  const bool includeEndpointSegments, const bool returnMidpoint, const double eps) {
    batchHdl.checkDevice(rayOrigins);
    batchHdl.checkDevice(rayDirections);
    batchHdl.checkDevice(tMin);
    batchHdl.checkDevice(tMax);
    TORCH_CHECK_TYPE(rayOrigins.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayDirections.is_floating_point(),
                     "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(tMin.is_floating_point(), "tmin must have a floating point type");
    TORCH_CHECK_TYPE(tMax.is_floating_point(), "tmax must have a floating point type");

    TORCH_CHECK_VALUE(batchHdl.batchSize() == rayOrigins.num_outer_lists(),
                      "ray_origins must have the same batch size as the grid batch");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == rayDirections.num_outer_lists(),
                      "ray_directions must have the same batch size as the grid batch");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == tMin.num_outer_lists(),
                      "t_min must have the same batch size as the grid batch");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == tMax.num_outer_lists(),
                      "t_max must have the same batch size as the grid batch");

    TORCH_CHECK_TYPE(rayOrigins.dtype() == rayDirections.dtype(),
                     "all tensors must have the same type");
    TORCH_CHECK_TYPE(tMin.dtype() == tMin.dtype(), "all tensors must have the same type");
    TORCH_CHECK_TYPE(tMin.dtype() == rayOrigins.dtype(), "all tensors must have the same type");

    TORCH_CHECK(rayOrigins.rdim() == 2,
                std::string("Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ") +
                    std::to_string(rayOrigins.rdim()) + " dimensions");
    TORCH_CHECK(
        rayDirections.rdim() == 2,
        std::string("Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(rayDirections.rdim()) + " dimensions");
    TORCH_CHECK(tMin.rdim() == 1,
                std::string("Expected tmin to have 1 dimension (shape (n,)) but got ") +
                    std::to_string(tMin.rdim()) + " dimensions");
    TORCH_CHECK(tMax.rdim() == 1,
                std::string("Expected tmin to have 1 dimension (shape (n,)) but got ") +
                    std::to_string(tMax.rdim()) + " dimensions");
    TORCH_CHECK(rayOrigins.rsize(0) == tMin.rsize(0),
                "ray_origins and tmin must have the same size in dimension 0 but got " +
                    std::to_string(rayOrigins.rsize(0)) + " and " + std::to_string(tMin.rsize(0)));
    TORCH_CHECK(rayOrigins.rsize(0) == tMax.rsize(0),
                "ray_origins and tmin must have the same size in dimension 0 but got " +
                    std::to_string(rayOrigins.rsize(0)) + " and " + std::to_string(tMin.rsize(0)));
    TORCH_CHECK(rayOrigins.rsize(0) == rayDirections.rsize(0),
                "ray_origins and ray_directions must have the same size in dimension 0 but got " +
                    std::to_string(rayOrigins.rsize(0)) + " and " +
                    std::to_string(rayDirections.rsize(0)));
    TORCH_CHECK(minStepSize > 0.0, "minStepSize must be positive");
    TORCH_CHECK(coneAngle >= 0.0, "coneAngle must be none negitive");
    TORCH_CHECK(rayOrigins.ldim() == 1, "Invalid list dimension for ray origins.");
    TORCH_CHECK(rayDirections.ldim() == 1, "Invalid list dimension for ray directions.");

    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> JaggedTensor {
        return AT_DISPATCH_V2(
            rayOrigins.scalar_type(), "UniformRaySamples", AT_WRAP([&]() -> JaggedTensor {
                int64_t numThreads = 256 + 128;
                if constexpr (nanovdb::util::is_same<scalar_t, double>::value) {
                    numThreads = 256;
                }
                const auto optsF =
                    torch::TensorOptions().dtype(rayOrigins.dtype()).device(rayOrigins.device());
                const auto optsI32 =
                    torch::TensorOptions().dtype(torch::kInt32).device(rayOrigins.device());
                const auto optsJIdx =
                    torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(rayOrigins.device());
                const auto optsJOffsets = torch::TensorOptions()
                                              .dtype(fvdb::JOffsetsScalarType)
                                              .device(rayOrigins.device());
                const auto optsJLIdx =
                    torch::TensorOptions().dtype(fvdb::JLIdxScalarType).device(rayOrigins.device());

                auto batchAcc         = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
                auto rayOriginsAcc    = jaggedAccessor<DeviceTag, scalar_t, 2>(rayOrigins);
                auto rayDirectionsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);

                auto tMinAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(tMin);
                auto tMaxAcc = jaggedAccessor<DeviceTag, scalar_t, 1>(tMax);

                // Count number of segments along each ray
                torch::Tensor rayCounts =
                    torch::zeros({ rayOrigins.rsize(0) + 1 }, optsI32); // [B*M]
                auto outCountsAcc = tensorAccessor<DeviceTag, int32_t, 1>(rayCounts);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__(int32_t bidx, int32_t eidx, int32_t cidx,
                                             JaggedRAcc32<scalar_t, 2> rayOriginsAcc) {
                        countSamplesPerRayCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                            bidx, eidx, rayOriginsAcc, rayDirectionsAcc, tMinAcc, tMaxAcc,
                            outCountsAcc, batchAcc, minStepSize, coneAngle, includeEndpointSegments,
                            eps);
                    };
                    forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb);
                } else {
                    auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx,
                                  JaggedAcc<scalar_t, 2> rayOriginsAcc) {
                        countSamplesPerRayCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                            bidx, eidx, rayOriginsAcc, rayDirectionsAcc, tMinAcc, tMaxAcc,
                            outCountsAcc, batchAcc, minStepSize, coneAngle, includeEndpointSegments,
                            eps);
                    };
                    forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb);
                }

                // Compute joffsets for the output samples
                torch::Tensor outJOffsets = rayCounts.cumsum(0, fvdb::JOffsetsScalarType); // [B*M]
                const fvdb::JOffsetsType totalSamples =
                    outJOffsets[outJOffsets.size(0) - 1].item<fvdb::JOffsetsType>();

                // Allocate output JaggedTensor indexing data
                torch::Tensor outJLidx =
                    torch::empty({ outJOffsets.size(0) - 1, 2 }, optsJLIdx); // [total_rays, 2]
                torch::Tensor outJIdx =
                    torch::zeros({ totalSamples }, optsJIdx); // [total_intersections]

                // Allocate output tensors
                torch::Tensor outRayTimes =
                    torch::zeros({ totalSamples, returnMidpoint ? 1 : 2 }, optsF); // [B*M*S, 2]

                // Compute output voxels and times
                auto outJOffsetsAcc = tensorAccessor<DeviceTag, fvdb::JOffsetsType, 1>(outJOffsets);
                auto outJIdxAcc     = tensorAccessor<DeviceTag, fvdb::JIdxType, 1>(outJIdx);
                auto outJLIdxAcc    = tensorAccessor<DeviceTag, fvdb::JLIdxType, 2>(outJLidx);

                auto outRayTimesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outRayTimes);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__(int32_t bidx, int32_t eidx, int32_t cidx,
                                             JaggedRAcc32<scalar_t, 2> rayOriginsAcc) {
                        generateRaySamplesCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                            bidx, eidx, rayOriginsAcc, rayDirectionsAcc, tMinAcc, tMaxAcc,
                            outJOffsetsAcc, outJIdxAcc, outJLIdxAcc, outRayTimesAcc, batchAcc,
                            minStepSize, coneAngle, includeEndpointSegments, returnMidpoint, eps);
                    };
                    forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb);
                } else {
                    auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx,
                                  JaggedAcc<scalar_t, 2> rayOriginsAcc) {
                        generateRaySamplesCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                            bidx, eidx, rayOriginsAcc, rayDirectionsAcc, tMinAcc, tMaxAcc,
                            outJOffsetsAcc, outJIdxAcc, outJLIdxAcc, outRayTimesAcc, batchAcc,
                            minStepSize, coneAngle, includeEndpointSegments, returnMidpoint, eps);
                    };
                    forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb);
                }

                if (returnMidpoint) {
                    outRayTimes = outRayTimes.squeeze(-1);
                }
                return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                    outRayTimes, outJOffsets, outJIdx, outJLidx, batchHdl.batchSize());
            }),
            AT_EXPAND(AT_FLOATING_TYPES), c10::kHalf);
    });
}

template <>
JaggedTensor
dispatchUniformRaySamples<torch::kCUDA>(const GridBatchImpl &batchHdl, const JaggedTensor &rayO,
                                        const JaggedTensor &rayD, const JaggedTensor &tMin,
                                        const JaggedTensor &tMax, const double minStepSize,
                                        const double coneAngle, const bool includeEndSegments,
                                        const bool returnMidpoint, const double eps) {
    return UniformRaySamples<torch::kCUDA>(batchHdl, rayO, rayD, tMin, tMax, minStepSize, coneAngle,
                                           includeEndSegments, returnMidpoint, eps);
}
template <>
JaggedTensor
dispatchUniformRaySamples<torch::kCPU>(const GridBatchImpl &batchHdl, const JaggedTensor &rayO,
                                       const JaggedTensor &rayD, const JaggedTensor &tMin,
                                       const JaggedTensor &tMax, const double minStepSize,
                                       const double coneAngle, const bool includeEndSegments,
                                       const bool returnMidpoint, const double eps) {
    return UniformRaySamples<torch::kCPU>(batchHdl, rayO, rayD, tMin, tMax, minStepSize, coneAngle,
                                          includeEndSegments, returnMidpoint, eps);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
