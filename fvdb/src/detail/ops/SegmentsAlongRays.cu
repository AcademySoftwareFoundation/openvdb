// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/ForEachCPU.h>
#include <detail/utils/cuda/ForEachCUDA.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
segmentsAlongRaysCallback(int32_t bidx, int32_t rayIdx,
                          const JaggedAccessor<ScalarType, 2>         rayOrigins,
                          const JaggedAccessor<ScalarType, 2>         rayDirections,
                          const TensorAccessor<fvdb::JOffsetsType, 1> outJOffsets,
                          TensorAccessor<fvdb::JIdxType, 1>           outJIdx,
                          TensorAccessor<fvdb::JLIdxType, 2>          outJLIdx,
                          TensorAccessor<ScalarType, 2>               outSegments,
                          GridBatchImpl::Accessor<GridType> batchAccessor, int64_t maxSegments,
                          ScalarType eps, bool ignoreMasked) {
    const nanovdb::NanoGrid<GridType> *gpuGrid = batchAccessor.grid(bidx);

    const VoxelCoordTransform &transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox         dualBbox  = batchAccessor.dualBbox(bidx);
    auto                       gridAcc   = gpuGrid->getAccessor();

    const auto                    &rayO = rayOrigins.data()[rayIdx];
    const auto                    &rayD = rayDirections.data()[rayIdx];
    nanovdb::math::Ray<ScalarType> rayVox =
        transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);

    if (outJLIdx.size(0) > 0) {
        const fvdb::JLIdxType batchStartIdx = rayOrigins.offsetStart(bidx);
        outJLIdx[rayIdx][0]                 = bidx;
        outJLIdx[rayIdx][1]                 = rayIdx - batchStartIdx;
    }

    if (!rayVox.clip(dualBbox)) {
        return;
    }

    fvdb::JOffsetsType       numSegments = 0;
    const fvdb::JOffsetsType startIdx    = outJOffsets[rayIdx];
    for (auto it =
             HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, ignoreMasked);
         it.isValid(); ++it) {
        const ScalarType t0     = it->t0;
        const ScalarType t1     = it->t1;
        const ScalarType deltaT = t1 - t0;

        if (deltaT < eps) {
            continue;
        }

        outSegments[startIdx + numSegments][0] = t0;
        outSegments[startIdx + numSegments][1] = t1;
        outJIdx[startIdx + numSegments]        = rayIdx;

        numSegments += 1;
        if (numSegments == maxSegments) {
            break;
        }
    }
    // assert(numSegments == outJOffsets[eidx][1]);
}

template <typename ScalarType, typename GridType,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
countSegmentsAlongRaysCallback(int32_t bidx, int32_t eidx,
                               const JaggedAccessor<ScalarType, 2> rayOrigins,    // [B*M, 3]
                               const JaggedAccessor<ScalarType, 2> rayDirections, // [B*M, 3]
                               TensorAccessor<int32_t, 1>          outCounts,     // [B*M]
                               BatchGridAccessor<GridType> batchAccessor, int64_t maxSegments,
                               ScalarType eps, bool ignoreMasked) {
    const nanovdb::NanoGrid<GridType> *gpuGrid = batchAccessor.grid(bidx);

    const VoxelCoordTransform &transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox         dualBbox  = batchAccessor.dualBbox(bidx);
    auto                       gridAcc   = gpuGrid->getAccessor();

    const auto                    &rayO = rayOrigins.data()[eidx];
    const auto                    &rayD = rayDirections.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox =
        transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);

    if (!rayVox.clip(dualBbox)) {
        outCounts[eidx + 1] = 0;
        return;
    }

    int32_t numSegments = 0;
    for (auto it =
             HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, ignoreMasked);
         it.isValid(); ++it) {
        const ScalarType t0     = it->t0;
        const ScalarType t1     = it->t1;
        const ScalarType deltaT = t1 - t0;

        if (deltaT < eps) {
            continue;
        }

        numSegments += 1;
        if (numSegments == maxSegments) {
            break;
        }
    }
    outCounts[eidx + 1] = numSegments;
}

template <c10::DeviceType DeviceTag>
JaggedTensor
SegmentsAlongRays(const GridBatchImpl &batchHdl, const JaggedTensor &rayOrigins,
                  const JaggedTensor &rayDirections, int64_t maxSegments, const double eps,
                  const bool ignoreMasked) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(rayOrigins);
    batchHdl.checkDevice(rayDirections);
    TORCH_CHECK_VALUE(rayOrigins.rsize(1) == 3, "ray_origins must have shape (n, 3)");
    TORCH_CHECK_VALUE(rayDirections.rsize(1) == 3, "ray_directions must have shape (n, 3)");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == rayOrigins.num_outer_lists(),
                      "ray_origins must have the same batch size as the grid batch");
    TORCH_CHECK_VALUE(batchHdl.batchSize() == rayDirections.num_outer_lists(),
                      "ray_directions must have the same batch size as the grid batch");
    TORCH_CHECK_TYPE(rayOrigins.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayDirections.is_floating_point(),
                     "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(rayOrigins.dtype() == rayDirections.dtype(),
                     "all tensors must have the same type");
    TORCH_CHECK_VALUE(
        rayOrigins.rdim() == 2,
        std::string("Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(rayOrigins.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(
        rayDirections.rdim() == 2,
        std::string("Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ") +
            std::to_string(rayDirections.rdim()) + " dimensions");
    TORCH_CHECK_VALUE(
        rayOrigins.rsize(0) == rayDirections.rsize(0),
        "ray_origins and ray_directions must have the same size in dimension 0 but got " +
            std::to_string(rayOrigins.rsize(0)) + " and " + std::to_string(rayDirections.rsize(0)));
    TORCH_CHECK_VALUE(eps >= 0.0, "eps must be positive or zero");
    TORCH_CHECK_VALUE(maxSegments > 0 || maxSegments == -1,
                      "max_segments must be greater than zero or -1");
    TORCH_CHECK_VALUE(rayOrigins.ldim() == 1, "Invalid list dimension for ray origins.");
    TORCH_CHECK_VALUE(rayDirections.ldim() == 1, "Invalid list dimension for ray directions.");

    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> JaggedTensor {
        return AT_DISPATCH_V2(
            rayOrigins.scalar_type(), "SegmentsAlongRays", AT_WRAP([&]() -> JaggedTensor {
                int64_t numThreads = 384;
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

                // Count number of segments along each ray
                torch::Tensor rayCounts =
                    torch::zeros({ rayOrigins.rsize(0) + 1 }, optsI32); // [B*M]
                auto outCountsAcc     = tensorAccessor<DeviceTag, int32_t, 1>(rayCounts);
                auto batchAcc         = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
                auto rayDirectionsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb1 = [=] __device__(int32_t bidx, int32_t eidx, int32_t cidx,
                                              JaggedRAcc32<scalar_t, 2> rOA) {
                        countSegmentsAlongRaysCallback<scalar_t, GridType, JaggedRAcc32,
                                                       TorchRAcc32>(
                            bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxSegments,
                            eps, ignoreMasked);
                    };
                    forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb1);
                } else {
                    auto cb1 = [=](int32_t bidx, int32_t eidx, int32_t cidx,
                                   JaggedAcc<scalar_t, 2> rOA) {
                        countSegmentsAlongRaysCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                            bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxSegments,
                            eps, ignoreMasked);
                    };
                    forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb1);
                }

                // Compute joffsets for output
                const torch::Tensor outJOffsets =
                    rayCounts.cumsum(0, fvdb::JOffsetsScalarType); // [B*M]
                const fvdb::JOffsetsType totalSegments =
                    outJOffsets[outJOffsets.size(0) - 1].item<fvdb::JOffsetsType>();

                // Allocate output JaggedTensor indexing data
                torch::Tensor outJLidx =
                    torch::empty({ outJOffsets.size(0) - 1, 2 }, optsJLIdx); // [total_rays, 2]
                torch::Tensor outJIdx =
                    torch::zeros({ totalSegments }, optsJIdx); // [total_intersections]

                // Allocate output jdata tensors
                torch::Tensor outSegments = torch::zeros({ totalSegments, 2 }, optsF); // [B*M*S, 2]

                // Compute output voxels and times
                const auto outJOffsetsAcc =
                    tensorAccessor<DeviceTag, fvdb::JOffsetsType, 1>(outJOffsets);
                const auto outJIdxAcc  = tensorAccessor<DeviceTag, fvdb::JIdxType, 1>(outJIdx);
                const auto outJLIdxAcc = tensorAccessor<DeviceTag, fvdb::JLIdxType, 2>(outJLidx);

                const auto outSegmentsAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outSegments);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb2 = [=] __device__(int32_t bidx, int32_t eidx, int32_t cidx,
                                              JaggedRAcc32<scalar_t, 2> rayOriginsAcc) {
                        segmentsAlongRaysCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                            bidx, eidx, rayOriginsAcc, rayDirectionsAcc, outJOffsetsAcc, outJIdxAcc,
                            outJLIdxAcc, outSegmentsAcc, batchAcc, maxSegments, eps, ignoreMasked);
                    };
                    forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb2);
                } else {
                    auto cb2 = [=](int32_t bidx, int32_t eidx, int32_t cidx,
                                   JaggedAcc<scalar_t, 2> rayOriginsAcc) {
                        segmentsAlongRaysCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                            bidx, eidx, rayOriginsAcc, rayDirectionsAcc, outJOffsetsAcc, outJIdxAcc,
                            outJLIdxAcc, outSegmentsAcc, batchAcc, maxSegments, eps, ignoreMasked);
                    };
                    forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb2);
                }

                return JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                    outSegments, outJOffsets, outJIdx, outJLidx, batchHdl.batchSize());
            }),
            AT_EXPAND(AT_FLOATING_TYPES), c10::kHalf);
    });
}

template <>
JaggedTensor
dispatchSegmentsAlongRays<torch::kCUDA>(const GridBatchImpl &batchHdl,
                                        const JaggedTensor  &rayOrigins,
                                        const JaggedTensor &rayDirections, int64_t maxSegments,
                                        const double eps, const bool ignoreMasked) {
    return SegmentsAlongRays<torch::kCUDA>(batchHdl, rayOrigins, rayDirections, maxSegments, eps,
                                           ignoreMasked);
}

template <>
JaggedTensor
dispatchSegmentsAlongRays<torch::kCPU>(const GridBatchImpl &batchHdl,
                                       const JaggedTensor  &rayOrigins,
                                       const JaggedTensor &rayDirections, int64_t maxSegments,
                                       const double eps, const bool ignoreMasked) {
    return SegmentsAlongRays<torch::kCPU>(batchHdl, rayOrigins, rayDirections, maxSegments, eps,
                                          ignoreMasked);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
