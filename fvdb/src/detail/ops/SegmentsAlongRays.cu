#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void segmentsAlongRaysCallback(int32_t bidx, int32_t eidx,
                                           const JaggedAccessor<ScalarType, 2> rayOrigins,
                                           const JaggedAccessor<ScalarType, 2> rayDirections,
                                           const TensorAccessor<int32_t, 2> packInfo,
                                           TensorAccessor<ScalarType, 2> outSegments,
                                           TensorAccessor<int16_t, 1> outBIdx,
                                           TensorAccessor<int32_t, 1> outRayIndices,
                                           GridBatchImpl::Accessor<GridType> batchAccessor,
                                           int64_t maxSegments,
                                           ScalarType eps,
                                           bool ignoreMasked) {
    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);

    const VoxelCoordTransform& transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox dualBbox = batchAccessor.dualBbox(bidx);
    auto gridAcc = gpuGrid->getAccessor();

    const bool outputBIdx = outBIdx.size(0) > 0;

    const auto& rayO = rayOrigins.data()[eidx];
    const auto& rayD = rayDirections.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);

    if (!rayVox.clip(dualBbox)) {
        return;
    }

    int32_t numSegments = 0;
    const int32_t startIdx = packInfo[eidx][0];
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, ignoreMasked); it.isValid(); ++it) {
        const ScalarType t0 = it->t0;
        const ScalarType t1 = it->t1;
        const ScalarType deltaT = t1 - t0;

        if (deltaT < eps) {
            continue;
        }

        outSegments[startIdx + numSegments][0] = t0;
        outSegments[startIdx + numSegments][1] = t1;
        outRayIndices[startIdx + numSegments] = eidx;
        if (outputBIdx) {
            outBIdx[startIdx + numSegments] = bidx;
        }
        numSegments += 1;
        if (numSegments == maxSegments) {
            break;
        }
    }
    assert(numSegments == packInfo[eidx][1]);
}


template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void countSegmentsAlongRaysCallback(int32_t bidx, int32_t eidx,
                                                const JaggedAccessor<ScalarType, 2> rayOrigins,     // [B*M, 3]
                                                const JaggedAccessor<ScalarType, 2> rayDirections,  // [B*M, 3]
                                                TensorAccessor<int32_t, 1> outCounts,               // [B*M]
                                                BatchGridAccessor<GridType> batchAccessor,
                                                int64_t maxSegments,
                                                ScalarType eps,
                                                bool ignoreMasked) {
    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);

    const VoxelCoordTransform& transform = batchAccessor.dualTransform(bidx);
    nanovdb::CoordBBox dualBbox = batchAccessor.dualBbox(bidx);
    auto gridAcc = gpuGrid->getAccessor();

    const auto& rayO = rayOrigins.data()[eidx];
    const auto& rayD = rayDirections.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);

    if (!rayVox.clip(dualBbox)) {
        outCounts[eidx] = 0;
        return;
    }

    int64_t numSegments = 0;
    for (auto it = HDDASegmentIterator<decltype(gridAcc), ScalarType>(rayVox, gridAcc, ignoreMasked); it.isValid(); ++it) {
        const ScalarType t0 = it->t0;
        const ScalarType t1 = it->t1;
        const ScalarType deltaT = t1 - t0;

        if (deltaT < eps) {
            continue;
        }

        numSegments += 1;
        if (numSegments == maxSegments) {
            break;
        }
    }
    outCounts[eidx] = numSegments;
}


template <c10::DeviceType DeviceTag>
std::vector<JaggedTensor> SegmentsAlongRays(const GridBatchImpl& batchHdl,
                                                  const JaggedTensor& rayOrigins,
                                                  const JaggedTensor& rayDirections,
                                                  int64_t maxSegments,
                                                  const double eps,
                                                  const bool ignoreMasked) {
    batchHdl.checkDevice(rayOrigins);
    batchHdl.checkDevice(rayDirections);
    TORCH_CHECK_TYPE(rayOrigins.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayDirections.is_floating_point(), "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(rayOrigins.dtype() == rayDirections.dtype(), "all tensors must have the same type");
    TORCH_CHECK(rayOrigins.dim() == 2, std::string("Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ") +
                                       std::to_string(rayOrigins.dim()) + " dimensions");
    TORCH_CHECK(rayDirections.dim() == 2, std::string("Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ") +
                                          std::to_string(rayDirections.dim()) + " dimensions");
    TORCH_CHECK(rayOrigins.size(0) == rayDirections.size(0),
                "ray_origins and ray_directions must have the same size in dimension 0 but got " +
                std::to_string(rayOrigins.size(0)) + " and " + std::to_string(rayDirections.size(0)));
    TORCH_CHECK(eps >= 0.0, "eps must be positive or zero");
    TORCH_CHECK(maxSegments > 0 || maxSegments == -1, "max_segments must be greater than zero or -1");

    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> std::vector<JaggedTensor> {
        return AT_DISPATCH_FLOATING_TYPES_AND_HALF(rayOrigins.scalar_type(), "SegmentsAlongRays", [&]() -> std::vector<JaggedTensor> {
            int64_t numThreads = 384;
            if constexpr (nanovdb::util::is_same<scalar_t, double>::value) {
                numThreads = 256;
            }
            auto optsF = torch::TensorOptions().dtype(rayOrigins.dtype()).device(rayOrigins.device());
            auto optsI = torch::TensorOptions().dtype(torch::kInt32).device(rayOrigins.device());
            auto optsI16 = torch::TensorOptions().dtype(torch::kInt16).device(rayOrigins.device());

            // Count number of segments along each ray
            torch::Tensor rayCounts = torch::zeros({rayOrigins.size(0)}, optsI);  // [B*M]
            auto outCountsAcc = tensorAccessor<DeviceTag, int32_t, 1>(rayCounts);
            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto rayDirectionsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb1 = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> rOA) {
                    countSegmentsAlongRaysCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxSegments, eps, ignoreMasked);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb1);
            } else {
                auto cb1 = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                    countSegmentsAlongRaysCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxSegments, eps, ignoreMasked);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb1);
            }

            // Compute pack offsets
            torch::Tensor cumRayCount = rayCounts.cumsum(0, torch::kInt32);  // [B*M]
            torch::Tensor outPackInfo = torch::stack({cumRayCount - rayCounts, rayCounts}, 1);  // [B*M, 2]
            const int32_t totalSegments = cumRayCount[cumRayCount.size(0) - 1].item<int32_t>();

            // Allocate output tensors
            torch::Tensor outSegments = torch::zeros({totalSegments, 2}, optsF);   // [B*M*S, 2]
            torch::Tensor outBIdx = rayOrigins.jagged_dim() == 0 ? torch::empty({0}, optsI16) : torch::zeros({totalSegments}, optsI16); // [B*M*S]
            torch::Tensor outRIdx = torch::zeros({totalSegments}, optsI); // [B*M*S]

            // Compute output voxels and times
            auto packInfoAcc = tensorAccessor<DeviceTag, int32_t, 2>(outPackInfo);
            auto outSegmentsAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outSegments);
            auto outBIdxAcc = tensorAccessor<DeviceTag, int16_t, 1>(outBIdx);
            auto outRayIndicesAcc = tensorAccessor<DeviceTag, int32_t, 1>(outRIdx);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb2 = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> rOA) {
                    segmentsAlongRaysCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, rOA, rayDirectionsAcc, packInfoAcc, outSegmentsAcc, outBIdxAcc, outRayIndicesAcc, batchAcc, maxSegments, eps, ignoreMasked);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb2);
            } else {
                auto cb2 = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                    segmentsAlongRaysCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, rOA, rayDirectionsAcc, packInfoAcc, outSegmentsAcc, outBIdxAcc, outRayIndicesAcc, batchAcc, maxSegments, eps, ignoreMasked);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb2);
            }

            // FIXME: outBIdx does not seem correct...
            return {
                rayOrigins.jagged_like(outPackInfo),
                JaggedTensor::from_data_and_jidx(outRIdx, outBIdx, batchHdl.batchSize()),
                JaggedTensor::from_data_and_jidx(outSegments, outBIdx, batchHdl.batchSize()),
            };
        });
    });
}



template <>
std::vector<JaggedTensor> dispatchSegmentsAlongRays<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                                      const JaggedTensor& rayOrigins,
                                                                      const JaggedTensor& rayDirections,
                                                                      int64_t maxSegments,
                                                                      const double eps,
                                                                      const bool ignoreMasked) {
    return SegmentsAlongRays<torch::kCUDA>(batchHdl, rayOrigins, rayDirections, maxSegments, eps, ignoreMasked);
}

template <>
std::vector<JaggedTensor> dispatchSegmentsAlongRays<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                                     const JaggedTensor& rayOrigins,
                                                                     const JaggedTensor& rayDirections,
                                                                     int64_t maxSegments,
                                                                     const double eps,
                                                                      const bool ignoreMasked) {
    return SegmentsAlongRays<torch::kCPU>(batchHdl, rayOrigins, rayDirections, maxSegments, eps, ignoreMasked);
}

} // namespace ops
} // namespace detail
} // namespace fvdb

