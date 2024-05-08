#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include "detail/utils/cuda/Utils.cuh"


namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void voxelsAlongRaysCallback(int32_t bidx, int32_t eidx,
                                         const JaggedAccessor<ScalarType, 2> rayOrigins,     // [B*M, 3]
                                         const JaggedAccessor<ScalarType, 2> rayDirections,  // [B*M, 3]
                                         const TensorAccessor<int32_t, 2> packInfo,           // [B*M, 2]
                                         TensorAccessor<int32_t, 2> outVoxels,                // [B*M*S, 3]
                                         TensorAccessor<ScalarType, 2> outTimes,              // [B*M*S, 2]
                                         TensorAccessor<int16_t, 1> outBIdx,                  // [B*M*S]
                                         GridBatchImpl::Accessor<GridType> batchAccessor,
                                         int64_t maxVox,
                                         ScalarType eps) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const VoxelCoordTransform& transform = batchAccessor.dualTransform(bidx);
    const nanovdb::CoordBBox dualBbox = batchAccessor.dualBbox(bidx);
    auto primalAcc = gpuGrid->getAccessor();

    const bool outputBIdx = outBIdx.size(0) > 0;

    const auto& rayO = rayOrigins.data()[eidx];
    const auto& rayD = rayDirections.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);

    if (!rayVox.clip(dualBbox)) {
        return;
    }
    int32_t numVox = 0;
    const int32_t startIdx = packInfo[eidx][0];
    for (auto it = HDDAVoxelIterator<decltype(primalAcc), ScalarType>(rayVox, primalAcc); it.isValid(); ++it) {
        const ScalarType t0 = it->second.t0, t1 = it->second.t1;
        const ScalarType deltaT = t1 - t0;
        const nanovdb::Coord& ijk = it->first;
        if (deltaT < eps) {
            continue;
        }
        if (numVox > 0 &&
            ijk[0] == outVoxels[startIdx + numVox - 1][0] &&
            ijk[1] == outVoxels[startIdx + numVox - 1][1] &&
            ijk[2] == outVoxels[startIdx + numVox - 1][2]) {
            outTimes[startIdx + numVox - 1][0] = c10::cuda::compat::min(t0, outTimes[startIdx + numVox - 1][0]);
            outTimes[startIdx + numVox - 1][1] = c10::cuda::compat::max(t1, outTimes[startIdx + numVox - 1][1]);
            if (outputBIdx) {
                outBIdx[startIdx + numVox - 1] = bidx;
            }
            continue;
        }

        #pragma unroll
        for (int i = 0; i < 3; i += 1) {
            outVoxels[startIdx + numVox][i] = ijk[i];
        }
        outTimes[startIdx + numVox][0] = t0;
        outTimes[startIdx + numVox][1] = t1;
        if (outputBIdx) {
            outBIdx[startIdx + numVox - 1] = bidx;
        }
        numVox += 1;
        if (numVox == maxVox) {
            break;
        }
    }
    assert(numVox == packInfo[eidx][1]);
}


template <typename ScalarType, typename GridType, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void countVoxelsAlongRaysCallback(int32_t bidx, int32_t eidx,
                                              const JaggedAccessor<ScalarType, 2> rayOrigins,
                                              const JaggedAccessor<ScalarType, 2> rayDirections,
                                              TensorAccessor<int32_t, 1> outCounts,
                                              BatchGridAccessor<GridType> batchAccessor,
                                              int64_t maxVox,
                                              ScalarType eps) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const VoxelCoordTransform& transform = batchAccessor.dualTransform(bidx);
    const nanovdb::CoordBBox dualBbox = batchAccessor.dualBbox(bidx);
    auto primalAcc = gpuGrid->getAccessor();

    const auto& rayO = rayOrigins.data()[eidx];
    const auto& rayD = rayDirections.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox = transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);

    if (!rayVox.clip(dualBbox)) {
        outCounts[eidx] = 0;
        return;
    }
    int32_t numVox = 0;
    nanovdb::Coord lastIjk;
    for (auto it = HDDAVoxelIterator<decltype(primalAcc), ScalarType>(rayVox, primalAcc); it.isValid(); ++it) {
        const ScalarType t0 = it->second.t0, t1 = it->second.t1;
        const ScalarType deltaT = t1 - t0;
        const nanovdb::Coord& ijk = it->first;
        if (deltaT < eps) {
            continue;
        }
        if (numVox > 0 && ijk == lastIjk) {
            continue;
        }

        lastIjk = ijk;
        numVox += 1;
        if (numVox == maxVox) {
            break;
        }
    }
    outCounts[eidx] = numVox;
}

template <c10::DeviceType DeviceTag>
std::vector<JaggedTensor> VoxelsAlongRays(const GridBatchImpl& batchHdl,
                                          const JaggedTensor& rayOrigins,
                                          const JaggedTensor& rayDirections,
                                          int64_t maxVox, float eps) {
    batchHdl.checkNonEmptyGrid();
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
    TORCH_CHECK(maxVox > 0 || maxVox == -1, "max_vox must be greater than zero or -1");

    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> std::vector<JaggedTensor> {
        return AT_DISPATCH_FLOATING_TYPES_AND_HALF(rayOrigins.scalar_type(), "VoxelsAlongRays", [&]() -> std::vector<JaggedTensor> {
            int64_t numThreads = 384;
            if constexpr (nanovdb::util::is_same<scalar_t, double>::value || nanovdb::util::is_same<GridType, nanovdb::ValueOnIndexMask>::value) {
                numThreads = 256;
            }
            auto optsF = torch::TensorOptions().dtype(rayOrigins.dtype()).device(rayOrigins.device());
            auto optsI = torch::TensorOptions().dtype(torch::kInt32).device(rayOrigins.device());
            auto optsI16 = torch::TensorOptions().dtype(torch::kInt16).device(rayOrigins.device());

            // Count number of voxels along each ray
            torch::Tensor rayCounts = torch::zeros({rayOrigins.size(0)}, optsI);  // [B*M]
            auto outCountsAcc = tensorAccessor<DeviceTag, int32_t, 1>(rayCounts);
            auto batchAcc = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto rayDirectionsAcc = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb1 = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> rOA) {
                    countVoxelsAlongRaysCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxVox, eps);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb1);
            } else {
                auto cb1 = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                    countVoxelsAlongRaysCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxVox, eps);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb1);
            }

            // Compute pack offsets
            torch::Tensor cumRayCount = rayCounts.cumsum(0, torch::kInt32);  // [B*M]
            torch::Tensor outPackInfo = torch::stack({cumRayCount - rayCounts, rayCounts}, 1);  // [B*M, 2]
            const int32_t totalSteps = cumRayCount[cumRayCount.size(0) - 1].item<int32_t>();

            // Allocate output tensors
            torch::Tensor outVoxels = torch::zeros({totalSteps, 3}, optsI);  // [B*M*S, 3]
            torch::Tensor outTimes = torch::zeros({totalSteps, 2}, optsF);   // [B*M*S, 2]
            torch::Tensor outBIdx = rayOrigins.jagged_dim() == 0 ? torch::empty({0}, optsI16) : torch::zeros({totalSteps}, optsI16); // [B*M*S]

            // Compute output voxels and times
            auto packInfoAcc = tensorAccessor<DeviceTag, int32_t, 2>(outPackInfo);
            auto outVoxelsAcc = tensorAccessor<DeviceTag, int32_t, 2>(outVoxels);
            auto outTimesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outTimes);
            auto outBIdxAcc = tensorAccessor<DeviceTag, int16_t, 1>(outBIdx);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb2 = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> rOA) {
                    voxelsAlongRaysCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, rOA, rayDirectionsAcc, packInfoAcc, outVoxelsAcc, outTimesAcc, outBIdxAcc, batchAcc, maxVox, eps);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb2);
            } else {
                auto cb2 = [=] (int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> rOA) {
                    voxelsAlongRaysCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                        bidx, eidx, rOA, rayDirectionsAcc, packInfoAcc, outVoxelsAcc, outTimesAcc, outBIdxAcc, batchAcc, maxVox, eps);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb2);
            }

            return {
                rayOrigins.jagged_like(outPackInfo),
                JaggedTensor::from_data_and_jidx(outVoxels, outBIdx, batchHdl.batchSize()),
                JaggedTensor::from_data_and_jidx(outTimes, outBIdx, batchHdl.batchSize()),
            };
        });
    });
}



template <>
std::vector<JaggedTensor> dispatchVoxelsAlongRays<torch::kCUDA>(const GridBatchImpl& batchHdl,
                                                              const JaggedTensor& rayOrigins,
                                                              const JaggedTensor& rayDirections,
                                                              int64_t maxVox,
                                                              float eps) {
    return VoxelsAlongRays<torch::kCUDA>(batchHdl, rayOrigins, rayDirections, maxVox, eps);
}

template <>
std::vector<JaggedTensor> dispatchVoxelsAlongRays<torch::kCPU>(const GridBatchImpl& batchHdl,
                                                             const JaggedTensor& rayOrigins,
                                                             const JaggedTensor& rayDirections,
                                                             int64_t maxVox,
                                                             float eps) {
    return VoxelsAlongRays<torch::kCPU>(batchHdl, rayOrigins, rayDirections, maxVox, eps);
}


} // namespace ops
} // namespace detail
} // namespace fvdb

