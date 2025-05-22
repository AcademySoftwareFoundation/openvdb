// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/ForEachCPU.h>
#include <detail/utils/cuda/ForEachCUDA.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

// Called for each ray
template <bool returnIjk,
          typename ScalarType,
          typename GridType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
voxelsAlongRaysCallback(int32_t bidx,
                        int32_t rayIdx,
                        const JaggedAccessor<ScalarType, 2> rayOrigins,          // [B*M, 3]
                        const JaggedAccessor<ScalarType, 2> rayDirections,       // [B*M, 3]
                        const TensorAccessor<fvdb::JOffsetsType, 1> outJOffsets, // [B*M, 2]
                        TensorAccessor<fvdb::JIdxType, 1> outJIdx,               // [B*M*S]
                        TensorAccessor<fvdb::JLIdxType, 2> outJLIdx,             // [B*M, 2]
                        TensorAccessor<int32_t, 2> outVoxels,                    // [B*M*S, 3]
                        TensorAccessor<ScalarType, 2> outTimes,                  // [B*M*S, 2]
                        GridBatchImpl::Accessor<GridType> batchAccessor,
                        int64_t maxVox,
                        ScalarType eps,
                        bool cumulative) {
    const nanovdb::NanoGrid<GridType> *gpuGrid = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform       = batchAccessor.dualTransform(bidx);
    const nanovdb::CoordBBox dualBbox          = batchAccessor.dualBbox(bidx);
    auto primalAcc                             = gpuGrid->getAccessor();

    const auto &rayO = rayOrigins.data()[rayIdx];
    const auto &rayD = rayDirections.data()[rayIdx];
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

    fvdb::JOffsetsType numVox         = 0;
    const fvdb::JOffsetsType startIdx = outJOffsets[rayIdx];
    for (auto it = HDDAVoxelIterator<decltype(primalAcc), ScalarType>(rayVox, primalAcc);
         it.isValid();
         ++it) {
        const ScalarType t0 = it->second.t0, t1 = it->second.t1;
        const ScalarType deltaT   = t1 - t0;
        const nanovdb::Coord &ijk = it->first;

        int32_t ijkIdx = -1;
        if constexpr (!returnIjk) {
            const int64_t baseOffset = cumulative ? batchAccessor.voxelOffset(bidx) : 0;
            ijkIdx                   = primalAcc.getValue(ijk) - 1 + baseOffset;
        }
        if (deltaT < eps) {
            continue;
        }

        // This check handles numerical errors where we can accidentally add the same voxel twice
        if (numVox > 0) {
            bool lastMatch = false;
            if constexpr (returnIjk) {
                lastMatch = (ijk[0] == outVoxels[startIdx + numVox - 1][0] &&
                             ijk[1] == outVoxels[startIdx + numVox - 1][1] &&
                             ijk[2] == outVoxels[startIdx + numVox - 1][2]);
            } else {
                lastMatch = (ijkIdx == outVoxels[startIdx + numVox - 1][0]);
            }
            if (lastMatch) {
                outTimes[startIdx + numVox - 1][0] =
                    c10::cuda::compat::min(t0, outTimes[startIdx + numVox - 1][0]);
                outTimes[startIdx + numVox - 1][1] =
                    c10::cuda::compat::max(t1, outTimes[startIdx + numVox - 1][1]);
                outJIdx[startIdx + numVox - 1] = rayIdx;
                continue;
            }
        }

        if constexpr (returnIjk) {
#pragma unroll
            for (int i = 0; i < 3; i += 1) {
                outVoxels[startIdx + numVox][i] = ijk[i];
            }
        } else {
            outVoxels[startIdx + numVox][0] = ijkIdx;
        }
        outTimes[startIdx + numVox][0] = t0;
        outTimes[startIdx + numVox][1] = t1;
        outJIdx[startIdx + numVox]     = rayIdx;

        numVox += 1;
        if (numVox == maxVox) {
            break;
        }
    }
    // assert(numVox == outJOffsets[rayIdx][1] - outJOffsets[rayIdx][0]);
}

template <typename ScalarType,
          typename GridType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ void
countVoxelsAlongRaysCallback(int32_t bidx,
                             int32_t eidx,
                             const JaggedAccessor<ScalarType, 2> rayOrigins,
                             const JaggedAccessor<ScalarType, 2> rayDirections,
                             TensorAccessor<int32_t, 1> outCounts,
                             BatchGridAccessor<GridType> batchAccessor,
                             int64_t maxVox,
                             ScalarType eps) {
    const nanovdb::NanoGrid<GridType> *gpuGrid = batchAccessor.grid(bidx);
    const VoxelCoordTransform &transform       = batchAccessor.dualTransform(bidx);
    const nanovdb::CoordBBox dualBbox          = batchAccessor.dualBbox(bidx);
    auto primalAcc                             = gpuGrid->getAccessor();

    const auto &rayO = rayOrigins.data()[eidx];
    const auto &rayD = rayDirections.data()[eidx];
    nanovdb::math::Ray<ScalarType> rayVox =
        transform.applyToRay(rayO[0], rayO[1], rayO[2], rayD[0], rayD[1], rayD[2]);

    if (!rayVox.clip(dualBbox)) {
        outCounts[eidx + 1] = 0;
        return;
    }
    int32_t numVox = 0;
    nanovdb::Coord lastIjk;
    for (auto it = HDDAVoxelIterator<decltype(primalAcc), ScalarType>(rayVox, primalAcc);
         it.isValid();
         ++it) {
        const ScalarType t0 = it->second.t0, t1 = it->second.t1;
        const ScalarType deltaT   = t1 - t0;
        const nanovdb::Coord &ijk = it->first;
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
    outCounts[eidx + 1] = numVox;
}

template <c10::DeviceType DeviceTag>
std::vector<JaggedTensor>
VoxelsAlongRays(const GridBatchImpl &batchHdl,
                const JaggedTensor &rayOrigins,
                const JaggedTensor &rayDirections,
                int64_t maxVox,
                float eps,
                bool returnIjk,
                bool cumulative) {
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
    TORCH_CHECK_VALUE(maxVox > 0 || maxVox == -1, "max_vox must be greater than zero or -1");
    TORCH_CHECK_VALUE(rayOrigins.ldim() == 1, "Invalid list dimension for ray origins.");
    TORCH_CHECK_VALUE(rayDirections.ldim() == 1, "Invalid list dimension for ray directions.");

    return FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() -> std::vector<JaggedTensor> {
        return AT_DISPATCH_V2(
            rayOrigins.scalar_type(),
            "VoxelsAlongRays",
            AT_WRAP([&]() -> std::vector<JaggedTensor> {
                int64_t numThreads = 384;
                if constexpr (nanovdb::util::is_same<scalar_t, double>::value ||
                              nanovdb::util::is_same<GridType, nanovdb::ValueOnIndexMask>::value) {
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

                // Count number of voxels along each ray
                torch::Tensor rayCounts = torch::zeros({rayOrigins.rsize(0) + 1}, optsI32); // [B*M]
                auto outCountsAcc       = tensorAccessor<DeviceTag, int32_t, 1>(rayCounts);
                auto batchAcc           = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
                auto rayDirectionsAcc   = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb1 = [=] __device__(int32_t bidx,
                                              int32_t eidx,
                                              int32_t cidx,
                                              JaggedRAcc32<scalar_t, 2> rOA) {
                        countVoxelsAlongRaysCallback<scalar_t, GridType, JaggedRAcc32, TorchRAcc32>(
                            bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxVox, eps);
                    };
                    forEachJaggedElementChannelCUDA<scalar_t, 2>(numThreads, 1, rayOrigins, cb1);
                } else {
                    auto cb1 = [=](int32_t bidx,
                                   int32_t eidx,
                                   int32_t cidx,
                                   JaggedAcc<scalar_t, 2> rOA) {
                        countVoxelsAlongRaysCallback<scalar_t, GridType, JaggedAcc, TorchAcc>(
                            bidx, eidx, rOA, rayDirectionsAcc, outCountsAcc, batchAcc, maxVox, eps);
                    };
                    forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb1);
                }

                // Compute joffsets for the output ray intersections
                const torch::Tensor outJOffsets =
                    rayCounts.cumsum(0, fvdb::JOffsetsScalarType); // [B*M]
                const fvdb::JOffsetsType totalIsects =
                    outJOffsets[outJOffsets.size(0) - 1].item<fvdb::JOffsetsType>();

                // Allocate output JaggedTensor indexing data
                torch::Tensor outJLidx =
                    torch::empty({outJOffsets.size(0) - 1, 2}, optsJLIdx); // [total_rays, 2]
                torch::Tensor outJIdx =
                    torch::zeros({totalIsects}, optsJIdx);                 // [total_intersections]

                // Allocate output jdata tensors
                torch::Tensor outVoxels =
                    torch::zeros({totalIsects, returnIjk ? 3 : 1}, optsI32);    // [B*M*S, 3]
                torch::Tensor outTimes = torch::zeros({totalIsects, 2}, optsF); // [B*M*S, 2]

                // Compute output voxels and times
                auto outJOffsetsAcc = tensorAccessor<DeviceTag, fvdb::JOffsetsType, 1>(outJOffsets);
                auto outJIdxAcc     = tensorAccessor<DeviceTag, fvdb::JIdxType, 1>(outJIdx);
                auto outJLIdxAcc    = tensorAccessor<DeviceTag, fvdb::JLIdxType, 2>(outJLidx);

                auto outVoxelsAcc = tensorAccessor<DeviceTag, int32_t, 2>(outVoxels);
                auto outTimesAcc  = tensorAccessor<DeviceTag, scalar_t, 2>(outTimes);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cbIjk = [=] __device__(int32_t bidx,
                                                int32_t eidx,
                                                int32_t cidx,
                                                JaggedRAcc32<scalar_t, 2> rayOriginsAcc) {
                        voxelsAlongRaysCallback<true,
                                                scalar_t,
                                                GridType,
                                                JaggedRAcc32,
                                                TorchRAcc32>(bidx,
                                                             eidx,
                                                             rayOriginsAcc,
                                                             rayDirectionsAcc,
                                                             outJOffsetsAcc,
                                                             outJIdxAcc,
                                                             outJLIdxAcc,
                                                             outVoxelsAcc,
                                                             outTimesAcc,
                                                             batchAcc,
                                                             maxVox,
                                                             eps,
                                                             cumulative);
                    };
                    auto cbIdx = [=] __device__(int32_t bidx,
                                                int32_t eidx,
                                                int32_t cidx,
                                                JaggedRAcc32<scalar_t, 2> rayOriginsAcc) {
                        voxelsAlongRaysCallback<false,
                                                scalar_t,
                                                GridType,
                                                JaggedRAcc32,
                                                TorchRAcc32>(bidx,
                                                             eidx,
                                                             rayOriginsAcc,
                                                             rayDirectionsAcc,
                                                             outJOffsetsAcc,
                                                             outJIdxAcc,
                                                             outJLIdxAcc,
                                                             outVoxelsAcc,
                                                             outTimesAcc,
                                                             batchAcc,
                                                             maxVox,
                                                             eps,
                                                             cumulative);
                    };

                    if (returnIjk) {
                        forEachJaggedElementChannelCUDA<scalar_t, 2>(
                            numThreads, 1, rayOrigins, cbIjk);
                    } else {
                        forEachJaggedElementChannelCUDA<scalar_t, 2>(
                            numThreads, 1, rayOrigins, cbIdx);
                    }
                } else {
                    auto cbIjk = [=](int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedAcc<scalar_t, 2> rayOriginsAcc) {
                        voxelsAlongRaysCallback<true, scalar_t, GridType, JaggedAcc, TorchAcc>(
                            bidx,
                            eidx,
                            rayOriginsAcc,
                            rayDirectionsAcc,
                            outJOffsetsAcc,
                            outJIdxAcc,
                            outJLIdxAcc,
                            outVoxelsAcc,
                            outTimesAcc,
                            batchAcc,
                            maxVox,
                            eps,
                            cumulative);
                    };
                    auto cbIdx = [=](int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedAcc<scalar_t, 2> rayOriginsAcc) {
                        voxelsAlongRaysCallback<false, scalar_t, GridType, JaggedAcc, TorchAcc>(
                            bidx,
                            eidx,
                            rayOriginsAcc,
                            rayDirectionsAcc,
                            outJOffsetsAcc,
                            outJIdxAcc,
                            outJLIdxAcc,
                            outVoxelsAcc,
                            outTimesAcc,
                            batchAcc,
                            maxVox,
                            eps,
                            cumulative);
                    };
                    if (returnIjk) {
                        forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cbIjk);
                    } else {
                        forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cbIdx);
                    }
                }

                if (!returnIjk) {
                    outVoxels = outVoxels.squeeze(-1);
                }

                const JaggedTensor retVox = JaggedTensor::from_jdata_joffsets_jidx_and_lidx_unsafe(
                    outVoxels, outJOffsets, outJIdx, outJLidx, batchHdl.batchSize());
                const JaggedTensor retTimes = retVox.jagged_like(outTimes);

                return {retVox, retTimes};
            }),
            AT_EXPAND(AT_FLOATING_TYPES),
            c10::kHalf);
    });
}

template <>
std::vector<JaggedTensor>
dispatchVoxelsAlongRays<torch::kCUDA>(const GridBatchImpl &batchHdl,
                                      const JaggedTensor &rayOrigins,
                                      const JaggedTensor &rayDirections,
                                      int64_t maxVox,
                                      float eps,
                                      bool returnIjk,
                                      bool cumulative) {
    return VoxelsAlongRays<torch::kCUDA>(
        batchHdl, rayOrigins, rayDirections, maxVox, eps, returnIjk, cumulative);
}

template <>
std::vector<JaggedTensor>
dispatchVoxelsAlongRays<torch::kCPU>(const GridBatchImpl &batchHdl,
                                     const JaggedTensor &rayOrigins,
                                     const JaggedTensor &rayDirections,
                                     int64_t maxVox,
                                     float eps,
                                     bool returnIjk,
                                     bool cumulative) {
    return VoxelsAlongRays<torch::kCPU>(
        batchHdl, rayOrigins, rayDirections, maxVox, eps, returnIjk, cumulative);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
