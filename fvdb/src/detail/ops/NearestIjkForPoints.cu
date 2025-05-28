// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/ForEachCPU.h>
#include <detail/utils/cuda/ForEachCUDA.cuh>
#include <detail/utils/cuda/RAIIRawDeviceBuffer.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarT>
__device__ void
nearestNeighborIJKForPointCallback(fvdb::JIdxType bidx,
                                   int32_t eidx,
                                   const JaggedRAcc32<ScalarT, 2> points,
                                   const VoxelCoordTransform *transforms,
                                   TorchRAcc64<int32_t, 2> outIJKData,
                                   TorchRAcc64<fvdb::JIdxType, 1> outIJKBIdx) {
    static_assert(sizeof(nanovdb::Coord) == 3 * sizeof(int32_t));

    using MathT = typename at::opmath_type<ScalarT>;

    const auto pt                        = points.data()[eidx];
    const VoxelCoordTransform &transform = transforms[bidx];
    const nanovdb::Coord ijk0 =
        transform
            .apply(static_cast<MathT>(pt[0]), static_cast<MathT>(pt[1]), static_cast<MathT>(pt[2]))
            .floor();
    const int32_t base = eidx * 8;
#pragma unroll
    for (int di = 0; di <= 1; di += 1) {
#pragma unroll
        for (int dj = 0; dj <= 1; dj += 1) {
#pragma unroll
            for (int dk = 0; dk <= 1; dk += 1) {
                const nanovdb::Coord ijk    = ijk0 + nanovdb::Coord(di, dj, dk);
                const int32_t count         = di * 4 + dj * 2 + dk;
                outIJKData[base + count][0] = ijk[0];
                outIJKData[base + count][1] = ijk[1];
                outIJKData[base + count][2] = ijk[2];
                outIJKBIdx[base + count]    = bidx;
            }
        }
    }
}

template <>
JaggedTensor
dispatchNearestNeighborIJKForPoints<torch::kCUDA>(
    const JaggedTensor &jaggedPoints, const std::vector<VoxelCoordTransform> &transforms) {
    TORCH_CHECK(jaggedPoints.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(jaggedPoints.device().has_index(), "GridBatchImpl must have a valid index");

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(jaggedPoints.device());
    const torch::TensorOptions optsBIdx =
        torch::TensorOptions().dtype(fvdb::JIdxScalarType).device(jaggedPoints.device());
    torch::Tensor outIJK     = torch::empty({jaggedPoints.jdata().size(0) * 8, 3}, optsData);
    torch::Tensor outIJKBIdx = torch::empty({jaggedPoints.jdata().size(0) * 8},
                                            optsBIdx); // TODO: Don't populate for single batch

    AT_DISPATCH_V2(
        jaggedPoints.scalar_type(),
        "nearestNeighborIJKForPoints",
        AT_WRAP([&] {
            RAIIRawDeviceBuffer<VoxelCoordTransform> transformsDVec(transforms.size(),
                                                                    jaggedPoints.device());
            transformsDVec.setData((VoxelCoordTransform *)transforms.data(), true /* blocking */);
            const VoxelCoordTransform *transformDevPtr = transformsDVec.devicePtr;

            auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
            auto outIJKBIdxAcc =
                outIJKBIdx.packed_accessor64<fvdb::JIdxType, 1, torch::RestrictPtrTraits>();

            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pacc) {
                nearestNeighborIJKForPointCallback(
                    bidx, eidx, pacc, transformDevPtr, outIJKAcc, outIJKBIdxAcc);
            };

            forEachJaggedElementChannelCUDA<scalar_t, 2>(256, 1, jaggedPoints, cb);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);

    return JaggedTensor::from_data_offsets_and_list_ids(
        outIJK, jaggedPoints.joffsets() * 8, jaggedPoints.jlidx());
}

} // namespace ops
} // namespace detail
} // namespace fvdb
