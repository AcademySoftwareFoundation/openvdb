// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/RAIIRawDeviceBuffer.h>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <thrust/universal_vector.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarT>
__device__ void
ijkForPointsCallback(int32_t bidx,
                     int32_t eidx,
                     const JaggedRAcc32<ScalarT, 2> points,
                     const VoxelCoordTransform *transforms,
                     TorchRAcc64<int32_t, 2> outIJKData) {
    using MathT                          = typename at::opmath_type<ScalarT>;
    const auto &point                    = points.data()[eidx];
    const VoxelCoordTransform &transform = transforms[bidx];
    const nanovdb::Coord ijk0            = transform
                                    .apply(static_cast<MathT>(point[0]),
                                           static_cast<MathT>(point[1]),
                                           static_cast<MathT>(point[2]))
                                    .round();
    outIJKData[eidx][0] = ijk0[0];
    outIJKData[eidx][1] = ijk0[1];
    outIJKData[eidx][2] = ijk0[2];
}

JaggedTensor
ijkForPoints(const JaggedTensor &jaggedPoints, const std::vector<VoxelCoordTransform> &transforms) {
    TORCH_CHECK(jaggedPoints.device().is_cuda(), "GridBatchImpl must be on CUDA device");
    TORCH_CHECK(jaggedPoints.device().has_index(), "GridBatchImpl must have a valid index");

    const torch::TensorOptions optsData =
        torch::TensorOptions().dtype(torch::kInt32).device(jaggedPoints.device());
    torch::Tensor outIJK = torch::empty({jaggedPoints.jdata().size(0), 3}, optsData);

    AT_DISPATCH_V2(
        jaggedPoints.scalar_type(),
        "ijkForPoints",
        AT_WRAP([&] {
            RAIIRawDeviceBuffer<VoxelCoordTransform> transformsDVec(transforms.size(),
                                                                    jaggedPoints.device());
            transformsDVec.setData((VoxelCoordTransform *)transforms.data(), true /* blocking */);
            const VoxelCoordTransform *transformDevPtr = transformsDVec.devicePtr;

            auto outIJKAcc = outIJK.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();

            auto cb = [=] __device__(int32_t bidx,
                                     int32_t eidx,
                                     int32_t cidx,
                                     JaggedRAcc32<scalar_t, 2> pacc) {
                ijkForPointsCallback(bidx, eidx, pacc, transformDevPtr, outIJKAcc);
            };
            forEachJaggedElementChannelCUDA<scalar_t, 2>(1024, 1, jaggedPoints, cb);
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
    return jaggedPoints.jagged_like(outIJK);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromPoints<torch::kCUDA>(const JaggedTensor &points,
                                          const std::vector<VoxelCoordTransform> &txs) {
    JaggedTensor coords = ijkForPoints(points, txs);
    return ops::dispatchCreateNanoGridFromIJK<torch::kCUDA>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromPoints<torch::kPrivateUse1>(const JaggedTensor &points,
                                                 const std::vector<VoxelCoordTransform> &txs) {
    TORCH_CHECK(points.device().is_privateuseone(), "GridBatchImpl must be on PrivateUse1 device");

    const torch::TensorOptions deviceOptions = torch::TensorOptions().device(points.device());
    const torch::TensorOptions ijkOptions    = deviceOptions.dtype(torch::kInt32);

    torch::Tensor ijk = torch::empty({points.jdata().size(0), 3}, ijkOptions);
    auto ijkAcc       = ijk.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();

    thrust::universal_vector<VoxelCoordTransform> transforms(txs.size());
    auto transformsPtr = transforms.data().get();
    cudaMemcpy(
        transformsPtr, txs.data(), sizeof(VoxelCoordTransform) * txs.size(), cudaMemcpyDefault);

    AT_DISPATCH_V2(points.scalar_type(),
                   "ijkForPoints",
                   AT_WRAP([&] {
                       auto cb = [=] __device__(int32_t bidx,
                                                int32_t eidx,
                                                int32_t cidx,
                                                JaggedRAcc32<scalar_t, 2> pacc) {
                           ijkForPointsCallback(bidx, eidx, pacc, transformsPtr, ijkAcc);
                       };
                       forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(1, points, cb);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES),
                   c10::kHalf);

    JaggedTensor coords = points.jagged_like(ijk);
    return ops::dispatchCreateNanoGridFromIJK<torch::kPrivateUse1>(coords);
}

template <>
nanovdb::GridHandle<TorchDeviceBuffer>
dispatchBuildGridFromPoints<torch::kCPU>(const JaggedTensor &pointsJagged,
                                         const std::vector<VoxelCoordTransform> &txs) {
    using GridType = nanovdb::ValueOnIndex;
    return AT_DISPATCH_V2(
        pointsJagged.scalar_type(),
        "buildPaddedGridFromPoints",
        AT_WRAP([&]() {
            using ScalarT = scalar_t;
            static_assert(is_floating_point_or_half<ScalarT>::value,
                          "Invalid type for points, must be floating point");
            using MathT      = typename at::opmath_type<ScalarT>;
            using ProxyGridT = nanovdb::tools::build::Grid<float>;

            pointsJagged.check_valid();

            const torch::TensorAccessor<ScalarT, 2> &pointsAcc =
                pointsJagged.jdata().accessor<ScalarT, 2>();
            const torch::TensorAccessor<fvdb::JOffsetsType, 1> &pointsBOffsetsAcc =
                pointsJagged.joffsets().accessor<fvdb::JOffsetsType, 1>();

            std::vector<nanovdb::GridHandle<TorchDeviceBuffer>> batchHandles;
            batchHandles.reserve(pointsBOffsetsAcc.size(0) - 1);
            for (int bi = 0; bi < (pointsBOffsetsAcc.size(0) - 1); bi += 1) {
                VoxelCoordTransform tx = txs[bi];

                auto proxyGrid         = std::make_shared<ProxyGridT>(-1.0f);
                auto proxyGridAccessor = proxyGrid->getWriteAccessor();

                const int64_t start = pointsBOffsetsAcc[bi];
                const int64_t end   = pointsBOffsetsAcc[bi + 1];

                for (int64_t pi = start; pi < end; pi += 1) {
                    nanovdb::Coord ijk = tx.apply(static_cast<MathT>(pointsAcc[pi][0]),
                                                  static_cast<MathT>(pointsAcc[pi][1]),
                                                  static_cast<MathT>(pointsAcc[pi][2]))
                                             .round();
                    proxyGridAccessor.setValue(ijk, 1.0f);
                }

                proxyGridAccessor.merge();
                auto ret = nanovdb::tools::createNanoGrid<ProxyGridT, GridType, TorchDeviceBuffer>(
                    *proxyGrid, 0u, false, false);
                ret.buffer().to(torch::kCPU);
                batchHandles.push_back(std::move(ret));
            }

            if (batchHandles.size() == 1) {
                return std::move(batchHandles[0]);
            } else {
                return nanovdb::mergeGrids(batchHandles);
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
