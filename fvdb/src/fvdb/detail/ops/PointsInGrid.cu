// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/cuda/ForEachPrivateUse1.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TorchAccessor>
__hostdev__ inline void
pointsInGridCallback(int32_t bidx,
                     int32_t eidx,
                     JaggedAccessor<ScalarType, 2> points,
                     TorchAccessor<bool, 1> outMask,
                     BatchGridAccessor<nanovdb::ValueOnIndex> batchAccessor) {
    const auto *gpuGrid                  = batchAccessor.grid(bidx);
    auto primalAcc                       = gpuGrid->getAccessor();
    const VoxelCoordTransform &transform = batchAccessor.primalTransform(bidx);

    const auto pointPos = points.data()[eidx];
    const nanovdb::Coord vox =
        transform.apply((ScalarType)pointPos[0], (ScalarType)pointPos[1], (ScalarType)pointPos[2])
            .round();

    const bool isActive = primalAcc.isActive(vox);
    outMask[eidx]       = isActive;
}

template <c10::DeviceType DeviceTag, typename scalar_t>
JaggedTensor
PointsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &points) {
    auto opts             = torch::TensorOptions().dtype(torch::kBool).device(points.device());
    torch::Tensor outMask = torch::empty({points.rsize(0)}, opts);

    auto batchAcc        = gridBatchAccessor<DeviceTag, nanovdb::ValueOnIndex>(batchHdl);
    auto outMaskAccessor = tensorAccessor<DeviceTag, bool, 1>(outMask);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                pointsInGridCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, ptsA, outMaskAccessor, batchAcc);
            };
        forEachJaggedElementChannelCUDA<scalar_t, 2>(1024, 1, points, cb);
    } else if constexpr (DeviceTag == torch::kPrivateUse1) {
        auto cb = [=]
            __device__(int32_t bidx, int32_t eidx, int32_t cidx, JaggedRAcc32<scalar_t, 2> ptsA) {
                pointsInGridCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                    bidx, eidx, ptsA, outMaskAccessor, batchAcc);
            };
        forEachJaggedElementChannelPrivateUse1<scalar_t, 2>(1, points, cb);
    } else {
        auto cb = [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ptsA) {
            pointsInGridCallback<scalar_t, JaggedAcc, TorchAcc>(
                bidx, eidx, ptsA, outMaskAccessor, batchAcc);
        };
        forEachJaggedElementChannelCPU<scalar_t, 2>(1, points, cb);
    }

    return points.jagged_like(outMask);
}

template <c10::DeviceType DeviceTag>
JaggedTensor
dispatchPointsInGrid<DeviceTag>(const GridBatchImpl &batchHdl, const JaggedTensor &points) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(points);
    TORCH_CHECK_TYPE(points.is_floating_point(), "points must have a floating point type");
    TORCH_CHECK(points.rdim() == 2,
                std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                    std::to_string(points.rdim()) + " dimensions");
    TORCH_CHECK(points.rsize(0) > 0, "Empty tensor (points)");
    TORCH_CHECK(points.rsize(1) == 3,
                "Expected 3 dimensional points but got points.shape[1] = " +
                    std::to_string(points.rsize(1)));

    return AT_DISPATCH_V2(
        points.scalar_type(),
        "PointsInGrid",
        AT_WRAP([&]() { return PointsInGrid<DeviceTag, scalar_t>(batchHdl, points); }),
        AT_EXPAND(AT_FLOATING_TYPES),
        c10::kHalf);
}

template JaggedTensor dispatchPointsInGrid<torch::kCPU>(const GridBatchImpl &,
                                                        const JaggedTensor &);
template JaggedTensor dispatchPointsInGrid<torch::kCUDA>(const GridBatchImpl &,
                                                         const JaggedTensor &);
template JaggedTensor dispatchPointsInGrid<torch::kPrivateUse1>(const GridBatchImpl &,
                                                                const JaggedTensor &);

} // namespace ops
} // namespace detail
} // namespace fvdb
