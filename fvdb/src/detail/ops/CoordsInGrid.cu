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

template <typename ScalarType,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
coordsInGridCallback(int32_t bidx,
                     int32_t eidx,
                     JaggedAccessor<ScalarType, 2> ijk,
                     TensorAccessor<bool, 1> outMask,
                     BatchGridAccessor<nanovdb::ValueOnIndex> batchAccessor) {
    const auto *gpuGrid = batchAccessor.grid(bidx);
    auto primalAcc      = gpuGrid->getAccessor();

    const auto &ijkCoord = ijk.data()[eidx];
    const nanovdb::Coord vox(ijkCoord[0], ijkCoord[1], ijkCoord[2]);
    const bool isActive = primalAcc.isActive(vox);
    outMask[eidx]       = isActive;
}

template <c10::DeviceType DeviceTag>
JaggedTensor
CoordsInGrid(const GridBatchImpl &batchHdl, const JaggedTensor &ijk) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(ijk);
    TORCH_CHECK_TYPE(!ijk.is_floating_point(), "ijk must have an integeral type");
    TORCH_CHECK(ijk.rdim() == 2,
                std::string("Expected ijk to have 2 dimensions (shape (n, 3)) but got ") +
                    std::to_string(ijk.rdim()) + " dimensions");
    TORCH_CHECK(ijk.rsize(0) > 0, "Empty tensor (ijk)");
    TORCH_CHECK(ijk.rsize(1) == 3,
                "Expected 3 dimensional ijk but got ijk.shape[1] = " +
                    std::to_string(ijk.rsize(1)));

    auto opts             = torch::TensorOptions().dtype(torch::kBool).device(ijk.device());
    torch::Tensor outMask = torch::empty({ijk.rsize(0)}, opts);

    AT_DISPATCH_V2(
        ijk.scalar_type(),
        "CoordsInGrid",
        AT_WRAP([&]() {
            auto batchAcc        = gridBatchAccessor<DeviceTag, nanovdb::ValueOnIndex>(batchHdl);
            auto outMaskAccessor = tensorAccessor<DeviceTag, bool, 1>(outMask);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__(int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedRAcc32<scalar_t, 2> ijkAcc) {
                    coordsInGridCallback<scalar_t, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, ijkAcc, outMaskAccessor, batchAcc);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(1024, 1, ijk, cb);
            } else {
                auto cb =
                    [=](int32_t bidx, int32_t eidx, int32_t cidx, JaggedAcc<scalar_t, 2> ijkAcc) {
                        coordsInGridCallback<scalar_t, JaggedAcc, TorchAcc>(
                            bidx, eidx, ijkAcc, outMaskAccessor, batchAcc);
                    };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, ijk, cb);
            }
        }),
        AT_EXPAND(AT_INTEGRAL_TYPES));

    return ijk.jagged_like(outMask);
}

template <>
JaggedTensor
dispatchCoordsInGrid<torch::kCUDA>(const GridBatchImpl &batchHdl, const JaggedTensor &coords) {
    return CoordsInGrid<torch::kCUDA>(batchHdl, coords);
}

template <>
JaggedTensor
dispatchCoordsInGrid<torch::kCPU>(const GridBatchImpl &batchHdl, const JaggedTensor &coords) {
    return CoordsInGrid<torch::kCPU>(batchHdl, coords);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
