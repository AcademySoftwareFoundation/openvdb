// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/cuda/Utils.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename GridType, typename ScalarType,
          template <typename T, int32_t D> typename JaggedAccessor,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ inline void
ijkToInvIndexCallback(fvdb::JIdxType bidx, int64_t eidx, BatchGridAccessor<GridType> batchAccessor,
                      const JaggedAccessor<ScalarType, 2> ijk,
                      TensorAccessor<int64_t, 1> outInvIndex, bool cumulative) {
    const nanovdb::NanoGrid<GridType> *grid     = batchAccessor.grid(bidx);
    const auto                         acc      = grid->getAccessor();
    const auto                        &ijkCoord = ijk.data()[eidx];
    const nanovdb::Coord               vox(ijkCoord[0], ijkCoord[1], ijkCoord[2]);
    if (acc.isActive(vox)) {
        const fvdb::JOffsetsType baseOffset = cumulative ? 0 : ijk.offsetStart(bidx);
        const int64_t index = int64_t(acc.getValue(vox)) - 1 + batchAccessor.voxelOffset(bidx);
        outInvIndex[index]  = eidx - baseOffset;
    }
}

template <c10::DeviceType DeviceTag>
JaggedTensor
IjkToInvIndex(const GridBatchImpl &batchHdl, const JaggedTensor &ijk, bool cumulative) {
    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(ijk);
    TORCH_CHECK_TYPE(at::isIntegralType(ijk.scalar_type(), false), "ijk must have an integer type");
    TORCH_CHECK(ijk.rdim() == 2,
                std::string("Expected points to have 2 dimensions (shape (n, 3)) but got ") +
                    std::to_string(ijk.rdim()) + " dimensions");
    TORCH_CHECK(ijk.rsize(0) > 0, "Empty tensor (ijk)");
    TORCH_CHECK(ijk.rsize(1) == 3, "Expected 3 dimensional ijk but got points.shape[1] = " +
                                       std::to_string(ijk.rsize(1)));

    auto          opts = torch::TensorOptions().dtype(torch::kLong).device(ijk.device());
    torch::Tensor outInvIndex =
        torch::full({ batchHdl.totalVoxels() }, at::Scalar(int64_t(-1)), opts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_INTEGRAL_TYPES(ijk.scalar_type(), "IjkToInvIndex", [&]() {
            auto batchAcc       = gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto outInvIndexAcc = tensorAccessor<DeviceTag, int64_t, 1>(outInvIndex);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__(fvdb::JIdxType bidx, int64_t eidx, int64_t cidx,
                                         JaggedRAcc32<scalar_t, 2> ijkAcc) {
                    ijkToInvIndexCallback<GridType, scalar_t, JaggedRAcc32, TorchRAcc32>(
                        bidx, eidx, batchAcc, ijkAcc, outInvIndexAcc, cumulative);
                };
                forEachJaggedElementChannelCUDA<scalar_t, 2>(512, 1, ijk, cb);
            } else {
                auto cb = [=](fvdb::JIdxType bidx, int64_t eidx, int64_t cidx,
                              JaggedAcc<scalar_t, 2> ijkAcc) {
                    ijkToInvIndexCallback<GridType, scalar_t, JaggedAcc, TorchAcc>(
                        bidx, eidx, batchAcc, ijkAcc, outInvIndexAcc, cumulative);
                };
                forEachJaggedElementChannelCPU<scalar_t, 2>(1, ijk, cb);
            }
        });
    });

    return batchHdl.jaggedTensor(outInvIndex, false);
}

template <>
JaggedTensor
dispatchIjkToInvIndex<torch::kCUDA>(const GridBatchImpl &batchHdl, const JaggedTensor &ijk,
                                    bool cumulative) {
    return IjkToInvIndex<torch::kCUDA>(batchHdl, ijk, cumulative);
}

template <>
JaggedTensor
dispatchIjkToInvIndex<torch::kCPU>(const GridBatchImpl &batchHdl, const JaggedTensor &ijk,
                                   bool cumulative) {
    return IjkToInvIndex<torch::kCPU>(batchHdl, ijk, cumulative);
}

} // namespace ops
} // namespace detail
} // namespace fvdb