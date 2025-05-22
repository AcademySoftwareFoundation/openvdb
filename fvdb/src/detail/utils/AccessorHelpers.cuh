// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_ACCESSORHELPERS_CUH
#define FVDB_DETAIL_UTILS_ACCESSORHELPERS_CUH

#include <JaggedTensor.h>
#include <detail/GridBatchImpl.h>
#include <detail/ops/Ops.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {

/// @brief Shorthand for torch::PackedTensorAccessor32 with RestrictPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using TorchRAcc32 = torch::PackedTensorAccessor32<ScalarType, DIMS, torch::RestrictPtrTraits>;

/// @brief Shorthand for torch::PackedTensorAccessor64 with RestrictPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using TorchRAcc64 = torch::PackedTensorAccessor64<ScalarType, DIMS, torch::RestrictPtrTraits>;

/// @brief Shorthand for a torch::TensorAccessor with DefaultPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using TorchAcc = torch::TensorAccessor<ScalarType, DIMS>;

/// @brief Shorthand for fvdb::PackedJaggedAccessor32 with RestrictPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using JaggedRAcc32 = fvdb::PackedJaggedAccessor32<ScalarType, DIMS, torch::RestrictPtrTraits>;

/// @brief Shorthand for fvdb::JaggedAccessor
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using JaggedAcc = fvdb::JaggedAccessor<ScalarType, DIMS>;

/// @brief Get an accessor for the given tensor with scalar type T and N dimensions
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the tensor
/// @tparam N The number of dimensions of the tensor
/// @tparam IndexT The type of index to use for packed tensors on the GPU (default is int32_t)
/// @param tensor The tensor to get an accessor for
/// @return A tensor accessor (either torch::TensorAccessor or torch::PackedTensorAccessor32)
template <c10::DeviceType DeviceTag, typename T, size_t N, typename IndexT = int32_t>
typename std::conditional<
    DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1,
    torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, IndexT>,
    torch::TensorAccessor<T, N>>::type
tensorAccessor(const torch::Tensor &tensor) {
    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        return tensor.generic_packed_accessor<T, N, torch::RestrictPtrTraits, IndexT>();
    } else {
        return tensor.accessor<T, N>();
    }
}

/// @brief Get an accessor for the given batched grid handle with scalar type T
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam GridT The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @param batchHdl The batched grid handle to get an accessor for
/// @return A fvdb::detail::GridBatchImpl::Accessor of the given type on the appropriate device
template <c10::DeviceType DeviceTag, typename GridT>
typename fvdb::detail::GridBatchImpl::Accessor<GridT>
gridBatchAccessor(const fvdb::detail::GridBatchImpl &batchHdl) {
    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        return batchHdl.deviceAccessor<GridT>();
    } else {
        return batchHdl.hostAccessor<GridT>();
    }
}

/// @brief Get an accessor for the given jagged tensor with scalar type T and N dimensions
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the JaggedTensor
/// @tparam N The number of dimensions of the JaggedTensor data tensor
/// @param jaggedTensor The JaggedTensor to get an accessor for
/// @return A JaggedTensor accessor (either JaggedAccessor or PackedJaggedAccessor32)
template <c10::DeviceType DeviceTag, typename T, size_t N>
typename std::conditional<DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1,
                          fvdb::JaggedRAcc32<T, N>,
                          fvdb::JaggedAcc<T, N>>::type
jaggedAccessor(const fvdb::JaggedTensor &jaggedTensor) {
    if constexpr (DeviceTag == torch::kCUDA || DeviceTag == torch::kPrivateUse1) {
        return jaggedTensor.packed_accessor32<T, N, torch::RestrictPtrTraits>();
    } else {
        return jaggedTensor.accessor<T, N>();
    }
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_ACCESSORHELPERS_CUH
