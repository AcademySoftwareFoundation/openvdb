// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef FVDB_TESTS_UTILS_TENSOR_H
#define FVDB_TESTS_UTILS_TENSOR_H

#include <torch/torch.h>

#include <vector>

namespace fvdb::test {

/// @brief Utility function to check if a tensor has the expected shape, type and device
/// @param tensor The tensor to check
/// @param shape The expected shape of the tensor, as a vector of integers
/// @param type The expected scalar (torch) type of the tensor
/// @param device The expected (torch) device of the tensor
/// @return true if the tensor has the expected shape, type and device, false otherwise
inline bool
checkTensor(const torch::Tensor &tensor, const std::vector<int64_t> &shape,
            const torch::ScalarType &type, const torch::Device &device = torch::kCPU) {
#if 0
    std::cout << "Tensor dimensions: " << tensor.dim() << std::endl;
    std::cout << "Expected shape: ";
    for (const auto &s: shape) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    std::cout << "Actual shape: ";
    for (const auto &s: tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    std::cout << "Expected type: " << type << std::endl;
    std::cout << "Actual type: " << tensor.scalar_type() << std::endl;
    std::cout << "Expected device: " << device << std::endl;
    std::cout << "Actual device: " << tensor.device() << std::endl;
#endif
    return tensor.dim() == shape.size() && tensor.sizes().vec() == shape &&
           tensor.scalar_type() == type && tensor.device() == device;
}

/// @brief Utility function to create a torch::TensorOptions object with the given scalar type and
///        device
/// @tparam T The scalar (C++) type of the tensor
/// @param device The device to create the tensor on (default: torch::kCUDA)
/// @return A torch::TensorOptions object with the given scalar type and device
template <typename T>
inline constexpr torch::TensorOptions
tensorOpts(torch::Device device = torch::kCUDA) {
    return torch::TensorOptions().device(device).dtype(torch::CppTypeToScalarType<T>::value);
}

} // namespace fvdb::test

#endif // FVDB_TESTS_UTILS_TENSOR_H
