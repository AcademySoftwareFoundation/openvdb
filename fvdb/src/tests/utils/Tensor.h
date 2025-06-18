// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_UTILS_TENSOR_H
#define TESTS_UTILS_TENSOR_H

#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#define FVDB_TEST_PRINT_LOADSTORE_INFO 0

namespace fvdb::test {

/// @brief Utility function to check if a tensor has the expected shape, type and device
/// @param tensor The tensor to check
/// @param shape The expected shape of the tensor, as a vector of integers
/// @param type The expected scalar (torch) type of the tensor
/// @param device The expected (torch) device of the tensor
/// @return true if the tensor has the expected shape, type and device, false otherwise
inline bool
checkTensor(const torch::Tensor &tensor,
            const std::vector<int64_t> &shape,
            const torch::ScalarType &type,
            const torch::Device &device = torch::kCPU) {
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
    return tensor.dim() == static_cast<int64_t>(shape.size()) && tensor.sizes().vec() == shape &&
           tensor.scalar_type() == type && tensor.device() == device;
}

/// @brief Utility function to create a torch::TensorOptions object with the given scalar type and
///        device
/// @tparam T The scalar (C++) type of the tensor
/// @param device The device to create the tensor on (default: torch::kCUDA)
/// @return A torch::TensorOptions object with the given scalar type and device
template <typename T>
inline torch::TensorOptions
tensorOpts(torch::Device device = torch::kCUDA) {
    return torch::TensorOptions().device(device).dtype(torch::CppTypeToScalarType<T>::value);
}

/// @brief Load tensors from a TorchScript module
/// @param filePath The path to the TorchScript module
/// @param names The names of the tensors to load
/// @return A vector of tensors loaded from the module
/// @throw c10::Error if the tensors are not loaded successfully
std::vector<torch::Tensor> loadTensorsFromTorchScript(const std::string &filePath,
                                                      const std::vector<std::string> &names);

/// @brief Load tensors from a file
/// @param filePath The path to the file
/// @return A vector of tensors loaded from the file
/// @throw c10::Error if the tensors are not loaded successfully
std::vector<torch::Tensor> loadTensorsDirect(const std::string &filePath);

/// @brief Load tensors from a file or TorchScript module
/// @param filePath The path to the file or TorchScript module
/// @param names The names of the tensors to load
/// @return A vector of tensors loaded from the file or TorchScript module
/// @throw c10::Error if the tensors are not loaded successfully
std::vector<torch::Tensor> loadTensors(const std::string &filePath,
                                       const std::vector<std::string> &names);

/// @brief Store tensors directly to a file
/// @param filePath The path to the file
/// @param tensors The tensors to store
/// @throw c10::Error if the tensors are not stored successfully
void storeTensorsDirect(const std::string &filePath, const std::vector<torch::Tensor> &tensors);

/// @brief Store tensors as named buffers in a module
/// @param filePath The path to the file
/// @param tensors The tensors to store
/// @param names The names of the tensors to store
/// @throw c10::Error if the tensors are not stored successfully
void storeTensorsModule(const std::string &filePath,
                        const std::vector<torch::Tensor> &tensors,
                        const std::vector<std::string> &names);

/// @brief Store tensors to a file or TorchScript module
/// @param filePath The path to the file or TorchScript module
/// @param tensors The tensors to store
/// @param names The names of the tensors to store
/// @throw c10::Error if the tensors are not stored successfully
void storeTensors(const std::string filePath,
                  const std::vector<torch::Tensor> &tensors,
                  const std::vector<std::string> &names);

} // namespace fvdb::test

#endif // TESTS_UTILS_TENSOR_H
