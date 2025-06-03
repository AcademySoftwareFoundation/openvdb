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
std::vector<torch::Tensor>
loadTensorsFromTorchScript(const std::string &filePath, const std::vector<std::string> &names) {
    std::vector<torch::Tensor> result;

    // load the module
    try {
        torch::jit::script::Module module = torch::jit::load(filePath);

        for (const auto &name: names) {
            try {
                auto tensor = module.attr(name).toTensor();
#if FVDB_TEST_PRINT_LOADSTORE_INFO
                std::cout << "Loaded tensor: " << name << " with shape: " << tensor.sizes()
                          << std::endl;
#endif
                result.push_back(tensor);
            } catch (const c10::Error &e) {
                std::cout << "Failed to load tensor: " << name << std::endl;
                throw e;
            }
        }
    } catch (const c10::Error &e) {
        std::cout << "Failed to load module: " << filePath << std::endl;
        throw e;
    }
    return result;
}

/// @brief Load tensors from a file
/// @param filePath The path to the file
/// @return A vector of tensors loaded from the file
/// @throw c10::Error if the tensors are not loaded successfully
std::vector<torch::Tensor>
loadTensorsDirect(const std::string &filePath) {
    std::vector<torch::Tensor> result;

    torch::load(result, filePath);

#if FVDB_TEST_PRINT_LOADSTORE_INFO
    if (!result.empty()) {
        std::cout << "Successfully loaded " << result.size() << " tensors directly" << std::endl;

        // Print shapes of loaded tensors for debugging
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << "  Tensor " << i << " shape: " << result[i].sizes() << std::endl;
        }
    }
#endif

    return result;
}

/// @brief Load tensors from a file or TorchScript module
/// @param filePath The path to the file or TorchScript module
/// @param names The names of the tensors to load
/// @return A vector of tensors loaded from the file or TorchScript module
/// @throw c10::Error if the tensors are not loaded successfully
std::vector<torch::Tensor>
loadTensors(const std::string &filePath, const std::vector<std::string> &names) {
    // First try direct tensor loading
    std::vector<torch::Tensor> result = loadTensorsDirect(filePath);

    // If direct loading failed, try module approach
    if (result.empty()) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
        std::cout << "Direct loading failed, trying module approach..." << std::endl;
#endif
        // Try original path
        result = loadTensorsFromTorchScript(filePath, names);

        // If that fails too, try with .module extension
        if (result.empty()) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
            std::cout << "Module loading failed, trying with .module extension..." << std::endl;
#endif
            result = loadTensorsFromTorchScript(filePath + ".module", names);
        }
    }

    return result;
}

/// @brief Store tensors directly to a file
/// @param filePath The path to the file
/// @param tensors The tensors to store
/// @throw c10::Error if the tensors are not stored successfully
void
storeTensorsDirect(const std::string &filePath, const std::vector<torch::Tensor> &tensors) {
    // Prepare tensors for direct serialization
    std::vector<torch::Tensor> processedTensors;
    for (const auto &tensor: tensors) {
        // Ensure tensors are on CPU
        processedTensors.push_back(tensor.detach().cpu());
    }

    try {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
        std::cout << "Directly saving tensors to: " << filePath << std::endl;
#endif
        torch::save(processedTensors, filePath);
    } catch (const c10::Error &e) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
        std::cerr << "Error saving tensors directly: " << e.what() << std::endl;
#endif
        throw e;
    }
}

/// @brief Store tensors as named buffers in a module
/// @param filePath The path to the file
/// @param tensors The tensors to store
/// @param names The names of the tensors to store
/// @throw c10::Error if the tensors are not stored successfully
void
storeTensorsModule(const std::string &filePath,
                   const std::vector<torch::Tensor> &tensors,
                   const std::vector<std::string> &names) {
    try {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
        std::cout << "Creating module for saving to: " << filePath << std::endl;
#endif

        torch::jit::Module module;

        for (size_t i = 0; i < tensors.size() && i < names.size(); ++i) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
            std::cout << "  Registering buffer: " << names[i] << std::endl;
#endif
            // Detach tensor and move to CPU before registering
            auto processedTensor = tensors[i].detach().cpu();
            module.register_buffer(names[i], processedTensor);
        }

#if FVDB_TEST_PRINT_LOADSTORE_INFO
        std::cout << "Saving module to: " << filePath << std::endl;
#endif
        module.save(filePath);
    } catch (const c10::Error &e) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
        std::cerr << "Error saving module: " << e.what() << std::endl;
#endif
        throw e;
    }
}

void
storeTensors(const std::string filePath,
             const std::vector<torch::Tensor> &tensors,
             const std::vector<std::string> &names) {
    try {
        storeTensorsModule(filePath, tensors, names);
    } catch (const c10::Error &e) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
        std::cerr << "Error storing tensors as module: " << e.what() << std::endl;
#endif

        try {
            storeTensorsDirect(filePath, tensors);
        } catch (const c10::Error &e) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
            std::cerr << "Error storing tensors directly: " << e.what() << std::endl;
#endif
            throw e;
        }
    }
}

} // namespace fvdb::test

#endif // TESTS_UTILS_TENSOR_H
