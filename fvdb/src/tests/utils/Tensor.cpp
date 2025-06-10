// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "Tensor.h"

namespace fvdb::test {

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

std::vector<torch::Tensor>
loadTensors(const std::string &filePath, const std::vector<std::string> &names) {
#if FVDB_TEST_PRINT_LOADSTORE_INFO
    std::cout << "Loading tensors from: " << filePath << std::endl;
#endif

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
