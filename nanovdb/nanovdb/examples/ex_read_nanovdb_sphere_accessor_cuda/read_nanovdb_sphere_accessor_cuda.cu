// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/cuda/DeviceBuffer.h> // required for CUDA memory management
#include <nanovdb/GridHandle.h>

extern "C" void launch_kernels(const nanovdb::NanoGrid<float>*,
                               const nanovdb::NanoGrid<float>*,
                               cudaStream_t stream);

/// @brief Read a NanoVDB grid from a file and print out multiple values on both the cpu and gpu.
///
/// @note Note This example does NOT depend on OpenVDB, only NanoVDB and CUDA.
int main(int, char**)
{
    try {
        // returns a GridHandle using CUDA for memory management.
        auto handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>("data/sphere.nvdb");

        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);

        handle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously

        auto* cpuGrid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* deviceGrid = handle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!deviceGrid || !cpuGrid)
            throw std::runtime_error("GridHandle did not contain a grid with value type float");

        launch_kernels(deviceGrid, cpuGrid, stream); // Call a host method to print a grid values on both the CPU and GPU

        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}