// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/util/IndexGridBuilder.h>// nanovdb::IndexGridBuilder
#include <nanovdb/util/Primitives.h>      // for nanovdb::createLevelSetSphere
#include <nanovdb/util/CudaDeviceBuffer.h>// for nanovdb::CudaDeviceBuffer

extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueIndex>*,// device grid
                               const nanovdb::NanoGrid<nanovdb::ValueIndex>*,// host grid
                               cudaStream_t stream);

/// @brief This examples depends on  NanoVDB and CUDA.
int main()
{
    try {
        // Create an NanoVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto srcHandle = nanovdb::createLevelSetSphere<float>();
        auto *srcGrid = srcHandle.grid<float>();

        // Converts the FloatGrid to an IndexGrid using CUDA for memory management.
        nanovdb::IndexGridBuilder<float> builder(*srcGrid, /*only active values*/true);
        auto idxHandle = builder.getHandle<nanovdb::CudaDeviceBuffer>("IndexGrid_test", /*number of channels*/1u);

        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);

        idxHandle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously
        auto* cpuGrid = idxHandle.grid<nanovdb::ValueIndex>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* gpuGrid = idxHandle.deviceGrid<nanovdb::ValueIndex>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!gpuGrid || !cpuGrid)
            throw std::runtime_error("GridHandle did not contain a grid with value type float");

        launch_kernels(cpuGrid, cpuGrid, stream); // Call a host method to print a grid value on both the CPU and GPU

        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}