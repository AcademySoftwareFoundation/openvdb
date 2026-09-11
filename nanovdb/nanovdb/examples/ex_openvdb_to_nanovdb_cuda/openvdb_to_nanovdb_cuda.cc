// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/tools/CreateNanoGrid.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/cuda/HandleStorage.h> // host-includable: cuda::copyTo transfers grids without any kernel

extern "C" void launch_kernels(const nanovdb::NanoGrid<float>*,
                               const nanovdb::NanoGrid<float>*,
                               cudaStream_t stream);

/// @brief This examples depends on OpenVDB, NanoVDB and CUDA.
int main(int, char**)
{
    using SrcGridT = openvdb::FloatGrid;
    try {
        // Create an OpenVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto srcGrid = openvdb::tools::createLevelSetSphere<SrcGridT>(100.0f, openvdb::Vec3f(0.0f), 1.0f);

        // Converts the OpenVDB to NanoVDB and returns a GridHandle that uses CUDA for memory management.
        auto handle = nanovdb::tools::createNanoGrid<SrcGridT, float>(*srcGrid);

        cudaStream_t stream; // stream that orders the transfer and the kernels below
        cudaStreamCreate(&stream);
        {
        // deep-copy the grid to the GPU (implemented in the CUDA translation unit)
        auto deviceHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(handle, stream);

        auto* grid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* deviceGrid = deviceHandle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!deviceGrid || !grid)
            throw std::runtime_error("GridHandle did not contain a grid with value type float");

        launch_kernels(deviceGrid, grid, stream); // Call a host method to print a grid value on both the CPU and GPU
        cudaStreamSynchronize(stream); // the kernels must finish before the device handle (whose buffer frees on this stream) goes away
        }
        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}