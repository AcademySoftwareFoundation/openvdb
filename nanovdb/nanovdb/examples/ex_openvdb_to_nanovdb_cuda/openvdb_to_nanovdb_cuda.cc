// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/tools/CreateNanoGrid.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/cuda/DeviceBuffer.h>

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
        auto handle = nanovdb::tools::createNanoGrid<SrcGridT, float, nanovdb::cuda::DeviceBuffer>(*srcGrid);

        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);

        handle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously

        auto* grid = handle.grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* deviceGrid = handle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!deviceGrid || !grid)
            throw std::runtime_error("GridHandle did not contain a grid with value type float");

        launch_kernels(deviceGrid, grid, stream); // Call a host method to print a grid value on both the CPU and GPU

        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}