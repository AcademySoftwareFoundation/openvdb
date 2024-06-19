// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#undef NANOVDB_USE_OPENVDB // Prevents include/openvdb/points/AttributeArray.h:1841:25: error: ‘stride’ cannot be used as a function

#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include <iostream>

extern "C" void launch_kernels(const nanovdb::NanoGrid<float>*,// GPU grid
                               const nanovdb::NanoGrid<float>*,// CPU grid
                               cudaStream_t stream);

/// @brief Creates a NanoVDB grids with custom values and access them.
///
/// @note This example only depends on NanoVDB.
int main()
{
    try {
        using GridT = nanovdb::tools::build::Grid<float>;
        GridT grid(0.0f);// empty grid with a background value of zero
        auto acc = grid.getAccessor();
        acc.setValue(nanovdb::Coord(1, 2, 3), 1.0f);
        printf("build::Grid: (%i,%i,%i)=%4.2f\n", 1, 2,-3, acc.getValue(nanovdb::Coord(1, 2,-3)));
        printf("build::Grid: (%i,%i,%i)=%4.2f\n", 1, 2, 3, acc.getValue(nanovdb::Coord(1, 2, 3)));

        // convert build::grid to a nanovdb::GridHandle using a Cuda buffer
        auto handle = nanovdb::tools::createNanoGrid<GridT, float, nanovdb::cuda::DeviceBuffer>(grid);

        auto* cpuGrid = handle.grid<float>(); //get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        if (!cpuGrid) throw std::runtime_error("GridHandle does not contain a grid with value type float");

        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);
        handle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously
        auto* gpuGrid = handle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        launch_kernels(gpuGrid, cpuGrid, stream); // Call a host method to print a grid values on both the CPU and GPU
        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}