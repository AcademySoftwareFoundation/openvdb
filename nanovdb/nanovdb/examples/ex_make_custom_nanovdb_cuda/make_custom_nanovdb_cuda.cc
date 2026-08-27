// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#undef NANOVDB_USE_OPENVDB // Prevents include/openvdb/points/AttributeArray.h:1841:25: error: ‘stride’ cannot be used as a function

#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/Buffer.h>// host-safe: declares the device buffer type the .cu side returns

extern nanovdb::GridHandle<nanovdb::cuda::Buffer<std::byte>> uploadGrid(const nanovdb::GridHandle<nanovdb::HostBuffer>& handle, cudaStream_t stream);

#include <iostream>

extern "C" void launch_kernels(const nanovdb::NanoGrid<float>*,// GPU grid
                               const nanovdb::NanoGrid<float>*,// CPU grid
                               cudaStream_t stream);

/// @brief Creates a NanoVDB grid with custom values and access them.
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

        // convert build::grid to a nanovdb::GridHandle in host memory
        auto handle = nanovdb::tools::createNanoGrid<GridT, float>(grid);

        auto* cpuGrid = handle.grid<float>(); //get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        if (!cpuGrid) throw std::runtime_error("GridHandle does not contain a grid with value type float");

        cudaStream_t stream; // stream that orders the transfer and the kernels below
        cudaStreamCreate(&stream);
        {
            // deep-copy the grid to the GPU (implemented in the CUDA translation unit)
            auto deviceHandle = uploadGrid(handle, stream);
            auto* gpuGrid = deviceHandle.deviceGrid<float>();

            launch_kernels(gpuGrid, cpuGrid, stream); // print grid values on both the CPU and GPU
            cudaStreamSynchronize(stream); // the kernels must finish before the device handle (whose buffer frees on this stream) goes away
        }
        cudaStreamDestroy(stream);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}