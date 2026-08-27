// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @brief This examples demonstrates how values in a NanpVDB grid can be
///        modified on the device. It depends on NanoVDB and CUDA thrust.

#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/cuda/HandleStorage.h>// host-includable: cuda::copyTo transfers grids without any kernel

extern "C"  void scaleActiveVoxels(nanovdb::FloatGrid *grid_d, uint64_t leafCount, float scale);

int main()
{
    try {
        // Create an NanoVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto handle = nanovdb::tools::createLevelSetSphere<float>(100.0f);
        using GridT = nanovdb::FloatGrid;

        // deep-copy the grid to the device -- callable right here, in a host-only file
        auto deviceHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(handle);

        const GridT* grid = handle.grid<float>(); // a (raw) const pointer to the grid on the CPU
        GridT* deviceGrid = deviceHandle.deviceGrid<float>(); // and its deep copy on the GPU

        if (!deviceGrid || !grid) {
            throw std::runtime_error("GridHandle did not contain a grid with value type float");
        }
        if (!grid->isSequential<0>()) {
            throw std::runtime_error("Grid does not support sequential access to leaf nodes!");
        }

        std::cout << "Value before scaling = " << grid->tree().getValue(nanovdb::Coord(101,0,0)) << std::endl;

        scaleActiveVoxels(deviceGrid, grid->tree().nodeCount(0), 2.0f);

        // copy the modified grid back to the host and read the result from the returned handle
        auto result = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(deviceHandle);

        std::cout << "Value after scaling  = " << result.grid<float>()->tree().getValue(nanovdb::Coord(101,0,0)) << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
