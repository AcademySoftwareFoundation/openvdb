// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @brief This examples demonstrates how values in a NanpVDB grid can be
///        modified on the device. It depends on NanoVDB and CUDA thrust.

#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/cuda/DeviceBuffer.h>

extern "C"  void scaleActiveVoxels(nanovdb::FloatGrid *grid_d, uint64_t leafCount, float scale);

int main()
{
    try {
        // Create an NanoVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto handle = nanovdb::tools::createLevelSetSphere<float, nanovdb::cuda::DeviceBuffer>(100.0f);
        using GridT = nanovdb::FloatGrid;

        handle.deviceUpload(0, false); // Copy the NanoVDB grid to the GPU asynchronously

        const GridT* grid = handle.grid<float>(); // get a (raw) const pointer to a NanoVDB grid of value type float on the CPU
        GridT* deviceGrid = handle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!deviceGrid || !grid) {
            throw std::runtime_error("GridHandle did not contain a grid with value type float");
        }
        if (!grid->isSequential<0>()) {
            throw std::runtime_error("Grid does not support sequential access to leaf nodes!");
        }

        std::cout << "Value before scaling = " << grid->tree().getValue(nanovdb::Coord(101,0,0)) << std::endl;

        scaleActiveVoxels(deviceGrid, grid->tree().nodeCount(0), 2.0f);

        handle.deviceDownload(0, true); // Copy the NanoVDB grid to the CPU synchronously

        std::cout << "Value after scaling  = " << grid->tree().getValue(nanovdb::Coord(101,0,0)) << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}