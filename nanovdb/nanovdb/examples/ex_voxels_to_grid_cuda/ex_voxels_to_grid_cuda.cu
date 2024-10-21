// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

/// @brief Demonstrates how to create a NanoVDB grid from voxel coordinates on the GPU
int main()
{
    try {
        // Define list of voxel coordinates and copy them to the device
        const size_t numVoxels = 3;
        nanovdb::Coord coords[numVoxels] = {nanovdb::Coord(1, 2, 3), nanovdb::Coord(-1,3,6), nanovdb::Coord(-90,100,5678)}, *d_coords = nullptr;
        cudaCheck(cudaMalloc(&d_coords, numVoxels * sizeof(nanovdb::Coord)));
        cudaCheck(cudaMemcpy(d_coords, coords, numVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// coords CPU -> GPU

        // Generate a NanoVDB grid that contains the list of voxels on the device
        auto handle = nanovdb::tools::cuda::voxelsToGrid<float>(d_coords, numVoxels);
        auto *d_grid = handle.deviceGrid<float>();

        // Define a list of values and copy them to the device
        float values[numVoxels] = {1.4f, 6.7f, -5.0f}, *d_values;
        cudaCheck(cudaMalloc(&d_values, numVoxels * sizeof(float)));
        cudaCheck(cudaMemcpy(d_values, values, numVoxels * sizeof(float), cudaMemcpyHostToDevice));// values CPU -> GPU

        // Launch a device kernel that sets the values of voxels define above and prints them
        const unsigned int numThreads = 128, numBlocks = nanovdb::util::cuda::blocksPerGrid(numVoxels, numThreads);
        nanovdb::util::cuda::lambdaKernel<<<numBlocks, numThreads>>>(numVoxels, [=] __device__(size_t tid) {
            using OpT = nanovdb::SetVoxel<float>;// defines type of random-access operation (set value)
            const nanovdb::Coord &ijk = d_coords[tid];
            d_grid->tree().set<OpT>(ijk, d_values[tid]);// normally one should use a ValueAccessor
            printf("GPU: voxel # %lu, grid(%4i,%4i,%4i) = %5.1f\n", tid, ijk[0], ijk[1], ijk[2], d_grid->tree().getValue(ijk));
        }); cudaCheckError();

        // Copy grid from GPU to CPU and print the voxel values for validation
        handle.deviceDownload();// creates a copy on the CPU
        auto *grid = handle.grid<float>();
        for (size_t i=0; i<numVoxels; ++i) {
            const nanovdb::Coord &ijk = coords[i];
            printf("CPU: voxel # %lu, grid(%4i,%4i,%4i) = %5.1f\n", i, ijk[0], ijk[1], ijk[2], grid->tree().getValue(ijk));
        }

        // free arrays allocated on the device
        cudaCheck(cudaFree(d_coords));
        cudaCheck(cudaFree(d_values));
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}