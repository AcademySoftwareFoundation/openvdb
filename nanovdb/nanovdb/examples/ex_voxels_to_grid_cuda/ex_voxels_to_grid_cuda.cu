// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/cuda/GridHandle.cuh> // for cuda::copyTo, the explicit device->host grid transfer

/// @brief Demonstrates how to create a NanoVDB grid from voxel coordinates on the GPU
int main()
{
    try {
        // Define list of voxel coordinates and copy them to the device
        const size_t numVoxels = 3;
        nanovdb::Coord coords[numVoxels] = {nanovdb::Coord(1, 2, 3), nanovdb::Coord(-1,3,6), nanovdb::Coord(-90,100,5678)};
        nanovdb::cuda::Buffer<nanovdb::Coord> coordBuffer(cudaStream_t(0), numVoxels, nanovdb::cuda::noInit);
        nanovdb::Coord *d_coords = coordBuffer.data();
        cudaCheck(cudaMemcpy(d_coords, coords, numVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// coords CPU -> GPU

        // Generate a NanoVDB grid from the voxels, stored in a single-space device buffer
        auto handle = nanovdb::tools::cuda::voxelsToGrid<float, nanovdb::Coord*, nanovdb::cuda::Buffer<std::byte>>(d_coords, numVoxels);
        auto *d_grid = handle.deviceGrid<float>();

        // Define a list of values and copy them to the device
        float values[numVoxels] = {1.4f, 6.7f, -5.0f};
        nanovdb::cuda::Buffer<float> valueBuffer(cudaStream_t(0), numVoxels, nanovdb::cuda::noInit);
        float *d_values = valueBuffer.data();
        cudaCheck(cudaMemcpy(d_values, values, numVoxels * sizeof(float), cudaMemcpyHostToDevice));// values CPU -> GPU

        // Launch a device kernel that sets the values of the voxels defined above and prints them
        const unsigned int numThreads = 128, numBlocks = nanovdb::util::cuda::blocksPerGrid(numVoxels, numThreads);
        nanovdb::util::cuda::lambdaKernel<<<numBlocks, numThreads>>>(numVoxels, [=] __device__(size_t tid) {
            using OpT = nanovdb::SetVoxel<float>;// defines type of random-access operation (set value)
            const nanovdb::Coord &ijk = d_coords[tid];
            d_grid->tree().set<OpT>(ijk, d_values[tid]);// normally one should use a ValueAccessor
            printf("GPU: voxel # %zu, grid(%4i,%4i,%4i) = %5.1f\n", tid, ijk[0], ijk[1], ijk[2], d_grid->tree().getValue(ijk));
        }); cudaCheckError();

        // Deep-copy the grid to a host handle and print the voxel values for validation
        auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle);
        auto *grid = hostHandle.grid<float>();
        for (size_t i=0; i<numVoxels; ++i) {
            const nanovdb::Coord &ijk = coords[i];
            printf("CPU: voxel # %zu, grid(%4i,%4i,%4i) = %5.1f\n", i, ijk[0], ijk[1], ijk[2], grid->tree().getValue(ijk));
        }

    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}
