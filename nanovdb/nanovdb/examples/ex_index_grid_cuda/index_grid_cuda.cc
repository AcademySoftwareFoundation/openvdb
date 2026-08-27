// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/CreatePrimitives.h>// for nanovdb::tools::createLevelSetSphere
#include <nanovdb/cuda/HandleStorage.h>// host-includable: cuda::copyTo transfers grids without any kernel

extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>*,// device grid
                               const nanovdb::NanoGrid<nanovdb::ValueOnIndex>*,// host grid
                               cudaStream_t stream);

/// @brief This examples depends on NanoVDB and CUDA.
int main(int, char**)
{
    using SrcGridT  = nanovdb::FloatGrid;
    using DstBuildT = nanovdb::ValueOnIndex;
    try {
        // Create an NanoVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto srcHandle = nanovdb::tools::createLevelSetSphere<float>();
        auto *srcGrid = srcHandle.grid<float>();

        // Converts the FloatGrid to an IndexGrid in host memory.
        auto idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT>(*srcGrid, 1u, false , false);// 1 channel, no tiles or stats

        cudaStream_t stream; // stream that orders the transfer and the kernels below
        cudaStreamCreate(&stream);
        {
            // deep-copy the grid to the GPU (implemented in the CUDA translation unit)
            auto deviceHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(idxHandle, stream);
            auto* cpuGrid = idxHandle.grid<DstBuildT>();
            auto* gpuGrid = deviceHandle.deviceGrid<DstBuildT>();

            if (!gpuGrid) throw std::runtime_error("GridHandle did not contain a device grid with value type float");
            if (!cpuGrid) throw std::runtime_error("GridHandle did not contain a host grid with value type float");

            launch_kernels(gpuGrid, cpuGrid, stream); // print a grid value on both the CPU and GPU
            cudaStreamSynchronize(stream); // the kernels must finish before the device handle (whose buffer frees on this stream) goes away
        }
        cudaStreamDestroy(stream);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}