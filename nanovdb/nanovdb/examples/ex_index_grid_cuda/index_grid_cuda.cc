// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/CreatePrimitives.h>      // for nanovdb::tools::createLevelSetSphere
#include <nanovdb/cuda/DeviceBuffer.h>// for nanovdb::cuda::DeviceBuffer

extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>*,// device grid
                               const nanovdb::NanoGrid<nanovdb::ValueOnIndex>*,// host grid
                               cudaStream_t stream);

/// @brief This examples depends on  NanoVDB and CUDA.
int main(int, char**)
{
    using SrcGridT  = nanovdb::FloatGrid;
    using DstBuildT = nanovdb::ValueOnIndex;
    using BufferT   = nanovdb::cuda::DeviceBuffer;
    try {
        // Create an NanoVDB grid of a sphere at the origin with radius 100 and voxel size 1.
        auto srcHandle = nanovdb::tools::createLevelSetSphere<float>();
        auto *srcGrid = srcHandle.grid<float>();

        // Converts the FloatGrid to an IndexGrid using CUDA for memory management.
        auto idxHandle = nanovdb::tools::createNanoGrid<SrcGridT, DstBuildT, BufferT>(*srcGrid, 1u, false , false);// 1 channel, no tiles or stats

        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);

        idxHandle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously
        auto* cpuGrid = idxHandle.grid<DstBuildT>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto* gpuGrid = idxHandle.deviceGrid<DstBuildT>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

        if (!gpuGrid) throw std::runtime_error("GridHandle did not contain a device grid with value type float");
        if (!cpuGrid) throw std::runtime_error("GridHandle did not contain a host grid with value type float");

        launch_kernels(cpuGrid, cpuGrid, stream); // Call a host method to print a grid value on both the CPU and GPU

        cudaStreamDestroy(stream); // Destroy the CUDA stream
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}