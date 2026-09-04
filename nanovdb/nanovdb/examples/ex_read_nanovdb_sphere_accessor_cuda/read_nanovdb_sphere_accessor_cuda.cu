// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

//! [read_nanovdb_sphere_accessor_cuda]
#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/cuda/GridHandle.cuh> // for cuda::copyTo, the explicit host<->device grid transfer

extern "C" void launch_kernels(const nanovdb::NanoGrid<float>*,
                               const nanovdb::NanoGrid<float>*,
                               cudaStream_t stream);

/// @brief Read a NanoVDB grid from a file and print out multiple values on both the cpu and gpu.
///
/// @note Note This example does NOT depend on OpenVDB, only NanoVDB and CUDA.
int main(int, char**)
{
    try {
        // read the grid into host memory (HostBuffer is the default buffer type)
        auto handle = nanovdb::io::readGrid("data/sphere.nvdb");

        cudaStream_t stream; // stream that orders the transfer and the kernels below
        cudaStreamCreate(&stream);
        {
            // Deep-copy the grid to the GPU: the copy is ordered on the stream, and the
            // returned handle validates the transferred grid on the device.
            auto deviceHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(handle, stream);

            auto* cpuGrid = handle.grid<float>(); // a (raw) pointer to the grid of value type float on the CPU
            auto* deviceGrid = deviceHandle.deviceGrid<float>(); // and its deep copy on the GPU

            if (!deviceGrid || !cpuGrid)
                throw std::runtime_error("GridHandle did not contain a grid with value type float");

            launch_kernels(deviceGrid, cpuGrid, stream); // print grid values on both the CPU and GPU

            cudaStreamSynchronize(stream); // the kernels must finish before the device handle (whose buffer frees on this stream) goes away
        }
        cudaStreamDestroy(stream); // safe: nothing outlives the stream now
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}
//! [read_nanovdb_sphere_accessor_cuda]