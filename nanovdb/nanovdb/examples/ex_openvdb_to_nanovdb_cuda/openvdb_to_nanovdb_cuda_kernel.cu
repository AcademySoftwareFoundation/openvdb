// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <nanovdb/cuda/GridHandle.cuh>// required since GridHandle<DeviceBuffer> has device code
#include <stdio.h> // for printf

// This is called by the host only
void cpu_kernel(const nanovdb::NanoGrid<float>* cpuGrid)
{
    printf("NanoVDB cpu; %4.2f\n", cpuGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));
}

// This is called by the device only
__global__ void gpu_kernel(const nanovdb::NanoGrid<float>* deviceGrid)
{
    printf("NanoVDB gpu: %4.2f\n", deviceGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));
}

// This is called by the client code on the host
extern "C" void launch_kernels(const nanovdb::NanoGrid<float>* deviceGrid,
                               const nanovdb::NanoGrid<float>* cpuGrid,
                               cudaStream_t                    stream)
{
    gpu_kernel<<<1, 1, 0, stream>>>(deviceGrid); // Launch the device kernel asynchronously

    cpu_kernel(cpuGrid); // Launch the host "kernel" (synchronously)
}