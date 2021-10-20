// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <stdio.h> // for printf

// This is called by the host only
void cpu_kernel(const nanovdb::NanoGrid<float>* cpuGrid)
{
    auto cpuAcc = cpuGrid->getAccessor();
    for (int i = 97; i < 104; ++i) {
        printf("(%3i,0,0) NanoVDB cpu: % -4.2f\n", i, cpuAcc.getValue(nanovdb::Coord(i, 0, 0)));
    }
}

// This is called by the device only
__global__ void gpu_kernel(const nanovdb::NanoGrid<float>* deviceGrid)
{
    if (threadIdx.x > 6)
        return;
    int  i = 97 + threadIdx.x;
    auto gpuAcc = deviceGrid->getAccessor();
    printf("(%3i,0,0) NanoVDB gpu: % -4.2f\n", i, gpuAcc.getValue(nanovdb::Coord(i, 0, 0)));
}

// This is called by the client code on the host
extern "C" void launch_kernels(const nanovdb::NanoGrid<float>* deviceGrid,
                               const nanovdb::NanoGrid<float>* cpuGrid,
                               cudaStream_t                    stream)
{
    // Launch the device kernel asynchronously
    gpu_kernel<<<1, 64, 0, stream>>>(deviceGrid);

    // Launch the host "kernel" (synchronously)
    cpu_kernel(cpuGrid);
}