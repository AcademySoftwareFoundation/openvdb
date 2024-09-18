// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <nanovdb/cuda/GridHandle.cuh>// required since GridHandle<DeviceBuffer> has device code
#include <stdio.h> // for printf

// This is called by the host only
void cpu_kernel(const nanovdb::NanoGrid<float>* cpuGrid)
{
    auto cpuAcc = cpuGrid->getAccessor();
    for (int k=-3; k<=3; k+=6) {
        printf("NanoVDB cpu: (%i,%i,%i)=%4.2f\n", 1, 2, k, cpuAcc.getValue(nanovdb::Coord(1, 2, k)));
    }
}

// This is called by the device only
__global__ void gpu_kernel(const nanovdb::NanoGrid<float>* deviceGrid)
{
    if (threadIdx.x != 0 && threadIdx.x != 6) return;
    int k = threadIdx.x - 3;
    auto gpuAcc = deviceGrid->getAccessor();
    printf("NanoVDB gpu: (%i,%i,%i)=%4.2f\n", 1, 2, k, gpuAcc.getValue(nanovdb::Coord(1, 2, k)));
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
