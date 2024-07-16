// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <nanovdb/cuda/GridHandle.cuh>// required since GridHandle<DeviceBuffer> has device code
#include <stdio.h> // for printf

// This is called by the host only
void cpu_kernel(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* cpuGrid)
{
    nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> acc(*cpuGrid);
    //printf("\nNanoVDB CPU: channels=%u values=%lu\n", acc.grid().blindDataCount(), acc.root().maximum());
    printf("NanoVDB CPU; %lu\n", acc.idx(  0, 0, 0));
    printf("NanoVDB CPU; %lu\n", acc.idx( 99, 0, 0));
    printf("NanoVDB CPU; %lu\n", acc.idx(100, 0, 0));
    printf("NanoVDB CPU; %4.2f\n",   acc(  0, 0, 0));
    printf("NanoVDB CPU; %4.2f\n",   acc( 99, 0, 0));
    printf("NanoVDB CPU; %4.2f\n",   acc(100, 0, 0));
}

// This is called by the device only
__global__ void gpu_kernel(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid)
{
    nanovdb::ChannelAccessor<float, nanovdb::ValueOnIndex> acc(*gpuGrid);
    //printf("\nNanoVDB GPU: channels=%u values=%lu\n", gpuGrid->blindDataCount(), acc.root().maximum());
    printf("NanoVDB GPU; %lu\n", acc.idx(  0, 0, 0));
    printf("NanoVDB GPU; %lu\n", acc.idx( 99, 0, 0));
    printf("NanoVDB GPU; %lu\n", acc.idx(100, 0, 0));
    printf("NanoVDB GPU; %4.2f\n",   acc(  0, 0, 0));
    printf("NanoVDB GPU; %4.2f\n",   acc( 99, 0, 0));
    printf("NanoVDB GPU; %4.2f\n",   acc(100, 0, 0));
}

// This is called by the client code on the host
extern "C" void launch_kernels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* gpuGrid,
                               const nanovdb::NanoGrid<nanovdb::ValueOnIndex>* cpuGrid,
                               cudaStream_t                                    stream)
{
    gpu_kernel<<<1, 1, 0, stream>>>(gpuGrid); // Launch the device kernel asynchronously

    cpu_kernel(cpuGrid); // Launch the host "kernel" (synchronously)
}