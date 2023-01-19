// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <nanovdb/util/NodeManager.h>
#include <stdio.h> // for printf

// This is called by the host only
void cpu_kernel(const nanovdb::NodeManager<float>* cpuMgr)
{
    printf("NanoVDB cpu; %4.2f\n", cpuMgr->tree().getValue(nanovdb::Coord(99, 0, 0)));
}

// This is called by the device only
__global__ void gpu_kernel(const nanovdb::NodeManager<float>* deviceMgr)
{
    printf("NanoVDB gpu: %4.2f\n", deviceMgr->tree().getValue(nanovdb::Coord(99, 0, 0)));
}

// This is called by the client code on the host
extern "C" void launch_kernels(const nanovdb::NodeManager<float>* deviceMgr,
                               const nanovdb::NodeManager<float>* cpuMgr,
                               cudaStream_t                    stream)
{
    gpu_kernel<<<1, 1, 0, stream>>>(deviceMgr); // Launch the device kernel asynchronously

    cpu_kernel(cpuMgr); // Launch the host "kernel" (synchronously)
}