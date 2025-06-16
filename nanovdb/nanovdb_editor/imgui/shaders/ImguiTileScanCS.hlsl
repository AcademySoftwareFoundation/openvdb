// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ImguiTileScanCS.hlsl

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/
// Copyright (c) 2014-2022 NVIDIA Corporation. All rights reserved.

#include "ImguiParams.h"

ConstantBuffer<imgui_renderer_params_t> paramsIn;

RWStructuredBuffer<uint> tileCountOut;
RWStructuredBuffer<uint> totalCountOut;

#include "ImguiBlockScan.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int tidx = int(dispatchThreadID.x);
    uint threadIdx = uint(tidx) & 255u;

    uint accumIdx = 0u;
    for (uint passIdx = 0u; passIdx < paramsIn.numTileBucketPasses; passIdx++)
    {
        uint tileBucketIdx = (passIdx << 8u) + threadIdx;

        uint bucketValue = (tileBucketIdx < paramsIn.numTileBuckets) ? tileCountOut[tileBucketIdx + paramsIn.tileLocalTotalOffset] : 0u;

        uint scanVal = blockScan(threadIdx, bucketValue);
        uint allocIdx = scanVal - bucketValue;

        if (tileBucketIdx < paramsIn.numTileBuckets)
        {
            tileCountOut[tileBucketIdx + paramsIn.tileGlobalScanOffset] = allocIdx + accumIdx;
        }

        GroupMemoryBarrierWithGroupSync();

        accumIdx += stotalCount;
    }

    if (threadIdx == 0u)
    {
        totalCountOut[0u] = accumIdx;
    }

    // temp feedback
    //totalCountOut[1u + tidx] = tileCountOut[tidx + paramsIn.tileNumTrianglesOffset];
}
