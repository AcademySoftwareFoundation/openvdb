// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ImguiBackgroundCS.hlsl

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/
// Copyright (c) 2014-2022 NVIDIA Corporation. All rights reserved.

RWTexture2D<float4> colorOut;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 tidx = int2(dispatchThreadID.xy);

    float4 color = float4(0.f, 0.f, 0.f, 0.f);
    bool parity = ((tidx.x ^ tidx.y) & 16) != 0u;
    if (parity)
    {
        color = float4(0.1f, 0.1f, 0.1f, 0.f);
    }
    colorOut[tidx] = color;
}
