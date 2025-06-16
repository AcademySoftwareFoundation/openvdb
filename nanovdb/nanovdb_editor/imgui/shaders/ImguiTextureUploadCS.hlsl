// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ImguiTextureUploadCS.hlsl

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/
// Copyright (c) 2014-2022 NVIDIA Corporation. All rights reserved.

StructuredBuffer<uint> uploadIn;

RWTexture2D<float4> colorOut;

[numthreads(128, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileIdx1D = dispatchThreadID.x >> 7u;
    uint threadIdx = dispatchThreadID.x & 127u;

    int2 tileIdx = int2(int(tileIdx1D & 511u), int(tileIdx1D >> 9u));

    for (uint subIdx1D = threadIdx; subIdx1D < 18u * 18u; subIdx1D +=128u)
    {
        uint colorRaw = uploadIn[tileIdx1D * 18u * 18u + subIdx1D];

        float4 color = float4(
            float((colorRaw >> 0u) & 255u) * (1.f / 255.f),
            float((colorRaw >> 8u) & 255u) * (1.f / 255.f),
            float((colorRaw >> 16u) & 255u) * (1.f / 255.f),
            float((colorRaw >> 24u) & 255u) * (1.f / 255.f),
        );

        int2 subIdx = int2(
            int(subIdx1D % 18u),
            int(subIdx1D / 18u)
        );

        colorOut[18u * tileIdx + subIdx] = color;
    }
}
