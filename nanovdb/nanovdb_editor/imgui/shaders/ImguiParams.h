// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ImguiParams.h

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#ifndef IMGUI_PARAMS_CPU
typedef uint pnanovdb_uint32_t;
#endif

struct imgui_renderer_params_t
{
    pnanovdb_uint32_t numVertices;
    pnanovdb_uint32_t numIndices;
    pnanovdb_uint32_t numDrawCmds;
    pnanovdb_uint32_t numBlocks;

    float width;
    float height;
    float widthInv;
    float heightInv;

    pnanovdb_uint32_t tileGridDim_x;
    pnanovdb_uint32_t tileGridDim_y;
    pnanovdb_uint32_t tileGridDim_xy;
    pnanovdb_uint32_t tileDimBits;

    pnanovdb_uint32_t maxTriangles;
    pnanovdb_uint32_t tileNumTrianglesOffset;
    pnanovdb_uint32_t tileLocalScanOffset;
    pnanovdb_uint32_t tileLocalTotalOffset;

    pnanovdb_uint32_t tileGlobalScanOffset;
    pnanovdb_uint32_t numTileBuckets;
    pnanovdb_uint32_t numTileBucketPasses;
    pnanovdb_uint32_t pad3;
};

struct imgui_renderer_draw_cmd_t
{
    float clipRect_x;
    float clipRect_y;
    float clipRect_z;
    float clipRect_w;
    pnanovdb_uint32_t elemCount;
    pnanovdb_uint32_t userTexture;
    pnanovdb_uint32_t vertexOffset;
    pnanovdb_uint32_t indexOffset;
};
