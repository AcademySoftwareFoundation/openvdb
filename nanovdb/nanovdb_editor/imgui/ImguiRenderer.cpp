
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ImguiRenderer.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include <imgui.h>

#include "ImguiRenderer.h"

#include "UploadBuffer.h"
#include "DynamicBuffer.h"

#define IMGUI_PARAMS_CPU
#include "shaders/ImguiParams.h"

#include <nanovdb_editor/compute/ComputeShader.h>

#include <memory>

namespace pnanovdb_imgui_renderer_default
{
    struct TextureTile
    {
        pnanovdb_uint8_t data[18u * 18u * 4u];
    };

    struct Texture
    {
        std::vector<TextureTile> tiles;
        int tileGridWidth;
        int tileGridHeight;
        int texWidth;
        int texHeight;
        pnanovdb_uint32_t textureId;
    };

    PNANOVDB_CAST_PAIR(pnanovdb_imgui_texture_t, Texture)

    enum imgui_shader
    {
        imgui_cs,
        imgui_copy_texture_cs,
        imgui_background_cs,
        imgui_build_cs,
        imgui_tile_cs,
        imgui_tile_count_cs,
        imgui_tile_scan_cs,
        imgui_texture_upload_cs,

        shader_count
    };

    static const char* imgui_shader_names[shader_count] = {
        "imgui/ImguiCS.hlsl",
        "imgui/ImguiCopyTextureCS.hlsl",
        "imgui/ImguiBackgroundCS.hlsl",
        "imgui/ImguiBuildCS.hlsl",
        "imgui/ImguiTileCS.hlsl",
        "imgui/ImguiTileCountCS.hlsl",
        "imgui/ImguiTileScanCS.hlsl",
        "imgui/ImguiTextureUploadCS.hlsl",
    };

    struct Renderer
    {
        pnanovdb_compute_interface_t compute_interface = {};
        pnanovdb_compute_shader_interface_t shaderInterface = {};

        pnanovdb_shader_context_t* shader_context[shader_count];

        pnanovdb_compute_upload_buffer_t vertexPosTexCoordBuffer = {};
        pnanovdb_compute_upload_buffer_t vertexColorBuffer = {};
        pnanovdb_compute_upload_buffer_t indicesBuffer = {};
        pnanovdb_compute_upload_buffer_t drawCmdsBuffer = {};
        pnanovdb_compute_upload_buffer_t constantBuffer = {};

        pnanovdb_compute_upload_buffer_t textureUpload = {};
        pnanovdb_compute_upload_buffer_t textureTableUpload = {};
        pnanovdb_compute_texture_t* textureDevice = nullptr;
        pnanovdb_compute_sampler_t* samplerLinear = nullptr;
        pnanovdb_uint32_t textureWidth = 0u;
        pnanovdb_uint32_t textureHeight = 0u;

        pnanovdb_compute_dynamic_buffer_t treeBuffer = {};
        pnanovdb_compute_dynamic_buffer_t tileCountBuffer = {};
        pnanovdb_compute_dynamic_buffer_t triangleBuffer = {};
        pnanovdb_compute_dynamic_buffer_t triangleRangeBuffer = {};

        pnanovdb_compute_buffer_t* totalCountBuffer = nullptr;

        std::vector<std::unique_ptr<Texture>> textures;
        pnanovdb_uint32_t textureIdCounter = 0u;
        bool textureDirty = true;

        std::vector<pnanovdb_uint32_t> textureTable;
        std::vector<pnanovdb_uint32_t> textureData;
    };

    PNANOVDB_CAST_PAIR(pnanovdb_imgui_renderer_t, Renderer)

    pnanovdb_imgui_texture_t* create_texture(
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        unsigned char* pixels,
        int texWidth,
        int texHeight
    );

    pnanovdb_imgui_renderer_t* create(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        unsigned char* pixels,
        int texWidth,
        int texHeight
    )
    {
        auto ptr = new Renderer();

        pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
        pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

        pnanovdb_compute_interface_t_duplicate(&ptr->compute_interface, compute_interface);
        pnanovdb_compute_shader_interface_t_duplicate(&ptr->shaderInterface, &compute->shader_interface);

        pnanovdb_compiler_settings_t compile_settings = {};
        pnanovdb_compiler_settings_init(&compile_settings);

        for (pnanovdb_uint32_t idx = 0u; idx < shader_count; idx++)
        {
            ptr->shader_context[idx] = compute->create_shader_context(imgui_shader_names[idx]);
            compute->init_shader(compute, queue, ptr->shader_context[idx], &compile_settings);
        }

        pnanovdb_compute_buffer_usage_t bufferUsage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC;

        pnanovdb_compute_upload_buffer_init(&ptr->compute_interface, context, &ptr->vertexPosTexCoordBuffer, bufferUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 4u * sizeof(float));
        pnanovdb_compute_upload_buffer_init(&ptr->compute_interface, context, &ptr->vertexColorBuffer, bufferUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, sizeof(pnanovdb_uint32_t));
        pnanovdb_compute_upload_buffer_init(&ptr->compute_interface, context, &ptr->indicesBuffer, bufferUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, sizeof(pnanovdb_uint32_t));
        pnanovdb_compute_upload_buffer_init(&ptr->compute_interface, context, &ptr->drawCmdsBuffer, bufferUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, sizeof(imgui_renderer_draw_cmd_t));
        pnanovdb_compute_upload_buffer_init(&ptr->compute_interface, context, &ptr->constantBuffer, PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 0u);

        pnanovdb_compute_upload_buffer_init(&ptr->compute_interface, context, &ptr->textureUpload, bufferUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 4u);
        pnanovdb_compute_upload_buffer_init(&ptr->compute_interface, context, &ptr->textureTableUpload, bufferUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 4u);

        pnanovdb_compute_sampler_desc_t samplerDesc = {};
        samplerDesc.filter_mode = PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_LINEAR;
        samplerDesc.address_mode_u = PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_WRAP;
        samplerDesc.address_mode_v = PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_WRAP;
        samplerDesc.address_mode_w = PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_WRAP;

        ptr->samplerLinear = ptr->compute_interface.create_sampler(context, &samplerDesc);

        pnanovdb_compute_buffer_usage_t deviceBufUsage = PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED;

        pnanovdb_compute_dynamic_buffer_init(&ptr->compute_interface, context, &ptr->treeBuffer, deviceBufUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 4u * sizeof(int));
        pnanovdb_compute_dynamic_buffer_init(&ptr->compute_interface, context, &ptr->tileCountBuffer, deviceBufUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, sizeof(pnanovdb_uint32_t));
        pnanovdb_compute_dynamic_buffer_init(&ptr->compute_interface, context, &ptr->triangleBuffer, deviceBufUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, sizeof(pnanovdb_uint32_t));
        pnanovdb_compute_dynamic_buffer_init(&ptr->compute_interface, context, &ptr->triangleRangeBuffer, deviceBufUsage, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 2u * sizeof(pnanovdb_uint32_t));

        pnanovdb_compute_buffer_desc_t totalCountDesc = {};
        totalCountDesc.usage = deviceBufUsage;
        totalCountDesc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        totalCountDesc.structure_stride = sizeof(pnanovdb_uint32_t);
        totalCountDesc.size_in_bytes = 1024u * sizeof(pnanovdb_uint32_t);

        ptr->totalCountBuffer = ptr->compute_interface.create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK, &totalCountDesc);

        create_texture(context, cast(ptr), pixels, texWidth, texHeight);

        return cast(ptr);
    }

    void destroy(const pnanovdb_compute_t* compute, pnanovdb_compute_queue_t* queue, pnanovdb_imgui_renderer_t* renderer)
    {
        auto ptr = cast(renderer);

        ptr->textures.clear();

        pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

        pnanovdb_compute_upload_buffer_destroy(context, &ptr->vertexPosTexCoordBuffer);
        pnanovdb_compute_upload_buffer_destroy(context, &ptr->vertexColorBuffer);
        pnanovdb_compute_upload_buffer_destroy(context, &ptr->indicesBuffer);
        pnanovdb_compute_upload_buffer_destroy(context, &ptr->drawCmdsBuffer);
        pnanovdb_compute_upload_buffer_destroy(context, &ptr->constantBuffer);

        pnanovdb_compute_upload_buffer_destroy(context, &ptr->textureUpload);
        pnanovdb_compute_upload_buffer_destroy(context, &ptr->textureTableUpload);
        if (ptr->textureDevice)
        {
            ptr->compute_interface.destroy_texture(context, ptr->textureDevice);
            ptr->textureDevice = nullptr;
        }
        ptr->compute_interface.destroy_sampler(context, ptr->samplerLinear);

        pnanovdb_compute_dynamic_buffer_destroy(context, &ptr->treeBuffer);
        pnanovdb_compute_dynamic_buffer_destroy(context, &ptr->tileCountBuffer);
        pnanovdb_compute_dynamic_buffer_destroy(context, &ptr->triangleBuffer);
        pnanovdb_compute_dynamic_buffer_destroy(context, &ptr->triangleRangeBuffer);

        ptr->compute_interface.destroy_buffer(context, ptr->totalCountBuffer);

        for (pnanovdb_uint32_t idx = 0u; idx < shader_count; idx++)
        {
            compute->destroy_shader_context(compute, queue, ptr->shader_context[idx]);
        }

        delete ptr;
    }

    void render(const pnanovdb_compute_t* compute,
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        ImDrawData* drawData,
        pnanovdb_uint32_t width,
        pnanovdb_uint32_t height,
        pnanovdb_compute_texture_transient_t* colorIn,
        pnanovdb_compute_texture_transient_t* colorOut)
    {
        auto ptr = cast(renderer);

        if (ptr->textureDirty)
        {
            ptr->textureDirty = false;

            ptr->textureTable.resize(0u);
            ptr->textureTable.push_back((pnanovdb_uint32_t)ptr->textures.size()); // tableSize
            ptr->textureTable.push_back(0u); // atlasGridWidthBits
            ptr->textureTable.push_back(0u); // atlasWidthInv
            ptr->textureTable.push_back(0u); // atlasHeightInv
            for (pnanovdb_uint64_t textureIdx = 0u; textureIdx < ptr->textures.size(); textureIdx++)
            {
                ptr->textureTable.push_back(ptr->textures[textureIdx]->textureId);
            }
            pnanovdb_uint32_t tileGridOffset = 0u;
            for (pnanovdb_uint64_t textureIdx = 0u; textureIdx < ptr->textures.size(); textureIdx++)
            {
                ptr->textureTable.push_back(ptr->textures[textureIdx]->texWidth);
                ptr->textureTable.push_back(ptr->textures[textureIdx]->texHeight);
                ptr->textureTable.push_back(ptr->textures[textureIdx]->tileGridWidth);
                ptr->textureTable.push_back(tileGridOffset);

                tileGridOffset += ptr->textures[textureIdx]->tileGridWidth *
                    ptr->textures[textureIdx]->tileGridHeight;
            }

            pnanovdb_uint32_t atlasGridWidth = tileGridOffset;
            pnanovdb_uint32_t atlasGridHeight = 1u;
            if (tileGridOffset > 512)
            {
                atlasGridWidth = 512;
                atlasGridHeight = (tileGridOffset + 511) / 512;
            }
            pnanovdb_uint32_t atlasWidth = 18u * atlasGridWidth;
            pnanovdb_uint32_t atlasHeight = 18u * atlasGridHeight;

            ptr->textureTable[1u] = 9u; // atlasGridWidthBits
            *((float*)&ptr->textureTable[2u]) = 1.f / float(atlasWidth); // atlasWidthInv
            *((float*)&ptr->textureTable[3u]) = 1.f / float(atlasHeight); // atlasHeightInv

            auto mappedTable = (pnanovdb_uint32_t*)pnanovdb_compute_upload_buffer_map(context, &ptr->textureTableUpload, ptr->textureTable.size() * 4u);
            for (pnanovdb_uint64_t idx = 0u; idx < ptr->textureTable.size(); idx++)
            {
                mappedTable[idx] = ptr->textureTable[idx];
            }
            pnanovdb_compute_upload_buffer_unmap_device(context,
                &ptr->textureTableUpload, 0llu, ptr->textureTable.size() * 4u, "ImguiTextureTableUpload");

            pnanovdb_uint64_t uploadSize = tileGridOffset * 18u * 18u * 4u;
            auto mapped = (unsigned char*)pnanovdb_compute_upload_buffer_map(context, &ptr->textureUpload, uploadSize);
            pnanovdb_uint64_t uploadOffset = 0u;
            for (pnanovdb_uint64_t textureIdx = 0u; textureIdx < ptr->textures.size(); textureIdx++)
            {
                for (pnanovdb_uint64_t tileIdx = 0u; tileIdx < ptr->textures[textureIdx]->tiles.size(); tileIdx++)
                {
                    memcpy(mapped + uploadOffset * 18u * 18u * 4u,
                        ptr->textures[textureIdx]->tiles[tileIdx].data,
                        18u * 18u * 4u);
                    uploadOffset++;
                }
            }
            pnanovdb_compute_buffer_transient_t* uploadTransient = pnanovdb_compute_upload_buffer_unmap(context, &ptr->textureUpload);

            if (ptr->textureDevice && (ptr->textureWidth != atlasWidth || ptr->textureHeight != atlasHeight))
            {
                ptr->compute_interface.destroy_texture(context, ptr->textureDevice);
                ptr->textureDevice = nullptr;
                ptr->textureWidth = 0u;
                ptr->textureHeight = 0u;
            }
            if (!ptr->textureDevice)
            {
                ptr->textureWidth = atlasWidth;
                ptr->textureHeight = atlasHeight;

                pnanovdb_compute_texture_desc_t texDesc = {};
                texDesc.texture_type = PNANOVDB_COMPUTE_TEXTURE_TYPE_2D;
                texDesc.usage = PNANOVDB_COMPUTE_TEXTURE_USAGE_RW_TEXTURE | PNANOVDB_COMPUTE_TEXTURE_USAGE_TEXTURE;
                texDesc.format = PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM;
                texDesc.width = ptr->textureWidth;
                texDesc.height = ptr->textureHeight;
                texDesc.depth = 1u;
                texDesc.mip_levels = 1u;

                ptr->textureDevice = ptr->compute_interface.create_texture(context, &texDesc);
            }

            pnanovdb_compute_texture_transient_t* textureTransient = ptr->compute_interface.register_texture_as_transient(context, ptr->textureDevice);

            {
                pnanovdb_compute_resource_t resources[2u] = {};
                resources[0u].buffer_transient = uploadTransient;
                resources[1u].texture_transient = textureTransient;

                compute->dispatch_shader(
                    &ptr->compute_interface,
                    context,
                    ptr->shader_context[imgui_texture_upload_cs],
                    resources,
                    tileGridOffset, 1u, 1u,
                    "imgui_texture_upload"
                );
            }
        }

        pnanovdb_uint32_t numVertices = drawData->TotalVtxCount;
        pnanovdb_uint32_t numIndices = drawData->TotalIdxCount;
        pnanovdb_uint32_t numDrawCmds = 0u;
        for (int listIdx = 0; listIdx < drawData->CmdListsCount; listIdx++)
        {
            numDrawCmds += drawData->CmdLists[listIdx]->CmdBuffer.Size;
        }

        pnanovdb_uint32_t numTriangles = numIndices / 3u;

        pnanovdb_uint32_t trianglesPerBlock = 256u;
        pnanovdb_uint32_t numBlocks = (numTriangles + trianglesPerBlock - 1u) / trianglesPerBlock;
        pnanovdb_uint64_t treeNumBytes = numBlocks * (1u + 4u + 16u + 64u + 256u) * 4u * sizeof(int);

        pnanovdb_compute_dynamic_buffer_resize(context, &ptr->treeBuffer, treeNumBytes);

        pnanovdb_uint32_t tileDimBits = 4u;
        pnanovdb_uint32_t tileDim = 1u << tileDimBits;
        pnanovdb_uint32_t tileGridDim_x = (width + tileDim - 1u) / tileDim;
        pnanovdb_uint32_t tileGridDim_y = (height + tileDim - 1u) / tileDim;
        pnanovdb_uint32_t tileGridDim_xy = tileGridDim_x * tileGridDim_y;
        pnanovdb_uint32_t numTileBuckets = (tileGridDim_xy + 255u) / 256u;
        pnanovdb_uint32_t numTileBucketPasses = (numTileBuckets + 255u) / 256u;

        pnanovdb_uint64_t tileCountNumBytes = tileGridDim_x * tileGridDim_y * 3u * sizeof(pnanovdb_uint32_t);

        pnanovdb_compute_dynamic_buffer_resize(context, &ptr->tileCountBuffer, tileCountNumBytes);

        pnanovdb_uint32_t maxTriangles = 4u * 256u * 1024u;
        pnanovdb_uint64_t triangleBufferNumBytes = maxTriangles * sizeof(pnanovdb_uint32_t);

        pnanovdb_compute_dynamic_buffer_resize(context, &ptr->triangleBuffer, triangleBufferNumBytes);

        pnanovdb_uint64_t triangleRangeBufferNumBytes = tileGridDim_xy * 2u * sizeof(pnanovdb_uint32_t);

        pnanovdb_compute_dynamic_buffer_resize(context, &ptr->triangleRangeBuffer, triangleRangeBufferNumBytes);

        pnanovdb_uint64_t numBytesPosTex = (numVertices + 1u) * 4u * sizeof(float);
        pnanovdb_uint64_t numBytesColor = (numVertices + 1u) * sizeof(pnanovdb_uint32_t);
        pnanovdb_uint64_t numBytesIndices = (numIndices + 1u) * sizeof(pnanovdb_uint32_t);
        pnanovdb_uint64_t numBytesDrawCmds = (numDrawCmds + 1u) * sizeof(imgui_renderer_draw_cmd_t);

        auto mappedPosTex = (float*)pnanovdb_compute_upload_buffer_map(context, &ptr->vertexPosTexCoordBuffer, numBytesPosTex);
        auto mappedColor = (pnanovdb_uint32_t*)pnanovdb_compute_upload_buffer_map(context, &ptr->vertexColorBuffer, numBytesColor);
        auto mappedIndices = (pnanovdb_uint32_t*)pnanovdb_compute_upload_buffer_map(context, &ptr->indicesBuffer, numBytesIndices);
        auto mappedDrawCmds = (imgui_renderer_draw_cmd_t*)pnanovdb_compute_upload_buffer_map(context, &ptr->drawCmdsBuffer, numBytesDrawCmds);
        auto mapped = (imgui_renderer_params_t*)pnanovdb_compute_upload_buffer_map(context, &ptr->constantBuffer, sizeof(imgui_renderer_params_t));

        pnanovdb_uint32_t vertexOffset = 0u;
        pnanovdb_uint32_t indexOffset = 0u;
        pnanovdb_uint32_t drawCmdOffset = 0u;

        for (int cmdListIdx = 0u; cmdListIdx < drawData->CmdListsCount; cmdListIdx++)
        {
            ImDrawList* cmdList = drawData->CmdLists[cmdListIdx];

            // copy vertices
            for (int vertIdx = 0; vertIdx < cmdList->VtxBuffer.Size; vertIdx++)
            {
                pnanovdb_uint32_t writeIdx = vertIdx + vertexOffset;
                mappedPosTex[4u * writeIdx + 0u] = cmdList->VtxBuffer[vertIdx].pos.x;
                mappedPosTex[4u * writeIdx + 1u] = cmdList->VtxBuffer[vertIdx].pos.y;
                mappedPosTex[4u * writeIdx + 2u] = cmdList->VtxBuffer[vertIdx].uv.x;
                mappedPosTex[4u * writeIdx + 3u] = cmdList->VtxBuffer[vertIdx].uv.y;
                mappedColor[writeIdx] = cmdList->VtxBuffer[vertIdx].col;
            }

            // copy indices
            for (int indexIdx = 0; indexIdx < cmdList->IdxBuffer.Size; indexIdx++)
            {
                pnanovdb_uint32_t writeIdx = indexIdx + indexOffset;
                mappedIndices[writeIdx] = cmdList->IdxBuffer[indexIdx] + vertexOffset;        // apply vertex offset on CPU
            }

            // copy drawCmds
            pnanovdb_uint32_t indexOffsetLocal = indexOffset;
            for (int drawCmdIdx = 0; drawCmdIdx < cmdList->CmdBuffer.Size; drawCmdIdx++)
            {
                pnanovdb_uint32_t writeIdx = drawCmdIdx + drawCmdOffset;
                auto& dst = mappedDrawCmds[writeIdx];
                auto& src = cmdList->CmdBuffer[drawCmdIdx];
                dst.clipRect_x = src.ClipRect.x;
                dst.clipRect_y = src.ClipRect.y;
                dst.clipRect_z = src.ClipRect.z;
                dst.clipRect_w = src.ClipRect.w;
                dst.elemCount = src.ElemCount;
                dst.userTexture = src.TextureId ? cast((pnanovdb_imgui_texture_t*)src.TextureId)->textureId : 0u;
                dst.vertexOffset = 0u;                                                    // vertex offset already applied
                dst.indexOffset = indexOffsetLocal;

                indexOffsetLocal += src.ElemCount;
            }

            vertexOffset += pnanovdb_uint32_t(cmdList->VtxBuffer.Size);
            indexOffset += pnanovdb_uint32_t(cmdList->IdxBuffer.Size);
            drawCmdOffset += pnanovdb_uint32_t(cmdList->CmdBuffer.Size);
        }

        mapped->numVertices = numVertices;
        mapped->numIndices = numIndices;
        mapped->numDrawCmds = numDrawCmds;
        mapped->numBlocks = numBlocks;

        mapped->width = float(width);
        mapped->height = float(height);
        mapped->widthInv = 1.f / float(width);
        mapped->heightInv = 1.f / float(height);

        mapped->tileGridDim_x = tileGridDim_x;
        mapped->tileGridDim_y = tileGridDim_y;
        mapped->tileGridDim_xy = tileGridDim_xy;
        mapped->tileDimBits = tileDimBits;

        mapped->maxTriangles = maxTriangles;
        mapped->tileNumTrianglesOffset = 0u;
        mapped->tileLocalScanOffset = tileGridDim_xy;
        mapped->tileLocalTotalOffset = 2u * tileGridDim_xy;

        mapped->tileGlobalScanOffset = 2u * tileGridDim_xy + numTileBuckets;
        mapped->numTileBuckets = numTileBuckets;
        mapped->numTileBucketPasses = numTileBucketPasses;
        mapped->pad3 = 0u;

        //pnanovdb_compute_buffer_transient_t* vertexPosTexCoordTransient = pnanovdb_compute_upload_buffer_unmap_device(context, &ptr->vertexPosTexCoordBuffer, 0llu, numBytesPosTex);
        //pnanovdb_compute_buffer_transient_t* vertexColorTransient = pnanovdb_compute_upload_buffer_unmap_device(context, &ptr->vertexColorBuffer, 0llu, numBytesColor);
        //pnanovdb_compute_buffer_transient_t* indicesTransient = pnanovdb_compute_upload_buffer_unmap_device(context, &ptr->indicesBuffer, 0llu, numBytesIndices);
        //pnanovdb_compute_buffer_transient_t* drawCmdsInTransient = pnanovdb_compute_upload_buffer_unmap_device(context, &ptr->drawCmdsBuffer, 0llu, numBytesDrawCmds);
        //pnanovdb_compute_buffer_transient_t* paramsInTransient = pnanovdb_compute_upload_buffer_unmap(context, &ptr->constantBuffer);

        pnanovdb_compute_buffer_transient_t* vertexPosTexCoordTransient = pnanovdb_compute_upload_buffer_unmap(context, &ptr->vertexPosTexCoordBuffer);
        pnanovdb_compute_buffer_transient_t* vertexColorTransient = pnanovdb_compute_upload_buffer_unmap(context, &ptr->vertexColorBuffer);
        pnanovdb_compute_buffer_transient_t* indicesTransient = pnanovdb_compute_upload_buffer_unmap(context, &ptr->indicesBuffer);
        pnanovdb_compute_buffer_transient_t* drawCmdsInTransient = pnanovdb_compute_upload_buffer_unmap(context, &ptr->drawCmdsBuffer);
        pnanovdb_compute_buffer_transient_t* paramsInTransient = pnanovdb_compute_upload_buffer_unmap(context, &ptr->constantBuffer);

        pnanovdb_compute_texture_transient_t* textureTransient = ptr->compute_interface.register_texture_as_transient(context, ptr->textureDevice);
        pnanovdb_compute_buffer_transient_t* textureTableTransient = ptr->compute_interface.register_buffer_as_transient(context, ptr->textureTableUpload.device_buffer);
        pnanovdb_compute_buffer_transient_t* treeTransient = pnanovdb_compute_dynamic_buffer_get_transient(context, &ptr->treeBuffer);
        pnanovdb_compute_buffer_transient_t* tileCountTransient = pnanovdb_compute_dynamic_buffer_get_transient(context, &ptr->tileCountBuffer);
        pnanovdb_compute_buffer_transient_t* triangleTransient = pnanovdb_compute_dynamic_buffer_get_transient(context, &ptr->triangleBuffer);
        pnanovdb_compute_buffer_transient_t* triangleRangeTransient = pnanovdb_compute_dynamic_buffer_get_transient(context, &ptr->triangleRangeBuffer);

        auto totalCountMapped = (pnanovdb_uint32_t*)ptr->compute_interface.map_buffer(context, ptr->totalCountBuffer);

        ptr->compute_interface.unmap_buffer(context, ptr->totalCountBuffer);

        pnanovdb_compute_buffer_transient_t* totalCountTransient = ptr->compute_interface.register_buffer_as_transient(context, ptr->totalCountBuffer);

        // build acceleration structure
        {
            pnanovdb_compute_resource_t resources[6u] = {};
            resources[0u].buffer_transient = paramsInTransient;
            resources[1u].buffer_transient = vertexPosTexCoordTransient;
            resources[2u].buffer_transient = vertexColorTransient;
            resources[3u].buffer_transient = indicesTransient;
            resources[4u].buffer_transient = drawCmdsInTransient;
            resources[5u].buffer_transient = treeTransient;

            compute->dispatch_shader(
                &ptr->compute_interface,
                context,
                ptr->shader_context[imgui_build_cs],
                resources,
                numBlocks, 1u, 1u,
                "imgui_build"
            );
        }

        // count triangles per tile
        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = paramsInTransient;
            resources[1u].buffer_transient = treeTransient;
            resources[2u].buffer_transient = tileCountTransient;

            compute->dispatch_shader(
                &ptr->compute_interface,
                context,
                ptr->shader_context[imgui_tile_count_cs],
                resources,
                (tileGridDim_xy + 255u) / 256u, 1u, 1u,
                "imgui_tile_count"
            );
        }

        // scan buckets
        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = paramsInTransient;
            resources[1u].buffer_transient = tileCountTransient;
            resources[2u].buffer_transient = totalCountTransient;

            compute->dispatch_shader(
                &ptr->compute_interface,
                context,
                ptr->shader_context[imgui_tile_scan_cs],
                resources,
                1u, 1u, 1u,
                "imgui_tile_scan"
            );
        }

        // generate tile data
        {
            pnanovdb_compute_resource_t resources[6u] = {};
            resources[0u].buffer_transient = paramsInTransient;
            resources[1u].buffer_transient = treeTransient;
            resources[2u].buffer_transient = tileCountTransient;
            resources[3u].buffer_transient = drawCmdsInTransient;
            resources[4u].buffer_transient = triangleTransient;
            resources[5u].buffer_transient = triangleRangeTransient;

            compute->dispatch_shader(
                &ptr->compute_interface,
                context,
                ptr->shader_context[imgui_tile_cs],
                resources,
                (tileGridDim_xy + 255u) / 256u, 1u, 1u,
                "imgui_tile"
            );
        }

        // produce background if missing
        if (!colorIn)
        {
            pnanovdb_compute_texture_desc_t tex_desc = {};
            tex_desc.texture_type = PNANOVDB_COMPUTE_TEXTURE_TYPE_2D;
            tex_desc.usage = PNANOVDB_COMPUTE_TEXTURE_USAGE_TEXTURE | PNANOVDB_COMPUTE_TEXTURE_USAGE_RW_TEXTURE;
            tex_desc.format = PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM;
            tex_desc.width = width;
            tex_desc.height = height;
            tex_desc.depth = 1u;
            tex_desc.mip_levels = 1u;

            colorIn = ptr->compute_interface.get_texture_transient(context, &tex_desc);

            pnanovdb_compute_resource_t resources[1u] = {};
            resources[0u].texture_transient = colorIn;

            compute->dispatch_shader(
                &ptr->compute_interface,
                context,
                ptr->shader_context[imgui_background_cs],
                resources,
                (width + 7u) / 8u,
                (height + 7u) / 8u,
                1u,
                "imgui_background"
            );
        }

        // render
        {
            pnanovdb_compute_resource_t resources[12u] = {};
            resources[0u].buffer_transient = paramsInTransient;
            resources[1u].buffer_transient = vertexPosTexCoordTransient;
            resources[2u].buffer_transient = vertexColorTransient;
            resources[3u].buffer_transient = indicesTransient;
            resources[4u].buffer_transient = drawCmdsInTransient;
            resources[5u].buffer_transient = textureTableTransient;
            resources[6u].texture_transient = textureTransient;
            resources[7u].sampler = ptr->samplerLinear;
            resources[8u].buffer_transient = triangleTransient;
            resources[9u].buffer_transient = triangleRangeTransient;
            resources[10u].texture_transient = colorIn;
            resources[11u].texture_transient = colorOut;

            compute->dispatch_shader(
                &ptr->compute_interface,
                context,
                ptr->shader_context[imgui_cs],
                resources,
                (width + 7u) / 8u,
                (height + 7u) / 8u,
                1u,
                "imgui_render"
            );
        }
    }

    void copy_texture(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        pnanovdb_uint32_t width,
        pnanovdb_uint32_t height,
        pnanovdb_compute_texture_transient_t* colorIn,
        pnanovdb_compute_texture_transient_t* colorOut
    )
    {
        auto ptr = cast(renderer);

        pnanovdb_compute_resource_t resources[2u] = {};
        resources[0u].texture_transient = colorIn;
        resources[1u].texture_transient = colorOut;

        compute->dispatch_shader(
            &ptr->compute_interface,
            context,
            ptr->shader_context[imgui_copy_texture_cs],
            resources,
            (width + 7u) / 8u,
            (height + 7u) / 8u,
            1u,
            "imgui_copy_texture"
        );
    }

    void update_texture(
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        pnanovdb_imgui_texture_t* texture,
        unsigned char* pixels,
        int texWidth,
        int texHeight
    )
    {
        auto ptr = cast(renderer);
        auto tex = cast(texture);

        ptr->textureDirty = true;

        tex->texWidth = texWidth;
        tex->texHeight = texHeight;
        tex->tileGridWidth = (tex->texWidth + 15u) / 16u;
        tex->tileGridHeight = (tex->texHeight + 15u) / 16u;

        tex->tiles.reserve(tex->tileGridWidth * tex->tileGridHeight);
        tex->tiles.resize(tex->tileGridWidth * tex->tileGridHeight);

        for (int tj = 0; tj < tex->tileGridHeight; tj++)
        {
            for (int ti = 0; ti < tex->tileGridWidth; ti++)
            {
                int tile_idx = tj * tex->tileGridWidth + ti;
                for (int j = -1; j < 17; j++)
                {
                    for (int i = -1; i < 17; i++)
                    {
                        unsigned char r = 0;
                        unsigned char g = 0;
                        unsigned char b = 0;
                        unsigned char a = 0;

                        int src_i = (int)((pnanovdb_uint32_t)((ti << 4u) + i) % (pnanovdb_uint32_t)tex->texWidth);
                        int src_j = (int)((pnanovdb_uint32_t)((tj << 4u) + j) % (pnanovdb_uint32_t)tex->texHeight);
                        int src_idx = src_j * tex->texWidth + src_i;
                        r = pixels[4u * src_idx + 0];
                        g = pixels[4u * src_idx + 1];
                        b = pixels[4u * src_idx + 2];
                        a = pixels[4u * src_idx + 3];

                        int tile_subidx = (j + 1) * 18 + (i + 1);
                        tex->tiles[tile_idx].data[tile_subidx * 4 + 0] = r;
                        tex->tiles[tile_idx].data[tile_subidx * 4 + 1] = g;
                        tex->tiles[tile_idx].data[tile_subidx * 4 + 2] = b;
                        tex->tiles[tile_idx].data[tile_subidx * 4 + 3] = a;
                    }
                }
            }
        }
    }

    pnanovdb_imgui_texture_t* create_texture(
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        unsigned char* pixels,
        int texWidth,
        int texHeight
    )
    {
        auto ptr = cast(renderer);
        auto tex = new Texture();
        ptr->textures.push_back(std::unique_ptr<Texture>(tex));

        update_texture(context, renderer, cast(tex), pixels, texWidth, texHeight);

        tex->textureId = ptr->textureIdCounter;
        ptr->textureIdCounter++;

        return cast(tex);
    }

    void destroy_texture(
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        pnanovdb_imgui_texture_t* texture
    )
    {
        auto ptr = cast(renderer);
        auto tex = cast(texture);

        tex->tiles.clear();

        // remove from array
        for (pnanovdb_uint64_t idx = 0u; idx < ptr->textures.size(); idx++)
        {
            if (ptr->textures[idx].get() == tex)
            {
                ptr->textures.erase(ptr->textures.begin() + idx);
                break;
            }
        }

        ptr->textureDirty = true;
    }
}

pnanovdb_imgui_renderer_interface_t* pnanovdb_imgui_get_renderer_interface()
{
    using namespace pnanovdb_imgui_renderer_default;
    static pnanovdb_imgui_renderer_interface_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_imgui_renderer_interface_t) };
    iface.create = create;
    iface.destroy = destroy;
    iface.render = render;
    iface.copy_texture = copy_texture;
    iface.create_texture = create_texture;
    iface.update_texture = update_texture;
    iface.destroy_texture = destroy_texture;
    return &iface;
}
