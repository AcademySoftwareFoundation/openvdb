
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ContextVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include "CommonVulkan.h"

#if defined(_WIN32)
// Windows.h already included
#else
#include <time.h>
#endif

namespace pnanovdb_vulkan
{

Context* context_create(DeviceQueue* deviceQueue)
{
    auto ptr = new Context();

    ptr->logPrint = deviceQueue->device->logPrint;

    ptr->deviceQueue = deviceQueue;

    pnanovdb_compute_sampler_desc_t samplerDesc = {};
    samplerDesc.address_mode_u = PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_BORDER;
    samplerDesc.address_mode_v = PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_BORDER;
    samplerDesc.address_mode_w = PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_BORDER;
    samplerDesc.filter_mode = PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_POINT;

    createSampler(cast(ptr), &samplerDesc);

    ptr->profiler = profiler_create(ptr);

    return ptr;
}

void context_destroy(Context* ptr)
{
    profiler_destroy(ptr, ptr->profiler);

    context_destroyBuffers(ptr);
    context_destroyTextures(ptr);
    context_destroySamplers(ptr);

    delete ptr;
}

void context_resetNodes(Context* context)
{
    context->nodes.resize(0u);
}

void context_resetNode(ContextNode* node)
{
    node->type = eContextNodeType_unknown;
    node->params.memory = ContextNodeMemoryParams{};
    node->descriptorWrites.resize(0u);
    node->resources.resize(0u);

    node->bufferBarriers.resize(0u);
    node->imageBarriers.resize(0u);
}

void addPassCompute(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_dispatch_params_t* params)
{
    auto context = cast(contextIn);
    context->nodes.push_back(ContextNode());
    ContextNode* node = &context->nodes.back();
    context_resetNode(node);

    node->type = eContextNodeType_compute;
    node->params.compute = *params;

    for (pnanovdb_uint32_t descriptorIdx = 0u; descriptorIdx < params->descriptor_write_count; descriptorIdx++)
    {
        node->descriptorWrites.push_back(params->descriptor_writes[descriptorIdx]);
        node->resources.push_back(params->resources[descriptorIdx]);
    }
    node->params.compute.descriptor_writes = node->descriptorWrites.data();
    node->params.compute.resources = node->resources.data();
}

void addPassCopyBuffer(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_copy_buffer_params_t* params)
{
    auto context = cast(contextIn);
    context->nodes.push_back(ContextNode());
    ContextNode* node = &context->nodes.back();
    context_resetNode(node);

    node->type = eContextNodeType_copyBuffer;
    node->params.copyBuffer = *params;

    pnanovdb_compute_resource_t src = {};
    pnanovdb_compute_resource_t dst = {};
    src.buffer_transient = params->src;
    dst.buffer_transient = params->dst;

    node->descriptorWrites.push_back(pnanovdb_compute_descriptor_write_t{ PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER_COPY_SRC });
    node->resources.push_back(src);
    node->descriptorWrites.push_back(pnanovdb_compute_descriptor_write_t{ PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER_COPY_DST });
    node->resources.push_back(dst);
}

void context_flushNodes(Context* context)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    // reset lifetimes
    for (pnanovdb_uint32_t idx = 0u; idx < context->bufferTransients.size(); idx++)
    {
        context->bufferTransients[idx]->nodeBegin = int(context->nodes.size());
        context->bufferTransients[idx]->nodeEnd = -1;
    }
    for (pnanovdb_uint32_t idx = 0u; idx < context->textureTransients.size(); idx++)
    {
        context->textureTransients[idx]->nodeBegin = int(context->nodes.size());
        context->textureTransients[idx]->nodeEnd = -1;
    }

    // already allocated resources begin life at -1, and if active live the whole frame
    for (pnanovdb_uint32_t idx = 0u; idx < context->bufferTransients.size(); idx++)
    {
        auto transient = context->bufferTransients[idx].get();
        if (transient->buffer)
        {
            transient->nodeBegin = -1;
            if (transient->buffer->refCount > 0)
            {
                transient->nodeEnd = int(context->nodes.size());
            }
        }
    }
    for (pnanovdb_uint32_t idx = 0u; idx < context->textureTransients.size(); idx++)
    {
        auto transient = context->textureTransients[idx].get();
        if (transient->texture)
        {
            transient->nodeBegin = -1;
            if (transient->texture->refCount > 0)
            {
                transient->nodeEnd = int(context->nodes.size());
            }
        }
    }
    // compute transient lifetimes
    for (pnanovdb_uint32_t nodeIdx = 0u; nodeIdx < context->nodes.size(); nodeIdx++)
    {
        ContextNode* node = &context->nodes[nodeIdx];
        for (pnanovdb_uint32_t descriptorIdx = 0u; descriptorIdx < node->descriptorWrites.size(); descriptorIdx++)
        {
            pnanovdb_compute_resource_t* resource = &node->resources[descriptorIdx];
            if (resource->buffer_transient)
            {
                BufferTransient* bufferTransient = cast(resource->buffer_transient);
                if (int(nodeIdx) < bufferTransient->nodeBegin)
                {
                    bufferTransient->nodeBegin = int(nodeIdx);
                }
                if (int(nodeIdx) > bufferTransient->nodeEnd)
                {
                    bufferTransient->nodeEnd = int(nodeIdx);
                }
            }
            if (resource->texture_transient)
            {
                TextureTransient* textureTransient = cast(resource->texture_transient);
                if (int(nodeIdx) < textureTransient->nodeBegin)
                {
                    textureTransient->nodeBegin = int(nodeIdx);
                }
                if (int(nodeIdx) > textureTransient->nodeEnd)
                {
                    textureTransient->nodeEnd = int(nodeIdx);
                }
            }
        }
    }
    // extend lifetime for acquired
    for (pnanovdb_uint32_t idx = 0u; idx < context->bufferAcquires.size(); idx++)
    {
        if (context->bufferAcquires[idx]->bufferTransient)
        {
            context->bufferAcquires[idx]->bufferTransient->nodeEnd = int(context->nodes.size());
        }
    }
    for (pnanovdb_uint32_t idx = 0u; idx < context->textureAcquires.size(); idx++)
    {
        if (context->textureAcquires[idx]->textureTransient)
        {
            context->textureAcquires[idx]->textureTransient->nodeEnd = int(context->nodes.size());
        }
    }

    // reset per node transient arrays
    for (pnanovdb_uint32_t nodeIdx = 0u; nodeIdx < context->nodes.size(); nodeIdx++)
    {
        ContextNode* node = &context->nodes[nodeIdx];
        node->bufferTransientsCreate.resize(0u);
        node->textureTransientsCreate.resize(0u);
        node->bufferTransientsDestroy.resize(0u);
        node->textureTransientsDestroy.resize(0u);
    }
    // scatter transients to per node arrays
    for (pnanovdb_uint32_t idx = 0u; idx < context->bufferTransients.size(); idx++)
    {
        auto transient = context->bufferTransients[idx].get();
        if (transient->nodeBegin >= 0 && transient->nodeBegin < int(context->nodes.size()))
        {
            context->nodes[transient->nodeBegin].bufferTransientsCreate.push_back(transient);
        }
        if (transient->nodeEnd >= 0 && transient->nodeEnd < int(context->nodes.size()))
        {
            context->nodes[transient->nodeEnd].bufferTransientsDestroy.push_back(transient);
        }
    }
    for (pnanovdb_uint32_t idx = 0u; idx < context->textureTransients.size(); idx++)
    {
        auto transient = context->textureTransients[idx].get();
        if (transient->nodeBegin >= 0 && transient->nodeBegin < int(context->nodes.size()))
        {
            context->nodes[transient->nodeBegin].textureTransientsCreate.push_back(transient);
        }
        if (transient->nodeEnd >= 0 && transient->nodeEnd < int(context->nodes.size()))
        {
            context->nodes[transient->nodeEnd].textureTransientsDestroy.push_back(transient);
        }
    }

    // for node -1, revive already allocated resources if referenced in a different node
    for (pnanovdb_uint32_t idx = 0u; idx < context->bufferTransients.size(); idx++)
    {
        auto transient = context->bufferTransients[idx].get();
        if (transient->buffer && transient->buffer->refCount == 0 && transient->nodeEnd != -1)
        {
            transient->buffer->refCount = 1;
            transient->buffer->lastActive = context->deviceQueue->nextFenceValue;
        }
    }
    for (pnanovdb_uint32_t idx = 0u; idx < context->textureTransients.size(); idx++)
    {
        auto transient = context->textureTransients[idx].get();
        if (transient->texture && transient->texture->refCount == 0 && transient->nodeEnd != -1)
        {
            transient->texture->refCount = 1;
            transient->texture->lastActive = context->deviceQueue->nextFenceValue;
        }
    }
    // resolve transient resources
    for (pnanovdb_uint32_t nodeIdx = 0u; nodeIdx < context->nodes.size(); nodeIdx++)
    {
        ContextNode* node = &context->nodes[nodeIdx];

        for (pnanovdb_uint32_t idx = 0u; idx < node->bufferTransientsCreate.size(); idx++)
        {
            auto transient = node->bufferTransientsCreate[idx];
            if (transient->aliasBuffer)
            {
                transient->buffer = transient->aliasBuffer->buffer;
                transient->buffer->refCount++;
            }
            else
            {
                if (transient->buffer)
                {
                    context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "pnanovdb_compute_context_t::BufferTransient double create");
                }
                transient->buffer = cast(createBuffer(cast(context), PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &transient->desc));
            }
        }
        for (pnanovdb_uint32_t idx = 0u; idx < node->textureTransientsCreate.size(); idx++)
        {
            auto transient = node->textureTransientsCreate[idx];
            if (transient->aliasTexture)
            {
                transient->texture = transient->aliasTexture->texture;
                transient->texture->refCount++;
            }
            else
            {
                if (transient->texture)
                {
                    context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "pnanovdb_compute_context_t::TextureTransient double create");
                }
                transient->texture = cast(createTexture(cast(context), &transient->desc));
            }
        }

        for (pnanovdb_uint32_t idx = 0u; idx < node->bufferTransientsDestroy.size(); idx++)
        {
            auto transient = node->bufferTransientsDestroy[idx];
            destroyBuffer(cast(context), cast(transient->buffer));
        }
        for (pnanovdb_uint32_t idx = 0u; idx < node->textureTransientsDestroy.size(); idx++)
        {
            auto transient = node->textureTransientsDestroy[idx];
            destroyTexture(cast(context), cast(transient->texture));
        }
    }
    // for the final node, allocate resources needed for capture
    for (pnanovdb_uint32_t idx = 0u; idx < context->bufferTransients.size(); idx++)
    {
        auto transient = context->bufferTransients[idx].get();
        if (transient->nodeBegin == int(context->nodes.size()) && transient->nodeEnd == int(context->nodes.size()))
        {
            transient->buffer = cast(createBuffer(cast(context), PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &transient->desc));
        }
    }
    for (pnanovdb_uint32_t idx = 0u; idx < context->textureTransients.size(); idx++)
    {
        auto transient = context->textureTransients[idx].get();
        if (transient->nodeBegin == int(context->nodes.size()) && transient->nodeEnd == int(context->nodes.size()))
        {
            transient->texture = cast(createTexture(cast(context), &transient->desc));
        }
    }

    // precompute barriers
    for (pnanovdb_uint32_t nodeIdx = 0u; nodeIdx < context->nodes.size(); nodeIdx++)
    {
        ContextNode* node = &context->nodes[nodeIdx];
        for (pnanovdb_uint32_t descriptorIdx = 0u; descriptorIdx < node->descriptorWrites.size(); descriptorIdx++)
        {
            pnanovdb_compute_descriptor_write_t* descriptorWrite = &node->descriptorWrites[descriptorIdx];
            pnanovdb_compute_resource_t* resource = &node->resources[descriptorIdx];

            if (resource->buffer_transient)
            {
                Buffer* buffer = cast(resource->buffer_transient)->buffer;

                VkBufferMemoryBarrier bufferBarrier = buffer->currentBarrier;

                // new becomes old
                bufferBarrier.srcAccessMask = bufferBarrier.dstAccessMask;
                bufferBarrier.srcQueueFamilyIndex = bufferBarrier.dstQueueFamilyIndex;

                // establish new
                if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_CONSTANT_BUFFER ||
                    descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_STRUCTURED_BUFFER ||
                    descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER)
                {
                    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                }
                else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_STRUCTURED_BUFFER ||
                         descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_BUFFER)
                {
                    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                }
                else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_INDIRECT_BUFFER)
                {
                    bufferBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
                }
                else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER_COPY_SRC)
                {
                    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                }
                else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER_COPY_DST)
                {
                    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                }
                bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

                // capture barrier
                node->bufferBarriers.push_back(bufferBarrier);

                buffer->currentBarrier = bufferBarrier;
            }
            if (resource->texture_transient)
            {
                Texture* texture = cast(resource->texture_transient)->texture;

                VkImageMemoryBarrier imageBarrier = texture->currentBarrier;

                // new becomes old
                imageBarrier.srcAccessMask = imageBarrier.dstAccessMask;
                imageBarrier.oldLayout = imageBarrier.newLayout;
                imageBarrier.srcQueueFamilyIndex = imageBarrier.dstQueueFamilyIndex;

                // establish new
                if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE)
                {
                    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
                else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE)
                {
                    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                }
                else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE_COPY_SRC)
                {
                    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                }
                else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE_COPY_DST)
                {
                    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                }
                imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

                // commit barrier, except if multiple reads detected
                pnanovdb_bool_t shouldCommit = PNANOVDB_TRUE;
                if (imageBarrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                    imageBarrier.newLayout == imageBarrier.oldLayout &&
                    imageBarrier.dstAccessMask == imageBarrier.srcAccessMask)
                {
                    shouldCommit = PNANOVDB_FALSE;
                }
                if (shouldCommit)
                {
                    node->imageBarriers.push_back(imageBarrier);

                    texture->currentBarrier = imageBarrier;
                }
            }
        }
    }

    profiler_beginCapture(context, context->profiler, context->nodes.size());

    // execute nodes
    for (pnanovdb_uint32_t nodeIdx = 0u; nodeIdx < context->nodes.size(); nodeIdx++)
    {
        ContextNode* node = &context->nodes[nodeIdx];

        if (node->bufferBarriers.size() > 0u ||
            node->imageBarriers.size() > 0u)
        {
            loader->vkCmdPipelineBarrier(
                context->deviceQueue->commandBuffer,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0u, nullptr,
                (uint32_t)node->bufferBarriers.size(), node->bufferBarriers.data(),
                (uint32_t)node->imageBarriers.size(), node->imageBarriers.data()
            );
        }

        if (node->type == eContextNodeType_compute)
        {
            const auto& params = node->params.compute;
            node->label = params.debug_label;

            computePipeline_dispatch(context, &params);
        }
        else if (node->type == eContextNodeType_copyBuffer)
        {
            const auto& params = node->params.copyBuffer;
            node->label = params.debug_label;

            VkBufferCopy region = {};
            region.srcOffset = params.src_offset;
            region.dstOffset = params.dst_offset;
            region.size = params.num_bytes;

            loader->vkCmdCopyBuffer(
                context->deviceQueue->commandBuffer,
                cast(params.src)->buffer->bufferVk,
                cast(params.dst)->buffer->bufferVk,
                1u, &region
            );
        }

        profiler_timestamp(context, context->profiler, node->label);
    }

    profiler_endCapture(context, context->profiler);

    profiler_processCaptures(context, context->profiler);

    // restore resource states
    context->restore_bufferBarriers.resize(0u);
    context->restore_imageBarriers.resize(0u);
    for (pnanovdb_uint32_t idx = 0u; idx < context->bufferTransients.size(); idx++)
    {
        auto transient = context->bufferTransients[idx].get();
        if (transient->buffer)
        {
            auto buffer = transient->buffer;

            VkBufferMemoryBarrier bufferBarrier = buffer->currentBarrier;

            // new becomes old
            bufferBarrier.srcAccessMask = bufferBarrier.dstAccessMask;
            bufferBarrier.srcQueueFamilyIndex = bufferBarrier.dstQueueFamilyIndex;

            // restore state
            bufferBarrier.dstAccessMask = buffer->restoreBarrier.dstAccessMask;
            bufferBarrier.dstQueueFamilyIndex = buffer->restoreBarrier.dstQueueFamilyIndex;

            // capture
            context->restore_bufferBarriers.push_back(bufferBarrier);

            buffer->currentBarrier = bufferBarrier;
        }
    }
    for (pnanovdb_uint32_t idx = 0u; idx < context->textureTransients.size(); idx++)
    {
        auto transient = context->textureTransients[idx].get();
        if (transient->texture)
        {
            auto texture = transient->texture;

            VkImageMemoryBarrier imageBarrier = texture->currentBarrier;

            // new becomes old
            imageBarrier.srcAccessMask = imageBarrier.dstAccessMask;
            imageBarrier.oldLayout = imageBarrier.newLayout;
            imageBarrier.srcQueueFamilyIndex = imageBarrier.dstQueueFamilyIndex;

            // restore state
            imageBarrier.dstAccessMask = texture->restoreBarrier.dstAccessMask;
            imageBarrier.newLayout = texture->restoreBarrier.newLayout;
            imageBarrier.dstQueueFamilyIndex = texture->restoreBarrier.dstQueueFamilyIndex;

            // commit barrier, except if multiple reads detected
            pnanovdb_bool_t shouldCommit = PNANOVDB_TRUE;
            if (imageBarrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                imageBarrier.newLayout == imageBarrier.oldLayout &&
                imageBarrier.dstAccessMask == imageBarrier.srcAccessMask)
            {
                shouldCommit = PNANOVDB_FALSE;
            }
            if (shouldCommit)
            {
                context->restore_imageBarriers.push_back(imageBarrier);

                texture->currentBarrier = imageBarrier;
            }
        }
    }
    if (context->restore_bufferBarriers.size() > 0u ||
        context->restore_imageBarriers.size() > 0u )
    {
        loader->vkCmdPipelineBarrier(
            context->deviceQueue->commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0u, nullptr,
            (uint32_t)context->restore_bufferBarriers.size(), context->restore_bufferBarriers.data(),
            (uint32_t)context->restore_imageBarriers.size(), context->restore_imageBarriers.data()
        );
    }

    // global barrier
    {
        VkMemoryBarrier barriers[3u] = {
            { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT},
            { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT},
            { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_WRITE_BIT},
        };

        loader->vkCmdPipelineBarrier(
            context->deviceQueue->commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            3u, barriers,
            0u, nullptr,
            0u, nullptr
        );
    }

    // process buffer acquires
    {
        for (pnanovdb_uint32_t idx = 0u; idx < context->bufferAcquires.size(); idx++)
        {
            auto bufferAcquire = context->bufferAcquires[idx].get();
            if (!bufferAcquire->buffer)
            {
                bufferAcquire->buffer = bufferAcquire->bufferTransient->buffer;
                bufferAcquire->bufferTransient = nullptr;
                if (bufferAcquire->buffer)
                {
                    bufferAcquire->buffer->refCount++;
                }
            }
        }
    }
    // process texture acquires
    {
        for (pnanovdb_uint32_t idx = 0u; idx < context->textureAcquires.size(); idx++)
        {
            auto textureAcquire = context->textureAcquires[idx].get();
            if (!textureAcquire->texture)
            {
                textureAcquire->texture = textureAcquire->textureTransient->texture;
                textureAcquire->textureTransient = nullptr;
                if (textureAcquire->texture)
                {
                    textureAcquire->texture->refCount++;
                }
            }
        }
    }

    // reset transient arrays
    context->bufferTransients.resize(0u);
    context->textureTransients.resize(0u);

    // apply deferred destroyBuffer
    for (pnanovdb_uint32_t idx = context->deferredReleaseBuffers.size() - 1u; idx < context->deferredReleaseBuffers.size(); idx--)
    {
        auto ptr = cast(context->deferredReleaseBuffers[idx]);
        if (ptr)
        {
            if (ptr->memory_type != PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE &&
                ptr->lastActive > context->deviceQueue->lastFenceCompleted)
            {
                continue;
            }
            ptr->refCount--;
            ptr->lastActive = context->deviceQueue->nextFenceValue;
        }
        context->deferredReleaseBuffers.erase(context->deferredReleaseBuffers.begin() + idx);
    }
    // apply deferred destroyTexture
    for (pnanovdb_uint32_t idx = context->deferredReleaseTextures.size() - 1u; idx < context->deferredReleaseTextures.size(); idx--)
    {
        auto ptr = cast(context->deferredReleaseTextures[idx]);
        if (context->deferredReleaseTextures[idx])
        {
            ptr->refCount--;
            ptr->lastActive = context->deviceQueue->nextFenceValue;
        }
        context->deferredReleaseTextures.erase(context->deferredReleaseTextures.begin() + idx);
    }

    // clean up unused resources
    for (pnanovdb_uint32_t idx = context->pool_buffers.size() - 1u; idx < context->pool_buffers.size(); idx--)
    {
        auto ptr = context->pool_buffers[idx].get();
        if (ptr->refCount == 0 && (ptr->lastActive + context->minLifetime) <= context->deviceQueue->lastFenceCompleted)
        {
            ptr = context->pool_buffers[idx].release();
            buffer_destroy(context, ptr);

            context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Vulkan destroy pool buffer %d", idx);

            context->pool_buffers.erase(context->pool_buffers.begin() + idx);
        }
    }
    for (pnanovdb_uint32_t idx = context->pool_textures.size() - 1u; idx < context->pool_textures.size(); idx--)
    {
        auto ptr = context->pool_textures[idx].get();
        if (ptr->refCount == 0 && (ptr->lastActive + context->minLifetime) <= context->deviceQueue->lastFenceCompleted)
        {
            ptr = context->pool_textures[idx].release();
            texture_destroy(context, ptr);

            context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Vulkan destroy pool texture %d", idx);

            context->pool_textures.erase(context->pool_textures.begin() + idx);
        }
    }
}

/// ***************************** TimerHeap *****************************************************

void profilerCapture_init(Context* context, ProfilerCapture* ptr, pnanovdb_uint64_t capacity)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkQueryPoolCreateInfo queryPoolInfo = {};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = (uint32_t)capacity;

    loader->vkCreateQueryPool(vulkanDevice, &queryPoolInfo, nullptr, &ptr->queryPool);

    VkBufferCreateInfo bufCreateInfo = {};
    bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCreateInfo.size = capacity * sizeof(pnanovdb_uint64_t);
    bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    loader->vkCreateBuffer(vulkanDevice, &bufCreateInfo, nullptr, &ptr->queryBuffer);

    VkMemoryRequirements bufMemReq = {};
    loader->vkGetBufferMemoryRequirements(vulkanDevice, ptr->queryBuffer, &bufMemReq);

    uint32_t bufMemType = context_getMemoryType(context, bufMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (bufMemType == ~0u)
    {
        bufMemType = context_getMemoryType(context, bufMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    VkMemoryAllocateInfo bufMemAllocInfo = {};
    bufMemAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufMemAllocInfo.allocationSize = bufMemReq.size;
    bufMemAllocInfo.memoryTypeIndex = bufMemType;

    loader->vkAllocateMemory(vulkanDevice, &bufMemAllocInfo, nullptr, &ptr->queryMemory);

    loader->vkBindBufferMemory(vulkanDevice, ptr->queryBuffer, ptr->queryMemory, 0u);

    loader->vkMapMemory(vulkanDevice, ptr->queryMemory, 0u, VK_WHOLE_SIZE, 0u, (void**)&ptr->queryMapped);

    ptr->capacity = capacity;

    ptr->entries.reserve(capacity);
    ptr->entries.resize(0u);
}

void profilerCapture_destroy(Context* context, ProfilerCapture* ptr)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    if (ptr->queryPool)
    {
        loader->vkDestroyQueryPool(loader->device, ptr->queryPool, nullptr);
        loader->vkDestroyBuffer(loader->device, ptr->queryBuffer, nullptr);
        loader->vkFreeMemory(loader->device, ptr->queryMemory, nullptr);
    }

    ptr->queryPool = nullptr;
    ptr->queryBuffer = nullptr;
    ptr->queryMemory = nullptr;
    ptr->queryMapped = nullptr;
    ptr->queryFrequency = 0u;
    ptr->queryReadbackFenceVal = ~0llu;

    ptr->state = 0u;
    ptr->captureID = 0llu;
    ptr->cpuFreq = 0llu;
    ptr->capacity = 0u;

    ptr->entries.resize(0u);
}

void profilerCapture_reset(Context* context, ProfilerCapture* ptr, pnanovdb_uint64_t minCapacity, pnanovdb_uint64_t captureID)
{
    if (ptr->state == 0u)
    {
        auto loader = &context->deviceQueue->device->loader;

        if (ptr->capacity == 0u || ptr->capacity < minCapacity)
        {
            profilerCapture_destroy(context, ptr);

            pnanovdb_uint64_t newCapacity = 128u;
            while (newCapacity < minCapacity)
            {
                newCapacity *= 2u;
            }

            profilerCapture_init(context, ptr, newCapacity);
        }

        ptr->queryFrequency = (pnanovdb_uint64_t)(double(1.0E9) / double(context->deviceQueue->device->physicalDeviceProperties.limits.timestampPeriod));
        loader->vkCmdResetQueryPool(context->deviceQueue->commandBuffer, ptr->queryPool, 0u, (uint32_t)ptr->capacity);

#if defined(_WIN32)
        LARGE_INTEGER tmpCpuFreq = {};
        QueryPerformanceFrequency(&tmpCpuFreq);
        ptr->cpuFreq = tmpCpuFreq.QuadPart;
#else
        ptr->cpuFreq = 1E9;
#endif
        ptr->entries.resize(0u);

        ptr->captureID = captureID;
        ptr->state = 1u;
    }
}

void profilerCapture_timestamp(Context* context, ProfilerCapture* ptr, const char* label)
{
    if (ptr->state == 1u && ptr->entries.size() < ptr->capacity)
    {
        auto loader = &context->deviceQueue->device->loader;

        uint32_t entry_idx = (uint32_t)ptr->entries.size();
        ptr->entries.push_back(ProfilerEntry());
        auto& entry = ptr->entries.back();

        entry.label = label;
        entry.cpuValue = 0llu;
        entry.gpuValue = 0llu;

        loader->vkCmdWriteTimestamp(context->deviceQueue->commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, ptr->queryPool, entry_idx);

#if defined(_WIN32)
        LARGE_INTEGER tmpCpuTime = {};
        QueryPerformanceCounter(&tmpCpuTime);
        entry.cpuValue = tmpCpuTime.QuadPart;
#else
        timespec timeValue = {};
        clock_gettime(CLOCK_MONOTONIC, &timeValue);
        entry.cpuValue = 1E9 * pnanovdb_uint64_t(timeValue.tv_sec) + pnanovdb_uint64_t(timeValue.tv_nsec);
#endif
    }
}

void profilerCapture_download(Context* context, ProfilerCapture* ptr)
{
    if (ptr->state == 1u)
    {
        auto loader = &context->deviceQueue->device->loader;

        loader->vkCmdCopyQueryPoolResults(context->deviceQueue->commandBuffer, ptr->queryPool, 0u, (uint32_t)ptr->entries.size(),
            ptr->queryBuffer, 0llu, sizeof(pnanovdb_uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        ptr->queryReadbackFenceVal = context->deviceQueue->nextFenceValue;

        ptr->state = 2u;
    }
}

pnanovdb_bool_t profilerCapture_mapResults(Context* context, ProfilerCapture* ptr, pnanovdb_uint64_t* pNumEntries, pnanovdb_compute_profiler_entry_t** pEntries)
{
    if (ptr->state == 2u)
    {
        if (ptr->queryReadbackFenceVal > context->deviceQueue->lastFenceCompleted)
        {
            return PNANOVDB_FALSE;
        }

        for (pnanovdb_uint32_t idx = 0u; idx < ptr->entries.size(); idx++)
        {
            ptr->entries[idx].gpuValue = ptr->queryMapped[idx];
        }

        ptr->deltaEntries.reserve(ptr->entries.size());
        ptr->deltaEntries.resize(ptr->entries.size());

        // compute time deltas
        ProfilerEntry prevEntry = {};
        if (ptr->entries.size() > 0u)
        {
            prevEntry = ptr->entries[0u];
        }
        for (pnanovdb_uint32_t idx = 0u; idx < ptr->entries.size(); idx++)
        {
            auto entry = &ptr->entries[idx];
            auto deltaEntry = &ptr->deltaEntries[idx];

            deltaEntry->label = entry->label;
            deltaEntry->cpu_delta_time = (float)(((double)(entry->cpuValue - prevEntry.cpuValue) / (double)(ptr->cpuFreq)));
            deltaEntry->gpu_delta_time = (float)(((double)(entry->gpuValue - prevEntry.gpuValue) / (double)(ptr->queryFrequency)));

            prevEntry = *entry;
        }

        // map results
        if (pNumEntries)
        {
            *pNumEntries = ptr->entries.size();
        }
        if (pEntries)
        {
            *pEntries = ptr->deltaEntries.data();
        }

        ptr->state = 3u;
        return PNANOVDB_TRUE;
    }
    return PNANOVDB_FALSE;
}

void profilerCapture_unmapResults(Context* context, ProfilerCapture* ptr)
{
    if (ptr->state == 3u)
    {
        ptr->state = 0u;
    }
}

/// ***************************** Profiler *****************************************************

Profiler* profiler_create(Context* context)
{
    auto ptr = new Profiler();

    return ptr;
}

void profiler_destroy(Context* context, Profiler* ptr)
{
    for (pnanovdb_uint32_t captureIndex = 0u; captureIndex < ptr->captures.size(); captureIndex++)
    {
        profilerCapture_destroy(context, &ptr->captures[captureIndex]);
    }
    ptr->captures.resize(0u);

    delete ptr;
}

void profiler_beginCapture(Context* context, Profiler* ptr, pnanovdb_uint64_t numEntries)
{
    if (!ptr->reportEntries)
    {
        return;
    }

    // account for implicit begin/end
    numEntries += 2u;

    pnanovdb_uint64_t captureIndex = 0u;
    for (; captureIndex < ptr->captures.size(); captureIndex++)
    {
        if (ptr->captures[captureIndex].state == 0u)
        {
            break;
        }
    }
    if (captureIndex == ptr->captures.size())
    {
        ptr->captures.push_back(ProfilerCapture());

        profilerCapture_init(context, &ptr->captures.back(), numEntries);
    }
    ptr->currentCaptureIndex = captureIndex;

    auto capture = &ptr->captures[captureIndex];

    ptr->currentCaptureID++;

    profilerCapture_reset(context, capture, numEntries, ptr->currentCaptureID);

    profilerCapture_timestamp(context, capture, "BeginCapture");
}

void profiler_endCapture(Context* context, Profiler* ptr)
{
    if (!ptr->reportEntries)
    {
        return;
    }

    if (ptr->currentCaptureIndex < ptr->captures.size())
    {
        auto capture = &ptr->captures[ptr->currentCaptureIndex];

        profilerCapture_timestamp(context, capture, "EndCapture");

        profilerCapture_download(context, capture);
    }
}

void profiler_processCaptures(Context* context, Profiler* ptr)
{
    if (!ptr->reportEntries)
    {
        return;
    }

    for (pnanovdb_uint64_t captureIndex = 0u; captureIndex < ptr->captures.size(); captureIndex++)
    {
        auto capture = &ptr->captures[captureIndex];

        pnanovdb_uint64_t numEntries = 0u;
        pnanovdb_compute_profiler_entry_t* entries = nullptr;
        if (profilerCapture_mapResults(context, capture, &numEntries, &entries))
        {
            ptr->reportEntries(ptr->userdata, capture->captureID, (pnanovdb_uint32_t)numEntries, entries);

            profilerCapture_unmapResults(context, capture);
        }
    }
}

void profiler_timestamp(Context* context, Profiler* ptr, const char* label)
{
    if (!ptr->reportEntries)
    {
        return;
    }

    if (ptr->currentCaptureIndex < ptr->captures.size())
    {
        auto capture = &ptr->captures[ptr->currentCaptureIndex];

        profilerCapture_timestamp(context, capture, label);
    }
}

void enableProfiler(pnanovdb_compute_context_t* contextIn, void* userdata, void(PNANOVDB_ABI* reportEntries)(void* userdata, pnanovdb_uint64_t captureID, pnanovdb_uint32_t numEntries, pnanovdb_compute_profiler_entry_t* entries))
{
    auto context = cast(contextIn);

    context->profiler->userdata = userdata;
    context->profiler->reportEntries = reportEntries;
}

void disableProfiler(pnanovdb_compute_context_t* contextIn)
{
    auto context = cast(contextIn);

    context->profiler->reportEntries = nullptr;
}

void setResourceMinLifetime(pnanovdb_compute_context_t* context, pnanovdb_uint64_t minLifetime)
{
    auto ctx = cast(context);
    ctx->minLifetime = minLifetime;
}

} // end namespace
