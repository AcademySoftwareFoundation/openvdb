
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ComputePipelineVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include "CommonVulkan.h"

namespace pnanovdb_vulkan
{

pnanovdb_uint32_t computePipeline_computeNumSets(pnanovdb_uint32_t descriptorsPerSet)
{
    static const pnanovdb_uint32_t targetCount = 4096u; // 65536u;
    pnanovdb_uint32_t numSets = 1u;
    if (descriptorsPerSet > 0)
    {
        numSets = targetCount / (descriptorsPerSet);
    }
    if (numSets == 0u)
    {
        numSets = 1u;
    }
    return numSets;
}

pnanovdb_compute_pipeline_t* createComputePipeline(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_pipeline_desc_t* desc)
{
    auto context = cast(contextIn);
    auto ptr = new ComputePipeline();

    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    ptr->desc = *desc;

    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_CONSTANT_BUFFER] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_STRUCTURED_BUFFER] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER] = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE] = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_SAMPLER] = VK_DESCRIPTOR_TYPE_SAMPLER;
    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_STRUCTURED_BUFFER] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_BUFFER] = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    ptr->pnanovdbDescriptorType_to_vkDescriptorType[PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE] = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

    // process bytecode
    if (ptr->desc.bytecode.size_in_bytes > 0u)
    {
        VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.codeSize = ptr->desc.bytecode.size_in_bytes;
        shaderModuleCreateInfo.pCode = (const uint32_t*)ptr->desc.bytecode.data;
        loader->vkCreateShaderModule(vulkanDevice, &shaderModuleCreateInfo, nullptr, &ptr->module);
    }
    else
    {
        ptr->module = nullptr;
    }

    // create descriptor layout
    {
        ptr->bindings.resize(0u);
        for (pnanovdb_uint32_t idx = 0u; idx < desc->binding_desc_count; idx++)
        {
            auto binding_desc = &desc->binding_descs[idx];

            VkDescriptorSetLayoutBinding binding = {};
            binding.binding = binding_desc->binding_desc.vulkan.binding;
            binding.descriptorType = ptr->pnanovdbDescriptorType_to_vkDescriptorType[binding_desc->type];
            binding.descriptorCount = binding_desc->binding_desc.vulkan.descriptor_count;
            binding.stageFlags = VK_SHADER_STAGE_ALL;
            binding.pImmutableSamplers = nullptr;

            // count descriptors
            ptr->descriptorCounts[binding_desc->type]++;
            ptr->totalDescriptors += binding_desc->binding_desc.vulkan.descriptor_count;

            ptr->bindings.push_back(binding);
        }

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {};
        descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutInfo.bindingCount = (uint32_t)ptr->bindings.size();
        descriptorSetLayoutInfo.pBindings = ptr->bindings.data();

        loader->vkCreateDescriptorSetLayout(vulkanDevice, &descriptorSetLayoutInfo, nullptr, &ptr->descriptorSetLayout);

        // precompute desriptor count for pool allocation
        {
            ptr->setsPerPool = computePipeline_computeNumSets(ptr->totalDescriptors);

            auto pushPool = [&](pnanovdb_compute_descriptor_type_t type)
            {
                if (ptr->descriptorCounts[type] > 0u)
                {
                    VkDescriptorType vkType = ptr->pnanovdbDescriptorType_to_vkDescriptorType[type];
                    pnanovdb_uint32_t descriptorCount = ptr->setsPerPool * ptr->descriptorCounts[type];
                    // check for match
                    pnanovdb_uint32_t idx = 0u;
                    for (; idx < ptr->poolSizeCount; idx++)
                    {
                        if (ptr->poolSizes[idx].type == vkType)
                        {
                            ptr->poolSizes[idx].descriptorCount += descriptorCount;
                            break;
                        }
                    }
                    if (idx == ptr->poolSizeCount)
                    {
                        ptr->poolSizes[ptr->poolSizeCount].type = vkType;
                        ptr->poolSizes[ptr->poolSizeCount].descriptorCount = descriptorCount;
                        ptr->poolSizeCount++;
                    }
                }
            };

            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_CONSTANT_BUFFER);
            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_STRUCTURED_BUFFER);
            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER);
            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE);
            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_SAMPLER);
            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_STRUCTURED_BUFFER);
            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_BUFFER);
            pushPool(PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE);
        }
    }

    // create pipeline layout
    {
        VkDescriptorSetLayout layouts[1u] = {
            ptr->descriptorSetLayout
        };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1u;
        pipelineLayoutInfo.pSetLayouts = layouts;
        pipelineLayoutInfo.pushConstantRangeCount = 0u;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        loader->vkCreatePipelineLayout(vulkanDevice, &pipelineLayoutInfo, nullptr, &ptr->pipelineLayout);
    }

    // create pipeline
    {
        VkPipelineShaderStageCreateInfo stage = {};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.pNext = nullptr;
        stage.flags = 0u;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.pName = "main";
        stage.pSpecializationInfo = nullptr;

        VkComputePipelineCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;

        stage.module = ptr->module;
        createInfo.stage = stage;
        createInfo.layout = ptr->pipelineLayout;

        loader->vkCreateComputePipelines(
            vulkanDevice,
            VK_NULL_HANDLE,
            1u,
            &createInfo,
            nullptr,
            &ptr->pipeline
        );
    }

    return cast(ptr);
}

void destroyComputePipeline(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_pipeline_t* pipeline)
{
    if (!pipeline)
    {
        return;
    }

    auto context = cast(contextIn);
    auto ptr = cast(pipeline);
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    for (pnanovdb_uint32_t idx = 0u; idx < ptr->pools.size(); idx++)
    {
        if (ptr->pools[idx].pool)
        {
            loader->vkDestroyDescriptorPool(loader->device, ptr->pools[idx].pool, nullptr);
            ptr->pools[idx].pool = VK_NULL_HANDLE;
        }
    }

    loader->vkDestroyDescriptorSetLayout(loader->device, ptr->descriptorSetLayout, nullptr);
    loader->vkDestroyPipelineLayout(loader->device, ptr->pipelineLayout, nullptr);
    loader->vkDestroyPipeline(loader->device, ptr->pipeline, nullptr);

    loader->vkDestroyShaderModule(loader->device, ptr->module, nullptr);

    delete ptr;
}

VkDescriptorSet computePipeline_allocate(Context* context, ComputePipeline* ptr)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    // allocate descriptor set
    VkDescriptorSetAllocateInfo descriptorAllocInfo = {};
    descriptorAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorAllocInfo.descriptorPool = VK_NULL_HANDLE;
    descriptorAllocInfo.descriptorSetCount = 1u;
    descriptorAllocInfo.pSetLayouts = &ptr->descriptorSetLayout;

    if (ptr->frontIdx < ptr->pools.size())
    {
        auto& pool = ptr->pools[ptr->frontIdx];
        if (pool.allocSetIdx < ptr->setsPerPool && pool.fenceValue == context->deviceQueue->nextFenceValue)
        {
            pool.allocSetIdx++;
            descriptorAllocInfo.descriptorPool = pool.pool;
            auto ret = loader->vkAllocateDescriptorSets(vulkanDevice, &descriptorAllocInfo, &descriptorSet);
            if (ret < 0)
            {
                descriptorSet = VK_NULL_HANDLE;
            }
        }
    }
    if (descriptorSet == VK_NULL_HANDLE)
    {
        pnanovdb_uint64_t poolIdx = 0u;
        for (; poolIdx < ptr->pools.size(); poolIdx++)
        {
            auto& pool = ptr->pools[poolIdx];
            if (pool.fenceValue <= context->deviceQueue->lastFenceCompleted)
            {
                loader->vkResetDescriptorPool(vulkanDevice, pool.pool, 0u);

                pool.allocSetIdx = 0u;
                pool.fenceValue = context->deviceQueue->nextFenceValue;

                pool.allocSetIdx++;
                descriptorAllocInfo.descriptorPool = pool.pool;
                loader->vkAllocateDescriptorSets(vulkanDevice, &descriptorAllocInfo, &descriptorSet);

                break;
            }
        }
        if (poolIdx == ptr->pools.size())
        {
            ptr->pools.push_back(DescriptorPool());
            auto& pool = ptr->pools.back();

            VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
            descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolInfo.flags = 0u;
            descriptorPoolInfo.maxSets = ptr->setsPerPool;
            descriptorPoolInfo.poolSizeCount = ptr->poolSizeCount;
            descriptorPoolInfo.pPoolSizes = ptr->poolSizes;

            loader->vkCreateDescriptorPool(vulkanDevice, &descriptorPoolInfo, nullptr, &pool.pool);

            pool.allocSetIdx = 0u;
            pool.fenceValue = context->deviceQueue->nextFenceValue;

            pool.allocSetIdx++;
            descriptorAllocInfo.descriptorPool = pool.pool;
            loader->vkAllocateDescriptorSets(vulkanDevice, &descriptorAllocInfo, &descriptorSet);
        }
        ptr->frontIdx = poolIdx;
    }
    return descriptorSet;
}

void computePipeline_updateDescriptorSet(Context* context, ComputePipeline* ptr, VkDescriptorSet descriptorSet, const pnanovdb_compute_dispatch_params_t* params)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    ptr->descriptorWrites.reserve(params->descriptor_write_count);
    ptr->descriptorWrites.resize(0u);

    ptr->bufferInfos.reserve(params->descriptor_write_count);
    ptr->bufferViews.reserve(params->descriptor_write_count);
    ptr->imageInfos.reserve(params->descriptor_write_count);

    ptr->bufferInfos.resize(0u);
    ptr->bufferViews.resize(0u);
    ptr->imageInfos.resize(0u);
    for (pnanovdb_uint32_t idx = 0u; idx < params->descriptor_write_count; idx++)
    {
        auto descriptorWrite = &params->descriptor_writes[idx];
        auto resource = &params->resources[idx];

        VkWriteDescriptorSet output = {};
        output.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        output.pNext = nullptr;
        output.dstSet = descriptorSet;
        output.dstBinding = descriptorWrite->write.vulkan.binding;
        output.dstArrayElement = descriptorWrite->write.vulkan.array_index;
        output.descriptorCount = 1u;
        output.descriptorType = ptr->pnanovdbDescriptorType_to_vkDescriptorType[descriptorWrite->type];

        if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_CONSTANT_BUFFER ||
            descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_STRUCTURED_BUFFER ||
            descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_STRUCTURED_BUFFER)
        {
            Buffer* buffer = cast(resource->buffer_transient)->buffer;

            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer = buffer->bufferVk;
            bufferInfo.offset = 0llu;
            bufferInfo.range = VK_WHOLE_SIZE;

            output.pBufferInfo = ptr->bufferInfos.data() + ptr->bufferInfos.size();
            ptr->bufferInfos.push_back(bufferInfo);
        }
        else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_BUFFER ||
            descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_BUFFER)
        {
            Buffer* buffer = cast(resource->buffer_transient)->buffer;
            VkBufferView bufferView = buffer_getBufferView(
                context,
                buffer,
                cast(resource->buffer_transient)->aliasFormat
            );
            output.pTexelBufferView = ptr->bufferViews.data() + ptr->bufferViews.size();
            ptr->bufferViews.push_back(bufferView);
        }
        else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE ||
            descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_SAMPLER ||
            descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE)
        {
            VkDescriptorImageInfo imageInfo = {};
            if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE)
            {
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            else
            {
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_SAMPLER)
            {
                Sampler* sampler = cast(resource->sampler);
                imageInfo.sampler = sampler->sampler;
            }
            else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_TEXTURE)
            {
                Texture* texture = cast(resource->texture_transient)->texture;
                imageInfo.imageView = texture_getImageViewAll(
                    context,
                    texture,
                    cast(resource->texture_transient)->aliasFormat
                );
            }
            else if (descriptorWrite->type == PNANOVDB_COMPUTE_DESCRIPTOR_TYPE_RW_TEXTURE)
            {
                Texture* texture = cast(resource->texture_transient)->texture;
                imageInfo.imageView = texture_getImageViewMipLevel(
                    context,
                    texture,
                    cast(resource->texture_transient)->aliasFormat
                );
            }

            output.pImageInfo = ptr->imageInfos.data() + ptr->imageInfos.size();
            ptr->imageInfos.push_back(imageInfo);
        }

        ptr->descriptorWrites.push_back(output);
    }

    loader->vkUpdateDescriptorSets(vulkanDevice, (uint32_t)ptr->descriptorWrites.size(), ptr->descriptorWrites.data(), 0u, nullptr);

}

void computePipeline_dispatch(Context* context, const pnanovdb_compute_dispatch_params_t* params)
{
    ComputePipeline* ptr = cast(params->pipeline);

    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    pnanovdb_uint32_t grid_dim_x = params->grid_dim_x;
    pnanovdb_uint32_t grid_dim_y = params->grid_dim_y;
    pnanovdb_uint32_t grid_dim_z = params->grid_dim_z;

    VkDescriptorSet descriptorSet = computePipeline_allocate(context, ptr);

    computePipeline_updateDescriptorSet(context, ptr, descriptorSet, params);

    loader->vkCmdBindDescriptorSets(context->deviceQueue->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        ptr->pipelineLayout, 0u, 1u, &descriptorSet, 0u, nullptr);

    loader->vkCmdBindPipeline(context->deviceQueue->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ptr->pipeline);

    if (grid_dim_x > 65535 || grid_dim_y > 65535 || grid_dim_z > 65535)
    {
        context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_WARNING, "Dispatch(%s) gridDim of (%d, %d, %d) exceeds maximum gridDim of 65535.",
            params->debug_label ? params->debug_label : "invalid",
            grid_dim_x, grid_dim_y, grid_dim_z);
    }

    if (grid_dim_x > 0 && grid_dim_y > 0 && grid_dim_z > 0)
    {
        loader->vkCmdDispatch(context->deviceQueue->commandBuffer, grid_dim_x, grid_dim_y, grid_dim_z);
    }
}

} // end namespace
