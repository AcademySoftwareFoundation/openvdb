
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   TextureVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include "CommonVulkan.h"

namespace pnanovdb_vulkan
{

void texture_descClamping(Texture* ptr)
{
    if (ptr->desc.mip_levels == 0u)
    {
        ptr->desc.mip_levels = 1u;
    }

    if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_1D)
    {
        ptr->desc.height = 1u;
        ptr->desc.depth = 1u;
    }
    else if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_2D)
    {
        ptr->desc.depth = 1u;
    }
}

void texture_createImageView(Context* context, Texture* ptr, VkImageView* view_all, VkImageView* view_mipLevel)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = ptr->imageVk;
    if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_1D)
    {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
    }
    else if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_2D)
    {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    }
    else //if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_3D)
    {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    }
    viewInfo.format = formatConverter_convertToVulkan(context->deviceQueue->device->formatConverter, ptr->desc.format);
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    viewInfo.subresourceRange.aspectMask = ptr->imageAspect;
    viewInfo.subresourceRange.levelCount = ptr->desc.mip_levels;
    viewInfo.subresourceRange.layerCount = 1u;

    // for views with depth and stencil, default to depth view
    if (viewInfo.subresourceRange.aspectMask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
    {
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    loader->vkCreateImageView(vulkanDevice, &viewInfo, nullptr, view_all);

    // single mipLevel default for write
    viewInfo.subresourceRange.baseMipLevel = 0u;
    viewInfo.subresourceRange.levelCount = 1u;

    loader->vkCreateImageView(vulkanDevice, &viewInfo, nullptr, view_mipLevel);
}

void texture_pushImageView(Context* context, Texture* ptr, pnanovdb_compute_format_t format)
{
    VkImageView imageViewVk_all = VK_NULL_HANDLE;
    VkImageView imageViewVk_mipLevel = VK_NULL_HANDLE;
    texture_createImageView(context, ptr, &imageViewVk_all, &imageViewVk_mipLevel);

    ptr->aliasFormats.push_back(format);
    ptr->aliasImageViewAlls.push_back(imageViewVk_all);
    ptr->aliasImageViewMipLevels.push_back(imageViewVk_mipLevel);
}

VkImageView texture_getImageViewAll(Context* context, Texture* ptr, pnanovdb_compute_format_t aliasFormat)
{
    if (aliasFormat == PNANOVDB_COMPUTE_FORMAT_UNKNOWN)
    {
        return ptr->imageViewVk_all;
    }
    for (pnanovdb_uint64_t idx = 0u; idx < ptr->aliasFormats.size(); idx++)
    {
        if (ptr->aliasFormats[idx] == aliasFormat)
        {
            return ptr->aliasImageViewAlls[idx];
        }
    }
    texture_pushImageView(context, ptr, aliasFormat);
    return texture_getImageViewAll(context, ptr, aliasFormat);
}

VkImageView texture_getImageViewMipLevel(Context* context, Texture* ptr, pnanovdb_compute_format_t aliasFormat)
{
    if (aliasFormat == PNANOVDB_COMPUTE_FORMAT_UNKNOWN)
    {
        return ptr->imageViewVk_mipLevel;
    }
    for (pnanovdb_uint64_t idx = 0u; idx < ptr->aliasFormats.size(); idx++)
    {
        if (ptr->aliasFormats[idx] == aliasFormat)
        {
            return ptr->aliasImageViewMipLevels[idx];
        }
    }
    texture_pushImageView(context, ptr, aliasFormat);
    return texture_getImageViewMipLevel(context, ptr, aliasFormat);
}

void texture_createImage(Context* context, Texture* ptr, VkImageUsageFlags usage)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkImageCreateInfo texCreateInfo = {};
    texCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_1D)
    {
        texCreateInfo.imageType = VK_IMAGE_TYPE_1D;
    }
    else if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_2D)
    {
        texCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    }
    else //if (ptr->desc.texture_type == PNANOVDB_COMPUTE_TEXTURE_TYPE_3D)
    {
        texCreateInfo.imageType = VK_IMAGE_TYPE_3D;
    }
    texCreateInfo.format = formatConverter_convertToVulkan(context->deviceQueue->device->formatConverter, ptr->desc.format);
    texCreateInfo.extent.width = ptr->desc.width;
    texCreateInfo.extent.height = ptr->desc.height;
    texCreateInfo.extent.depth = ptr->desc.depth;
    texCreateInfo.mipLevels = ptr->desc.mip_levels;
    texCreateInfo.arrayLayers = 1u;
    texCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    texCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    texCreateInfo.usage = usage;
    texCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    texCreateInfo.queueFamilyIndexCount = VK_QUEUE_FAMILY_IGNORED;
    texCreateInfo.pQueueFamilyIndices = nullptr;
    texCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    loader->vkCreateImage(vulkanDevice, &texCreateInfo, nullptr, &ptr->imageVk);

    VkMemoryRequirements texMemReq = {};
    loader->vkGetImageMemoryRequirements(vulkanDevice, ptr->imageVk, &texMemReq);

    uint32_t texMemType_device = context_getMemoryType(context, texMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    uint32_t texMemType_sysmem = context_getMemoryType(context, texMemReq.memoryTypeBits, 0);

    ptr->allocationBytes = texMemReq.size;
    device_reportMemoryAllocate(context->deviceQueue->device, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, ptr->allocationBytes);

    VkMemoryAllocateInfo texMemAllocInfo = {};
    texMemAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    texMemAllocInfo.allocationSize = texMemReq.size;
    texMemAllocInfo.memoryTypeIndex = texMemType_device;

    VkResult result = loader->vkAllocateMemory(vulkanDevice, &texMemAllocInfo, nullptr, &ptr->memoryVk);
    if (result == VK_SUCCESS)
    {
        context->deviceQueue->device->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Texture vidmem allocate %lld bytes", texMemReq.size);
    }
    else
    {
        texMemAllocInfo.memoryTypeIndex = texMemType_sysmem;
        result = loader->vkAllocateMemory(vulkanDevice, &texMemAllocInfo, nullptr, &ptr->memoryVk);
        if (result == VK_SUCCESS)
        {
            context->deviceQueue->device->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Texture sysmem allocate %lld bytes", texMemReq.size);
        }
        else
        {
            context->deviceQueue->device->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Texture allocate failed %lld bytes", texMemReq.size);
        }
    }

    loader->vkBindImageMemory(vulkanDevice, ptr->imageVk, ptr->memoryVk, 0u);
}

void texture_initRestoreBarrier(Context* context, Texture* ptr)
{
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = ptr->imageVk;
    barrier.subresourceRange.aspectMask = ptr->imageAspect;
    barrier.subresourceRange.baseMipLevel = 0u;
    barrier.subresourceRange.levelCount = ptr->desc.mip_levels;
    barrier.subresourceRange.baseArrayLayer = 0u;
    barrier.subresourceRange.layerCount = 1u;

    ptr->restoreBarrier = barrier;
}

void texture_initCurrentBarrier(Context* context, Texture* ptr)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkImageMemoryBarrier barrier = ptr->restoreBarrier;
    barrier.srcAccessMask = 0u;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    barrier.dstAccessMask = 0u;
    barrier.newLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    ptr->currentBarrier = barrier;
}

Texture* texture_create(Context* context, const pnanovdb_compute_texture_desc_t* desc)
{
    auto ptr = new Texture();

    ptr->desc = *desc;
    ptr->imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;

    texture_descClamping(ptr);

    VkImageUsageFlags usage = 0u;
    if (ptr->desc.usage & PNANOVDB_COMPUTE_TEXTURE_USAGE_RW_TEXTURE)
    {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_TEXTURE_USAGE_TEXTURE)
    {
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_TEXTURE_USAGE_COPY_SRC)
    {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_TEXTURE_USAGE_COPY_DST)
    {
        usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    texture_createImage(context, ptr, usage);
    texture_createImageView(context, ptr, &ptr->imageViewVk_all, &ptr->imageViewVk_mipLevel);

    //texture_computeSubresources(context, ptr);
    texture_initRestoreBarrier(context, ptr);

    texture_initCurrentBarrier(context, ptr);

    return ptr;
}

Texture* texture_createExternal(Context* context, const pnanovdb_compute_texture_desc_t* desc, VkImage externalImage, VkImageLayout defaultLayout)
{
    auto ptr = new Texture();

    ptr->desc = *desc;
    ptr->imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;

    texture_descClamping(ptr);

    ptr->memoryVk = VK_NULL_HANDLE;
    ptr->imageVk = externalImage;
    texture_createImageView(context, ptr, &ptr->imageViewVk_all, &ptr->imageViewVk_mipLevel);

    //texture_computeSubresources(context, ptr);
    texture_initRestoreBarrier(context, ptr);

    ptr->restoreBarrier.oldLayout = defaultLayout;
    ptr->restoreBarrier.newLayout = defaultLayout;

    texture_initCurrentBarrier(context, ptr);

    return ptr;
}

void texture_destroy(Context* context, Texture* ptr)
{
    auto loader = &context->deviceQueue->device->loader;

    loader->vkDestroyImageView(loader->device, ptr->imageViewVk_mipLevel, nullptr);
    loader->vkDestroyImageView(loader->device, ptr->imageViewVk_all, nullptr);
    for (pnanovdb_uint64_t idx = 0u; idx < ptr->aliasImageViewAlls.size(); idx++)
    {
        loader->vkDestroyImageView(loader->device, ptr->aliasImageViewAlls[idx], nullptr);
    }
    for (pnanovdb_uint64_t idx = 0u; idx < ptr->aliasImageViewMipLevels.size(); idx++)
    {
        loader->vkDestroyImageView(loader->device, ptr->aliasImageViewMipLevels[idx], nullptr);
    }

    // use memoryVk as indicator of ownership of original image
    if (ptr->memoryVk)
    {
        loader->vkDestroyImage(loader->device, ptr->imageVk, nullptr);
        loader->vkFreeMemory(loader->device, ptr->memoryVk, nullptr);

        device_reportMemoryFree(context->deviceQueue->device, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, ptr->allocationBytes);
    }

    delete ptr;
}

pnanovdb_bool_t textureDesc_compare(const pnanovdb_compute_texture_desc_t* a, const pnanovdb_compute_texture_desc_t* b)
{
    if (a->texture_type == b->texture_type &&
        a->usage == b->usage &&
        a->format == b->format &&
        a->width == b->width &&
        a->height == b->height &&
        a->depth == b->depth &&
        a->mip_levels == b->mip_levels)
    {
        return PNANOVDB_TRUE;
    }
    return PNANOVDB_FALSE;
}

pnanovdb_compute_texture_t* createTexture(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_texture_desc_t* desc)
{
    auto context = cast(contextIn);

    for (pnanovdb_uint32_t idx = 0u; idx < context->pool_textures.size(); idx++)
    {
        auto ptr = context->pool_textures[idx].get();
        if (ptr && ptr->refCount == 0 && textureDesc_compare(&ptr->desc, desc))
        {
            ptr->refCount = 1;
            ptr->lastActive = context->deviceQueue->nextFenceValue;
            return cast(ptr);
        }
    }

    auto ptr = texture_create(context, desc);

    ptr->refCount = 1;
    ptr->lastActive = context->deviceQueue->nextFenceValue;
    context->pool_textures.push_back(std::unique_ptr<Texture>(ptr));

    return cast(ptr);
}

pnanovdb_compute_texture_t* createTextureExternal(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_texture_desc_t* desc, VkImage externalImage, VkImageLayout defaultLayout)
{
    auto context = cast(contextIn);

    auto ptr = texture_createExternal(context, desc, externalImage, defaultLayout);

    ptr->refCount = 1;
    ptr->lastActive = context->deviceQueue->nextFenceValue;
    context->pool_textures.push_back(std::unique_ptr<Texture>(ptr));

    return cast(ptr);
}

void destroyTexture(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_texture_t* texture)
{
    auto context = cast(contextIn);
    auto ptr = cast(texture);

    if (ptr->refCount <= 0)
    {
        context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "pnanovdb_compute_context_t::destroyTexture() called on inactive texture %p", texture);
    }

    // defer all except external
    if (ptr->memoryVk)
    {
        // defer release to end of frame for now
        // resource getting created/destroyed within single frame were being too eagerly recycled
        context->deferredReleaseTextures.push_back(texture);
    }
    else // if external, actually destroy
    {
        ptr->refCount--;
        ptr->lastActive = context->deviceQueue->nextFenceValue;
    }

    // if external, actually destroy
    if (!ptr->memoryVk)
    {
        // remove from pool
        Texture* destroy_texture = nullptr;
        for (pnanovdb_uint64_t idx = 0u; idx < context->pool_textures.size(); idx++)
        {
            if (context->pool_textures[idx].get() == ptr)
            {
                destroy_texture = context->pool_textures[idx].release();
                context->pool_textures.erase(context->pool_textures.begin() + idx);
                break;
            }
        }
        if (destroy_texture != ptr)
        {
            context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "pnanovdb_compute_context_t::destroyTexture() texture %p not found in pool %p",
                ptr, destroy_texture);
        }
        if (destroy_texture)
        {
            texture_destroy(context, destroy_texture);
        }
    }
}

void context_destroyTextures(Context* context)
{
    context->textureTransients.clear();
    context->textureAcquires.clear();

    for (pnanovdb_uint32_t idx = 0u; idx < context->pool_textures.size(); idx++)
    {
        auto ptr = context->pool_textures[idx].release();
        texture_destroy(context, ptr);
    }
}

pnanovdb_compute_texture_transient_t* getTextureTransient(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_texture_desc_t* desc)
{
    auto context = cast(contextIn);
    auto ptr = new TextureTransient();
    context->textureTransients.push_back(std::unique_ptr<TextureTransient>(ptr));
    ptr->desc = *desc;
    ptr->texture = nullptr;
    ptr->aliasTexture = nullptr;
    ptr->aliasFormat = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    ptr->nodeBegin = 0;
    ptr->nodeEnd = 0;
    return cast(ptr);
}

pnanovdb_compute_texture_transient_t* registerTextureAsTransient(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_texture_t* texture)
{
    auto context = cast(contextIn);
    auto ptr = new TextureTransient();
    context->textureTransients.push_back(std::unique_ptr<TextureTransient>(ptr));
    ptr->desc = cast(texture)->desc;
    ptr->texture = cast(texture);
    ptr->aliasTexture = nullptr;
    ptr->aliasFormat = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    ptr->nodeBegin = 0;
    ptr->nodeEnd = 0;
    return cast(ptr);
}

pnanovdb_compute_texture_transient_t* aliasTextureTransient(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_texture_transient_t* texture, pnanovdb_compute_format_t format)
{
    auto context = cast(contextIn);
    auto ptr = new TextureTransient();
    context->textureTransients.push_back(std::unique_ptr<TextureTransient>(ptr));
    ptr->desc = cast(texture)->desc;
    ptr->texture = nullptr;
    ptr->aliasTexture = cast(texture);
    ptr->aliasFormat = format;
    ptr->nodeBegin = 0;
    ptr->nodeEnd = 0;
    return cast(ptr);
}

pnanovdb_compute_texture_acquire_t* enqueueAcquireTexture(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_texture_transient_t* textureTransient)
{
    auto context = cast(contextIn);
    auto ptr = new TextureAcquire();
    context->textureAcquires.push_back(std::unique_ptr<TextureAcquire>(ptr));
    ptr->textureTransient = cast(textureTransient);
    ptr->texture = nullptr;
    return cast(ptr);
}

pnanovdb_bool_t getAcquiredTexture(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_texture_acquire_t* acquire, pnanovdb_compute_texture_t** outTexture)
{
    auto context = cast(contextIn);
    auto ptr = cast(acquire);
    if (ptr->texture)
    {
        *outTexture = cast(ptr->texture);

        // remove from acquire array
        for (pnanovdb_uint64_t idx = 0u; idx < context->textureAcquires.size(); idx++)
        {
            if (context->textureAcquires[idx].get() == ptr)
            {
                context->textureAcquires.erase(context->textureAcquires.begin() + idx);
                break;
            }
        }

        return PNANOVDB_TRUE;
    }
    return PNANOVDB_FALSE;
}

/// ***************************** Samplers ********************************************

VkSamplerAddressMode sampler_convertAddressMode(pnanovdb_compute_sampler_address_mode_t addressMode)
{
    switch (addressMode)
    {
    case PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_WRAP:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_CLAMP:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_MIRROR:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case PNANOVDB_COMPUTE_SAMPLER_ADDRESS_MODE_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    }
}

VkFilter sampler_convertFilter(pnanovdb_compute_sampler_filter_mode_t filter)
{
    switch (filter)
    {
    case PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_POINT:
        return VK_FILTER_NEAREST;
    case PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_LINEAR:
        return VK_FILTER_LINEAR;
    default:
        return VK_FILTER_NEAREST;
    }
}

VkSamplerMipmapMode sampler_convertMipmapMode(pnanovdb_compute_sampler_filter_mode_t filter)
{
    switch (filter)
    {
    case PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_POINT:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case PNANOVDB_COMPUTE_SAMPLER_FILTER_MODE_LINEAR:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    default:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    }
}

Sampler* sampler_create(Context* context, const pnanovdb_compute_sampler_desc_t* desc)
{
    auto loader = &context->deviceQueue->device->loader;
    auto ptr = new Sampler();

    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = sampler_convertFilter(desc->filter_mode);
    samplerCreateInfo.minFilter = sampler_convertFilter(desc->filter_mode);
    samplerCreateInfo.mipmapMode = sampler_convertMipmapMode(desc->filter_mode);
    samplerCreateInfo.addressModeU = sampler_convertAddressMode(desc->address_mode_u);
    samplerCreateInfo.addressModeV = sampler_convertAddressMode(desc->address_mode_v);
    samplerCreateInfo.addressModeW = sampler_convertAddressMode(desc->address_mode_w);
    samplerCreateInfo.minLod = 0.f;
    samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerCreateInfo.maxAnisotropy = 1.f;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;

    loader->vkCreateSampler(context->deviceQueue->vulkanDevice, &samplerCreateInfo, nullptr, &ptr->sampler);

    return ptr;
}

void sampler_destroy(Context* context, Sampler* ptr)
{
    auto loader = &context->deviceQueue->device->loader;

    loader->vkDestroySampler(loader->device, ptr->sampler, nullptr);

    delete ptr;
}

pnanovdb_compute_sampler_t* createSampler(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_sampler_desc_t* desc)
{
    auto context = cast(contextIn);
    auto ptr = sampler_create(context, desc);

    ptr->isActive = PNANOVDB_TRUE;
    ptr->lastActive = context->deviceQueue->nextFenceValue;
    context->pool_samplers.push_back(std::unique_ptr<Sampler>(ptr));

    return cast(ptr);
}

pnanovdb_compute_sampler_t* getDefaultSampler(pnanovdb_compute_context_t* contextIn)
{
    auto context = cast(contextIn);
    return cast(context->pool_samplers[0u].get());
}

void destroySampler(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_sampler_t* sampler)
{
    auto context = cast(contextIn);
    auto ptr = cast(sampler);

    ptr->isActive = PNANOVDB_FALSE;
    ptr->lastActive = context->deviceQueue->nextFenceValue;
}

void context_destroySamplers(Context* context)
{
    for (pnanovdb_uint32_t idx = 0u; idx < context->pool_samplers.size(); idx++)
    {
        auto ptr = context->pool_samplers[idx].release();
        sampler_destroy(context, ptr);
    }
}

} // end namespace
