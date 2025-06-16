
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   BufferVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include "CommonVulkan.h"

#if defined(_WIN32)

#else
#include <unistd.h>
#include <string.h>
#endif

namespace pnanovdb_vulkan
{

void buffer_createBuffer(Context* context, Buffer* ptr, const pnanovdb_compute_interop_handle_t* interopHandle)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkBufferCreateInfo bufCreateInfo = {};
    bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCreateInfo.size = ptr->desc.size_in_bytes;
    bufCreateInfo.usage = 0u;
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_BUFFER)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_RW_BUFFER)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_INDIRECT)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST)
    {
        bufCreateInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    bufCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfoKHR externalMemoryBufferCreateInfo = {};
    if (context->deviceQueue->device->desc.enable_external_usage && ptr->memory_type == PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE)
    {
        externalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR;
#if defined(_WIN32)
        externalMemoryBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
        externalMemoryBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        bufCreateInfo.pNext = &externalMemoryBufferCreateInfo;
    }

    loader->vkCreateBuffer(vulkanDevice, &bufCreateInfo, nullptr, &ptr->bufferVk);

    VkMemoryRequirements bufMemReq = {};
    loader->vkGetBufferMemoryRequirements(vulkanDevice, ptr->bufferVk, &bufMemReq);

    uint32_t bufMemType = 0u;
    uint32_t bufMemType_sysmem = 0u;
    if (ptr->memory_type == PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD)
    {
        bufMemType = context_getMemoryType(context, bufMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    else if (ptr->memory_type == PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK)
    {
        bufMemType = context_getMemoryType(context, bufMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (bufMemType == ~0u)
        {
            bufMemType = context_getMemoryType(context, bufMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    }
    else // (ptr->memory_type == PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE)
    {
        bufMemType = context_getMemoryType(context, bufMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        bufMemType_sysmem = context_getMemoryType(context, bufMemReq.memoryTypeBits, 0);
    }

    VkMemoryAllocateInfo bufMemAllocInfo = {};
    bufMemAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufMemAllocInfo.allocationSize = bufMemReq.size;
    bufMemAllocInfo.memoryTypeIndex = bufMemType;

    VkExportMemoryAllocateInfoKHR exportAllocInfo = {};
    if (!interopHandle && context->deviceQueue->device->desc.enable_external_usage && ptr->memory_type == PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE)
    {
        exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#if defined(_WIN32)
        exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
        exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        bufMemAllocInfo.pNext = &exportAllocInfo;
    }

#if defined(_WIN32)
    VkImportMemoryWin32HandleInfoKHR importAllocInfo = {};
    importAllocInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    importAllocInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
    if (interopHandle)
    {
        importAllocInfo.handle = (HANDLE)interopHandle->value;
        bufMemAllocInfo.pNext = &importAllocInfo;
    }
#else
    VkImportMemoryFdInfoKHR importAllocInfo = {};
    importAllocInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    importAllocInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    if (interopHandle)
    {
        importAllocInfo.fd = (int)interopHandle->value;
        bufMemAllocInfo.pNext = &importAllocInfo;
    }
#endif

    VkResult result = loader->vkAllocateMemory(vulkanDevice, &bufMemAllocInfo, nullptr, &ptr->memoryVk);
    if (result == VK_SUCCESS)
    {
        context->deviceQueue->device->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Buffer mem allocate %lld bytes", bufMemReq.size);
    }
    else
    {
        bufMemAllocInfo.memoryTypeIndex = bufMemType_sysmem;
        result = loader->vkAllocateMemory(vulkanDevice, &bufMemAllocInfo, nullptr, &ptr->memoryVk);
        if (result == VK_SUCCESS)
        {
            context->deviceQueue->device->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Buffer sysmem fallback allocate %lld bytes", bufMemReq.size);
        }
        else
        {
            context->deviceQueue->device->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Buffer allocate failed %lld bytes", bufMemReq.size);
        }
    }

    if (result == VK_SUCCESS)
    {
        ptr->allocationBytes = bufMemReq.size;
        device_reportMemoryAllocate(context->deviceQueue->device, ptr->memory_type, ptr->allocationBytes);

        loader->vkBindBufferMemory(vulkanDevice, ptr->bufferVk, ptr->memoryVk, 0u);

        if (ptr->memory_type == PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD ||
            ptr->memory_type == PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK)
        {
            loader->vkMapMemory(vulkanDevice, ptr->memoryVk, 0u, VK_WHOLE_SIZE, 0u, &ptr->mappedData);
        }
    }
    else // free buffer and set null
    {
        loader->vkDestroyBuffer(loader->device, ptr->bufferVk, nullptr);
        ptr->bufferVk = VK_NULL_HANDLE;
    }
}

void buffer_createBufferView(Context* context, Buffer* ptr, VkBufferView* view)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkBufferViewCreateInfo bufViewCreateInfo = {};
    bufViewCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    bufViewCreateInfo.buffer = ptr->bufferVk;
    bufViewCreateInfo.format = formatConverter_convertToVulkan(context->deviceQueue->device->formatConverter, ptr->desc.format);
    bufViewCreateInfo.offset = 0u;
    bufViewCreateInfo.range = VK_WHOLE_SIZE;

    if (bufViewCreateInfo.format == VK_FORMAT_UNDEFINED)
    {
        return;
    }

    loader->vkCreateBufferView(vulkanDevice, &bufViewCreateInfo, nullptr, view);
}

VkBufferView buffer_getBufferView(Context* context, Buffer* ptr, pnanovdb_compute_format_t aliasFormat)
{
    if (aliasFormat == PNANOVDB_COMPUTE_FORMAT_UNKNOWN)
    {
        return ptr->bufferViewVk;
    }
    for (pnanovdb_uint64_t idx = 0u; idx < ptr->aliasFormats.size(); idx++)
    {
        if (ptr->aliasFormats[idx] == aliasFormat)
        {
            return ptr->aliasBufferViews[idx];
        }
    }
    VkBufferView view = VK_NULL_HANDLE;
    buffer_createBufferView(context, ptr, &view);
    ptr->aliasFormats.push_back(aliasFormat);
    ptr->aliasBufferViews.push_back(view);
    return buffer_getBufferView(context, ptr, aliasFormat);
}

void buffer_initRestoreBarrier(Context* context, Buffer* ptr)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = 0u;
    barrier.dstAccessMask = 0u;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = ptr->bufferVk;
    barrier.offset = 0u;
    barrier.size = VK_WHOLE_SIZE;

    VkAccessFlags accessFlags = 0u;
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT)
    {
        accessFlags |= VK_ACCESS_UNIFORM_READ_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED)
    {
        accessFlags |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_BUFFER)
    {
        accessFlags |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED)
    {
        accessFlags |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_RW_BUFFER)
    {
        accessFlags |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_INDIRECT)
    {
        accessFlags |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC)
    {
        accessFlags |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (ptr->desc.usage & PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST)
    {
        accessFlags |= VK_ACCESS_SHADER_READ_BIT;
    }
    barrier.srcAccessMask = accessFlags;
    barrier.dstAccessMask = accessFlags;

    ptr->restoreBarrier = barrier;
}

void buffer_initCurrentBarrier(Context* context, Buffer* ptr)
{
    auto loader = &context->deviceQueue->device->loader;
    auto vulkanDevice = context->deviceQueue->device->vulkanDevice;

    VkBufferMemoryBarrier barrier = ptr->restoreBarrier;
    barrier.srcAccessMask = 0u;
    barrier.dstAccessMask = 0u;

    ptr->currentBarrier = barrier;
}

Buffer* buffer_create(Context* context, pnanovdb_compute_memory_type_t memory_type, const pnanovdb_compute_buffer_desc_t* desc, const pnanovdb_compute_interop_handle_t* interopHandle)
{
    auto ptr = new Buffer();

    ptr->desc = *desc;
    ptr->memory_type = memory_type;

    buffer_createBuffer(context, ptr, interopHandle);
    if (!ptr->bufferVk)
    {
        delete ptr;
        return nullptr;
    }
    buffer_createBufferView(context, ptr, &ptr->bufferViewVk);

    buffer_initRestoreBarrier(context, ptr);
    buffer_initCurrentBarrier(context, ptr);

    return ptr;
}

void buffer_destroy(Context* context, Buffer* ptr)
{
    auto loader = &context->deviceQueue->device->loader;

    loader->vkDestroyBufferView(loader->device, ptr->bufferViewVk, nullptr);
    for (pnanovdb_uint64_t idx = 0u; idx < ptr->aliasBufferViews.size(); idx++)
    {
        loader->vkDestroyBufferView(loader->device, ptr->aliasBufferViews[idx], nullptr);
    }
    ptr->mappedData = nullptr;

    // use memoryVk as indicator of ownership of original buffer
    if (ptr->memoryVk)
    {
        loader->vkDestroyBuffer(loader->device, ptr->bufferVk, nullptr);
        loader->vkFreeMemory(loader->device, ptr->memoryVk, nullptr);

        device_reportMemoryFree(context->deviceQueue->device, ptr->memory_type, ptr->allocationBytes);
    }

    delete ptr;
}

pnanovdb_bool_t bufferDesc_compare(const pnanovdb_compute_buffer_desc_t* a, const pnanovdb_compute_buffer_desc_t* b)
{
    if (a->usage == b->usage &&
        a->format == b->format &&
        a->size_in_bytes == b->size_in_bytes)
    {
        return PNANOVDB_TRUE;
    }
    return PNANOVDB_FALSE;
}

pnanovdb_compute_buffer_t* createBuffer(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_memory_type_t memory_type, const pnanovdb_compute_buffer_desc_t* desc)
{
    auto context = cast(contextIn);

    for (pnanovdb_uint32_t idx = 0u; idx < context->pool_buffers.size(); idx++)
    {
        auto ptr = context->pool_buffers[idx].get();
        if (ptr && ptr->refCount == 0 && bufferDesc_compare(&ptr->desc, desc) && ptr->memory_type == memory_type)
        {
            ptr->refCount = 1;
            ptr->lastActive = context->deviceQueue->nextFenceValue;
            return cast(ptr);
        }
    }

    auto ptr = buffer_create(context, memory_type, desc, nullptr);

    ptr->refCount = 1;
    ptr->lastActive = context->deviceQueue->nextFenceValue;
    context->pool_buffers.push_back(std::unique_ptr<Buffer>(ptr));

    return cast(ptr);
}

pnanovdb_compute_buffer_t* createBufferFromExternalHandle(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_buffer_desc_t* desc, const pnanovdb_compute_interop_handle_t* interopHandle)
{
    auto context = cast(contextIn);

    // do not recycle from pool for external resources

    auto ptr = buffer_create(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, desc, interopHandle);
    if (!ptr)
    {
        return nullptr;
    }

    ptr->refCount = 1;
    ptr->lastActive = context->deviceQueue->nextFenceValue;
    context->pool_buffers.push_back(std::unique_ptr<Buffer>(ptr));

    return cast(ptr);
}

void destroyBuffer(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_buffer_t* buffer)
{
    auto context = cast(contextIn);
    auto ptr = cast(buffer);

    if (ptr->refCount <= 0)
    {
        context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "pnanovdb_compute_context_t::destroyBuffer() called on inactive buffer %p", buffer);
        context->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "buffer_desc_t usage(%d) format(%d) structure_stride(%d) size_in_bytes(%zu) memory_type(%d)",
            ptr->desc.usage, ptr->desc.format, ptr->desc.structure_stride, ptr->desc.size_in_bytes, ptr->memory_type);
    }

    // defer release to end of frame for now
    // resource getting created/destroyed within single frame were being too eagerly recycled
    context->deferredReleaseBuffers.push_back(buffer);
    //ptr->refCount--;
    //ptr->lastActive = context->deviceQueue->nextFenceValue;
}

void context_destroyBuffers(Context* context)
{
    context->bufferTransients.clear();
    context->bufferAcquires.clear();

    for (pnanovdb_uint32_t idx = 0u; idx < context->pool_buffers.size(); idx++)
    {
        auto ptr = context->pool_buffers[idx].release();
        buffer_destroy(context, ptr);
    }
}

pnanovdb_compute_buffer_transient_t* getBufferTransient(pnanovdb_compute_context_t* contextIn, const pnanovdb_compute_buffer_desc_t* desc)
{
    auto context = cast(contextIn);
    auto ptr = new BufferTransient();
    context->bufferTransients.push_back(std::unique_ptr<BufferTransient>(ptr));
    ptr->desc = *desc;
    ptr->buffer = nullptr;
    ptr->aliasBuffer = nullptr;
    ptr->aliasFormat = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    ptr->nodeBegin = 0;
    ptr->nodeEnd = 0;
    return cast(ptr);
}

pnanovdb_compute_buffer_transient_t* registerBufferAsTransient(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_buffer_t* buffer)
{
    auto context = cast(contextIn);
    auto ptr = new BufferTransient();
    context->bufferTransients.push_back(std::unique_ptr<BufferTransient>(ptr));
    ptr->desc = cast(buffer)->desc;
    ptr->buffer = cast(buffer);
    ptr->aliasBuffer = nullptr;
    ptr->aliasFormat = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    ptr->nodeBegin = 0;
    ptr->nodeEnd = 0;
    return cast(ptr);
}

pnanovdb_compute_buffer_transient_t* aliasBufferTransient(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_buffer_transient_t* buffer, pnanovdb_compute_format_t format, pnanovdb_uint32_t structureStride)
{
    auto context = cast(contextIn);
    auto ptr = new BufferTransient();
    context->bufferTransients.push_back(std::unique_ptr<BufferTransient>(ptr));
    ptr->desc = cast(buffer)->desc;
    ptr->buffer = nullptr;
    ptr->aliasBuffer = cast(buffer);
    ptr->aliasFormat = format;
    ptr->nodeBegin = 0;
    ptr->nodeEnd = 0;
    return cast(ptr);
}

pnanovdb_compute_buffer_acquire_t* enqueueAcquireBuffer(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_buffer_transient_t* bufferTransient)
{
    auto context = cast(contextIn);
    auto ptr = new BufferAcquire();
    context->bufferAcquires.push_back(std::unique_ptr<BufferAcquire>(ptr));
    ptr->bufferTransient = cast(bufferTransient);
    ptr->buffer = nullptr;
    return cast(ptr);
}

pnanovdb_bool_t getAcquiredBuffer(pnanovdb_compute_context_t* contextIn, pnanovdb_compute_buffer_acquire_t* acquire, pnanovdb_compute_buffer_t** outBuffer)
{
    auto context = cast(contextIn);
    auto ptr = cast(acquire);
    if (ptr->buffer)
    {
        *outBuffer = cast(ptr->buffer);

        // remove from acquire array
        for (pnanovdb_uint64_t idx = 0u; idx < context->bufferAcquires.size(); idx++)
        {
            if (context->bufferAcquires[idx].get() == ptr)
            {
                context->bufferAcquires.erase(context->bufferAcquires.begin() + idx);
                break;
            }
        }

        return PNANOVDB_TRUE;
    }
    return PNANOVDB_FALSE;
}

void* mapBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer)
{
    return (cast(buffer))->mappedData;
}

void unmapBuffer(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer)
{
    // NOP
}

void device_getBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, void* dstHandle, pnanovdb_uint64_t dstHandleSize, pnanovdb_uint64_t* pBufferSizeInBytes)
{
    auto ctx = cast(context);
    auto ptr = cast(buffer);
#if defined(_WIN32)
    HANDLE handle = {};
    VkMemoryGetWin32HandleInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory = ptr->memoryVk;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    ctx->deviceQueue->device->loader.vkGetMemoryWin32HandleKHR(ctx->deviceQueue->device->vulkanDevice, &handleInfo, &handle);

    memset(dstHandle, 0, dstHandleSize);
    if (dstHandleSize >= sizeof(handle))
    {
        memcpy(dstHandle, &handle, sizeof(handle));
    }
#else
    int fd = 0;
    VkMemoryGetFdInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    handleInfo.memory = ptr->memoryVk;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    ctx->deviceQueue->device->loader.vkGetMemoryFdKHR(ctx->deviceQueue->device->vulkanDevice, &handleInfo, &fd);

    memset(dstHandle, 0, dstHandleSize);
    if (dstHandleSize >= sizeof(fd))
    {
        memcpy(dstHandle, &fd, sizeof(fd));
    }
#endif
    if (pBufferSizeInBytes)
    {
        *pBufferSizeInBytes = ptr->desc.size_in_bytes;
    }
}

void device_closeBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, const void* srcHandle, pnanovdb_uint64_t srcHandleSize)
{
    auto ctx = cast(context);
    auto ptr = cast(buffer);
#if defined(_WIN32)
    HANDLE handle = {};
    if (srcHandleSize >= sizeof(handle))
    {
        memcpy(&handle, srcHandle, sizeof(handle));

        CloseHandle(handle);
    }
#else
    int fd = 0;
    if (srcHandleSize >= sizeof(fd))
    {
        memcpy(&fd, srcHandle, sizeof(fd));

        close(fd);
    }
#endif
}

void getBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, pnanovdb_compute_interop_handle_t* dstHandle)
{
#if defined(_WIN32)
    dstHandle->type = PNANOVDB_COMPUTE_INTEROP_HANDLE_TYPE_OPAQUE_WIN32;
#else
    dstHandle->type = PNANOVDB_COMPUTE_INTEROP_HANDLE_TYPE_OPAQUE_FD;
#endif
    device_getBufferExternalHandle(context, buffer, &dstHandle->value, sizeof(dstHandle->value), &dstHandle->resource_size_in_bytes);
}

void closeBufferExternalHandle(pnanovdb_compute_context_t* context, pnanovdb_compute_buffer_t* buffer, const pnanovdb_compute_interop_handle_t* srcHandle)
{
    device_closeBufferExternalHandle(context, buffer, &srcHandle->value, sizeof(srcHandle->value));
}

} // end namespace
