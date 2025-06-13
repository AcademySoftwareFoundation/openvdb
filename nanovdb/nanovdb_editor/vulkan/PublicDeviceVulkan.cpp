// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   PublicDeviceVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include "CommonVulkan.h"

namespace pnanovdb_vulkan
{
    pnanovdb_compute_api_t get_api(pnanovdb_compute_context_t* context)
    {
        return PNANOVDB_COMPUTE_API_VULKAN;
    }

    pnanovdb_bool_t is_feature_supported(pnanovdb_compute_context_t* context, pnanovdb_compute_feature_t feature)
    {
        Context* ctx = cast(context);
        pnanovdb_bool_t isSupported = PNANOVDB_FALSE;
        if (feature == PNANOVDB_COMPUTE_FEATURE_ALIAS_RESOURCE_FORMATS)
        {
            isSupported = PNANOVDB_TRUE;
        }
        else if (feature == PNANOVDB_COMPUTE_FEATURE_BUFFER_EXTERNAL_HANDLE)
        {
            isSupported = ctx->deviceQueue->device->desc.enable_external_usage;
        }
        return isSupported;
    }

    void get_frame_info(pnanovdb_compute_context_t* context, pnanovdb_compute_frame_info_t* frame_info)
    {
        Context* ctx = cast(context);
        frame_info->frame_local_current = ctx->deviceQueue->nextFenceValue;
        frame_info->frame_local_completed = ctx->deviceQueue->lastFenceCompleted;
        frame_info->frame_global_current = ctx->deviceQueue->nextFenceValue;
        frame_info->frame_global_completed = ctx->deviceQueue->lastFenceCompleted;
    }

    pnanovdb_compute_log_print_t get_log_print(pnanovdb_compute_context_t* context)
    {
        auto ctx = cast(context);
        return ctx->logPrint;
    }

    void execute_tasks(pnanovdb_compute_context_t* context, pnanovdb_uint32_t taskCount, pnanovdb_uint32_t taskGranularity, pnanovdb_compute_thread_pool_task_t task, void* userdata)
    {
        Context* ctx = cast(context);

        static pnanovdb_uint8_t smem[1024u * 1024u];
        for (pnanovdb_uint32_t task_idx = 0u; task_idx < taskCount; task_idx++)
        {
            task(task_idx, 0u, smem, userdata);
        }
    }
}

pnanovdb_compute_interface_t* pnanovdbGetContextInterface_vulkan()
{
    using namespace pnanovdb_vulkan;
    static pnanovdb_compute_interface_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_compute_interface_t) };

    iface.get_api = get_api;
    iface.is_feature_supported = is_feature_supported;
    iface.get_frame_info = get_frame_info;
    iface.get_log_print = get_log_print;

    iface.execute_tasks = execute_tasks;

    iface.create_buffer = createBuffer;
    iface.destroy_buffer = destroyBuffer;
    iface.get_buffer_transient = getBufferTransient;
    iface.register_buffer_as_transient = registerBufferAsTransient;
    iface.alias_buffer_transient = aliasBufferTransient;
    iface.enqueue_acquire_buffer = enqueueAcquireBuffer;
    iface.get_acquired_buffer = getAcquiredBuffer;
    iface.map_buffer = mapBuffer;
    iface.unmap_buffer = unmapBuffer;
    iface.get_buffer_external_handle = getBufferExternalHandle;
    iface.close_buffer_external_handle = closeBufferExternalHandle;
    iface.create_buffer_from_external_handle = createBufferFromExternalHandle;

    iface.create_texture = createTexture;
    iface.destroy_texture = destroyTexture;
    iface.get_texture_transient = getTextureTransient;
    iface.register_texture_as_transient = registerTextureAsTransient;
    iface.alias_texture_transient = aliasTextureTransient;
    iface.enqueue_acquire_texture = enqueueAcquireTexture;
    iface.get_acquired_texture = getAcquiredTexture;

    iface.create_sampler = createSampler;
    iface.get_default_sampler = getDefaultSampler;
    iface.destroy_sampler = destroySampler;

    iface.create_compute_pipeline = createComputePipeline;
    iface.destroy_compute_pipeline = destroyComputePipeline;

    iface.dispatch = addPassCompute;
    iface.copy_buffer = addPassCopyBuffer;

    return &iface;
}

pnanovdb_compute_device_interface_t* pnanovdb_get_compute_device_interface(pnanovdb_compute_api_t api)
{
    if (api != PNANOVDB_COMPUTE_API_VULKAN)
    {
        return nullptr;
    }

    using namespace pnanovdb_vulkan;
    static pnanovdb_compute_device_interface_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_compute_device_interface_t) };

    iface.create_device_manager = createDeviceManager;
    iface.destroy_device_manager = destroyDeviceManager;
    iface.enumerate_devices = enumerateDevices;

    iface.create_device = createDevice;
    iface.destroy_device = destroyDevice;
    iface.get_memory_stats = getMemoryStats;

    iface.create_semaphore = createSemaphore;
    iface.destroy_semaphore = destroySemaphore;
    iface.get_semaphore_external_handle = getSemaphoreExternalHandle;
    iface.close_semaphore_external_handle = closeSemaphoreExternalHandle;

    iface.get_device_queue = getDeviceQueue;
    iface.get_compute_queue = getComputeQueue;
    iface.flush = flush;
    iface.get_frame_global_completed = getLastFrameCompleted;
    iface.wait_for_frame = waitForFrame;
    iface.wait_idle = waitIdle;
    iface.get_compute_interface = getContextInterface;
    iface.get_compute_context = getContext;

    iface.create_swapchain = createSwapchain;
    iface.destroy_swapchain = destroySwapchain;
    iface.resize_swapchain = resizeSwapchain;
    iface.present_swapchain = presentSwapchain;
    iface.get_swapchain_front_texture = getSwapchainFrontTexture;

    iface.enable_profiler = enableProfiler;
    iface.disable_profiler = disableProfiler;

    iface.set_resource_min_lifetime = setResourceMinLifetime;

    return &iface;
}
