
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   DeviceVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include "CommonVulkan.h"

#if defined(_WIN32)

#else
#include <unistd.h>
#endif

#include <string.h>

pnanovdb_compute_interface_t* pnanovdbGetContextInterface_vulkan();

namespace pnanovdb_vulkan
{
/// ************************** Device Manager **************************************

pnanovdb_compute_device_manager_t* createDeviceManager(pnanovdb_bool_t enableValidationOnDebugBuild)
{
    auto ptr = new DeviceManager();

    ptr->vulkan_module = pnanovdb_load_library("vulkan-1.dll", "libvulkan.so.1", "libvulkan.1.dylib");

    auto getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)pnanovdb_get_proc_address(ptr->vulkan_module, "vkGetInstanceProcAddr");

    pnanovdb_vulkan_loader_global(&ptr->loader, getInstanceProcAddr);

    auto loader = &ptr->loader;

    // select extensions
    std::vector<const char*> instanceExtensionsEnabled;
    selectInstanceExtensions(ptr, instanceExtensionsEnabled);

    // create instance
    uint32_t numLayers = 0u;
    const char** layers = nullptr;
#ifdef _DEBUG
    const uint32_t numLayers_validation = 1u;
    const char* layers_validation[numLayers_validation] = {
        "VK_LAYER_KHRONOS_validation"
    };
    if (enableValidationOnDebugBuild)
    {
        numLayers = numLayers_validation;
        layers = layers_validation;
    }
#endif

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = nullptr;
    instanceCreateInfo.enabledLayerCount = numLayers;
    instanceCreateInfo.ppEnabledLayerNames = layers;
    instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensionsEnabled.size();
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensionsEnabled.data();
#if defined(__APPLE__)
    instanceCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkResult res = loader->vkCreateInstance(&instanceCreateInfo, nullptr, &ptr->vulkanInstance);

    // for non-dev systems, gracefully fall back to no validation layers
    if (!ptr->vulkanInstance)
    {
        instanceCreateInfo.enabledLayerCount = 0u;
        instanceCreateInfo.ppEnabledLayerNames = nullptr;

        loader->vkCreateInstance(&instanceCreateInfo, nullptr, &ptr->vulkanInstance);
    }

    pnanovdb_vulkan_loader_instance(&ptr->loader, ptr->vulkanInstance);

    // enumerate devices
    {
        uint32_t gpuCount = 0u;
        loader->vkEnumeratePhysicalDevices(ptr->vulkanInstance, &gpuCount, nullptr);

        ptr->physicalDevices.reserve(gpuCount);
        ptr->deviceProps.reserve(gpuCount);
        ptr->physicalDeviceDescs.reserve(gpuCount);

        ptr->physicalDevices.resize(gpuCount);
        ptr->deviceProps.resize(gpuCount);
        ptr->physicalDeviceDescs.resize(gpuCount);

        loader->vkEnumeratePhysicalDevices(ptr->vulkanInstance, &gpuCount, ptr->physicalDevices.data());

        for (uint32_t i = 0; i < gpuCount; i++)
        {
            if (ptr->enabledExtensions.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2 && ptr->enabledExtensions.VK_KHR_EXTERNAL_FENCE_CAPABILITIES)
            {
                VkPhysicalDeviceIDPropertiesKHR deviceIdVulkan = {};
                deviceIdVulkan.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;

                VkPhysicalDeviceProperties2KHR devicePropsVulkan = {};
                devicePropsVulkan.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
                devicePropsVulkan.pNext = &deviceIdVulkan;

                loader->vkGetPhysicalDeviceProperties2KHR(ptr->physicalDevices[i], &devicePropsVulkan);

                pnanovdb_compute_physical_device_desc_t physicalDeviceDesc = {};
                pnanovdb_uint64_t uuidNumBytes =
                    sizeof(deviceIdVulkan.deviceUUID) < sizeof(physicalDeviceDesc.device_uuid) ?
                    sizeof(deviceIdVulkan.deviceUUID) : sizeof(physicalDeviceDesc.device_uuid);
                for (pnanovdb_uint32_t uuidIdx = 0u; uuidIdx < uuidNumBytes; uuidIdx++)
                {
                    physicalDeviceDesc.device_uuid[uuidIdx] = deviceIdVulkan.deviceUUID[uuidIdx];
                }
                pnanovdb_uint64_t luidNumBytes =
                    sizeof(deviceIdVulkan.deviceLUID) < sizeof(physicalDeviceDesc.device_luid) ?
                    sizeof(deviceIdVulkan.deviceLUID) : sizeof(physicalDeviceDesc.device_luid);
                for (pnanovdb_uint32_t luidIdx = 0u; luidIdx < luidNumBytes; luidIdx++)
                {
                    physicalDeviceDesc.device_luid[luidIdx] = deviceIdVulkan.deviceLUID[luidIdx];
                }
                physicalDeviceDesc.device_node_mask = deviceIdVulkan.deviceNodeMask;
                physicalDeviceDesc.device_luid_valid = deviceIdVulkan.deviceLUIDValid;

                ptr->deviceProps[i] = devicePropsVulkan.properties;
                ptr->physicalDeviceDescs[i] = physicalDeviceDesc;
            }
            else
            {
                pnanovdb_compute_physical_device_desc_t physicalDeviceDesc = {};

                loader->vkGetPhysicalDeviceProperties(ptr->physicalDevices[i], &ptr->deviceProps[i]);
                ptr->physicalDeviceDescs[i] = physicalDeviceDesc;
            }
        }
    }

    return cast(ptr);
}

void destroyDeviceManager(pnanovdb_compute_device_manager_t* deviceManager)
{
    auto ptr = cast(deviceManager);

    ptr->loader.vkDestroyInstance(ptr->vulkanInstance, nullptr);

    pnanovdb_free_library(ptr->vulkan_module);
    ptr->vulkan_module = nullptr;

    delete ptr;
}

pnanovdb_bool_t enumerateDevices(pnanovdb_compute_device_manager_t* manager, pnanovdb_uint32_t deviceIndex, pnanovdb_compute_physical_device_desc_t* pDesc)
{
    auto ptr = cast(manager);
    if (deviceIndex < ptr->physicalDeviceDescs.size())
    {
        *pDesc = ptr->physicalDeviceDescs[deviceIndex];
        return PNANOVDB_TRUE;
    }
    return PNANOVDB_FALSE;
}

/// ************************** Device **************************************

void logDefault(pnanovdb_compute_log_level_t level, const char* format, ...)
{
    // NOP
}

pnanovdb_compute_device_t* createDevice(pnanovdb_compute_device_manager_t* deviceManagerIn, const pnanovdb_compute_device_desc_t* desc)
{
    auto deviceManager = cast(deviceManagerIn);

    pnanovdb_compute_physical_device_desc_t physicalDeviceDesc = {};
    if (!enumerateDevices(deviceManagerIn, desc->device_index, &physicalDeviceDesc))
    {
        return nullptr;
    }

    auto ptr = new Device();

    ptr->desc = *desc;
    if (desc->log_print)
    {
        ptr->logPrint = desc->log_print;
    }
    else
    {
        ptr->logPrint = logDefault;
    }

    ptr->deviceManager = deviceManager;
    ptr->formatConverter = formatConverter_create();

    auto instanceLoader = &ptr->deviceManager->loader;
    auto deviceLoader = &ptr->loader;

    // set physical device
    {
        ptr->physicalDevice = deviceManager->physicalDevices[desc->device_index];
    }

    // identify graphics and compute queues
    {
        std::vector<VkQueueFamilyProperties> queueProps;

        uint32_t queueCount = 0u;
        instanceLoader->vkGetPhysicalDeviceQueueFamilyProperties(ptr->physicalDevice, &queueCount, nullptr);

        queueProps.reserve(queueCount);
        queueProps.resize(queueCount);

        instanceLoader->vkGetPhysicalDeviceQueueFamilyProperties(ptr->physicalDevice, &queueCount, queueProps.data());

        uint32_t graphicsQueueFamilyIdx = 0u;
        for (uint32_t i = 0u; i < queueCount; i++)
        {
            if (queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                graphicsQueueFamilyIdx = i;
                break;
            }
        }
        ptr->graphicsQueueFamilyIdx = graphicsQueueFamilyIdx;

        uint32_t computeQueueFamilyIdx = ~0u;
        // prefer compute and no graphics
        for (uint32_t i = 0u; i < queueCount; i++)
        {
            if ((queueProps[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) == VK_QUEUE_COMPUTE_BIT)
            {
                computeQueueFamilyIdx = i;
                break;
            }
        }
        // prefer any compute capable queue that is not the selected graphics queue family
        if (computeQueueFamilyIdx == ~0u)
        {
            for (uint32_t i = 0u; i < queueCount; i++)
            {
                if (i != ptr->graphicsQueueFamilyIdx &&
                    queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                {
                    computeQueueFamilyIdx = i;
                    break;
                }
            }
        }
        // use same queue family as graphics, if no other option
        if (computeQueueFamilyIdx == ~0u)
        {
            computeQueueFamilyIdx = ptr->graphicsQueueFamilyIdx;
        }
        ptr->computeQueueFamilyIdx = computeQueueFamilyIdx;

        if (ptr->logPrint)
        {
            ptr->logPrint(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Vulkan graphics_queue_family_idx(%d) compute_queue_family_idx(%d)",
                ptr->graphicsQueueFamilyIdx, ptr->computeQueueFamilyIdx);
        }
    }

    std::vector<const char*> deviceExtensionsEnabled;
    selectDeviceExtensions(ptr, deviceExtensionsEnabled);

    // create device
    {
        const float queuePriorities[2] = { 1.f, 1.f };

        pnanovdb_uint32_t queueCreateInfoCount = 1u;
        VkDeviceQueueCreateInfo queueCreateInfo[2] = {};
        queueCreateInfo[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo[0].queueFamilyIndex = ptr->graphicsQueueFamilyIdx;
        queueCreateInfo[0].queueCount = 1u;
        queueCreateInfo[0].pQueuePriorities = queuePriorities;
        if (ptr->computeQueueFamilyIdx == ptr->graphicsQueueFamilyIdx)
        {
            queueCreateInfo[0].queueCount = 2u;
        }
        else
        {
            queueCreateInfoCount = 2u;
            queueCreateInfo[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo[1].queueFamilyIndex = ptr->computeQueueFamilyIdx;
            queueCreateInfo[1].queueCount = 1u;
            queueCreateInfo[1].pQueuePriorities = queuePriorities;
        }

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = queueCreateInfoCount;
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensionsEnabled.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionsEnabled.data();

        VkPhysicalDeviceFeatures features = {};
        instanceLoader->vkGetPhysicalDeviceFeatures(ptr->physicalDevice, &features);

        VkPhysicalDeviceFeatures enabledFeaturesVk = {};

#define PNANOVDB_VULKAN_TRY_ENABLE_FEATURE(X) \
        if (features.X) \
        { \
            ptr->enabledFeatures.X = PNANOVDB_TRUE; \
            enabledFeaturesVk.X = VK_TRUE; \
        }

        PNANOVDB_VULKAN_TRY_ENABLE_FEATURE(shaderStorageImageWriteWithoutFormat)
        PNANOVDB_VULKAN_TRY_ENABLE_FEATURE(shaderInt64)

#undef PNANOVDB_VULKAN_TRY_ENABLE_FEATURE

        deviceCreateInfo.pEnabledFeatures = &enabledFeaturesVk;

        instanceLoader->vkCreateDevice(ptr->physicalDevice, &deviceCreateInfo, nullptr, &ptr->vulkanDevice);

        pnanovdb_vulkan_loader_device(&ptr->loader, ptr->vulkanDevice, instanceLoader->vkGetDeviceProcAddr);
    }

    // get properties
    instanceLoader->vkGetPhysicalDeviceProperties(ptr->physicalDevice, &ptr->physicalDeviceProperties);
    instanceLoader->vkGetPhysicalDeviceMemoryProperties(ptr->physicalDevice, &ptr->memoryProperties);

    // get graphics queue
    deviceLoader->vkGetDeviceQueue(ptr->vulkanDevice, ptr->graphicsQueueFamilyIdx, 0u, &ptr->graphicsQueueVk);

    // get compute queue
    pnanovdb_uint32_t compute_queue_idx = 0u;
    if (ptr->computeQueueFamilyIdx == ptr->graphicsQueueFamilyIdx)
    {
        compute_queue_idx = 1u;
    }
    deviceLoader->vkGetDeviceQueue(ptr->vulkanDevice, ptr->computeQueueFamilyIdx, compute_queue_idx, &ptr->computeQueueVk);

    ptr->deviceQueue = deviceQueue_create(ptr, ptr->graphicsQueueFamilyIdx, ptr->graphicsQueueVk);
    ptr->computeQueue = deviceQueue_create(ptr, ptr->computeQueueFamilyIdx, ptr->computeQueueVk);

    return cast(ptr);
}

void destroyDevice(pnanovdb_compute_device_manager_t* manager, pnanovdb_compute_device_t* device)
{
    auto ptr = cast(device);

    deviceQueue_destroy(ptr->deviceQueue);

    ptr->loader.vkDestroyDevice(ptr->vulkanDevice, nullptr);

    formatConverter_destroy(ptr->formatConverter);

    delete ptr;
}

pnanovdb_compute_queue_t* getDeviceQueue(const pnanovdb_compute_device_t* device)
{
    auto ptr = cast(device);
    return cast(ptr->deviceQueue);
}

pnanovdb_compute_queue_t* getComputeQueue(const pnanovdb_compute_device_t* device)
{
    auto ptr = cast(device);
    return cast(ptr->computeQueue);
}

void getMemoryStats(pnanovdb_compute_device_t* device, pnanovdb_compute_device_memory_stats_t* dstStats)
{
    auto ptr = cast(device);
    if (dstStats)
    {
        *dstStats = ptr->memoryStats;
    }
}

void device_reportMemoryAllocate(Device* device, pnanovdb_compute_memory_type_t type, pnanovdb_uint64_t bytes)
{
    if (type == PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE)
    {
        device->memoryStats.device_memory_bytes += bytes;
    }
    else if (type == PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD)
    {
        device->memoryStats.upload_memory_bytes += bytes;
    }
    else if (type == PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK)
    {
        device->memoryStats.readback_memory_bytes += bytes;
    }
}

void device_reportMemoryFree(Device* device, pnanovdb_compute_memory_type_t type, pnanovdb_uint64_t bytes)
{
    if (type == PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE)
    {
        device->memoryStats.device_memory_bytes -= bytes;
    }
    else if (type == PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD)
    {
        device->memoryStats.upload_memory_bytes -= bytes;
    }
    else if (type == PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK)
    {
        device->memoryStats.readback_memory_bytes -= bytes;
    }
}

/// ************************** DeviceSemaphore **************************************

pnanovdb_compute_semaphore_t* createSemaphore(pnanovdb_compute_device_t* device)
{
    auto ptr = new DeviceSemaphore();

    ptr->device = cast(device);

    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
    exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#if defined(_WIN32)
    exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
    exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
    semaphoreCreateInfo.pNext = &exportSemaphoreCreateInfo;

    ptr->device->loader.vkCreateSemaphore(
        ptr->device->vulkanDevice, &semaphoreCreateInfo, nullptr, &ptr->semaphoreVk);

    return cast(ptr);
}

void destroySemaphore(pnanovdb_compute_semaphore_t* semaphore)
{
    auto ptr = cast(semaphore);

    ptr->device->loader.vkDestroySemaphore(
        ptr->device->vulkanDevice, ptr->semaphoreVk, nullptr);

    delete ptr;
}

void getSemaphoreExternalHandle(pnanovdb_compute_semaphore_t* semaphore, void* dstHandle, pnanovdb_uint64_t dstHandleSize)
{
    auto ptr = cast(semaphore);
#if defined(_WIN32)
    HANDLE handle = {};
    VkSemaphoreGetWin32HandleInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = ptr->semaphoreVk;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    ptr->device->loader.vkGetSemaphoreWin32HandleKHR(ptr->device->vulkanDevice, &handleInfo, &handle);

    memset(dstHandle, 0, dstHandleSize);
    if (dstHandleSize >= sizeof(handle))
    {
        memcpy(dstHandle, &handle, sizeof(handle));
    }
#else
    int fd = 0;
    VkSemaphoreGetFdInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    handleInfo.semaphore = ptr->semaphoreVk;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    ptr->device->loader.vkGetSemaphoreFdKHR(ptr->device->vulkanDevice, &handleInfo, &fd);

    memset(dstHandle, 0, dstHandleSize);
    if (dstHandleSize >= sizeof(fd))
    {
        memcpy(dstHandle, &fd, sizeof(fd));
    }
#endif
}

void closeSemaphoreExternalHandle(pnanovdb_compute_semaphore_t* semaphore, const void* srcHandle, pnanovdb_uint64_t srcHandleSize)
{
    auto ptr = cast(semaphore);
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

/// ************************** DeviceQueue **************************************

DeviceQueue* deviceQueue_create(Device* device, uint32_t queueFamilyIdx, VkQueue queue)
{
    auto ptr = new DeviceQueue();

    ptr->device = device;

    ptr->vulkanDevice = ptr->device->vulkanDevice;
    ptr->queueFamilyIdx = queueFamilyIdx;
    ptr->queueVk = queue;

    auto loader = &ptr->device->loader;

    VkCommandPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = queueFamilyIdx;

    loader->vkCreateCommandPool(ptr->vulkanDevice, &poolCreateInfo, nullptr, &ptr->commandPool);

    VkCommandBufferAllocateInfo commandInfo = {};
    commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandInfo.commandPool = ptr->commandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1u;

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0u; // VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < kMaxFramesInFlight; i++)
    {
        loader->vkAllocateCommandBuffers(ptr->vulkanDevice, &commandInfo, &ptr->commandBuffers[i]);

        loader->vkCreateFence(ptr->vulkanDevice, &fenceCreateInfo, nullptr, &ptr->fences[i].fence);
        ptr->fences[i].active = PNANOVDB_FALSE;
        ptr->fences[i].value = 0u;
    }

    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    loader->vkCreateSemaphore(ptr->vulkanDevice, &semaphoreCreateInfo, nullptr, &ptr->beginFrameSemaphore);
    loader->vkCreateSemaphore(ptr->vulkanDevice, &semaphoreCreateInfo, nullptr, &ptr->endFrameSemaphore);

    // Second step of flush to prime command buffer
    flushStepB(ptr);

    ptr->context = context_create(ptr);

    return ptr;
}

void deviceQueue_destroy(DeviceQueue* ptr)
{
    auto loader = &ptr->device->loader;

    // Wait idle, since context destroy will force destroy resources
    waitIdle(cast(ptr));

    context_destroy(ptr->context);

    loader->vkDestroySemaphore(ptr->vulkanDevice, ptr->beginFrameSemaphore, nullptr);
    loader->vkDestroySemaphore(ptr->vulkanDevice, ptr->endFrameSemaphore, nullptr);

    for (int i = 0; i < kMaxFramesInFlight; i++)
    {
        loader->vkFreeCommandBuffers(ptr->vulkanDevice, ptr->commandPool, 1u, &ptr->commandBuffers[i]);
        loader->vkDestroyFence(ptr->vulkanDevice, ptr->fences[i].fence, nullptr);
    }
    loader->vkDestroyCommandPool(ptr->vulkanDevice, ptr->commandPool, nullptr);

    ptr->device = nullptr;

    delete ptr;
}

int flushStepA(DeviceQueue* ptr, pnanovdb_compute_semaphore_t* waitSemaphore, pnanovdb_compute_semaphore_t* signalSemaphore)
{
    auto loader = &ptr->device->loader;

    if (ptr->context)
    {
        context_flushNodes(ptr->context);
    }

    loader->vkEndCommandBuffer(ptr->commandBuffer);

    VkPipelineStageFlags stageFlags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSemaphore waitSemaphores[2u] = {};
    VkSemaphore signalSemaphores[2u] = {};

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 0u;
    if (ptr->currentBeginFrameSemaphore)
    {
        waitSemaphores[submitInfo.waitSemaphoreCount] = ptr->currentBeginFrameSemaphore;
        submitInfo.waitSemaphoreCount++;
        submitInfo.pWaitSemaphores = waitSemaphores;
    }
    if (waitSemaphore)
    {
        waitSemaphores[submitInfo.waitSemaphoreCount] = cast(waitSemaphore)->semaphoreVk;
        submitInfo.waitSemaphoreCount++;
        submitInfo.pWaitSemaphores = waitSemaphores;
    }
    submitInfo.pWaitDstStageMask = &stageFlags;
    submitInfo.commandBufferCount = 1u;
    submitInfo.pCommandBuffers = &ptr->commandBuffer;
    submitInfo.signalSemaphoreCount = 0u;
    if (ptr->currentEndFrameSemaphore)
    {
        signalSemaphores[submitInfo.signalSemaphoreCount] = ptr->currentEndFrameSemaphore;
        submitInfo.signalSemaphoreCount++;
        submitInfo.pSignalSemaphores = signalSemaphores;
    }
    if (signalSemaphore)
    {
        signalSemaphores[submitInfo.signalSemaphoreCount] = cast(signalSemaphore)->semaphoreVk;
        submitInfo.signalSemaphoreCount++;
        submitInfo.pSignalSemaphores = signalSemaphores;
    }

    VkResult result = loader->vkQueueSubmit(ptr->queueVk, 1u, &submitInfo, ptr->fences[ptr->commandBufferIdx].fence);

    // mark signaled fence value
    ptr->fences[ptr->commandBufferIdx].value = ptr->nextFenceValue;
    ptr->fences[ptr->commandBufferIdx].active = PNANOVDB_TRUE;

    // increment fence value
    ptr->nextFenceValue++;

    ptr->currentBeginFrameSemaphore = VK_NULL_HANDLE;

    return (result == VK_ERROR_DEVICE_LOST) ? 1 : 0;
}

void deviceQueue_fenceUpdate(DeviceQueue* ptr, pnanovdb_uint32_t fenceIdx, pnanovdb_bool_t blocking)
{
    if (ptr->fences[fenceIdx].active)
    {
        auto loader = &ptr->device->loader;

        uint64_t timeout = blocking ? ~0llu : 0llu;

        if (VK_SUCCESS == loader->vkWaitForFences(ptr->vulkanDevice, 1u, &ptr->fences[fenceIdx].fence, VK_TRUE, timeout))
        {
            loader->vkResetFences(ptr->vulkanDevice, 1u, &ptr->fences[fenceIdx].fence);

            ptr->fences[fenceIdx].active = PNANOVDB_FALSE;
            if (ptr->fences[fenceIdx].value > ptr->lastFenceCompleted)
            {
                ptr->lastFenceCompleted = ptr->fences[fenceIdx].value;
            }
        }
    }
}

void flushStepB(DeviceQueue* ptr)
{
    auto loader = &ptr->device->loader;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    ptr->commandBufferIdx = (ptr->commandBufferIdx + 1) % kMaxFramesInFlight;
    ptr->commandBuffer = ptr->commandBuffers[ptr->commandBufferIdx];

    // non-blocking update of fence values
    for (pnanovdb_uint32_t fenceIdx = 0u; fenceIdx < kMaxFramesInFlight; fenceIdx++)
    {
        deviceQueue_fenceUpdate(ptr, fenceIdx, PNANOVDB_FALSE);
    }

    // blocking update of fence values
    deviceQueue_fenceUpdate(ptr, ptr->commandBufferIdx, PNANOVDB_TRUE);

    loader->vkResetCommandBuffer(ptr->commandBuffer, 0);

    //loader->vkResetCommandPool(ptr->vulkanDevice, ptr->commandPool, 0);

    loader->vkBeginCommandBuffer(ptr->commandBuffer, &beginInfo);

    if (ptr->context)
    {
        context_resetNodes(ptr->context);
    }
}

int flush(pnanovdb_compute_queue_t* deviceQueue, pnanovdb_uint64_t* flushedFrameID, pnanovdb_compute_semaphore_t* waitSemaphore, pnanovdb_compute_semaphore_t* signalSemaphore)
{
    auto ptr = cast(deviceQueue);

    *flushedFrameID = ptr->nextFenceValue;

    int ret = flushStepA(ptr, waitSemaphore, signalSemaphore);
    flushStepB(ptr);

    return ret;
}

void waitIdle(pnanovdb_compute_queue_t* deviceQueue)
{
    auto ptr = cast(deviceQueue);

    ptr->device->loader.vkQueueWaitIdle(ptr->queueVk);

    for (pnanovdb_uint32_t fenceIdx = 0u; fenceIdx < kMaxFramesInFlight; fenceIdx++)
    {
        deviceQueue_fenceUpdate(ptr, fenceIdx, PNANOVDB_TRUE);
    }

    // update internal context
}

void waitForFrame(pnanovdb_compute_queue_t* deviceQueue, pnanovdb_uint64_t frameID)
{
    auto ptr = cast(deviceQueue);

    // avoid waiting on future frames
    if (frameID >= ptr->nextFenceValue)
    {
        return;
    }

    while (ptr->lastFenceCompleted < frameID)
    {
        pnanovdb_uint64_t minFenceValue = 0llu;
        pnanovdb_uint32_t minFenceIdx = 0u;
        for (pnanovdb_uint32_t fenceIdx = 0u; fenceIdx < kMaxFramesInFlight; fenceIdx++)
        {
            if (ptr->fences[fenceIdx].active)
            {
                pnanovdb_uint64_t frameFenceValue = ptr->fences[fenceIdx].value;
                if (minFenceValue == 0 || frameFenceValue < minFenceValue)
                {
                    minFenceValue = frameFenceValue;
                    minFenceIdx = fenceIdx;
                }
            }
        }
        // wait for min frame
        if (minFenceValue > 0u)
        {
            deviceQueue_fenceUpdate(ptr, minFenceIdx, PNANOVDB_TRUE);
        }
    }

    // update internal context
}

pnanovdb_uint64_t getLastFrameCompleted(pnanovdb_compute_queue_t* queue)
{
    auto ptr = cast(queue);
    return ptr->lastFenceCompleted;
}

pnanovdb_compute_interface_t* getContextInterface(const pnanovdb_compute_queue_t* ptr)
{
    return pnanovdbGetContextInterface_vulkan();
}

pnanovdb_compute_context_t* getContext(const pnanovdb_compute_queue_t* queue)
{
    auto deviceQueue = cast(queue);

    return cast(deviceQueue->context);
}

/// ************************** Swapchain **************************************

pnanovdb_compute_swapchain_t* createSwapchain(pnanovdb_compute_queue_t* queue, const pnanovdb_compute_swapchain_desc_t* desc)
{
    auto ptr = new Swapchain();

    ptr->desc = *desc;
    ptr->deviceQueue = cast(queue);

    auto device = ptr->deviceQueue->device;
    auto instanceLoader = &ptr->deviceQueue->device->deviceManager->loader;
    auto deviceLoader = &ptr->deviceQueue->device->loader;

#if defined(_WIN32)
    VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceCreateInfo.hinstance = ptr->desc.hinstance;
    surfaceCreateInfo.hwnd = ptr->desc.hwnd;

    instanceLoader->vkCreateWin32SurfaceKHR(device->deviceManager->vulkanInstance, &surfaceCreateInfo, nullptr, &ptr->surfaceVulkan);
#elif defined(__APPLE__)
    VkMacOSSurfaceCreateInfoMVK surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
    surfaceCreateInfo.pView = ptr->desc.nsview;

    if (ptr->desc.create_surface)
    {
        ptr->desc.create_surface(ptr->desc.window_userdata, device->deviceManager->vulkanInstance, (void**)&ptr->surfaceVulkan);
    }
    else
    {
        instanceLoader->vkCreateMacOSSurfaceMVK(device->deviceManager->vulkanInstance, &surfaceCreateInfo, nullptr, &ptr->surfaceVulkan);
    }
#else
    VkXlibSurfaceCreateInfoKHR surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    surfaceCreateInfo.dpy = ptr->desc.dpy;
    surfaceCreateInfo.window = ptr->desc.window;

    instanceLoader->vkCreateXlibSurfaceKHR(device->deviceManager->vulkanInstance, &surfaceCreateInfo, nullptr, &ptr->surfaceVulkan);
#endif

    VkBool32 supported = false;
    instanceLoader->vkGetPhysicalDeviceSurfaceSupportKHR(device->physicalDevice, ptr->deviceQueue->queueFamilyIdx, ptr->surfaceVulkan, &supported);

    uint32_t surfaceCount = 0;
    VkSurfaceFormatKHR surfaceFormats[32u] = {};
    instanceLoader->vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, ptr->surfaceVulkan, &surfaceCount, nullptr);
    if (surfaceCount > 32u) surfaceCount = 32u;
    instanceLoader->vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, ptr->surfaceVulkan, &surfaceCount, surfaceFormats);

    ptr->swapchainFormat = formatConverter_convertToVulkan(ptr->deviceQueue->device->formatConverter, ptr->desc.format);

    return cast(ptr);
}

void destroySwapchain(pnanovdb_compute_swapchain_t* swapchain)
{
    auto ptr = cast(swapchain);
    auto device = ptr->deviceQueue->device;
    auto instanceLoader = &ptr->deviceQueue->device->deviceManager->loader;
    auto deviceLoader = &ptr->deviceQueue->device->loader;

    swapchain_destroySwapchain(ptr);

    if (ptr->surfaceVulkan)
    {
        instanceLoader->vkDestroySurfaceKHR(device->deviceManager->vulkanInstance, ptr->surfaceVulkan, nullptr);
        ptr->surfaceVulkan = VK_NULL_HANDLE;
    }

#if defined(_WIN32)
#elif defined(__APPLE__)
#else
    if (ptr->moduleX11)
    {
        pnanovdb_free_library(ptr->moduleX11);
    }
#endif

    delete ptr;
}

void swapchain_getWindowSize(Swapchain* ptr, pnanovdb_uint32_t* width, pnanovdb_uint32_t* height)
{
#if defined(_WIN32)
    RECT rc;
    GetClientRect(ptr->desc.hwnd, &rc);
    *width = rc.right - rc.left;
    *height = rc.bottom - rc.top;
#elif defined(__APPLE__)
    int widthi, heighti;
    ptr->desc.get_framebuffer_size(ptr->desc.window_userdata, &widthi, &heighti);
    *width = widthi;
    *height = heighti;
#else
    if (!ptr->moduleX11)
    {
        ptr->moduleX11 = pnanovdb_load_library("X11.dll", "libX11.so", "libX11.dylib");
        ptr->getWindowAttrib = (decltype(&XGetWindowAttributes))pnanovdb_get_proc_address(ptr->moduleX11, "XGetWindowAttributes");
    }

    XWindowAttributes winAttr = {};
    ptr->getWindowAttrib(ptr->desc.dpy, ptr->desc.window, &winAttr);

    *width = winAttr.width;
    *height = winAttr.height;
#endif
}

void swapchain_initSwapchain(Swapchain* ptr)
{
    auto device = ptr->deviceQueue->device;
    auto instanceLoader = &ptr->deviceQueue->device->deviceManager->loader;
    auto deviceLoader = &ptr->deviceQueue->device->loader;

    pnanovdb_uint32_t width = 0u;
    pnanovdb_uint32_t height = 0u;
    swapchain_getWindowSize(ptr, &width, &height);

    VkExtent2D imageExtent = {};
    imageExtent.width = width;
    imageExtent.height = height;

    ptr->width = width;
    ptr->height = height;

    // catch this before throwing error
    if (ptr->width == 0 || ptr->height == 0)
    {
        ptr->valid = PNANOVDB_FALSE;
        return;
    }

    VkSurfaceCapabilitiesKHR surfaceCaps = {};
    instanceLoader->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device->physicalDevice, ptr->surfaceVulkan, &surfaceCaps);

    // clamp width and height based on surfaceCaps
    if (ptr->width < int(surfaceCaps.minImageExtent.width))
    {
        ptr->width = surfaceCaps.minImageExtent.width;
    }
    if (ptr->height < int(surfaceCaps.minImageExtent.height))
    {
        ptr->height = surfaceCaps.minImageExtent.height;
    }
    if (ptr->width > int(surfaceCaps.maxImageExtent.width))
    {
        ptr->width = surfaceCaps.maxImageExtent.width;
    }
    if (ptr->height > int(surfaceCaps.maxImageExtent.height))
    {
        ptr->height = surfaceCaps.maxImageExtent.height;
    }

    VkPresentModeKHR presentModes[8u] = {};
    uint32_t numPresentModes = 0u;
    instanceLoader->vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, ptr->surfaceVulkan, &numPresentModes, nullptr);
    if (numPresentModes > 8u) numPresentModes = 8u;
    instanceLoader->vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, ptr->surfaceVulkan, &numPresentModes, presentModes);

    VkPresentModeKHR presentOptions[3u] = { VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR };
    if (ptr->vsync)
    {
        presentOptions[0u] = VK_PRESENT_MODE_FIFO_KHR;
        presentOptions[1u] = VK_PRESENT_MODE_IMMEDIATE_KHR;
        presentOptions[2u] = VK_PRESENT_MODE_MAILBOX_KHR;
    }
    for (int j = 0; j < 3; j++)
    {
        for (uint32_t i = 0; i < numPresentModes; i++)
        {
            if (presentModes[i] == presentOptions[j])
            {
                ptr->presentMode = presentOptions[j];
                j = 3;
                break;
            }
        }
    }

    VkSwapchainKHR oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainCreateInfoKHR swapchainDesc = {};
    swapchainDesc.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainDesc.surface = ptr->surfaceVulkan;
    swapchainDesc.minImageCount = 3;
    swapchainDesc.imageFormat = ptr->swapchainFormat;
    swapchainDesc.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainDesc.imageExtent = imageExtent;
    swapchainDesc.imageArrayLayers = 1;
    swapchainDesc.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    swapchainDesc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainDesc.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchainDesc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainDesc.presentMode = ptr->presentMode;
    swapchainDesc.clipped = VK_TRUE;
    swapchainDesc.oldSwapchain = oldSwapchain;

    VkResult result = deviceLoader->vkCreateSwapchainKHR(device->vulkanDevice, &swapchainDesc, nullptr, &ptr->swapchainVulkan);

    if (result != VK_SUCCESS)
    {
        ptr->valid = PNANOVDB_FALSE;
        return;
    }

    ptr->numSwapchainImages = 0u;
    deviceLoader->vkGetSwapchainImagesKHR(device->vulkanDevice, ptr->swapchainVulkan, &ptr->numSwapchainImages, nullptr);
    if (ptr->numSwapchainImages > kMaxSwapchainImages)
    {
        ptr->numSwapchainImages = kMaxSwapchainImages;
    }
    deviceLoader->vkGetSwapchainImagesKHR(device->vulkanDevice, ptr->swapchainVulkan, &ptr->numSwapchainImages, ptr->swapchainImages);

    for (pnanovdb_uint32_t idx = 0; idx < ptr->numSwapchainImages; idx++)
    {
        pnanovdb_compute_texture_desc_t texDesc = {};
        texDesc.texture_type = PNANOVDB_COMPUTE_TEXTURE_TYPE_2D;
        texDesc.usage = PNANOVDB_COMPUTE_TEXTURE_USAGE_RW_TEXTURE | PNANOVDB_COMPUTE_TEXTURE_USAGE_COPY_DST;
        texDesc.format = ptr->desc.format;
        texDesc.width = imageExtent.width;
        texDesc.height = imageExtent.height;

        ptr->textures[idx] = createTextureExternal(cast(ptr->deviceQueue->context), &texDesc, ptr->swapchainImages[idx], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }

    ptr->valid = PNANOVDB_TRUE;
}

void swapchain_destroySwapchain(Swapchain* ptr)
{
    auto device = ptr->deviceQueue->device;
    auto instanceLoader = &ptr->deviceQueue->device->deviceManager->loader;
    auto deviceLoader = &ptr->deviceQueue->device->loader;

    waitIdle(cast(ptr->deviceQueue));

    for (pnanovdb_uint32_t idx = 0; idx < ptr->numSwapchainImages; idx++)
    {
        destroyTexture(cast(ptr->deviceQueue->context), ptr->textures[idx]);
        ptr->textures[idx] = nullptr;
    }

    if (ptr->swapchainVulkan != VK_NULL_HANDLE)
    {
        deviceLoader->vkDestroySwapchainKHR(device->vulkanDevice, ptr->swapchainVulkan, nullptr);
        ptr->swapchainVulkan = VK_NULL_HANDLE;
    }
}

void resizeSwapchain(pnanovdb_compute_swapchain_t* swapchain, pnanovdb_uint32_t width, pnanovdb_uint32_t height)
{
    auto ptr = cast(swapchain);

    if (ptr->valid)
    {
        if (width != ptr->width ||
            height != ptr->height)
        {
            swapchain_destroySwapchain(ptr);
            ptr->valid = PNANOVDB_FALSE;
        }
    }

    if (ptr->valid == PNANOVDB_FALSE)
    {
        swapchain_initSwapchain(ptr);
    }
}

int presentSwapchain(pnanovdb_compute_swapchain_t* swapchain, pnanovdb_bool_t vsync, pnanovdb_uint64_t* flushedFrameID)
{
    auto ptr = cast(swapchain);

    auto loader = &ptr->deviceQueue->device->loader;

    if (ptr->valid == PNANOVDB_FALSE)
    {
        return flush(cast(ptr->deviceQueue), flushedFrameID, nullptr, nullptr);
    }

    *flushedFrameID = ptr->deviceQueue->nextFenceValue;

    ptr->deviceQueue->currentEndFrameSemaphore = ptr->deviceQueue->endFrameSemaphore;

    int deviceReset = flushStepA(ptr->deviceQueue, nullptr, nullptr);

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &ptr->swapchainVulkan;
    presentInfo.pImageIndices = &ptr->currentSwapchainIdx;
    presentInfo.waitSemaphoreCount = 1u;
    presentInfo.pWaitSemaphores = &ptr->deviceQueue->currentEndFrameSemaphore;

    VkResult result = loader->vkQueuePresentKHR(ptr->deviceQueue->queueVk, &presentInfo);

    ptr->deviceQueue->currentEndFrameSemaphore = VK_NULL_HANDLE;

    flushStepB(ptr->deviceQueue);

    if (result != VK_SUCCESS || ptr->vsync != vsync)
    {
        ptr->vsync = vsync;
        swapchain_destroySwapchain(ptr);
        ptr->valid = PNANOVDB_FALSE;
    }

    if (ptr->valid)
    {
        pnanovdb_uint32_t compWidth = 0u;
        pnanovdb_uint32_t compHeight = 0u;
        swapchain_getWindowSize(ptr, &compWidth, &compHeight);

        if (compWidth != ptr->width ||
            compHeight != ptr->height)
        {
            swapchain_destroySwapchain(ptr);
            ptr->valid = PNANOVDB_FALSE;
        }
    }

    return deviceReset;
}

pnanovdb_compute_texture_t* getSwapchainFrontTexture(pnanovdb_compute_swapchain_t* swapchain)
{
    auto ptr = cast(swapchain);

    auto device = ptr->deviceQueue->device;
    auto loader = &ptr->deviceQueue->device->loader;

    if (ptr->valid == PNANOVDB_FALSE)
    {
        swapchain_initSwapchain(ptr);
    }

    if (ptr->valid == PNANOVDB_FALSE)
    {
        return nullptr;
    }

    ptr->deviceQueue->currentBeginFrameSemaphore = ptr->deviceQueue->beginFrameSemaphore;

    VkResult result = loader->vkAcquireNextImageKHR(device->vulkanDevice, ptr->swapchainVulkan, UINT64_MAX, ptr->deviceQueue->currentBeginFrameSemaphore, VK_NULL_HANDLE, &ptr->currentSwapchainIdx);

    if (result != VK_SUCCESS)
    {
        swapchain_destroySwapchain(ptr);
        ptr->valid = PNANOVDB_FALSE;
        return nullptr;
    }

    return ptr->textures[ptr->currentSwapchainIdx];
}

} // end namespace
