
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   LoaderVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#pragma once

#include "nanovdb/PNanoVDB.h"

#define PNANOVDB_VK_LOADER_PTR(FUNC_REF) \
    PFN_##FUNC_REF FUNC_REF = nullptr

#define PNANOVDB_VK_LOADER_INSTANCE(X) \
    ptr-> X = (PFN_##X)ptr->vkGetInstanceProcAddr(vulkanInstance, #X )

#define PNANOVDB_VK_LOADER_DEVICE(X) \
    ptr-> X = (PFN_##X)ptr->vkGetDeviceProcAddr(vulkanDevice, #X )

typedef struct pnanovdb_vulkan_enabled_features_t
{
    pnanovdb_bool_t shaderStorageImageWriteWithoutFormat;
    pnanovdb_bool_t shaderInt64;
}pnanovdb_vulkan_enabled_features_t;

typedef struct pnanovdb_vulkan_enabled_instance_extensions_t
{
    pnanovdb_bool_t VK_KHR_SURFACE;
    pnanovdb_bool_t VK_KHR_WIN32_SURFACE;
    pnanovdb_bool_t VK_EXT_METAL_SURFACE;
    pnanovdb_bool_t VK_MVK_MACOS_SURFACE;
    pnanovdb_bool_t VK_KHR_XLIB_SURFACE;
    pnanovdb_bool_t VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2;
    pnanovdb_bool_t VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES;
    pnanovdb_bool_t VK_KHR_EXTERNAL_FENCE_CAPABILITIES;
    pnanovdb_bool_t VK_KHR_EXTERNAL_MEMORY_CAPABILITIES;
    pnanovdb_bool_t VK_KHR_PORTABILITY_ENUMERATION;
}pnanovdb_vulkan_enabled_instance_extensions_t;

typedef struct pnanovdb_vulkan_enabled_device_extensions_t
{
    pnanovdb_bool_t VK_KHR_SWAPCHAIN;
    pnanovdb_bool_t VK_KHR_EXTERNAL_MEMORY;
    pnanovdb_bool_t VK_KHR_EXTERNAL_SEMAPHORE;
    pnanovdb_bool_t VK_KHR_EXTERNAL_MEMORY_WIN32;
    pnanovdb_bool_t VK_KHR_EXTERNAL_SEMAPHORE_WIN32;
    pnanovdb_bool_t VK_KHR_EXTERNAL_MEMORY_FD;
    pnanovdb_bool_t VK_KHR_EXTERNAL_SEMAPHORE_FD;
}pnanovdb_vulkan_enabled_device_extensions_t;

typedef struct pnanovdb_vulkan_instance_loader_t
{
    VkInstance instance = nullptr;

    // Instance functions
    PNANOVDB_VK_LOADER_PTR(vkGetInstanceProcAddr);
    PNANOVDB_VK_LOADER_PTR(vkCreateInstance);
    PNANOVDB_VK_LOADER_PTR(vkEnumerateInstanceExtensionProperties);

    PNANOVDB_VK_LOADER_PTR(vkDestroyInstance);
    PNANOVDB_VK_LOADER_PTR(vkGetDeviceProcAddr);
    PNANOVDB_VK_LOADER_PTR(vkEnumeratePhysicalDevices);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceProperties);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceProperties2KHR);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceQueueFamilyProperties);

    PNANOVDB_VK_LOADER_PTR(vkCreateDevice);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceMemoryProperties);
    PNANOVDB_VK_LOADER_PTR(vkEnumerateDeviceExtensionProperties);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceFeatures);

    // Instance surface
#if defined(_WIN32)
    PNANOVDB_VK_LOADER_PTR(vkCreateWin32SurfaceKHR);
#elif defined(__APPLE__)
    PNANOVDB_VK_LOADER_PTR(vkCreateMacOSSurfaceMVK);
#else
    PNANOVDB_VK_LOADER_PTR(vkCreateXlibSurfaceKHR);
#endif
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceSurfaceSupportKHR);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceSurfaceFormatsKHR);
    PNANOVDB_VK_LOADER_PTR(vkGetPhysicalDeviceSurfacePresentModesKHR);
    PNANOVDB_VK_LOADER_PTR(vkDestroySurfaceKHR);
}pnanovdb_vulkan_instance_loader_t;

typedef struct pnanovdb_vulkan_device_loader_t
{
    VkDevice device = nullptr;

    PNANOVDB_VK_LOADER_PTR(vkGetDeviceProcAddr);

    // Device functions
    PNANOVDB_VK_LOADER_PTR(vkDestroyDevice);
    PNANOVDB_VK_LOADER_PTR(vkGetDeviceQueue);
    PNANOVDB_VK_LOADER_PTR(vkCreateCommandPool);
    PNANOVDB_VK_LOADER_PTR(vkAllocateCommandBuffers);
    PNANOVDB_VK_LOADER_PTR(vkCreateFence);

    PNANOVDB_VK_LOADER_PTR(vkCreateSemaphore);
    PNANOVDB_VK_LOADER_PTR(vkDestroySemaphore);
    PNANOVDB_VK_LOADER_PTR(vkFreeCommandBuffers);
    PNANOVDB_VK_LOADER_PTR(vkDestroyFence);
    PNANOVDB_VK_LOADER_PTR(vkDestroyCommandPool);

    PNANOVDB_VK_LOADER_PTR(vkEndCommandBuffer);
    PNANOVDB_VK_LOADER_PTR(vkQueueSubmit);
    PNANOVDB_VK_LOADER_PTR(vkWaitForFences);
    PNANOVDB_VK_LOADER_PTR(vkResetFences);
    PNANOVDB_VK_LOADER_PTR(vkResetCommandBuffer);

    PNANOVDB_VK_LOADER_PTR(vkBeginCommandBuffer);
    PNANOVDB_VK_LOADER_PTR(vkQueueWaitIdle);
    PNANOVDB_VK_LOADER_PTR(vkCreateImage);
    PNANOVDB_VK_LOADER_PTR(vkGetImageMemoryRequirements);
    PNANOVDB_VK_LOADER_PTR(vkAllocateMemory);

    PNANOVDB_VK_LOADER_PTR(vkBindImageMemory);
    PNANOVDB_VK_LOADER_PTR(vkCreateImageView);
    PNANOVDB_VK_LOADER_PTR(vkDestroyImageView);
    PNANOVDB_VK_LOADER_PTR(vkDestroyImage);
    PNANOVDB_VK_LOADER_PTR(vkFreeMemory);

    PNANOVDB_VK_LOADER_PTR(vkCreateBuffer);
    PNANOVDB_VK_LOADER_PTR(vkGetBufferMemoryRequirements);
    PNANOVDB_VK_LOADER_PTR(vkBindBufferMemory);
    PNANOVDB_VK_LOADER_PTR(vkMapMemory);
    PNANOVDB_VK_LOADER_PTR(vkDestroyBuffer);

    PNANOVDB_VK_LOADER_PTR(vkCmdPipelineBarrier);
    PNANOVDB_VK_LOADER_PTR(vkCmdCopyBuffer);
    PNANOVDB_VK_LOADER_PTR(vkCreateShaderModule);
    PNANOVDB_VK_LOADER_PTR(vkDestroyPipeline);
    PNANOVDB_VK_LOADER_PTR(vkDestroyShaderModule);

    PNANOVDB_VK_LOADER_PTR(vkCreateDescriptorSetLayout);
    PNANOVDB_VK_LOADER_PTR(vkCreatePipelineLayout);
    PNANOVDB_VK_LOADER_PTR(vkDestroyPipelineLayout);
    PNANOVDB_VK_LOADER_PTR(vkDestroyDescriptorSetLayout);
    PNANOVDB_VK_LOADER_PTR(vkCreateDescriptorPool);

    PNANOVDB_VK_LOADER_PTR(vkAllocateDescriptorSets);
    PNANOVDB_VK_LOADER_PTR(vkDestroyDescriptorPool);
    PNANOVDB_VK_LOADER_PTR(vkUpdateDescriptorSets);
    PNANOVDB_VK_LOADER_PTR(vkCmdBindDescriptorSets);
    PNANOVDB_VK_LOADER_PTR(vkCmdBindPipeline);

    PNANOVDB_VK_LOADER_PTR(vkResetDescriptorPool);
    PNANOVDB_VK_LOADER_PTR(vkCreateBufferView);
    PNANOVDB_VK_LOADER_PTR(vkDestroyBufferView);
    PNANOVDB_VK_LOADER_PTR(vkCreateComputePipelines);
    PNANOVDB_VK_LOADER_PTR(vkCreateSampler);

    PNANOVDB_VK_LOADER_PTR(vkDestroySampler);
    PNANOVDB_VK_LOADER_PTR(vkCmdDispatch);
    PNANOVDB_VK_LOADER_PTR(vkCmdDispatchIndirect);
    PNANOVDB_VK_LOADER_PTR(vkCreateQueryPool);
    PNANOVDB_VK_LOADER_PTR(vkDestroyQueryPool);

    PNANOVDB_VK_LOADER_PTR(vkCmdResetQueryPool);
    PNANOVDB_VK_LOADER_PTR(vkCmdWriteTimestamp);
    PNANOVDB_VK_LOADER_PTR(vkCmdCopyQueryPoolResults);
    PNANOVDB_VK_LOADER_PTR(vkGetImageSubresourceLayout);
    PNANOVDB_VK_LOADER_PTR(vkCmdCopyImage);

    PNANOVDB_VK_LOADER_PTR(vkCmdCopyBufferToImage);
    PNANOVDB_VK_LOADER_PTR(vkCmdCopyImageToBuffer);
    PNANOVDB_VK_LOADER_PTR(vkCmdPushConstants);
#if defined(_WIN32)
    PNANOVDB_VK_LOADER_PTR(vkGetMemoryWin32HandleKHR);
    PNANOVDB_VK_LOADER_PTR(vkGetSemaphoreWin32HandleKHR);
#else
    PNANOVDB_VK_LOADER_PTR(vkGetMemoryFdKHR);
    PNANOVDB_VK_LOADER_PTR(vkGetSemaphoreFdKHR);
#endif

    // Device surface
    PNANOVDB_VK_LOADER_PTR(vkQueuePresentKHR);
    PNANOVDB_VK_LOADER_PTR(vkCreateSwapchainKHR);
    PNANOVDB_VK_LOADER_PTR(vkGetSwapchainImagesKHR);
    PNANOVDB_VK_LOADER_PTR(vkDestroySwapchainKHR);
    PNANOVDB_VK_LOADER_PTR(vkAcquireNextImageKHR);
}pnanovdb_vulkan_device_loader_t;

PNANOVDB_INLINE void pnanovdb_vulkan_loader_global(pnanovdb_vulkan_instance_loader_t* ptr, PFN_vkGetInstanceProcAddr getInstanceProcAddr)
{
    VkInstance vulkanInstance = nullptr;
    ptr->vkGetInstanceProcAddr = getInstanceProcAddr;
    PNANOVDB_VK_LOADER_INSTANCE(vkCreateInstance);
    PNANOVDB_VK_LOADER_INSTANCE(vkEnumerateInstanceExtensionProperties);
}

PNANOVDB_INLINE void pnanovdb_vulkan_loader_instance(pnanovdb_vulkan_instance_loader_t* ptr, VkInstance vulkanInstance)
{
    ptr->instance = vulkanInstance;

    PNANOVDB_VK_LOADER_INSTANCE(vkDestroyInstance);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetDeviceProcAddr);
    PNANOVDB_VK_LOADER_INSTANCE(vkEnumeratePhysicalDevices);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceProperties);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceProperties2KHR);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties);

    PNANOVDB_VK_LOADER_INSTANCE(vkCreateDevice);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceMemoryProperties);
    PNANOVDB_VK_LOADER_INSTANCE(vkEnumerateDeviceExtensionProperties);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceFeatures);

    // surface extensions
#if defined(_WIN32)
    PNANOVDB_VK_LOADER_INSTANCE(vkCreateWin32SurfaceKHR);
#elif defined(__APPLE__)
    PNANOVDB_VK_LOADER_INSTANCE(vkCreateMacOSSurfaceMVK);
#else
    PNANOVDB_VK_LOADER_INSTANCE(vkCreateXlibSurfaceKHR);
#endif
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceSurfaceSupportKHR);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceSurfaceFormatsKHR);
    PNANOVDB_VK_LOADER_INSTANCE(vkGetPhysicalDeviceSurfacePresentModesKHR);
    PNANOVDB_VK_LOADER_INSTANCE(vkDestroySurfaceKHR);
}

PNANOVDB_INLINE void pnanovdb_vulkan_loader_device(pnanovdb_vulkan_device_loader_t* ptr, VkDevice vulkanDevice, PFN_vkGetDeviceProcAddr getDeviceProcAddr)
{
    ptr->device = vulkanDevice;
    ptr->vkGetDeviceProcAddr = getDeviceProcAddr;

    PNANOVDB_VK_LOADER_DEVICE(vkDestroyDevice);
    PNANOVDB_VK_LOADER_DEVICE(vkGetDeviceQueue);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateCommandPool);
    PNANOVDB_VK_LOADER_DEVICE(vkAllocateCommandBuffers);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateFence);

    PNANOVDB_VK_LOADER_DEVICE(vkCreateSemaphore);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroySemaphore);
    PNANOVDB_VK_LOADER_DEVICE(vkFreeCommandBuffers);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyFence);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyCommandPool);

    PNANOVDB_VK_LOADER_DEVICE(vkEndCommandBuffer);
    PNANOVDB_VK_LOADER_DEVICE(vkQueueSubmit);
    PNANOVDB_VK_LOADER_DEVICE(vkWaitForFences);
    PNANOVDB_VK_LOADER_DEVICE(vkResetFences);
    PNANOVDB_VK_LOADER_DEVICE(vkResetCommandBuffer);

    PNANOVDB_VK_LOADER_DEVICE(vkBeginCommandBuffer);
    PNANOVDB_VK_LOADER_DEVICE(vkQueueWaitIdle);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateImage);
    PNANOVDB_VK_LOADER_DEVICE(vkGetImageMemoryRequirements);
    PNANOVDB_VK_LOADER_DEVICE(vkAllocateMemory);

    PNANOVDB_VK_LOADER_DEVICE(vkBindImageMemory);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateImageView);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyImageView);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyImage);
    PNANOVDB_VK_LOADER_DEVICE(vkFreeMemory);

    PNANOVDB_VK_LOADER_DEVICE(vkCreateBuffer);
    PNANOVDB_VK_LOADER_DEVICE(vkGetBufferMemoryRequirements);
    PNANOVDB_VK_LOADER_DEVICE(vkBindBufferMemory);
    PNANOVDB_VK_LOADER_DEVICE(vkMapMemory);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyBuffer);

    PNANOVDB_VK_LOADER_DEVICE(vkCmdPipelineBarrier);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdCopyBuffer);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateShaderModule);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyPipeline);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyShaderModule);

    PNANOVDB_VK_LOADER_DEVICE(vkCreateDescriptorSetLayout);
    PNANOVDB_VK_LOADER_DEVICE(vkCreatePipelineLayout);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyPipelineLayout);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyDescriptorSetLayout);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateDescriptorPool);

    PNANOVDB_VK_LOADER_DEVICE(vkAllocateDescriptorSets);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyDescriptorPool);
    PNANOVDB_VK_LOADER_DEVICE(vkUpdateDescriptorSets);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdBindDescriptorSets);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdBindPipeline);

    PNANOVDB_VK_LOADER_DEVICE(vkResetDescriptorPool);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateBufferView);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyBufferView);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateComputePipelines);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateSampler);

    PNANOVDB_VK_LOADER_DEVICE(vkDestroySampler);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdDispatch);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdDispatchIndirect);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateQueryPool);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroyQueryPool);

    PNANOVDB_VK_LOADER_DEVICE(vkCmdResetQueryPool);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdWriteTimestamp);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdCopyQueryPoolResults);
    PNANOVDB_VK_LOADER_DEVICE(vkGetImageSubresourceLayout);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdCopyImage);

    PNANOVDB_VK_LOADER_DEVICE(vkCmdCopyBufferToImage);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdCopyImageToBuffer);
    PNANOVDB_VK_LOADER_DEVICE(vkCmdPushConstants);
#if defined(_WIN32)
    PNANOVDB_VK_LOADER_DEVICE(vkGetMemoryWin32HandleKHR);
    PNANOVDB_VK_LOADER_DEVICE(vkGetSemaphoreWin32HandleKHR);
#else
    PNANOVDB_VK_LOADER_DEVICE(vkGetMemoryFdKHR);
    PNANOVDB_VK_LOADER_DEVICE(vkGetSemaphoreFdKHR);
#endif

    // device surface extensions
    PNANOVDB_VK_LOADER_DEVICE(vkQueuePresentKHR);
    PNANOVDB_VK_LOADER_DEVICE(vkCreateSwapchainKHR);
    PNANOVDB_VK_LOADER_DEVICE(vkGetSwapchainImagesKHR);
    PNANOVDB_VK_LOADER_DEVICE(vkDestroySwapchainKHR);
    PNANOVDB_VK_LOADER_DEVICE(vkAcquireNextImageKHR);
}
