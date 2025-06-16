// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   DeviceUtilsVulkan.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#include "CommonVulkan.h"

namespace pnanovdb_vulkan
{
    /// FormatConverter

    struct FormatConverterElement
    {
        pnanovdb_compute_format_t format_pnanovdb;
        VkFormat format_vulkan;
        pnanovdb_uint32_t size_in_bytes;
    };

    struct FormatConverter
    {
        FormatConverterElement formats_pnanovdb[PNANOVDB_COMPUTE_FORMAT_COUNT] = { { PNANOVDB_COMPUTE_FORMAT_UNKNOWN, VK_FORMAT_UNDEFINED , 0u } };
        FormatConverterElement formats_vulkan[PNANOVDB_COMPUTE_FORMAT_COUNT] = { { PNANOVDB_COMPUTE_FORMAT_UNKNOWN, VK_FORMAT_UNDEFINED , 0u } };
    };

    void formatConverter_placeElement(FormatConverter* ptr, pnanovdb_compute_format_t format_pnanovdb, VkFormat format_vulkan, pnanovdb_uint32_t size_in_bytes)
    {
        FormatConverterElement e = { format_pnanovdb, format_vulkan, size_in_bytes };
        ptr->formats_pnanovdb[format_pnanovdb] = e;
        ptr->formats_vulkan[format_vulkan] = e;
    }

    FormatConverter* formatConverter_create()
    {
        auto ptr = new FormatConverter();

        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, VK_FORMAT_UNDEFINED, 1u);
        // 128-bit
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32G32B32A32_FLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, 16u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_UINT, 16u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SINT, 16u);
        // 64-bit
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 8u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_UINT, 8u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_SNORM, 8u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SINT, 8u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32G32_FLOAT, VK_FORMAT_R32G32_SFLOAT, 8u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_UINT, 8u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SINT, 8u);
        // 32-bit
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM_SRGB, VK_FORMAT_R8G8B8A8_SRGB, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_UINT, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_SNORM, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SINT, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_B8G8R8A8_UNORM_SRGB, VK_FORMAT_B8G8R8A8_SRGB, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UNORM, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_UINT, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_SNORM, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_SINT, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, 4u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R32_SINT, VK_FORMAT_R32_SINT, 4u);
        // 16-bit
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_UNORM, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_UINT, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_SNORM, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8G8_SINT, VK_FORMAT_R8G8_SINT, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16_FLOAT, VK_FORMAT_R16_SFLOAT, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16_UNORM, VK_FORMAT_R16_UNORM, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16_UINT, VK_FORMAT_R16_UINT, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16_SNORM, VK_FORMAT_R16_SNORM, 2u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R16_SINT, VK_FORMAT_R16_SINT, 2u);
        //8-bit
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, 1u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8_UINT, VK_FORMAT_R8_UINT, 1u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8_SNORM, VK_FORMAT_R8_SNORM, 1u);
        formatConverter_placeElement(ptr, PNANOVDB_COMPUTE_FORMAT_R8_SINT, VK_FORMAT_R8_SINT, 1u);

        return ptr;
    }

    void formatConverter_destroy(FormatConverter* ptr)
    {
        delete ptr;
    }

    VkFormat formatConverter_convertToVulkan(FormatConverter* ptr, pnanovdb_compute_format_t format)
    {
        return ptr->formats_pnanovdb[format].format_vulkan;
    }

    pnanovdb_compute_format_t formatConverter_convertToPnanovdb(FormatConverter* ptr, VkFormat format)
    {
        return ptr->formats_vulkan[format].format_pnanovdb;
    }

    pnanovdb_uint32_t formatConverter_getFormatSizeInBytes(FormatConverter* ptr, pnanovdb_compute_format_t format)
    {
        return ptr->formats_pnanovdb[format].size_in_bytes;
    }

    /// Utils

    void determineMatches(std::vector<const char*>& extensionsEnabled, std::vector<ExtensionRequest>& extensionsRequest, std::vector<VkExtensionProperties>& extensions)
    {
        for (pnanovdb_uint32_t idx = 0u; idx < extensions.size(); idx++)
        {
            auto& ext = extensions[idx];

            for (pnanovdb_uint32_t reqIdx = 0u; reqIdx < extensionsRequest.size(); reqIdx++)
            {
                auto& extReq = extensionsRequest[reqIdx];

                if (pnanovdb_reflect_string_compare(extReq.name, ext.extensionName) == 0)
                {
                    extensionsEnabled.push_back(extReq.name);
                    *extReq.pEnabled = PNANOVDB_TRUE;
                }
            }
        }
    }

    void selectInstanceExtensions(DeviceManager* ptr, std::vector<const char*>& instanceExtensionsEnabled)
    {
        std::vector<VkExtensionProperties> instanceExtensions;
        std::vector<ExtensionRequest> instanceExtensionsRequest;

        auto& enabledExt = ptr->enabledExtensions;

#define PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(X) \
    instanceExtensionsRequest.push_back({ X##_EXTENSION_NAME, &enabledExt.X })

        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_SURFACE);
#if defined(_WIN32)
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_WIN32_SURFACE);
#elif defined(__APPLE__)
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_EXT_METAL_SURFACE);
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_MVK_MACOS_SURFACE);
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_PORTABILITY_ENUMERATION);
#else
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_XLIB_SURFACE);
#endif
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2);
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES);
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_EXTERNAL_FENCE_CAPABILITIES);
        PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES);

#undef PNANOVDB_VULKAN_TRY_ENABLE_INSTANCE_EXTENSION

        // determine what requested extensions are supported
        {
            pnanovdb_uint32_t numInstanceExtensions = 0u;

            ptr->loader.vkEnumerateInstanceExtensionProperties(nullptr, &numInstanceExtensions, nullptr);

            instanceExtensions.reserve(numInstanceExtensions);
            instanceExtensions.resize(numInstanceExtensions);

            ptr->loader.vkEnumerateInstanceExtensionProperties(nullptr, &numInstanceExtensions, instanceExtensions.data());

            determineMatches(instanceExtensionsEnabled, instanceExtensionsRequest, instanceExtensions);
        }
    }

    void selectDeviceExtensions(Device* ptr, std::vector<const char*>& deviceExtensionsEnabled)
    {
        std::vector<VkExtensionProperties> deviceExtensions;
        std::vector<ExtensionRequest> deviceExtensionsRequest;

        auto& enabledExt = ptr->enabledExtensions;

#define PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(X) \
    deviceExtensionsRequest.push_back({ X##_EXTENSION_NAME, &enabledExt.X })

        PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(VK_KHR_SWAPCHAIN);

        PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(VK_KHR_EXTERNAL_MEMORY);
        PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(VK_KHR_EXTERNAL_SEMAPHORE);
#if defined(_WIN32)
        PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(VK_KHR_EXTERNAL_MEMORY_WIN32);
        PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(VK_KHR_EXTERNAL_SEMAPHORE_WIN32);
#else
        PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(VK_KHR_EXTERNAL_MEMORY_FD);
        PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION(VK_KHR_EXTERNAL_SEMAPHORE_FD);
#endif

#undef PNANOVDB_VULKAN_TRY_ENABLE_DEVICE_EXTENSION

        // determine what requested extensions are supported
        {
            pnanovdb_uint32_t numDeviceExtensions = 0u;
            ptr->deviceManager->loader.vkEnumerateDeviceExtensionProperties(ptr->physicalDevice, nullptr, &numDeviceExtensions, nullptr);

            deviceExtensions.reserve(numDeviceExtensions);
            deviceExtensions.resize(numDeviceExtensions);

            ptr->deviceManager->loader.vkEnumerateDeviceExtensionProperties(ptr->physicalDevice, nullptr, &numDeviceExtensions, deviceExtensions.data());

            determineMatches(deviceExtensionsEnabled, deviceExtensionsRequest, deviceExtensions);
        }
    }
}
