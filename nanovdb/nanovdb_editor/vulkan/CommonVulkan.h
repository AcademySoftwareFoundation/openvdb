
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   CommonVulkan.h

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#pragma once

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR 1
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_MACOS_MVK 1
#define VK_USE_PLATFORM_METAL_EXT 1
#else
#define VK_USE_PLATFORM_XLIB_KHR 1
#endif

#include <vulkan/vulkan.h>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#define PNANOVDB_SWAPCHAIN_DESC 1

#include "nanovdb_editor/putil/Compute.h"

#include "LoaderVulkan.h"

#include "DeviceVulkan.h"
