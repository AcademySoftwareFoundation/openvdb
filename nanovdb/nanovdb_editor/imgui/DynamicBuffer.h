// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   DynamicBuffer.h

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#pragma once

#include "nanovdb_editor/putil/Compute.h"

struct pnanovdb_compute_dynamic_buffer_t
{
    pnanovdb_compute_interface_t* compute_interface = nullptr;
    pnanovdb_compute_buffer_usage_t usage = 0u;
    pnanovdb_compute_format_t format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    pnanovdb_uint32_t structure_stride = 0u;

    pnanovdb_compute_buffer_t* device_buffer = nullptr;
    pnanovdb_uint64_t device_num_bytes = 0llu;

    pnanovdb_compute_buffer_transient_t* transient_buffer = nullptr;
    pnanovdb_uint64_t transient_frame = ~0llu;
};

PNANOVDB_INLINE void pnanovdb_compute_dynamic_buffer_init(
    pnanovdb_compute_interface_t* compute_interface,
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_dynamic_buffer_t* ptr,
    pnanovdb_compute_buffer_usage_t usage,
    pnanovdb_compute_format_t format,
    pnanovdb_uint32_t structure_stride)
{
    ptr->compute_interface = compute_interface;

    ptr->usage = usage;
    ptr->format = format;
    ptr->structure_stride = structure_stride;
}

PNANOVDB_INLINE void pnanovdb_compute_dynamic_buffer_destroy(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_dynamic_buffer_t* ptr)
{
    if (ptr->device_buffer)
    {
        ptr->compute_interface->destroy_buffer(context, ptr->device_buffer);
        ptr->device_buffer = nullptr;
    }
}

PNANOVDB_INLINE void pnanovdb_compute_dynamic_buffer_resize(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_dynamic_buffer_t* ptr,
    pnanovdb_uint64_t num_bytes)
{
    if (ptr->device_buffer && ptr->device_num_bytes < num_bytes)
    {
        ptr->compute_interface->destroy_buffer(context, ptr->device_buffer);
        ptr->device_buffer = nullptr;
        ptr->device_num_bytes = 0llu;
        ptr->transient_buffer = nullptr;
        ptr->transient_frame = ~0llu;
    }
    if (!ptr->device_buffer)
    {
        pnanovdb_compute_buffer_desc_t bufDesc = {};
        bufDesc.format = ptr->format;
        bufDesc.usage = ptr->usage;
        bufDesc.structure_stride = ptr->structure_stride;
        bufDesc.size_in_bytes = 65536u;
        while (bufDesc.size_in_bytes < num_bytes)
        {
            bufDesc.size_in_bytes *= 2u;
        }

        ptr->device_num_bytes = bufDesc.size_in_bytes;
        ptr->device_buffer = ptr->compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &bufDesc);
        ptr->transient_buffer = nullptr;
        ptr->transient_frame = ~0llu;
    }
}

PNANOVDB_INLINE pnanovdb_compute_buffer_transient_t* pnanovdb_compute_dynamic_buffer_get_transient(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_dynamic_buffer_t* ptr)
{
    pnanovdb_compute_frame_info_t frame_info = {};
    ptr->compute_interface->get_frame_info(context, &frame_info);

    if (ptr->transient_buffer &&
        ptr->transient_frame == frame_info.frame_local_current)
    {
        return ptr->transient_buffer;
    }
    if (ptr->device_buffer)
    {
        ptr->transient_buffer = ptr->compute_interface->register_buffer_as_transient(context, ptr->device_buffer);
        ptr->transient_frame = frame_info.frame_local_current;
    }
    return ptr->transient_buffer;
}
