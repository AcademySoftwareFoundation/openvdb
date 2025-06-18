// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   UploadBuffer.h

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#pragma once

#include "nanovdb_editor/putil/Compute.h"
#include <vector>

#define PNANOVDB_DISPATCH_BATCH_SIZE 32768u
//#define PNANOVDB_DISPATCH_BATCH_SIZE 256

struct pnanovdb_compute_dispatch_batch_t
{
    pnanovdb_compute_buffer_transient_t* global_transient = nullptr;
    pnanovdb_uint32_t block_idx_offset = 0u;
    pnanovdb_uint32_t block_count = 0u;
};

typedef std::vector<pnanovdb_compute_dispatch_batch_t> pnanovdb_compute_dispatch_batches_t;

PNANOVDB_INLINE void pnanovdb_compute_dispatch_batches_init_custom(
    pnanovdb_compute_dispatch_batches_t* ptr,
    pnanovdb_uint32_t total_block_count,
    pnanovdb_uint32_t batch_size)
{
    ptr->resize(0u);
    for (pnanovdb_uint32_t block_idx_offset = 0u; block_idx_offset < total_block_count; block_idx_offset += batch_size)
    {
        pnanovdb_compute_dispatch_batch_t batch = {};
        batch.global_transient = nullptr;
        batch.block_idx_offset = block_idx_offset;
        batch.block_count = total_block_count - block_idx_offset;
        if (batch.block_count > batch_size)
        {
            batch.block_count = batch_size;
        }
        ptr->push_back(batch);
    }
}

PNANOVDB_INLINE void pnanovdb_compute_dispatch_batches_init(
    pnanovdb_compute_dispatch_batches_t* ptr,
    pnanovdb_uint32_t total_block_count)
{
    pnanovdb_compute_dispatch_batches_init_custom(ptr, total_block_count, PNANOVDB_DISPATCH_BATCH_SIZE);
}

struct pnanovdb_compute_buffer_versioning_t
{
    pnanovdb_uint64_t mapped_idx = ~0llu;
    pnanovdb_uint64_t front_idx = ~0llu;
    std::vector<pnanovdb_uint64_t> recycle_fence_values;
};

PNANOVDB_INLINE pnanovdb_uint64_t pnanovdb_compute_buffer_versioning_map(
    pnanovdb_compute_buffer_versioning_t* ptr,
    pnanovdb_uint64_t last_fence_completed)
{
    pnanovdb_uint64_t index = ptr->front_idx + 1u;
    for (; index < ptr->recycle_fence_values.size(); index++)
    {
        if (ptr->recycle_fence_values[index] <= last_fence_completed)
        {
            break;
        }
    }
    if (index == ptr->recycle_fence_values.size())
    {
        for (index = 0; index < ptr->front_idx && index < ptr->recycle_fence_values.size(); index++)
        {
            if (ptr->recycle_fence_values[index] <= last_fence_completed)
            {
                break;
            }
        }
    }
    if (!(index < ptr->recycle_fence_values.size() && ptr->recycle_fence_values[index] <= last_fence_completed))
    {
        index = ptr->recycle_fence_values.size();
        ptr->recycle_fence_values.push_back(~0llu);
    }
    ptr->recycle_fence_values[index] = ~0llu;
    ptr->mapped_idx = index;
    return ptr->mapped_idx;
}

PNANOVDB_INLINE void pnanovdb_compute_buffer_versioning_unmap(
    pnanovdb_compute_buffer_versioning_t* ptr,
    pnanovdb_uint64_t next_fence_value)
{
    if (ptr->front_idx < ptr->recycle_fence_values.size())
    {
        ptr->recycle_fence_values[ptr->front_idx] = next_fence_value;
    }
    ptr->front_idx = ptr->mapped_idx;
}

struct pnanovdb_compute_upload_buffer_t
{
    pnanovdb_compute_interface_t* compute_interface = nullptr;
    pnanovdb_compute_buffer_t*(PNANOVDB_ABI* create_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_memory_type_t memory_type, const pnanovdb_compute_buffer_desc_t* desc, void* userdata) = nullptr;
    void(PNANOVDB_ABI* copy_buffer)(pnanovdb_compute_context_t* context, const pnanovdb_compute_copy_buffer_params_t* params, void* userdata) = nullptr;
    void* userdata = nullptr;
    pnanovdb_compute_buffer_usage_t usage = 0u;
    pnanovdb_compute_format_t format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    pnanovdb_uint32_t structure_stride = 0u;

    pnanovdb_compute_buffer_versioning_t versioning;
    std::vector<pnanovdb_compute_buffer_t*> buffers;
    std::vector<pnanovdb_uint64_t> buffer_sizes;

    pnanovdb_compute_buffer_t* device_buffer = nullptr;
    pnanovdb_uint64_t device_num_bytes = 0llu;

    pnanovdb_compute_buffer_transient_t* device_buffer_transient = nullptr;
    pnanovdb_uint64_t device_buffer_transient_frame = ~0llu;
};

PNANOVDB_INLINE void pnanovdb_compute_upload_buffer_init_custom(
    pnanovdb_compute_interface_t* compute_interface,
    pnanovdb_compute_context_t* context, pnanovdb_compute_upload_buffer_t* ptr,
    pnanovdb_compute_buffer_usage_t usage, pnanovdb_compute_format_t format, pnanovdb_uint32_t structure_stride,
    pnanovdb_compute_buffer_t*(PNANOVDB_ABI* create_buffer)(pnanovdb_compute_context_t* context, pnanovdb_compute_memory_type_t memory_type, const pnanovdb_compute_buffer_desc_t* desc, void* userdata),
    void(PNANOVDB_ABI* copy_buffer)(pnanovdb_compute_context_t* context, const pnanovdb_compute_copy_buffer_params_t* params, void* userdata),
    void* userdata
)
{
    ptr->compute_interface = compute_interface;
    ptr->create_buffer = create_buffer;
    ptr->copy_buffer = copy_buffer;
    ptr->userdata = userdata;

    ptr->usage = usage;
    ptr->format = format;
    ptr->structure_stride = structure_stride;
}

PNANOVDB_INLINE pnanovdb_compute_buffer_t* pnanovdb_compute_upload_buffer_create_buffer(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_memory_type_t memoryType,
    const pnanovdb_compute_buffer_desc_t* desc,
    void* userdata)
{
    pnanovdb_compute_upload_buffer_t* ptr = (pnanovdb_compute_upload_buffer_t*)userdata;
    return ptr->compute_interface->create_buffer(context, memoryType, desc);
}

PNANOVDB_INLINE void pnanovdb_compute_upload_buffer_copy_buffer(
    pnanovdb_compute_context_t* context,
    const pnanovdb_compute_copy_buffer_params_t* params,
    void* userdata)
{
    pnanovdb_compute_upload_buffer_t* ptr = (pnanovdb_compute_upload_buffer_t*)userdata;
    ptr->compute_interface->copy_buffer(context, params);
}

PNANOVDB_INLINE void pnanovdb_compute_upload_buffer_init(
    pnanovdb_compute_interface_t* compute_interface,
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_upload_buffer_t* ptr,
    pnanovdb_compute_buffer_usage_t usage,
    pnanovdb_compute_format_t format,
    pnanovdb_uint32_t structure_stride)
{
    pnanovdb_compute_upload_buffer_init_custom(
        compute_interface,
        context,
        ptr,
        usage,
        format,
        structure_stride,
        pnanovdb_compute_upload_buffer_create_buffer,
        pnanovdb_compute_upload_buffer_copy_buffer,
        ptr);
}

PNANOVDB_INLINE void pnanovdb_compute_upload_buffer_destroy(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_upload_buffer_t* ptr)
{
    for (pnanovdb_uint64_t idx = 0u; idx < ptr->buffers.size(); idx++)
    {
        if (ptr->buffers[idx])
        {
            ptr->compute_interface->destroy_buffer(context, ptr->buffers[idx]);
            ptr->buffers[idx] = nullptr;
        }
    }
    ptr->buffers.clear();
    ptr->buffer_sizes.clear();

    if (ptr->device_buffer)
    {
        ptr->compute_interface->destroy_buffer(context, ptr->device_buffer);
        ptr->device_buffer = nullptr;
    }
}

PNANOVDB_INLINE pnanovdb_uint64_t pnanovdb_compute_upload_buffer_compute_buffer_size(pnanovdb_uint64_t requested)
{
    pnanovdb_uint64_t buffer_size = 65536u;
    while (buffer_size < requested)
    {
        buffer_size *= 2u;
    }
    return buffer_size;
}

PNANOVDB_INLINE void* pnanovdb_compute_upload_buffer_map(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_upload_buffer_t* ptr,
    pnanovdb_uint64_t num_bytes)
{
    pnanovdb_compute_frame_info_t frame_info = {};
    ptr->compute_interface->get_frame_info(context, &frame_info);

    pnanovdb_uint64_t instance_idx = pnanovdb_compute_buffer_versioning_map(&ptr->versioning, frame_info.frame_local_completed);
    while (instance_idx >= ptr->buffers.size())
    {
        ptr->buffers.push_back(nullptr);
        ptr->buffer_sizes.push_back(0llu);
    }

    if (ptr->buffers[instance_idx] && ptr->buffer_sizes[instance_idx] < num_bytes)
    {
        ptr->compute_interface->destroy_buffer(context, ptr->buffers[instance_idx]);
        ptr->buffers[instance_idx] = nullptr;
    }

    if (!ptr->buffers[instance_idx])
    {
        pnanovdb_compute_buffer_desc_t buf_desc = {};
        buf_desc.format = ptr->format;
        buf_desc.usage = ptr->usage;
        buf_desc.structure_stride = ptr->structure_stride;
        buf_desc.size_in_bytes = pnanovdb_compute_upload_buffer_compute_buffer_size(num_bytes);

        ptr->buffer_sizes[instance_idx] = buf_desc.size_in_bytes;
        ptr->buffers[instance_idx] = ptr->compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);
    }

    return ptr->compute_interface->map_buffer(context, ptr->buffers[instance_idx]);
}

PNANOVDB_INLINE pnanovdb_compute_buffer_transient_t* pnanovdb_compute_upload_buffer_unmap(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_upload_buffer_t* ptr)
{
    pnanovdb_compute_frame_info_t frame_info = {};
    ptr->compute_interface->get_frame_info(context, &frame_info);

    ptr->compute_interface->unmap_buffer(context, ptr->buffers[ptr->versioning.mapped_idx]);

    pnanovdb_compute_buffer_versioning_unmap(&ptr->versioning, frame_info.frame_local_current);

    return ptr->compute_interface->register_buffer_as_transient(context, ptr->buffers[ptr->versioning.front_idx]);
}

struct pnanovdb_compute_upload_buffer_copy_range_t
{
    pnanovdb_uint64_t offset;
    pnanovdb_uint64_t num_bytes;
};

PNANOVDB_INLINE pnanovdb_compute_buffer_transient_t* pnanovdb_compute_upload_buffer_get_device(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_upload_buffer_t* ptr,
    pnanovdb_uint64_t num_bytes)
{
    pnanovdb_compute_frame_info_t frame_info = {};
    ptr->compute_interface->get_frame_info(context, &frame_info);

    pnanovdb_uint64_t src_num_bytes = pnanovdb_compute_upload_buffer_compute_buffer_size(num_bytes);

    if (ptr->device_buffer && ptr->device_num_bytes < src_num_bytes)
    {
        ptr->compute_interface->destroy_buffer(context, ptr->device_buffer);
        ptr->device_buffer = nullptr;
        ptr->device_num_bytes = 0llu;
        ptr->device_buffer_transient = nullptr;
        ptr->device_buffer_transient_frame = ~0llu;
    }
    if (!ptr->device_buffer)
    {
        pnanovdb_compute_buffer_desc_t buf_desc = {};
        buf_desc.format = ptr->format;
        buf_desc.usage = ptr->usage | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
        buf_desc.structure_stride = ptr->structure_stride;
        buf_desc.size_in_bytes = src_num_bytes;

        ptr->device_buffer = ptr->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc, ptr->userdata);
        ptr->device_num_bytes = src_num_bytes;
        ptr->device_buffer_transient = nullptr;
        ptr->device_buffer_transient_frame = ~0llu;
    }

    if (ptr->device_buffer_transient &&
        ptr->device_buffer_transient_frame == frame_info.frame_local_current)
    {
        return ptr->device_buffer_transient;
    }
    ptr->device_buffer_transient = ptr->compute_interface->register_buffer_as_transient(context, ptr->device_buffer);
    ptr->device_buffer_transient_frame = frame_info.frame_local_current;
    return ptr->device_buffer_transient;
}

PNANOVDB_INLINE pnanovdb_compute_buffer_transient_t* pnanovdb_compute_upload_buffer_unmap_device_n(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_upload_buffer_t* ptr,
    pnanovdb_compute_upload_buffer_copy_range_t* copy_ranges,
    pnanovdb_uint64_t copy_range_count,
    const char* debug_name)
{
    pnanovdb_compute_buffer_transient_t* src = pnanovdb_compute_upload_buffer_unmap(context, ptr);

    pnanovdb_uint64_t src_num_bytes = ptr->buffer_sizes[ptr->versioning.front_idx];

    pnanovdb_compute_buffer_transient_t* dst = pnanovdb_compute_upload_buffer_get_device(context, ptr, src_num_bytes);

    pnanovdb_uint32_t active_copy_count = 0u;
    for (pnanovdb_uint64_t copy_range_idx = 0u; copy_range_idx < copy_range_count; copy_range_idx++)
    {
        pnanovdb_compute_copy_buffer_params_t copy_params = {};
        copy_params.src_offset = copy_ranges[copy_range_idx].offset;
        copy_params.dst_offset = copy_ranges[copy_range_idx].offset;
        copy_params.num_bytes = copy_ranges[copy_range_idx].num_bytes;
        copy_params.src = src;
        copy_params.dst = dst;

        copy_params.debug_label = debug_name ? debug_name : "upload_buffer_unmap_device";

        if (copy_params.num_bytes > 0u)
        {
            ptr->copy_buffer(context, &copy_params, ptr->userdata);
            active_copy_count++;
        }
    }
    // this ensures proper barriers
    if (active_copy_count == 0u)
    {
        pnanovdb_compute_copy_buffer_params_t copy_params = {};
        copy_params.src_offset = 0llu;
        copy_params.dst_offset = 0llu;
        copy_params.num_bytes = 0llu;
        copy_params.src = src;
        copy_params.dst = dst;

        copy_params.debug_label = debug_name ? debug_name : "upload_buffer_unmap_device";

        ptr->copy_buffer(context, &copy_params, ptr->userdata);
    }

    return dst;
}

PNANOVDB_INLINE pnanovdb_compute_buffer_transient_t* pnanovdb_compute_upload_buffer_unmap_device(
    pnanovdb_compute_context_t* context,
    pnanovdb_compute_upload_buffer_t* ptr,
    pnanovdb_uint64_t offset,
    pnanovdb_uint64_t num_bytes,
    const char* debug_name)
{
    pnanovdb_compute_upload_buffer_copy_range_t copy_range = { offset, num_bytes };
    return pnanovdb_compute_upload_buffer_unmap_device_n(context, ptr, &copy_range, 1u, debug_name);
}
