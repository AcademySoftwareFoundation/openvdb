// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/raster/Common.h

    \author Andrew Reidmeyer

    \brief
*/

#pragma once

#include "nanovdb_editor/putil/Compute.h"

#include <math.h>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <time.h>
#endif

PNANOVDB_INLINE void timestamp_capture(pnanovdb_uint64_t* ptr)
{
#if defined(_WIN32)
    LARGE_INTEGER tmpCpuTime = {};
    QueryPerformanceCounter(&tmpCpuTime);
    (*ptr) = tmpCpuTime.QuadPart;
#else
    timespec timeValue = {};
    clock_gettime(CLOCK_MONOTONIC, &timeValue);
    (*ptr) = 1E9 * pnanovdb_uint64_t(timeValue.tv_sec) + pnanovdb_uint64_t(timeValue.tv_nsec);
#endif
}
PNANOVDB_INLINE pnanovdb_uint64_t timestamp_frequency()
{
#if defined(_WIN32)
    LARGE_INTEGER tmpCpuFreq = {};
    QueryPerformanceFrequency(&tmpCpuFreq);
    return tmpCpuFreq.QuadPart;
#else
    return 1E9;
#endif
}
PNANOVDB_INLINE float timestamp_diff(pnanovdb_uint64_t begin, pnanovdb_uint64_t end, pnanovdb_uint64_t freq)
{
    return (float)(((double)(end - begin) / (double)(freq)));
}

struct compute_gpu_array_t
{
    pnanovdb_compute_buffer_t* upload_buffer;
    pnanovdb_compute_buffer_t* device_buffer;
    pnanovdb_compute_buffer_t* readback_buffer;
};

static compute_gpu_array_t* gpu_array_create()
{
    compute_gpu_array_t* ptr = new compute_gpu_array_t();
    ptr->upload_buffer = nullptr;
    ptr->device_buffer = nullptr;
    ptr->readback_buffer = nullptr;
    return ptr;
}

static void gpu_array_destroy(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    compute_gpu_array_t* ptr)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    if (ptr->upload_buffer)
    {
        compute_interface->destroy_buffer(context, ptr->upload_buffer);
        ptr->upload_buffer = nullptr;
    }
    if (ptr->device_buffer)
    {
        compute_interface->destroy_buffer(context, ptr->device_buffer);
        ptr->device_buffer = nullptr;
    }
    if (ptr->readback_buffer)
    {
        compute_interface->destroy_buffer(context, ptr->readback_buffer);
        ptr->readback_buffer = nullptr;
    }
    delete ptr;
}

static void gpu_array_alloc_device(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    compute_gpu_array_t* ptr,
    pnanovdb_compute_array_t* arr)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_desc_t buf_desc = {};
    if (!ptr->device_buffer)
    {
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 4u;
        buf_desc.size_in_bytes = arr->element_count * arr->element_size;
        ptr->device_buffer = compute_interface->create_buffer(
            context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    }
}

static void gpu_array_upload(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    compute_gpu_array_t* ptr,
    pnanovdb_compute_array_t* arr)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_desc_t buf_desc = {};
    if (!ptr->upload_buffer)
    {
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 0u;
        buf_desc.size_in_bytes = arr->element_count * arr->element_size;
        ptr->upload_buffer = compute_interface->create_buffer(
            context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);
    }
    gpu_array_alloc_device(compute, queue, ptr, arr);

    // copy arr
    void* mapped_arr = compute_interface->map_buffer(context, ptr->upload_buffer);
    memcpy(mapped_arr, arr->data, arr->element_count * arr->element_size);
    compute_interface->unmap_buffer(context, ptr->upload_buffer);

    // upload arr
    pnanovdb_compute_copy_buffer_params_t copy_params = {};
    copy_params.num_bytes = arr->element_count * arr->element_size;
    copy_params.src = compute_interface->register_buffer_as_transient(context, ptr->upload_buffer);
    copy_params.dst = compute_interface->register_buffer_as_transient(context, ptr->device_buffer);
    copy_params.debug_label = "gpu_array_upload";
    compute_interface->copy_buffer(context, &copy_params);
}

static void gpu_array_copy(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    compute_gpu_array_t* ptr,
    pnanovdb_compute_buffer_t* src_buffer,
    pnanovdb_uint64_t src_offset,
    pnanovdb_uint64_t num_bytes)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    // copy src to device array
    pnanovdb_compute_copy_buffer_params_t copy_params = {};
    copy_params.num_bytes = num_bytes;
    copy_params.src_offset = src_offset;
    copy_params.src = compute_interface->register_buffer_as_transient(context, src_buffer);
    copy_params.dst = compute_interface->register_buffer_as_transient(context, ptr->device_buffer);
    copy_params.debug_label = "gpu_array_copy";
    compute_interface->copy_buffer(context, &copy_params);
}

static void gpu_array_readback(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    compute_gpu_array_t* ptr,
    pnanovdb_compute_array_t* arr)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_desc_t buf_desc = {};
    if (!ptr->readback_buffer)
    {
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 0u;
        buf_desc.size_in_bytes = arr->element_count * arr->element_size;
        ptr->readback_buffer = compute_interface->create_buffer(
            context, PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK, &buf_desc);
    }

    // readback arr
    pnanovdb_compute_copy_buffer_params_t copy_params = {};
    copy_params.num_bytes = arr->element_count * arr->element_size;
    copy_params.src = compute_interface->register_buffer_as_transient(context, ptr->device_buffer);
    copy_params.dst = compute_interface->register_buffer_as_transient(context, ptr->readback_buffer);
    copy_params.debug_label = "gpu_array_readback";
    compute_interface->copy_buffer(context, &copy_params);
}

static void gpu_array_map(
    const pnanovdb_compute_t* compute,
    const pnanovdb_compute_queue_t* queue,
    compute_gpu_array_t* ptr,
    pnanovdb_compute_array_t* arr)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    // copy arr
    void* mapped_arr = compute_interface->map_buffer(context, ptr->readback_buffer);
        memcpy(arr->data, mapped_arr, arr->element_count * arr->element_size);
    compute_interface->unmap_buffer(context, ptr->readback_buffer);
}
