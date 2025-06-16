// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/raster/Raster.cpp

    \author Andrew Reidmeyer

    \brief
*/

#define PNANOVDB_BUF_BOUNDS_CHECK
#include "Common.h"
#include "nanovdb_editor/putil/ParallelPrimitives.h"
#include "nanovdb_editor/putil/ThreadPool.hpp"

#include "nanovdb/PNanoVDB2.h"

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <future>

namespace
{

enum shader
{
    scan1_max_slang,
    scan1_uint64_slang,
    scan1_slang,
    scan2_max_slang,
    scan2_uint64_slang,
    scan2_slang,
    scan3_max_slang,
    scan3_uint64_slang,
    scan3_slang,

    radix_sort_dual1_slang,
    radix_sort_dual2_slang,
    radix_sort_dual3_slang,
    radix_sort_dual4_slang,
    radix_sort1_slang,
    radix_sort2_slang,
    radix_sort3_slang,

    shader_count
};

static const char* s_shader_names[shader_count] = {
    "raster/scan1_max.slang",
    "raster/scan1_uint64.slang",
    "raster/scan1.slang",
    "raster/scan2_max.slang",
    "raster/scan2_uint64.slang",
    "raster/scan2.slang",
    "raster/scan3_max.slang",
    "raster/scan3_uint64.slang",
    "raster/scan3.slang",

    "raster/radix_sort_dual1.slang",
    "raster/radix_sort_dual2.slang",
    "raster/radix_sort_dual3.slang",
    "raster/radix_sort_dual4.slang",
    "raster/radix_sort1.slang",
    "raster/radix_sort2.slang",
    "raster/radix_sort3.slang"
};

struct parallel_primitives_context_t
{
    pnanovdb_shader_context_t* shader_ctx[shader_count];
};

PNANOVDB_CAST_PAIR(pnanovdb_parallel_primitives_context_t, parallel_primitives_context_t)

static pnanovdb_parallel_primitives_context_t* create_context(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue)
{
    parallel_primitives_context_t* ctx = new parallel_primitives_context_t();

    pnanovdb_compiler_settings_t compile_settings = {};
    pnanovdb_compiler_settings_init(&compile_settings);

    pnanovdb_util::ThreadPool pool;
    std::vector<std::future<bool>> futures;

    for (pnanovdb_uint32_t idx = 0u; idx < shader_count; idx++)
    {
        auto future = pool.enqueue([compute, queue, ctx, idx, &compile_settings]() -> bool
            {
                ctx->shader_ctx[idx] = compute->create_shader_context(s_shader_names[idx]);
                return compute->init_shader(compute, queue, ctx->shader_ctx[idx], &compile_settings) == PNANOVDB_TRUE;
            });
        futures.push_back(std::move(future));
    }

    for (auto& future : futures)
    {
        bool success = future.get();
        if (!success)
        {
            return nullptr;
        }
    }

    return cast(ctx);
}

static void destroy_context(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in)
{
    auto ctx = cast(context_in);

    for (pnanovdb_uint32_t idx = 0u; idx < shader_count; idx++)
    {
        compute->destroy_shader_context(compute, queue, ctx->shader_ctx[idx]);
    }

    delete ctx;
}

static void global_scan_generic(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_buffer_t* val_in,
    pnanovdb_compute_buffer_t* val_out,
    pnanovdb_uint64_t val_count,
    pnanovdb_bool_t scan_uint64,
    pnanovdb_bool_t scan_max,
    pnanovdb_uint32_t dispatch_count)
{
    auto ctx = cast(context_in);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_uint32_t scan1_generic = scan1_slang;
    pnanovdb_uint32_t scan2_generic = scan2_slang;
    pnanovdb_uint32_t scan3_generic = scan3_slang;
    if (scan_uint64)
    {
        scan1_generic = scan1_uint64_slang;
        scan2_generic = scan2_uint64_slang;
        scan3_generic = scan3_uint64_slang;
    }
    else if (scan_max)
    {
        scan1_generic = scan1_max_slang;
        scan2_generic = scan2_max_slang;
        scan3_generic = scan3_max_slang;
    }

    pnanovdb_uint32_t element_size = 4u;
    if (scan_uint64)
    {
        element_size = 8u;
    }

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    struct constants_t
    {
        pnanovdb_uint32_t val_count;
        pnanovdb_uint32_t pad0;
        pnanovdb_uint32_t pad1;
        pnanovdb_uint32_t pad2;
    };
    constants_t constants = {};
    constants.val_count = val_count;

    pnanovdb_uint32_t workgroup_count = (val_count + 1023u) / 1024u;

    // constants
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 0u;
    buf_desc.size_in_bytes = sizeof(constants_t);
    pnanovdb_compute_buffer_t* constant_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

    // copy constants
    void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
    memcpy(mapped_constants, &constants, sizeof(constants_t));
    compute_interface->unmap_buffer(context, constant_buffer);

    // reduce and reduce_scan buffers
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = element_size;
    buf_desc.size_in_bytes = workgroup_count * element_size;
    pnanovdb_compute_buffer_t* reduce_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* reduce_scan_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    pnanovdb_compute_buffer_transient_t* constant_transient =
        compute_interface->register_buffer_as_transient(context, constant_buffer);
    pnanovdb_compute_buffer_transient_t* val_in_transient =
        compute_interface->register_buffer_as_transient(context, val_in);
    pnanovdb_compute_buffer_transient_t* val_out_transient =
        compute_interface->register_buffer_as_transient(context, val_out);
    pnanovdb_compute_buffer_transient_t* reduce_transient =
        compute_interface->register_buffer_as_transient(context, reduce_buffer);
    pnanovdb_compute_buffer_transient_t* reduce_scan_transient =
        compute_interface->register_buffer_as_transient(context, reduce_scan_buffer);

    for (pnanovdb_uint32_t dispatch_idx = 0u; dispatch_idx < dispatch_count; dispatch_idx++)
    {
        // scan 1
        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = val_in_transient;
            resources[1u].buffer_transient = constant_transient;
            resources[2u].buffer_transient = reduce_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[scan1_generic],
                resources,
                workgroup_count, 1u, 1u,
                "scan1"
            );
        }
        // scan 2
        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = reduce_transient;
            resources[1u].buffer_transient = constant_transient;
            resources[2u].buffer_transient = reduce_scan_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[scan2_generic],
                resources,
                1u, 1u, 1u,
                "scan2"
            );
        }
        // scan 3
        {
            pnanovdb_compute_resource_t resources[4u] = {};
            resources[0u].buffer_transient = val_in_transient;
            resources[1u].buffer_transient = constant_transient;
            resources[2u].buffer_transient = reduce_scan_transient;
            resources[3u].buffer_transient = val_out_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[scan3_generic],
                resources,
                workgroup_count, 1u, 1u,
                "scan3"
            );
        }
    }

    compute_interface->destroy_buffer(context, constant_buffer);
    compute_interface->destroy_buffer(context, reduce_buffer);
    compute_interface->destroy_buffer(context, reduce_scan_buffer);
}

static void global_scan(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_buffer_t* val_in,
    pnanovdb_compute_buffer_t* val_out,
    pnanovdb_uint64_t val_count,
    pnanovdb_uint32_t dispatch_count)
{
    global_scan_generic(compute, queue, context_in, val_in, val_out, val_count, PNANOVDB_FALSE, PNANOVDB_FALSE, dispatch_count);
}

static void global_scan_uint64(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_buffer_t* val_in,
    pnanovdb_compute_buffer_t* val_out,
    pnanovdb_uint64_t val_count,
    pnanovdb_uint32_t dispatch_count)
{
    global_scan_generic(compute, queue, context_in, val_in, val_out, val_count, PNANOVDB_TRUE, PNANOVDB_FALSE, dispatch_count);
}

static void global_scan_max(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_buffer_t* val_in,
    pnanovdb_compute_buffer_t* val_out,
    pnanovdb_uint64_t val_count,
    pnanovdb_uint32_t dispatch_count)
{
    global_scan_generic(compute, queue, context_in, val_in, val_out, val_count, PNANOVDB_FALSE, PNANOVDB_TRUE, dispatch_count);
}

static int global_scan_array(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_array_t* val_in,
    pnanovdb_compute_array_t* val_out,
    pnanovdb_uint64_t val_count,
    pnanovdb_uint32_t dispatch_count)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    compute_gpu_array_t* val_in_gpu_array = gpu_array_create();
    compute_gpu_array_t* val_out_gpu_array = gpu_array_create();

    gpu_array_upload(compute, queue, val_in_gpu_array, val_in);
    gpu_array_alloc_device(compute, queue, val_out_gpu_array, val_out);

    global_scan(
        compute,
        queue,
        context_in,
        val_in_gpu_array->device_buffer,
        val_out_gpu_array->device_buffer,
        val_count,
        dispatch_count
    );

    gpu_array_readback(compute, queue, val_out_gpu_array, val_out);

    pnanovdb_uint64_t flushed_frame = 0llu;
    compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

    compute->device_interface.wait_idle(queue);

    gpu_array_map(compute, queue, val_out_gpu_array, val_out);

    gpu_array_destroy(compute, queue, val_in_gpu_array);
    gpu_array_destroy(compute, queue, val_out_gpu_array);

    return 0;
}

static void radix_sort(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_buffer_t* key_inout,
    pnanovdb_compute_buffer_t* val_inout,
    pnanovdb_uint64_t key_count,
    pnanovdb_uint32_t key_bit_count)
{
    auto ctx = cast(context_in);

    if (key_count == 0u)
    {
        return;
    }

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    struct constants_t
    {
        pnanovdb_uint32_t workgroup_count;
        pnanovdb_uint32_t pass_start;
        pnanovdb_uint32_t pass_mask;
        pnanovdb_uint32_t pass_bit_count;
        pnanovdb_uint32_t counter_count;
        pnanovdb_uint32_t key_bits_count;
        pnanovdb_uint32_t key_count;
        pnanovdb_uint32_t pad1;
    };
    constants_t constants = {};
    constants.workgroup_count = (key_count + 1023u) / 1024u;
    constants.pass_start = 0u;
    constants.pass_mask = 0x0F;
    constants.pass_bit_count = 4u;
    constants.counter_count = 16u * constants.workgroup_count;
    constants.key_bits_count = key_bit_count;
    constants.key_count = key_count;

    // counter buffers
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 65536u;
    while (buf_desc.size_in_bytes < constants.counter_count * 2u * 4u)
    {
        buf_desc.size_in_bytes *= 2u;
    }
    pnanovdb_compute_buffer_t* counters_a_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* counters_b_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    // tmp buffers
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 65536u;
    while (buf_desc.size_in_bytes < constants.key_count * 4u)
    {
        buf_desc.size_in_bytes *= 2u;
    }
    pnanovdb_compute_buffer_t* key_tmp_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* val_tmp_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    pnanovdb_compute_buffer_transient_t* key_transient =
        compute_interface->register_buffer_as_transient(context, key_inout);
    pnanovdb_compute_buffer_transient_t* val_transient =
        compute_interface->register_buffer_as_transient(context, val_inout);

    pnanovdb_compute_buffer_transient_t* counters_a_transient =
        compute_interface->register_buffer_as_transient(context, counters_a_buffer);
    pnanovdb_compute_buffer_transient_t* counters_b_transient =
        compute_interface->register_buffer_as_transient(context, counters_b_buffer);
    pnanovdb_compute_buffer_transient_t* key_tmp_transient =
        compute_interface->register_buffer_as_transient(context, key_tmp_buffer);
    pnanovdb_compute_buffer_transient_t* val_tmp_transient =
        compute_interface->register_buffer_as_transient(context, val_tmp_buffer);

    pnanovdb_compute_buffer_transient_t* counters4_a_transient =
        compute_interface->alias_buffer_transient(context, counters_a_transient, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 16u);
    pnanovdb_compute_buffer_transient_t* counters4_b_transient =
        compute_interface->alias_buffer_transient(context, counters_b_transient, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 16u);

    pnanovdb_compute_buffer_transient_t* key4_transient =
        compute_interface->alias_buffer_transient(context, key_transient, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 16u);
    pnanovdb_compute_buffer_transient_t* val4_transient =
        compute_interface->alias_buffer_transient(context, val_transient, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 16u);
    pnanovdb_compute_buffer_transient_t* key4_tmp_transient =
        compute_interface->alias_buffer_transient(context, key_tmp_transient, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 16u);
    pnanovdb_compute_buffer_transient_t* val4_tmp_transient =
        compute_interface->alias_buffer_transient(context, val_tmp_transient, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 16u);

    pnanovdb_uint32_t pass_count = 2u * ((constants.key_bits_count + 7u) / 8u);
    for (pnanovdb_uint32_t pass_id = 0u; pass_id < pass_count; pass_id++)
    {
        constants.pass_start = 4u * pass_id;
        constants.pass_bit_count = 0u;
        if (4u * pass_id < constants.key_bits_count)
        {
            constants.pass_bit_count = constants.key_bits_count - 4u * pass_id;
        }
        if (constants.pass_bit_count > 4u)
        {
            constants.pass_bit_count = 4u;
        }
        constants.pass_mask = (1u << constants.pass_bit_count) - 1u;

        // for shared memory reasons, must take a least one pass
        if (constants.pass_bit_count == 0u)
        {
            constants.pass_bit_count = 1u;
        }

        // constants
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 0u;
        buf_desc.size_in_bytes = sizeof(constants_t);
        pnanovdb_compute_buffer_t* constant_buffer =
            compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

        // copy constants
        void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
        memcpy(mapped_constants, &constants, sizeof(constants_t));
        compute_interface->unmap_buffer(context, constant_buffer);

        pnanovdb_compute_buffer_transient_t* constant_transient =
        compute_interface->register_buffer_as_transient(context, constant_buffer);

        // radix sort 1
        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = (pass_id & 1) == 0u ? key4_transient : key4_tmp_transient;
            resources[1u].buffer_transient = constant_transient;
            resources[2u].buffer_transient = counters_a_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[radix_sort1_slang],
                resources,
                constants.workgroup_count, 1u, 1u,
                "radix_sort1"
            );
        }
        // radix sort 2
        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = counters4_a_transient;
            resources[1u].buffer_transient = constant_transient;
            resources[2u].buffer_transient = counters4_b_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[radix_sort2_slang],
                resources,
                2u, 1u, 1u,
                "radix_sort2"
            );
        }
        // radix sort 3
        {
            pnanovdb_compute_resource_t resources[6u] = {};
            resources[0u].buffer_transient = (pass_id & 1) == 0u ? key4_transient : key4_tmp_transient;
            resources[1u].buffer_transient = (pass_id & 1) == 0u ? val4_transient : val4_tmp_transient;
            resources[2u].buffer_transient = counters_b_transient;
            resources[3u].buffer_transient = constant_transient;
            resources[4u].buffer_transient = (pass_id & 1) == 0u ? key_tmp_transient : key_transient;
            resources[5u].buffer_transient = (pass_id & 1) == 0u ? val_tmp_transient : val_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[radix_sort3_slang],
                resources,
                constants.workgroup_count, 1u, 1u,
                "radix_sort3"
            );
        }

        compute_interface->destroy_buffer(context, constant_buffer);
    }

    compute_interface->destroy_buffer(context, counters_a_buffer);
    compute_interface->destroy_buffer(context, counters_b_buffer);
    compute_interface->destroy_buffer(context, key_tmp_buffer);
    compute_interface->destroy_buffer(context, val_tmp_buffer);
}

static void radix_sort_dual_key(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_buffer_t* key_low_inout,
    pnanovdb_compute_buffer_t* key_high_inout,
    pnanovdb_compute_buffer_t* val_inout,
    pnanovdb_uint64_t key_count,
    pnanovdb_uint32_t key_low_bit_count,
    pnanovdb_uint32_t key_high_bit_count)
{
    auto ctx = cast(context_in);

    if (key_count == 0u)
    {
        return;
    }

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    struct constants_t
    {
        pnanovdb_uint32_t workgroup_count;
        pnanovdb_uint32_t key_count;
    };
    constants_t constants = {};
    constants.workgroup_count = (key_count + 1023u) / 1024u;
    constants.key_count = key_count;

    // constants
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 0u;
    buf_desc.size_in_bytes = sizeof(constants_t);
    pnanovdb_compute_buffer_t* constant_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

    // copy constants
    void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
    memcpy(mapped_constants, &constants, sizeof(constants_t));
    compute_interface->unmap_buffer(context, constant_buffer);

    pnanovdb_compute_buffer_transient_t* constant_transient =
    compute_interface->register_buffer_as_transient(context, constant_buffer);

    // tmp buffers
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 65536u;
    while (buf_desc.size_in_bytes < constants.key_count * 4u)
    {
        buf_desc.size_in_bytes *= 2u;
    }
    pnanovdb_compute_buffer_t* key_tmp_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* val_tmp_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* key_low_copy_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* key_high_copy_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* val_copy_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    pnanovdb_compute_buffer_transient_t* key_tmp_transient =
        compute_interface->register_buffer_as_transient(context, key_tmp_buffer);
    pnanovdb_compute_buffer_transient_t* val_tmp_transient =
        compute_interface->register_buffer_as_transient(context, val_tmp_buffer);
    pnanovdb_compute_buffer_transient_t* key_low_copy_transient =
        compute_interface->register_buffer_as_transient(context, key_low_copy_buffer);
    pnanovdb_compute_buffer_transient_t* key_high_copy_transient =
        compute_interface->register_buffer_as_transient(context, key_high_copy_buffer);
    pnanovdb_compute_buffer_transient_t* val_copy_transient =
        compute_interface->register_buffer_as_transient(context, val_copy_buffer);

    pnanovdb_compute_buffer_transient_t* key_low_transient =
        compute_interface->register_buffer_as_transient(context, key_low_inout);
    pnanovdb_compute_buffer_transient_t* key_high_transient =
        compute_interface->register_buffer_as_transient(context, key_high_inout);
    pnanovdb_compute_buffer_transient_t* val_transient =
        compute_interface->register_buffer_as_transient(context, val_inout);

    // generate key and val for first sort pass
    {
        pnanovdb_compute_resource_t resources[4u] = {};
        resources[0u].buffer_transient = key_low_transient;
        resources[1u].buffer_transient = constant_transient;
        resources[2u].buffer_transient = key_tmp_transient;
        resources[3u].buffer_transient = val_tmp_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[radix_sort_dual1_slang],
            resources,
            constants.workgroup_count, 1u, 1u,
            "radix_sort_dual1"
        );
    }

    radix_sort(
        compute,
        queue,
        context_in,
        key_tmp_buffer,
        val_tmp_buffer,
        key_count,
        key_low_bit_count
    );

    // gather key high to current sorted indices
    {
        pnanovdb_compute_resource_t resources[4u] = {};
        resources[0u].buffer_transient = key_high_transient;
        resources[1u].buffer_transient = val_tmp_transient;
        resources[2u].buffer_transient = constant_transient;
        resources[3u].buffer_transient = key_tmp_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[radix_sort_dual2_slang],
            resources,
            constants.workgroup_count, 1u, 1u,
            "radix_sort_dual2"
        );
    }

    radix_sort(
        compute,
        queue,
        context_in,
        key_tmp_buffer,
        val_tmp_buffer,
        key_count,
        key_high_bit_count
    );

    // gather values
    {
        pnanovdb_compute_resource_t resources[8u] = {};
        resources[0u].buffer_transient = val_tmp_transient;
        resources[1u].buffer_transient = key_low_transient;
        resources[2u].buffer_transient = key_high_transient;
        resources[3u].buffer_transient = val_transient;
        resources[4u].buffer_transient = constant_transient;
        resources[5u].buffer_transient = key_low_copy_transient;
        resources[6u].buffer_transient = key_high_copy_transient;
        resources[7u].buffer_transient = val_copy_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[radix_sort_dual3_slang],
            resources,
            constants.workgroup_count, 1u, 1u,
            "radix_sort_dual3"
        );
    }

    // copy values
    {
        pnanovdb_compute_resource_t resources[7u] = {};
        resources[0u].buffer_transient = key_low_copy_transient;
        resources[1u].buffer_transient = key_high_copy_transient;
        resources[2u].buffer_transient = val_copy_transient;
        resources[3u].buffer_transient = constant_transient;
        resources[4u].buffer_transient = key_low_transient;
        resources[5u].buffer_transient = key_high_transient;
        resources[6u].buffer_transient = val_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[radix_sort_dual4_slang],
            resources,
            constants.workgroup_count, 1u, 1u,
            "radix_sort_dual4"
        );
    }

    compute_interface->destroy_buffer(context, key_tmp_buffer);
    compute_interface->destroy_buffer(context, val_tmp_buffer);
    compute_interface->destroy_buffer(context, key_low_copy_buffer);
    compute_interface->destroy_buffer(context, key_high_copy_buffer);
    compute_interface->destroy_buffer(context, val_copy_buffer);

    compute_interface->destroy_buffer(context, constant_buffer);
}

static int radix_sort_array(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in,
    pnanovdb_compute_array_t* key_inout,
    pnanovdb_compute_array_t* val_inout,
    pnanovdb_uint64_t key_count,
    pnanovdb_uint32_t key_bit_count)
{
    auto ctx = cast(context_in);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    compute_gpu_array_t* key_gpu_array = gpu_array_create();
    compute_gpu_array_t* val_gpu_array = gpu_array_create();

    gpu_array_upload(compute, queue, key_gpu_array, key_inout);
    gpu_array_upload(compute, queue, val_gpu_array, val_inout);

    radix_sort(
        compute,
        queue,
        context_in,
        key_gpu_array->device_buffer,
        val_gpu_array->device_buffer,
        key_count,
        key_bit_count
    );

    gpu_array_readback(compute, queue, key_gpu_array, key_inout);
    gpu_array_readback(compute, queue, val_gpu_array, val_inout);

    pnanovdb_uint64_t flushed_frame = 0llu;
    compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

    compute->device_interface.wait_idle(queue);

    gpu_array_map(compute, queue, key_gpu_array, key_inout);
    gpu_array_map(compute, queue, val_gpu_array, val_inout);

    gpu_array_destroy(compute, queue, key_gpu_array);
    gpu_array_destroy(compute, queue, val_gpu_array);

    return 0;
}

static void test_radix_sort(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_parallel_primitives_context_t* context_in)
{
    // test radix sort
    pnanovdb_uint32_t element_count = ((rand() & 0xFFFF) | ((rand() & 0xFFFF) << 16u)) & 0x00FFFFFF;

    pnanovdb_compute_array_t* key_arr = compute->create_array(4u, element_count, nullptr);
    pnanovdb_compute_array_t* val_arr = compute->create_array(4u, element_count, nullptr);

    pnanovdb_uint64_t pre_checksum = 0u;
    pnanovdb_uint32_t* key_mapped = (pnanovdb_uint32_t*)compute->map_array(key_arr);
    pnanovdb_uint32_t* val_mapped = (pnanovdb_uint32_t*)compute->map_array(val_arr);
    for (pnanovdb_uint32_t idx = 0u; idx < element_count; idx++)
    {
        key_mapped[idx] = ((rand() & 0xFFFF) | ((rand() & 0xFFFF) << 16u));
        val_mapped[idx] = idx;

        pre_checksum += key_mapped[idx];
    }
    compute->unmap_array(key_arr);
    compute->unmap_array(val_arr);

    radix_sort_array(compute, queue, context_in, key_arr, val_arr, element_count, 32u);

    pnanovdb_uint32_t old_key = 0u;
    pnanovdb_uint32_t sort_fail_count = 0u;
    pnanovdb_uint64_t post_checksum = 0u;
    key_mapped = (pnanovdb_uint32_t*)compute->map_array(key_arr);
    val_mapped = (pnanovdb_uint32_t*)compute->map_array(val_arr);
    for (pnanovdb_uint32_t idx = 0u; idx < element_count; idx++)
    {
        if (idx < 32u)
        {
            printf("[%u] key(%u) val(%u)\n", idx, key_mapped[idx], val_mapped[idx]);
        }
        if (key_mapped[idx] < old_key)
        {
            sort_fail_count++;
        }
        post_checksum += key_mapped[idx];
        old_key = key_mapped[idx];
    }
    compute->unmap_array(key_arr);
    compute->unmap_array(val_arr);

    printf("radix_sort fail_count(%u of %u) pre_checksum(0x%llx) post_checksum(0x%llx)\n",
        sort_fail_count, element_count, (unsigned long long int)pre_checksum, (unsigned long long int)post_checksum);
}

}

pnanovdb_parallel_primitives_t* pnanovdb_get_parallel_primitives()
{
    static pnanovdb_parallel_primitives_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_parallel_primitives_t) };

    iface.create_context = create_context;
    iface.destroy_context = destroy_context;
    iface.global_scan = global_scan;
    iface.global_scan_uint64 = global_scan_uint64;
    iface.global_scan_max = global_scan_max;
    iface.radix_sort = radix_sort;
    iface.radix_sort_dual_key = radix_sort_dual_key;

    return &iface;
}
