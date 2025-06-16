// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/raster/Raster.cpp

    \author Andrew Reidmeyer

    \brief
*/

#define PNANOVDB_BUF_BOUNDS_CHECK
#include "Raster.h"

#include "nanovdb/PNanoVDB2.h"
#include "nanovdb_editor/putil/WorkerThread.hpp"

#include <stdlib.h>
#include <math.h>
#include <vector>

//#include "Node2Cpp.h"

//#define PNANOVDB_RASTER_VALIDATE 1

namespace pnanovdb_raster
{

static void raster_profiler_report(void* userdata, pnanovdb_uint64_t capture_id, pnanovdb_uint32_t num_entries, pnanovdb_compute_profiler_entry_t* entries)
{
    printf("raster_points() profiler results capture_id(%llu):\n", (unsigned long long int)capture_id);
    for (pnanovdb_uint32_t idx = 0u; idx < num_entries; idx++)
    {
        printf("[%d] name(%s) cpu_ms(%f) gpu_ms(%f)\n",
            idx, entries[idx].label, 1000.f * entries[idx].cpu_delta_time, 1000.f * entries[idx].gpu_delta_time);
    }
}

static int point_frag_alloc(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    raster_context_t* ctx,
    pnanovdb_compute_buffer_t* positions_in,
    pnanovdb_compute_buffer_t* ijk_out,
    pnanovdb_uint64_t point_count,
    float voxel_size,
    pnanovdb_uint32_t dispatch_count)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    struct constants_t
    {
        pnanovdb_uint32_t point_count;
        float voxel_size;
        float voxel_size_inv;
        float pad3;
    };
    constants_t constants = {};
    constants.point_count = (pnanovdb_uint32_t)point_count;
    constants.voxel_size = voxel_size;
    constants.voxel_size_inv = 1.f / voxel_size;

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    // positions to ijk
    {
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
        pnanovdb_compute_buffer_transient_t* positions_transient =
            compute_interface->register_buffer_as_transient(context, positions_in);
        pnanovdb_compute_buffer_transient_t* ijk_transient =
            compute_interface->register_buffer_as_transient(context, ijk_out);

        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = constant_transient;
            resources[1u].buffer_transient = positions_transient;
            resources[2u].buffer_transient = ijk_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[point_frag_alloc_slang],
                resources,
                (constants.point_count + 255u) / 256u, 1u, 1u,
                "point_to_ijks"
            );
        }

        compute_interface->destroy_buffer(context, constant_buffer);
    }

    return 0;
}

struct raster_constants_t
{
    pnanovdb_uint32_t prim_batch_size;
    pnanovdb_uint32_t ijk_batch_size;
    pnanovdb_uint64_t fragment_window_min;
    pnanovdb_uint64_t fragment_window_max;
    pnanovdb_uint32_t global_prim_begin_idx;
    pnanovdb_uint32_t global_prim_count;
    float voxel_size;
    float voxel_size_inv;
};

static pnanovdb_compute_buffer_t* raster_get_constant_buffer(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_fanout_state_t* state,
    raster_constants_t* constants_out)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    raster_constants_t constants = {};
    constants.prim_batch_size = state->prim_batch_size;
    constants.ijk_batch_size = state->ijk_batch_size;
    constants.fragment_window_min = (pnanovdb_uint64_t)state->ijk_batch_idx * (pnanovdb_uint64_t)state->ijk_batch_size;
    constants.fragment_window_max = constants.fragment_window_min + constants.ijk_batch_size;
    constants.global_prim_begin_idx = state->prim_batch_idx * state->prim_batch_size;
    constants.global_prim_count = state->prim_count,
    constants.voxel_size = state->voxel_size;
    constants.voxel_size_inv = 1.f / state->voxel_size;

    //printf("window(%d:%d) prim_offset(%d) prim_count(%d)\n",
    //    constants.fragment_window_min, constants.fragment_window_max,
    //    constants.global_prim_begin_idx, constants.global_prim_count);

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    // constants
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 0u;
    buf_desc.size_in_bytes = sizeof(raster_constants_t);
    pnanovdb_compute_buffer_t* constant_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

    // copy constants
    void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
    memcpy(mapped_constants, &constants, sizeof(raster_constants_t));
    compute_interface->unmap_buffer(context, constant_buffer);

    if (constants_out)
    {
        *constants_out = constants;
    }
    return constant_buffer;
}

static int gaussian_prim(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    raster_context_t* ctx,
    pnanovdb_grid_build_fanout_state_t* state,
    pnanovdb_compute_buffer_t* positions_in,
    pnanovdb_compute_buffer_t* quaternions_in,
    pnanovdb_compute_buffer_t* scales_in,
    pnanovdb_compute_buffer_t* opacities_in,
    pnanovdb_uint32_t dispatch_count)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    raster_constants_t constants = {};
    pnanovdb_compute_buffer_t* constant_buffer = raster_get_constant_buffer(compute, queue, state, &constants);

    pnanovdb_compute_buffer_transient_t* constant_transient =
        compute_interface->register_buffer_as_transient(context, constant_buffer);

    pnanovdb_compute_buffer_transient_t* positions_transient =
        compute_interface->register_buffer_as_transient(context, positions_in);
    pnanovdb_compute_buffer_transient_t* quaternions_transient =
        compute_interface->register_buffer_as_transient(context, quaternions_in);
    pnanovdb_compute_buffer_transient_t* scales_transient =
        compute_interface->register_buffer_as_transient(context, scales_in);
    pnanovdb_compute_buffer_transient_t* opacities_transient =
        compute_interface->register_buffer_as_transient(context, opacities_in);

    // per prim buffers
    pnanovdb_compute_buffer_transient_t* fragment_counts_transient =
        compute_interface->register_buffer_as_transient(context, state->fragment_counts_buffer);
    pnanovdb_compute_buffer_transient_t* prim_bbox_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_bbox_buffer);

    // gaussian prim shader
    {
        pnanovdb_compute_resource_t resources[7u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = positions_transient;
        resources[2u].buffer_transient = quaternions_transient;
        resources[3u].buffer_transient = scales_transient;
        resources[4u].buffer_transient = opacities_transient;
        resources[5u].buffer_transient = fragment_counts_transient;
        resources[6u].buffer_transient = prim_bbox_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_prim_slang],
            resources,
            (constants.prim_batch_size + 255u) / 256u, 1u, 1u,
            "gaussian_prim"
        );
    }

    compute_interface->destroy_buffer(context, constant_buffer);

    return 0;
}

static int gaussian_frag_alloc(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    raster_context_t* ctx,
    pnanovdb_grid_build_fanout_state_t* state,
    pnanovdb_compute_buffer_t* positions_in,
    pnanovdb_compute_buffer_t* quaternions_in,
    pnanovdb_compute_buffer_t* scales_in,
    pnanovdb_compute_buffer_t* opacities_in,
    pnanovdb_compute_buffer_t* ijk_out,
    pnanovdb_uint32_t dispatch_count)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    raster_constants_t constants = {};
    pnanovdb_compute_buffer_t* constant_buffer = raster_get_constant_buffer(compute, queue, state, &constants);

    pnanovdb_compute_buffer_transient_t* constant_transient =
        compute_interface->register_buffer_as_transient(context, constant_buffer);

    // external buffers
    pnanovdb_compute_buffer_transient_t* positions_transient =
        compute_interface->register_buffer_as_transient(context, positions_in);
    pnanovdb_compute_buffer_transient_t* quaternions_transient =
        compute_interface->register_buffer_as_transient(context, quaternions_in);
    pnanovdb_compute_buffer_transient_t* scales_transient =
        compute_interface->register_buffer_as_transient(context, scales_in);
    pnanovdb_compute_buffer_transient_t* opacities_transient =
        compute_interface->register_buffer_as_transient(context, opacities_in);
    pnanovdb_compute_buffer_transient_t* ijk_transient =
        compute_interface->register_buffer_as_transient(context, ijk_out);

    // state buffers
    pnanovdb_compute_buffer_transient_t* prim_bbox_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_bbox_buffer);
    pnanovdb_compute_buffer_transient_t* prim_idxs_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_idxs_buffer);
    pnanovdb_compute_buffer_transient_t* prim_raster_idxs_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_raster_idxs_buffer);

    // gaussian frag shader
    {
        pnanovdb_compute_resource_t resources[9u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = positions_transient;
        resources[2u].buffer_transient = quaternions_transient;
        resources[3u].buffer_transient = scales_transient;
        resources[4u].buffer_transient = opacities_transient;
        resources[5u].buffer_transient = prim_idxs_transient;
        resources[6u].buffer_transient = prim_raster_idxs_transient;
        resources[7u].buffer_transient = prim_bbox_transient;
        resources[8u].buffer_transient = ijk_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_frag_alloc_slang],
            resources,
            (constants.ijk_batch_size + 255u) / 256u, 1u, 1u,
            "gaussian_frag_alloc"
        );
    }

    compute_interface->destroy_buffer(context, constant_buffer);

    return 0;
}

static int gaussian_frag_color(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    raster_context_t* ctx,
    pnanovdb_grid_build_fanout_state_t* state,
    pnanovdb_compute_buffer_t* positions_in,
    pnanovdb_compute_buffer_t* quaternions_in,
    pnanovdb_compute_buffer_t* scales_in,
    pnanovdb_compute_buffer_t* opacities_in,
    pnanovdb_compute_buffer_t* colors_in,
    pnanovdb_compute_buffer_t* nanovdb_inout,
    pnanovdb_uint32_t dispatch_count)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    raster_constants_t constants = {};
    pnanovdb_compute_buffer_t* constant_buffer = raster_get_constant_buffer(compute, queue, state, &constants);

    pnanovdb_compute_buffer_transient_t* constant_transient =
        compute_interface->register_buffer_as_transient(context, constant_buffer);

    // external buffers
    pnanovdb_compute_buffer_transient_t* positions_transient =
        compute_interface->register_buffer_as_transient(context, positions_in);
    pnanovdb_compute_buffer_transient_t* quaternions_transient =
        compute_interface->register_buffer_as_transient(context, quaternions_in);
    pnanovdb_compute_buffer_transient_t* scales_transient =
        compute_interface->register_buffer_as_transient(context, scales_in);
    pnanovdb_compute_buffer_transient_t* opacities_transient =
        compute_interface->register_buffer_as_transient(context, opacities_in);
    pnanovdb_compute_buffer_transient_t* colors_transient =
        compute_interface->register_buffer_as_transient(context, colors_in);
    pnanovdb_compute_buffer_transient_t* nanovdb_transient =
        compute_interface->register_buffer_as_transient(context, nanovdb_inout);

    // state buffers
    pnanovdb_compute_buffer_transient_t* prim_bbox_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_bbox_buffer);
    pnanovdb_compute_buffer_transient_t* prim_idxs_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_idxs_buffer);
    pnanovdb_compute_buffer_transient_t* prim_raster_idxs_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_raster_idxs_buffer);

    // gaussian frag shader
    {
        pnanovdb_compute_resource_t resources[10u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = positions_transient;
        resources[2u].buffer_transient = quaternions_transient;
        resources[3u].buffer_transient = scales_transient;
        resources[4u].buffer_transient = opacities_transient;
        resources[5u].buffer_transient = colors_transient;
        resources[6u].buffer_transient = prim_idxs_transient;
        resources[7u].buffer_transient = prim_raster_idxs_transient;
        resources[8u].buffer_transient = prim_bbox_transient;
        resources[9u].buffer_transient = nanovdb_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_frag_color_slang],
            resources,
            (constants.ijk_batch_size + 255u) / 256u, 1u, 1u,
            "gaussian_frag_color"
        );
    }

    compute_interface->destroy_buffer(context, constant_buffer);

    return 0;
}

static int point_frag_color(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    raster_context_t* ctx,
    pnanovdb_compute_buffer_t* colors_in,
    pnanovdb_compute_buffer_t* ijk_in,
    pnanovdb_compute_buffer_t* nanovdb_inout,
    pnanovdb_uint64_t point_count,
    pnanovdb_uint32_t dispatch_count)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    struct constants_t
    {
        pnanovdb_uint32_t point_count;
        float pad1;
        float pad2;
        float pad3;
    };
    constants_t constants = {};
    constants.point_count = (pnanovdb_uint32_t)point_count;

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    // splat colors
    {
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
        pnanovdb_compute_buffer_transient_t* colors_transient =
            compute_interface->register_buffer_as_transient(context, colors_in);
        pnanovdb_compute_buffer_transient_t* ijk_transient =
            compute_interface->register_buffer_as_transient(context, ijk_in);
        pnanovdb_compute_buffer_transient_t* nanovdb_transient =
            compute_interface->register_buffer_as_transient(context, nanovdb_inout);

        {
            pnanovdb_compute_resource_t resources[4u] = {};
            resources[0u].buffer_transient = constant_transient;
            resources[1u].buffer_transient = colors_transient;
            resources[2u].buffer_transient = ijk_transient;
            resources[3u].buffer_transient = nanovdb_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[point_frag_color_slang],
                resources,
                (constants.point_count + 255u) / 256u, 1u, 1u,
                "point_frag_color"
            );
        }

        compute_interface->destroy_buffer(context, constant_buffer);
    }

    return 0;
}

#if PNANOVDB_RASTER_VALIDATE
static void raster_validate(
    const pnanovdb_compute_t* compute,
    float voxel_size,
    pnanovdb_compute_array_t* positions,
    pnanovdb_compute_array_t* colors,
    pnanovdb_compute_array_t* nanovdb_arr
);
#endif

void raster_gaussian_3d(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context_in,
    float voxel_size,
    pnanovdb_raster_gaussian_data_t* data_in,
    pnanovdb_compute_buffer_t* nanovdb_out,
    pnanovdb_uint64_t nanovdb_word_count,
    void* userdata
)
{
    auto ctx = cast(context_in);
    auto data = cast(data_in);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_util::WorkerThread* worker = static_cast<pnanovdb_util::WorkerThread*>(userdata);
    if (worker)
    {
        worker->updateTaskProgress(0.f, "Rastering gaussian data");
    }

    pnanovdb_uint32_t dispatch_count = 1u;

    pnanovdb_uint32_t prim_batch_size = 256u * 1024u;
    pnanovdb_uint32_t ijk_batch_size = 8u * 1024u * 1024u;
    pnanovdb_uint32_t ijk_batch_count_max = ~0u;

    // increase min lifetime to ensure buffers recycle
    compute->device_interface.set_resource_min_lifetime(context, 512u);

    // init loop state
    pnanovdb_grid_build_fanout_state_t state = {};
    ctx->grid_build.fanout_state_init(
        compute,
        queue,
        ctx->grid_build_ctx,
        &state,
        prim_batch_size,
        ijk_batch_size,
        ijk_batch_count_max,
        data->point_count,
        voxel_size
    );

    pnanovdb_grid_build_state_t grid_state = {};
    ctx->grid_build.grid_build_init(
        compute,
        queue,
        ctx->grid_build_ctx,
        &grid_state,
        nanovdb_out,
        nanovdb_word_count,
        voxel_size,
        dispatch_count
    );

    if (worker)
    {
        worker->updateTaskProgress(0.1f);
    }

    const pnanovdb_uint32_t raster_count = 1u;
    for (pnanovdb_uint32_t raster_idx = 0u; raster_idx < raster_count; raster_idx++)
    {
        pnanovdb_uint64_t time_begin;
        timestamp_capture(&time_begin);

        if (worker)
        {
            worker->updateTaskProgress(0.1f + 0.9f * (float)raster_idx / (float)raster_count);
        }

        // init loop state
        ctx->grid_build.fanout_state_reset(
            compute,
            queue,
            ctx->grid_build_ctx,
            &state
        );

        ctx->grid_build.grid_build_reset(
            compute,
            queue,
            ctx->grid_build_ctx,
            &grid_state,
            nanovdb_out,
            nanovdb_word_count,
            voxel_size,
            dispatch_count
        );

        while (ctx->grid_build.fanout_state_valid(
            compute,
            queue,
            ctx->grid_build_ctx,
            &state))
        {
            // gaussian prim shader
            gaussian_prim(
                compute,
                queue,
                ctx,
                &state,
                data->means_gpu_array->device_buffer,
                data->quaternions_gpu_array->device_buffer,
                data->scales_gpu_array->device_buffer,
                data->opacities_gpu_array->device_buffer,
                dispatch_count
            );

            // raster fanout
            ctx->grid_build.fanout(
                compute,
                queue,
                ctx->grid_build_ctx,
                &state,
                dispatch_count
            );

            // gaussian frag alloc
            gaussian_frag_alloc(
                compute,
                queue,
                ctx,
                &state,
                data->means_gpu_array->device_buffer,
                data->quaternions_gpu_array->device_buffer,
                data->scales_gpu_array->device_buffer,
                data->opacities_gpu_array->device_buffer,
                state.prim_raster_ijks_buffer,
                1u
            );

            ctx->grid_build.grid_build(
                compute,
                queue,
                ctx->grid_build_ctx,
                &grid_state,
                state.prim_raster_ijks_buffer,
                nanovdb_out,
                state.ijk_batch_size,
                nanovdb_word_count,
                voxel_size,
                dispatch_count
            );

            pnanovdb_uint64_t flushed_frame = 0llu;
            compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

            ctx->grid_build.fanout_state_increment(
                compute,
                queue,
                ctx->grid_build_ctx,
                &state
            );
        }

        ctx->grid_build.grid_build_finalize(
            compute,
            queue,
            ctx->grid_build_ctx,
            &grid_state,
            state.prim_raster_ijks_buffer,
            nanovdb_out,
            state.ijk_batch_size,
            nanovdb_word_count,
            voxel_size,
            dispatch_count
        );

        printf("Feedback total_fragment_count(%llu)\n", (long long unsigned int)state.total_fragment_count);

        // init loop state
        ctx->grid_build.fanout_state_reset(
            compute,
            queue,
            ctx->grid_build_ctx,
            &state
        );

        while (ctx->grid_build.fanout_state_valid(
            compute,
            queue,
            ctx->grid_build_ctx,
            &state))
        {
            // gaussian prim shader
            gaussian_prim(
                compute,
                queue,
                ctx,
                &state,
                data->means_gpu_array->device_buffer,
                data->quaternions_gpu_array->device_buffer,
                data->scales_gpu_array->device_buffer,
                data->opacities_gpu_array->device_buffer,
                dispatch_count
            );

            // raster fanout
            ctx->grid_build.fanout(
                compute,
                queue,
                ctx->grid_build_ctx,
                &state,
                dispatch_count
            );

            // gaussian frag color
            gaussian_frag_color(
                compute,
                queue,
                ctx,
                &state,
                data->means_gpu_array->device_buffer,
                data->quaternions_gpu_array->device_buffer,
                data->scales_gpu_array->device_buffer,
                data->opacities_gpu_array->device_buffer,
                data->colors_gpu_array->device_buffer,
                nanovdb_out,
                1u
            );

            pnanovdb_uint64_t flushed_frame = 0llu;
            compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

            ctx->grid_build.fanout_state_increment(
                compute,
                queue,
                ctx->grid_build_ctx,
                &state
            );
        }

        pnanovdb_uint64_t flushed_frame = 0llu;
        compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

        compute->device_interface.wait_idle(queue);

        pnanovdb_uint64_t time_end;
        timestamp_capture(&time_end);
        pnanovdb_uint64_t time_freq = timestamp_frequency();

        float raster_dt = timestamp_diff(time_begin, time_end, time_freq);
        printf("Raster time %f ms\n", 1000.f * raster_dt);
    }

    ctx->grid_build.grid_build_destroy(
        compute,
        queue,
        &grid_state
    );

    ctx->grid_build.fanout_state_destroy(
        compute,
        queue,
        &state
    );

    if (worker)
    {
        worker->updateTaskProgress(1.f);
    }
}

pnanovdb_compute_array_t* raster_points(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    float voxel_size,
    pnanovdb_compute_array_t* means,
    pnanovdb_compute_array_t* quaternions,
    pnanovdb_compute_array_t* scales,
    pnanovdb_compute_array_t* colors,
    pnanovdb_compute_array_t* spherical_harmonics,
    pnanovdb_compute_array_t* opacities,
    pnanovdb_profiler_report_t profiler_report,
    void* userdata)
{
    raster_context_t* ctx = cast(create_context(compute, queue));
    if (!ctx)
    {
        return nullptr;
    }

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    compute->device_interface.enable_profiler(context, (void*)("raster"), profiler_report);

    // note: colors duplicate for now, since no SH in interface
    pnanovdb_raster_gaussian_data_t* data = create_gaussian_data(
        compute, cast(ctx),
        means, quaternions, scales, colors, spherical_harmonics, opacities);

    upload_gaussian_data(compute, queue, cast(ctx), data);

    pnanovdb_uint64_t nanovdb_word_count = 3u * 256u * 1024u * 1024u;
    pnanovdb_compute_array_t* nanovdb_array = compute->create_array(4u, nanovdb_word_count, nullptr);

    compute_gpu_array_t* nanovdb_gpu_array = gpu_array_create();
    gpu_array_alloc_device(compute, queue, nanovdb_gpu_array, nanovdb_array);

    raster_gaussian_3d(
        compute,
        queue,
        cast(ctx),
        voxel_size,
        data,
        nanovdb_gpu_array->device_buffer,
        nanovdb_word_count,
        userdata
    );

    gpu_array_readback(compute, queue, nanovdb_gpu_array, nanovdb_array);

    pnanovdb_uint64_t flushed_frame = 0llu;
    compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

    compute->device_interface.wait_idle(queue);

    // to flush profile
    flushed_frame = 0llu;
    compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

    // restore min lifetime to default
    compute->device_interface.set_resource_min_lifetime(context, 60u);

    compute->device_interface.disable_profiler(context);

    destroy_context(compute, queue, cast(ctx));

    gpu_array_map(compute, queue, nanovdb_gpu_array, nanovdb_array);

    destroy_gaussian_data(compute, queue, cast(ctx), data);

    gpu_array_destroy(compute, queue, nanovdb_gpu_array);

    // to flush destroys
    for (pnanovdb_uint32_t flush_count = 0u; flush_count < 64u; flush_count++)
    {
        compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);
    }

#if PNANOVDB_RASTER_VALIDATE
    raster_validate(compute, voxel_size, means, colors, nanovdb_array);
#endif

    {
        pnanovdb_uint32_t* mapped_nanovdb = (pnanovdb_uint32_t*)compute->map_array(nanovdb_array);

        pnanovdb_buf_t buf = pnanovdb_make_buf(mapped_nanovdb, nanovdb_array->element_count);

        pnanovdb_grid_handle_t grid = {};
        pnanovdb_uint64_t grid_size = pnanovdb_grid_get_grid_size(buf, grid);
        printf("grid_size(%llu)\n", (unsigned long long int)grid_size);
        pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
        pnanovdb_uint32_t upper_count = pnanovdb_tree_get_node_count_upper(buf, tree);
        pnanovdb_uint32_t lower_count = pnanovdb_tree_get_node_count_lower(buf, tree);
        pnanovdb_uint32_t leaf_count = pnanovdb_tree_get_node_count_leaf(buf, tree);
        printf("upper_count(%llu), lower_count(%llu), leaf_count(%llu)\n",
            (unsigned long long int)upper_count, (unsigned long long int)lower_count, (unsigned long long int)leaf_count);
        pnanovdb_gridblindmetadata_handle_t meta = pnanovdb_grid_get_gridblindmetadata(buf, grid, 1u);
        pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(buf, meta);
        pnanovdb_uint32_t value_size = pnanovdb_gridblindmetadata_get_value_size(buf, meta);
        printf("value_count(%llu) value_size(%u)\n", (unsigned long long int)value_count, value_size);

        pnanovdb_address_t bbox_addr = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 0u);
        pnanovdb_coord_t bbox_min = pnanovdb_read_coord(buf, pnanovdb_address_offset(bbox_addr, 0u));
        pnanovdb_coord_t bbox_max = pnanovdb_read_coord(buf, pnanovdb_address_offset(bbox_addr, 12u));
        printf("bbox_min(%d,%d,%d) bbox_max(%d,%d,%d)\n",
            bbox_min.x, bbox_min.y, bbox_min.z, bbox_max.x, bbox_max.y, bbox_max.z);

        // trim element count to save upload size later
        if (grid_size <= nanovdb_array->element_size * nanovdb_array->element_count)
        {
            nanovdb_array->element_count = (grid_size + nanovdb_array->element_size - 1u) / nanovdb_array->element_size;
        }
        printf("nanovdb_array size trimmed to %llu bytes\n",
            (unsigned long long int)nanovdb_array->element_size * nanovdb_array->element_count);

        compute->unmap_array(nanovdb_array);
    }

    return nanovdb_array;
}

#if PNANOVDB_RASTER_VALIDATE
static void raster_validate(
    const pnanovdb_compute_t* compute,
    float voxel_size,
    pnanovdb_compute_array_t* positions,
    pnanovdb_compute_array_t* colors,
    pnanovdb_compute_array_t* nanovdb_arr
)
{
    pnanovdb_uint32_t* mapped_nanovdb = (pnanovdb_uint32_t*)compute->map_array(nanovdb_arr);

    pnanovdb_buf_t buf = pnanovdb_make_buf(mapped_nanovdb, nanovdb_arr->element_count);

    pnanovdb_grid_handle_t grid = {};
    pnanovdb_uint64_t grid_size = pnanovdb_grid_get_grid_size(buf, grid);
    printf("grid_size(%zu)\n", grid_size);
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_uint32_t tree_upper_count = pnanovdb_tree_get_node_count_upper(buf, tree);
    pnanovdb_uint32_t tree_lower_count = pnanovdb_tree_get_node_count_lower(buf, tree);
    pnanovdb_uint32_t tree_leaf_count = pnanovdb_tree_get_node_count_leaf(buf, tree);
    printf("upper_count(%llu), lower_count(%llu), leaf_count(%llu)\n",
        (unsigned long long int)tree_upper_count, (unsigned long long int)tree_lower_count, (unsigned long long int)tree_leaf_count);

    pnanovdb_node2_handle_t root = {pnanovdb_tree_get_root(buf, tree).address};

    pnanovdb_address_t bboxes = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 0u);
    pnanovdb_coord_t bbox_min = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 0u));
    pnanovdb_coord_t bbox_max = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 12u));

    printf("bbox_min(%d,%d,%d) bbox_max(%d,%d,%d)\n",
        bbox_min.x, bbox_min.y, bbox_min.z,
        bbox_max.x, bbox_max.y, bbox_max.z);

    printf("root prefix_sum(%zu)\n",
        pnanovdb_node2_get_child_mask_prefix_sum(buf, root, PNANOVDB_NODE2_TYPE_ROOT, 127u) >> 48u);


    pnanovdb_uint32_t lower_dense_child_bytes = 0u;
    pnanovdb_uint32_t lower_sparse_child_bytes = 0u;

    pnanovdb_uint32_t upper_count = 0u;
    pnanovdb_uint32_t lower_count = 0u;
    pnanovdb_uint32_t leaf_count = 0u;
    pnanovdb_uint32_t voxel_count = 0u;
    for (pnanovdb_uint32_t root_n = 0u; root_n < 32768u; root_n++)
    {
        if (pnanovdb_node2_get_child_mask_bit(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n))
        {
            upper_count++;
            pnanovdb_node2_handle_t upper =
                pnanovdb_node2_get_child_as_node2(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
            //printf("upper_addr(%zu)\n", upper.address.byte_offset);
            for (pnanovdb_uint32_t upper_n = 0u; upper_n < 32768u; upper_n++)
            {
                if (pnanovdb_node2_get_child_mask_bit(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n))
                {
                    lower_count++;
                    pnanovdb_node2_handle_t lower =
                        pnanovdb_node2_get_child_as_node2(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);

                    lower_dense_child_bytes += pnanovdb_node2_compute_size(PNANOVDB_NODE2_TYPE_LOWER);
                    lower_sparse_child_bytes += pnanovdb_node2_compute_size(PNANOVDB_NODE2_TYPE_LOWER) -
                        (pnanovdb_node2_type_fanout(PNANOVDB_NODE2_TYPE_LOWER) -
                        pnanovdb_node2_get_child_count(buf, lower, PNANOVDB_NODE2_TYPE_LOWER)) * 4u;

                    //printf("lower_addr(%zu)\n", lower.address.byte_offset);
                    for (pnanovdb_uint32_t lower_n = 0u; lower_n < 4096u; lower_n++)
                    {
                        if (pnanovdb_node2_get_child_mask_bit(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n))
                        {
                            leaf_count++;
                            pnanovdb_node2_handle_t leaf =
                                pnanovdb_node2_get_child_as_node2(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                            if (leaf_count < 32u)
                            {
                                pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_idx(buf, leaf);
                                printf("leaf(%d) value_idx(%llu)\n", leaf_count, (unsigned long long int)value_idx);
                            }
                            for (pnanovdb_uint32_t leaf_n = 0u; leaf_n < 512u; leaf_n++)
                            {
                                if (pnanovdb_node2_get_value_mask_bit(buf, leaf, PNANOVDB_NODE2_TYPE_LEAF, leaf_n))
                                {
                                    voxel_count++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    printf("upper_count(%d) lower_count(%d) leaf_count(%d) voxel_count(%d)\n", upper_count, lower_count, leaf_count, voxel_count);

    printf("lower_dense_child_bytes(%d) lower_sparse_child_bytes(%d)\n", lower_dense_child_bytes, lower_sparse_child_bytes);

    pnanovdb_uint32_t fault_count = 0u;

    pnanovdb_uint64_t point_count = positions->element_count / 3u;
    const float* positions_data = (const float*)positions->data;

    pnanovdb_uint32_t root_flags = pnanovdb_node2_get_flags(buf, root);

    float voxel_size_inv = 1.f / voxel_size;
    for (pnanovdb_uint64_t point_idx = 0u; point_idx < point_count; point_idx++)
    {
        pnanovdb_coord_t ijk;
        ijk.x = (int)floorf(voxel_size_inv * positions_data[3u * point_idx + 0u]);
        ijk.y = (int)floorf(voxel_size_inv * positions_data[3u * point_idx + 1u]);
        ijk.z = (int)floorf(voxel_size_inv * positions_data[3u * point_idx + 2u]);

        pnanovdb_uint32_t dense_n = 0u;
        pnanovdb_uint32_t node_type = 0u;
        pnanovdb_node2_handle_t node = pnanovdb_node2_root_find_node(buf, root, root_flags, PNANOVDB_REF(ijk), PNANOVDB_REF(dense_n), PNANOVDB_REF(node_type));

        pnanovdb_address_t values = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 1u);
        pnanovdb_address_t val_addr = pnanovdb_node2_get_value_address(buf, root, root_flags, values, 32u, 0u, PNANOVDB_REF(ijk));
        if (point_idx < 64u)
        {
            pnanovdb_uint32_t value_raw = pnanovdb_read_uint32(buf, val_addr);
            float color[4] = {
                float((value_raw >>  0) & 255) * (1.f / 255.f),
                float((value_raw >>  8) & 255) * (1.f / 255.f),
                float((value_raw >> 16) & 255) * (1.f / 255.f),
                float((value_raw >> 24) & 255) * (1.f / 255.f)
            };
            printf("val_addr(%llu) vcolor(%f,%f,%f,%f)\n",
                (unsigned long long int)val_addr.byte_offset,
                color[0], color[1], color[2], color[3]);
        }

        if (node_type == PNANOVDB_NODE2_TYPE_LEAF)
        {
            if (!pnanovdb_node2_get_value_mask_bit(buf, node, PNANOVDB_NODE2_TYPE_LEAF, dense_n))
            {
                fault_count++;
            }
        }
        else
        {
            fault_count++;
        }
    }
    printf("fault_count(%d)\n", fault_count);

    compute->unmap_array(nanovdb_arr);
}
#endif
}

pnanovdb_raster_t* pnanovdb_get_raster()
{
    static pnanovdb_raster_t raster = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_raster_t) };

    raster.create_context = pnanovdb_raster::create_context;
    raster.destroy_context = pnanovdb_raster::destroy_context;
    raster.create_gaussian_data = pnanovdb_raster::create_gaussian_data;
    raster.upload_gaussian_data = pnanovdb_raster::upload_gaussian_data;
    raster.destroy_gaussian_data = pnanovdb_raster::destroy_gaussian_data;
    raster.raster_gaussian_3d = pnanovdb_raster::raster_gaussian_3d;
    raster.raster_gaussian_2d = pnanovdb_raster::raster_gaussian_2d;
    raster.raster_points = pnanovdb_raster::raster_points;

    return &raster;
}
