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
#include "nanovdb_editor/putil/GridBuild.h"
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
    grid_build_blindmetadata_slang,
    grid_build_clear_slang,
    grid_build_compute_bbox_slang,
    grid_build_count_new_slang,
    grid_build_finalize_slang,
    grid_build_init_slang,
    grid_build_scatter_new_slang,
    grid_build_set_masks_slang,

    fanout_clear_prim_idxs_slang,
    fanout_compute_fragment_idxs_slang,
    fanout_scatter_prim_idxs_slang,

    shader_count
};

static const char* s_shader_names[shader_count] = {
    "raster/grid_build_blindmetadata.slang",
    "raster/grid_build_clear.slang",
    "raster/grid_build_compute_bbox.slang",
    "raster/grid_build_count_new.slang",
    "raster/grid_build_finalize.slang",
    "raster/grid_build_init.slang",
    "raster/grid_build_scatter_new.slang",
    "raster/grid_build_set_masks.slang",

    "raster/fanout_clear_prim_idxs.slang",
    "raster/fanout_compute_fragment_idxs.slang",
    "raster/fanout_scatter_prim_idxs.slang"
};

struct grid_build_context_t
{
    pnanovdb_shader_context_t* shader_ctx[shader_count];

    pnanovdb_parallel_primitives_t parallel_primitives;
    pnanovdb_parallel_primitives_context_t* parallel_primitives_ctx;
};

PNANOVDB_CAST_PAIR(pnanovdb_grid_build_context_t, grid_build_context_t)

static pnanovdb_grid_build_context_t* create_context(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue)
{
    grid_build_context_t* ctx = new grid_build_context_t();

    pnanovdb_parallel_primitives_load(&ctx->parallel_primitives, compute);
    ctx->parallel_primitives_ctx = ctx->parallel_primitives.create_context(compute, queue);

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
    pnanovdb_grid_build_context_t* context_in)
{
    auto ctx = cast(context_in);

    for (pnanovdb_uint32_t idx = 0u; idx < shader_count; idx++)
    {
        compute->destroy_shader_context(compute, queue, ctx->shader_ctx[idx]);
    }

    ctx->parallel_primitives.destroy_context(compute, queue, ctx->parallel_primitives_ctx);
    pnanovdb_parallel_primitives_free(&ctx->parallel_primitives);

    delete ctx;
}

static pnanovdb_uint32_t empty_grid_size()
{
    pnanovdb_uint32_t grid_size = PNANOVDB_GRID_SIZE + PNANOVDB_TREE_SIZE;
    // node2 root
    grid_size += pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_ROOT];
    return grid_size;
}

static int empty_grid_generate(void* dst_data, pnanovdb_uint64_t dst_size, float voxel_size)
{
    float voxel_size_inv = 1.f / voxel_size;

    pnanovdb_buf_t buf = pnanovdb_make_buf((pnanovdb_uint32_t*)dst_data, dst_size / 4u);

    pnanovdb_node2_memclear(buf, pnanovdb_address_null(), dst_size);

    pnanovdb_grid_handle_t grid = {pnanovdb_address_null()};
    pnanovdb_grid_set_magic(buf, grid, PNANOVDB_MAGIC_GRID);
    pnanovdb_grid_set_version(buf, grid,
        pnanovdb_make_version(PNANOVDB_MAJOR_VERSION_NUMBER, PNANOVDB_MINOR_VERSION_NUMBER, PNANOVDB_PATCH_VERSION_NUMBER));
    pnanovdb_grid_set_flags(buf, grid, 0u);
    pnanovdb_grid_set_grid_index(buf, grid, 0u);
    pnanovdb_grid_set_grid_count(buf, grid, 1u);
    pnanovdb_grid_set_grid_size(buf, grid, dst_size);
    pnanovdb_grid_set_grid_name(buf, grid, 0u, 0x65646f6e);     // "node2"
    pnanovdb_grid_set_grid_name(buf, grid, 1u, 0x00000032);
    pnanovdb_grid_set_voxel_size(buf, grid, 0u, voxel_size);
    pnanovdb_grid_set_voxel_size(buf, grid, 1u, voxel_size);
    pnanovdb_grid_set_voxel_size(buf, grid, 2u, voxel_size);
    pnanovdb_grid_set_grid_class(buf, grid, PNANOVDB_GRID_CLASS_UNKNOWN);
    pnanovdb_grid_set_grid_type(buf, grid, PNANOVDB_GRID_TYPE_NODE2);

    pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
    pnanovdb_map_set_matf(buf, map, 0u, voxel_size);
    pnanovdb_map_set_matf(buf, map, 4u, voxel_size);
    pnanovdb_map_set_matf(buf, map, 8u, voxel_size);
    pnanovdb_map_set_invmatf(buf, map, 0u, voxel_size_inv);
    pnanovdb_map_set_invmatf(buf, map, 4u, voxel_size_inv);
    pnanovdb_map_set_invmatf(buf, map, 8u, voxel_size_inv);
    pnanovdb_map_set_matd(buf, map, 0u, voxel_size);
    pnanovdb_map_set_matd(buf, map, 4u, voxel_size);
    pnanovdb_map_set_matd(buf, map, 8u, voxel_size);
    pnanovdb_map_set_invmatd(buf, map, 0u, voxel_size_inv);
    pnanovdb_map_set_invmatd(buf, map, 4u, voxel_size_inv);
    pnanovdb_map_set_invmatd(buf, map, 8u, voxel_size_inv);
    pnanovdb_map_set_vecf(buf, map, 0u, 0.f);
    pnanovdb_map_set_vecf(buf, map, 1u, 0.f);
    pnanovdb_map_set_vecf(buf, map, 2u, 0.f);
    pnanovdb_map_set_vecd(buf, map, 0u, 0.0);
    pnanovdb_map_set_vecd(buf, map, 1u, 0.0);
    pnanovdb_map_set_vecd(buf, map, 2u, 0.0);

    // legacy tree for ABI compatibility
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    // initialized by pnanovdb_node2_memclear()

    pnanovdb_address_t root_addr = pnanovdb_address_offset(tree.address, PNANOVDB_TREE_SIZE);
    pnanovdb_node2_handle_t root = {pnanovdb_uint32_t(root_addr.byte_offset >> 3u)};
    // initialized by pnanovdb_node2_memclear()

    // set keys to default
    for (pnanovdb_uint32_t node_n = 0u; node_n < 2u * pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT]; node_n++)
    {
        pnanovdb_node2_write(buf, root.idx64 + pnanovdb_node2_off_children[PNANOVDB_NODE2_TYPE_ROOT] + node_n, pnanovdb_node2_end_key);
    }

    pnanovdb_address_t addr_end = pnanovdb_address_offset64(root_addr, pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_ROOT]);

    // point tree at root
    pnanovdb_tree_set_node_offset_root(buf, tree, pnanovdb_address_diff(root_addr, tree.address));

    return 0;
}

static void grid_build_init(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_state_t* state,
    pnanovdb_compute_buffer_t* nanovdb_out,
    pnanovdb_uint64_t nanovdb_word_count,
    float voxel_size,
    pnanovdb_uint32_t dispatch_count)
{
    auto ctx = cast(context_in);

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    struct constants_t
    {
        pnanovdb_uint32_t workgroup_count;
        pnanovdb_uint32_t max_node_count;
        pnanovdb_uint32_t buf_word_count;
        pnanovdb_uint32_t point_count;
        pnanovdb_uint32_t empty_grid_word_count;
        pnanovdb_uint32_t active_node_type;
        pnanovdb_uint32_t child_node_type;
    };
    constants_t constants = {};
    constants.workgroup_count = 4096u;
    constants.max_node_count = 2u * 1024u * 1024u;
    constants.buf_word_count = nanovdb_word_count;
    constants.point_count = 0llu;
    constants.empty_grid_word_count = empty_grid_size() / 4u;
    constants.active_node_type = PNANOVDB_NODE2_TYPE_ROOT;
    constants.child_node_type = PNANOVDB_NODE2_TYPE_UPPER;

    // per node_idx buffers
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = constants.max_node_count * 4u;
    state->node_addresses_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    state->node_types_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    state->new_child_counts_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    state->scan_new_child_counts_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    // bbox buffer
    buf_desc.size_in_bytes = constants.max_node_count * 6u * 4u;
    state->bboxes_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
}

static void grid_build_reset(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_state_t* state,
    pnanovdb_compute_buffer_t* nanovdb_out,
    pnanovdb_uint64_t nanovdb_word_count,
    float voxel_size,
    pnanovdb_uint32_t dispatch_count)
{
    auto ctx = cast(context_in);

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_transient_t* nanovdb_transient =
        compute_interface->register_buffer_as_transient(context, nanovdb_out);

    struct constants_t
    {
        pnanovdb_uint32_t workgroup_count;
        pnanovdb_uint32_t max_node_count;
        pnanovdb_uint32_t buf_word_count;
        pnanovdb_uint32_t point_count;
        pnanovdb_uint32_t empty_grid_word_count;
        pnanovdb_uint32_t active_node_type;
        pnanovdb_uint32_t child_node_type;
    };
    constants_t constants = {};
    constants.workgroup_count = 4096u;
    constants.max_node_count = 2u * 1024u * 1024u;
    constants.buf_word_count = nanovdb_word_count;
    constants.point_count = 0llu;
    constants.empty_grid_word_count = empty_grid_size() / 4u;
    constants.active_node_type = PNANOVDB_NODE2_TYPE_ROOT;
    constants.child_node_type = PNANOVDB_NODE2_TYPE_UPPER;

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

    pnanovdb_uint32_t upload_size = constants.empty_grid_word_count * 4u;

    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = upload_size;
    pnanovdb_compute_buffer_t* upload_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

    void* mapped_upload = compute_interface->map_buffer(context, upload_buffer);
    empty_grid_generate(mapped_upload, upload_size, voxel_size);
    compute_interface->unmap_buffer(context, upload_buffer);

    pnanovdb_compute_buffer_transient_t* node_addresses_transient =
        compute_interface->register_buffer_as_transient(context, state->node_addresses_buffer);
    pnanovdb_compute_buffer_transient_t* node_types_transient =
        compute_interface->register_buffer_as_transient(context, state->node_types_buffer);

    // clear nanovdb buf with empty grid
    {
        pnanovdb_compute_resource_t resources[3u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = compute_interface->register_buffer_as_transient(context, upload_buffer);
        resources[2u].buffer_transient = nanovdb_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[grid_build_clear_slang],
            resources,
            constants.workgroup_count, 1u, 1u,
            "grid_build_clear"
        );
    }
    // initialize per node_idx buffers
    {
        pnanovdb_compute_resource_t resources[4u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = nanovdb_transient;
        resources[2u].buffer_transient = node_addresses_transient;
        resources[3u].buffer_transient = node_types_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[grid_build_init_slang],
            resources,
            (constants.max_node_count + 255u) / 256u, 1u, 1u,
            "grid_build_init"
        );
    }

    compute_interface->destroy_buffer(context, constant_buffer);
    compute_interface->destroy_buffer(context, upload_buffer);
}

static void grid_build_destroy(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_state_t* state)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    // release temporary buffers
    compute_interface->destroy_buffer(context, state->node_addresses_buffer);
    compute_interface->destroy_buffer(context, state->node_types_buffer);
    compute_interface->destroy_buffer(context, state->new_child_counts_buffer);
    compute_interface->destroy_buffer(context, state->scan_new_child_counts_buffer);

    compute_interface->destroy_buffer(context, state->bboxes_buffer);
}

static void grid_build(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_state_t* state,
    pnanovdb_compute_buffer_t* point_ijk_in,
    pnanovdb_compute_buffer_t* nanovdb_out,
    pnanovdb_uint64_t point_count,
    pnanovdb_uint64_t nanovdb_word_count,
    float voxel_size,
    pnanovdb_uint32_t dispatch_count)
{
    auto ctx = cast(context_in);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_transient_t* point_ijk_transient =
        compute_interface->register_buffer_as_transient(context, point_ijk_in);
    pnanovdb_compute_buffer_transient_t* nanovdb_transient =
        compute_interface->register_buffer_as_transient(context, nanovdb_out);

    struct constants_t
    {
        pnanovdb_uint32_t workgroup_count;
        pnanovdb_uint32_t max_node_count;
        pnanovdb_uint32_t buf_word_count;
        pnanovdb_uint32_t point_count;
        pnanovdb_uint32_t empty_grid_word_count;
        pnanovdb_uint32_t active_node_type;
        pnanovdb_uint32_t child_node_type;
    };
    constants_t constants = {};
    constants.workgroup_count = 4096u;
    constants.max_node_count = 2u * 1024u * 1024u;
    constants.buf_word_count = nanovdb_word_count;
    constants.point_count = point_count;
    constants.empty_grid_word_count = empty_grid_size() / 4u;
    constants.active_node_type = PNANOVDB_NODE2_TYPE_ROOT;
    constants.child_node_type = PNANOVDB_NODE2_TYPE_UPPER;

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    pnanovdb_compute_buffer_transient_t* node_addresses_transient =
        compute_interface->register_buffer_as_transient(context, state->node_addresses_buffer);
    pnanovdb_compute_buffer_transient_t* node_types_transient =
        compute_interface->register_buffer_as_transient(context, state->node_types_buffer);
    pnanovdb_compute_buffer_transient_t* new_child_counts_transient =
        compute_interface->register_buffer_as_transient(context, state->new_child_counts_buffer);
    pnanovdb_compute_buffer_transient_t* scan_new_child_counts_transient =
        compute_interface->register_buffer_as_transient(context, state->scan_new_child_counts_buffer);

    for (pnanovdb_uint32_t tree_level = 0u; tree_level < 4u; tree_level++)
    {
        // constants
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 0u;
        buf_desc.size_in_bytes = sizeof(constants_t);
        pnanovdb_compute_buffer_t* constant_buffer =
            compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

        if (tree_level == 0u)
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_ROOT;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_UPPER;
        }
        else if (tree_level == 1u)
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_UPPER;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_LOWER;
        }
        else if (tree_level == 2)
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_LOWER;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_LEAF;
        }
        else
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_LEAF;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_LEAF;
        }

        // copy constants
        void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
        memcpy(mapped_constants, &constants, sizeof(constants_t));
        compute_interface->unmap_buffer(context, constant_buffer);

        pnanovdb_compute_buffer_transient_t* constant_transient =
            compute_interface->register_buffer_as_transient(context, constant_buffer);

        // set masks
        {
            pnanovdb_compute_resource_t resources[3u] = {};
            resources[0u].buffer_transient = constant_transient;
            resources[1u].buffer_transient = point_ijk_transient;
            resources[2u].buffer_transient = nanovdb_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[grid_build_set_masks_slang],
                resources,
                (constants.point_count + 255u) / 256u, 1u, 1u,
                "grid_build_set_masks"
            );
        }
        // count new
        {
            pnanovdb_compute_resource_t resources[5u] = {};
            resources[0u].buffer_transient = constant_transient;
            resources[1u].buffer_transient = node_addresses_transient;
            resources[2u].buffer_transient = node_types_transient;
            resources[3u].buffer_transient = nanovdb_transient;
            resources[4u].buffer_transient = new_child_counts_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[grid_build_count_new_slang],
                resources,
                constants.workgroup_count, 1u, 1u,
                "grid_build_count_new"
            );
        }
        // global scan new_child_counts
        {
            ctx->parallel_primitives.global_scan(
                compute,
                queue,
                ctx->parallel_primitives_ctx,
                state->new_child_counts_buffer,
                state->scan_new_child_counts_buffer,
                constants.max_node_count,
                1u
            );
        }
        // scatter new
        {
            pnanovdb_compute_resource_t resources[6u] = {};
            resources[0u].buffer_transient = constant_transient;
            resources[1u].buffer_transient = new_child_counts_transient;
            resources[2u].buffer_transient = scan_new_child_counts_transient;
            resources[3u].buffer_transient = nanovdb_transient;
            resources[4u].buffer_transient = node_addresses_transient;
            resources[5u].buffer_transient = node_types_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[grid_build_scatter_new_slang],
                resources,
                constants.workgroup_count, 1u, 1u,
                "grid_build_scatter_new"
            );
        }
        // finalize
        {
            pnanovdb_compute_resource_t resources[4u] = {};
            resources[0u].buffer_transient = constant_transient;
            resources[1u].buffer_transient = new_child_counts_transient;
            resources[2u].buffer_transient = scan_new_child_counts_transient;
            resources[3u].buffer_transient = nanovdb_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[grid_build_finalize_slang],
                resources,
                1u, 1u, 1u,
                "grid_build_finalize"
            );
        }

        compute_interface->destroy_buffer(context, constant_buffer);
    }
}

static void grid_build_finalize(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_state_t* state,
    pnanovdb_compute_buffer_t* point_ijk_in,
    pnanovdb_compute_buffer_t* nanovdb_out,
    pnanovdb_uint64_t point_count,
    pnanovdb_uint64_t nanovdb_word_count,
    float voxel_size,
    pnanovdb_uint32_t dispatch_count)
{
    auto ctx = cast(context_in);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_transient_t* point_ijk_transient =
        compute_interface->register_buffer_as_transient(context, point_ijk_in);
    pnanovdb_compute_buffer_transient_t* nanovdb_transient =
        compute_interface->register_buffer_as_transient(context, nanovdb_out);

    struct constants_t
    {
        pnanovdb_uint32_t workgroup_count;
        pnanovdb_uint32_t max_node_count;
        pnanovdb_uint32_t buf_word_count;
        pnanovdb_uint32_t point_count;
        pnanovdb_uint32_t empty_grid_word_count;
        pnanovdb_uint32_t active_node_type;
        pnanovdb_uint32_t child_node_type;
    };
    constants_t constants = {};
    constants.workgroup_count = 4096u;
    constants.max_node_count = 2u * 1024u * 1024u;
    constants.buf_word_count = nanovdb_word_count;
    constants.point_count = point_count;
    constants.empty_grid_word_count = empty_grid_size() / 4u;
    constants.active_node_type = PNANOVDB_NODE2_TYPE_ROOT;
    constants.child_node_type = PNANOVDB_NODE2_TYPE_UPPER;

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    pnanovdb_compute_buffer_transient_t* node_addresses_transient =
        compute_interface->register_buffer_as_transient(context, state->node_addresses_buffer);
    pnanovdb_compute_buffer_transient_t* node_types_transient =
        compute_interface->register_buffer_as_transient(context, state->node_types_buffer);
    pnanovdb_compute_buffer_transient_t* new_child_counts_transient =
        compute_interface->register_buffer_as_transient(context, state->new_child_counts_buffer);
    pnanovdb_compute_buffer_transient_t* scan_new_child_counts_transient =
        compute_interface->register_buffer_as_transient(context, state->scan_new_child_counts_buffer);

    pnanovdb_compute_buffer_transient_t* bboxes_transient =
        compute_interface->register_buffer_as_transient(context, state->bboxes_buffer);

    // add blindmetadata
    {
        // constants
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 0u;
        buf_desc.size_in_bytes = sizeof(constants_t);
        pnanovdb_compute_buffer_t* constant_buffer =
            compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

        constants.active_node_type = PNANOVDB_NODE2_TYPE_LEAF;
        constants.child_node_type = PNANOVDB_NODE2_TYPE_LEAF;

        // copy constants
        void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
        memcpy(mapped_constants, &constants, sizeof(constants_t));
        compute_interface->unmap_buffer(context, constant_buffer);

        pnanovdb_compute_buffer_transient_t* constant_transient =
            compute_interface->register_buffer_as_transient(context, constant_buffer);

        pnanovdb_compute_resource_t resources[4u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = new_child_counts_transient;
        resources[2u].buffer_transient = scan_new_child_counts_transient;
        resources[3u].buffer_transient = nanovdb_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[grid_build_blindmetadata_slang],
            resources,
            1u, 1u, 1u,
            "grid_build_blindmetadata"
        );

        compute_interface->destroy_buffer(context, constant_buffer);
    }

    // compute bbox
    for (pnanovdb_uint32_t tree_level = 3u; tree_level < 4u; tree_level--)
    {
        // constants
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 0u;
        buf_desc.size_in_bytes = sizeof(constants_t);
        pnanovdb_compute_buffer_t* constant_buffer =
            compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

        if (tree_level == 0u)
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_ROOT;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_UPPER;
        }
        else if (tree_level == 1u)
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_UPPER;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_LOWER;
        }
        else if (tree_level == 2)
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_LOWER;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_LEAF;
        }
        else
        {
            constants.active_node_type = PNANOVDB_NODE2_TYPE_LEAF;
            constants.child_node_type = PNANOVDB_NODE2_TYPE_LEAF;
        }

        // copy constants
        void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
        memcpy(mapped_constants, &constants, sizeof(constants_t));
        compute_interface->unmap_buffer(context, constant_buffer);

        pnanovdb_compute_buffer_transient_t* constant_transient =
            compute_interface->register_buffer_as_transient(context, constant_buffer);

        // compute bbox
        {
            pnanovdb_compute_resource_t resources[5u] = {};
            resources[0u].buffer_transient = constant_transient;
            resources[1u].buffer_transient = node_addresses_transient;
            resources[2u].buffer_transient = node_types_transient;
            resources[3u].buffer_transient = nanovdb_transient;
            resources[4u].buffer_transient = bboxes_transient;

            compute->dispatch_shader(
                compute_interface,
                context,
                ctx->shader_ctx[grid_build_compute_bbox_slang],
                resources,
                constants.workgroup_count, 1u, 1u,
                "grid_build_compute_bbox"
            );
        }

        compute_interface->destroy_buffer(context, constant_buffer);
    }
}

//-------------------------------- fanout ---------------------------------

static void fanout_state_reset(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_fanout_state_t* state)
{
    state->prim_batch_idx = 0u;
    state->ijk_batch_idx = 0u;
    state->total_fragment_count = 0llu;
}

static pnanovdb_bool_t fanout_state_valid(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_fanout_state_t* state)
{
    return state->prim_batch_idx < state->prim_batch_count;
}

static void fanout_state_increment(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_fanout_state_t* state)
{
    state->ijk_batch_idx++;
    if (state->ijk_batch_idx >= state->ijk_batch_count)
    {
        state->prim_batch_idx++;
        state->ijk_batch_idx = 0u;
        state->ijk_batch_count = state->ijk_batch_count_max;
    }
}

static void fanout_state_init(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_fanout_state_t* state,
    pnanovdb_uint32_t prim_batch_size,
    pnanovdb_uint32_t ijk_batch_size,
    pnanovdb_uint32_t ijk_batch_count_max,
    pnanovdb_uint32_t prim_count,
    float voxel_size)
{
    state->prim_batch_size = prim_batch_size;
    state->ijk_batch_size = ijk_batch_size;
    state->prim_count = prim_count;
    state->voxel_size = voxel_size;

    state->prim_batch_count = (state->prim_count + state->prim_batch_size - 1u) / state->prim_batch_size;
    state->ijk_batch_count = ijk_batch_count_max;
    state->ijk_batch_count_max = ijk_batch_count_max;
    fanout_state_reset(compute, queue, context_in, state);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    // per prim buffers
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 8u;
    buf_desc.size_in_bytes = state->prim_batch_size * 8u;   // uint64_t
    state->fragment_counts_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    state->scan_fragment_counts_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.structure_stride = 4u;         // int
    buf_desc.size_in_bytes = state->prim_batch_size * 6u * 4u;
    state->prim_bbox_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    // per fragment buffers
    buf_desc.structure_stride = 4u;         // uint
    buf_desc.size_in_bytes = state->ijk_batch_size * 4u;
    state->scatter_prim_idxs_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    state->prim_idxs_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    state->prim_raster_idxs_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.size_in_bytes = state->ijk_batch_size * 3u * 4u;
    state->prim_raster_ijks_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    // readback buffer
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
    buf_desc.size_in_bytes = state->prim_batch_count * 8u;      // uint64_t
    state->readback_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK, &buf_desc);

    state->readback_frame_count = state->prim_batch_count;
    state->readback_frames = new pnanovdb_uint64_t[state->readback_frame_count];
    for (size_t idx = 0u; idx < state->readback_frame_count; idx++)
    {
        state->readback_frames[idx] = ~0llu;
    }
}

static void fanout_state_destroy(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_fanout_state_t* state)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    compute_interface->destroy_buffer(context, state->fragment_counts_buffer);
    compute_interface->destroy_buffer(context, state->scan_fragment_counts_buffer);
    compute_interface->destroy_buffer(context, state->prim_bbox_buffer);

    compute_interface->destroy_buffer(context, state->scatter_prim_idxs_buffer);
    compute_interface->destroy_buffer(context, state->prim_idxs_buffer);
    compute_interface->destroy_buffer(context, state->prim_raster_idxs_buffer);
    compute_interface->destroy_buffer(context, state->prim_raster_ijks_buffer);

    compute_interface->destroy_buffer(context, state->readback_buffer);

    if (state->readback_frames)
    {
        delete [] state->readback_frames;
        state->readback_frames = nullptr;
        state->readback_frame_count = 0llu;
    }
}

struct fanout_constants_t
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

static pnanovdb_compute_buffer_t* fanout_get_constant_buffer(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_fanout_state_t* state,
    fanout_constants_t* constants_out)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    fanout_constants_t constants = {};
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
    buf_desc.size_in_bytes = sizeof(fanout_constants_t);
    pnanovdb_compute_buffer_t* constant_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

    // copy constants
    void* mapped_constants = compute_interface->map_buffer(context, constant_buffer);
    memcpy(mapped_constants, &constants, sizeof(fanout_constants_t));
    compute_interface->unmap_buffer(context, constant_buffer);

    if (constants_out)
    {
        *constants_out = constants;
    }
    return constant_buffer;
}

static void fanout(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_grid_build_context_t* context_in,
    pnanovdb_grid_build_fanout_state_t* state,
    pnanovdb_uint32_t dispatch_count)
{
    auto ctx = cast(context_in);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    fanout_constants_t constants = {};
    pnanovdb_compute_buffer_t* constant_buffer = fanout_get_constant_buffer(compute, queue, state, &constants);

    pnanovdb_compute_buffer_transient_t* constant_transient =
        compute_interface->register_buffer_as_transient(context, constant_buffer);

    // per prim buffers
    pnanovdb_compute_buffer_transient_t* fragment_counts_transient =
        compute_interface->register_buffer_as_transient(context, state->fragment_counts_buffer);
    pnanovdb_compute_buffer_transient_t* scan_fragment_counts_transient =
        compute_interface->register_buffer_as_transient(context, state->scan_fragment_counts_buffer);

    // per fragment buffers
    pnanovdb_compute_buffer_transient_t* scatter_prim_idxs_transient =
        compute_interface->register_buffer_as_transient(context, state->scatter_prim_idxs_buffer);
    pnanovdb_compute_buffer_transient_t* prim_idxs_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_idxs_buffer);
    pnanovdb_compute_buffer_transient_t* prim_raster_idxs_transient =
        compute_interface->register_buffer_as_transient(context, state->prim_raster_idxs_buffer);

    // global scan fragment_count
    {
        ctx->parallel_primitives.global_scan_uint64(
            compute,
            queue,
            ctx->parallel_primitives_ctx,
            state->fragment_counts_buffer,
            state->scan_fragment_counts_buffer,
            constants.prim_batch_size,
            1u
        );
    }
    if (state->readback_frames[state->prim_batch_idx] == ~0llu)
    {
        pnanovdb_compute_copy_buffer_params_t copy_params = {};
        copy_params.num_bytes = 8u;     // uint64_t
        copy_params.src = compute_interface->register_buffer_as_transient(context, state->scan_fragment_counts_buffer);
        copy_params.src_offset = (state->prim_batch_size - 1u) * 8u;
        copy_params.dst = compute_interface->register_buffer_as_transient(context, state->readback_buffer);
        copy_params.dst_offset = state->prim_batch_idx * 8u;
        copy_params.debug_label = "raster_feedback_copy";
        compute_interface->copy_buffer(context, &copy_params);

        pnanovdb_compute_frame_info_t frame_info = {};
        compute_interface->get_frame_info(context, &frame_info);
        state->readback_frames[state->prim_batch_idx] = frame_info.frame_local_current;
    }
    else if (state->ijk_batch_count == state->ijk_batch_count_max)
    {
        pnanovdb_compute_frame_info_t frame_info = {};
        compute_interface->get_frame_info(context, &frame_info);

        if (state->readback_frames[state->prim_batch_idx] <= frame_info.frame_local_completed)
        {
            pnanovdb_uint64_t* mapped = (pnanovdb_uint64_t*)compute_interface->map_buffer(context, state->readback_buffer);

            pnanovdb_uint64_t fragment_count = mapped[state->prim_batch_idx];
            state->total_fragment_count += fragment_count;
            state->ijk_batch_count = (fragment_count + state->ijk_batch_size - 1u) / state->ijk_batch_size;
            printf("Feedback ijk_batch_count(%d)\n", state->ijk_batch_count);

            compute_interface->unmap_buffer(context, state->readback_buffer);
        }
    }

    // clear prim idxs
    {
        pnanovdb_compute_resource_t resources[2u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = scatter_prim_idxs_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[fanout_clear_prim_idxs_slang],
            resources,
            (constants.ijk_batch_size + 255u) / 256u, 1u, 1u,
            "fanout_clear_prim_idxs"
        );
    }
    // scatter prim idxs
    {
        pnanovdb_compute_resource_t resources[4u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = fragment_counts_transient;
        resources[2u].buffer_transient = scan_fragment_counts_transient;
        resources[3u].buffer_transient = scatter_prim_idxs_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[fanout_scatter_prim_idxs_slang],
            resources,
            (constants.prim_batch_size + 255u) / 256u, 1u, 1u,
            "fanout_prim_idxs"
        );
    }
    // global max scan to spread prim ids
    {
        ctx->parallel_primitives.global_scan_max(
            compute,
            queue,
            ctx->parallel_primitives_ctx,
            state->scatter_prim_idxs_buffer,
            state->prim_idxs_buffer,
            constants.ijk_batch_size,
            1u
        );
    }
    // compute fragment idxs
    {
        pnanovdb_compute_resource_t resources[5u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = fragment_counts_transient;
        resources[2u].buffer_transient = scan_fragment_counts_transient;
        resources[3u].buffer_transient = prim_idxs_transient;
        resources[4u].buffer_transient = prim_raster_idxs_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[fanout_compute_fragment_idxs_slang],
            resources,
            (constants.ijk_batch_size + 255u) / 256u, 1u, 1u,
            "fanout_compute_fragment_idxs"
        );
    }

    compute_interface->destroy_buffer(context, constant_buffer);
}

}

pnanovdb_grid_build_t* pnanovdb_get_grid_build()
{
    static pnanovdb_grid_build_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_grid_build_t) };

    iface.create_context = create_context;
    iface.destroy_context = destroy_context;

    iface.grid_build_init = grid_build_init;
    iface.grid_build_reset = grid_build_reset;
    iface.grid_build_destroy = grid_build_destroy;
    iface.grid_build = grid_build;
    iface.grid_build_finalize = grid_build_finalize;

    iface.fanout_state_reset = fanout_state_reset;
    iface.fanout_state_valid = fanout_state_valid;
    iface.fanout_state_increment = fanout_state_increment;
    iface.fanout_state_init = fanout_state_init;
    iface.fanout_state_destroy = fanout_state_destroy;
    iface.fanout = fanout;

    return &iface;
}
