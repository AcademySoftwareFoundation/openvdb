// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/putil/GridBuild.h

    \author Andrew Reidmeyer

    \brief
*/

#ifndef NANOVDB_PUTILS_GRID_BUILD_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_GRID_BUILD_H_HAS_BEEN_INCLUDED

#include "nanovdb_editor/putil/ParallelPrimitives.h"

/// ********************************* GridBuild ***************************************

struct pnanovdb_grid_build_context_t;
typedef struct pnanovdb_grid_build_context_t pnanovdb_grid_build_context_t;

typedef struct pnanovdb_grid_build_state_t
{
    pnanovdb_compute_buffer_t* node_addresses_buffer;
    pnanovdb_compute_buffer_t* node_types_buffer;
    pnanovdb_compute_buffer_t* new_child_counts_buffer;
    pnanovdb_compute_buffer_t* scan_new_child_counts_buffer;
    pnanovdb_compute_buffer_t* bboxes_buffer;
}pnanovdb_grid_build_state_t;

typedef struct pnanovdb_grid_build_fanout_state_t
{
    pnanovdb_uint32_t prim_batch_size;
    pnanovdb_uint32_t ijk_batch_size;
    pnanovdb_uint32_t prim_count;
    float voxel_size;

    pnanovdb_uint32_t prim_batch_idx;
    pnanovdb_uint32_t prim_batch_count;
    pnanovdb_uint32_t ijk_batch_idx;
    pnanovdb_uint32_t ijk_batch_count;
    pnanovdb_uint32_t ijk_batch_count_max;

    pnanovdb_uint64_t total_fragment_count;

    pnanovdb_compute_buffer_t* fragment_counts_buffer;
    pnanovdb_compute_buffer_t* scan_fragment_counts_buffer;
    pnanovdb_compute_buffer_t* prim_bbox_buffer;

    pnanovdb_compute_buffer_t* scatter_prim_idxs_buffer;
    pnanovdb_compute_buffer_t* prim_idxs_buffer;
    pnanovdb_compute_buffer_t* prim_raster_idxs_buffer;
    pnanovdb_compute_buffer_t* prim_raster_ijks_buffer;

    pnanovdb_compute_buffer_t* readback_buffer;
    pnanovdb_uint64_t* readback_frames;
    pnanovdb_uint64_t readback_frame_count;
}pnanovdb_grid_build_fanout_state_t;

typedef struct pnanovdb_grid_build_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_grid_build_context_t* (PNANOVDB_ABI* create_context)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue);

    void (PNANOVDB_ABI* destroy_context)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context);

    void (PNANOVDB_ABI* grid_build_init)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_state_t* state,
        pnanovdb_compute_buffer_t* nanovdb_out,
        pnanovdb_uint64_t nanovdb_word_count,
        float voxel_size,
        pnanovdb_uint32_t dispatch_count);

    void (PNANOVDB_ABI* grid_build_reset)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_state_t* state,
        pnanovdb_compute_buffer_t* nanovdb_out,
        pnanovdb_uint64_t nanovdb_word_count,
        float voxel_size,
        pnanovdb_uint32_t dispatch_count);

    void (PNANOVDB_ABI* grid_build_destroy)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_state_t* state);

    void (PNANOVDB_ABI* grid_build)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_state_t* state,
        pnanovdb_compute_buffer_t* point_ijk_in,
        pnanovdb_compute_buffer_t* nanovdb_out,
        pnanovdb_uint64_t point_count,
        pnanovdb_uint64_t nanovdb_word_count,
        float voxel_size,
        pnanovdb_uint32_t dispatch_count);

    void (PNANOVDB_ABI* grid_build_finalize)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_state_t* state,
        pnanovdb_compute_buffer_t* point_ijk_in,
        pnanovdb_compute_buffer_t* nanovdb_out,
        pnanovdb_uint64_t point_count,
        pnanovdb_uint64_t nanovdb_word_count,
        float voxel_size,
        pnanovdb_uint32_t dispatch_count);

    void (PNANOVDB_ABI* fanout_state_reset)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_fanout_state_t* state);

    pnanovdb_bool_t (PNANOVDB_ABI* fanout_state_valid)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_fanout_state_t* state);

    void (PNANOVDB_ABI* fanout_state_increment)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_fanout_state_t* state);

    void (PNANOVDB_ABI* fanout_state_init)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_fanout_state_t* state,
        pnanovdb_uint32_t prim_batch_size,
        pnanovdb_uint32_t ijk_batch_size,
        pnanovdb_uint32_t ijk_batch_count_max,
        pnanovdb_uint32_t prim_count,
        float voxel_size);

    void (PNANOVDB_ABI* fanout_state_destroy)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_fanout_state_t* state);

    void (PNANOVDB_ABI* fanout)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_grid_build_context_t* context,
        pnanovdb_grid_build_fanout_state_t* state,
        pnanovdb_uint32_t dispatch_count);

    const pnanovdb_compute_t* compute;

}pnanovdb_grid_build_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_grid_build_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(create_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(grid_build_init, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(grid_build_reset, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(grid_build_destroy, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(grid_build, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(grid_build_finalize, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(fanout_state_reset, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(fanout_state_valid, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(fanout_state_increment, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(fanout_state_init, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(fanout_state_destroy, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(fanout, 0, 0)
PNANOVDB_REFLECT_POINTER(pnanovdb_compute_t, compute, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_grid_build_t* (PNANOVDB_ABI* PFN_pnanovdb_get_grid_build)();

PNANOVDB_API pnanovdb_grid_build_t* pnanovdb_get_grid_build();

static void pnanovdb_grid_build_load(
    pnanovdb_grid_build_t* grid_build, const pnanovdb_compute_t* compute)
{
    auto get_grid_build = (PFN_pnanovdb_get_grid_build)
        pnanovdb_get_proc_address(compute->module, "pnanovdb_get_grid_build");
    if (!get_grid_build)
    {
        printf("Error: Failed to acquire grid build\n");
        return;
    }
    *grid_build = *get_grid_build();

    grid_build->compute = compute;
}

static void pnanovdb_grid_build_free(pnanovdb_grid_build_t* grid_build)
{
    // NOP for now
}

#endif
