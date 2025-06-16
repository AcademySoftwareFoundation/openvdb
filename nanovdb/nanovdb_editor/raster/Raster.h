// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/raster/Raster.h

    \author Andrew Reidmeyer

    \brief
*/

#include "Common.h"

#include "nanovdb_editor/putil/Raster.h"
#include "nanovdb_editor/putil/GridBuild.h"
#include "nanovdb_editor/putil/ParallelPrimitives.h"

namespace pnanovdb_raster
{

enum shader
{
    gaussian_frag_alloc_slang,
    gaussian_frag_color_slang,
    gaussian_prim_slang,
    point_frag_alloc_slang,
    point_frag_color_slang,

    gaussian_count_tiles_slang,
    gaussian_projection_slang,
    gaussian_rasterize_2d_slang,
    gaussian_spherical_harmonics_slang,
    gaussian_tile_intersections_slang,
    gaussian_tile_offsets_slang,

    shader_count
};

static const char* s_shader_names[shader_count] = {
    "raster/gaussian_frag_alloc.slang",
    "raster/gaussian_frag_color.slang",
    "raster/gaussian_prim.slang",
    "raster/point_frag_alloc.slang",
    "raster/point_frag_color.slang",

    "raster/gaussian_count_tiles.slang",
    "raster/gaussian_projection.slang",
    "raster/gaussian_rasterize_2d.slang",
    "raster/gaussian_spherical_harmonics.slang",
    "raster/gaussian_tile_intersections.slang",
    "raster/gaussian_tile_offsets.slang"
};

struct raster_context_t
{
    pnanovdb_shader_context_t* shader_ctx[shader_count];

    pnanovdb_parallel_primitives_t parallel_primitives;
    pnanovdb_parallel_primitives_context_t* parallel_primitives_ctx;
    pnanovdb_grid_build_t grid_build;
    pnanovdb_grid_build_context_t* grid_build_ctx;
};

PNANOVDB_CAST_PAIR(pnanovdb_raster_context_t, raster_context_t)

struct gaussian_data_t
{
    pnanovdb_uint64_t point_count;

    pnanovdb_bool_t has_uploaded;

    pnanovdb_compute_array_t* means_cpu_array;
    pnanovdb_compute_array_t* quaternions_cpu_array;
    pnanovdb_compute_array_t* scales_cpu_array;
    pnanovdb_compute_array_t* colors_cpu_array;
    pnanovdb_compute_array_t* spherical_harmonics_cpu_array;
    pnanovdb_compute_array_t* opacities_cpu_array;

    compute_gpu_array_t* means_gpu_array;
    compute_gpu_array_t* quaternions_gpu_array;
    compute_gpu_array_t* scales_gpu_array;
    compute_gpu_array_t* colors_gpu_array;
    compute_gpu_array_t* spherical_harmonics_gpu_array;
    compute_gpu_array_t* opacities_gpu_array;
};

PNANOVDB_CAST_PAIR(pnanovdb_raster_gaussian_data_t, gaussian_data_t)

pnanovdb_raster_context_t* create_context(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue);

void destroy_context(
    const pnanovdb_compute_t *compute,
    pnanovdb_compute_queue_t *queue,
    pnanovdb_raster_context_t *context);

pnanovdb_raster_gaussian_data_t* create_gaussian_data(
    const pnanovdb_compute_t* compute,
    pnanovdb_raster_context_t* context,
    pnanovdb_compute_array_t* means,
    pnanovdb_compute_array_t* quaternions,
    pnanovdb_compute_array_t* scales,
    pnanovdb_compute_array_t* colors,
    pnanovdb_compute_array_t* spherical_harmonics,
    pnanovdb_compute_array_t* opacities
);

void upload_gaussian_data(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context,
    pnanovdb_raster_gaussian_data_t* data
);

void destroy_gaussian_data(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context,
    pnanovdb_raster_gaussian_data_t* data
);

void raster_gaussian_2d(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context,
    pnanovdb_raster_gaussian_data_t* data,
    pnanovdb_compute_texture_t* color_2d,
    pnanovdb_uint32_t image_width,
    pnanovdb_uint32_t image_height,
    const pnanovdb_camera_mat_t* view,
    const pnanovdb_camera_mat_t* projection
);

void raster_gaussian_3d(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context,
    float voxel_size,
    pnanovdb_raster_gaussian_data_t* data,
    pnanovdb_compute_buffer_t* nanovdb_out,
    pnanovdb_uint64_t nanovdb_word_count
);

pnanovdb_raster_t* raster_file(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    const char* filename,
    pnanovdb_compute_array_t** nanovdb_arr,
    pnanovdb_raster_gaussian_data_t** gaussian_data,
    pnanovdb_raster_context_t** raster_context,
    pnanovdb_profiler_report_t profiler_report
);
}
