// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Raster.h

    \author Andrew Reidmeyer

    \brief  This file provides an interface for rasterization of voxels to NanoVDB.
*/

#ifndef NANOVDB_PUTILS_RASTER_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_RASTER_H_HAS_BEEN_INCLUDED

#include "nanovdb_editor/putil/Compute.h"
#include "nanovdb/putil/Camera.h"

/// ********************************* Raster ***************************************

struct pnanovdb_raster_context_t;
typedef struct pnanovdb_raster_context_t pnanovdb_raster_context_t;

struct pnanovdb_raster_gaussian_data_t;
typedef struct pnanovdb_raster_gaussian_data_t pnanovdb_raster_gaussian_data_t;

typedef struct pnanovdb_raster_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_raster_context_t* (PNANOVDB_ABI* create_context)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue);

    void (PNANOVDB_ABI* destroy_context)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_raster_context_t* context);

    pnanovdb_raster_gaussian_data_t*(PNANOVDB_ABI* create_gaussian_data)(
        const pnanovdb_compute_t* compute,
        pnanovdb_raster_context_t* context,
        pnanovdb_compute_array_t* means,
        pnanovdb_compute_array_t* quaternions,
        pnanovdb_compute_array_t* scales,
        pnanovdb_compute_array_t* colors,
        pnanovdb_compute_array_t* spherical_harmonics,
        pnanovdb_compute_array_t* opacities
    );

    void (PNANOVDB_ABI* upload_gaussian_data)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_raster_context_t* context,
        pnanovdb_raster_gaussian_data_t* data
    );

    void (PNANOVDB_ABI* destroy_gaussian_data)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_raster_context_t* context,
        pnanovdb_raster_gaussian_data_t* data
    );

    void(PNANOVDB_ABI* raster_gaussian_2d)(
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

    void(PNANOVDB_ABI* raster_gaussian_3d)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_raster_context_t* context,
        float voxel_size,
        pnanovdb_raster_gaussian_data_t* data,
        pnanovdb_compute_buffer_t* nanovdb_out,
        pnanovdb_uint64_t nanovdb_word_count,
        void* userdata
    );

    pnanovdb_compute_array_t*(PNANOVDB_ABI* raster_points)(
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
        void* userdata);

    const pnanovdb_compute_t* compute;

}pnanovdb_raster_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_raster_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(create_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_gaussian_data, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(upload_gaussian_data, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_gaussian_data, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(raster_gaussian_2d, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(raster_gaussian_3d, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(raster_points, 0, 0)
PNANOVDB_REFLECT_POINTER(pnanovdb_compute_t, compute, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_raster_t* (PNANOVDB_ABI* PFN_pnanovdb_get_raster)();

PNANOVDB_API pnanovdb_raster_t* pnanovdb_get_raster();

static void pnanovdb_raster_load(pnanovdb_raster_t* raster, const pnanovdb_compute_t* compute)
{
    auto get_raster = (PFN_pnanovdb_get_raster)pnanovdb_get_proc_address(compute->module, "pnanovdb_get_raster");
    if (!get_raster)
    {
        printf("Error: Failed to acquire raster\n");
        return;
    }
    *raster = *get_raster();

    raster->compute = compute;
}

static void pnanovdb_raster_free(pnanovdb_raster_t* raster)
{
    // NOP for now
}

#endif
