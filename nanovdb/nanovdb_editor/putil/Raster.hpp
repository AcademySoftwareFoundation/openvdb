// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Raster.hpp

    \author Petra Hapalova

    \brief
*/

#pragma once

#include "nanovdb_editor/putil/Raster.h"
#include "nanovdb_editor/putil/WorkerThread.hpp"
#include "nanovdb_editor/putil/FileFormat.h"

#define PNANOVDB_RASTER_TEST_CONVERT 1
#if PNANOVDB_RASTER_TEST_CONVERT
#include "nanovdb/putil/Convert.h"
#endif

namespace pnanovdb_raster
{
bool raster_file(
    pnanovdb_raster_t* raster,
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    const char* filename,
    float voxel_size,
    pnanovdb_compute_array_t** nanovdb_arr,
    pnanovdb_raster_gaussian_data_t** gaussian_data,
    pnanovdb_raster_context_t** raster_context,
    pnanovdb_profiler_report_t profiler_report,
    void* userdata)
{
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    auto log_print = compute_interface->get_log_print(context);

    pnanovdb_util::WorkerThread* worker = static_cast<pnanovdb_util::WorkerThread*>(userdata);
    if (worker)
    {
        worker->updateTaskProgress(0.f, "Reading file");
    }

    pnanovdb_fileformat_t fileformat = {};
    pnanovdb_fileformat_load(&fileformat, compute);

    const char* array_names_gaussian[] = { "means", "opacities", "quaternions", "scales", "sh" };
    pnanovdb_compute_array_t* arrays_gaussian[5] = {};

    pnanovdb_bool_t loaded_gaussian = fileformat.load_file(filename, 5, array_names_gaussian, arrays_gaussian);
    if (loaded_gaussian == PNANOVDB_TRUE)
    {
        pnanovdb_uint64_t point_count = arrays_gaussian[0]->element_count / 3u;

        pnanovdb_compute_array_t* means_arr = arrays_gaussian[0];
        pnanovdb_compute_array_t* opacity_arr = arrays_gaussian[1];
        pnanovdb_compute_array_t* quat_arr = arrays_gaussian[2];
        pnanovdb_compute_array_t* scale_arr = arrays_gaussian[3];
        pnanovdb_compute_array_t* spherical_harmonics_arr = arrays_gaussian[4];
        pnanovdb_compute_array_t* color_arr = compute->create_array(4u, point_count * 3u, nullptr);

        if (worker)
        {
            worker->updateTaskProgress(0.4f);
        }

        float* mapped_quat = (float*)compute->map_array(quat_arr);
        for (pnanovdb_uint64_t point_idx = 0u; point_idx < point_count; point_idx++)
        {
            float x = mapped_quat[4u * point_idx + 1u];
            float y = mapped_quat[4u * point_idx + 2u];
            float z = mapped_quat[4u * point_idx + 3u];
            float w = mapped_quat[4u * point_idx + 0u];

            float magn_inv = 1.f / sqrtf(x * x + y * y + z * z + w * w);

            mapped_quat[4u * point_idx + 1u] = x * magn_inv;
            mapped_quat[4u * point_idx + 2u] = y * magn_inv;
            mapped_quat[4u * point_idx + 3u] = z * magn_inv;
            mapped_quat[4u * point_idx + 0u] = w * magn_inv;
        }
        compute->unmap_array(quat_arr);

        if (worker)
        {
            worker->updateTaskProgress(0.5f);
        }

        float* mapped_scale = (float*)compute->map_array(scale_arr);
        for (pnanovdb_uint64_t point_idx = 0u; point_idx < point_count; point_idx++)
        {
            mapped_scale[3u * point_idx + 0u] = expf(mapped_scale[3u * point_idx + 0u]);
            mapped_scale[3u * point_idx + 1u] = expf(mapped_scale[3u * point_idx + 1u]);
            mapped_scale[3u * point_idx + 2u] = expf(mapped_scale[3u * point_idx + 2u]);
        }
        compute->unmap_array(scale_arr);

        if (worker)
        {
            worker->updateTaskProgress(0.7f);
        }

        float* mapped_spherical_harmonics = (float*)compute->map_array(spherical_harmonics_arr);
        float* mapped_color = (float*)compute->map_array(color_arr);
        for (pnanovdb_uint64_t point_idx = 0u; point_idx < point_count; point_idx++)
        {
            const float c0 = 0.28209479177387814f;
            mapped_color[3u * point_idx + 0u] = c0 * mapped_spherical_harmonics[48u * point_idx + 0u] + 0.5f;
            mapped_color[3u * point_idx + 1u] = c0 * mapped_spherical_harmonics[48u * point_idx + 1u] + 0.5f;
            mapped_color[3u * point_idx + 2u] = c0 * mapped_spherical_harmonics[48u * point_idx + 2u] + 0.5f;
        }
        compute->unmap_array(spherical_harmonics_arr);
        compute->unmap_array(color_arr);

        if (worker)
        {
            worker->updateTaskProgress(0.8f);
        }

        float* mapped_opacity = (float*)compute->map_array(opacity_arr);
        for (pnanovdb_uint64_t point_idx = 0u; point_idx < point_count; point_idx++)
        {
            mapped_opacity[point_idx] = 1.f / (1.f + expf(-mapped_opacity[point_idx]));
        }
        compute->unmap_array(opacity_arr);

        if (worker)
        {
            worker->updateTaskProgress(0.9f);
        }

        compute->compute_array_print_range(compute, log_print, "means", means_arr, 3u);
        compute->compute_array_print_range(compute, log_print, "quats", quat_arr, 4u);
        compute->compute_array_print_range(compute, log_print, "scales", scale_arr, 3u);
        compute->compute_array_print_range(compute, log_print, "colors", color_arr, 3u);
        compute->compute_array_print_range(compute, log_print, "opacities", opacity_arr, 1u);

        if (worker)
        {
            worker->updateTaskProgress(1.f);
        }

        if (gaussian_data != nullptr)
        {
            if (worker)
            {
                worker->updateTaskProgress(0.f, "Creating gaussian data");
            }

            pnanovdb_raster_context_t* raster_ctx = raster->create_context(raster->compute, queue);
            if (raster_ctx == nullptr)
            {
                if (log_print)
                {
                    log_print(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "Failed to create raster context for Gaussian data from file '%s'", filename);
                }
                pnanovdb_fileformat_free(&fileformat);
                return false;
            }
            pnanovdb_raster_gaussian_data_t* raster_data = raster->create_gaussian_data(
                raster->compute,
                raster_ctx,
                means_arr, quat_arr, scale_arr,
                color_arr, spherical_harmonics_arr, opacity_arr
            );
            *gaussian_data = raster_data;

            if (raster_context != nullptr)
            {
                *raster_context = raster_ctx;
            }
            else
            {
                raster->destroy_context(raster->compute, queue, raster_ctx);
            }
        }
        else if (nanovdb_arr != nullptr)
        {
            *nanovdb_arr = raster->raster_points(
                raster->compute,
                queue,
                voxel_size,
                means_arr,
                quat_arr,
                scale_arr,
                color_arr,
                spherical_harmonics_arr,
                opacity_arr,
                profiler_report,
                userdata
            );
            if (*nanovdb_arr == nullptr)
            {
                if (log_print)
                {
                    log_print(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "Failed to rasterize Gaussian data from file '%s'", filename);
                }
                pnanovdb_fileformat_free(&fileformat);
                return false;
            }
#if PNANOVDB_RASTER_TEST_CONVERT
            pnanovdb_compute_array_t* node2_arr = *nanovdb_arr;
            *nanovdb_arr = compute->create_array(4u, 3u  * 256u * 1024u * 1024u, nullptr);

            pnanovdb_uint32_t* mapped_src = (pnanovdb_uint32_t*)compute->map_array(node2_arr);
            pnanovdb_uint32_t* mapped_dst = (pnanovdb_uint32_t*)compute->map_array(*nanovdb_arr);

            pnanovdb_buf_t src_buf = pnanovdb_make_buf(mapped_src, node2_arr->element_count);
            pnanovdb_buf_t dst_buf = pnanovdb_make_buf(mapped_dst, (*nanovdb_arr)->element_count);

            pnanovdb_address_t dst_addr_max = {4u * 3u  * 256u * 1024u * 1024u};
            pnanovdb_node2_convert_to_grid_type(dst_buf, dst_addr_max, PNANOVDB_GRID_TYPE_ONINDEX, src_buf);

            compute->unmap_array(*nanovdb_arr);
            compute->unmap_array(node2_arr);

            compute->destroy_array(node2_arr);
#endif
        }

        compute->destroy_array(means_arr);
        compute->destroy_array(quat_arr);
        compute->destroy_array(scale_arr);
        compute->destroy_array(color_arr);
        compute->destroy_array(spherical_harmonics_arr);
        compute->destroy_array(opacity_arr);

        pnanovdb_fileformat_free(&fileformat);

        if (log_print)
        {
            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Rasterized Gaussian data from file '%s'", filename);
        }
        return true;
    }

    const char* array_names_svraster[] = { "corner_ijk", "corner_indices", "levels", "morton_keys", "per_corner_opacities", "per_voxel_sh0", "per_voxel_shN"};
    pnanovdb_compute_array_t* arrays_svraster[7] = {};

    pnanovdb_bool_t loaded_svraster = fileformat.load_file(filename, 7, array_names_svraster, arrays_svraster);
    if (loaded_svraster == PNANOVDB_TRUE)
    {
        // TODO: svraster code

        pnanovdb_fileformat_free(&fileformat);

        if (log_print)
        {
            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "Rasterized data from file '%s'", filename);
        }
        return true;
    }

    if (log_print)
    {
        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "Failed to load Gaussian data from file '%s'", filename);
    }
    return false;
}
}
