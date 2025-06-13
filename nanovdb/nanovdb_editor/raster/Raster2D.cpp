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

#include <stdlib.h>
#include <math.h>
#include <vector>

namespace pnanovdb_raster
{

static void extract_camera_info(
    pnanovdb_camera_mat_t view,
    pnanovdb_camera_mat_t proj,
    pnanovdb_vec3_t* view_dir_out,
    float* near_plane_out,
    float* far_plane_out)
{
    pnanovdb_camera_mat_t view_inv = pnanovdb_camera_mat_inverse(view);
    pnanovdb_camera_mat_t proj_inv = pnanovdb_camera_mat_inverse(proj);

    pnanovdb_vec4_t pos_d0 = pnanovdb_camera_vec4_transform(pnanovdb_vec4_t{0.f, 0.f, 0.f, 1.f}, proj_inv);
    pnanovdb_vec4_t pos_d1 = pnanovdb_camera_vec4_transform(pnanovdb_vec4_t{0.f, 0.f, 1.f, 1.f}, proj_inv);

    float z_d0 = pos_d0.z * (1.f / pos_d0.w);
    float z_d1 = pos_d1.z * (1.f / pos_d1.w);
    bool is_reverse_z = abs(z_d0) > abs(z_d1);
    pnanovdb_vec4_t ray_dir_near = is_reverse_z ? pos_d1 : pos_d0;

    pnanovdb_vec4_t ray_dir_far = pnanovdb_vec4_add(ray_dir_near,
        pnanovdb_camera_vec4_transform(pnanovdb_vec4_t{0.f, 0.f, 1.f, 0.f}, proj_inv));
    pnanovdb_vec3_t rayDir = {
        (ray_dir_far.x / ray_dir_far.w) - (ray_dir_near.x / ray_dir_near.w),
        (ray_dir_far.y / ray_dir_far.w) - (ray_dir_near.y / ray_dir_near.w),
        (ray_dir_far.z / ray_dir_far.w) - (ray_dir_near.z / ray_dir_near.w)
    };
    rayDir = pnanovdb_camera_vec3_normalize(rayDir);
    if (is_reverse_z)
    {
        rayDir.x = -rayDir.x;
        rayDir.y = -rayDir.y;
        rayDir.z = -rayDir.z;
    }

    pnanovdb_vec4_t rayDir4 = pnanovdb_camera_vec4_transform(
        pnanovdb_vec4_t{rayDir.x, rayDir.y, rayDir.z, 0.f}, view_inv);
    rayDir.x = rayDir4.x;
    rayDir.y = rayDir4.y;
    rayDir.z = rayDir4.z;

    *view_dir_out = rayDir;
    *near_plane_out = is_reverse_z ? z_d1 : z_d0;
    *far_plane_out = is_reverse_z ? z_d0 : z_d1;
}

void raster_gaussian_2d(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context_in,
    pnanovdb_raster_gaussian_data_t* data_in,
    pnanovdb_compute_texture_t* color_2d,
    pnanovdb_uint32_t image_width,
    pnanovdb_uint32_t image_height,
    const pnanovdb_camera_mat_t* view,
    const pnanovdb_camera_mat_t* projection
)
{
    auto ctx = cast(context_in);
    auto data = cast(data_in);

    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    upload_gaussian_data(compute, queue, context_in, data_in);

    struct constants_t
    {
        pnanovdb_camera_mat_t view;
        pnanovdb_vec4_t view_rot0;
        pnanovdb_vec4_t view_rot1;
        pnanovdb_vec4_t view_rot2;

        float near_plane;
        float far_plane;
        float fx;
        float fy;

        float cx;
        float cy;
        pnanovdb_uint32_t image_width;
        pnanovdb_uint32_t image_height;

        float eps2d;
        float radius_clip;
        pnanovdb_uint32_t prim_count;
        pnanovdb_uint32_t n_isects;

        pnanovdb_uint32_t image_origin_w;
        pnanovdb_uint32_t image_origin_h;
        pnanovdb_uint32_t tile_origin_w;
        pnanovdb_uint32_t tile_origin_h;

        pnanovdb_uint32_t tile_size;
        pnanovdb_uint32_t tile_width;
        pnanovdb_uint32_t tile_height;
        pnanovdb_uint32_t num_tiles_w;

        pnanovdb_vec3_t view_dir;
        pnanovdb_uint32_t num_tiles_h;

        pnanovdb_uint32_t num_tiles;
        pnanovdb_uint32_t pad1;
        pnanovdb_uint32_t pad2;
        pnanovdb_uint32_t pad3;
    };
    constants_t constants = {};

    pnanovdb_vec3_t view_dir = {};
    float near_plane = 0.f;
    float far_plane = 0.f;
    extract_camera_info(*view, *projection, &view_dir, &near_plane, &far_plane);

    constants.view = pnanovdb_camera_mat_transpose(*view);
    constants.view_rot0 = view->x;
    constants.view_rot1 = view->y;
    constants.view_rot2 = view->z;
    constants.near_plane = near_plane;
    constants.far_plane = far_plane;
    constants.fx = (float)image_width * projection->x.x;
    constants.fy = (float)image_height * projection->y.y;
    constants.cx = (float)image_width / 2.f;    // TODO: this could be extracted from proj matrix
    constants.cy = (float)image_height / 2.f;
    constants.image_width = image_width;
    constants.image_height = image_height;
    constants.eps2d = 0.3f;
    constants.radius_clip = 0.f;
    constants.prim_count = data->point_count;
    constants.n_isects = 0u;
    constants.image_origin_w = 0u;
    constants.image_origin_h = 0u;
    constants.tile_origin_w = 0u;
    constants.tile_origin_h = 0u;
    constants.tile_size = 16u;
    constants.tile_width = (image_width + 15u) / 16u;
    constants.tile_height = (image_height + 15u) / 16u;
    constants.num_tiles_w = (image_width + 15u) / 16u;
    constants.view_dir = view_dir;
    constants.num_tiles_h = (image_height + 15u) / 16u;
    constants.num_tiles = constants.num_tiles_w * constants.num_tiles_h;

    //printf("fx(%f) fy(%f) cx(%f) cy(%f)\n", constants.fx, constants.fy, constants.cx, constants.cy);

    pnanovdb_compute_buffer_desc_t buf_desc = {};
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

    pnanovdb_compute_buffer_transient_t* means_transient =
        compute_interface->register_buffer_as_transient(context, data->means_gpu_array->device_buffer);
    pnanovdb_compute_buffer_transient_t* quats_transient =
        compute_interface->register_buffer_as_transient(context, data->quaternions_gpu_array->device_buffer);
    pnanovdb_compute_buffer_transient_t* scales_transient =
        compute_interface->register_buffer_as_transient(context, data->scales_gpu_array->device_buffer);
    pnanovdb_compute_buffer_transient_t* colors_transient =
        compute_interface->register_buffer_as_transient(context, data->colors_gpu_array->device_buffer);
    pnanovdb_compute_buffer_transient_t* spherical_harmonics_transient =
        compute_interface->register_buffer_as_transient(context, data->spherical_harmonics_gpu_array->device_buffer);
    pnanovdb_compute_buffer_transient_t* opacities_transient =
        compute_interface->register_buffer_as_transient(context, data->opacities_gpu_array->device_buffer);

    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED |
        PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 4u * constants.prim_count;
    pnanovdb_compute_buffer_t* radii_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.structure_stride = 8u;
    buf_desc.size_in_bytes = 8u * constants.prim_count;
    pnanovdb_compute_buffer_t* means2d_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 4u * constants.prim_count;
    pnanovdb_compute_buffer_t* depths_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 12u * constants.prim_count;
    pnanovdb_compute_buffer_t* conics_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 4u * constants.prim_count;
    pnanovdb_compute_buffer_t* compensations_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 12u * constants.prim_count;
    pnanovdb_compute_buffer_t* resolved_color_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 4u * constants.prim_count;
    pnanovdb_compute_buffer_t* num_tiles_per_gaussian_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* scan_tiles_per_gaussian_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    pnanovdb_compute_buffer_transient_t* radii_transient =
        compute_interface->register_buffer_as_transient(context, radii_buffer);
    pnanovdb_compute_buffer_transient_t* means2d_transient =
        compute_interface->register_buffer_as_transient(context, means2d_buffer);
    pnanovdb_compute_buffer_transient_t* depths_transient =
        compute_interface->register_buffer_as_transient(context, depths_buffer);
    pnanovdb_compute_buffer_transient_t* conics_transient =
        compute_interface->register_buffer_as_transient(context, conics_buffer);
    pnanovdb_compute_buffer_transient_t* compensations_transient =
        compute_interface->register_buffer_as_transient(context, compensations_buffer);
    pnanovdb_compute_buffer_transient_t* resolved_color_transient =
        compute_interface->register_buffer_as_transient(context, resolved_color_buffer);
    pnanovdb_compute_buffer_transient_t* num_tiles_per_gaussian_transient =
        compute_interface->register_buffer_as_transient(context, num_tiles_per_gaussian_buffer);
    pnanovdb_compute_buffer_transient_t* scan_tiles_per_gaussian_transient =
        compute_interface->register_buffer_as_transient(context, scan_tiles_per_gaussian_buffer);

    // projection
    {
        pnanovdb_compute_resource_t resources[9u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = means_transient;
        resources[2u].buffer_transient = quats_transient;
        resources[3u].buffer_transient = scales_transient;
        resources[4u].buffer_transient = radii_transient;
        resources[5u].buffer_transient = means2d_transient;
        resources[6u].buffer_transient = depths_transient;
        resources[7u].buffer_transient = conics_transient;
        resources[8u].buffer_transient = compensations_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_projection_slang],
            resources,
            (constants.prim_count + 255u) / 256u, 1u, 1u,
            "gaussian_projection"
        );
    }

    // spherical harmonics
    {
        pnanovdb_compute_resource_t resources[3u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = spherical_harmonics_transient;
        resources[2u].buffer_transient = resolved_color_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_spherical_harmonics_slang],
            resources,
            (constants.prim_count + 255u) / 256u, 1u, 1u,
            "gaussian_spherical_harmonics"
        );
    }

    // counts tiles
    {
        pnanovdb_compute_resource_t resources[4u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = means2d_transient;
        resources[2u].buffer_transient = radii_transient;
        resources[3u].buffer_transient = num_tiles_per_gaussian_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_count_tiles_slang],
            resources,
            (constants.prim_count + 255u) / 256u, 1u, 1u,
            "gaussian_count_tiles"
        );
    }

    // prefix sum tile counts
    {
        ctx->parallel_primitives.global_scan(
            compute,
            queue,
            ctx->parallel_primitives_ctx,
            num_tiles_per_gaussian_buffer,
            scan_tiles_per_gaussian_buffer,
            constants.prim_count,
            1u
        );
    }

    // readback total count, allocate key/val buffers
    pnanovdb_uint32_t total_count = 0u;
    {
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
        buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
        buf_desc.structure_stride = 4u;
        buf_desc.size_in_bytes = 4u;
        pnanovdb_compute_buffer_t* readback_buffer =
            compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK, &buf_desc);

        pnanovdb_compute_copy_buffer_params_t copy_params = {};
        copy_params.num_bytes = 4u;     // uint32_t
        copy_params.src = compute_interface->register_buffer_as_transient(context, scan_tiles_per_gaussian_buffer);
        copy_params.src_offset = (constants.prim_count - 1u) * 4u;
        copy_params.dst = compute_interface->register_buffer_as_transient(context, readback_buffer);
        copy_params.dst_offset = 0u;
        copy_params.debug_label = "raster_2d_feedback_copy";
        compute_interface->copy_buffer(context, &copy_params);

        pnanovdb_uint64_t flushed_frame = 0llu;
        compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

        compute->device_interface.wait_idle(queue);

        pnanovdb_uint32_t* mapped = (pnanovdb_uint32_t*)compute_interface->map_buffer(context, readback_buffer);

        total_count = mapped[0u];

        compute_interface->unmap_buffer(context, readback_buffer);

        compute_interface->destroy_buffer(context, readback_buffer);
    }

    // update constants with total count
    constants.n_isects = total_count;

    //printf("raster_2d total_intersections(%u)\n", total_count);

    compute_interface->destroy_buffer(context, constant_buffer);
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 0u;
    buf_desc.size_in_bytes = sizeof(constants_t);
    constant_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

    mapped_constants = compute_interface->map_buffer(context, constant_buffer);
    memcpy(mapped_constants, &constants, sizeof(constants_t));
    compute_interface->unmap_buffer(context, constant_buffer);

    // due to flush, transients must be refreshed
    constant_transient = compute_interface->register_buffer_as_transient(context, constant_buffer);

    means_transient = compute_interface->register_buffer_as_transient(context, data->means_gpu_array->device_buffer);
    quats_transient = compute_interface->register_buffer_as_transient(context, data->quaternions_gpu_array->device_buffer);
    scales_transient = compute_interface->register_buffer_as_transient(context, data->scales_gpu_array->device_buffer);
    colors_transient = compute_interface->register_buffer_as_transient(context, data->colors_gpu_array->device_buffer);
    spherical_harmonics_transient = compute_interface->register_buffer_as_transient(context, data->spherical_harmonics_gpu_array->device_buffer);
    opacities_transient = compute_interface->register_buffer_as_transient(context, data->opacities_gpu_array->device_buffer);

    radii_transient = compute_interface->register_buffer_as_transient(context, radii_buffer);
    means2d_transient = compute_interface->register_buffer_as_transient(context, means2d_buffer);
    depths_transient = compute_interface->register_buffer_as_transient(context, depths_buffer);
    conics_transient = compute_interface->register_buffer_as_transient(context, conics_buffer);
    compensations_transient = compute_interface->register_buffer_as_transient(context, compensations_buffer);
    resolved_color_transient = compute_interface->register_buffer_as_transient(context, resolved_color_buffer);
    num_tiles_per_gaussian_transient = compute_interface->register_buffer_as_transient(context, num_tiles_per_gaussian_buffer);
    scan_tiles_per_gaussian_transient = compute_interface->register_buffer_as_transient(context, scan_tiles_per_gaussian_buffer);

    // create sort keys/vals
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 65536u;
    while (buf_desc.size_in_bytes < 4u * constants.n_isects)
    {
        buf_desc.size_in_bytes *= 2u;
    }
    pnanovdb_compute_buffer_t* intersection_keys_low_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* intersection_keys_high_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    pnanovdb_compute_buffer_t* intersection_vals_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    pnanovdb_compute_buffer_transient_t* intersection_keys_low_transient =
        compute_interface->register_buffer_as_transient(context, intersection_keys_low_buffer);
    pnanovdb_compute_buffer_transient_t* intersection_keys_high_transient =
        compute_interface->register_buffer_as_transient(context, intersection_keys_high_buffer);
    pnanovdb_compute_buffer_transient_t* intersection_vals_transient =
        compute_interface->register_buffer_as_transient(context, intersection_vals_buffer);

    // tile intersections
    {
        pnanovdb_compute_resource_t resources[8u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = means2d_transient;
        resources[2u].buffer_transient = radii_transient;
        resources[3u].buffer_transient = depths_transient;
        resources[4u].buffer_transient = scan_tiles_per_gaussian_transient;
        resources[5u].buffer_transient = intersection_keys_low_transient;
        resources[6u].buffer_transient = intersection_keys_high_transient;
        resources[7u].buffer_transient = intersection_vals_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_tile_intersections_slang],
            resources,
            (constants.prim_count + 255u) / 256u, 1u, 1u,
            "gaussian_tile_intersections"
        );
    }

    // radix sort
    {
        pnanovdb_uint32_t num_tile_id_bits = 0u;
        while ((1u << num_tile_id_bits) < constants.num_tiles)
        {
            num_tile_id_bits++;
        }

        ctx->parallel_primitives.radix_sort_dual_key(
            compute,
            queue,
            ctx->parallel_primitives_ctx,
            intersection_keys_low_buffer,
            intersection_keys_high_buffer,
            intersection_vals_buffer,
            constants.n_isects,
            32u,
            num_tile_id_bits
        );
    }

    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = 4u * constants.num_tiles;
    if (buf_desc.size_in_bytes < 65536u)
    {
        buf_desc.size_in_bytes = 65536u;
    }
    pnanovdb_compute_buffer_t* tile_offsets_buffer =
        compute_interface->create_buffer(context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    pnanovdb_compute_buffer_transient_t* tile_offsets_transient =
        compute_interface->register_buffer_as_transient(context, tile_offsets_buffer);

    // compute tile offsets
    {
        pnanovdb_compute_resource_t resources[3u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = intersection_keys_high_transient;
        resources[2u].buffer_transient = tile_offsets_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_tile_offsets_slang],
            resources,
            (constants.n_isects + 255u) / 256u, 1u, 1u,
            "gaussian_tile_offsets"
        );
    }

    pnanovdb_compute_texture_transient_t* color_2d_transient =
        compute_interface->register_texture_as_transient(context, color_2d);

    // raster
    {
        pnanovdb_compute_resource_t resources[8u] = {};
        resources[0u].buffer_transient = constant_transient;
        resources[1u].buffer_transient = means2d_transient;
        resources[2u].buffer_transient = conics_transient;
        resources[3u].buffer_transient = resolved_color_transient;
        resources[4u].buffer_transient = opacities_transient;
        resources[5u].buffer_transient = tile_offsets_transient;
        resources[6u].buffer_transient = intersection_vals_transient;
        resources[7u].texture_transient = color_2d_transient;

        compute->dispatch_shader(
            compute_interface,
            context,
            ctx->shader_ctx[gaussian_rasterize_2d_slang],
            resources,
            (constants.image_height + 15u) / 16u, (constants.image_width + 15u) / 16u, 1u,
            "gaussian_rasterize_2d"
        );
    }

#if 0
    pnanovdb_compute_array_t* debug_cpu = compute->create_array(4u, constants.num_tiles, nullptr);
    compute_gpu_array_t* debug = gpu_array_create();
    gpu_array_alloc_device(compute, queue, debug, debug_cpu);
    gpu_array_copy(compute, queue, debug, tile_offsets_buffer, 0u, 4u * constants.num_tiles);
    gpu_array_readback(compute, queue, debug, debug_cpu);

    // flush for debug
    {
        pnanovdb_uint64_t flushed_frame = 0llu;
        compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

        compute->device_interface.wait_idle(queue);
    }
    gpu_array_map(compute, queue, debug, debug_cpu);
    pnanovdb_uint32_t* debug_mapped = (pnanovdb_uint32_t*)compute->map_array(debug_cpu);
    pnanovdb_uint32_t debug_zero_count = 0u;
    for (pnanovdb_uint32_t idx = 0u; idx < constants.num_tiles; idx++)
    {
        if (debug_mapped[idx] == 0u) {debug_zero_count++;}
    }
    printf("debug_zero_count(%d) total_count(%d)\n", debug_zero_count, constants.num_tiles);
    compute->unmap_array(debug_cpu);
    gpu_array_destroy(compute, queue, debug);
    debug = nullptr;
    compute->destroy_array(debug_cpu);
    debug_cpu = nullptr;
#endif

    compute_interface->destroy_buffer(context, constant_buffer);

    compute_interface->destroy_buffer(context, tile_offsets_buffer);

    compute_interface->destroy_buffer(context, intersection_keys_low_buffer);
    compute_interface->destroy_buffer(context, intersection_keys_high_buffer);
    compute_interface->destroy_buffer(context, intersection_vals_buffer);

    compute_interface->destroy_buffer(context, radii_buffer);
    compute_interface->destroy_buffer(context, means2d_buffer);
    compute_interface->destroy_buffer(context, depths_buffer);
    compute_interface->destroy_buffer(context, conics_buffer);
    compute_interface->destroy_buffer(context, compensations_buffer);
    compute_interface->destroy_buffer(context, resolved_color_buffer);
    compute_interface->destroy_buffer(context, num_tiles_per_gaussian_buffer);
    compute_interface->destroy_buffer(context, scan_tiles_per_gaussian_buffer);
}

}
