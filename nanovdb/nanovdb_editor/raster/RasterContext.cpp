// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/raster/RasterContext.cpp

    \author Andrew Reidmeyer

    \brief
*/

#define PNANOVDB_BUF_BOUNDS_CHECK
#include "Raster.h"

#include "nanovdb/PNanoVDB2.h"
#include "nanovdb_editor/putil/ThreadPool.hpp"

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <future>

namespace pnanovdb_raster
{

pnanovdb_raster_context_t* create_context(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue)
{
    raster_context_t* ctx = new raster_context_t();

    pnanovdb_parallel_primitives_load(&ctx->parallel_primitives, compute);
    ctx->parallel_primitives_ctx = ctx->parallel_primitives.create_context(compute, queue);
    if (!ctx->parallel_primitives_ctx)
    {
        return nullptr;
    }

    pnanovdb_grid_build_load(&ctx->grid_build, compute);
    ctx->grid_build_ctx = ctx->grid_build.create_context(compute, queue);
    if (!ctx->grid_build_ctx)
    {
        return nullptr;
    }

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

void destroy_context(
    const pnanovdb_compute_t *compute,
    pnanovdb_compute_queue_t *queue,
    pnanovdb_raster_context_t *context_in)
{
    auto ctx = cast(context_in);

    for (pnanovdb_uint32_t idx = 0u; idx < shader_count; idx++)
    {
        compute->destroy_shader_context(compute, queue, ctx->shader_ctx[idx]);
    }

    ctx->parallel_primitives.destroy_context(compute, queue, ctx->parallel_primitives_ctx);
    pnanovdb_parallel_primitives_free(&ctx->parallel_primitives);
    ctx->grid_build.destroy_context(compute, queue, ctx->grid_build_ctx);
    pnanovdb_grid_build_free(&ctx->grid_build);

    delete ctx;
}

pnanovdb_raster_gaussian_data_t* create_gaussian_data(
    const pnanovdb_compute_t* compute,
    pnanovdb_raster_context_t* context,
    pnanovdb_compute_array_t* means,
    pnanovdb_compute_array_t* quaternions,
    pnanovdb_compute_array_t* scales,
    pnanovdb_compute_array_t* colors,
    pnanovdb_compute_array_t* spherical_harmonics,
    pnanovdb_compute_array_t* opacities)
{
    auto ptr = new gaussian_data_t();

    ptr->point_count = means->element_count / 3u;

    ptr->has_uploaded = PNANOVDB_FALSE;

    ptr->means_cpu_array = compute->create_array(means->element_size, means->element_count, means->data);
    ptr->quaternions_cpu_array = compute->create_array(quaternions->element_size, quaternions->element_count, quaternions->data);
    ptr->scales_cpu_array = compute->create_array(scales->element_size, scales->element_count, scales->data);
    ptr->colors_cpu_array = compute->create_array(colors->element_size, colors->element_count, colors->data);
    ptr->spherical_harmonics_cpu_array = compute->create_array(spherical_harmonics->element_size, spherical_harmonics->element_count, spherical_harmonics->data);
    ptr->opacities_cpu_array = compute->create_array(opacities->element_size, opacities->element_count, opacities->data);

    ptr->means_gpu_array = gpu_array_create();
    ptr->quaternions_gpu_array = gpu_array_create();
    ptr->scales_gpu_array = gpu_array_create();
    ptr->colors_gpu_array = gpu_array_create();
    ptr->spherical_harmonics_gpu_array = gpu_array_create();
    ptr->opacities_gpu_array = gpu_array_create();

    return cast(ptr);
}

void upload_gaussian_data(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context,
    pnanovdb_raster_gaussian_data_t* data
)
{
    auto ptr = cast(data);

    if (!ptr->has_uploaded)
    {
        ptr->has_uploaded = PNANOVDB_TRUE;

        gpu_array_upload(compute, queue, ptr->means_gpu_array, ptr->means_cpu_array);
        gpu_array_upload(compute, queue, ptr->quaternions_gpu_array, ptr->quaternions_cpu_array);
        gpu_array_upload(compute, queue, ptr->scales_gpu_array, ptr->scales_cpu_array);
        gpu_array_upload(compute, queue, ptr->colors_gpu_array, ptr->colors_cpu_array);
        gpu_array_upload(compute, queue, ptr->spherical_harmonics_gpu_array, ptr->spherical_harmonics_cpu_array);
        gpu_array_upload(compute, queue, ptr->opacities_gpu_array, ptr->opacities_cpu_array);
    }
}

void destroy_gaussian_data(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_raster_context_t* context,
    pnanovdb_raster_gaussian_data_t* data
)
{
    auto ptr = cast(data);

    gpu_array_destroy(compute, queue, ptr->means_gpu_array);
    gpu_array_destroy(compute, queue, ptr->quaternions_gpu_array);
    gpu_array_destroy(compute, queue, ptr->scales_gpu_array);
    gpu_array_destroy(compute, queue, ptr->colors_gpu_array);
    gpu_array_destroy(compute, queue, ptr->spherical_harmonics_gpu_array);
    gpu_array_destroy(compute, queue, ptr->opacities_gpu_array);

    delete ptr;
}

}
