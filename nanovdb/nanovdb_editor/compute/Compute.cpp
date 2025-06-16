// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/compute/Compute.cpp

    \author Petra Hapalova

    \brief
*/

#include "Compute.h"
#include "ComputeShader.h"

#include <nanovdb/io/IO.h>

#include <stdio.h>

namespace pnanovdb_compute
{
static void compute_profiler_report(void* userdata, pnanovdb_uint64_t capture_id, pnanovdb_uint32_t num_entries, pnanovdb_compute_profiler_entry_t* entries)
{
    const char* name = (const char*)userdata;
    printf("'%s' profiler results capture_id(%llu):\n", name, (unsigned long long int)capture_id);
    for (pnanovdb_uint32_t idx = 0u; idx < num_entries; idx++)
    {
        printf("[%d] name(%s) cpu_ms(%f) gpu_ms(%f)\n",
            idx, entries[idx].label, 1000.f * entries[idx].cpu_delta_time, 1000.f * entries[idx].gpu_delta_time);
    }
}

struct shader_context_t
{
    pnanovdb_compute_shader_t* shader;
    pnanovdb_compute_shader_build_t* shader_build;
    pnanovdb_compute_pipeline_t* pipeline;
    pnanovdb_compute_shader_source_t source;
};

PNANOVDB_CAST_PAIR(pnanovdb_shader_context_t, shader_context_t)

pnanovdb_shader_context_t* create_shader_context(const char* filename)
{
    shader_context_t* ctx = new shader_context_t();

    ctx->source.source_filename = filename;

    return cast(ctx);
}

void destroy_shader_context(const pnanovdb_compute_t* compute, pnanovdb_compute_queue_t* queue, pnanovdb_shader_context_t* context)
{
    shader_context_t* ctx = cast(context);
    if (ctx)
    {
        compute->destroy_shader(compute->device_interface.get_compute_interface(queue), &compute->shader_interface, compute->device_interface.get_compute_context(queue), context);
        assert(ctx->shader == nullptr);
        delete ctx;
    }
}

pnanovdb_compute_array_t* load_nanovdb(const char* filepath)
{
    nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle;
    try
    {
        gridHandle = nanovdb::io::readGrid(filepath, 0);
    }
    catch (const std::ios_base::failure& e)
    {
        printf("Error: Could not open nanovdb '%s'", filepath);
        return nullptr;
    }

    pnanovdb_compute_array_t* array = new pnanovdb_compute_array_t();
    array->filepath = filepath;
    array->element_size = sizeof(pnanovdb_uint32_t);
    array->element_count = gridHandle.bufferSize() / array->element_size;
    array->data = new char[gridHandle.bufferSize()];
    memcpy(array->data, gridHandle.data(), gridHandle.bufferSize());
    return array;
}

pnanovdb_bool_t save_nanovdb(pnanovdb_compute_array_t* array, const char* filepath)
{
    nanovdb::HostBuffer buffer(array->element_count * array->element_size);
    memcpy(buffer.data(), array->data, array->element_count * array->element_size);

    nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle(std::move(buffer));
    try
    {
        nanovdb::io::writeGrid(filepath, gridHandle);
    }
    catch (const std::ios_base::failure& e)
    {
        printf("Error: Could not save nanovdb '%s' (%s)\n", filepath, e.what());
        return PNANOVDB_FALSE;
    }
    return PNANOVDB_TRUE;
}

pnanovdb_bool_t init_shader(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_queue_t* queue,
    pnanovdb_shader_context_t* shaderContext,
    pnanovdb_compiler_settings_t* compileSettings)
{
    const pnanovdb_compute_shader_interface_t* shader_interface = &compute->shader_interface;
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(queue);

    auto log_print = compute_interface->get_log_print(context);

    shader_context_t* shader_ctx = cast(shaderContext);

    pnanovdb_bool_t shader_updated = PNANOVDB_FALSE;
    if (compute->compiler)
    {
        // shader will be recompiled only if the source has changed
        pnanovdb_bool_t result = compute->compiler->compile_shader_from_file(nullptr, shader_ctx->source.source_filename, compileSettings, &shader_updated);
        if (shader_updated && log_print)
        {
            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "shader '%s' updated", shader_ctx->source.source_filename);
        }
    }
    if (shader_updated == PNANOVDB_TRUE)
    {
        compute->destroy_shader(compute_interface, shader_interface, context, shaderContext);
    }

    shader_ctx->shader = shader_interface->create_shader(&shader_ctx->source);
    shader_ctx->shader_build = nullptr;
    bool result = shader_interface->map_shader_build(shader_ctx->shader, &shader_ctx->shader_build);
    if (!result && log_print)
    {
        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "mapping shader build failed, check the shader exists and is compiled");
        return PNANOVDB_FALSE;
    }

    if (log_print)
    {
        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "shader(%s) descriptor_write_count(%d) byte_code_size(%d)",
            shader_ctx->shader_build->debug_label,
            shader_ctx->shader_build->descriptor_write_count,
            (pnanovdb_uint32_t)shader_ctx->shader_build->pipeline_desc.bytecode.size_in_bytes);
        for (uint32_t idx = 0u; idx < shader_ctx->shader_build->descriptor_write_count; idx++)
        {
            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "resource[%d] name(%s)",
                idx, shader_ctx->shader_build->resource_names[idx]);
        }
    }

    shader_ctx->pipeline = compute_interface->create_compute_pipeline(context, &shader_ctx->shader_build->pipeline_desc);
    return PNANOVDB_TRUE;
}

void destroy_shader(
    pnanovdb_compute_interface_t* computeInterface,
    const pnanovdb_compute_shader_interface_t* shaderInterface,
    pnanovdb_compute_context_t* computeContext,
    pnanovdb_shader_context_t* shaderContext)
{
    if (!shaderContext)
    {
        return;
    }

    shader_context_t* shader = cast(shaderContext);
    if (shader->pipeline)
    {
        computeInterface->destroy_compute_pipeline(computeContext, shader->pipeline);
        shader->pipeline = nullptr;
    }
    if (shader->shader)
    {
        shaderInterface->destroy_shader(shader->shader);
        shader->shader = nullptr;
    }
}

void dispatch_shader(
    pnanovdb_compute_interface_t* contextInterface,
    pnanovdb_compute_context_t* computeContext,
    const pnanovdb_shader_context_t* shaderContext,
    pnanovdb_compute_resource_t* resources,
    pnanovdb_uint32_t grid_dim_x,
    pnanovdb_uint32_t grid_dim_y,
    pnanovdb_uint32_t grid_dim_z,
    const char* debug_label
)
{
    auto shader = cast(shaderContext);

    pnanovdb_compute_dispatch_params_t dispatch_params = {};
    dispatch_params.pipeline = shader->pipeline;
    dispatch_params.grid_dim_x = grid_dim_x;
    dispatch_params.grid_dim_y = grid_dim_y;
    dispatch_params.grid_dim_z = grid_dim_z;

    dispatch_params.descriptor_writes = shader->shader_build->descriptor_writes;
    dispatch_params.resources = resources;
    dispatch_params.descriptor_write_count = shader->shader_build->descriptor_write_count;

    dispatch_params.debug_label = debug_label ? debug_label : shader->shader_build->debug_label;

    contextInterface->dispatch(computeContext, &dispatch_params);
}

pnanovdb_bool_t dispatch_shader_on_nanovdb_array(
    const pnanovdb_compute_t* compute,
    const pnanovdb_compute_device_t* device,
    const pnanovdb_shader_context_t* shader_context,
    pnanovdb_compute_array_t* nanovdb_array,
    pnanovdb_int32_t image_width,
    pnanovdb_int32_t image_height,
    pnanovdb_compute_texture_t* background_image,
    pnanovdb_compute_buffer_transient_t* upload_buffer,
    pnanovdb_compute_buffer_transient_t* user_upload_buffer,
    pnanovdb_compute_buffer_t** nanovdb_buffer,
    pnanovdb_compute_buffer_transient_t** readback_buffer)
{
    if (!nanovdb_array)
    {
        return PNANOVDB_FALSE;
    }
    const pnanovdb_uint64_t size_in_bytes = nanovdb_array->element_count * nanovdb_array->element_size;
    if (size_in_bytes == 0)
    {
        return PNANOVDB_FALSE;
    }
    auto shader = cast(shader_context);
    if (!shader->shader_build)
    {
        return PNANOVDB_FALSE;
    }

    pnanovdb_compute_queue_t* queue = compute->device_interface.get_device_queue(device);
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* compute_context = compute->device_interface.get_compute_context(queue);

    pnanovdb_compute_buffer_t* nanovdb_upload_buffer = nullptr;
    if (nanovdb_array && *nanovdb_buffer == nullptr)
    {
        // nanovdb buffer to upload
        pnanovdb_compute_buffer_desc_t upload_desc = {};
        upload_desc.size_in_bytes = size_in_bytes;
        upload_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC;

        pnanovdb_compute_buffer_t* nanovdb_upload_buffer = compute_interface->create_buffer(compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &upload_desc);
        if (!nanovdb_upload_buffer)
        {
            return PNANOVDB_FALSE;
        }
        void* mapped_upload = compute_interface->map_buffer(compute_context, nanovdb_upload_buffer);
        memcpy(mapped_upload, nanovdb_array->data, size_in_bytes);
        compute_interface->unmap_buffer(compute_context, nanovdb_upload_buffer);

        // uploaded nanovdb buffer
        pnanovdb_compute_buffer_desc_t buf_desc = {};
        buf_desc.size_in_bytes = size_in_bytes;
        buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
        buf_desc.structure_stride = nanovdb_array->element_size;

        *nanovdb_buffer = compute_interface->create_buffer(compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
        if (*nanovdb_buffer == nullptr)
        {
            return PNANOVDB_FALSE;
        }

        // upload nanovdb
        pnanovdb_compute_copy_buffer_params_t upload_params = {};
        upload_params.num_bytes = size_in_bytes;
        upload_params.src = compute_interface->register_buffer_as_transient(compute_context, nanovdb_upload_buffer);
        upload_params.dst = compute_interface->register_buffer_as_transient(compute_context, *nanovdb_buffer);
        upload_params.debug_label = "dispatch_shader_on_nanovdb_array_upload";
        compute_interface->copy_buffer(compute_context, &upload_params);
    }

    pnanovdb_compute_buffer_desc_t image_buf_desc = {};
    image_buf_desc.size_in_bytes = pnanovdb_uint64_t(image_width * image_height * 4u);
    image_buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC;
    image_buf_desc.structure_stride = 4u;
    auto* image_buffer = compute_interface->get_buffer_transient(compute_context, &image_buf_desc);

    pnanovdb_compute_resource_t resources[5u] = {};
    resources[0u].buffer_transient = compute_interface->register_buffer_as_transient(compute_context, *nanovdb_buffer);
    resources[1u].buffer_transient = image_buffer;
    resources[2u].texture_transient = compute_interface->register_texture_as_transient(compute_context, background_image);
    resources[3u].buffer_transient = upload_buffer;
    resources[4u].buffer_transient = user_upload_buffer;

    compute->dispatch_shader(
        compute_interface,
        compute_context,
        shader_context,
        resources,
        (image_width + 31u) / 32u, (image_height + 3u) / 4u, 1u,
        "dispatch_shader_on_nanovdb_array");

    if (*readback_buffer)
    {
        // reading final image back to CPU
        pnanovdb_compute_copy_buffer_params_t readback_params = {};
        readback_params.num_bytes = pnanovdb_uint64_t(image_width * image_height * 4u);
        readback_params.src = image_buffer;
        readback_params.dst = *readback_buffer;
        readback_params.debug_label = "dispatch_shader_on_nanovdb_array_readback";
        compute_interface->copy_buffer(compute_context, &readback_params);
    }

    return PNANOVDB_TRUE;
}

pnanovdb_bool_t dispatch_shader_on_array(
    const pnanovdb_compute_t* compute,
    const pnanovdb_compute_device_t* device,
    const char* shader_path,
    pnanovdb_uint32_t grid_dim_x,
    pnanovdb_uint32_t grid_dim_y,
    pnanovdb_uint32_t grid_dim_z,
    pnanovdb_compute_array_t* data_in,
    pnanovdb_compute_array_t* constants,
    pnanovdb_compute_array_t* data_out,
    pnanovdb_uint32_t dispatch_count,
    pnanovdb_uint64_t scratch_size,
    pnanovdb_uint64_t scratch_clear_size)
{
    if (!compute || !device)
    {
        printf("Error: shader dispatch failed, null compute or device\n");
        return PNANOVDB_FALSE;
    }
    if (!data_in->data)
    {
        printf("Error: shader dispatch failed, null data_in\n");
        return PNANOVDB_FALSE;
    }
    if (!constants->data)
    {
        printf("Error: shader dispatch failed, null constants\n");
        return PNANOVDB_FALSE;
    }

    pnanovdb_compiler_settings_t compile_settings = {};
    pnanovdb_compiler_settings_init(&compile_settings);

    pnanovdb_compute_queue_t* queue = compute->device_interface.get_device_queue(device);
    pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
    pnanovdb_compute_context_t* compute_context = compute->device_interface.get_compute_context(queue);

    pnanovdb_shader_context_t* shader_context = compute->create_shader_context(shader_path);
    if (compute->init_shader(compute, queue, shader_context, &compile_settings) == PNANOVDB_FALSE)
    {
        compute->destroy_shader_context(compute, queue, shader_context);
        return PNANOVDB_FALSE;
    }

    pnanovdb_compiler_settings_t clear_compile_settings = {};
    pnanovdb_compiler_settings_init(&clear_compile_settings);

    pnanovdb_shader_context_t* clear_shader_context = compute->create_shader_context("compute/clear_buffer.slang");
    if (compute->init_shader(compute, queue, clear_shader_context, &clear_compile_settings) == PNANOVDB_FALSE)
    {
        compute->destroy_shader_context(compute, queue, clear_shader_context);
        return PNANOVDB_FALSE;
    }

    auto* shader = cast(shader_context);
    auto* clear_shader = cast(clear_shader_context);

    compute->device_interface.enable_profiler(compute_context, (void*)"dispatch_shader_on_array", compute_profiler_report);

    pnanovdb_compute_buffer_desc_t buf_desc = {};

    // data_in upload and device
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 0u;
    buf_desc.size_in_bytes = data_in->element_count * data_in->element_size;
    pnanovdb_compute_buffer_t* data_in_upload = compute_interface->create_buffer(
        compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
    buf_desc.structure_stride = 4u;
    pnanovdb_compute_buffer_t* data_in_device = compute_interface->create_buffer(
        compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    // constants
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 0u;
    buf_desc.size_in_bytes = constants->element_count * constants->element_size;
    pnanovdb_compute_buffer_t* constant_buffer =
        compute_interface->create_buffer(compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_UPLOAD, &buf_desc);

    // data_out device and readback
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED | PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_SRC;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = data_out->element_count == 0u ? 65536u : data_out->element_count * data_out->element_size;
    pnanovdb_compute_buffer_t* data_out_device = compute_interface->create_buffer(
        compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 0u;
    buf_desc.size_in_bytes = data_out->element_count == 0u ? 65536u : data_out->element_count * data_out->element_size;
    pnanovdb_compute_buffer_t* data_out_readback = compute_interface->create_buffer(
        compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK, &buf_desc);

    // scratch buffer
    buf_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_RW_STRUCTURED;
    buf_desc.format = PNANOVDB_COMPUTE_FORMAT_UNKNOWN;
    buf_desc.structure_stride = 4u;
    buf_desc.size_in_bytes = scratch_size == 0u ? 65536u : scratch_size;
    pnanovdb_compute_buffer_t* scratch_device = compute_interface->create_buffer(
        compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_DEVICE, &buf_desc);

    // copy data_in
    void* mapped_data_in = compute_interface->map_buffer(compute_context, data_in_upload);
    memcpy(mapped_data_in, data_in->data, data_in->element_count * data_in->element_size);
    compute_interface->unmap_buffer(compute_context, data_in_upload);

    // copy constants
    void* mapped_constants = compute_interface->map_buffer(compute_context, constant_buffer);
    memcpy(mapped_constants, constants->data, constants->element_count * constants->element_size);
    compute_interface->unmap_buffer(compute_context, constant_buffer);

    // upload data_in
    pnanovdb_compute_copy_buffer_params_t copy_params = {};
    copy_params.num_bytes = data_in->element_count * data_in->element_size;
    copy_params.src = compute_interface->register_buffer_as_transient(compute_context, data_in_upload);
    copy_params.dst = compute_interface->register_buffer_as_transient(compute_context, data_in_device);
    copy_params.debug_label = "dispatch_shader_on_array_upload";
    compute_interface->copy_buffer(compute_context, &copy_params);

    pnanovdb_compute_buffer_transient_t* scratch_transient = compute_interface->register_buffer_as_transient(compute_context, scratch_device);

    // ------------------ clear dispatch params --------------------
    pnanovdb_compute_resource_t clear_resources[1u] = {};
    clear_resources[0u].buffer_transient = scratch_transient;

    // note: clear size in bytes, but compute shader writes words, so 4x workgroup size
    pnanovdb_uint32_t clear_grid_dim_x = (scratch_clear_size + 1023u) / 1024u;

    pnanovdb_compute_dispatch_params_t clear_dispatch_params = {};
    clear_dispatch_params.pipeline = clear_shader->pipeline;
    clear_dispatch_params.grid_dim_x = clear_grid_dim_x;
    clear_dispatch_params.grid_dim_y = 1u;
    clear_dispatch_params.grid_dim_z = 1u;

    clear_dispatch_params.descriptor_writes = clear_shader->shader_build->descriptor_writes;
    clear_dispatch_params.resources = clear_resources;
    clear_dispatch_params.descriptor_write_count = clear_shader->shader_build->descriptor_write_count;

    clear_dispatch_params.debug_label = "dispatch_shader_on_array_clear";

    // ------------------ compute dispatch params --------------------
    pnanovdb_compute_resource_t resources[4u] = {};
    resources[0u].buffer_transient = compute_interface->register_buffer_as_transient(compute_context, data_in_device);
    resources[1u].buffer_transient = compute_interface->register_buffer_as_transient(compute_context, constant_buffer);
    resources[2u].buffer_transient = compute_interface->register_buffer_as_transient(compute_context, data_out_device);
    resources[3u].buffer_transient = scratch_transient;

    pnanovdb_compute_dispatch_params_t dispatch_params = {};
    dispatch_params.pipeline = shader->pipeline;
    dispatch_params.grid_dim_x = grid_dim_x;
    dispatch_params.grid_dim_y = grid_dim_y;
    dispatch_params.grid_dim_z = grid_dim_z;

    dispatch_params.descriptor_writes = shader->shader_build->descriptor_writes;
    dispatch_params.resources = resources;
    dispatch_params.descriptor_write_count = shader->shader_build->descriptor_write_count;

    dispatch_params.debug_label = "dispatch_shader_on_array_dispatch";

    // ------------------ dispatch --------------------
    for (pnanovdb_uint32_t dispatch_idx = 0u; dispatch_idx < dispatch_count; dispatch_idx++)
    {
        if (scratch_clear_size != 0llu)
        {
            compute_interface->dispatch(compute_context, &clear_dispatch_params);
        }
        compute_interface->dispatch(compute_context, &dispatch_params);
    }

    // readback data_out
    copy_params.num_bytes = data_out->element_count * data_out->element_size;
    copy_params.src = compute_interface->register_buffer_as_transient(compute_context, data_out_device);
    copy_params.dst = compute_interface->register_buffer_as_transient(compute_context, data_out_readback);
    copy_params.debug_label = "dispatch_shader_on_array_readback";
    compute_interface->copy_buffer(compute_context, &copy_params);

    pnanovdb_uint64_t flushed_frame = 0llu;
    compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

    compute->device_interface.wait_idle(queue);

    // to flush profile
    compute->device_interface.flush(queue, &flushed_frame, nullptr, nullptr);

    // copy data_out
    void* mapped_data_out = compute_interface->map_buffer(compute_context, data_out_readback);
        memcpy(data_out->data, mapped_data_out, data_out->element_count * data_out->element_size);
    compute_interface->unmap_buffer(compute_context, data_out_readback);

    compute_interface->destroy_buffer(compute_context, data_in_upload);
    compute_interface->destroy_buffer(compute_context, data_in_device);
    compute_interface->destroy_buffer(compute_context, constant_buffer);
    compute_interface->destroy_buffer(compute_context, data_out_device);
    compute_interface->destroy_buffer(compute_context, data_out_readback);
    compute_interface->destroy_buffer(compute_context, scratch_device);

    compute->device_interface.disable_profiler(compute_context);

    compute->destroy_shader(compute_interface, &compute->shader_interface, compute_context, shader_context);
    compute->destroy_shader(compute_interface, &compute->shader_interface, compute_context, clear_shader_context);

    compute->destroy_shader_context(compute, queue, shader_context);
    compute->destroy_shader_context(compute, queue, clear_shader_context);

    return PNANOVDB_TRUE;
}

pnanovdb_compute_array_t* create_array(size_t element_size, pnanovdb_uint64_t element_count, void* data)
{
    pnanovdb_compute_array_t* array = new pnanovdb_compute_array_t();
    array->element_count = element_count;
    array->element_size = element_size;
    array->data = new char[array->element_size * array->element_count];
    if (data)
    {
        memcpy(array->data, data, array->element_size * array->element_count);
    }
    else
    {
        memset(array->data, 0, array->element_size * array->element_count);
    }
    return array;
}

void destroy_array(pnanovdb_compute_array_t* array)
{
    if (array && array->data)
    {
        delete[](char*)array->data;
        array->data = nullptr;
        delete array;
        array = nullptr;
    }
}

void* map_array(pnanovdb_compute_array_t* array)
{
    if (!array)
    {
        printf("Error: map_array failed, null array\n");
        return nullptr;
    }
    return array->data;
}

void unmap_array(pnanovdb_compute_array_t* array)
{
    if (!array)
    {
        printf("Error: unmap_array failed, null array\n");
        return;
    }
}


void compute_array_print_range(
    const pnanovdb_compute_t* compute,
    pnanovdb_compute_log_print_t log_print,
    const char* name,
    pnanovdb_compute_array_t* arr,
    pnanovdb_uint32_t channel_count)
{
    if (channel_count > 4u)
    {
        channel_count = 4u;
    }
    int print_count = 0;
    float val_min[4] = { INFINITY, INFINITY, INFINITY, INFINITY };
    float val_max[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float val_min_anisotropy = INFINITY;
    float val_max_anisotropy = -INFINITY;
    double val_ave_anisotropy = 0.0;
    double val_ave[4] = { 0.0, 0.0, 0.0, 0.0 };
    float* mapped_val = (float*)compute->map_array(arr);
    int aniso_type[4] = {};
    for (pnanovdb_uint64_t idx = 0u; idx < arr->element_count; idx++)
    {
        pnanovdb_uint64_t channel_idx = idx % channel_count;
        val_min[channel_idx] = fminf(val_min[channel_idx], mapped_val[idx]);
        val_max[channel_idx] = fmaxf(val_max[channel_idx], mapped_val[idx]);
        val_ave[channel_idx] = val_ave[channel_idx] + (double)mapped_val[idx];
        if (channel_idx == 0u)
        {
            float abs_val_min = INFINITY;
            float abs_val_max = -INFINITY;
            for (pnanovdb_uint32_t channel_idx_off = 0u; channel_idx_off < channel_count; channel_idx_off++)
            {
                float abs_val = fabsf(mapped_val[idx + channel_idx_off]);
                abs_val_min = fminf(abs_val_min, abs_val);
                abs_val_max = fmaxf(abs_val_max, abs_val);
            }
            float val_anisotropy = abs_val_max / abs_val_min;
            val_min_anisotropy = fminf(val_min_anisotropy, val_anisotropy);
            val_max_anisotropy = fmaxf(val_max_anisotropy, val_anisotropy);
            val_ave_anisotropy = val_ave_anisotropy + (double)val_anisotropy;
            if (print_count < 4u && channel_count == 3u && val_anisotropy > 30000000.f)
            {
                print_count++;
                if (log_print)
                {
                    log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "array(%s) val(%e,%e,%e)", name, mapped_val[idx], mapped_val[idx + 1], mapped_val[idx + 2]);
                }
            }
            if (channel_count == 3u)
            {
                float sx = fabsf(mapped_val[idx]);
                float sy = fabsf(mapped_val[idx + 1]);
                float sz = fabsf(mapped_val[idx + 2]);
                float smax = fmaxf(fmaxf(sx, sy), sz);
                int s_type = 0;
                s_type += (sx >= 0.5f * smax) ? 1 : 0;
                s_type += (sy >= 0.5f * smax) ? 1 : 0;
                s_type += (sz >= 0.5f * smax) ? 1 : 0;
                aniso_type[s_type]++;
            }
        }
    }
    compute->unmap_array(arr);
    for (pnanovdb_uint32_t channel_idx = 0u; channel_idx < channel_count; channel_idx++)
    {
        val_ave[channel_idx] /= (double)(arr->element_count / channel_count);
        if (log_print)
        {
            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "array(%s) channel(%d) min(%f) max(%f) ave(%f)",
                name, channel_idx, val_min[channel_idx], val_max[channel_idx], val_ave[channel_idx]);
        }
    }
    val_ave_anisotropy /= (double)(arr->element_count / channel_count);
    if (log_print)
    {
        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "array(%s) min_anisotropy(%f) max_anisotropy(%f) ave_anisotropy(%f)",
            name, val_min_anisotropy, val_max_anisotropy, val_ave_anisotropy);
        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "array(%s) aniso_type_count(%d,%d,%d,%d)",
            name, aniso_type[0], aniso_type[1], aniso_type[2], aniso_type[3]);
    }
}

PNANOVDB_API pnanovdb_compute_t* pnanovdb_get_compute()
{
    static pnanovdb_compute_t compute = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_compute_t) };

    compute.load_nanovdb = load_nanovdb;
    compute.save_nanovdb = save_nanovdb;
    compute.create_shader_context = create_shader_context;
    compute.destroy_shader_context = destroy_shader_context;
    compute.init_shader = init_shader;
    compute.destroy_shader = destroy_shader;
    compute.dispatch_shader = dispatch_shader;
    compute.dispatch_shader_on_array = dispatch_shader_on_array;
    compute.dispatch_shader_on_nanovdb_array = dispatch_shader_on_nanovdb_array;
    compute.create_array = create_array;
    compute.destroy_array = destroy_array;
    compute.map_array = map_array;
    compute.unmap_array = unmap_array;
    compute.compute_array_print_range = compute_array_print_range;

    return &compute;
}
}
