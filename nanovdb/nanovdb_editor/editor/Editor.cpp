// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/Editor.cpp

    \author Petra Hapalova

    \brief
*/

#include "Editor.h"

#include "ShaderMonitor.h"
#include "Console.h"
#include "Profiler.h"

#ifdef USE_IMGUI_INSTANCE
#include "ImguiInstance.h"
#endif

#include "imgui/ImguiWindow.h"
#include "imgui/UploadBuffer.h"

#include "compute/ComputeShader.h"

#include "nanovdb_editor/putil/Raster.h"
#include "nanovdb_editor/putil/Raster.hpp"
#include "nanovdb_editor/putil/WorkerThread.hpp"

#include <nanovdb/io/IO.h>

#define TEST_IO 0

static const pnanovdb_uint32_t s_default_width = 1440u;
static const pnanovdb_uint32_t s_default_height = 720u;

static const char* s_raster_file = "./data/splats.npz";

namespace pnanovdb_editor
{
enum class ViewportShader : int
{
    Editor
};

static const char* s_viewport_shaders[] = {
    "editor/editor.slang"
};

// default shader used for the NanoVDB viewer
static const char* s_default_shader = s_viewport_shaders[(int)ViewportShader::Editor];
static const char* s_raster_shader = s_viewport_shaders[(int)ViewportShader::Editor];

// mirrored from shader
struct EditorParams
{
    pnanovdb_camera_mat_t view_inv;
    pnanovdb_camera_mat_t projection_inv;
    uint32_t width;
    uint32_t height;
    uint32_t narrow_band_only;
    float slice_plane_thickness;
    float slice_plane[4u];
    uint32_t highlight_bbox;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
};

struct UserParams
{
#if NANOVDB_EDITOR_USER_PARAMS_SIZE
    uint64_t pad[NANOVDB_EDITOR_USER_PARAMS_SIZE];
#else
    uint64_t pad[16u];
#endif
};

static void save_image(const char* filename, float* mapped_data, uint32_t image_width, uint32_t image_height)
{
    FILE* file = fopen(filename, "wb");
    if (!file)
    {
        printf("Could not create file to save the capture '%s'", filename);
        return;
    }

    char headerField0 = 'B';
    char headerField1 = 'M';
    uint32_t size = 54 + image_width * image_height * 4u;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t offset = 54;
    uint32_t headerSize = 40;
    uint32_t width = image_width;
    uint32_t height = image_height;
    uint16_t colorPlanes = 1;
    uint16_t bitsPerPixel = 32;
    uint32_t compressionMethod = 0;
    uint32_t imageSize = image_width * image_height * 4u;
    uint32_t hRes = 2000;
    uint32_t vRes = 2000;
    uint32_t numColors = 0;
    uint32_t numImportantColors = 0;

    fwrite(&headerField0, 1, 1, file);
    fwrite(&headerField1, 1, 1, file);
    fwrite(&size, 4, 1, file);
    fwrite(&reserved1, 2, 1, file);
    fwrite(&reserved2, 2, 1, file);
    fwrite(&offset, 4, 1, file);
    fwrite(&headerSize, 4, 1, file);
    fwrite(&width, 4, 1, file);
    fwrite(&height, 4, 1, file);
    fwrite(&colorPlanes, 2, 1, file);
    fwrite(&bitsPerPixel, 2, 1, file);
    fwrite(&compressionMethod, 4, 1, file);
    fwrite(&imageSize, 4, 1, file);
    fwrite(&hRes, 4, 1, file);
    fwrite(&vRes, 4, 1, file);
    fwrite(&numColors, 4, 1, file);
    fwrite(&numImportantColors, 4, 1, file);
    fwrite(mapped_data, 1u, image_width * image_height * 4u, file);

    fclose(file);
}

void init(pnanovdb_editor_t* editor)
{
}

void shutdown(pnanovdb_editor_t* editor)
{
    if (editor->data_array)
    {
        editor->compute->destroy_array(editor->data_array);
        editor->data_array = nullptr;
    }
    if (editor->nanovdb_array)
    {
        editor->compute->destroy_array(editor->nanovdb_array);
        editor->nanovdb_array = nullptr;
    }
}

void add_nanovdb(pnanovdb_editor_t* editor, pnanovdb_compute_array_t* nanovdb_array)
{
    if (editor->nanovdb_array)
    {
        editor->compute->destroy_array(editor->nanovdb_array);
    }
    editor->nanovdb_array = nanovdb_array;
}

void add_array(pnanovdb_editor_t* editor, pnanovdb_compute_array_t* data_array)
{
    if (editor->data_array)
    {
        editor->compute->destroy_array(editor->data_array);
    }
    editor->data_array = data_array;
}

void add_callable(pnanovdb_editor_t* editor, const char* name, pnanovdb_editor_callable_t callable)
{
    editor->callable_func = callable;
    strcpy(editor->callable_name, name);
}

void show(pnanovdb_editor_t* editor, pnanovdb_compute_device_t* device)
{
    if (!editor->compute || !editor->compiler)
    {
        return;
    }

    pnanovdb_int32_t image_width = s_default_width;
    pnanovdb_int32_t image_height = s_default_height;

    pnanovdb_imgui_window_interface_t* imgui_window_iface = pnanovdb_imgui_get_window_interface();
    if (!imgui_window_iface)
    {
        return;
    }
    pnanovdb_imgui_settings_render_t* imgui_user_settings = nullptr;

#ifdef USE_IMGUI_INSTANCE
    pnanovdb_imgui_instance_interface_t* imgui_instance_iface = get_user_imgui_instance_interface();
    imgui_instance_user::Instance* imgui_user_instance = nullptr;
    void* imgui_instance_userdata = &imgui_user_instance;
#endif
    pnanovdb_imgui_window_t* imgui_window = imgui_window_iface->create(
        editor->compute,
        device,
        image_width,
        image_height,
        (void**)&imgui_user_settings,
#ifdef USE_IMGUI_INSTANCE
        PNANOVDB_FALSE,
        &imgui_instance_iface,
        &imgui_instance_userdata,
        1u
#else
        PNANOVDB_TRUE,
        nullptr,
        nullptr,
        0u
#endif
    );

    if (!imgui_window)
    {
        return;
    }

#ifdef USE_IMGUI_INSTANCE
    if (!imgui_user_instance)
    {
        return;
    }
#endif

    pnanovdb_compiler_instance_t* compiler_inst = editor->compiler->create_instance();
    pnanovdb_compiler_settings_t compile_settings = {};
    pnanovdb_compiler_settings_init(&compile_settings);

    pnanovdb_compute_queue_t* device_queue = editor->compute->device_interface.get_device_queue(device);
    pnanovdb_compute_queue_t* compute_queue = editor->compute->device_interface.get_compute_queue(device);      // used for a worker thread
    pnanovdb_compute_interface_t* compute_interface = editor->compute->device_interface.get_compute_interface(device_queue);
    pnanovdb_compute_context_t* compute_context = editor->compute->device_interface.get_compute_context(device_queue);

    pnanovdb_camera_mat_t view = {};
    pnanovdb_camera_mat_t projection = {};

    pnanovdb_compute_upload_buffer_t compute_upload_buffer;
    pnanovdb_compute_upload_buffer_init(compute_interface, compute_context, &compute_upload_buffer,
        PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 0u);

    pnanovdb_compute_upload_buffer_t user_upload_buffer;
    pnanovdb_compute_upload_buffer_init(compute_interface, compute_context, &user_upload_buffer,
        PNANOVDB_COMPUTE_BUFFER_USAGE_CONSTANT, PNANOVDB_COMPUTE_FORMAT_UNKNOWN, 0u);

    pnanovdb_compute_texture_t* background_image = nullptr;
    pnanovdb_compute_buffer_t* readback_buffer = nullptr;

    std::string capture_filename = "./data/pnanovdbeditor_capture.bmp";

    pnanovdb_shader_context_t* shader_context = nullptr;
    pnanovdb_compute_buffer_t* nanovdb_buffer = nullptr;
    pnanovdb_compute_array_t* uploaded_nanovdb_array = nullptr;

    pnanovdb_raster_t raster = {};
    pnanovdb_raster_load(&raster, editor->compute);
    pnanovdb_raster_context_t* raster_ctx = nullptr;
    pnanovdb_raster_gaussian_data_t* raster_data = nullptr;

    pnanovdb_util::WorkerThread raster_worker;
    pnanovdb_util::WorkerThread::TaskId raster_task_id = pnanovdb_util::WorkerThread::invalidTaskId();
    std::string pending_raster_filepath;
    float pending_voxel_size = 1.f / 128.f;
    pnanovdb_raster_gaussian_data_t* pending_gaussian_data = nullptr;
    pnanovdb_raster_context_t* pending_raster_ctx = nullptr;
    pnanovdb_compute_array_t* pending_nanovdb_array = nullptr;

    editor->compute->device_interface.enable_profiler(compute_context, (void*)"editor", pnanovdb_editor::Profiler::report_callback);
    editor->compute->device_interface.get_memory_stats(device, Profiler::getInstance().getMemoryStats());

#ifdef USE_IMGUI_INSTANCE
    // called from a worker thread, recompile shader cache
    ShaderCallback callback = [&](const std::string& path)
    {
        pnanovdb_compiler_settings_t settings = imgui_user_instance->compiler_settings;
        std::string shader_name = pnanovdb_shader::getShaderName(path.c_str());
        Console::getInstance().addLog("Compiling shader: %s...", shader_name.c_str());

        // compile to HLSL first
        if (settings.hlsl_output)
        {
            pnanovdb_bool_t result = editor->compiler->compile_shader_from_file(compiler_inst, path.c_str(), &settings, nullptr);
            if (!result)
            {
                Console::getInstance().addLog("Failed to compile shader to HLSL: %s", shader_name.c_str());
            }
            imgui_user_instance->pending.update_hlsl = true;
            settings.hlsl_output = false;
        }
        // then compile the shader
        pnanovdb_bool_t shader_updated = false;
        bool result = editor->compiler->compile_shader_from_file(compiler_inst, path.c_str(), &settings, &shader_updated);
        if (result)
        {
            Console::getInstance().addLog("Compilation successful: %s", shader_name.c_str());
        }
        else
        {
            shader_updated = PNANOVDB_TRUE;
            Console::getInstance().addLog("Failed to compile shader: %s", shader_name.c_str());
        }

        if (shader_updated == PNANOVDB_TRUE && pnanovdb_shader::getShaderFilePath(imgui_user_instance->shader_name.c_str()) == path)
        {
            // update current shader on a main thread
            imgui_user_instance->pending.update_shader = true;
        }
    };
    monitor_shader_dir(pnanovdb_shader::getShaderDir().c_str(), callback);

    imgui_user_instance->shader_name = s_default_shader;
    imgui_user_instance->pending.shader_name = imgui_user_instance->shader_name;

    if (editor->nanovdb_array && editor->nanovdb_array->filepath)
    {
        imgui_user_instance->nanovdb_filepath = editor->nanovdb_array->filepath;
    }

    imgui_user_instance->raster_filepath = std::string(s_raster_file);

    for (const char* shader : s_viewport_shaders)
    {
        imgui_user_instance->viewport_shaders.push_back(shader);
    }

    imgui_user_instance->compiler = editor->compiler;
    imgui_user_instance->compute = editor->compute;

    bool dispatch_shader = true;

    editor->compiler->set_diagnostic_callback(compiler_inst,
        [](const char* message)
        {
            if (message && message[0] != '\0')
            {
                pnanovdb_editor::Console::getInstance().addLog("%s", message);
            }
        });
#endif

    auto cleanup_background = [&]()
        {
            if (background_image)
            {
                compute_interface->destroy_texture(compute_context, background_image);
            }
            background_image = nullptr;
        };

    bool should_run = true;
    while (should_run)
    {
        // create background image texture
        pnanovdb_compute_texture_desc_t tex_desc = {};
        tex_desc.texture_type = PNANOVDB_COMPUTE_TEXTURE_TYPE_2D;
        tex_desc.usage = PNANOVDB_COMPUTE_TEXTURE_USAGE_TEXTURE | PNANOVDB_COMPUTE_TEXTURE_USAGE_RW_TEXTURE;
        tex_desc.format = PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM;
        tex_desc.width = image_width;
        tex_desc.height = image_height;
        tex_desc.depth = 1u;
        tex_desc.mip_levels = 1u;
        background_image = compute_interface->create_texture(compute_context, &tex_desc);

        // get camera view and projection matrices
        imgui_window_iface->get_camera(imgui_window, &image_width, &image_height, &view, &projection);
        pnanovdb_camera_mat_t view_inv = pnanovdb_camera_mat_inverse(view);
        pnanovdb_camera_mat_t projection_inv = pnanovdb_camera_mat_inverse(projection);

        bool should_capture = false;
#ifdef USE_IMGUI_INSTANCE
        // update pending GUI states
        if (imgui_user_instance->pending.capture_image)
        {
            imgui_user_instance->pending.capture_image = false;
            should_capture = true;
        }
        if (imgui_user_instance->pending.load_nvdb)
        {
            imgui_user_instance->pending.load_nvdb = false;
            const char* nvdb_filepath = imgui_user_instance->nanovdb_filepath.c_str();
            pnanovdb_compute_array_t* loaded_array = editor->compute->load_nanovdb(nvdb_filepath);
            if (loaded_array)
            {
                editor->add_nanovdb(editor, loaded_array);
            }
            auto nvdb_shader = s_default_shader;
            if (imgui_user_instance->shader_name != nvdb_shader)
            {
                imgui_user_instance->shader_name = nvdb_shader;
                imgui_user_instance->pending.shader_name = nvdb_shader;
                imgui_user_instance->pending.update_shader = true;
            }
        }
        if (imgui_user_instance->pending.save_nanovdb)
        {
            imgui_user_instance->pending.save_nanovdb = false;
            const char* nvdb_filepath = imgui_user_instance->nanovdb_filepath.c_str();
            if (editor->nanovdb_array)
            {
                pnanovdb_bool_t result = editor->compute->save_nanovdb(editor->nanovdb_array, nvdb_filepath);
                if (result == PNANOVDB_TRUE)
                {
                    pnanovdb_editor::Console::getInstance().addLog("NanoVDB saved to '%s'", nvdb_filepath);
                }
                else
                {
                    pnanovdb_editor::Console::getInstance().addLog("Failed to save NanoVDB to '%s'", nvdb_filepath);
                }
            }
        }
        if (imgui_user_instance->pending.print_slice)
        {
            imgui_user_instance->pending.print_slice = false;
#if TEST_IO
            FILE* input_file = fopen("./data/smoke.nvdb", "rb");
            FILE* output_file = fopen("./data/slice_output.bmp", "wb");
            test_pnanovdb_io_print_slice(input_file, output_file);
            fclose(input_file);
            fclose(output_file);
#endif
        }

        // update raster
        if (imgui_user_instance->pending.update_raster)
        {
            imgui_user_instance->pending.update_raster = false;

            if (raster_worker.hasRunningTask())
            {
                pnanovdb_editor::Console::getInstance().addLog("Error: Rasterization already in progress", imgui_user_instance->raster_filepath.c_str());
            }
            else
            {
                pending_raster_filepath = imgui_user_instance->raster_filepath;
                pending_voxel_size = 1.f / imgui_user_instance->raster_voxels_per_unit;

                raster_task_id = raster_worker.enqueue(
                    [&raster_worker](pnanovdb_raster_t* raster,
                       const pnanovdb_compute_t* compute,
                       pnanovdb_compute_queue_t* queue,
                       const char* filepath,
                       float voxel_size,
                       pnanovdb_compute_array_t** nanovdb_array,
                       pnanovdb_raster_gaussian_data_t** gaussian_data,
                       pnanovdb_raster_context_t** raster_context,
                       pnanovdb_profiler_report_t profiler) -> bool
                    {
                        return pnanovdb_raster::raster_file(raster, compute, queue, filepath, voxel_size, nanovdb_array, gaussian_data, raster_context, profiler, (void*)(&raster_worker));
                    },
                    &raster,
                    raster.compute,
                    compute_queue,
                    pending_raster_filepath.c_str(),
                    pending_voxel_size,
                    imgui_user_instance->viewport_option == imgui_instance_user::ViewportOption::NanoVDB ? &pending_nanovdb_array : nullptr,
                    imgui_user_instance->viewport_option == imgui_instance_user::ViewportOption::Raster2D ? &pending_gaussian_data : nullptr,
                    imgui_user_instance->viewport_option == imgui_instance_user::ViewportOption::Raster2D ? &pending_raster_ctx : nullptr,
                    pnanovdb_editor::Profiler::report_callback
                );

                pnanovdb_editor::Console::getInstance().addLog("Running rasterization: '%s'...", pending_raster_filepath.c_str());
            }
        }
        {
            if (raster_worker.isTaskRunning(raster_task_id))
            {
                imgui_user_instance->progress.text = raster_worker.getTaskProgressText(raster_task_id);
                imgui_user_instance->progress.value = raster_worker.getTaskProgress(raster_task_id);
            }
            else if (raster_worker.isTaskCompleted(raster_task_id))
            {
                // Update with new data and reset
                if (pending_gaussian_data)
                {
                    if (editor->gaussian_data)
                    {
                        raster.destroy_gaussian_data(raster.compute, compute_queue, raster_ctx, editor->gaussian_data);
                    }
                    editor->gaussian_data = pending_gaussian_data;
                    pending_gaussian_data = nullptr;
                }
                if (pending_raster_ctx)
                {
                    if (raster_ctx)
                    {
                        raster.destroy_context(raster.compute, compute_queue, raster_ctx);
                    }
                    raster_ctx = pending_raster_ctx;
                    pending_raster_ctx = nullptr;
                }
                if (pending_nanovdb_array)
                {
                    if (editor->nanovdb_array)
                    {
                        editor->compute->destroy_array(editor->nanovdb_array);
                    }
                    editor->nanovdb_array = pending_nanovdb_array;
                    pending_nanovdb_array = nullptr;
                }

                if (raster_worker.isTaskSuccessful(raster_task_id))
                {
                    // Update viewport shader if needed
                    if (imgui_user_instance->viewport_option == imgui_instance_user::ViewportOption::NanoVDB)
                    {
                        if (imgui_user_instance->shader_name != s_raster_shader)
                        {
                            imgui_user_instance->shader_name = s_raster_shader;
                            imgui_user_instance->pending.shader_name = s_raster_shader;
                            imgui_user_instance->pending.update_shader = true;
                        }
                    }
                    pnanovdb_editor::Console::getInstance().addLog("Rasterization of '%s' was successful", pending_raster_filepath.c_str());
                }
                else
                {
                    pnanovdb_editor::Console::getInstance().addLog("Rasterization of '%s' failed", pending_raster_filepath.c_str());
                }

                pending_raster_filepath = "";
                raster_worker.removeCompletedTask(raster_task_id);

                imgui_user_instance->progress.reset();
            }
        }

        // update memory stats periodically
        if (imgui_user_instance && imgui_user_instance->pending.update_memory_stats)
        {
            editor->compute->device_interface.get_memory_stats(device, Profiler::getInstance().getMemoryStats());
            imgui_user_instance->pending.update_memory_stats = false;
        }

        // update viewport according to the selected option
        if (imgui_user_instance->viewport_option == imgui_instance_user::ViewportOption::NanoVDB)
        {
            if (imgui_user_instance->pending.update_shader)
            {
                imgui_user_instance->pending.update_shader = false;
                editor->compute->destroy_shader_context(editor->compute, device_queue, shader_context);
                shader_context = editor->compute->create_shader_context(imgui_user_instance->shader_name.c_str());
                if (editor->compute->init_shader(
                    editor->compute,
                    device_queue,
                    shader_context,
                    &compile_settings) == PNANOVDB_FALSE)
                {
                    // compilation has failed, don't dispatch the shader
                    dispatch_shader = false;
                    cleanup_background();
                }
                else
                {
                    dispatch_shader = true;
                }
            }
            if (dispatch_shader && editor->nanovdb_array)
            {
                EditorParams editor_params = {};
                editor_params.view_inv = pnanovdb_camera_mat_transpose(view_inv);
                editor_params.projection_inv = pnanovdb_camera_mat_transpose(projection_inv);
                editor_params.width = image_width;
                editor_params.height = image_height;
                editor_params.narrow_band_only = imgui_user_instance->narrow_band_only ? 1u : 0u;
                editor_params.slice_plane_thickness = imgui_user_instance->slice_plane_thickness;
                editor_params.slice_plane[0] = imgui_user_instance->slice_plane[0];
                editor_params.slice_plane[1] = imgui_user_instance->slice_plane[1];
                editor_params.slice_plane[2] = imgui_user_instance->slice_plane[2];
                editor_params.slice_plane[3] = imgui_user_instance->slice_plane[3];
                editor_params.highlight_bbox = imgui_user_instance->highlight_bbox ? 1u : 0u;

                UserParams user_params = {};
                imgui_user_instance->user_params.load(imgui_user_instance->shader_name);
                char* user_params_ptr = reinterpret_cast<char*>(&user_params);
                for (auto& user_param : imgui_user_instance->user_params.getParams())
                {
                    if (user_param.value != nullptr)
                    {
                        std::memcpy(user_params_ptr, user_param.value, user_param.num_elements * user_param.size);
                        user_params_ptr += user_param.num_elements * user_param.size;
                    }
                }

                EditorParams* mapped = (EditorParams*)pnanovdb_compute_upload_buffer_map(compute_context, &compute_upload_buffer, 16u);
                *mapped = editor_params;
                auto* upload_transient = pnanovdb_compute_upload_buffer_unmap(compute_context, &compute_upload_buffer);

                UserParams* user_mapped = (UserParams*)pnanovdb_compute_upload_buffer_map(compute_context, &user_upload_buffer, 16u);
                *user_mapped = user_params;
                auto* user_upload_transient = pnanovdb_compute_upload_buffer_unmap(compute_context, &user_upload_buffer);

                pnanovdb_compute_buffer_transient_t* readback_transient = nullptr;
                if (should_capture)
                {
                    pnanovdb_compute_buffer_desc_t readback_desc = {};
                    readback_desc.size_in_bytes = pnanovdb_uint64_t(image_width * image_height * 4u);
                    readback_desc.usage = PNANOVDB_COMPUTE_BUFFER_USAGE_COPY_DST;
                    readback_buffer = compute_interface->create_buffer(compute_context, PNANOVDB_COMPUTE_MEMORY_TYPE_READBACK, &readback_desc);
                    readback_transient = compute_interface->register_buffer_as_transient(compute_context, readback_buffer);
                }

                if (editor->nanovdb_array != uploaded_nanovdb_array && nanovdb_buffer)
                {
                    compute_interface->destroy_buffer(compute_context, nanovdb_buffer);
                    nanovdb_buffer = nullptr;
                }

                editor->compute->dispatch_shader_on_nanovdb_array(
                    editor->compute,
                    device,
                    shader_context,
                    editor->nanovdb_array,
                    image_width,
                    image_height,
                    background_image,
                    upload_transient,
                    user_upload_transient,
                    &nanovdb_buffer,
                    &readback_transient
                );

                if (nanovdb_buffer)
                {
                    uploaded_nanovdb_array = editor->nanovdb_array;
                }
            }
            else
            {
                cleanup_background();
            }
        }
        else if (imgui_user_instance->viewport_option == imgui_instance_user::ViewportOption::Raster2D)
        {
            if (editor->gaussian_data)
            {
                raster.raster_gaussian_2d(
                    raster.compute,
                    device_queue,
                    raster_ctx,
                    editor->gaussian_data,
                    background_image,
                    image_width,
                    image_height,
                    &view,
                    &projection
                );
            }
            else
            {
                cleanup_background();
            }
        }

        // update camera settings
        if (imgui_user_instance->pending.save_camera)
        {
            imgui_user_instance->pending.save_camera = false;

            imgui_user_instance->saved_render_settings[imgui_user_instance->render_settings_name] = *imgui_user_settings;

            pnanovdb_camera_state_t camera_state = {};
            imgui_window_iface->get_camera_state(imgui_window, &camera_state);
            imgui_user_instance->saved_render_settings[imgui_user_instance->render_settings_name].camera_state = camera_state;

            imgui_user_instance->pending.save_render_settings = true;
        }
        if (imgui_user_instance->pending.load_camera)
        {
            imgui_user_instance->pending.load_camera = false;

            *imgui_user_settings = imgui_user_instance->saved_render_settings[imgui_user_instance->render_settings_name];
            imgui_user_settings->sync_camera_state = true;

            imgui_user_instance->pending.load_render_settings = true;
        }
#else
        // default to NanoVDB viewport if there is no imgui instance
        if (editor->nanovdb_array != uploaded_nanovdb_array)
        {
            shader_context = editor->compute->create_shader_context(s_default_shader);
            editor->compute->init_shader(
                editor->compute,
                device_queue,
                shader_context,
                &compile_settings
            );

            UserParams user_params = {};
            EditorParams editor_params = {};
            editor_params.view_inv = pnanovdb_camera_mat_transpose(view_inv);
            editor_params.projection_inv = pnanovdb_camera_mat_transpose(projection_inv);
            editor_params.width = image_width;
            editor_params.height = image_height;

            editor->compute->dispatch_shader_on_nanovdb_array(
                compute_interface,
                compute_context,
                shader_context,
                editor->nanovdb_array,
                image_width,
                image_height,
                background_image,
                upload_transient,
                user_upload_transient,
                nanovdb_buffer,
                nullptr,
                );
            uploaded_nanovdb_array = editor->nanovdb_array;
        }
#endif
        imgui_window_iface->update_camera(imgui_window, imgui_user_settings);

        // update viewport image
        should_run = imgui_window_iface->update(
            editor->compute,
            device_queue,
            background_image ? compute_interface->register_texture_as_transient(compute_context, background_image) : nullptr,
            &image_width,
            &image_height,
            imgui_window,
            imgui_user_settings
        );

        if (background_image)
        {
            compute_interface->destroy_texture(compute_context, background_image);
        }

        if (should_capture && readback_buffer)
        {
            editor->compute->device_interface.wait_idle(device_queue);

            float* mapped_data = (float*)compute_interface->map_buffer(compute_context, readback_buffer);
            save_image(capture_filename.c_str(), mapped_data, image_width, image_height);
            compute_interface->unmap_buffer(compute_context, readback_buffer);
        }
    }
    editor->compute->device_interface.wait_idle(device_queue);

    if (editor->gaussian_data)
    {
        raster.destroy_gaussian_data(raster.compute, compute_queue, raster_ctx, editor->gaussian_data);
        editor->gaussian_data = nullptr;
    }
    if (raster_ctx)
    {
        raster.destroy_context(raster.compute, compute_queue, raster_ctx);
        raster_ctx = nullptr;
    }
    pnanovdb_raster_free(&raster);

    editor->compute->device_interface.disable_profiler(compute_context);

    editor->compute->destroy_shader(compute_interface, &editor->compute->shader_interface, compute_context, shader_context);
    editor->compiler->destroy_instance(compiler_inst);

    pnanovdb_compute_upload_buffer_destroy(compute_context, &compute_upload_buffer);
    pnanovdb_compute_upload_buffer_destroy(compute_context, &user_upload_buffer);

    imgui_window_iface->destroy(editor->compute, device_queue, imgui_window, imgui_user_settings);
}

PNANOVDB_API pnanovdb_editor_t* pnanovdb_get_editor()
{
    static pnanovdb_editor_t editor = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_editor_t) };

    editor.init = init;
    editor.shutdown = shutdown;
    editor.add_nanovdb = add_nanovdb;
    editor.add_array = add_array;
    editor.add_callable = add_callable;
    editor.show = show;

    return &editor;
}
}
