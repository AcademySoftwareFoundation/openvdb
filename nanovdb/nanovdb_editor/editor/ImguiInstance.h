// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/ImguiInstance.h

    \author Andrew Reidmeyer, Petra Hapalova

    \brief
*/

#pragma once

#include "UserParams.h"

#include "imgui/ImguiWindow.h"
#include "putil/Editor.h"
#include "putil/Shader.hpp"

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif  // IMGUI_DEFINE_MATH_OPERATORS

#include <imgui.h>

#include <string>
#include <atomic>
#include <map>

namespace imgui_instance_user
{
    static const char* s_render_settings_default = "default";

    enum class ViewportOption : int
    {
        NanoVDB,
        Raster2D,
        Last,
    };

    struct ViewportSettings
    {
        std::string render_settings_name = s_render_settings_default;
    };

    struct PendingState
    {
        std::atomic<bool> update_shader = true;     // needs to be initialized with true to map the shader after init
        std::atomic<bool> update_hlsl = false;
        bool capture_image = false;
        bool print_slice = false;
        bool load_nvdb = false;
        bool save_nanovdb = false;
        bool find_raster_file = false;
        bool find_callable_file = false;
        bool open_file = false;
        bool save_file = false;
        bool load_camera = false;                   // load camera state in editor update loop
        bool save_camera = true;                    // save default camera first
        bool save_render_settings = false;
        bool load_render_settings = false;
        std::string shader_name = "";
        bool update_memory_stats = false;
        bool update_raster = false;
    };

    struct ProgressBar
    {
        std::string text;
        float value;

        ProgressBar()
        {
            reset();
        }

        void reset()
        {
            text = "";
            value = 0.f;
        }
    };

    struct WindowState
    {
        bool show_profiler = false;
        bool show_code_editor = false;
        bool show_console = true;
        bool show_viewport_settings = true;
        bool show_render_settings = true;
        bool show_user_settings = true;
        bool show_compiler_settings = true;
        bool show_user_params = true;
        bool show_benchmark = true;
    };

    struct UniformState
    {
    };

    struct Instance
    {
        PendingState pending;
        WindowState window;

        const pnanovdb_compiler_t* compiler;
        const pnanovdb_compute_t* compute;

        pnanovdb_imgui_settings_render_t* render_settings;
        pnanovdb_compiler_settings_t compiler_settings;

        bool highlight_bbox = false;
        bool narrow_band_only = true;
        float slice_plane_thickness = 0.f;
        float slice_plane[4] = {1.f, 0.f, 0.f, 0.f};
        bool animate_slice = false;

        pnanovdb_uint64_t last_timestamp = 0llu;

        ViewportOption viewport_option = ViewportOption::NanoVDB;
        ViewportSettings viewport_settings[(int)ViewportOption::Last];

        std::string shader_name = "";           // shader used for the viewport
        std::string nanovdb_filepath = "";      // filename selected in the ImGuiFileDialog
        std::string raster_filepath = "";
        float raster_voxels_per_unit = 128.f;

        UserParams user_params;

        ImVec2 dialog_size{768.f, 512.f};

        std::string callable_file = "";
        std::string callable_name = "";
        pnanovdb_editor_callable_t callable_func = nullptr;

        std::string render_settings_name = s_render_settings_default;
        std::map<std::string, pnanovdb_imgui_settings_render_t> saved_render_settings;

        std::vector<std::string> viewport_shaders;

        ProgressBar progress;

        pnanovdb_shader::run_shader_func_t run_shader = [this](const char* shaderName,
            pnanovdb_uint32_t grid_dim_x,
            pnanovdb_uint32_t grid_dim_y,
            pnanovdb_uint32_t grid_dim_z)
        {
            const uint32_t compileTarget = pnanovdb_shader::getCompileTarget(shaderName);
            if (compileTarget == PNANOVDB_COMPILE_TARGET_CPU)
            {
                assert(compiler);
                pnanovdb_compiler_instance_t* compiler_inst = compiler->create_instance();
                pnanovdb_compiler_settings_t compile_settings = {};
                pnanovdb_compiler_settings_init(&compile_settings);
                compiler->compile_shader_from_file(compiler_inst, shaderName, &compiler_settings, nullptr);

                UniformState uniformState = {};
                compiler->execute_cpu(compiler_inst, shaderName, grid_dim_x, grid_dim_y, grid_dim_z, nullptr, (void*)&uniformState);
                compiler->destroy_instance(compiler_inst);
            }
            else if (compileTarget == PNANOVDB_COMPILE_TARGET_VULKAN)
            {
                assert(compute);
                //compute->dispatch_shader_on_array(compute, compute_device, shaderName, grid_dim_x, grid_dim_y, grid_dim_z);
            }
        };
    };

    PNANOVDB_CAST_PAIR(pnanovdb_imgui_instance_t, Instance)
}

pnanovdb_imgui_instance_interface_t* get_user_imgui_instance_interface();
