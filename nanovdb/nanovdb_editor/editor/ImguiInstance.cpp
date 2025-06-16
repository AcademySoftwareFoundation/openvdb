// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/ImguiInstance.cpp

    \author Andrew Reidmeyer, Petra Hapalova

    \brief
*/

#include "ImguiInstance.h"

#include "imgui/ImguiRenderer.h"
#include "compute/ComputeShader.h"
#include "CodeEditor.h"
#include "Console.h"
#include "Node2Verify.h"
#include "Profiler.h"
#include "RenderSettingsHandler.h"

#include <ImGuiFileDialog.h>

#include <stdio.h>
#include <imgui_internal.h>           // for the docking branch
#include "misc/cpp/imgui_stdlib.h"    // for std::string text input

#if defined(_WIN32)
#include <Windows.h>
#else
#include <time.h>
#endif

#include <filesystem>

#define IMGUI_CHECKBOX_SYNC(label, var) \
    { \
        bool temp_bool = ((var) != PNANOVDB_FALSE); \
        if (ImGui::Checkbox((label), &temp_bool)) \
        { \
            (var) = temp_bool ? PNANOVDB_TRUE : PNANOVDB_FALSE; \
        } \
    }

PNANOVDB_INLINE void timestamp_capture(pnanovdb_uint64_t* ptr)
{
#if defined(_WIN32)
    LARGE_INTEGER tmpCpuTime = {};
    QueryPerformanceCounter(&tmpCpuTime);
    (*ptr) = tmpCpuTime.QuadPart;
#else
    timespec timeValue = {};
    clock_gettime(CLOCK_MONOTONIC, &timeValue);
    (*ptr) = 1E9 * pnanovdb_uint64_t(timeValue.tv_sec) + pnanovdb_uint64_t(timeValue.tv_nsec);
#endif
}
PNANOVDB_INLINE pnanovdb_uint64_t timestamp_frequency()
{
#if defined(_WIN32)
    LARGE_INTEGER tmpCpuFreq = {};
    QueryPerformanceFrequency(&tmpCpuFreq);
    return tmpCpuFreq.QuadPart;
#else
    return 1E9;
#endif
}
PNANOVDB_INLINE float timestamp_diff(pnanovdb_uint64_t begin, pnanovdb_uint64_t end, pnanovdb_uint64_t freq)
{
    return (float)(((double)(end - begin) / (double)(freq)));
}

namespace imgui_instance_user
{
pnanovdb_imgui_instance_t* create(void* userdata, void* user_settings, const pnanovdb_reflect_data_type_t* user_settings_data_type)
{
    auto ptr = new Instance();

    if (pnanovdb_reflect_layout_compare(user_settings_data_type, PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_imgui_settings_render_t)))
    {
        ptr->render_settings = (pnanovdb_imgui_settings_render_t*)user_settings;
    }

    *((Instance**)userdata) = ptr;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Register settings handlers
    ImGuiContext* context = ImGui::GetCurrentContext();
    RenderSettingsHandler::Register(context, ptr);
    pnanovdb_editor::CodeEditor::getInstance().registerSettingsHandler(context);

    return cast(ptr);
}

void destroy(pnanovdb_imgui_instance_t* instance)
{
    auto ptr = cast(instance);

    ImGui::DestroyContext();

    delete ptr;
}

static const char* VIEWPORT_SETTINGS = "Viewport";
static const char* RENDER_SETTINGS = "Render";
static const char* USER_SETTINGS = "User";
static const char* COMPILER_SETTINGS = "Compiler";
static const char* PROFILER_WINDOW = "Profiler";
static const char* CODE_EDITOR = "Shader Editor";
static const char* CONSOLE = "Output";
static const char* USER_PARAMS = "Params";
static const char* BENCHMARK = "Benchmark";

static void initializeDocking()
{
    ImGuiID dockspace_id = 1u;
    ImGui::DockSpaceOverViewport(dockspace_id, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);

    static bool is_docking_setup = false;
    if (!is_docking_setup)
    {
        // setup docking once
        is_docking_setup = true;

        float window_width = ImGui::GetIO().DisplaySize.x;
        float window_height = ImGui::GetIO().DisplaySize.y;

        float left_dock_width = window_width * 0.25f;
        float right_dock_width = window_width * 0.33f;
        float bottom_dock_height = window_height * 0.2f;

        // clear existing layout
        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);

        // create dock spaces for various windows, the order matters!
        ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Right, 0.f, nullptr, &dockspace_id);
        ImGui::DockBuilderSetNodeSize(dock_id_right, ImVec2(right_dock_width, window_height));
        ImGui::DockBuilderDockWindow(CODE_EDITOR, dock_id_right);
        ImGui::DockBuilderDockWindow(PROFILER_WINDOW, dock_id_right);

        ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.f, nullptr, &dockspace_id);
        ImGui::DockBuilderSetNodeSize(dock_id_bottom, ImVec2(window_width, bottom_dock_height));
        ImGui::DockBuilderDockWindow(CONSOLE, dock_id_bottom);

        ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.f, nullptr, &dockspace_id);
        ImGui::DockBuilderSetNodeSize(dock_id_left, ImVec2(left_dock_width, window_height));
        ImGui::DockBuilderDockWindow(VIEWPORT_SETTINGS, dock_id_left);
        ImGui::DockBuilderDockWindow(RENDER_SETTINGS, dock_id_left);
        ImGui::DockBuilderDockWindow(COMPILER_SETTINGS, dock_id_left);

        ImGuiID dock_id_left_bottom = ImGui::DockBuilderSplitNode(dock_id_left, ImGuiDir_Down, 0.f, nullptr, &dock_id_left);
        ImGui::DockBuilderSetNodeSize(dock_id_left_bottom, ImVec2(left_dock_width, window_height));
        ImGui::DockBuilderDockWindow(USER_SETTINGS, dock_id_left_bottom);
        ImGui::DockBuilderDockWindow(USER_PARAMS, dock_id_left_bottom);
        ImGui::DockBuilderDockWindow(BENCHMARK, dock_id_left_bottom);

        ImGui::DockBuilderFinish(dockspace_id);
    }
}

static void createMenu(Instance* ptr)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            ImGui::MenuItem("Open...", "", &ptr->pending.open_file);
            ImGui::MenuItem("Save...", "", &ptr->pending.save_file);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window"))
        {
            ImGui::MenuItem(CODE_EDITOR, "", &ptr->window.show_code_editor);
            ImGui::MenuItem(PROFILER_WINDOW, "", &ptr->window.show_profiler);
            ImGui::MenuItem(CONSOLE, "", &ptr->window.show_console);
            ImGui::MenuItem(USER_PARAMS, "", &ptr->window.show_user_params);
            ImGui::MenuItem(BENCHMARK, "", &ptr->window.show_benchmark);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Settings"))
        {
            ImGui::MenuItem("Viewport", "", &ptr->window.show_viewport_settings);
            ImGui::MenuItem("Render", "", &ptr->window.show_render_settings);
            ImGui::MenuItem("User", "", &ptr->window.show_user_settings);
            ImGui::MenuItem("Compiler", "", &ptr->window.show_compiler_settings);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

static void showWindows(Instance* ptr, float delta_time)
{
    if (ptr->window.show_viewport_settings)
    {
        if (ImGui::Begin(VIEWPORT_SETTINGS, &ptr->window.show_viewport_settings))
        {
            // viewport options
            {
                ImGui::BeginGroup();
                ViewportOption selectedOption = ptr->viewport_option;
                if (ImGui::RadioButton("NanoVDB", selectedOption == ViewportOption::NanoVDB))
                {
                    ptr->viewport_option = ViewportOption::NanoVDB;
                    ptr->render_settings_name = ptr->viewport_settings[(int)ptr->viewport_option].render_settings_name;
                    ptr->pending.load_camera = true;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Raster2D", selectedOption == ViewportOption::Raster2D))
                {
                    ptr->viewport_option = ViewportOption::Raster2D;
                    ptr->render_settings_name = ptr->viewport_settings[(int)ptr->viewport_option].render_settings_name;
                    ptr->pending.load_camera = true;
                }
                ImGui::EndGroup();
            }

            ImGui::Text("Camera View");
            {
                ImGui::BeginGroup();
                if (ImGui::BeginCombo("##render_settings", "Select..."))
                {
                    for (const auto& pair : ptr->saved_render_settings)
                    {
                        bool is_selected = (ptr->render_settings_name == pair.first);
                        if (ImGui::Selectable(pair.first.c_str(), is_selected))
                        {
                            ptr->render_settings_name = pair.first;
                            ptr->viewport_settings[(int)ptr->viewport_option].render_settings_name = ptr->render_settings_name;
                            ptr->pending.load_camera = true;
                        }
                        if (is_selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::InputText("##render_settings_name", &ptr->render_settings_name);
                ImGui::SameLine();
                if (ImGui::Button("Save"))
                {
                    if (!ptr->render_settings_name.empty())
                    {
                        if (ptr->saved_render_settings.find(ptr->render_settings_name) == ptr->saved_render_settings.end())
                        {
                            // save camera state in editor update loop
                            ptr->pending.save_camera = true;
                        }
                        else
                        {
                            pnanovdb_editor::Console::getInstance().addLog("Render settings '%s' already exists", ptr->render_settings_name.c_str());
                        }
                    }
                    else
                    {
                        pnanovdb_editor::Console::getInstance().addLog("Please enter a name for the render settings");
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Remove"))
                {
                    if (ptr->saved_render_settings.erase(ptr->render_settings_name))
                    {
                        pnanovdb_editor::Console::getInstance().addLog("Render settings '%s' removed", ptr->render_settings_name.c_str());
                        ptr->render_settings_name = "";
                    }
                    else
                    {
                        pnanovdb_editor::Console::getInstance().addLog("Render settings '%s' not found", ptr->render_settings_name.c_str());
                    }
                }
                ImGui::EndGroup();
            }

            ImGui::Text("Gaussian File");
            {
                ImGui::BeginGroup();
                ImGui::InputText("##viewport_raster_file", &ptr->raster_filepath);
                ImGui::SameLine();
                if (ImGui::Button("...##open_raster_file"))
                {
                    ptr->pending.find_raster_file = true;

                    IGFD::FileDialogConfig config;
                    config.path = ".";

                    ImGuiFileDialog::Instance()->OpenDialog("OpenRasterFileDlgKey", "Open Gaussian File", "Gaussian Files (*.npy *.npz *.ply){.npy,.npz,.ply}", config);
                }
                ImGui::SameLine();
                if (ImGui::Button("Show##Gaussian"))
                {
                    // runs rasterization on a worker thread first
                    ptr->pending.update_raster = true;
                }
                ImGui::InputFloat("VoxelsPerUnit", &ptr->raster_voxels_per_unit);
                ImGui::EndGroup();
            }

            if (ptr->viewport_option == ViewportOption::NanoVDB)
            {
                ImGui::Text("NanoVDB File");
                {
                    ImGui::BeginGroup();
                    ImGui::InputText("##viewport_nanovdb_file", &ptr->nanovdb_filepath);
                    ImGui::SameLine();
                    if (ImGui::Button("...##open_nanovddb_file"))
                    {
                        ptr->pending.open_file = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Show##NanoVDB"))
                    {
                        // TODO: does it need to be on a worker thread?
                        pnanovdb_editor::Console::getInstance().addLog("Opening file '%s'", ptr->nanovdb_filepath.c_str());
                        ptr->pending.load_nvdb = true;
                    }
                    ImGui::EndGroup();
                }
            }
            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_render_settings)
    {
        if (ImGui::Begin(RENDER_SETTINGS, &ptr->window.show_render_settings))
        {
            ImGui::Text("Viewport Shader");
            {
                ImGui::BeginGroup();
                if (ImGui::BeginCombo("##viewport_shader", "Select..."))
                {
                    for (const auto& shader : ptr->viewport_shaders)
                    {
                        if (ImGui::Selectable(shader.c_str()))
                        {
                            ptr->pending.shader_name = shader;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::InputText("##viewport_shader_name", &ptr->pending.shader_name);
                ImGui::SameLine();
                if (ImGui::Button("Update"))
                {
                    pnanovdb_editor::CodeEditor::getInstance().setSelectedShader(ptr->pending.shader_name);
                    ptr->shader_name = ptr->pending.shader_name;
                    ptr->pending.update_shader = true;
                }
                ImGui::EndGroup();
            }

            auto settings = ptr->render_settings;

            IMGUI_CHECKBOX_SYNC("VSync", settings->vsync);
            IMGUI_CHECKBOX_SYNC("Projection RH", settings->is_projection_rh);
            IMGUI_CHECKBOX_SYNC("Orthographic", settings->is_orthographic);
            IMGUI_CHECKBOX_SYNC("Reverse Z", settings->is_reverse_z);
            IMGUI_CHECKBOX_SYNC("Y up", settings->is_y_up);
            IMGUI_CHECKBOX_SYNC("Upside down", settings->is_upside_down);

            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_user_settings)
    {
        if (ImGui::Begin(USER_SETTINGS, &ptr->window.show_user_settings))
        {
            if (ptr->callable_func)
            {
                ImGui::BeginGroup();
                ImGui::InputText("##callable_file", &ptr->callable_file);
                ImGui::SameLine();
                if (ImGui::Button("...##open_callable_file"))
                {
                    ptr->pending.find_callable_file = false;

                    IGFD::FileDialogConfig config;
                    config.path = ".";

                    ImGuiFileDialog::Instance()->OpenDialog("FindCallableFileDlgKey", "Find File", "All Files (*.*){.*}", config);
                }
                if (ImGui::Button(ptr->callable_name.c_str()))
                {
                    std::thread workerThread([ptr]()
                        {
                            ptr->callable_func(ptr->callable_file.c_str(), &pnanovdb_editor::Profiler::report_callback);

                            pnanovdb_editor::Console::getInstance().addLog("Function '%s' was completed", ptr->callable_name.c_str());
                        });
                    workerThread.detach();

                    pnanovdb_editor::Console::getInstance().addLog("Function '%s' was started", ptr->callable_name.c_str());
                }
                ImGui::EndGroup();
            }
            ImGui::Checkbox("Highlight Bounding Box", &ptr->highlight_bbox);
            ImGui::Checkbox("Narrow Band Only", &ptr->narrow_band_only);
            ImGui::DragFloat("Slice Plane Thickness", &ptr->slice_plane_thickness, 1.f, 0.f, 100.0f);
            ImGui::DragFloat4("Slice Plane", ptr->slice_plane, 1.f, -1000.f, 1000.0f);
            ImGui::Checkbox("Animate Slice", &ptr->animate_slice);
            if (ptr->animate_slice)
            {
                ptr->slice_plane[3] += delta_time * 250.f;
                if (ptr->slice_plane[3] > 1000.f)
                {
                    ptr->slice_plane[3] = -1000.f;
                }
            }
            if (ImGui::Button("Print Slice"))
            {
                ptr->pending.print_slice = true;
            }
            if (ImGui::Button("Capture Image"))
            {
                ptr->pending.capture_image = true;
            }
            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_compiler_settings)
    {
        if (ImGui::Begin(COMPILER_SETTINGS, &ptr->window.show_compiler_settings))
        {
            IMGUI_CHECKBOX_SYNC("Row Major Matrix Layout", ptr->compiler_settings.is_row_major);
            IMGUI_CHECKBOX_SYNC("Create HLSL Output", ptr->compiler_settings.hlsl_output);
            // TODO: add downstream compilers dropdown
            IMGUI_CHECKBOX_SYNC("Use glslang", ptr->compiler_settings.use_glslang);

            const char* compile_targets[] = { "Select...", "Vulkan", "CPU" };
            int compile_target = (int)ptr->compiler_settings.compile_target;
            if (ImGui::Combo("Target", &compile_target, compile_targets, IM_ARRAYSIZE(compile_targets)))
            {
                ptr->compiler_settings.compile_target = (pnanovdb_compile_target_type_t)(compile_target);
            }
            if (ImGui::Button("Clear Shader Cache"))
            {
                const char* compiledShadersDir = PNANOVDB_REFLECT_XSTR(COMPILED_SHADERS_DIR);
                if (compiledShadersDir)
                {
                    std::filesystem::path shaderDir(compiledShadersDir);
                    if (std::filesystem::exists(shaderDir) && std::filesystem::is_directory(shaderDir))
                    {
                        for (const auto& entry : std::filesystem::directory_iterator(shaderDir))
                        {
                            std::filesystem::remove_all(entry.path());
                        }
                        pnanovdb_editor::Console::getInstance().addLog("Shader cache cleared");
                    }
                }
            }

            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_user_params)
    {
        if (ImGui::Begin(USER_PARAMS, &ptr->window.show_user_params))
        {
            if (ImGui::Button("Create"))
            {
                ptr->user_params.create(ptr->shader_name);
            }
            ImGui::SameLine();
            if (ImGui::Button("Save & Load"))
            {
                pnanovdb_editor::CodeEditor::getInstance().saveUserParams();

                if (!ptr->user_params.load(ptr->shader_name))
                {
                    pnanovdb_editor::Console::getInstance().addLog("Failed to reload user params for '%s'", ptr->shader_name.c_str());
                }
            }
            ptr->user_params.render();

            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_benchmark)
    {
        if (ImGui::Begin(BENCHMARK, &ptr->window.show_benchmark))
        {
            if (ImGui::Button("Run Benchmark"))
            {
                // For now, guess the ref nvdb by convention
                std::string ref_nvdb = ptr->nanovdb_filepath;
                std::string suffix = "_node2.nvdb";
                if (ref_nvdb.size() > suffix.size())
                {
                    ref_nvdb.erase(ref_nvdb.size() - suffix.size(), suffix.size());
                }
                ref_nvdb.append(".nvdb");
                printf("nanovdbFile_(%s) ref_nvdb(%s) suffix(%s)\n", ptr->nanovdb_filepath.c_str(), ref_nvdb.c_str(), suffix.c_str());
                pnanovdb_editor::node2_verify_gpu(ref_nvdb.c_str(), ptr->nanovdb_filepath.c_str());
            }
            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_code_editor)
    {
        pnanovdb_editor::CodeEditor::getInstance().setup(&ptr->shader_name, &ptr->pending.update_shader, ptr->dialog_size, ptr->run_shader);
        if (ImGui::Begin(CODE_EDITOR, &ptr->window.show_code_editor, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_MenuBar))
        {
            if (!pnanovdb_editor::CodeEditor::getInstance().render())
            {
                ptr->window.show_code_editor = false;
            }
            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_profiler)
    {
        if (ImGui::Begin(PROFILER_WINDOW, &ptr->window.show_profiler))
        {
            if (!pnanovdb_editor::Profiler::getInstance().render(&ptr->pending.update_memory_stats, delta_time))
            {
                ptr->window.show_profiler = false;
            }
            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }

    if (ptr->window.show_console)
    {
        if (ImGui::Begin(CONSOLE, &ptr->window.show_console))
        {
            if (!pnanovdb_editor::Console::getInstance().render())
            {
                ptr->window.show_console = false;
            }

            ImGui::ProgressBar(ptr->progress.value, ImVec2(-1, 0), "");

            ImVec2 pos = ImGui::GetItemRectMin();
            ImVec2 size_bar = ImGui::GetItemRectSize();
            ImVec2 text_size = ImGui::CalcTextSize(ptr->progress.text.c_str());
            ImVec2 text_pos = ImVec2(pos.x + 5.f, pos.y + (size_bar.y - text_size.y) * 0.5f);

            ImGui::GetWindowDrawList()->AddText(text_pos, ImGui::GetColorU32(ImGuiCol_Text), ptr->progress.text.c_str());

            ImGui::End();
        }
        else
        {
            ImGui::End();
        }
    }
}

void update(pnanovdb_imgui_instance_t* instance)
{
    auto ptr = cast(instance);

    // compute delta_time
    float delta_time = 0.f;
    pnanovdb_uint64_t end;
    timestamp_capture(&end);

    if (ptr->last_timestamp != 0llu)
    {
        delta_time = timestamp_diff(end, ptr->last_timestamp, timestamp_frequency());
    }
    ptr->last_timestamp = end;
    if (delta_time > 1.f / 30.f)    // limit maximum time step to cover hitching
    {
        delta_time = 1.f / 30.f;
    }

    ImGui::NewFrame();

    initializeDocking();

    createMenu(ptr);

    //bool show_demo_window = true;
    //ImGui::ShowDemoWindow(&show_demo_window);

    showWindows(ptr, delta_time);

    if (ptr->pending.update_hlsl)
    {
        pnanovdb_editor::CodeEditor::getInstance().updateViewer();
        ptr->pending.update_hlsl = false;
    }

    if (ptr->pending.save_render_settings)
    {
        ptr->pending.save_render_settings = false;

        // Mark settings as dirty to trigger save
        ImGui::MarkIniSettingsDirty();

        pnanovdb_editor::Console::getInstance().addLog("Render settings '%s' saved", ptr->render_settings_name.c_str());
    }
    if (ptr->pending.load_render_settings)
    {
        ptr->pending.load_render_settings = false;

        auto it = ptr->saved_render_settings.find(ptr->render_settings_name);
        if (it != ptr->saved_render_settings.end())
        {
            // Copy saved camera state to current camera state
            ptr->render_settings_name = it->first;
            //pnanovdb_editor::Console::getInstance().addLog("Render settings '%s' loaded", ptr->render_settings_name.c_str());
        }
        else
        {
            pnanovdb_editor::Console::getInstance().addLog("Render settings '%s' not found", ptr->render_settings_name.c_str());
        }
    }

    if (ptr->pending.open_file)
    {
        ptr->pending.open_file = false;

        IGFD::FileDialogConfig config;
        config.path = ".";

        ImGuiFileDialog::Instance()->OpenDialog("OpenNvdbFileDlgKey", "Open NanoVDB File", "NanoVDB Files (*.nvdb){.nvdb}", config);
    }
    if (ptr->pending.save_file)
    {
        ptr->pending.save_file = false;

        IGFD::FileDialogConfig config;
        config.path = ".";

        ImGuiFileDialog::Instance()->OpenDialog("SaveNvdbFileDlgKey", "Save NanoVDB File", "NanoVDB Files (*.nvdb){.nvdb}", config);
    }

    if (ImGuiFileDialog::Instance()->IsOpened())
    {
        ImGui::SetNextWindowSize(ptr->dialog_size, ImGuiCond_Appearing);
        if (ImGuiFileDialog::Instance()->Display("OpenNvdbFileDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                ptr->nanovdb_filepath = ImGuiFileDialog::Instance()->GetFilePathName();
                //ptr->nanovdb_path = ImGuiFileDialog::Instance()->GetCurrentPath();
                pnanovdb_editor::Console::getInstance().addLog("Opening file '%s'", ptr->nanovdb_filepath.c_str());
                ptr->pending.load_nvdb = true;
            }
            ImGuiFileDialog::Instance()->Close();
        }
        else if (ImGuiFileDialog::Instance()->Display("OpenRasterFileDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                ptr->raster_filepath = ImGuiFileDialog::Instance()->GetFilePathName();
                ptr->pending.find_raster_file = false;
            }
            ImGuiFileDialog::Instance()->Close();
        }
        else if (ImGuiFileDialog::Instance()->Display("SaveNvdbFileDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                ptr->nanovdb_filepath = ImGuiFileDialog::Instance()->GetFilePathName();
                pnanovdb_editor::Console::getInstance().addLog("Saving file '%s'...", ptr->nanovdb_filepath.c_str());
                ptr->pending.save_nanovdb = true;
            }
            ImGuiFileDialog::Instance()->Close();
        }
        else if (ImGuiFileDialog::Instance()->Display("FindCallableFileDlgKey"))
        {
            if (ImGuiFileDialog::Instance()->IsOk())
            {
                ptr->callable_file = ImGuiFileDialog::Instance()->GetFilePathName();
                ptr->pending.find_callable_file = false;
            }
            ImGuiFileDialog::Instance()->Close();
        }
    }

    ImGui::Render();
}

ImGuiStyle* get_style(pnanovdb_imgui_instance_t* instance)
{
    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();

    return &s;
}

ImGuiIO* get_io(pnanovdb_imgui_instance_t* instance)
{
    ImGuiIO& io = ImGui::GetIO();

    return &io;
}

void get_tex_data_as_rgba32(
    pnanovdb_imgui_instance_t* instance,
    unsigned char** out_pixels,
    int* out_width,
    int* out_height)
{
    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->GetTexDataAsRGBA32(out_pixels, out_width, out_height);
}

ImDrawData* get_draw_data(pnanovdb_imgui_instance_t* instance)
{
    return ImGui::GetDrawData();
}
}

pnanovdb_imgui_instance_interface_t* get_user_imgui_instance_interface()
{
    using namespace imgui_instance_user;
    static pnanovdb_imgui_instance_interface_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_imgui_instance_interface_t) };
    iface.create = create;
    iface.destroy = destroy;
    iface.update = update;
    iface.get_style = get_style;
    iface.get_io = get_io;
    iface.get_tex_data_as_rgba32 = get_tex_data_as_rgba32;
    iface.get_draw_data = get_draw_data;
    return &iface;
}
