// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ImguiWindow.cpp

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#if defined(__APPLE__)
#define VK_USE_PLATFORM_MACOS_MVK 1
#include <vulkan/vulkan.h>
#endif

#define GLFW_DLL

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#define PNANOVDB_SWAPCHAIN_DESC 1

#include <imgui.h>

#include "ImguiRenderer.h"
#include "ImguiWindow.h"

#include "nanovdb/putil/Camera.h"

#include <vector>

#define GLFW_PTR(X) decltype(&X) p_##X = nullptr

#define GLFW_PTR_LOAD(X) ptr->p_##X = (decltype(&X))pnanovdb_get_proc_address(ptr->glfw_module, #X)

namespace pnanovdb_imgui_window_default
{
    static pnanovdb_imgui_instance_interface_t* get_default_imgui_instance_interface();

    void setStyle_NvidiaDark(ImGuiStyle& s);

    void windowSizeCallback(GLFWwindow* win, int width, int height);
    void keyboardCallback(GLFWwindow* win, int key, int scanCode, int action, int modifiers);
    void charInputCallback(GLFWwindow* win, uint32_t input);
    void mouseMoveCallback(GLFWwindow* win, double mouseX, double mouseY);
    void mouseButtonCallback(GLFWwindow* win, int button, int action, int modifiers);
    void mouseWheelCallback(GLFWwindow* win, double scrollX, double scrollY);

    GLFW_PTR(glfwGetWindowUserPointer);

    struct ImguiInstance
    {
        pnanovdb_imgui_instance_interface_t instance_interface;
        pnanovdb_imgui_instance_t* instance;
        pnanovdb_imgui_renderer_interface_t renderer_interface;
        pnanovdb_imgui_renderer_t* renderer;
        void* userdata;
    };

    struct Window
    {
        void* glfw_module;

        GLFW_PTR(glfwInit);
        GLFW_PTR(glfwWindowHint);
        GLFW_PTR(glfwCreateWindow);
        GLFW_PTR(glfwGetPrimaryMonitor);
        GLFW_PTR(glfwGetVideoMode);
        GLFW_PTR(glfwSetWindowUserPointer);
        GLFW_PTR(glfwSetWindowPos);
        GLFW_PTR(glfwSetWindowSizeCallback);
        GLFW_PTR(glfwSetKeyCallback);
        GLFW_PTR(glfwSetCharCallback);
        GLFW_PTR(glfwSetMouseButtonCallback);
        GLFW_PTR(glfwSetCursorPosCallback);
        GLFW_PTR(glfwSetScrollCallback);
    #if defined(_WIN32)
        GLFW_PTR(glfwGetWin32Window);
    #elif defined(__APPLE__)
        GLFW_PTR(glfwGetCocoaView);
        GLFW_PTR(glfwCreateWindowSurface);
    #else
        GLFW_PTR(glfwGetX11Display);
        GLFW_PTR(glfwGetX11Window);
    #endif
        GLFW_PTR(glfwDestroyWindow);
        GLFW_PTR(glfwTerminate);
        GLFW_PTR(glfwPollEvents);
        GLFW_PTR(glfwWindowShouldClose);
        GLFW_PTR(glfwGetWindowUserPointer);
        GLFW_PTR(glfwSetWindowMonitor);
        GLFW_PTR(glfwGetMouseButton);
        GLFW_PTR(glfwGetFramebufferSize);
        GLFW_PTR(glfwGetKey);

        pnanovdb_compute_device_interface_t device_interface = {};
        pnanovdb_compute_interface_t compute_interface = {};

        pnanovdb_compute_swapchain_t* swapchain = nullptr;

        std::vector<ImguiInstance> imgui_instances;
        bool enable_default_imgui = false;

        GLFWwindow* window = nullptr;

        int mouse_x = 0;
        int mouse_y = 0;
        int mouse_y_inv = 0;
        pnanovdb_bool_t mouse_just_pressed[5u] = {};

        pnanovdb_uint32_t width = 0;
        pnanovdb_uint32_t height = 0;

        pnanovdb_camera_t camera = {};
    };

    PNANOVDB_CAST_PAIR(pnanovdb_imgui_window_t, Window)

    pnanovdb_imgui_window_t* create(
        const pnanovdb_compute_t* compute,
        const pnanovdb_compute_device_t* device,
        pnanovdb_int32_t width,
        pnanovdb_int32_t height,
        void** imgui_user_settings,
        pnanovdb_bool_t enable_default_imgui,
        pnanovdb_imgui_instance_interface_t** imgui_instance_interfaces,
        void** imgui_instance_userdatas,
        pnanovdb_uint64_t imgui_instance_instance_count)
    {
        pnanovdb_compute_queue_t* queue = compute->device_interface.get_device_queue(device);
        pnanovdb_compute_interface_t* compute_interface = compute->device_interface.get_compute_interface(queue);
        pnanovdb_compute_context_t* compute_context = compute->device_interface.get_compute_context(queue);

        const pnanovdb_compute_shader_interface_t* shader_interface = &compute->shader_interface;

        auto log_print = compute_interface->get_log_print(compute_context);

        void* glfw_module = pnanovdb_load_library("glfw3.dll", "libglfw.so.3", "libglfw.3.dylib");
        if (!glfw_module)
        {
            if (log_print)
            {
                log_print(PNANOVDB_COMPUTE_LOG_LEVEL_WARNING, "Failed to load GLFW, attempting typical absolute path.");

                #if defined(_WIN32)
                    // Print PATH environment variable
                    char* path = getenv("PATH");
                    if (path) {
                        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "\nPATH environment variable search paths:");
                        char* context = nullptr;
                        char* token = strtok_s(path, ";", &context);
                        while (token) {
                            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, token);
                            token = strtok_s(nullptr, ";", &context);
                        }
                    }

                    // Print application directory
                    WCHAR buffer[32768];
                    DWORD result = GetModuleFileNameW(NULL, buffer, 32768);
                    if (result > 0) {
                        WCHAR* lastSlash = wcsrchr(buffer, L'\\');
                        if (lastSlash) {
                            *lastSlash = L'\0';
                            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "\nApplication directory:");
                            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "%ls", buffer);
                        }
                    }

                    // Print system directory
                    if (GetSystemDirectoryW(buffer, 32768) > 0) {
                        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "\nSystem directory:");
                        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "%ls", buffer);
                    }

                    // Print Windows directory
                    if (GetWindowsDirectoryW(buffer, 32768) > 0) {
                        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "\nWindows directory:");
                        log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "%ls", buffer);
                    }
                #endif
            }
            glfw_module = pnanovdb_load_library("glfw3.dll", "/usr/lib/libglfw.so.3", "/opt/homebrew/lib/libglfw.3.dylib");
        }
        if (!glfw_module)
        {
            if (log_print)
            {
                log_print(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "Failed to load GLFW");
                return nullptr;
            }
        }

        auto ptr = new Window();
        auto settings = new pnanovdb_imgui_settings_render_t();
        settings->datatype = PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_imgui_settings_render_t);
        *imgui_user_settings = settings;

        ptr->width = width;
        ptr->height = height;

        ptr->glfw_module = glfw_module;

        GLFW_PTR_LOAD(glfwInit);
        GLFW_PTR_LOAD(glfwWindowHint);
        GLFW_PTR_LOAD(glfwCreateWindow);
        GLFW_PTR_LOAD(glfwGetPrimaryMonitor);
        GLFW_PTR_LOAD(glfwGetVideoMode);
        GLFW_PTR_LOAD(glfwSetWindowUserPointer);
        GLFW_PTR_LOAD(glfwSetWindowPos);
        GLFW_PTR_LOAD(glfwSetWindowSizeCallback);
        GLFW_PTR_LOAD(glfwSetKeyCallback);
        GLFW_PTR_LOAD(glfwSetCharCallback);
        GLFW_PTR_LOAD(glfwSetMouseButtonCallback);
        GLFW_PTR_LOAD(glfwSetCursorPosCallback);
        GLFW_PTR_LOAD(glfwSetScrollCallback);
    #if defined(_WIN32)
        GLFW_PTR_LOAD(glfwGetWin32Window);
    #elif defined(__APPLE__)
        GLFW_PTR_LOAD(glfwGetCocoaView);
        GLFW_PTR_LOAD(glfwCreateWindowSurface);
    #else
        GLFW_PTR_LOAD(glfwGetX11Display);
        GLFW_PTR_LOAD(glfwGetX11Window);
    #endif
        GLFW_PTR_LOAD(glfwDestroyWindow);
        GLFW_PTR_LOAD(glfwTerminate);
        GLFW_PTR_LOAD(glfwPollEvents);
        GLFW_PTR_LOAD(glfwWindowShouldClose);
        GLFW_PTR_LOAD(glfwGetWindowUserPointer);
        GLFW_PTR_LOAD(glfwSetWindowMonitor);
        GLFW_PTR_LOAD(glfwGetMouseButton);
        GLFW_PTR_LOAD(glfwGetFramebufferSize);
        GLFW_PTR_LOAD(glfwGetKey);

        // need global access on this one
        p_glfwGetWindowUserPointer = ptr->p_glfwGetWindowUserPointer;

        pnanovdb_compute_device_interface_t_duplicate(&ptr->device_interface, &compute->device_interface);
        pnanovdb_compute_interface_t_duplicate(&ptr->compute_interface, compute_interface);

        if (!ptr->p_glfwInit() && log_print)
        {
            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "Failed to initialize GLFW");
        }

        ptr->p_glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        //ptr->p_glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        const char* window_name = "NanoVDB Editor";

        ptr->window = ptr->p_glfwCreateWindow(ptr->width, ptr->height, window_name, nullptr, nullptr);
        if (!ptr->window && log_print)
        {
            log_print(PNANOVDB_COMPUTE_LOG_LEVEL_ERROR, "Failed to create GLFW window");
        }

        GLFWmonitor* monitor = ptr->p_glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = ptr->p_glfwGetVideoMode(monitor);

        ptr->p_glfwSetWindowUserPointer(ptr->window, ptr);

        ptr->p_glfwSetWindowPos(ptr->window, mode->width / 2 - ptr->width / 2, mode->height / 2 - ptr->height / 2);

        ptr->p_glfwSetWindowSizeCallback(ptr->window, windowSizeCallback);
        ptr->p_glfwSetKeyCallback(ptr->window, keyboardCallback);
        ptr->p_glfwSetCharCallback(ptr->window, charInputCallback);
        ptr->p_glfwSetMouseButtonCallback(ptr->window, mouseButtonCallback);
        ptr->p_glfwSetCursorPosCallback(ptr->window, mouseMoveCallback);
        ptr->p_glfwSetScrollCallback(ptr->window, mouseWheelCallback);

        // initialize swapchain
        pnanovdb_compute_swapchain_desc_t swapchain_desc = {};
        swapchain_desc.format = PNANOVDB_COMPUTE_FORMAT_B8G8R8A8_UNORM;
    #if defined(_WIN32)
        swapchain_desc.hwnd = ptr->p_glfwGetWin32Window(ptr->window);
        swapchain_desc.hinstance = (HINSTANCE)GetWindowLongPtr(swapchain_desc.hwnd, GWLP_HINSTANCE);
    #elif defined(__APPLE__)
        swapchain_desc.nsview = ptr->p_glfwGetCocoaView(ptr->window);
        auto get_framebuffer_size = [](void* window_userdata, int* width, int* height) {
            auto ptr = (Window*)window_userdata;
            ptr->p_glfwGetFramebufferSize(ptr->window, width, height);
        };
        swapchain_desc.get_framebuffer_size = get_framebuffer_size;
        swapchain_desc.window_userdata = ptr;
        auto create_surface = [](void* window_userdata, void* vkinstance, void** out_surface) {
            auto ptr = (Window*)window_userdata;
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            ptr->p_glfwCreateWindowSurface((VkInstance)vkinstance, ptr->window, nullptr, &surface);
            *out_surface = surface;
        };
        swapchain_desc.create_surface = create_surface;
    #else
        swapchain_desc.dpy = ptr->p_glfwGetX11Display();
        swapchain_desc.window = ptr->p_glfwGetX11Window(ptr->window);
    #endif

        ptr->swapchain = ptr->device_interface.create_swapchain(queue, &swapchain_desc);

        // initialize imgui
        ptr->imgui_instances.resize(0u);

        if (enable_default_imgui)
        {
            ptr->enable_default_imgui = true;
            ImguiInstance instance = {};
            pnanovdb_imgui_instance_interface_t_duplicate(
                &instance.instance_interface, get_default_imgui_instance_interface());
            instance.userdata = ptr;
            ptr->imgui_instances.push_back(instance);
        }
        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < imgui_instance_instance_count; instance_idx++)
        {
            ImguiInstance instance = {};
            pnanovdb_imgui_instance_interface_t_duplicate(
                &instance.instance_interface, imgui_instance_interfaces[instance_idx]);
            instance.userdata = imgui_instance_userdatas[instance_idx];
            ptr->imgui_instances.push_back(instance);
        }

        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            inst.instance = inst.instance_interface.create(inst.userdata, settings, settings->datatype);

            setStyle_NvidiaDark(*inst.instance_interface.get_style(inst.instance));

            ImGuiIO& io = *inst.instance_interface.get_io(inst.instance);

            // keymap obsolete
#if 0
            io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
            io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
            io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
            io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
            io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
            io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
            io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
            io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
            io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
            io.KeyMap[ImGuiKey_Insert] = GLFW_KEY_INSERT;
            io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
            io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
            io.KeyMap[ImGuiKey_Space] = GLFW_KEY_SPACE;
            io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
            io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
#if defined(_WIN32)
            io.KeyMap[ImGuiKey_KeypadEnter] = GLFW_KEY_KP_ENTER;        // ImGuiKey_KeyPadEnter is obsolete from 1.90.1
#else
            io.KeyMap[ImGuiKey_KeyPadEnter] = GLFW_KEY_KP_ENTER;
#endif
            io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
            io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
            io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
            io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
            io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
            io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;
#endif

            unsigned char* pixels = nullptr;
            int tex_width = 0;
            int tex_height = 0;
            inst.instance_interface.get_tex_data_as_rgba32(
                inst.instance,
                &pixels,
                &tex_width,
                &tex_height
            );

            pnanovdb_imgui_renderer_interface_t_duplicate(
                &inst.renderer_interface, pnanovdb_imgui_get_renderer_interface());

            inst.renderer = inst.renderer_interface.create(
                compute,
                queue,
                pixels,
                tex_width,
                tex_height
            );
        }

        pnanovdb_camera_init(&ptr->camera);

        return cast(ptr);
    }

    void destroy(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_imgui_window_t* window,
        pnanovdb_imgui_settings_render_t* settings)
    {
        auto ptr = cast(window);

        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            inst.instance_interface.destroy(inst.instance);
            inst.instance = nullptr;

            inst.renderer_interface.destroy(
                compute,
                queue,
                inst.renderer
            );
            inst.renderer = nullptr;
        }

        ptr->device_interface.destroy_swapchain(ptr->swapchain);

        ptr->p_glfwDestroyWindow(ptr->window);

        ptr->p_glfwTerminate();

        pnanovdb_free_library(ptr->glfw_module);

        delete ptr;
        delete settings;
    }

    pnanovdb_bool_t update(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* compute_queue,
        pnanovdb_compute_texture_transient_t* background,
        pnanovdb_int32_t* out_width,
        pnanovdb_int32_t* out_height,
        pnanovdb_imgui_window_t* window,
        pnanovdb_imgui_settings_render_t* user_settings)
    {
        auto ptr = cast(window);

        pnanovdb_compute_context_t* context = compute->device_interface.get_compute_context(compute_queue);
        auto log_print = ptr->compute_interface.get_log_print(context);

        float delta_time = 1.f / 60.f;

        pnanovdb_camera_animation_tick(&ptr->camera, delta_time);

        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            ImGuiIO& io = *inst.instance_interface.get_io(inst.instance);

            io.DisplaySize = ImVec2(float(ptr->width), float(ptr->height));
            io.DeltaTime = delta_time;
            for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); i++)
            {
                io.MouseDown[i] = ptr->mouse_just_pressed[i] != PNANOVDB_FALSE || ptr->p_glfwGetMouseButton(ptr->window, i) != 0;
                ptr->mouse_just_pressed[i] = PNANOVDB_FALSE;
            }
            io.MousePos.x = (float)ptr->mouse_x;
            io.MousePos.y = (float)ptr->mouse_y;

            inst.instance_interface.update(inst.instance);
        }

        pnanovdb_compute_texture_t* swapchain_texture =
            ptr->device_interface.get_swapchain_front_texture(ptr->swapchain);

        if (swapchain_texture == nullptr)
        {
            return PNANOVDB_FALSE;
        }

        pnanovdb_compute_texture_transient_t* swapchain_transient =
            ptr->compute_interface.register_texture_as_transient(context, swapchain_texture);

        pnanovdb_compute_texture_transient_t* front_texture = background;
        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            ImDrawData* draw_data = inst.instance_interface.get_draw_data(inst.instance);

            pnanovdb_compute_texture_desc_t tex_desc = {};
            tex_desc.texture_type = PNANOVDB_COMPUTE_TEXTURE_TYPE_2D;
            tex_desc.usage = PNANOVDB_COMPUTE_TEXTURE_USAGE_TEXTURE | PNANOVDB_COMPUTE_TEXTURE_USAGE_RW_TEXTURE;
            tex_desc.format = PNANOVDB_COMPUTE_FORMAT_R8G8B8A8_UNORM;
            tex_desc.width = ptr->width;
            tex_desc.height = ptr->height;
            tex_desc.depth = 1u;
            tex_desc.mip_levels = 1u;

            pnanovdb_compute_texture_transient_t* back_texture =
                ptr->compute_interface.get_texture_transient(context, &tex_desc);

            inst.renderer_interface.render(
                compute,
                context,
                inst.renderer,
                draw_data,
                ptr->width,
                ptr->height,
                front_texture,
                back_texture
            );

            // update front_texture
            front_texture = back_texture;
        }
        // copy final texture to swapchain
        if (ptr->imgui_instances.size() != 0u)
        {
            auto& inst = ptr->imgui_instances[0u];
            inst.renderer_interface.copy_texture(
                compute,
                context,
                inst.renderer,
                ptr->width,
                ptr->height,
                front_texture,
                swapchain_transient
            );
        }

        pnanovdb_uint64_t flushed_frame = 0llu;
        ptr->device_interface.present_swapchain(ptr->swapchain, user_settings->vsync, &flushed_frame);

        ptr->p_glfwPollEvents();

        if (out_width)
        {
            *out_width = ptr->width;
        }
        if (out_height)
        {
            *out_height = ptr->height;
        }

        if (ptr->p_glfwWindowShouldClose(ptr->window))
        {
            if (log_print)
            {
                log_print(PNANOVDB_COMPUTE_LOG_LEVEL_INFO, "GLFW Close Window.");
            }
            return PNANOVDB_FALSE;
        }

        return PNANOVDB_TRUE;
    }

    void get_camera(
        pnanovdb_imgui_window_t* window,
        pnanovdb_int32_t* out_width,
        pnanovdb_int32_t* out_height,
        pnanovdb_camera_mat_t* out_view,
        pnanovdb_camera_mat_t* out_projection
    )
    {
        auto ptr = cast(window);

        if (out_width)
        {
            *out_width = ptr->width;
        }
        if (out_height)
        {
            *out_height = ptr->height;
        }
        if (out_view)
        {
            pnanovdb_camera_get_view(&ptr->camera, out_view);
        }
        if (out_projection)
        {
            pnanovdb_camera_get_projection(&ptr->camera, out_projection, (float)ptr->width, (float)ptr->height);
        }
    }

    void get_camera_state(pnanovdb_imgui_window_t* window, pnanovdb_camera_state_t* camera_state)
    {
        auto ptr = cast(window);

        *camera_state = ptr->camera.state;
    }

    void update_camera(pnanovdb_imgui_window_t* window, pnanovdb_imgui_settings_render_t* user_settings)
    {
        auto ptr = cast(window);

        if (user_settings->is_projection_rh != ptr->camera.config.is_projection_rh)
        {
            ptr->camera.config.is_projection_rh = user_settings->is_projection_rh;
        }
        if (user_settings->is_reverse_z != ptr->camera.config.is_reverse_z)
        {
            ptr->camera.config.is_reverse_z = user_settings->is_reverse_z;
            if (ptr->camera.config.is_reverse_z && !ptr->camera.config.is_orthographic)
            {
                ptr->camera.config.far_plane = INFINITY;
            }
            else
            {
                ptr->camera.config.far_plane = 10000.f;
            }
        }
        if (user_settings->is_orthographic != ptr->camera.config.is_orthographic)
        {
            ptr->camera.config.is_orthographic = user_settings->is_orthographic;
            if (ptr->camera.config.is_reverse_z && !ptr->camera.config.is_orthographic)
            {
                ptr->camera.config.far_plane = INFINITY;
            }
            else
            {
                ptr->camera.config.far_plane = 10000.f;
            }
        }
        if (user_settings->is_y_up != fabsf(ptr->camera.state.eye_up.y) > 0.5f)
        {
            pnanovdb_camera_state_default(&ptr->camera.state, user_settings->is_y_up);
        }
        if (user_settings->is_upside_down == (ptr->camera.state.eye_up.y + ptr->camera.state.eye_up.z) > 0.f)
        {
            ptr->camera.state.eye_up.y = -ptr->camera.state.eye_up.y;
            ptr->camera.state.eye_up.z = -ptr->camera.state.eye_up.z;
        }
        if (user_settings->sync_camera_state)
        {
            ptr->camera.state = user_settings->camera_state;
            user_settings->sync_camera_state = false;
        }
    }

    void windowSizeCallback(GLFWwindow* win, int width, int height)
    {
        auto ptr = (Window*)p_glfwGetWindowUserPointer(win);

        // resize swapchain
        ptr->device_interface.resize_swapchain(
            ptr->swapchain, (pnanovdb_uint32_t)width, (pnanovdb_uint32_t)height);

        if (width == 0 || height == 0)
        {
            return;
        }

        ptr->width = width;
        ptr->height = height;
    }

    struct Instance
    {
        Window* window = nullptr;
        pnanovdb_imgui_settings_render_t* settings = nullptr;
    };

    PNANOVDB_CAST_PAIR(pnanovdb_imgui_instance_t, Instance)

    pnanovdb_imgui_instance_t* imgui_create(void* userdata, void* user_settings, const pnanovdb_reflect_data_type_t* user_settings_data_type)
    {
        auto ptr = new Instance();

        ptr->window = (Window*)userdata;

        if (pnanovdb_reflect_layout_compare(user_settings_data_type, PNANOVDB_REFLECT_DATA_TYPE(pnanovdb_imgui_settings_render_t)))
        {
            ptr->settings = (pnanovdb_imgui_settings_render_t*)user_settings;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        return cast(ptr);
    }

    void imgui_destroy(pnanovdb_imgui_instance_t* instance)
    {
        auto ptr = cast(instance);

        ImGui::DestroyContext();

        delete ptr;
    }

    void imgui_update(pnanovdb_imgui_instance_t* instance)
    {
        auto inst = cast(instance);
        auto ptr = (Window*)inst->window;

        ImGui::NewFrame();

        if (inst->settings->is_projection_rh != ptr->camera.config.is_projection_rh)
        {
            ptr->camera.config.is_projection_rh = !ptr->camera.config.is_projection_rh;
        }
        if (inst->settings->is_orthographic != ptr->camera.config.is_orthographic)
        {
            ptr->camera.config.is_orthographic = !ptr->camera.config.is_orthographic;
            if (ptr->camera.config.is_reverse_z && !ptr->camera.config.is_orthographic)
            {
                ptr->camera.config.far_plane = INFINITY;
            }
            else
            {
                ptr->camera.config.far_plane = 10000.f;
            }
        }
        if (inst->settings->is_reverse_z != ptr->camera.config.is_reverse_z)
        {
            ptr->camera.config.is_reverse_z = !ptr->camera.config.is_reverse_z;
            if (ptr->camera.config.is_reverse_z && !ptr->camera.config.is_orthographic)
            {
                ptr->camera.config.far_plane = INFINITY;
            }
            else
            {
                ptr->camera.config.far_plane = 10000.f;
            }
        }
        if (inst->settings->is_y_up != fabsf(ptr->camera.state.eye_up.y) > 0.5f)
        {
            pnanovdb_camera_state_default(&ptr->camera.state, inst->settings->is_y_up);
        }
        if (inst->settings->is_upside_down == (ptr->camera.state.eye_up.y + ptr->camera.state.eye_up.z) > 0.f)
        {
            ptr->camera.state.eye_up.y = -ptr->camera.state.eye_up.y;
            ptr->camera.state.eye_up.z = -ptr->camera.state.eye_up.z;
        }

        ImGui::Render();
    }

    ImGuiStyle* imgui_get_style(pnanovdb_imgui_instance_t* instance)
    {
        ImGui::StyleColorsDark();
        ImGuiStyle& s = ImGui::GetStyle();

        return &s;
    }

    ImGuiIO* imgui_get_io(pnanovdb_imgui_instance_t* instance)
    {
        ImGuiIO& io = ImGui::GetIO();

        return &io;
    }

    void imgui_get_tex_data_as_rgba32(
        pnanovdb_imgui_instance_t* instance,
        unsigned char** out_pixels,
        int* out_width,
        int* out_height)
    {
        ImGuiIO& io = ImGui::GetIO();

        io.Fonts->GetTexDataAsRGBA32(out_pixels, out_width, out_height);
    }

    ImDrawData* imgui_get_draw_data(pnanovdb_imgui_instance_t* instance)
    {
        return ImGui::GetDrawData();
    }

    static pnanovdb_imgui_instance_interface_t* get_default_imgui_instance_interface()
    {
        static pnanovdb_imgui_instance_interface_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_imgui_instance_interface_t) };
        iface.create = imgui_create;
        iface.destroy = imgui_destroy;
        iface.update = imgui_update;
        iface.get_style = imgui_get_style;
        iface.get_io = imgui_get_io;
        iface.get_tex_data_as_rgba32 = imgui_get_tex_data_as_rgba32;
        iface.get_draw_data = imgui_get_draw_data;
        return &iface;
    }

    ImGuiKey keyToImguiKey(int keycode);

    void keyboardCallback(GLFWwindow* win, int key, int scanCode, int action, int modifiers)
    {
        auto ptr = (Window*)p_glfwGetWindowUserPointer(win);

        bool zeroWantCaptureKeyboard = true;
        bool zeroWantCaptureMouse = true;
        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            ImGuiIO& io = *inst.instance_interface.get_io(inst.instance);

            if (io.WantCaptureKeyboard)
            {
                zeroWantCaptureKeyboard = true;
            }
            if (io.WantCaptureMouse)
            {
                zeroWantCaptureMouse = false;
            }
            // imgui always captures
            {
                io.AddKeyEvent(ImGuiMod_Ctrl,
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) ||
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS));
                io.AddKeyEvent(ImGuiMod_Shift,
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_LEFT_SHIFT)   == GLFW_PRESS) ||
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS));
                io.AddKeyEvent(ImGuiMod_Alt,
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_LEFT_ALT)     == GLFW_PRESS) ||
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_RIGHT_ALT)     == GLFW_PRESS));
                io.AddKeyEvent(ImGuiMod_Super,
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_LEFT_SUPER)   == GLFW_PRESS) ||
                    (ptr->p_glfwGetKey(ptr->window, GLFW_KEY_RIGHT_SUPER)   == GLFW_PRESS));
                if (action == GLFW_PRESS)
                {
                    io.AddKeyEvent(keyToImguiKey(key), true);
                }
                else if (action == GLFW_RELEASE)
                {
                    io.AddKeyEvent(keyToImguiKey(key), false);
                }
            }
        }
        if (zeroWantCaptureKeyboard && zeroWantCaptureMouse)
        {
            pnanovdb_camera_action_t p_action = PNANOVDB_CAMERA_ACTION_UNKNOWN;
            if (action == GLFW_PRESS)
            {
                p_action = PNANOVDB_CAMERA_ACTION_DOWN;
            }
            else if (action == GLFW_RELEASE)
            {
                p_action = PNANOVDB_CAMERA_ACTION_UP;
            }
            pnanovdb_camera_key_t p_key = PNANOVDB_CAMERA_KEY_UNKNOWN;
            if (key == GLFW_KEY_UP)
            {
                p_key = PNANOVDB_CAMERA_KEY_UP;
            }
            else if (key == GLFW_KEY_DOWN)
            {
                p_key = PNANOVDB_CAMERA_KEY_DOWN;
            }
            else if (key == GLFW_KEY_LEFT)
            {
                p_key = PNANOVDB_CAMERA_KEY_LEFT;
            }
            else if (key == GLFW_KEY_RIGHT)
            {
                p_key = PNANOVDB_CAMERA_KEY_RIGHT;
            }
            pnanovdb_camera_key_update(&ptr->camera, p_key, p_action);
        }
    }

    void charInputCallback(GLFWwindow* win, uint32_t input)
    {
        auto ptr = (Window*)p_glfwGetWindowUserPointer(win);

        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            ImGuiIO& io = *inst.instance_interface.get_io(inst.instance);
            // imgui always captures
            {
                io.AddInputCharacter(input);
            }
        }
    }

    void mouseMoveCallback(GLFWwindow* win, double mouseX, double mouseY)
    {
        auto ptr = (Window*)p_glfwGetWindowUserPointer(win);

        int x = int(mouseX);
        int y = int(mouseY);

        ptr->mouse_x = x;
        ptr->mouse_y = y;
        ptr->mouse_y_inv = ptr->height - 1 - y;

        bool zeroWantCaptureMouse = true;
        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            ImGuiIO& io = *inst.instance_interface.get_io(inst.instance);

            if (io.WantCaptureMouse)
            {
                zeroWantCaptureMouse = false;
            }
        }
        if (zeroWantCaptureMouse)
        {
            pnanovdb_camera_mouse_update(&ptr->camera, PNANOVDB_CAMERA_MOUSE_BUTTON_UNKNOWN, PNANOVDB_CAMERA_ACTION_UNKNOWN,
                ptr->mouse_x, ptr->mouse_y, (int)ptr->width, (int)ptr->height);
        }
    }

    void mouseButtonCallback(GLFWwindow* win, int button, int action, int modifiers)
    {
        auto ptr = (Window*)p_glfwGetWindowUserPointer(win);

        bool zeroWantCaptureMouse = true;
        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            ImGuiIO& io = *inst.instance_interface.get_io(inst.instance);

            if (io.WantCaptureMouse)
            {
                zeroWantCaptureMouse = false;
            }

            // imgui
            if (action == GLFW_PRESS && button >= 0 && button < 5)
            {
                ptr->mouse_just_pressed[button] = PNANOVDB_TRUE;
            }
        }
        if (zeroWantCaptureMouse)
        {
            pnanovdb_camera_action_t p_action = PNANOVDB_CAMERA_ACTION_UNKNOWN;
            if (action == GLFW_PRESS)
            {
                p_action = PNANOVDB_CAMERA_ACTION_DOWN;
            }
            else if (action == GLFW_RELEASE)
            {
                p_action = PNANOVDB_CAMERA_ACTION_UP;
            }
            pnanovdb_camera_mouse_button_t p_mouse = PNANOVDB_CAMERA_MOUSE_BUTTON_UNKNOWN;
            if (button == GLFW_MOUSE_BUTTON_LEFT)
            {
                p_mouse = PNANOVDB_CAMERA_MOUSE_BUTTON_LEFT;
            }
            else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
            {
                p_mouse = PNANOVDB_CAMERA_MOUSE_BUTTON_MIDDLE;
            }
            else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            {
                p_mouse = PNANOVDB_CAMERA_MOUSE_BUTTON_RIGHT;
            }
            pnanovdb_camera_mouse_update(&ptr->camera, p_mouse, p_action, ptr->mouse_x, ptr->mouse_y,
                (int)ptr->width, (int)ptr->height);
        }
    }

    void mouseWheelCallback(GLFWwindow* win, double scrollX, double scrollY)
    {
        auto ptr = (Window*)p_glfwGetWindowUserPointer(win);

        for (pnanovdb_uint64_t instance_idx = 0u; instance_idx < ptr->imgui_instances.size(); instance_idx++)
        {
            auto& inst = ptr->imgui_instances[instance_idx];

            ImGuiIO& io = *inst.instance_interface.get_io(inst.instance);

            io.MouseWheelH += (float)scrollX;
            io.MouseWheel += (float)scrollY;
        }
    }

    ImGuiKey keyToImguiKey(int keycode)
    {
        switch (keycode)
        {
            case GLFW_KEY_TAB: return ImGuiKey_Tab;
            case GLFW_KEY_LEFT: return ImGuiKey_LeftArrow;
            case GLFW_KEY_RIGHT: return ImGuiKey_RightArrow;
            case GLFW_KEY_UP: return ImGuiKey_UpArrow;
            case GLFW_KEY_DOWN: return ImGuiKey_DownArrow;
            case GLFW_KEY_PAGE_UP: return ImGuiKey_PageUp;
            case GLFW_KEY_PAGE_DOWN: return ImGuiKey_PageDown;
            case GLFW_KEY_HOME: return ImGuiKey_Home;
            case GLFW_KEY_END: return ImGuiKey_End;
            case GLFW_KEY_INSERT: return ImGuiKey_Insert;
            case GLFW_KEY_DELETE: return ImGuiKey_Delete;
            case GLFW_KEY_BACKSPACE: return ImGuiKey_Backspace;
            case GLFW_KEY_SPACE: return ImGuiKey_Space;
            case GLFW_KEY_ENTER: return ImGuiKey_Enter;
            case GLFW_KEY_ESCAPE: return ImGuiKey_Escape;
            case GLFW_KEY_APOSTROPHE: return ImGuiKey_Apostrophe;
            case GLFW_KEY_COMMA: return ImGuiKey_Comma;
            case GLFW_KEY_MINUS: return ImGuiKey_Minus;
            case GLFW_KEY_PERIOD: return ImGuiKey_Period;
            case GLFW_KEY_SLASH: return ImGuiKey_Slash;
            case GLFW_KEY_SEMICOLON: return ImGuiKey_Semicolon;
            case GLFW_KEY_EQUAL: return ImGuiKey_Equal;
            case GLFW_KEY_LEFT_BRACKET: return ImGuiKey_LeftBracket;
            case GLFW_KEY_BACKSLASH: return ImGuiKey_Backslash;
            case GLFW_KEY_RIGHT_BRACKET: return ImGuiKey_RightBracket;
            case GLFW_KEY_GRAVE_ACCENT: return ImGuiKey_GraveAccent;
            case GLFW_KEY_CAPS_LOCK: return ImGuiKey_CapsLock;
            case GLFW_KEY_SCROLL_LOCK: return ImGuiKey_ScrollLock;
            case GLFW_KEY_NUM_LOCK: return ImGuiKey_NumLock;
            case GLFW_KEY_PRINT_SCREEN: return ImGuiKey_PrintScreen;
            case GLFW_KEY_PAUSE: return ImGuiKey_Pause;
            case GLFW_KEY_KP_0: return ImGuiKey_Keypad0;
            case GLFW_KEY_KP_1: return ImGuiKey_Keypad1;
            case GLFW_KEY_KP_2: return ImGuiKey_Keypad2;
            case GLFW_KEY_KP_3: return ImGuiKey_Keypad3;
            case GLFW_KEY_KP_4: return ImGuiKey_Keypad4;
            case GLFW_KEY_KP_5: return ImGuiKey_Keypad5;
            case GLFW_KEY_KP_6: return ImGuiKey_Keypad6;
            case GLFW_KEY_KP_7: return ImGuiKey_Keypad7;
            case GLFW_KEY_KP_8: return ImGuiKey_Keypad8;
            case GLFW_KEY_KP_9: return ImGuiKey_Keypad9;
            case GLFW_KEY_KP_DECIMAL: return ImGuiKey_KeypadDecimal;
            case GLFW_KEY_KP_DIVIDE: return ImGuiKey_KeypadDivide;
            case GLFW_KEY_KP_MULTIPLY: return ImGuiKey_KeypadMultiply;
            case GLFW_KEY_KP_SUBTRACT: return ImGuiKey_KeypadSubtract;
            case GLFW_KEY_KP_ADD: return ImGuiKey_KeypadAdd;
            case GLFW_KEY_KP_ENTER: return ImGuiKey_KeypadEnter;
            case GLFW_KEY_KP_EQUAL: return ImGuiKey_KeypadEqual;
            case GLFW_KEY_LEFT_SHIFT: return ImGuiKey_LeftShift;
            case GLFW_KEY_LEFT_CONTROL: return ImGuiKey_LeftCtrl;
            case GLFW_KEY_LEFT_ALT: return ImGuiKey_LeftAlt;
            case GLFW_KEY_LEFT_SUPER: return ImGuiKey_LeftSuper;
            case GLFW_KEY_RIGHT_SHIFT: return ImGuiKey_RightShift;
            case GLFW_KEY_RIGHT_CONTROL: return ImGuiKey_RightCtrl;
            case GLFW_KEY_RIGHT_ALT: return ImGuiKey_RightAlt;
            case GLFW_KEY_RIGHT_SUPER: return ImGuiKey_RightSuper;
            case GLFW_KEY_MENU: return ImGuiKey_Menu;
            case GLFW_KEY_0: return ImGuiKey_0;
            case GLFW_KEY_1: return ImGuiKey_1;
            case GLFW_KEY_2: return ImGuiKey_2;
            case GLFW_KEY_3: return ImGuiKey_3;
            case GLFW_KEY_4: return ImGuiKey_4;
            case GLFW_KEY_5: return ImGuiKey_5;
            case GLFW_KEY_6: return ImGuiKey_6;
            case GLFW_KEY_7: return ImGuiKey_7;
            case GLFW_KEY_8: return ImGuiKey_8;
            case GLFW_KEY_9: return ImGuiKey_9;
            case GLFW_KEY_A: return ImGuiKey_A;
            case GLFW_KEY_B: return ImGuiKey_B;
            case GLFW_KEY_C: return ImGuiKey_C;
            case GLFW_KEY_D: return ImGuiKey_D;
            case GLFW_KEY_E: return ImGuiKey_E;
            case GLFW_KEY_F: return ImGuiKey_F;
            case GLFW_KEY_G: return ImGuiKey_G;
            case GLFW_KEY_H: return ImGuiKey_H;
            case GLFW_KEY_I: return ImGuiKey_I;
            case GLFW_KEY_J: return ImGuiKey_J;
            case GLFW_KEY_K: return ImGuiKey_K;
            case GLFW_KEY_L: return ImGuiKey_L;
            case GLFW_KEY_M: return ImGuiKey_M;
            case GLFW_KEY_N: return ImGuiKey_N;
            case GLFW_KEY_O: return ImGuiKey_O;
            case GLFW_KEY_P: return ImGuiKey_P;
            case GLFW_KEY_Q: return ImGuiKey_Q;
            case GLFW_KEY_R: return ImGuiKey_R;
            case GLFW_KEY_S: return ImGuiKey_S;
            case GLFW_KEY_T: return ImGuiKey_T;
            case GLFW_KEY_U: return ImGuiKey_U;
            case GLFW_KEY_V: return ImGuiKey_V;
            case GLFW_KEY_W: return ImGuiKey_W;
            case GLFW_KEY_X: return ImGuiKey_X;
            case GLFW_KEY_Y: return ImGuiKey_Y;
            case GLFW_KEY_Z: return ImGuiKey_Z;
            case GLFW_KEY_F1: return ImGuiKey_F1;
            case GLFW_KEY_F2: return ImGuiKey_F2;
            case GLFW_KEY_F3: return ImGuiKey_F3;
            case GLFW_KEY_F4: return ImGuiKey_F4;
            case GLFW_KEY_F5: return ImGuiKey_F5;
            case GLFW_KEY_F6: return ImGuiKey_F6;
            case GLFW_KEY_F7: return ImGuiKey_F7;
            case GLFW_KEY_F8: return ImGuiKey_F8;
            case GLFW_KEY_F9: return ImGuiKey_F9;
            case GLFW_KEY_F10: return ImGuiKey_F10;
            case GLFW_KEY_F11: return ImGuiKey_F11;
            case GLFW_KEY_F12: return ImGuiKey_F12;
            case GLFW_KEY_F13: return ImGuiKey_F13;
            case GLFW_KEY_F14: return ImGuiKey_F14;
            case GLFW_KEY_F15: return ImGuiKey_F15;
            case GLFW_KEY_F16: return ImGuiKey_F16;
            case GLFW_KEY_F17: return ImGuiKey_F17;
            case GLFW_KEY_F18: return ImGuiKey_F18;
            case GLFW_KEY_F19: return ImGuiKey_F19;
            case GLFW_KEY_F20: return ImGuiKey_F20;
            case GLFW_KEY_F21: return ImGuiKey_F21;
            case GLFW_KEY_F22: return ImGuiKey_F22;
            case GLFW_KEY_F23: return ImGuiKey_F23;
            case GLFW_KEY_F24: return ImGuiKey_F24;
            default: return ImGuiKey_None;
        }
    }

    void setStyle_NvidiaDark(ImGuiStyle& s)
    {
        s.FrameRounding = 4.0f;

        // Settings
        s.WindowPadding = ImVec2(8.0f, 8.0f);
        s.PopupRounding = 4.0f;
        s.FramePadding = ImVec2(8.0f, 4.0f);
        s.ItemSpacing = ImVec2(6.0f, 6.0f);
        s.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
        s.TouchExtraPadding = ImVec2(0.0f, 0.0f);
        s.IndentSpacing = 21.0f;
        s.ScrollbarSize = 16.0f;
        s.GrabMinSize = 8.0f;

        // BorderSize
        s.WindowBorderSize = 1.0f;
        s.ChildBorderSize = 1.0f;
        s.PopupBorderSize = 1.0f;
        s.FrameBorderSize = 0.0f;
        s.TabBorderSize = 0.0f;

        // Rounding
        s.WindowRounding = 4.0f;
        s.ChildRounding = 4.0f;
        s.FrameRounding = 4.0f;
        s.ScrollbarRounding = 4.0f;
        s.GrabRounding = 4.0f;
        s.TabRounding = 4.0f;

        // Alignment
        s.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        s.ButtonTextAlign = ImVec2(0.48f, 0.5f);

        s.DisplaySafeAreaPadding = ImVec2(3.0f, 3.0f);

        // Colors
        s.Colors[::ImGuiCol_Text] = ImVec4(0.89f, 0.89f, 0.89f, 1.00f);
        s.Colors[::ImGuiCol_Text] = ImVec4(0.89f, 0.89f, 0.89f, 1.00f);
        s.Colors[::ImGuiCol_TextDisabled] = ImVec4(0.43f, 0.43f, 0.43f, 1.00f);
        s.Colors[::ImGuiCol_WindowBg] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
        s.Colors[::ImGuiCol_ChildBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        s.Colors[::ImGuiCol_PopupBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        s.Colors[::ImGuiCol_Border] = ImVec4(0.29f, 0.29f, 0.29f, 1.00f);
        s.Colors[::ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
        s.Colors[::ImGuiCol_FrameBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        s.Colors[::ImGuiCol_FrameBgHovered] = ImVec4(0.29f, 0.29f, 0.29f, 1.00f);
        s.Colors[::ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_TitleBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        s.Colors[::ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        s.Colors[::ImGuiCol_TitleBgCollapsed] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        s.Colors[::ImGuiCol_MenuBarBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        s.Colors[::ImGuiCol_ScrollbarBg] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_ScrollbarGrab] = ImVec4(0.51f, 0.50f, 0.50f, 1.00f);
        s.Colors[::ImGuiCol_ScrollbarGrabHovered] = ImVec4(1.00f, 0.99f, 0.99f, 0.58f);
        s.Colors[::ImGuiCol_ScrollbarGrabActive] = ImVec4(0.47f, 0.53f, 0.54f, 0.76f);
        s.Colors[::ImGuiCol_CheckMark] = ImVec4(0.89f, 0.89f, 0.89f, 1.00f);
        s.Colors[::ImGuiCol_SliderGrab] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
        s.Colors[::ImGuiCol_SliderGrabActive] = ImVec4(0.47f, 0.53f, 0.54f, 0.76f);
        s.Colors[::ImGuiCol_Button] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_ButtonHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
        s.Colors[::ImGuiCol_ButtonActive] = ImVec4(0.47f, 0.53f, 0.54f, 0.76f);
        s.Colors[::ImGuiCol_Header] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_HeaderHovered] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
        s.Colors[::ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
        s.Colors[::ImGuiCol_Separator] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_SeparatorHovered] = ImVec4(0.23f, 0.44f, 0.69f, 1.00f);
        s.Colors[::ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        s.Colors[::ImGuiCol_ResizeGrip] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_ResizeGripHovered] = ImVec4(0.23f, 0.44f, 0.69f, 1.00f);
        s.Colors[::ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        s.Colors[::ImGuiCol_Tab] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_TabHovered] = ImVec4(0.6f, 0.6f, 0.6f, 0.58f);
        s.Colors[::ImGuiCol_TabActive] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
        s.Colors[::ImGuiCol_TabUnfocused] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
        s.Colors[::ImGuiCol_TabUnfocusedActive] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        //s.Colors[::ImGuiCol_DockingPreview] = ImVec4(0.26f, 0.59f, 0.98f, 0.70f);
        //s.Colors[::ImGuiCol_DockingEmptyBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
        s.Colors[::ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        s.Colors[::ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        s.Colors[::ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        s.Colors[::ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        s.Colors[::ImGuiCol_TextSelectedBg] = ImVec4(0.97f, 0.97f, 0.97f, 0.19f);
        s.Colors[::ImGuiCol_DragDropTarget] = ImVec4(0.38f, 0.62f, 0.80f, 1.0f);
        s.Colors[::ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        s.Colors[::ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        s.Colors[::ImGuiCol_NavWindowingDimBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        s.Colors[::ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
        // TODO add setting for scaling
        //s.ScaleAllSizes(1.2f);
    }
}

pnanovdb_imgui_window_interface_t* pnanovdb_imgui_get_window_interface()
{
    using namespace pnanovdb_imgui_window_default;
    static pnanovdb_imgui_window_interface_t iface = { PNANOVDB_REFLECT_INTERFACE_INIT(pnanovdb_imgui_window_interface_t) };
    iface.create = create;
    iface.destroy = destroy;
    iface.update = update;
    iface.get_camera = get_camera;
    iface.get_camera_state = get_camera_state;
    iface.update_camera = update_camera;
    return &iface;
}
