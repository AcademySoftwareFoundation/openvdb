
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   ImguiRenderer.h

    \author Andrew Reidmeyer

    \brief  This file is part of the PNanoVDB Compute Vulkan reference implementation.
*/

#pragma once

#include "nanovdb_editor/putil/Compute.h"

struct pnanovdb_imgui_renderer_t;
typedef struct pnanovdb_imgui_renderer_t pnanovdb_imgui_renderer_t;

struct pnanovdb_imgui_texture_t;
typedef struct pnanovdb_imgui_texture_t pnanovdb_imgui_texture_t;

struct ImDrawData;
typedef struct ImDrawData ImDrawData;

typedef struct pnanovdb_imgui_renderer_interface_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_imgui_renderer_t*(PNANOVDB_ABI* create)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        unsigned char* pixels,
        int texWidth,
        int texHeight
    );

    void(PNANOVDB_ABI* destroy)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_imgui_renderer_t* renderer);

    void(PNANOVDB_ABI* render)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        ImDrawData* drawData,
        pnanovdb_uint32_t width,
        pnanovdb_uint32_t height,
        pnanovdb_compute_texture_transient_t* colorIn,
        pnanovdb_compute_texture_transient_t* colorOut
    );

    void(PNANOVDB_ABI* copy_texture)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        pnanovdb_uint32_t width,
        pnanovdb_uint32_t height,
        pnanovdb_compute_texture_transient_t* colorIn,
        pnanovdb_compute_texture_transient_t* colorOut
    );

    pnanovdb_imgui_texture_t*(PNANOVDB_ABI* create_texture)(
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        unsigned char* pixels,
        int texWidth,
        int texHeight
    );

    void(PNANOVDB_ABI* update_texture)(
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        pnanovdb_imgui_texture_t* texture,
        unsigned char* pixels,
        int texWidth,
        int texHeight
    );

    void(PNANOVDB_ABI* destroy_texture)(
        pnanovdb_compute_context_t* context,
        pnanovdb_imgui_renderer_t* renderer,
        pnanovdb_imgui_texture_t* texture
    );
}pnanovdb_imgui_renderer_interface_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_imgui_renderer_interface_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(create, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(render, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(copy_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(create_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(update_texture, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_texture, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_imgui_renderer_interface_t* (PNANOVDB_ABI* PFN_pnanovdb_imgui_get_renderer_interface)();

PNANOVDB_API pnanovdb_imgui_renderer_interface_t* pnanovdb_imgui_get_renderer_interface();
