// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Editor.h

    \author Andrew Reidmeyer

    \brief  This file provides a GPU compute abstraction.
*/

#ifndef NANOVDB_PNANOVDBEDITOR_H_HAS_BEEN_INCLUDED
#define NANOVDB_PNANOVDBEDITOR_H_HAS_BEEN_INCLUDED

#include "nanovdb_editor/putil/Compute.h"
#include "nanovdb_editor/putil/Raster.h"

// ------------------------------------------------ Editor -----------------------------------------------------------

struct pnanovdb_editor_t;
typedef struct pnanovdb_editor_t pnanovdb_editor_t;

typedef void(*pnanovdb_editor_callable_t)(const char* filename, pnanovdb_profiler_report_t profiler_report);

typedef struct pnanovdb_editor_t
{
    PNANOVDB_REFLECT_INTERFACE();

    const pnanovdb_compiler_t* compiler;
    const pnanovdb_compute_t* compute;

    void(PNANOVDB_ABI* init)(pnanovdb_editor_t* editor);

    void(PNANOVDB_ABI* shutdown)(pnanovdb_editor_t* editor);

    void(PNANOVDB_ABI* add_nanovdb)(pnanovdb_editor_t* editor, pnanovdb_compute_array_t* array);

    void(PNANOVDB_ABI* add_array)(pnanovdb_editor_t* editor, pnanovdb_compute_array_t* array);

    void(PNANOVDB_ABI* add_callable)(pnanovdb_editor_t* editor, const char* name, pnanovdb_editor_callable_t function);

    void(PNANOVDB_ABI* show)(pnanovdb_editor_t* editor, pnanovdb_compute_device_t* device);

    void* module;
    pnanovdb_raster_gaussian_data_t* gaussian_data;
    pnanovdb_compute_array_t* nanovdb_array;
    pnanovdb_compute_array_t* data_array;
    pnanovdb_editor_callable_t callable_func;
    char callable_name[64];

}pnanovdb_editor_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_editor_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_POINTER(pnanovdb_compiler_t, compiler, 0, 0)
PNANOVDB_REFLECT_POINTER(pnanovdb_compute_t, compute, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(init, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(shutdown, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(add_nanovdb, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(add_array, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(add_callable, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(show, 0, 0)
PNANOVDB_REFLECT_VOID_POINTER(module, 0, 0)
PNANOVDB_REFLECT_POINTER(pnanovdb_compute_array_t, nanovdb_array, 0, 0)
PNANOVDB_REFLECT_POINTER(pnanovdb_compute_array_t, data_array, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_editor_t* (PNANOVDB_ABI* PFN_pnanovdb_get_editor)();

PNANOVDB_API pnanovdb_editor_t* pnanovdb_get_editor();

static void pnanovdb_editor_load(pnanovdb_editor_t* editor, const pnanovdb_compute_t* compute, const pnanovdb_compiler_t* compiler)
{
    void* editor_module = pnanovdb_load_library("pnanovdbeditor.dll", "libpnanovdbeditor.so", "libpnanovdbeditor.dylib");
    if (!editor_module)
    {
#if defined(_WIN32)
        printf("Error: Editor module failed to load\n");
#else
        printf("Error: Editor module failed to load: %s\n", dlerror());
#endif
        return;
    }
    PFN_pnanovdb_get_editor get_editor = (PFN_pnanovdb_get_editor)pnanovdb_get_proc_address(editor_module, "pnanovdb_get_editor");
    if (!get_editor)
    {
        printf("Error: Failed to acquire editor getter\n");
        return;
    }
    pnanovdb_editor_t_duplicate(editor, get_editor());
    if (!editor)
    {
        printf("Error: Failed to acquire editor\n");
        return;
    }

    editor->module = editor_module;
    editor->compute = compute;
    editor->compiler = compiler;
    editor->nanovdb_array = NULL;
    editor->data_array = NULL;
    editor->callable_func = NULL;
    editor->callable_name[0] = '\0';
    editor->init(editor);
}

static void pnanovdb_editor_free(pnanovdb_editor_t* editor)
{
    if (!editor)
    {
        return;
    }
    editor->shutdown(editor);

    pnanovdb_free_library(editor->module);
}

#endif
