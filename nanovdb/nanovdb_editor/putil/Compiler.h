// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/Compiler.h

    \author Andrew Reidmeyer

    \brief  This file provides a compiler abstraction.
*/

#ifndef NANOVDB_PUTILS_COMPILER_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_COMPILER_H_HAS_BEEN_INCLUDED

#include "nanovdb_editor/putil/Loader.h"
#include "nanovdb_editor/putil/Reflect.h"
#include <stdio.h>

/// ********************************* Compiler ***************************************

struct pnanovdb_compiler_instance_t;
typedef struct pnanovdb_compiler_instance_t pnanovdb_compiler_instance_t;

typedef pnanovdb_uint32_t pnanovdb_compile_target_type_t;

#define PNANOVDB_COMPILE_TARGET_UNKNOWN 0
#define PNANOVDB_COMPILE_TARGET_VULKAN 1
#define PNANOVDB_COMPILE_TARGET_CPU 2
//TODO support CUDA

struct pnanovdb_compiler_settings_t;
typedef struct pnanovdb_compiler_settings_t
{
    pnanovdb_bool_t is_row_major;
    pnanovdb_bool_t use_glslang;
    pnanovdb_bool_t hlsl_output;
    pnanovdb_compile_target_type_t compile_target;
    char entry_point_name[64];
}pnanovdb_compiler_settings_t;

typedef pnanovdb_uint32_t pnanovdb_compiler_api_t;

typedef void (*pnanovdb_compiler_diagnostic_callback)(const char* message);

typedef struct pnanovdb_compiler_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_compiler_instance_t*(PNANOVDB_ABI* create_instance)();

    void(PNANOVDB_ABI* set_diagnostic_callback)(
        pnanovdb_compiler_instance_t* instance,
        pnanovdb_compiler_diagnostic_callback callback);

    pnanovdb_bool_t(PNANOVDB_ABI* compile_shader_from_file)(
        pnanovdb_compiler_instance_t* instance,
        const char* filename,
        pnanovdb_compiler_settings_t* settings,
        pnanovdb_bool_t* shader_updated);

    pnanovdb_bool_t(PNANOVDB_ABI* execute_cpu)(
        pnanovdb_compiler_instance_t* instance,
        const char* filename,
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ,
        void* uniformParams,
        void* uniformState);

    void(PNANOVDB_ABI* destroy_instance)(pnanovdb_compiler_instance_t* instance);

    void* module;

}pnanovdb_compiler_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_compiler_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(create_instance, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(set_diagnostic_callback, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(compile_shader_from_file, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(execute_cpu, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_instance, 0, 0)
PNANOVDB_REFLECT_VOID_POINTER(module, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_compiler_t* (PNANOVDB_ABI* PFN_pnanovdb_get_compiler)();

PNANOVDB_API pnanovdb_compiler_t* pnanovdb_get_compiler();

static void pnanovdb_compiler_load(pnanovdb_compiler_t* compiler)
{
    void* compiler_module = pnanovdb_load_library("pnanovdbcompiler.dll", "libpnanovdbcompiler.so", "libpnanovdbcompiler.dylib");
    if (!compiler_module)
    {
#if defined(_WIN32)
        printf("Error: Compiler module failed to load\n");
#else
        printf("Error: Compiler module failed to load: %s\n", dlerror());
#endif
        return;
    }
    PFN_pnanovdb_get_compiler get_compiler = (PFN_pnanovdb_get_compiler)pnanovdb_get_proc_address(compiler_module, "pnanovdb_get_compiler");
    if (!get_compiler)
    {
        printf("Error: Failed to acquire compiler getter\n");
        return;
    }
    pnanovdb_compiler_t_duplicate(compiler, get_compiler());
    if (!compiler)
    {
        printf("Error: Failed to acquire compiler\n");
        return;
    }

    compiler->module = compiler_module;
}

static void pnanovdb_compiler_free(pnanovdb_compiler_t* compiler)
{
    pnanovdb_free_library(compiler->module);
}

static void pnanovdb_compiler_settings_init(pnanovdb_compiler_settings_t* settings)
{
    settings->is_row_major = PNANOVDB_FALSE;
    settings->use_glslang = PNANOVDB_FALSE;
    settings->hlsl_output = PNANOVDB_FALSE;
    settings->compile_target = PNANOVDB_COMPILE_TARGET_UNKNOWN;
    settings->entry_point_name[0] = '\0';
}

// ------------------------------------------------ Interface Implementation Helpers -----------------------------------------------------------

#define PNANOVDB_CAST_PAIR(X, Y) \
    PNANOVDB_INLINE X* cast(Y* ptr) { return (X*)ptr; } \
    PNANOVDB_INLINE Y* cast(X* ptr) { return (Y*)ptr; } \
    PNANOVDB_INLINE const X* cast(const Y* ptr) { return (X*)ptr; } \
    PNANOVDB_INLINE const Y* cast(const X* ptr) { return (Y*)ptr; }

#define PNANOVDB_CAST_PAIR_NAMED(name, X, Y) \
    PNANOVDB_INLINE X* name##_cast(Y* ptr) { return (X*)ptr; } \
    PNANOVDB_INLINE Y* name##_cast(X* ptr) { return (Y*)ptr; } \
    PNANOVDB_INLINE const X* name##_cast(const Y* ptr) { return (X*)ptr; } \
    PNANOVDB_INLINE const Y* name##_cast(const X* ptr) { return (Y*)ptr; }

#endif
