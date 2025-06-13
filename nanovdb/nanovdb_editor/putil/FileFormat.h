// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb_editor/putil/FileFormat.h

    \author Petra Hapalova

    \brief
*/

#ifndef NANOVDB_PUTILS_FILEFORMAT_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_FILEFORMAT_H_HAS_BEEN_INCLUDED

#include "nanovdb_editor/putil/Compute.h"

/// ********************************* FileFormat ***************************************

typedef struct pnanovdb_fileformat_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_bool_t (PNANOVDB_ABI* can_load_file)(const char* filename);

    pnanovdb_bool_t(PNANOVDB_ABI* load_file)(const char* filename, pnanovdb_uint32_t array_count, const char** array_names, pnanovdb_compute_array_t** out_arrays);

    void(PNANOVDB_ABI* e57_to_float)(const char* filename, size_t* array_size, float** positions_array, float** colors_array, float** normals_array);

    void* loaded_library;
} pnanovdb_fileformat_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_fileformat_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(can_load_file, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(load_file, 0, 0)
PNANOVDB_REFLECT_VOID_POINTER(loaded_library, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_fileformat_t* (PNANOVDB_ABI* PFN_pnanovdb_get_fileformat)();

PNANOVDB_API pnanovdb_fileformat_t* pnanovdb_get_fileformat();

static void pnanovdb_fileformat_load(pnanovdb_fileformat_t* fileformat, const pnanovdb_compute_t* compute)
{
    void* module = pnanovdb_load_library("pnanovdbfileformat.dll", "libpnanovdbfileformat.so", "libpnanovdbfileformat.dylib");
    auto get_fileformat = (PFN_pnanovdb_get_fileformat)pnanovdb_get_proc_address(module, "pnanovdb_get_fileformat");
    if (!get_fileformat)
    {
        printf("Error: Failed to acquire fileformat\n");
        return;
    }
    *fileformat = *get_fileformat();

    fileformat->loaded_library = module;
}

static void pnanovdb_fileformat_free(pnanovdb_fileformat_t* fileformat)
{
    pnanovdb_free_library(fileformat->loaded_library);
}

#endif // NANOVDB_PUTILS_FILEFORMAT_H_HAS_BEEN_INCLUDED
