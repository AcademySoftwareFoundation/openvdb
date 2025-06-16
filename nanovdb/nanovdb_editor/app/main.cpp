// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/app/main.cpp

    \author Petra Hapalova, Andrew Reidmeyer

    \brief
*/

#include "Log.h"
#include "Node2Convert.h"

#include <nanovdb_editor/putil/Editor.h>

#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>

#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#define CONVERT_NODE2 1

int main(int argc, char *argv[])
{
#if CONVERT_NODE2
    #ifndef _DEBUG
        pnanovdb_editor::node2_convert("./data/dragon.nvdb", "./data/dragon_node2.nvdb");
        //pnanovdb_editor::node2_sphere("./data/sphere.nvdb");
    #endif
#endif

    pnanovdb_compiler_t compiler = {};
    pnanovdb_compiler_load(&compiler);

    pnanovdb_compute_t compute = {};
    pnanovdb_compute_load(&compute, &compiler);

    pnanovdb_compute_device_desc_t device_desc = {};
    device_desc.log_print = pnanovdb_compute_log_print;

    pnanovdb_compute_device_manager_t* device_manager = compute.device_interface.create_device_manager(PNANOVDB_FALSE);
    pnanovdb_compute_device_t* device = compute.device_interface.create_device(device_manager, &device_desc);

#if TEST_NODE2
    const char* file = "./data/dragon_node2.nvdb";
    //file = "./data/sphere.nvdb";
#else
    const char* file = "./data/dragon.nvdb";
#endif

    pnanovdb_editor_t editor = {};
    pnanovdb_editor_load(&editor, &compute, &compiler);

    editor.compute = &compute;

    pnanovdb_compute_array_t* data_in = compute.load_nanovdb(file);
    editor.add_nanovdb(&editor, data_in);
    editor.show(&editor, device);

    pnanovdb_editor_free(&editor);

    compute.device_interface.destroy_device(device_manager, device);
    compute.device_interface.destroy_device_manager(device_manager);

    pnanovdb_compute_free(&compute);
    pnanovdb_compiler_free(&compiler);

    return 0;
}
