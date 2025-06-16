// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/test/main.cpp

    \author Petra Hapalova, Andrew Reidmeyer

    \brief
*/

#include <nanovdb_editor/putil/Editor.h>
#include "nanovdb_editor/putil/FileFormat.h"
#include <nanovdb_editor/putil/Raster.hpp>

#include <nanovdb/io/IO.h>

#include <cstdarg>

#include <slang.h>

#define SLANG_PRELUDE_NAMESPACE CPPPrelude
#include <slang-cpp-types.h>

#define TEST_COMPILER
#define TEST_CPU_COMPILER
//#define TEST_EMPTY_COMPILER
#define TEST_COMPUTE
#define TEST_EDITOR
//#define TEST_RASTER
//#define TEST_SVRASTER
//#define TEST_E57
//#define TEST_FILE_FORMAT
//#define FORMAT_INGP
#define FORMAT_PLY

struct constants_t
{
    int magic_number;
    int pad1;
    int pad2;
    int pad3;
};

void pnanovdb_compute_log_print(pnanovdb_compute_log_level_t level, const char* format, ...)
{
    va_list args;
    va_start(args, format);

    const char* prefix = "Unknown";
    if (level == PNANOVDB_COMPUTE_LOG_LEVEL_ERROR)
    {
        prefix = "Error";
    }
    else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_WARNING)
    {
        prefix = "Warning";
    }
    else if (level == PNANOVDB_COMPUTE_LOG_LEVEL_INFO)
    {
        prefix = "Info";
    }
    printf("%s: ", prefix);
    vprintf(format, args);
    printf("\n");

    va_end(args);
}

int main()
{
    const char* shader_path = "test/test.slang";
#if TEST_NODE2
    const char* nvdb_filepath = "./data/dragon_node2.nvdb";
#else
    const char* nvdb_filepath = "./data/dragon.nvdb";
#endif

    constants_t params = { 4 };
    std::vector<int> test_data = { 0, 1, 2, 3, 4, 5, 6, 7 };

#ifdef TEST_COMPILER
    // uses dlopen to load compiler and get symbols
    pnanovdb_compiler_t compiler = {};
    pnanovdb_compiler_load(&compiler);

    pnanovdb_compiler_instance_t* compiler_inst = compiler.create_instance();

    pnanovdb_compiler_settings_t compile_settings = {};
    pnanovdb_compiler_settings_init(&compile_settings);
    compile_settings.compile_target = PNANOVDB_COMPILE_TARGET_VULKAN;
    strcpy(compile_settings.entry_point_name, "computeMain");

    pnanovdb_bool_t result = compiler.compile_shader_from_file(compiler_inst, shader_path, &compile_settings, nullptr);
    if (result == PNANOVDB_FALSE)
    {
        printf("Error: Compilation of shader '%s' failed\n", shader_path);
    }
#endif

#ifdef TEST_COMPUTE
    // uses dlopen to load compute and get symbols
    pnanovdb_compute_t compute = {};

#ifdef TEST_EMPTY_COMPILER
    // by setting compiler to null, SPIR-V must be cached or dispatch will fail
    pnanovdb_compiler_t* compiler_optional = nullptr;

    pnanovdb_compute_load(&compute, &compiler_optional);
#else
    pnanovdb_compute_load(&compute, &compiler);
#endif

    pnanovdb_compute_device_desc_t device_desc = {};
    device_desc.log_print = pnanovdb_compute_log_print;

    pnanovdb_compute_device_manager_t* device_manager = compute.device_interface.create_device_manager(PNANOVDB_FALSE);
    pnanovdb_compute_device_t* device = compute.device_interface.create_device(device_manager, &device_desc);

    pnanovdb_compute_array_t* data_nanovdb = compute.load_nanovdb(nvdb_filepath);
    if (!data_nanovdb)
    {
        printf("Error: Could not laod file '%s'\n", nvdb_filepath);
        return 1;
    }

    pnanovdb_uint64_t data_out_size = test_data.size();

    pnanovdb_compute_array_t* data_in = compute.create_array(sizeof(int), test_data.size(), test_data.data());
    pnanovdb_compute_array_t* constants = compute.create_array(sizeof(constants_t), 1u, &params);
    pnanovdb_compute_array_t* data_out = compute.create_array(sizeof(int), data_out_size, nullptr);

    const int dispatch_result = compute.dispatch_shader_on_array(
        &compute,
        device,
        shader_path,
        8u, 1u, 1u,
        data_in,
        constants,
        data_out,
        1u,
        0llu, 0llu);

    if (dispatch_result == 0)
    {
        int* mapped = (int*)compute.map_array(data_out);
        bool success = true;

        for (int i = 0; i < data_out_size; ++i)
        {
            if (mapped[i] != test_data[i] + params.magic_number)
            {
                printf("Error: Shader test failed!\n");
                success = false;
                break;
            }
        }
        if (success)
        {
            printf("Shader test passed!\n");
        }
    }
    else
    {
        printf("Error: Shader test dispatch failed!\n");
    }

    compute.unmap_array(data_out);

    compute.destroy_array(data_in);
    compute.destroy_array(constants);
#endif

#ifdef TEST_CPU_COMPILER
    pnanovdb_compiler_t cpu_compiler = {};
    pnanovdb_compiler_load(&cpu_compiler);

    pnanovdb_compiler_instance_t* cpu_compiler_inst = cpu_compiler.create_instance();

    pnanovdb_compiler_settings_t cpu_compile_settings = {};
    pnanovdb_compiler_settings_init(&cpu_compile_settings);
    cpu_compile_settings.compile_target = PNANOVDB_COMPILE_TARGET_CPU;
    strcpy(cpu_compile_settings.entry_point_name, "computeMain");

    pnanovdb_bool_t result_cpu = cpu_compiler.compile_shader_from_file(cpu_compiler_inst, shader_path, &cpu_compile_settings, nullptr);
    if (result_cpu == PNANOVDB_FALSE)
    {
        printf("Error: Compilation of shader '%s' failed\n", shader_path);
    }
    else
    {
        std::vector<int> test_data_out(test_data.size(), 0);
        std::vector<unsigned int> scratch_buffer(test_data.size(), 0);

        struct UniformState
        {
            CPPPrelude::StructuredBuffer<int> data_in;
            constants_t* constants;
            CPPPrelude::RWStructuredBuffer<int> data_out;
            CPPPrelude::RWStructuredBuffer<unsigned int> scratch;
        };

        UniformState uniformState;
        uniformState.data_in = { test_data.data(), test_data.size() };
        uniformState.constants = &params;
        uniformState.data_out = { test_data_out.data(), test_data.size() };
        uniformState.scratch = { scratch_buffer.data(), scratch_buffer.size() };

        result_cpu = cpu_compiler.execute_cpu(cpu_compiler_inst, shader_path, 1u, 1u, 1u, nullptr, (void*)&uniformState);
        if (result_cpu != PNANOVDB_FALSE)
        {
            bool success = true;
            for (int i = 0; i < test_data.size(); ++i)
            {
                if (test_data_out[i] != test_data[i] + params.magic_number)
                {
                    printf("Error: CPU Shader test failed!\n");
                    success = false;
                    break;
                }
            }
            if (success)
            {
                printf("CPU Shader test passed!\n");
            }
        }
        else
        {
            printf("Error: Shader test execute failed!\n");
        }
    }
#endif

#ifdef TEST_RASTER
    const char* npy_file = "./data/splats.npz";
    pnanovdb_raster_t raster = {};
    pnanovdb_raster_load(&raster, &compute);

    pnanovdb_compute_queue_t* queue = compute.device_interface.get_compute_queue(device);

    pnanovdb_raster::raster_file(&raster, &compute, queue, npy_file, &data_nanovdb, nullptr, nullptr, nullptr, nullptr);

    compute.save_nanovdb(data_nanovdb, "./data/splats.nvdb");

    pnanovdb_raster_free(&raster);
#endif

#ifdef TEST_SVRASTER
    const char* npy_file = "./data/svraster.npz";
    pnanovdb_raster_t raster = {};
    pnanovdb_raster_load(&raster, &compute);

    pnanovdb_compute_queue_t* queue = compute.device_interface.get_compute_queue(device);

    pnanovdb_raster::raster_file(&raster, &compute, queue, npy_file, &data_nanovdb, nullptr, nullptr, nullptr, nullptr);

    compute.save_nanovdb(data_nanovdb, "./data/splats.nvdb");

    pnanovdb_raster_free(&raster);
#endif

#ifdef TEST_E57
#define ARRAY_COUNT 3
    size_t runs = 10u;
    size_t totalTime = 0u;
    size_t array_size = 0u;

    const char* e57_file = "./data/kantonalschule/20042020-kantonalschule-_Setip_001.e57";
    const char* array_names[] = {"positions", "colors", "normals"};

    pnanovdb_fileformat_t fileformat = {};
    pnanovdb_fileformat_load(&fileformat, &compute);

    for (int i = 0; i < runs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();

        pnanovdb_compute_array_t* arrays[ARRAY_COUNT] = {};
        pnanovdb_bool_t loaded = fileformat.load_file(e57_file, ARRAY_COUNT, array_names, arrays);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        totalTime += duration.count();
        array_size = arrays[0]->element_count;

        float* positions = (float*)compute.map_array(arrays[0]);
        float* colors = (float*)compute.map_array(arrays[1]);

        for (int i = 0; i < 10; i++)
        {
            printf("[%d] position: %f, %f, %f\t", i, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            printf("color: %f, %f, %f\n", colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]);
        }
        compute.unmap_array(arrays[0]);
        compute.unmap_array(arrays[1]);

        for (int i = 0; i < ARRAY_COUNT; i++)
        {
            compute.destroy_array(arrays[i]);
        }

        //printf("Load time: %zu ms (%zu points)\n", (size_t)duration.count(), array_size / 3);
    }
    printf("E57 average load: %zu ms (%zu points)\n", totalTime / runs, array_size / 3);

    pnanovdb_fileformat_free(&fileformat);
    return 0;
#endif


#ifdef TEST_FILE_FORMAT

#if defined(FORMAT_INGP)
    const char* filename = "./data/ficus-30k.ingp";
    const char* array_names[] = { "mog_positions", "mog_densities", "mog_rotations", "mog_scales", "mog_features" };
#elif defined(FORMAT_PLY)
    const char* filename = "./data/ficus.ply";
    const char* array_names[] = { "means", "opacities", "quaternions", "scales", "sh" };
#else
#error "No valid format defined"
#endif

#define ARRAY_COUNT 5
    pnanovdb_compute_array_t* arrays[ARRAY_COUNT] = {};

    pnanovdb_fileformat_t fileformat = {};
    pnanovdb_fileformat_load(&fileformat, &compute);
    pnanovdb_bool_t loaded = fileformat.load_file(filename, ARRAY_COUNT, array_names, arrays);
    if (loaded == PNANOVDB_TRUE)
    {
        printf("Successfully loaded file with arrays:\n");
        for (int i = 0; i < ARRAY_COUNT; i++)
        {
            if (arrays[i])
            {
                printf("  %s: %llu elements of size %llu\n", array_names[i], arrays[i]->element_count, arrays[i]->element_size);
                compute.destroy_array(arrays[i]);
            }
        }
    }
    else
    {
        printf("Failed to load file '%s'\n", filename);
    }

    pnanovdb_fileformat_free(&fileformat);
    return 0;
#endif

#ifdef TEST_EDITOR
    // add editing optionally and late
    pnanovdb_editor_t editor = {};
    pnanovdb_editor_load(&editor, &compute, &compiler);

    editor.add_nanovdb(&editor, data_nanovdb);
    editor.add_array(&editor, data_out);
    editor.show(&editor, device);

    pnanovdb_editor_free(&editor);
#endif

#ifdef TEST_COMPUTE
    compute.device_interface.destroy_device(device_manager, device);
    compute.device_interface.destroy_device_manager(device_manager);

    pnanovdb_compute_free(&compute);
#endif

#ifdef TEST_COMPILER
    compiler.destroy_instance(compiler_inst);

    pnanovdb_compiler_free(&compiler);
#endif

    return 0;
}
