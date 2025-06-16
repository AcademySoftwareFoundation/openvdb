// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>  // Use C assert instead of C++ cassert
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <nanovdb_editor/putil/Editor.h>

//#define TEST_EMPTY_COMPILER

typedef struct {
    int magic_number;
} constants_t;

void pnanovdb_compute_log_print(pnanovdb_compute_log_level_t level, const char* format, ...) {
    va_list args;
    va_start(args, format);

    const char* prefix = "Unknown";
    switch (level) {
        case PNANOVDB_COMPUTE_LOG_LEVEL_ERROR:
            prefix = "Error";
            break;
        case PNANOVDB_COMPUTE_LOG_LEVEL_WARNING:
            prefix = "Warning";
            break;
        case PNANOVDB_COMPUTE_LOG_LEVEL_INFO:
            prefix = "Info";
            break;
    }

    printf("%s: ", prefix);
    vprintf(format, args);
    printf("\n");

    va_end(args);
}

int main() {
    const char* shader_name = "../test/shaders/test.slang";

    printf("Running test with shader: %s\n", shader_name);

    // Resolve shader path
    char shader_path[1024];
    const char* source_path = __FILE__;
    const char* last_slash = strrchr(source_path, '/');  // for Unix-style paths
    if (!last_slash) last_slash = strrchr(source_path, '\\');  // for Windows-style paths
    if (last_slash) {
        size_t dir_length = last_slash - source_path + 1;
        strncpy(shader_path, source_path, dir_length);
        shader_path[dir_length] = '\0';
        strcat(shader_path, shader_name);
    }

    // Initialize all handles to NULL/zero
    pnanovdb_compiler_t compiler = {0};
    pnanovdb_compute_t compute = {0};
    pnanovdb_editor_t editor = {0};
    pnanovdb_compute_device_manager_t* device_manager = NULL;
    pnanovdb_compute_device_t* device = NULL;
    pnanovdb_compute_array_t* data_nanovdb = NULL;
    pnanovdb_compute_array_t* data_in = NULL;
    pnanovdb_compute_array_t* constants = NULL;
    pnanovdb_compute_array_t* data_out = NULL;
    int result = 0;

#ifdef TEST_COMPILER
    printf("Testing compiler...\n");

    // Initialize compiler
    pnanovdb_compiler_load(&compiler);

    pnanovdb_compiler_instance_t* compiler_inst = compiler.create_instance();
    if (!compiler_inst) {
        printf("Failed to create compiler instance\n");
        result = 1;
        goto cleanup;
    }

    pnanovdb_compiler_settings_t compile_settings = {0};
    pnanovdb_compiler_settings_init(&compile_settings);

    pnanovdb_bool_t shader_updated = PNANOVDB_FALSE;
    pnanovdb_bool_t compile_result = compiler.compile_shader_from_file(compiler_inst, shader_path, &compile_settings, &shader_updated);
    if (!compile_result) {
        printf("Failed to compile shader\n");
        result = 1;
        goto cleanup;
    }
#endif

#ifdef TEST_COMPUTE
    printf("Testing compute...\n");

#ifdef TEST_EMPTY_COMPILER
    pnanovdb_compiler_t empty_compiler = {0};
    // By setting compiler to null, SPIR-V must be cached or dispatch will fail
    pnanovdb_compute_load(&compute, &empty_compiler);
#else
    // Initialize compute
    pnanovdb_compute_load(&compute, &compiler);
#endif

    // Setup device
    pnanovdb_compute_device_desc_t device_desc = {0};
    device_desc.log_print = pnanovdb_compute_log_print;

    device_manager = compute.device_interface.create_device_manager(PNANOVDB_FALSE);
    if (!device_manager) {
        printf("Failed to create device manager\n");
        result = 1;
        goto cleanup;
    }

    device = compute.device_interface.create_device(device_manager, &device_desc);
    if (!device) {
        printf("Failed to create device\n");
        result = 1;
        goto cleanup;
    }

    // Create test data
    constants_t params = { .magic_number = 4 };
    int test_data[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    size_t test_data_size = sizeof(test_data) / sizeof(test_data[0]);

    // Create compute arrays
    data_in = compute.create_array(sizeof(int), test_data_size, test_data);
    constants = compute.create_array(sizeof(constants_t), 1, &params);
    data_out = compute.create_array(sizeof(int), test_data_size, NULL);

    if (!data_in || !constants || !data_out) {
        printf("Failed to create compute arrays\n");
        result = 1;
        goto cleanup;
    }

    // Dispatch compute shader
    if (compute.dispatch_shader_on_array(
            &compute,
            device,
            shader_path,
            8, 1, 1,
            data_in,
            constants,
            data_out,
            1u,
            0llu, 0llu) == 1) {
        printf("Failed to dispatch shader\n");
        result = 1;
        goto cleanup;
    }

    // Verify results
    int* mapped = (int*)compute.map_array(data_out);
    if (mapped) {
        int test_passed = 1;
        for (size_t i = 0; i < test_data_size; ++i) {
            if (mapped[i] != test_data[i] + params.magic_number) {
                printf("Error: Shader test failed at index %zu!\n", i);
                test_passed = 0;
                break;
            }
        }
        if (test_passed) {
            printf("Shader test was successful\n");
        }
        compute.unmap_array(data_out);
    }
#endif

#ifdef TEST_EDITOR
    printf("Testing editor...\n");

    // Initialize editor
    pnanovdb_editor_load(&editor, &compute, &compiler);

    editor.add_array(&editor, data_out);

    editor.show(&editor, device);

    // Already released by editor
    data_out = NULL;
#endif

cleanup:
#ifdef TEST_EDITOR
    pnanovdb_editor_free(&editor);
#endif

#ifdef TEST_COMPUTE
    if (data_in) compute.destroy_array(data_in);
    if (constants) compute.destroy_array(constants);
    if (data_out) compute.destroy_array(data_out);
    if (device) compute.device_interface.destroy_device(device_manager, device);
    if (device_manager) compute.device_interface.destroy_device_manager(device_manager);
    pnanovdb_compute_free(&compute);
#endif

#ifdef TEST_COMPILER
    if (compiler_inst) compiler.destroy_instance(compiler_inst);
    pnanovdb_compiler_free(&compiler);
#endif

    return result;
}
