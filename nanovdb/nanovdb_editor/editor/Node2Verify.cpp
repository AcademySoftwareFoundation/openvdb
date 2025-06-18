// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/Node2Convert.cpp

    \author Andrew Reidmeyer

    \brief
*/

#include "Node2Verify.h"

#include <nanovdb_editor/putil/Editor.h>
#include <nanovdb/io/IO.h>

#include "Console.h"

#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#include "raster/Node2Cpp.h"

namespace pnanovdb_editor
{
static void node2_verify_gpu_log_print(pnanovdb_compute_log_level_t level, const char* format, ...)
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

struct shader_params_t
{
    pnanovdb_coord_t bbox_min;
    pnanovdb_uint32_t grid_dim_x;
    pnanovdb_coord_t bbox_max;
    pnanovdb_uint32_t grid_dim_y;
};

static void node2_profiler_report(void* userdata, pnanovdb_uint64_t capture_id, pnanovdb_uint32_t num_entries, pnanovdb_compute_profiler_entry_t* entries)
{
    printf("run_test_shader() profiler results capture_id(%llu):\n", (unsigned long long int)capture_id);
    for (pnanovdb_uint32_t idx = 0u; idx < num_entries; idx++)
    {
        printf("[%d] name(%s) cpu_ms(%f) gpu_ms(%f)\n",
            idx, entries[idx].label, 1000.f * entries[idx].cpu_delta_time, 1000.f * entries[idx].gpu_delta_time);
    }
}

int node2_verify_gpu(const char* nvdb_ref_filepath, const char* nvdb_filepath)
{
    pnanovdb_compiler_t compiler = {};
    pnanovdb_compiler_load(&compiler);

    pnanovdb_compute_t compute = {};
    pnanovdb_compute_load(&compute, &compiler);

    pnanovdb_compute_device_desc_t device_desc = {};
    device_desc.log_print = node2_verify_gpu_log_print;

    pnanovdb_compute_device_manager_t* device_manager = compute.device_interface.create_device_manager(PNANOVDB_FALSE);
    //pnanovdb_compute_device_t* device = compute.device_interface.create_device(device_manager, &device_desc);

    pnanovdb_compute_array_t* data_in = compute.load_nanovdb(nvdb_filepath);
    pnanovdb_compute_array_t* data_in_ref = compute.load_nanovdb(nvdb_ref_filepath);
    pnanovdb_compute_array_t* data_in_onindex = compute.load_nanovdb("./data/ls_dragon_onindex.nvdb");

    printf("node2 grid_size(%lld)\n", 4u * (unsigned long long int)data_in->element_count);
    printf("float grid_size(%lld)\n", 4u * (unsigned long long int)data_in_ref->element_count);
    printf("onindex grid_size(%lld)\n", 4u * (unsigned long long int)data_in_onindex->element_count);

    pnanovdb_buf_t buf = pnanovdb_make_buf((pnanovdb_uint32_t*)data_in->data, data_in->element_count);
    pnanovdb_grid_handle_t grid = {};
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_node2_handle_t root = {pnanovdb_uint32_t(pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u)};
    pnanovdb_node2_levelset_values_t values = {};
    values.values = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 1u);
    values.node_inactive_idxs = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 2u);
    values.inactive_value_idxs = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 3u);

    pnanovdb_address_t bboxes = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 0u);
    pnanovdb_coord_t bbox_min = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 0u));
    pnanovdb_coord_t bbox_max = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 12u));

    shader_params_t params = {};
    params.bbox_min = bbox_min;
    params.bbox_max = bbox_max;
    params.grid_dim_x = (bbox_max.y - bbox_min.y + 15u) / 16u;
    params.grid_dim_y = (bbox_max.z - bbox_min.z + 15u) / 16u;

    static const pnanovdb_uint32_t element_words = 1u;
    pnanovdb_uint64_t data_out_element_count = (bbox_max.y - bbox_min.y) * (bbox_max.z - bbox_min.z) * element_words;

    pnanovdb_compute_array_t* constants = compute.create_array(sizeof(shader_params_t), 1u, &params);
    pnanovdb_compute_array_t* data_out = compute.create_array(4u, data_out_element_count, nullptr);
    pnanovdb_compute_array_t* data_out_ref = compute.create_array(4u, data_out_element_count, nullptr);
    pnanovdb_compute_array_t* data_out_onindex = compute.create_array(4u, data_out_element_count, nullptr);

    pnanovdb_compute_device_t* device = compute.device_interface.create_device(device_manager, &device_desc);

    printf("Benchmark for Node2 NanoVDB:\n");
    compute.dispatch_shader_on_array(
        &compute,
        device,
        "editor/node2_verify_gpu.slang",
        params.grid_dim_x, params.grid_dim_y, 1u,
        data_in,
        constants,
        data_out,
        8u,
        0llu, 0llu);

    compute.device_interface.destroy_device(device_manager, device);
    device = compute.device_interface.create_device(device_manager, &device_desc);

    printf("Benchmark for Float NanoVDB:\n");
    compute.dispatch_shader_on_array(
        &compute,
        device,
        "editor/node2_verify_gpu_ref.slang",
        params.grid_dim_x, params.grid_dim_y, 1u,
        data_in_ref,
        constants,
        data_out_ref,
        8u,
        0llu, 0llu);

    compute.device_interface.destroy_device(device_manager, device);
    device = compute.device_interface.create_device(device_manager, &device_desc);

    printf("Benchmark for OnIndex NanoVDB:\n");
    compute.dispatch_shader_on_array(
        &compute,
        device,
        "editor/node2_verify_gpu_onindex.slang",
        params.grid_dim_x, params.grid_dim_y, 1u,
        data_in_onindex,
        constants,
        data_out_onindex,
        8u,
        0llu, 0llu);

    compute.device_interface.destroy_device(device_manager, device);

    pnanovdb_uint32_t* mapped_data_out = (pnanovdb_uint32_t*)compute.map_array(data_out);
    pnanovdb_uint32_t* mapped_data_out_ref = (pnanovdb_uint32_t*)compute.map_array(data_out_ref);
    pnanovdb_uint32_t* mapped_data_out_onindex = (pnanovdb_uint32_t*)compute.map_array(data_out_onindex);

    uint64_t total_count = 0;
    uint64_t mismatched_count = 0;
    uint64_t mismatched_count_onindex = 0;
    int print_count = 0;
    //for (pnanovdb_int32_t i = bbox_min.x; i < bbox_max.x; i++)
    pnanovdb_int32_t i = (bbox_max.x + bbox_min.x) / 2;
    {
        for (pnanovdb_int32_t j = bbox_min.y; j < bbox_max.y; j++)
        {
            for (pnanovdb_int32_t k = bbox_min.z; k < bbox_max.z; k++)
            {
                pnanovdb_coord_t ijk = { i, j, k };

                float val_cpp = 0.f;
                float val_cpu = 0.f;
                for (int i = 0; i < PNANOVDB_NODE2_TEST_X_DEPTH; i++)
                {
                    pnanovdb_node2_handle_t node = {};
                    pnanovdb_uint32_t node_type = 0u;
                    pnanovdb_uint32_t node_n = 0u;
                    pnanovdb_uint32_t level = 0;
                    pnanovdb_node2_find_node(buf, root, &node, &node_type, &node_n, &level, ijk);
                    pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, node, node_type, node_n, PNANOVDB_TRUE, values.node_inactive_idxs, values.inactive_value_idxs);
                    pnanovdb_address_t val_addr = pnanovdb_address_offset64_product(values.values, value_idx, 4u);

                    uint64_t* buf_64 = (uint64_t*)buf.data;
                    uint32_t root_idx64 = root.idx64;
                    uint32_t cpp_node = 0u;
                    uint32_t cpp_node_type = 0u;
                    uint32_t cpp_node_n = 0u;
                    uint32_t cpp_local_value_idx = 0u;
                    uint64_t cpp_val_idx = nanovdb::node2_get_value_index(buf_64, root_idx64, nanovdb::Coord(ijk.x, ijk.y, ijk.z),
                        true, uint32_t(values.node_inactive_idxs.byte_offset >> 3u), uint32_t(values.inactive_value_idxs.byte_offset >> 3u));

                    uint32_t values_idx64 = uint32_t(values.values.byte_offset >> 3u);
                    pnanovdb_address_t cpp_val_addr = pnanovdb_address_offset64_product(values.values, cpp_val_idx, 4u);

                    val_cpp = fmaxf(val_cpp, pnanovdb_read_float(buf, cpp_val_addr));

                    val_cpu = fmaxf(val_cpu, pnanovdb_read_float(buf, val_addr));
                    ijk.x++;
                }

                pnanovdb_uint64_t idx = (ijk.y - bbox_min.y) * (bbox_max.z - bbox_min.z) + (ijk.z - bbox_min.z);
                float val_gpu = pnanovdb_uint32_as_float(mapped_data_out[element_words * idx + 0]);
                float val_gpu_ref = pnanovdb_uint32_as_float(mapped_data_out_ref[element_words * idx + 0]);
                float val_gpu_onindex = pnanovdb_uint32_as_float(mapped_data_out_onindex[element_words * idx + 0]);

                if (val_cpu != val_gpu ||
                    val_gpu != val_gpu_ref ||
                    val_cpp != val_gpu)
                {
                    mismatched_count++;
                    if (print_count < 32u)
                    {
                        pnanovdb_node2_handle_t node = {};
                        pnanovdb_uint32_t node_type = 0u;
                        pnanovdb_uint32_t node_n = 0u;
                        pnanovdb_uint32_t level = 0u;
                        pnanovdb_node2_find_node(buf, root, &node, &node_type, &node_n, &level, ijk);

                        print_count++;
                        pnanovdb_editor::Console::getInstance().addLog("val_cpu(%f) val_gpu(%f) val_gpu_ref(%f) ijk(%d,%d,%d) node_type(0x%x) dense_n(%d)",
                            val_cpu, val_gpu, val_gpu_ref, i, j, k, node_type, node_n);
                        printf("element[%llu] val_cpu(%f) val_cpp(%f) val_gpu(%f) val_gpu_ref(%f) ijk(%d,%d,%d) node_type(0x%x) dense_n(%d)\n",
                            (unsigned long long int)idx, val_cpu, val_cpp, val_gpu, val_gpu_ref, i, j, k, node_type, node_n);
                    }
                }
                if (val_cpu != val_gpu_onindex)
                {
                    mismatched_count_onindex++;
                }
                total_count++;
            }
        }
    }

    pnanovdb_editor::Console::getInstance().addLog("node2_verify_gpu() mismatched(%lld) total_count(%lld)",
        (unsigned long long int)mismatched_count, (unsigned long long int)total_count);
    pnanovdb_editor::Console::getInstance().addLog("node2_verify_gpu() mismatched_onindex(%lld) total_count(%lld)",
        (unsigned long long int)mismatched_count_onindex, (unsigned long long int)total_count);

    printf("node2_verify_gpu() mismatched(%lld) total_count(%lld)\n",
        (unsigned long long int)mismatched_count, (unsigned long long int)total_count);
    printf("node2_verify_gpu() mismatched_onindex(%lld) total_count(%lld)\n",
        (unsigned long long int)mismatched_count_onindex, (unsigned long long int)total_count);

    compute.unmap_array(data_out);
    compute.unmap_array(data_out_ref);
    compute.unmap_array(data_out_onindex);

    compute.destroy_array(data_in);
    compute.destroy_array(data_in_ref);
    compute.destroy_array(data_in_onindex);
    compute.destroy_array(constants);
    compute.destroy_array(data_out);
    compute.destroy_array(data_out_ref);
    compute.destroy_array(data_out_onindex);

    //compute.device_interface.destroy_device(device_manager, device);
    compute.device_interface.destroy_device_manager(device_manager);

    pnanovdb_compute_free(&compute);

    pnanovdb_compiler_free(&compiler);

    return 0;
}

}
