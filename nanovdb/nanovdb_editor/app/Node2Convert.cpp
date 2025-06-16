// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/app/Node2Convert.cpp

    \author Andrew Reidmeyer

    \brief
*/

#include "Node2Convert.h"

#include <nanovdb/io/IO.h>

#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#include "raster/Sphere.h"

namespace pnanovdb_editor
{

void node2_convert(const char* nvdb_path, const char* dst_path)
{
    nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle = nanovdb::io::readGrid(nvdb_path, 0);
    const void* nanovdb_data = gridHandle.data();
    uint64_t nanovdb_size = gridHandle.bufferSize();
    pnanovdb_buf_t buf = pnanovdb_make_buf((pnanovdb_uint32_t*)nanovdb_data, nanovdb_size / 4u);
    pnanovdb_grid_handle_t grid = {};
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_grid_type_t grid_type = pnanovdb_grid_get_grid_type(buf, grid);

    // convert
    {
        std::vector<uint32_t> dst_data;
        dst_data.resize(1024u * 1024u * 1024u);
        pnanovdb_buf_t dst_buf = pnanovdb_make_buf(dst_data.data(), dst_data.size());
        pnanovdb_address_t dst_addr_max = {1024u * 1024u * 1024u};

        pnanovdb_convert_to_node2(dst_buf, dst_addr_max, buf, grid);

        pnanovdb_grid_handle_t dst_grid = {};
        pnanovdb_uint64_t node2_grid_size = pnanovdb_grid_get_grid_size(dst_buf, dst_grid);

        printf("node2_convert in_grid_size(%llu) out_grid_size(%llu)\n",
            (unsigned long long int)nanovdb_size,
            (unsigned long long int)node2_grid_size);

        pnanovdb_address_t bboxes = pnanovdb_grid_get_gridblindmetadata_value_address(dst_buf, dst_grid, 0u);
        pnanovdb_coord_t bbox_min = pnanovdb_read_coord(dst_buf, pnanovdb_address_offset(bboxes, 0u));
        pnanovdb_coord_t bbox_max = pnanovdb_read_coord(dst_buf, pnanovdb_address_offset(bboxes, 12u));

        printf("node2 bbox_min(%d,%d,%d) bbox_max(%d,%d,%d)\n",
            bbox_min.x, bbox_min.y, bbox_min.z, bbox_max.x, bbox_max.y, bbox_max.z);

        FILE* file = fopen(dst_path, "wb");
        fwrite(dst_buf.data, 1u, node2_grid_size, file);
        fclose(file);
    }

    // verify and cpu perf
#if 0
    {
        pnanovdb_coord_t ijk_min = pnanovdb_root_get_bbox_min(buf, root);
        pnanovdb_coord_t ijk_max = pnanovdb_root_get_bbox_max(buf, root);

        nanovdb::GridHandle<nanovdb::HostBuffer> dst_gridHandle = nanovdb::io::readGrid(dst_path, 0);
        const void* dst_nanovdb_data = dst_gridHandle.data();
        uint64_t dst_nanovdb_size = dst_gridHandle.bufferSize();
        pnanovdb_buf_t dst_buf = pnanovdb_make_buf((pnanovdb_uint32_t*)dst_nanovdb_data, dst_nanovdb_size / 4u);
        pnanovdb_grid_handle_t dst_grid = {};

        pnanovdb_node2_handle_t dst_root = pnanovdb_grid_get_node2_root(dst_buf, dst_grid);
        pnanovdb_address_t values = pnanovdb_grid_get_gridblindmetadata_value_address(dst_buf, dst_grid, 1u);

        double val_sum_ref = 0.0;
        double val_sum_node2 = 0.0;
        uint64_t total_count = 0;
        uint64_t mismatched_count = 0;
        for (pnanovdb_int32_t i = ijk_min.x; i < ijk_max.x; i++)
        {
            if ((i & 127) == 0u)
            {
                printf("verifing x_slice(%d)\n", i);
            }
            for (pnanovdb_int32_t j = ijk_min.y; j < ijk_max.y; j++)
            {
                for (pnanovdb_int32_t k = ijk_min.z; k < ijk_max.z; k++)
                {
                    pnanovdb_coord_t ijk = { i, j, k };

                    pnanovdb_address_t val_addr = pnanovdb_root_get_value_address(grid_type, buf, root, PNANOVDB_REF(ijk));
                    float val_ref = pnanovdb_read_float(buf, val_addr);

                    val_addr = pnanovdb_node2_get_value_address(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, values, 32u, 1u, PNANOVDB_REF(ijk));
                    float val_node2 = pnanovdb_read_float(dst_buf, val_addr);

                    if (val_ref != val_node2)
                    {
                        mismatched_count++;
                    }
                    total_count++;

                    val_sum_ref += val_ref;
                    val_sum_node2 += val_node2;
                }
            }
        }
        printf("verification val_sum_ref(%f) val_sum_node2(%f) mismatched_count(%lld) total_count(%lld)\n",
            val_sum_ref, val_sum_node2, (unsigned long long int)mismatched_count, (unsigned long long int)total_count);

        timespec time_val = {};
        clock_gettime(CLOCK_MONOTONIC, &time_val);
        pnanovdb_uint64_t time_0 = 1E9 * pnanovdb_uint64_t(time_val.tv_sec) + pnanovdb_uint64_t(time_val.tv_nsec);

        // ref perf
        val_sum_ref = 0.0;
        for (pnanovdb_int32_t i = ijk_min.x; i < ijk_max.x; i++)
        {
            for (pnanovdb_int32_t j = ijk_min.y; j < ijk_max.y; j++)
            {
                for (pnanovdb_int32_t k = ijk_min.z; k < ijk_max.z; k++)
                {
                    pnanovdb_coord_t ijk = { i, j, k };

                    pnanovdb_address_t val_addr = pnanovdb_root_get_value_address(grid_type, buf, root, PNANOVDB_REF(ijk));
                    float val_ref = pnanovdb_read_float(buf, val_addr);

                    val_sum_ref += val_ref;
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &time_val);
        pnanovdb_uint64_t time_1 = 1E9 * pnanovdb_uint64_t(time_val.tv_sec) + pnanovdb_uint64_t(time_val.tv_nsec);

        // node2 perf
        val_sum_node2 = 0.0;
        for (pnanovdb_int32_t i = ijk_min.x; i < ijk_max.x; i++)
        {
            for (pnanovdb_int32_t j = ijk_min.y; j < ijk_max.y; j++)
            {
                for (pnanovdb_int32_t k = ijk_min.z; k < ijk_max.z; k++)
                {
                    pnanovdb_coord_t ijk = { i, j, k };

                    pnanovdb_address_t val_addr = pnanovdb_node2_get_value_address(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, values, 32u, 1u, PNANOVDB_REF(ijk));
                    float val_node2 = pnanovdb_read_float(dst_buf, val_addr);

                    val_sum_node2 += val_node2;
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &time_val);
        pnanovdb_uint64_t time_2 = 1E9 * pnanovdb_uint64_t(time_val.tv_sec) + pnanovdb_uint64_t(time_val.tv_nsec);

        double ref_ms = (double)(time_1 - time_0) / 1.0E6;
        double node2_ms = (double)(time_2 - time_1) / 1.0E6;
        printf("ref_ms(%f) node2_ms(%f) sum_ref(%f) sum_node2(%f)\n", ref_ms, node2_ms, val_sum_ref, val_sum_node2);
    }
#endif
}

void node2_sphere(const char* dst_path)
{
    // convert
    {
        std::vector<uint32_t> dst_data;
        dst_data.resize(1024u * 1024u * 1024u);
        pnanovdb_buf_t dst_buf = pnanovdb_make_buf(dst_data.data(), dst_data.size());
        pnanovdb_address_t dst_addr_max = {1024u * 1024u * 1024u};

        pnanovdb_node2_generate_sphere(dst_buf, dst_addr_max, 0.125f, 128.f);

        pnanovdb_grid_handle_t dst_grid = {};
        pnanovdb_uint64_t node2_grid_size = pnanovdb_grid_get_grid_size(dst_buf, dst_grid);

        printf("node2_sphere out_grid_size(%llu)\n",
            (unsigned long long int)node2_grid_size);

        FILE* file = fopen(dst_path, "wb");
        fwrite(dst_buf.data, 1u, node2_grid_size, file);
        fclose(file);
    }
}

}
