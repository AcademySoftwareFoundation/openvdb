// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   pnanovdb_validate_strides.h

    \author Andrew Reidmeyer

    \brief  This header implements validation tests for the strides used
            in PNanaoVDB.h (instead of pointers). It can be used both for
            unit-testing (hence its location), but also to update PNanoVDB.h
            if the ABI changes in NanoVDB.h.
*/

#ifndef NANOVDB_PNANOVDB_VALIDATE_STRIDES_H_HAS_BEEN_INCLUDED
#define NANOVDB_PNANOVDB_VALIDATE_STRIDES_H_HAS_BEEN_INCLUDED

#include <nanovdb/PNanoVDB.h>

static pnanovdb_uint32_t allocate(pnanovdb_uint32_t* poffset, pnanovdb_uint32_t size, pnanovdb_uint32_t alignment)
{
    if (alignment > 0u)
    {
        (*poffset) = alignment * (((*poffset) + (alignment - 1)) / alignment);
    }
    pnanovdb_uint32_t ret = (*poffset);
    (*poffset) += size;
    return ret;
}

static void compute_root_strides(
    pnanovdb_uint32_t grid_type,
    pnanovdb_uint32_t* background_off,
    pnanovdb_uint32_t* min_off, pnanovdb_uint32_t* max_off,
    pnanovdb_uint32_t* ave_off, pnanovdb_uint32_t* stddev_off,
    pnanovdb_uint32_t* total_size)
{
    pnanovdb_uint32_t offset = 0u;
    allocate(&offset, PNANOVDB_ROOT_BASE_SIZE, 32u);

    pnanovdb_uint32_t minmaxStride = pnanovdb_grid_type_minmax_strides_bits[grid_type] / 8u;
    pnanovdb_uint32_t minmaxAlign = pnanovdb_grid_type_minmax_aligns_bits[grid_type] / 8u;
    pnanovdb_uint32_t statStride = pnanovdb_grid_type_stat_strides_bits[grid_type] / 8u;

    *background_off = allocate(&offset, minmaxStride, minmaxAlign);
    *min_off = allocate(&offset, minmaxStride, minmaxAlign);
    *max_off = allocate(&offset, minmaxStride, minmaxAlign);
    *ave_off = allocate(&offset, statStride, statStride);
    *stddev_off = allocate(&offset, statStride, statStride);
    *total_size = allocate(&offset, 0u, 32u);
}

static void compute_tile_strides(pnanovdb_uint32_t grid_type, pnanovdb_uint32_t* value_off, pnanovdb_uint32_t* total_size)
{
    pnanovdb_uint32_t offset = 0u;
    allocate(&offset, PNANOVDB_ROOT_TILE_BASE_SIZE, 32u);

    pnanovdb_uint32_t valueStride = pnanovdb_grid_type_minmax_strides_bits[grid_type] / 8u;
    pnanovdb_uint32_t valueAlign = pnanovdb_grid_type_minmax_aligns_bits[grid_type] / 8u;

    *value_off = allocate(&offset, valueStride, valueAlign);
    *total_size = allocate(&offset, 0u, 32u);
}

static void compute_node_strides(
    pnanovdb_uint32_t grid_type,
    pnanovdb_uint32_t nodeLevel,
    pnanovdb_uint32_t* min_off, pnanovdb_uint32_t* max_off,
    pnanovdb_uint32_t* ave_off, pnanovdb_uint32_t* stddev_off,
    pnanovdb_uint32_t* table_off,
    pnanovdb_uint32_t* total_size)
{
    static const pnanovdb_uint32_t node_size[3] = { PNANOVDB_LEAF_BASE_SIZE, PNANOVDB_LOWER_BASE_SIZE, PNANOVDB_UPPER_BASE_SIZE };
    static const pnanovdb_uint32_t node_elements[3] = { PNANOVDB_LEAF_TABLE_COUNT, PNANOVDB_LOWER_TABLE_COUNT, PNANOVDB_UPPER_TABLE_COUNT };
    pnanovdb_uint32_t offset = 0u;
    allocate(&offset, node_size[nodeLevel], 32u);

    pnanovdb_uint32_t valueStrideBits = pnanovdb_grid_type_value_strides_bits[grid_type];
    pnanovdb_uint32_t tableStrideBits = nodeLevel == 0u ? valueStrideBits : pnanovdb_grid_type_table_strides_bits[grid_type];
    pnanovdb_uint32_t tableAlign = 32u;
    pnanovdb_uint32_t tableFullStride = (tableStrideBits * node_elements[nodeLevel]) / 8u;

    pnanovdb_uint32_t minmaxStride = pnanovdb_grid_type_minmax_strides_bits[grid_type] / 8u;
    pnanovdb_uint32_t minmaxAlign = pnanovdb_grid_type_minmax_aligns_bits[grid_type] / 8u;
    pnanovdb_uint32_t statStride = pnanovdb_grid_type_stat_strides_bits[grid_type] / 8u;
    pnanovdb_uint32_t indexMaskStride = 0u;
    if (nodeLevel == 0u)
    {
        if (pnanovdb_grid_type_leaf_type[grid_type] == PNANOVDB_LEAF_TYPE_LITE)
        {
            minmaxStride = 0u;
            minmaxAlign = 0u;
            statStride = 0u;
        }
        else if (pnanovdb_grid_type_leaf_type[grid_type] == PNANOVDB_LEAF_TYPE_FP)
        {
            minmaxStride = 2u;
            minmaxAlign = 2u;
            statStride = 2u;
            // allocate minimum and quantum
            allocate(&offset, 4u, 4u);
            allocate(&offset, 4u, 4u);
        }
        else if (pnanovdb_grid_type_leaf_type[grid_type] == PNANOVDB_LEAF_TYPE_INDEX)
        {
            minmaxStride = 0u;
            minmaxAlign = 0u;
            statStride = 0u;
            tableAlign = 8u;
            tableFullStride = 8u;
        }
        else if (pnanovdb_grid_type_leaf_type[grid_type] == PNANOVDB_LEAF_TYPE_INDEXMASK)
        {
            minmaxStride = 0u;
            minmaxAlign = 0u;
            statStride = 0u;
            tableAlign = 8u;
            tableFullStride = 8u;
            indexMaskStride = 64u;
        }
        else if (pnanovdb_grid_type_leaf_type[grid_type] == PNANOVDB_LEAF_TYPE_POINTINDEX)
        {
            minmaxStride = 8u;
            minmaxAlign = 8u;
            statStride = 0u;
            tableAlign = 2u;
            tableFullStride = (16u * node_elements[nodeLevel]) / 8u;
        }
    }
    *min_off = allocate(&offset, minmaxStride, minmaxAlign);
    *max_off = allocate(&offset, minmaxStride, minmaxAlign);
    *ave_off = allocate(&offset, statStride, statStride);
    *stddev_off = allocate(&offset, statStride, statStride);
    *table_off = allocate(&offset, tableFullStride, tableAlign);
    allocate(&offset, indexMaskStride, tableAlign);
    *total_size = allocate(&offset, 0u, 32u);
}

static bool validate_strides(int(*local_printf)(const char* format, ...))
{
    pnanovdb_grid_type_constants_t constants[PNANOVDB_GRID_TYPE_END];

    for (pnanovdb_uint32_t idx = 0u; idx < PNANOVDB_GRID_TYPE_END; idx++)
    {

        pnanovdb_uint32_t root_background, root_min, root_max, root_ave, root_stddev, root_size;
        compute_root_strides(idx, &root_background, &root_min, &root_max, &root_ave, &root_stddev, &root_size);

        pnanovdb_uint32_t tile_value, tile_size;
        compute_tile_strides(idx, &tile_value, &tile_size);

        pnanovdb_uint32_t upper_min, upper_max, upper_ave, upper_stddev, upper_table, upper_size;
        compute_node_strides(idx, 2, &upper_min, &upper_max, &upper_ave, &upper_stddev, &upper_table, &upper_size);

        pnanovdb_uint32_t lower_min, lower_max, lower_ave, lower_stddev, lower_table, lower_size;
        compute_node_strides(idx, 1, &lower_min, &lower_max, &lower_ave, &lower_stddev, &lower_table, &lower_size);

        pnanovdb_uint32_t leaf_min, leaf_max, leaf_ave, leaf_stddev, leaf_table, leaf_size;
        compute_node_strides(idx, 0, &leaf_min, &leaf_max, &leaf_ave, &leaf_stddev, &leaf_table, &leaf_size);

        pnanovdb_uint32_t valueStrideBits = pnanovdb_grid_type_value_strides_bits[idx];
        pnanovdb_uint32_t tableStrideBits = pnanovdb_grid_type_table_strides_bits[idx];
        pnanovdb_uint32_t tableStride = tableStrideBits / 8u;

        // For FP, always return the base of the table
        if (pnanovdb_grid_type_leaf_type[idx] == PNANOVDB_LEAF_TYPE_FP)
        {
            valueStrideBits = 0u;
        }

        pnanovdb_grid_type_constants_t local_constants = {
            root_background, root_min, root_max, root_ave, root_stddev, root_size,
            valueStrideBits, tableStride, tile_value, tile_size,
            upper_min, upper_max, upper_ave, upper_stddev, upper_table, upper_size,
            lower_min, lower_max, lower_ave, lower_stddev, lower_table, lower_size,
            leaf_min, leaf_max, leaf_ave, leaf_stddev, leaf_table, leaf_size
        };
        constants[idx] = local_constants;
    }

    bool mismatch = false;
    for (pnanovdb_uint32_t idx = 0u; idx < PNANOVDB_GRID_TYPE_END; idx++)
    {
        pnanovdb_grid_type_constants_t c = constants[idx];
        pnanovdb_grid_type_constants_t t = pnanovdb_grid_type_constants[idx];
        if (memcmp(&c, &t, sizeof(pnanovdb_grid_type_constants_t)) != 0)
        {
            mismatch = true;
        }
    }
    if (mismatch)
    {
        local_printf("Error: Mismatch between constant tables.\n");
        for (pnanovdb_uint32_t pass = 0u; pass < 2u; pass++)
        {
            if (pass == 0u)
            {
                local_printf("Printing expected values:\n");
            }
            else
            {
                local_printf("Printing current header values:\n");
            }
            for (pnanovdb_uint32_t idx = 0u; idx < PNANOVDB_GRID_TYPE_END; idx++)
            {
                pnanovdb_grid_type_constants_t c = (pass == 0u) ? constants[idx] : pnanovdb_grid_type_constants[idx];
                local_printf("{%d, %d, %d, %d, %d, %d,  %d, %d, %d, %d,  %d, %d, %d, %d, %d, %d,  %d, %d, %d, %d, %d, %d,  %d, %d, %d, %d, %d, %d},\n",
                    c.root_off_background, c.root_off_min, c.root_off_max, c.root_off_ave, c.root_off_stddev, c.root_size,
                    c.value_stride_bits, c.table_stride, c.root_tile_off_value, c.root_tile_size,
                    c.upper_off_min, c.upper_off_max, c.upper_off_ave, c.upper_off_stddev, c.upper_off_table, c.upper_size,
                    c.lower_off_min, c.lower_off_max, c.lower_off_ave, c.lower_off_stddev, c.lower_off_table, c.lower_size,
                    c.leaf_off_min, c.leaf_off_max, c.leaf_off_ave, c.leaf_off_stddev, c.leaf_off_table, c.leaf_size
                );
            }
        }
    }
    return !mismatch;
}

#endif// end of NANOVDB_PNANOVDB_VALIDATE_STRIDES_H_HAS_BEEN_INCLUDED