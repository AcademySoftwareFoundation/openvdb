// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb/putil/Convert.h

    \author Andrew Reidmeyer

    \brief
*/

#ifndef NANOVDB_CONVERT_H_HAS_BEEN_INCLUDED
#define NANOVDB_CONVERT_H_HAS_BEEN_INCLUDED

#define PNANOVDB_C
#define PNANOVDB_CMATH
#include <nanovdb/PNanoVDB2.h>

#define PNANOVDB_CONVERT_VALIDATION 0

#if PNANOVDB_CONVERT_VALIDATION
#include <stdio.h>
#endif

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_convert_n_to_ijk(pnanovdb_uint32_t node_type, pnanovdb_uint32_t n)
{
    const pnanovdb_uint32_t tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
    const pnanovdb_uint32_t fanout_bits2 = pnanovdb_node2_fanout_bits2[node_type];
    const pnanovdb_uint32_t fanout_bits = pnanovdb_node2_fanout_bits[node_type];
    const pnanovdb_uint32_t mask = (1u << fanout_bits) - 1u;

    pnanovdb_int32_t i = pnanovdb_uint32_as_int32(((n >> fanout_bits2) & mask) << tiledim_bits);
    pnanovdb_int32_t j = pnanovdb_uint32_as_int32(((n >> fanout_bits) & mask) << tiledim_bits);
    pnanovdb_int32_t k = pnanovdb_uint32_as_int32((n & mask) << tiledim_bits);

    pnanovdb_coord_t ret = {i, j, k};
    return ret;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_convert_to_grid_type(
    pnanovdb_buf_t dst_buf,
    pnanovdb_address_t dst_addr_max,
    pnanovdb_uint32_t dst_grid_type,
    pnanovdb_buf_t buf
)
{
    pnanovdb_grid_handle_t grid = {};
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);

    pnanovdb_address_t bboxes = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 0u);
    pnanovdb_address_t values = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 1u);

    pnanovdb_address_t dst_addr_end = pnanovdb_address_null();

    // allocate grid, tree
    pnanovdb_grid_handle_t dst_grid = {dst_addr_end};
    pnanovdb_uint32_t dst_grid_size = PNANOVDB_GRID_SIZE + PNANOVDB_TREE_SIZE;
    dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_grid_size);

    // copy grid header, but change type
    pnanovdb_node2_memcpy(dst_buf, dst_grid.address, buf, grid.address, PNANOVDB_GRID_SIZE);
    pnanovdb_grid_set_grid_type(dst_buf, dst_grid, dst_grid_type);

    pnanovdb_tree_handle_t dst_tree = pnanovdb_grid_get_tree(dst_buf, dst_grid);
    pnanovdb_node2_memclear(dst_buf, dst_tree.address, PNANOVDB_TREE_SIZE);

    pnanovdb_node2_handle_t root = {pnanovdb_uint32_t(pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u)};
    for (pnanovdb_uint32_t node_type = 0u; node_type < 5u; node_type++)
    {
        // alloc root
        if (node_type == 0u)
        {
            pnanovdb_root_handle_t dst_root = {dst_addr_end};
            pnanovdb_uint32_t dst_root_size = PNANOVDB_GRID_TYPE_GET(dst_grid_type, root_size);
            dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_root_size);
            pnanovdb_node2_memclear(dst_buf, dst_root.address, dst_root_size);
            // alloc root tile for every active key
            for (pnanovdb_uint32_t root_n = 0u; root_n < 32768u; root_n++)
            {
                pnanovdb_uint64_t key = pnanovdb_node2_get_key(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
                if (key != pnanovdb_node2_end_key)
                {
                    pnanovdb_root_tile_handle_t dst_root_tile = {dst_addr_end};
                    pnanovdb_uint32_t dst_root_tile_size = PNANOVDB_GRID_TYPE_GET(dst_grid_type, root_tile_size);
                    dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_root_tile_size);
                    pnanovdb_node2_memclear(dst_buf, dst_root_tile.address, dst_root_tile_size);

                    pnanovdb_uint32_t state = pnanovdb_node2_get_value_mask_bit(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n) ? 1 : 0;

                    pnanovdb_root_tile_set_key(dst_buf, dst_root_tile, key);
                    pnanovdb_root_tile_set_state(dst_buf, dst_root_tile, state);

                    pnanovdb_uint32_t tile_count = pnanovdb_root_get_tile_count(dst_buf, dst_root);
                    tile_count++;
                    pnanovdb_root_set_tile_count(dst_buf, dst_root, tile_count);
                }
            }
            pnanovdb_coord_t bbox_min = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 0u));
            pnanovdb_coord_t bbox_max = pnanovdb_read_coord(buf, pnanovdb_address_offset(bboxes, 12u));
            pnanovdb_root_set_bbox_min(dst_buf, dst_root, PNANOVDB_REF(bbox_min));
            pnanovdb_root_set_bbox_max(dst_buf, dst_root, PNANOVDB_REF(bbox_max));
            pnanovdb_tree_set_first_root(dst_buf, dst_tree, dst_root);
            continue;
        }
        // get dst root
        pnanovdb_uint64_t root_value_idx = pnanovdb_node2_read(buf, root.idx64 + pnanovdb_node2_off_value_idx);
        pnanovdb_root_handle_t dst_root = pnanovdb_tree_get_root(dst_buf, dst_tree);
        for (pnanovdb_uint32_t root_n = 0u; root_n < 32768u; root_n++)
        {
            pnanovdb_uint64_t key = pnanovdb_node2_get_key(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
            pnanovdb_coord_t ijk = pnanovdb_node2_key_to_coord(key);
            pnanovdb_root_tile_handle_t dst_root_tile = {};
            if (key != pnanovdb_node2_end_key)
            {
                dst_root_tile = pnanovdb_root_find_tile(dst_grid_type, dst_buf, dst_root, PNANOVDB_REF(ijk));
            }
            if (node_type == 4u)
            {
                pnanovdb_bool_t root_tile_is_active;
                pnanovdb_uint32_t root_tile_local_idx = pnanovdb_node2_mask_n_to_idx(buf,
                    root.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[PNANOVDB_NODE2_TYPE_ROOT],
                    PNANOVDB_NODE2_TYPE_ROOT, root_n, PNANOVDB_REF(root_tile_is_active));
                if (root_tile_is_active && dst_grid_type == PNANOVDB_GRID_TYPE_ONINDEX)
                {
                    pnanovdb_address_t val_addr = pnanovdb_root_tile_get_value_address(dst_grid_type, dst_buf, dst_root_tile);
                    pnanovdb_write_uint64(dst_buf, val_addr, root_value_idx + root_tile_local_idx);
                }
            }
            pnanovdb_node2_handle_t upper = pnanovdb_node2_get_child(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
            if (upper.idx64 == 0u)
            {
                continue;
            }
            // alloc upper
            if (node_type == 1u)
            {
                pnanovdb_upper_handle_t dst_upper = {dst_addr_end};
                pnanovdb_uint32_t dst_upper_size = PNANOVDB_GRID_TYPE_GET(dst_grid_type, upper_size);
                dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_upper_size);
                pnanovdb_node2_memclear(dst_buf, dst_upper.address, dst_upper_size);

                pnanovdb_node2_memcpy(dst_buf, pnanovdb_address_offset(dst_upper.address, PNANOVDB_UPPER_OFF_CHILD_MASK),
                    buf, pnanovdb_address_offset_product(pnanovdb_address_null(),
                    upper.idx64 + pnanovdb_node2_off_child_mask[PNANOVDB_NODE2_TYPE_UPPER], 8u), 32768u / 8u);
                pnanovdb_node2_memcpy(dst_buf, pnanovdb_address_offset(dst_upper.address, PNANOVDB_UPPER_OFF_VALUE_MASK),
                    buf, pnanovdb_address_offset_product(pnanovdb_address_null(),
                    upper.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_UPPER], 8u), 32768u / 8u);

                pnanovdb_root_tile_set_child(dst_buf, dst_root_tile, pnanovdb_address_diff(dst_upper.address, dst_root.address));

                continue;
            }
            // get dst upper
            pnanovdb_uint64_t upper_value_idx = pnanovdb_node2_read(buf, upper.idx64 + pnanovdb_node2_off_value_idx);
            pnanovdb_upper_handle_t dst_upper = pnanovdb_root_get_child(dst_grid_type, dst_buf, dst_root, dst_root_tile);
            for (pnanovdb_uint32_t upper_n = 0u; upper_n < 32768u; upper_n++)
            {
                if (node_type == 4u)
                {
                    pnanovdb_bool_t upper_is_active;
                    pnanovdb_uint32_t upper_local_idx = pnanovdb_node2_mask_n_to_idx(buf,
                        upper.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[PNANOVDB_NODE2_TYPE_UPPER],
                        PNANOVDB_NODE2_TYPE_UPPER, upper_n, PNANOVDB_REF(upper_is_active));
                    if (upper_is_active && dst_grid_type == PNANOVDB_GRID_TYPE_ONINDEX)
                    {
                        pnanovdb_address_t val_addr = pnanovdb_upper_get_table_address(dst_grid_type, dst_buf, dst_upper, upper_n);
                        pnanovdb_write_uint64(dst_buf, val_addr, upper_value_idx + upper_local_idx);
                    }
                }
                pnanovdb_node2_handle_t lower = pnanovdb_node2_get_child(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);
                if (lower.idx64 == 0u)
                {
                    continue;
                }
                // alloc lower
                if (node_type == 2u)
                {
                    pnanovdb_lower_handle_t dst_lower = {dst_addr_end};
                    pnanovdb_uint32_t dst_lower_size = PNANOVDB_GRID_TYPE_GET(dst_grid_type, lower_size);
                    dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_lower_size);
                    pnanovdb_node2_memclear(dst_buf, dst_lower.address, dst_lower_size);

                    pnanovdb_node2_memcpy(dst_buf, pnanovdb_address_offset(dst_lower.address, PNANOVDB_LOWER_OFF_CHILD_MASK),
                        buf, pnanovdb_address_offset_product(pnanovdb_address_null(),
                        lower.idx64 + pnanovdb_node2_off_child_mask[PNANOVDB_NODE2_TYPE_LOWER], 8u), 4096u / 8u);
                    pnanovdb_node2_memcpy(dst_buf, pnanovdb_address_offset(dst_lower.address, PNANOVDB_LOWER_OFF_VALUE_MASK),
                        buf, pnanovdb_address_offset_product(pnanovdb_address_null(),
                        lower.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_LOWER], 8u), 4096u / 8u);

                    pnanovdb_upper_set_child(dst_grid_type, dst_buf, dst_upper, upper_n, dst_lower);

                    continue;
                }
                // get dst lower
                pnanovdb_uint64_t lower_value_idx = pnanovdb_node2_read(buf, lower.idx64 + pnanovdb_node2_off_value_idx);
                pnanovdb_lower_handle_t dst_lower = pnanovdb_upper_get_child(dst_grid_type, dst_buf, dst_upper, upper_n);
                for (pnanovdb_uint32_t lower_n = 0u; lower_n < 4096u; lower_n++)
                {
                    if (node_type == 4u)
                    {
                        pnanovdb_bool_t lower_is_active;
                        pnanovdb_uint32_t lower_local_idx = pnanovdb_node2_mask_n_to_idx(buf,
                            lower.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[PNANOVDB_NODE2_TYPE_LOWER],
                            PNANOVDB_NODE2_TYPE_LOWER, lower_n, PNANOVDB_REF(lower_is_active));
                        if (lower_is_active && dst_grid_type == PNANOVDB_GRID_TYPE_ONINDEX)
                        {
                            pnanovdb_address_t val_addr = pnanovdb_lower_get_table_address(dst_grid_type, dst_buf, dst_lower, lower_n);
                            pnanovdb_write_uint64(dst_buf, val_addr, lower_value_idx + lower_local_idx);
                        }
                    }
                    pnanovdb_node2_handle_t leaf = pnanovdb_node2_get_child(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                    if (leaf.idx64 == 0u)
                    {
                        continue;
                    }
                    // alloc leaf
                    if (node_type == 3u)
                    {
                        pnanovdb_leaf_handle_t dst_leaf = {dst_addr_end};
                        pnanovdb_uint32_t dst_leaf_size = PNANOVDB_GRID_TYPE_GET(dst_grid_type, leaf_size);
                        dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_leaf_size);
                        pnanovdb_node2_memclear(dst_buf, dst_leaf.address, dst_leaf_size);

                        pnanovdb_node2_memcpy(dst_buf, pnanovdb_address_offset(dst_leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK),
                            buf, pnanovdb_address_offset_product(pnanovdb_address_null(),
                            leaf.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_LEAF], 8u), 512u / 8u);

                        pnanovdb_lower_set_child(dst_grid_type, dst_buf, dst_lower, lower_n, dst_leaf);

                        continue;
                    }
                    if (node_type == 4u)
                    {
                        pnanovdb_uint64_t leaf_value_idx = pnanovdb_node2_read(buf, leaf.idx64 + pnanovdb_node2_off_value_idx);
                        pnanovdb_leaf_handle_t dst_leaf = pnanovdb_lower_get_child(dst_grid_type, dst_buf, dst_lower, lower_n);
                        if (dst_grid_type == PNANOVDB_GRID_TYPE_ONINDEX)
                        {
                            pnanovdb_address_t offset_addr = pnanovdb_leaf_get_table_address(dst_grid_type, dst_buf, dst_leaf, 0u);
                            pnanovdb_write_uint64(dst_buf, offset_addr, leaf_value_idx);
                            // compute prefix sum
                            pnanovdb_uint64_t sum = 0llu;
                            pnanovdb_uint32_t accum = 0u;
                            for (pnanovdb_uint32_t idx = 0u; idx < 7u; idx++)
                            {
                                pnanovdb_uint64_t mask = pnanovdb_read_uint64(dst_buf,
                                    pnanovdb_address_offset(dst_leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * idx));
                                accum += pnanovdb_uint64_countbits(mask);
                                sum |= (pnanovdb_uint64_t(accum) << (9 * idx));
                            }
                            pnanovdb_write_uint64(dst_buf, pnanovdb_address_offset(offset_addr, 8u), sum);
                        }
                        //else if (dst_grid_type == PNANOVDB_GRID_TYPE_RGBA8)
                        //{
                        //    for (pnanovdb_uint32_t leaf_n = 0u; leaf_n < 512u; leaf_n++)
                        //    {
                        //        // copy voxels
                        //    }
                        //}
                    }
                }
            }
        }
    }
    // copy metadata
    if (dst_grid_type == PNANOVDB_GRID_TYPE_ONINDEX)
    {
        // value data is at idx==1 by convention
        pnanovdb_gridblindmetadata_handle_t metadata = pnanovdb_grid_get_gridblindmetadata(buf, grid, 1u);
        pnanovdb_address_t src_data = pnanovdb_grid_get_gridblindmetadata_value_address(buf, grid, 1u);

        pnanovdb_gridblindmetadata_handle_t dst_metadata = {dst_addr_end};
        pnanovdb_uint32_t dst_metadata_size = PNANOVDB_GRIDBLINDMETADATA_SIZE;
        dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_metadata_size);

        pnanovdb_node2_memcpy(dst_buf, dst_metadata.address, buf, metadata.address, dst_metadata_size);

        pnanovdb_uint64_t value_count = pnanovdb_gridblindmetadata_get_value_count(buf, metadata);
        pnanovdb_uint32_t value_size = pnanovdb_gridblindmetadata_get_value_size(buf, metadata);
        pnanovdb_address_t dst_data = dst_addr_end;
        pnanovdb_uint32_t dst_data_size = value_count * value_size;
        dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_data_size);

        pnanovdb_node2_memcpy(dst_buf, dst_data, buf, src_data, dst_data_size);

        pnanovdb_gridblindmetadata_set_data_offset(dst_buf, dst_metadata, pnanovdb_address_diff(dst_data, dst_metadata.address));

        pnanovdb_grid_set_first_gridblindmetadata(dst_buf, dst_grid, dst_metadata);
        pnanovdb_grid_set_blind_metadata_count(dst_buf, dst_grid, 1u);
    }

#if PNANOVDB_CONVERT_VALIDATION
    // validation
    {
        pnanovdb_root_handle_t dst_root = pnanovdb_tree_get_root(dst_buf, dst_tree);

        pnanovdb_uint32_t dst_tile_count = pnanovdb_root_get_tile_count(dst_buf, dst_root);
        for (pnanovdb_uint32_t dst_tile_idx = 0u; dst_tile_idx < dst_tile_count; dst_tile_idx++)
        {
            pnanovdb_root_tile_handle_t dst_tile = pnanovdb_root_get_tile(dst_grid_type, dst_root, dst_tile_idx);
            pnanovdb_int64_t child = pnanovdb_root_tile_get_child(dst_buf, dst_tile);
            pnanovdb_uint64_t key = pnanovdb_root_tile_get_key(dst_buf, dst_tile);
            pnanovdb_uint32_t state = pnanovdb_root_tile_get_state(dst_buf, dst_tile);
            printf("tile[%d] child(%llu) key(%llx) state(%u)\n",
                dst_tile_idx, (unsigned long long int)child, (unsigned long long int)key, state);
        }

        pnanovdb_uint32_t mismatch_count = 0u;
        pnanovdb_uint32_t match_count = 0u;
        for (pnanovdb_uint32_t root_n = 0u; root_n < 32768u; root_n++)
        {
            pnanovdb_uint64_t key = pnanovdb_node2_get_key(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
            pnanovdb_coord_t root_ijk = pnanovdb_node2_key_to_coord(key);
            pnanovdb_node2_handle_t upper = pnanovdb_node2_get_child(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
            if (upper.idx64 == 0llu)
            {
                continue;
            }
            for (pnanovdb_uint32_t upper_n = 0u; upper_n < 32768u; upper_n++)
            {
                pnanovdb_coord_t upper_ijk = pnanovdb_node2_convert_n_to_ijk(PNANOVDB_NODE2_TYPE_UPPER, upper_n);
                pnanovdb_node2_handle_t lower = pnanovdb_node2_get_child(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);
                if (lower.idx64 == 0llu)
                {
                    continue;
                }
                for (pnanovdb_uint32_t lower_n = 0u; lower_n < 4096u; lower_n++)
                {
                    pnanovdb_coord_t lower_ijk = pnanovdb_node2_convert_n_to_ijk(PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                    pnanovdb_node2_handle_t leaf = pnanovdb_node2_get_child(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                    if (leaf.idx64 == 0u)
                    {
                        continue;
                    }
                    for (pnanovdb_uint32_t leaf_n = 0u; leaf_n < 512u; leaf_n++)
                    {
                        pnanovdb_coord_t leaf_ijk = pnanovdb_node2_convert_n_to_ijk(PNANOVDB_NODE2_TYPE_LEAF, leaf_n);
                        pnanovdb_coord_t ijk = {
                            root_ijk.x + upper_ijk.x + lower_ijk.x + leaf_ijk.x,
                            root_ijk.y + upper_ijk.y + lower_ijk.y + leaf_ijk.y,
                            root_ijk.z + upper_ijk.z + lower_ijk.z + leaf_ijk.z
                        };
                        pnanovdb_readaccessor_t acc = {};
                        pnanovdb_readaccessor_init(PNANOVDB_REF(acc), dst_root);
                        pnanovdb_uint32_t level = 0u;
                        pnanovdb_address_t val_addr = pnanovdb_readaccessor_get_value_address_and_level(dst_grid_type, dst_buf,
                            PNANOVDB_REF(acc), PNANOVDB_REF(ijk), PNANOVDB_REF(level));
                        pnanovdb_uint64_t val_idx;
                        if (level == 0)
                        {
                            val_idx = pnanovdb_leaf_onindex_get_value_index(dst_buf, val_addr, PNANOVDB_REF(ijk));
                        }
                        else
                        {
                            val_idx = pnanovdb_read_uint64(dst_buf, val_addr);
                        }

                        pnanovdb_uint64_t leaf_value_idx = pnanovdb_node2_read(buf, leaf.idx64 + pnanovdb_node2_off_value_idx);
                        pnanovdb_bool_t leaf_is_active;
                        pnanovdb_uint32_t leaf_local_idx = pnanovdb_node2_mask_n_to_idx(buf,
                            leaf.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[PNANOVDB_NODE2_TYPE_LEAF],
                            PNANOVDB_NODE2_TYPE_LEAF, leaf_n, PNANOVDB_REF(leaf_is_active));
                        pnanovdb_uint64_t node2_val_idx = leaf_value_idx + leaf_local_idx;
                        if (leaf_is_active)
                        {
                            if (val_idx == node2_val_idx)
                            {
                                match_count++;
                            }
                            else
                            {
                                mismatch_count++;
                                if (mismatch_count <= 32u)
                                {
                                    printf("tree(%d,%d,%d,%d) acc(%llu,%llu,%llu,%llu) val_idx(%llu) vs node2_val_idx(%llu)\n",
                                        root_n, upper_n, lower_n, leaf_n,
                                        (unsigned long long int)acc.root.address.byte_offset, (unsigned long long int)acc.upper.address.byte_offset,
                                        (unsigned long long int)acc.lower.address.byte_offset, (unsigned long long int)acc.leaf.address.byte_offset,
                                        (unsigned long long int)val_idx, (unsigned long long int)node2_val_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
        printf("pnanovdb_node2_convert_to_grid_type() match_count(%d) mismatch_count(%d)\n", match_count, mismatch_count);
    }
#endif
}

#endif
