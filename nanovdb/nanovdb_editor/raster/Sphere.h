// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/raster/Sphere.h

    \author Andrew Reidmeyer

    \brief
*/

#ifndef NANOVDB_SPHERE_H_HAS_BEEN_INCLUDED
#define NANOVDB_SPHERE_H_HAS_BEEN_INCLUDED

#define PNANOVDB_C
#define PNANOVDB_CMATH
#include <nanovdb/PNanoVDB2.h>

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_read_uint32_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint64_t index)
{
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset64_product(addr, index, 4u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_node2_write_uint32_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint64_t index, pnanovdb_uint32_t val)
{
    pnanovdb_write_uint32(buf, pnanovdb_address_offset64_product(addr, index, 4u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_empty_grid_size()
{
    pnanovdb_uint32_t grid_size = PNANOVDB_GRID_SIZE + PNANOVDB_TREE_SIZE;
    // node2 root
    grid_size += pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_ROOT];
    return grid_size;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_init_grid(
    pnanovdb_buf_t buf,
    pnanovdb_uint32_t empty_grid_size,
    float voxel_size
)
{
    float voxel_size_inv = 1.f / voxel_size;

    pnanovdb_node2_memclear(buf, pnanovdb_address_null(), empty_grid_size);

    pnanovdb_grid_handle_t grid = {pnanovdb_address_null()};
    pnanovdb_grid_set_magic(buf, grid, PNANOVDB_MAGIC_GRID);
    pnanovdb_grid_set_version(buf, grid,
        pnanovdb_make_version(PNANOVDB_MAJOR_VERSION_NUMBER, PNANOVDB_MINOR_VERSION_NUMBER, PNANOVDB_PATCH_VERSION_NUMBER));
    pnanovdb_grid_set_flags(buf, grid, 0u);
    pnanovdb_grid_set_grid_index(buf, grid, 0u);
    pnanovdb_grid_set_grid_count(buf, grid, 1u);
    pnanovdb_grid_set_grid_size(buf, grid, empty_grid_size);
    pnanovdb_grid_set_grid_name(buf, grid, 0u, 0x65646f6e);     // "node2"
    pnanovdb_grid_set_grid_name(buf, grid, 1u, 0x00000032);
    pnanovdb_grid_set_voxel_size(buf, grid, 0u, voxel_size);
    pnanovdb_grid_set_voxel_size(buf, grid, 1u, voxel_size);
    pnanovdb_grid_set_voxel_size(buf, grid, 2u, voxel_size);
    pnanovdb_grid_set_grid_class(buf, grid, PNANOVDB_GRID_CLASS_UNKNOWN);
    pnanovdb_grid_set_grid_type(buf, grid, PNANOVDB_GRID_TYPE_NODE2);

    pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
    pnanovdb_map_set_matf(buf, map, 0u, voxel_size);
    pnanovdb_map_set_matf(buf, map, 4u, voxel_size);
    pnanovdb_map_set_matf(buf, map, 8u, voxel_size);
    pnanovdb_map_set_invmatf(buf, map, 0u, voxel_size_inv);
    pnanovdb_map_set_invmatf(buf, map, 4u, voxel_size_inv);
    pnanovdb_map_set_invmatf(buf, map, 8u, voxel_size_inv);
    pnanovdb_map_set_matd(buf, map, 0u, voxel_size);
    pnanovdb_map_set_matd(buf, map, 4u, voxel_size);
    pnanovdb_map_set_matd(buf, map, 8u, voxel_size);
    pnanovdb_map_set_invmatd(buf, map, 0u, voxel_size_inv);
    pnanovdb_map_set_invmatd(buf, map, 4u, voxel_size_inv);
    pnanovdb_map_set_invmatd(buf, map, 8u, voxel_size_inv);
    pnanovdb_map_set_vecf(buf, map, 0u, 0.f);
    pnanovdb_map_set_vecf(buf, map, 1u, 0.f);
    pnanovdb_map_set_vecf(buf, map, 2u, 0.f);
    pnanovdb_map_set_vecd(buf, map, 0u, 0.0);
    pnanovdb_map_set_vecd(buf, map, 1u, 0.0);
    pnanovdb_map_set_vecd(buf, map, 2u, 0.0);

    // legacy tree for ABI compatibility
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    // initialized by pnanovdb_node2_memclear()

    pnanovdb_address_t root_addr = pnanovdb_address_offset(tree.address, PNANOVDB_TREE_SIZE);
    pnanovdb_node2_handle_t root = {pnanovdb_uint32_t(root_addr.byte_offset >> 3u)};
    // initialized by pnanovdb_node2_memclear()

    // set keys to default
    for (pnanovdb_uint32_t node_n = 0u; node_n < 2u * pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT]; node_n++)
    {
        pnanovdb_node2_write(buf, root.idx64 + pnanovdb_node2_off_children[PNANOVDB_NODE2_TYPE_ROOT] + node_n, pnanovdb_node2_end_key);
    }

    pnanovdb_address_t addr_end = pnanovdb_address_offset64(root_addr, pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_ROOT]);

    // point tree at root
    pnanovdb_tree_set_node_offset_root(buf, tree, pnanovdb_address_diff(root_addr, tree.address));
}

// -1 sphere complete inside box
//  0 sphere overlapping box
// +1 sphere does not overlap box
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_sphere_box_overlap(
    pnanovdb_coord_t ijk_box_min,
    pnanovdb_coord_t ijk_box_max,
    float radius
)
{
    pnanovdb_vec3_t box_min = {
        pnanovdb_int32_to_float(ijk_box_min.x),
        pnanovdb_int32_to_float(ijk_box_min.y),
        pnanovdb_int32_to_float(ijk_box_min.z)};
    pnanovdb_vec3_t box_max = {
        pnanovdb_int32_to_float(ijk_box_max.x),
        pnanovdb_int32_to_float(ijk_box_max.y),
        pnanovdb_int32_to_float(ijk_box_max.z)};


    pnanovdb_vec3_t box_min2 = pnanovdb_vec3_mul(box_min, box_min);
    pnanovdb_vec3_t box_max2 = pnanovdb_vec3_mul(box_max, box_max);

    float d2 = radius * radius;

    if (box_min.x > 0.f) { d2 -= box_min2.x;}
    else if (box_max.x < 0.f) { d2 -= box_max2.x;}
    if (box_min.y > 0.f) { d2 -= box_min2.y;}
    else if (box_max.y < 0.f) { d2 -= box_max2.y;}
    if (box_min.z > 0.f) { d2 -= box_min2.z;}
    else if (box_max.z < 0.f) { d2 -= box_max2.z;}

    int ret = 1;
    if (d2 > 0.f)
    {
        ret = 0;
        pnanovdb_vec3_t box_abs2 = pnanovdb_vec3_max(box_min2, box_max2);
        if (box_abs2.x + box_abs2.y + box_abs2.z < radius * radius)
        {
            ret = -1;
        }

    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_n_to_ijk(pnanovdb_uint32_t node_type, pnanovdb_uint32_t n)
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

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_fastroot_ijk_wrap(pnanovdb_uint32_t node_type, pnanovdb_coord_t ijk_in)
{
    pnanovdb_coord_t ijk = ijk_in;
    if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
    {
        const pnanovdb_uint32_t tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
        const pnanovdb_uint32_t fanout_bits = pnanovdb_node2_fanout_bits[node_type];
        const pnanovdb_uint32_t range_half = (1u << (fanout_bits + tiledim_bits)) >> 1u;
        ijk.x = ijk.x >= range_half ? ijk.x - range_half - range_half : ijk.x;
        ijk.y = ijk.y >= range_half ? ijk.y - range_half - range_half : ijk.y;
        ijk.z = ijk.z >= range_half ? ijk.z - range_half - range_half : ijk.z;
    }
    return ijk;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_estimate_child_capacity(
    pnanovdb_buf_t buf,
    pnanovdb_uint32_t node_type,
    float voxel_radius,
    pnanovdb_coord_t parent_ijk
)
{
    pnanovdb_uint32_t child_capacity = 0u;
    pnanovdb_uint32_t fanout = pnanovdb_node2_fanout_1d[node_type];
    for (pnanovdb_uint32_t n = 0u; n < fanout; n++)
    {
        const pnanovdb_uint32_t tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
        const pnanovdb_int32_t tile_size = pnanovdb_uint32_as_int32(1u << tiledim_bits);

        pnanovdb_coord_t ijk = pnanovdb_node2_n_to_ijk(node_type, n);
        ijk = pnanovdb_node2_fastroot_ijk_wrap(node_type, ijk);
        pnanovdb_coord_t ijk_min = pnanovdb_coord_add(ijk, parent_ijk);
        pnanovdb_coord_t ijk_max = {ijk_min.x + tile_size, ijk_min.y + tile_size, ijk_min.z + tile_size};

        pnanovdb_int32_t overlap = pnanovdb_sphere_box_overlap(ijk_min, ijk_max, voxel_radius);

        if (overlap == 0)
        {
            child_capacity++;
        }
    }
    // to ensure 32 byte alignment, round to nearest 4 uint64_t
    child_capacity = 4u * ((child_capacity + 3u) / 4u);
    return child_capacity;
}

PNANOVDB_FORCE_INLINE pnanovdb_node2_handle_t pnanovdb_node2_generate_sphere_node(
    pnanovdb_buf_t buf,
    pnanovdb_address_t addr_max,
    pnanovdb_uint32_t node_type,
    pnanovdb_uint32_t child_type,
    float voxel_radius,
    pnanovdb_node2_handle_t node,
    pnanovdb_coord_t parent_ijk,
    pnanovdb_uint32_t n_in,
    PNANOVDB_INOUT(pnanovdb_coord_t) out_ijk,
    PNANOVDB_INOUT(pnanovdb_address_t) addr_end,
    PNANOVDB_INOUT(pnanovdb_uint64_t) node_idx
)
{
    pnanovdb_node2_handle_t ret = {};

    // on first pass, compute the value level masks
    if (n_in == 0)
    {
        const pnanovdb_uint32_t fanout_bits = pnanovdb_node2_fanout_bits[node_type];
        pnanovdb_uint32_t value_level_count = pnanovdb_node2_fanout_1d[node_type] >> 3u;
        for (pnanovdb_uint32_t idx = 0u; idx < value_level_count; idx++)
        {
            pnanovdb_uint32_t shift = fanout_bits - 1u;
            pnanovdb_uint32_t mask = (1u << shift) - 1u;
            pnanovdb_int32_t i = pnanovdb_uint32_as_int32(((idx >> (shift + shift)) & mask) << 1u);
            pnanovdb_int32_t j = pnanovdb_uint32_as_int32(((idx >> shift) & mask) << 1u);
            pnanovdb_int32_t k = pnanovdb_uint32_as_int32((idx & mask) << 1u);

            // for leaf, ijk here in 0 to 3 range
            pnanovdb_uint32_t value_level = fanout_bits;
            // for fastroot, automatically one less due to modulo wrap at origin
            if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
            {
                value_level--;
            }
            while (value_level > 0u)
            {
                pnanovdb_int32_t bitmask = (1u << value_level) - 1u;
                pnanovdb_coord_t ijk_min = {i, j, k};
                ijk_min.x &= ~bitmask;
                ijk_min.y &= ~bitmask;
                ijk_min.z &= ~bitmask;

                const pnanovdb_uint32_t tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
                ijk_min.x = ijk_min.x << tiledim_bits;
                ijk_min.y = ijk_min.y << tiledim_bits;
                ijk_min.z = ijk_min.z << tiledim_bits;

                ijk_min = pnanovdb_coord_add(ijk_min, parent_ijk);

                ijk_min = pnanovdb_node2_fastroot_ijk_wrap(node_type, ijk_min);

                pnanovdb_coord_t ijk_max = ijk_min;
                ijk_max.x += 1u << (tiledim_bits + value_level);
                ijk_max.y += 1u << (tiledim_bits + value_level);
                ijk_max.z += 1u << (tiledim_bits + value_level);

                pnanovdb_int32_t overlap = pnanovdb_sphere_box_overlap(ijk_min, ijk_max, voxel_radius);
                if (overlap != 0)
                {
                    break;
                }

                value_level--;
            }

            pnanovdb_uint64_t value_level_word = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_level + (idx >> 4u));
            value_level_word = value_level_word & ~(pnanovdb_uint32_as_uint64_low(15) << (4 * (idx & 15)));
            value_level_word = value_level_word | (pnanovdb_uint32_as_uint64_low(value_level) << (4 * (idx & 15)));
            pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_value_level + (idx >> 4u), value_level_word);
        }
    }

    pnanovdb_uint32_t n = n_in;
    pnanovdb_uint32_t level = 0u;
    n = pnanovdb_node2_adaptive_n(buf, node, node_type, n, PNANOVDB_REF(level));

    if (n != n_in)
    {
        return ret;
    }

    const pnanovdb_uint32_t tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
    const pnanovdb_int32_t tile_size = pnanovdb_uint32_as_int32(1u << tiledim_bits);

    pnanovdb_coord_t ijk = pnanovdb_node2_n_to_ijk(node_type, n);
    ijk = pnanovdb_node2_fastroot_ijk_wrap(node_type, ijk);
    pnanovdb_coord_t ijk_min = pnanovdb_coord_add(ijk, parent_ijk);
    pnanovdb_coord_t ijk_max = {ijk_min.x + tile_size, ijk_min.y + tile_size, ijk_min.z + tile_size};

    PNANOVDB_DEREF(out_ijk) = ijk_min;

    pnanovdb_int32_t overlap = pnanovdb_sphere_box_overlap(ijk_min, ijk_max, voxel_radius);

    // max depth condition
    if (node_type == child_type && overlap == 0)
    {
        overlap = -1;
    }

    if (overlap == -1) // fully inside, add tile value
    {
        pnanovdb_uint64_t mask = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + (n >> 6u));
        pnanovdb_uint64_t mask_old = mask;
        mask |= (1llu << (n & 63));
        pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + (n >> 6u), mask);

        if (mask != mask_old)
        {
            // any time mask changes, update prefix sum to avoid incoherence
            pnanovdb_node2_compute_prefix_sums(buf, node, node_type);
        }
    }
    else if (overlap == 0) // partial overlap, add child
    {
        pnanovdb_uint64_t mask = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + (n >> 6u));
        pnanovdb_uint64_t mask_old = mask;
        mask |= (1llu << (n & 63));
        pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + (n >> 6u), mask);

        if (mask != mask_old)
        {
            // any time mask changes, update prefix sum to avoid incoherence
            pnanovdb_node2_compute_prefix_sums(buf, node, node_type);

            // estimate child's child capacity and needed size
            pnanovdb_uint32_t child_child_cap = pnanovdb_node2_estimate_child_capacity(buf, child_type, voxel_radius, ijk_min);
            pnanovdb_uint32_t child_size = pnanovdb_node2_max_size[node_type];
            if (node_type != PNANOVDB_NODE2_TYPE_ROOT)
            {
                child_size = child_size - ((pnanovdb_node2_fanout_1d[node_type] - child_child_cap) << 3u);
            }

            // allocate child
            pnanovdb_address_t child_addr = PNANOVDB_DEREF(addr_end);
            pnanovdb_node2_handle_t child = {pnanovdb_uint32_t(child_addr.byte_offset >> 3u)};
            PNANOVDB_DEREF(addr_end) = pnanovdb_address_offset(PNANOVDB_DEREF(addr_end), child_size);
            pnanovdb_node2_memclear(buf, child_addr, child_size);

            pnanovdb_node2_write(buf, child.idx64 + pnanovdb_node2_off_node_idx, PNANOVDB_DEREF(node_idx));
            PNANOVDB_DEREF(node_idx)++;

            pnanovdb_node2_set_child(buf, node, node_type, n, child);

            ret = child;
        }
        else
        {
            ret = pnanovdb_node2_get_child(buf, node, node_type, n);
        }
    }

    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_compute_active_values_word_count(
    pnanovdb_buf_t buf,
    pnanovdb_node2_handle_t node,
    pnanovdb_uint32_t node_type
)
{
    pnanovdb_uint32_t active_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type])) & 0xFFFF;
    return active_count;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_value_level_to_rgba(pnanovdb_uint32_t value_level)
{
    pnanovdb_uint32_t idx = value_level % 6u;
    pnanovdb_uint32_t rgba = 0u;
    if (idx == 0) {rgba = 0xFF0000FF;}
    if (idx == 1) {rgba = 0xFF00FF00;}
    if (idx == 2) {rgba = 0xFFFF0000;}
    if (idx == 3) {rgba = 0xFF00FFFF;}
    if (idx == 4) {rgba = 0xFFFF00FF;}
    if (idx == 5) {rgba = 0xFFFFFF00;}
    return rgba;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_generate_sphere(
    pnanovdb_buf_t buf,
    pnanovdb_address_t addr_max,
    float voxel_size,
    float radius
)
{
    float voxel_size_inv = 1.f / voxel_size;
    float voxel_radius = radius * voxel_size_inv;

    pnanovdb_address_t addr_end = pnanovdb_address_null();
    pnanovdb_uint64_t node_idx = 0u;

    // allocate grid, tree, root
    pnanovdb_uint32_t empty_grid_size = pnanovdb_node2_empty_grid_size();
    addr_end = pnanovdb_address_offset(addr_end, empty_grid_size);

    pnanovdb_node2_init_grid(buf, empty_grid_size, voxel_size);

    pnanovdb_grid_handle_t grid = {};
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_node2_handle_t root = {pnanovdb_uint32_t(pnanovdb_tree_get_root(buf, tree).address.byte_offset >> 3u)};

    // allocate node_idx for root
    pnanovdb_node2_write(buf, root.idx64 + pnanovdb_node2_off_node_idx, node_idx);
    node_idx++;

    // Note: this assumes fast root, which is reasonable because sphere is origin centered.
    pnanovdb_coord_t root_ijk = {0, 0, 0};
    pnanovdb_uint32_t root_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT];

    pnanovdb_uint32_t upper_count = 0u;
    pnanovdb_uint32_t lower_count = 0u;
    pnanovdb_uint32_t leaf_count = 0u;

    // allocate breadth first
    for (pnanovdb_uint32_t depth = 0u; depth < 4u; depth++)
    {
        if (depth == 0u)
        {
            pnanovdb_upper_handle_t upper = {addr_end};
            pnanovdb_tree_set_first_upper(buf, tree, upper);
        }
        else if (depth == 1u)
        {
            pnanovdb_lower_handle_t lower = {addr_end};
            pnanovdb_tree_set_first_lower(buf, tree, lower);
        }
        else if (depth == 2u)
        {
            pnanovdb_leaf_handle_t leaf = {addr_end};
            pnanovdb_tree_set_first_leaf(buf, tree, leaf);
        }
        for (pnanovdb_uint32_t root_n = 0u; root_n < root_fanout; root_n++)
        {
            pnanovdb_coord_t upper_ijk = {};
            pnanovdb_node2_handle_t upper = pnanovdb_node2_generate_sphere_node(
                buf,
                addr_max,
                PNANOVDB_NODE2_TYPE_ROOT,
                PNANOVDB_NODE2_TYPE_UPPER,
                voxel_radius,
                root,
                root_ijk,
                root_n,
                PNANOVDB_REF(upper_ijk),
                PNANOVDB_REF(addr_end),
                PNANOVDB_REF(node_idx)
            );
            if (depth == 0u && upper.idx64 != 0u)
            {
                upper_count++;
            }
            if (depth >= 1u && upper.idx64 != 0u)
            {
                pnanovdb_uint32_t upper_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER];
                for (pnanovdb_uint32_t upper_n = 0u; upper_n < upper_fanout; upper_n++)
                {
                    pnanovdb_coord_t lower_ijk = {};
                    pnanovdb_node2_handle_t lower = pnanovdb_node2_generate_sphere_node(
                        buf,
                        addr_max,
                        PNANOVDB_NODE2_TYPE_UPPER,
                        PNANOVDB_NODE2_TYPE_LOWER,
                        voxel_radius,
                        upper,
                        upper_ijk,
                        upper_n,
                        PNANOVDB_REF(lower_ijk),
                        PNANOVDB_REF(addr_end),
                        PNANOVDB_REF(node_idx)
                    );
                    if (depth == 1u && lower.idx64 != 0u)
                    {
                        lower_count++;
                    }
                    if (depth >= 2u && lower.idx64 != 0u)
                    {
                        pnanovdb_uint32_t lower_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER];
                        for (pnanovdb_uint32_t lower_n = 0u; lower_n < lower_fanout; lower_n++)
                        {
                            pnanovdb_coord_t leaf_ijk = {};
                            pnanovdb_node2_handle_t leaf = pnanovdb_node2_generate_sphere_node(
                                buf,
                                addr_max,
                                PNANOVDB_NODE2_TYPE_LOWER,
                                PNANOVDB_NODE2_TYPE_LEAF,
                                voxel_radius,
                                lower,
                                lower_ijk,
                                lower_n,
                                PNANOVDB_REF(leaf_ijk),
                                PNANOVDB_REF(addr_end),
                                PNANOVDB_REF(node_idx)
                            );
                            if (depth == 2u && leaf.idx64 != 0u)
                            {
                                leaf_count++;
                            }
                            if (depth >= 3u && leaf.idx64 != 0u)
                            {
                                pnanovdb_uint32_t leaf_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LEAF];
                                for (pnanovdb_uint32_t leaf_n = 0u; leaf_n < leaf_fanout; leaf_n++)
                                {
                                    pnanovdb_coord_t voxel_ijk = {};
                                    pnanovdb_node2_generate_sphere_node(
                                        buf,
                                        addr_max,
                                        PNANOVDB_NODE2_TYPE_LEAF,
                                        PNANOVDB_NODE2_TYPE_LEAF,
                                        voxel_radius,
                                        leaf,
                                        leaf_ijk,
                                        leaf_n,
                                        PNANOVDB_REF(voxel_ijk),
                                        PNANOVDB_REF(addr_end),
                                        PNANOVDB_REF(node_idx)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // finalize tree node counts
    pnanovdb_tree_set_node_count_upper(buf, tree, upper_count);
    pnanovdb_tree_set_node_count_lower(buf, tree, lower_count);
    pnanovdb_tree_set_node_count_leaf(buf, tree, leaf_count);
    // todo: set tile counts

    // generically walk tree, compute total active values
    pnanovdb_uint64_t value_count = pnanovdb_uint32_as_uint64_low(1u);

    pnanovdb_node2_write(buf, root.idx64 + pnanovdb_node2_off_value_idx, value_count);
    value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_active_values_word_count(buf, root, PNANOVDB_NODE2_TYPE_ROOT));

    root_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT];
    for (pnanovdb_uint32_t root_n = 0u; root_n < root_fanout; root_n++)
    {
        pnanovdb_node2_handle_t upper = pnanovdb_node2_get_child(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
        if (upper.idx64 == 0u)
        {
            continue;
        }

        pnanovdb_node2_write(buf, upper.idx64 + pnanovdb_node2_off_value_idx, value_count);
        value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_active_values_word_count(buf, upper, PNANOVDB_NODE2_TYPE_UPPER));

        pnanovdb_uint32_t node2_upper_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER];
        for (pnanovdb_uint32_t upper_n = 0u; upper_n < node2_upper_fanout; upper_n++)
        {
            pnanovdb_node2_handle_t lower = pnanovdb_node2_get_child(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);
            if (lower.idx64 == 0u)
            {
                continue;
            }

            pnanovdb_node2_write(buf, lower.idx64 + pnanovdb_node2_off_value_idx, value_count);
            value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_active_values_word_count(buf, lower, PNANOVDB_NODE2_TYPE_LOWER));

            pnanovdb_uint32_t node2_lower_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER];
            for (pnanovdb_uint32_t lower_n = 0u; lower_n < node2_lower_fanout; lower_n++)
            {
                pnanovdb_node2_handle_t leaf = pnanovdb_node2_get_child(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                if (leaf.idx64 == 0u)
                {
                    continue;
                }

                pnanovdb_node2_write(buf, leaf.idx64 + pnanovdb_node2_off_value_idx, value_count);
                value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_active_values_word_count(buf, leaf, PNANOVDB_NODE2_TYPE_LEAF));
            }
        }
    }

    // allocate blindmetadata header for values
    pnanovdb_gridblindmetadata_handle_t metadata = {addr_end};
    pnanovdb_uint32_t metadata_size = PNANOVDB_GRIDBLINDMETADATA_SIZE + PNANOVDB_GRIDBLINDMETADATA_SIZE;
    addr_end = pnanovdb_address_offset(addr_end, metadata_size);

    // link headers to grid
    pnanovdb_grid_set_first_gridblindmetadata(buf, grid, metadata);
    pnanovdb_grid_set_blind_metadata_count(buf, grid, 2u);

    // allocate coords for root bbox
    pnanovdb_address_t bboxes = {addr_end};
    pnanovdb_uint64_t bboxes_size = pnanovdb_node2_uint64_align32(6u * 4u);
    addr_end = pnanovdb_address_offset64(addr_end, bboxes_size);

    // allocate floats for active values, bits for inactive values
    pnanovdb_address_t values = {addr_end};
    pnanovdb_uint64_t values_size = pnanovdb_node2_uint64_align32(value_count * 4u);
    addr_end = pnanovdb_address_offset64(addr_end, values_size);

    // bbox header
    pnanovdb_gridblindmetadata_set_data_offset(buf, metadata, pnanovdb_address_diff(bboxes, metadata.address));
    pnanovdb_gridblindmetadata_set_value_count(buf, metadata, 2u);
    pnanovdb_gridblindmetadata_set_value_size(buf, metadata, 12u);
    pnanovdb_gridblindmetadata_set_semantic(buf, metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_class(buf, metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_type(buf, metadata, PNANOVDB_GRID_TYPE_VEC3F);
    pnanovdb_gridblindmetadata_set_name(buf, metadata, 0u, 0u);

    // values header
    metadata.address = pnanovdb_address_offset(metadata.address, PNANOVDB_GRIDBLINDMETADATA_SIZE);
    pnanovdb_gridblindmetadata_set_data_offset(buf, metadata, pnanovdb_address_diff(values, metadata.address));
    pnanovdb_gridblindmetadata_set_value_count(buf, metadata, value_count);
    pnanovdb_gridblindmetadata_set_value_size(buf, metadata, 4u);
    pnanovdb_gridblindmetadata_set_semantic(buf, metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_class(buf, metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_type(buf, metadata, PNANOVDB_GRID_TYPE_RGBA8);
    pnanovdb_gridblindmetadata_set_name(buf, metadata, 0u, 0u);

    // write root bbox
    pnanovdb_coord_t root_bbox_min = {
        pnanovdb_float_to_int32(pnanovdb_floor(-voxel_radius)),
        pnanovdb_float_to_int32(pnanovdb_floor(-voxel_radius)),
        pnanovdb_float_to_int32(pnanovdb_floor(-voxel_radius))
    };
    pnanovdb_coord_t root_bbox_max = {
        pnanovdb_float_to_int32(-pnanovdb_floor(-voxel_radius)),
        pnanovdb_float_to_int32(-pnanovdb_floor(-voxel_radius)),
        pnanovdb_float_to_int32(-pnanovdb_floor(-voxel_radius))
    };
    pnanovdb_write_coord(buf, pnanovdb_address_offset(bboxes, 0u), PNANOVDB_REF(root_bbox_min));
    pnanovdb_write_coord(buf, pnanovdb_address_offset(bboxes, 12u), PNANOVDB_REF(root_bbox_max));

    printf("sphere value_count(%llu)\n", (unsigned long long int)value_count);

    // finalize grid size
    pnanovdb_grid_set_grid_size(buf, grid, pnanovdb_address_diff(addr_end, grid.address));

    // write values
    pnanovdb_node2_write_uint32_index(buf, values, pnanovdb_uint32_as_uint64_low(0u), 0x00000000);

    root_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT];
    for (pnanovdb_uint32_t root_n = 0u; root_n < root_fanout; root_n++)
    {
        if (pnanovdb_node2_get_value_mask_bit(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n))
        {
            pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n, PNANOVDB_FALSE, pnanovdb_address_null(), pnanovdb_address_null());
            pnanovdb_uint32_t value_level = 0u;
            pnanovdb_node2_adaptive_n(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n, PNANOVDB_REF(value_level));
            pnanovdb_node2_write_uint32_index(buf, values, value_idx, pnanovdb_node2_value_level_to_rgba(value_level));
        }
        pnanovdb_node2_handle_t upper = pnanovdb_node2_get_child(buf, root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
        if (upper.idx64 == 0u)
        {
            continue;
        }

        pnanovdb_uint32_t node2_upper_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER];
        for (pnanovdb_uint32_t upper_n = 0u; upper_n < node2_upper_fanout; upper_n++)
        {
            if (pnanovdb_node2_get_value_mask_bit(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n))
            {
                pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n, PNANOVDB_FALSE, pnanovdb_address_null(), pnanovdb_address_null());
                pnanovdb_uint32_t value_level = 0u;
                pnanovdb_node2_adaptive_n(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n, PNANOVDB_REF(value_level));
                pnanovdb_node2_write_uint32_index(buf, values, value_idx, pnanovdb_node2_value_level_to_rgba(value_level));
            }
            pnanovdb_node2_handle_t lower = pnanovdb_node2_get_child(buf, upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);
            if (lower.idx64 == 0u)
            {
                continue;
            }

            pnanovdb_uint32_t node2_lower_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER];
            for (pnanovdb_uint32_t lower_n = 0u; lower_n < node2_lower_fanout; lower_n++)
            {
                if (pnanovdb_node2_get_value_mask_bit(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n))
                {
                    pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n, PNANOVDB_FALSE, pnanovdb_address_null(), pnanovdb_address_null());
                    pnanovdb_uint32_t value_level = 0u;
                    pnanovdb_node2_adaptive_n(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n, PNANOVDB_REF(value_level));
                    pnanovdb_node2_write_uint32_index(buf, values, value_idx, pnanovdb_node2_value_level_to_rgba(value_level));
                }
                pnanovdb_node2_handle_t leaf = pnanovdb_node2_get_child(buf, lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                if (leaf.idx64 == 0u)
                {
                    continue;
                }

                pnanovdb_uint32_t node2_leaf_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LEAF];
                for (pnanovdb_uint32_t leaf_n = 0u; leaf_n < node2_leaf_fanout; leaf_n++)
                {
                    if (pnanovdb_node2_get_value_mask_bit(buf, leaf, PNANOVDB_NODE2_TYPE_LEAF, leaf_n))
                    {
                        pnanovdb_uint64_t value_idx = pnanovdb_node2_get_value_index(buf, leaf, PNANOVDB_NODE2_TYPE_LEAF, leaf_n, PNANOVDB_FALSE, pnanovdb_address_null(), pnanovdb_address_null());
                        pnanovdb_uint32_t value_level = 0u;
                        pnanovdb_node2_adaptive_n(buf, leaf, PNANOVDB_NODE2_TYPE_LEAF, leaf_n, PNANOVDB_REF(value_level));
                        pnanovdb_node2_write_uint32_index(buf, values, value_idx, pnanovdb_node2_value_level_to_rgba(value_level));
                    }
                }
            }
        }
    }
}

#endif
