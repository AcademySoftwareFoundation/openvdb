// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/raster/Node2Cpp.h

    \author Andrew Reidmeyer

    \brief
*/

#ifndef NANOVDB_NODE2CPP_H_HAS_BEEN_INCLUDED
#define NANOVDB_NODE2CPP_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Util.h>

namespace nanovdb {// =================================================================

template<uint32_t fanout1D>
struct node2_value_t
{
    uint64_t node_idx;                                  // index for per node sidecar values
    uint64_t value_idx;                                 // index for per voxel sidecar values
    uint64_t value_level[fanout1D / 128];               // pow2 voxel size, 0 is normal VDB behavior
    uint16_t value_mask_prefix_sum[fanout1D / 64u];     // prefix sum of value mask
    uint64_t value_mask[fanout1D / 64u];                // active value mask
};
template<uint32_t fanout1D>
struct node2_child_t : public node2_value_t<fanout1D>
{
    uint16_t child_mask_prefix_sum[fanout1D / 64u];     // prefix sum of child mask
    uint64_t child_mask[fanout1D / 64u];                // active child mask
};
template<uint32_t fanout1D>
struct node2_key_t : public node2_child_t<fanout1D>
{
    uint16_t key_mask_prefix_sum[fanout1D / 64u];       // prefix sum of child mask
    uint64_t key_mask[fanout1D / 64u];                  // active child mask
    uint64_t key_range[2u];                             // key min/max in 3D space
};
// children_and_keys follows end of node2_child_t or node2_key_t, array is dynamically sized
// uint64_t children_and_keys[2u * fanout1D];          // byte offsets to child node2 and keys

// offsets in 64-bit words
static const uint32_t node2_off_node_idx = 0u;
static const uint32_t node2_off_value_idx = 1u;
static const uint32_t node2_off_value_level = 2u;
static const uint32_t node2_off_value_mask_prefix_sum[4u] = { 258, 258, 34, 6 };
static const uint32_t node2_off_value_mask[4u] = { 258 + 128, 258 + 128, 34 + 16, 6 + 2 };
static const uint32_t node2_off_child_mask_prefix_sum[4u] = { 898, 898, 114, 6};
static const uint32_t node2_off_child_mask[4u] = { 898 + 128, 898 + 128, 114 + 16, 6 + 2 };
static const uint32_t node2_off_key_mask_prefix_sum[4u] = { 1538, 898, 114, 6 };
static const uint32_t node2_off_key_mask[4u] = { 1538 + 128, 898 + 128, 114 + 16, 6 + 2 };
static const uint32_t node2_off_key_range = 2178;
static const uint32_t node2_off_children[4u] = { 2180, 1538, 194, 16 };

static const uint32_t node2_prefix_sum_off_mask[4u] = {128, 128, 16, 2};

static const uint32_t node2_max_size[4] = {541728, 274464, 34336, 128};

// masks for each level
static const uint32_t node2_ijk_mask[4u] = {0x1FFFF, 0xFFF, 0x7F, 0x7};
static const uint32_t node2_fanout_bits2[4u] = {10, 10, 8, 6};
static const uint32_t node2_fanout_bits[4u] = {5, 5, 4, 3};
static const uint32_t node2_tiledim_bits[4u] = {12, 7, 3, 0};
static const uint32_t node2_fanout_1d[4u] = {32768, 32768, 4096, 512};

static const uint32_t node2_level_mask_i[4u] = {0x7800, 0x7800, 0xE00, 0x180};
static const uint32_t node2_level_mask_j[4u] = {0x03C0, 0x03C0, 0x0E0, 0x030};
static const uint32_t node2_level_mask_k[4u] = {0x001E, 0x001E, 0x00E, 0x006};

static const uint64_t node2_end_key = 1llu << 63u;

uint32_t node2_adaptive_n(uint64_t* buf, uint32_t node, uint32_t node_type, uint32_t node_n)
{
    // shift to 2x2x2 smaller index space
    uint32_t level_idx =
        ((node_n & node2_level_mask_i[node_type]) >> 3) |
        ((node_n & node2_level_mask_j[node_type]) >> 2) |
        ((node_n & node2_level_mask_k[node_type]) >> 1);
    // fetch 4-bit level
    uint32_t level = uint32_t(
        buf[node + node2_off_value_level + (level_idx >> 4u)] >>
        ((level_idx << 2u) & 63u)) & 15;
    // clear bits to apply level
    uint32_t level_mask = (1u << level) - 1u;
    uint32_t level_mask_3d = (level_mask << node2_fanout_bits2[node_type]) |
        (level_mask << node2_fanout_bits[node_type]) | level_mask;
    node_n = node_n & ~level_mask_3d;
    return node_n;
}

uint32_t node2_mask_n_to_idx(uint64_t* buf, uint32_t prefix_sum_idx64, uint32_t node_type, uint32_t node_n, bool* out_is_active)
{
    uint64_t mask = buf[prefix_sum_idx64 + node2_prefix_sum_off_mask[node_type] + (node_n >> 6u)];
    uint32_t prefix_sum = uint32_t(buf[prefix_sum_idx64 + (node_n >> 8u)] >> ((node_n >> 2u) & 0x30)) & 0xFFFF;
    uint64_t cmp_mask = 1llu << (node_n & 63u);
    uint32_t idx = util::countOn(mask & (cmp_mask - 1u)) + (node_n >= 64u ? prefix_sum : 0u);
    *out_is_active = (mask & cmp_mask) != 0llu;
    return idx;
}

void node2_find_node_pass(uint64_t* buf, uint32_t node, int32_t root_n_off, Coord ijk,
    uint32_t* out_node, uint32_t* out_node_type, uint32_t* out_node_n, bool* out_key_is_match)
{
    *out_key_is_match = true;
    // find finest node in tree
    uint32_t node_type = 0u;
    uint32_t node_n = 0u;
    for (; node_type < 4u; node_type++)
    {
        node_n =
            (((ijk.x() & node2_ijk_mask[node_type]) >> node2_tiledim_bits[node_type]) << node2_fanout_bits2[node_type]) |
            (((ijk.y() & node2_ijk_mask[node_type]) >> node2_tiledim_bits[node_type]) << node2_fanout_bits[node_type]) |
            ((ijk.z() & node2_ijk_mask[node_type]) >> node2_tiledim_bits[node_type]);
        node_n = node2_adaptive_n(buf, node, node_type, node_n);
        if (node_type == 0u)
        {
            node_n += root_n_off;
            // verify key match
            uint64_t key =
                uint64_t(uint32_t(ijk.z()) >> 12) |
                (uint64_t(uint32_t(ijk.y()) >> 12) << 21) |
                (uint64_t(uint32_t(ijk.x()) >> 12) << 42);
            uint32_t child_count = uint32_t(buf[node + node2_off_child_mask_prefix_sum[node_type]]) & 0xFFFF;
            bool key_active;
            uint32_t key_idx = node2_mask_n_to_idx(buf, node + node2_off_key_mask_prefix_sum[node_type], node_type, node_n, &key_active);
            uint64_t cmp_key = buf[node + node2_off_children[node_type] + child_count + key_idx];
            *out_key_is_match = !key_active || (key_active && (key == cmp_key || node2_end_key == cmp_key));
        }
        if (node_type == 3u)
        {
            break;
        }

        bool is_active;
        uint32_t child_idx = node2_mask_n_to_idx(buf, node + node2_off_child_mask_prefix_sum[node_type], node_type, node_n, &is_active);
        uint32_t child = 0u;
        if (is_active)
        {
            child = uint32_t(buf[node + node2_off_children[node_type] + child_idx] >> 3u);
        }
        if (child == 0u)
        {
            break;
        }
        // advance to next node
        node = node + child;
    }
    *out_node = node;
    *out_node_type = node_type;
    *out_node_n = node_n;
}

void node2_find_node(uint64_t* buf, uint32_t root, Coord ijk, uint32_t* out_node, uint32_t* out_node_type, uint32_t* out_node_n)
{
    bool key_is_match;
    node2_find_node_pass(buf, root, 0, ijk, out_node, out_node_type, out_node_n, &key_is_match);
    if (!key_is_match)
    {
        uint32_t node_n =
            (((ijk.x() & node2_ijk_mask[0u]) >> node2_tiledim_bits[0u]) << node2_fanout_bits2[0u]) |
            (((ijk.y() & node2_ijk_mask[0u]) >> node2_tiledim_bits[0u]) << node2_fanout_bits[0u]) |
            ((ijk.z() & node2_ijk_mask[0u]) >> node2_tiledim_bits[0u]);
        node_n = node2_adaptive_n(buf, root, 0u, node_n);
        // search for matching key
        uint64_t key =
            uint64_t(uint32_t(ijk.z()) >> 12) |
            (uint64_t(uint32_t(ijk.y()) >> 12) << 21) |
            (uint64_t(uint32_t(ijk.x()) >> 12) << 42);
        int n_off_match = 0;
        uint32_t child_count = uint32_t(buf[root + node2_off_child_mask_prefix_sum[0u]]) & 0xFFFF;
        for (uint32_t n_off = 1u; n_off < 32768u; n_off++)
        {
            uint32_t cmp_n = (node_n + n_off) & (32767u);
            bool key_active;
            uint32_t key_idx = node2_mask_n_to_idx(buf, root + node2_off_key_mask_prefix_sum[0u], 0u, cmp_n, &key_active);
            if (key_active)
            {
                uint64_t cmp_key = buf[root + node2_off_children[0u] + child_count + key_idx];
                if (key == cmp_key || node2_end_key == cmp_key)
                {
                    n_off_match = int32_t(cmp_n - node_n);
                    break;
                }
            }
        }
        node2_find_node_pass(buf, root, n_off_match, ijk, out_node, out_node_type, out_node_n, &key_is_match);
    }
}

uint64_t node2_get_value_index(uint64_t* buf, uint32_t node, Coord ijk, bool is_levelset, uint32_t first_node_inactive_idx, uint32_t first_inactive_value_idx)
{
    pnanovdb_uint32_t node_type = 0u;
    pnanovdb_uint32_t node_n = 0u;
    node2_find_node(buf, node, ijk, &node, &node_type, &node_n);

    // compute node local active value index
    bool is_active;
    uint32_t local_value_idx = node2_mask_n_to_idx(buf, node + node2_off_value_mask_prefix_sum[node_type], node_type, node_n, &is_active);

    // compute global value index
    uint64_t value_idx = 0u;
    if (is_active)
    {
        value_idx = buf[node + node2_off_value_idx] + local_value_idx;
    }
    else if (is_levelset)
    {
        // flip node_value_idx for inactive
        local_value_idx = node_n - local_value_idx;

        pnanovdb_uint64_t node_idx = buf[node + pnanovdb_node2_off_node_idx];
        pnanovdb_uint64_t inactive_idx = buf[first_node_inactive_idx + node_idx];

        // compute single bit address and load, assuming 1-bit inactive value
        pnanovdb_uint64_t bit_idx = (first_inactive_value_idx << 6u) + inactive_idx + local_value_idx;
        pnanovdb_uint64_t val_raw = buf[(bit_idx >> 6u)];
        value_idx = (val_raw >> (bit_idx & 63u)) & 1u;
    }
    return value_idx;
}

} // namespace nanovdb ===================================================================

#endif
