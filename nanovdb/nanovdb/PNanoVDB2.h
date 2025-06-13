
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/PNanoVDB2.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
            of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PNANOVDB2_H_HAS_BEEN_INCLUDED
#define NANOVDB_PNANOVDB2_H_HAS_BEEN_INCLUDED

#if defined(PNANOVDB_C)
#include "nanovdb/PNanoVDB.h"
#endif

#define PNANOVDB_NODE2_TEST_WITH_ACCESSOR 1
#define PNANOVDB_NODE2_TEST_X_DEPTH 32

// ----------------------------- core structures ------------------------------

struct pnanovdb_node2_t
{
    pnanovdb_uint64_t node_idx;                      // enables any per node values to be indexed in sidecar array
    pnanovdb_uint64_t value_idx;                     // start of values for this node
    //pnanovdb_uint64_t value_level[fanout_count / 128];
    //pnanovdb_uint64_t value_mask_prefix_sum[fanout_count / 256];
    //pnanovdb_uint64_t value_mask[fanout_count / 64];
    //pnanovdb_uint64_t child_mask_prefix_sum[fanout_count / 256];
    //pnanovdb_uint64_t child_mask[fanout_count / 64];
    //pnanovdb_uint64_t key_mask_prefix_sum[fanout_count / 256];
    //pnanovdb_uint64_t key_mask[fanout_count / 64];
    //pnanovdb_uint64_t key_range[2u];
    //pnanovdb_uint64_t children_and_keys[2u * fanout_count];    // worst case size
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node2_t)
struct pnanovdb_node2_handle_t { pnanovdb_uint32_t idx64; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node2_handle_t)

// offsets in 64-bit words
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_node_idx = 0;
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_value_idx = 1;
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_value_level = 2;
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_value_mask_prefix_sum[4u] = { 258, 258, 34, 6 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_value_mask[4u] = { 258 + 128, 258 + 128, 34 + 16, 6 + 2 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_child_mask_prefix_sum[4u] = { 898, 898, 114, 6 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_child_mask[4u] = { 898 + 128, 898 + 128, 114 + 16, 6 + 2 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_key_mask_prefix_sum[4u] = { 1538, 898, 114, 6 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_key_mask[4u] = { 1538 + 128, 898 + 128, 114 + 16, 6 + 2 };
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_key_range = 2178;
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_off_children[4u] = { 2180, 1538, 194, 16 };

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_prefix_sum_off_mask[4u] = {128, 128, 16, 2};

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_max_size[4] = {541728, 274464, 34336, 128};

// node types
#define PNANOVDB_NODE2_TYPE_ROOT 0
#define PNANOVDB_NODE2_TYPE_UPPER 1
#define PNANOVDB_NODE2_TYPE_LOWER 2
#define PNANOVDB_NODE2_TYPE_LEAF 3

// masks for each node type
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_ijk_mask[4u] = {0x1FFFF, 0xFFF, 0x7F, 0x7};
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_fanout_bits2[4u] = {10, 10, 8, 6};
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_fanout_bits[4u] = {5, 5, 4, 3};
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_tiledim_bits[4u] = {12, 7, 3, 0};
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_fanout_1d[4u] = {32768, 32768, 4096, 512};

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_level_mask_i[4u] = {0x7800, 0x7800, 0xE00, 0x180};
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_level_mask_j[4u] = {0x03C0, 0x03C0, 0x0E0, 0x030};
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_node2_level_mask_k[4u] = {0x001E, 0x001E, 0x00E, 0x006};

PNANOVDB_STATIC_CONST pnanovdb_uint64_t pnanovdb_node2_end_key = 1llu << 63u;

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_node2_read(pnanovdb_buf_t buf, pnanovdb_uint32_t index)
{
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset64_product(pnanovdb_address_null(), index, 8u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_node2_write(pnanovdb_buf_t buf, pnanovdb_uint32_t index, pnanovdb_uint64_t val)
{
    pnanovdb_write_uint64(buf, pnanovdb_address_offset64_product(pnanovdb_address_null(), index, 8u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_coord_to_n(pnanovdb_uint32_t node_type, pnanovdb_coord_t ijk)
{
    pnanovdb_uint32_t node_n =
        (((ijk.x & pnanovdb_node2_ijk_mask[node_type]) >> pnanovdb_node2_tiledim_bits[node_type]) << pnanovdb_node2_fanout_bits2[node_type]) |
        (((ijk.y & pnanovdb_node2_ijk_mask[node_type]) >> pnanovdb_node2_tiledim_bits[node_type]) << pnanovdb_node2_fanout_bits[node_type]) |
        ((ijk.z & pnanovdb_node2_ijk_mask[node_type]) >> pnanovdb_node2_tiledim_bits[node_type]);
    return node_n;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_adaptive_n(
    pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_uint32_t node_n, PNANOVDB_INOUT(pnanovdb_uint32_t) out_level)
{
    // shift to 2x2x2 smaller index space
    pnanovdb_uint32_t level_idx =
        ((node_n & pnanovdb_node2_level_mask_i[node_type]) >> 3) |
        ((node_n & pnanovdb_node2_level_mask_j[node_type]) >> 2) |
        ((node_n & pnanovdb_node2_level_mask_k[node_type]) >> 1);
    // fetch 4-bit level
    pnanovdb_uint32_t level = pnanovdb_uint32_t(
        pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_level + (level_idx >> 4u)) >>
        ((level_idx << 2u) & 63u)) & 15;
    // clear bits to apply level
    pnanovdb_uint32_t level_mask = (1u << level) - 1u;
    pnanovdb_uint32_t level_mask_3d = (level_mask << pnanovdb_node2_fanout_bits2[node_type]) |
        (level_mask << pnanovdb_node2_fanout_bits[node_type]) | level_mask;
    node_n = node_n & ~level_mask_3d;
    PNANOVDB_DEREF(out_level) = level + pnanovdb_node2_tiledim_bits[node_type];
    return node_n;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_mask_n_to_idx(
    pnanovdb_buf_t buf, pnanovdb_uint32_t prefix_sum_idx64, pnanovdb_uint32_t node_type, pnanovdb_uint32_t node_n, PNANOVDB_INOUT(pnanovdb_bool_t) out_is_active)
{
    pnanovdb_uint64_t mask = pnanovdb_node2_read(buf, prefix_sum_idx64 + pnanovdb_node2_prefix_sum_off_mask[node_type] + (node_n >> 6u));
    pnanovdb_uint32_t prefix_sum = pnanovdb_uint32_t(pnanovdb_node2_read(buf, prefix_sum_idx64 + (node_n >> 8u)) >> ((node_n >> 2u) & 0x30)) & 0xFFFF;
    pnanovdb_uint64_t cmp_mask = 1llu << (node_n & 63u);
    pnanovdb_uint32_t idx = pnanovdb_uint64_countbits(mask & (cmp_mask - 1u)) + (node_n >= 64u ? prefix_sum : 0u);
    PNANOVDB_DEREF(out_is_active) = (mask & cmp_mask) != 0llu;
    return idx;
}

PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_node2_find_n_offset_to_key(
    pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_uint32_t n_off, pnanovdb_uint32_t node_n, pnanovdb_coord_t ijk)
{
    pnanovdb_int32_t n_off_match = 0;
    if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
    {
        // keyed root, searches hash table to verify key matches, for incredibly sparse cases
        pnanovdb_uint64_t key =
            pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.z) >> 12) |
            (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.y) >> 12) << 21) |
            (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.x) >> 12) << 42);
        pnanovdb_uint32_t child_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type])) & 0xFFFF;
        for (; n_off < 32768u; n_off++)
        {
            pnanovdb_uint32_t cmp_n = (node_n + n_off) & (32767u);
            pnanovdb_bool_t key_active;
            pnanovdb_uint32_t key_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type], node_type, cmp_n, PNANOVDB_REF(key_active));
            if (key_active)
            {
                pnanovdb_uint64_t cmp_key = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_count + key_idx);
                if (key == cmp_key || pnanovdb_node2_end_key == cmp_key)
                {
                    n_off_match = pnanovdb_int32_t(cmp_n - node_n);
                    break;
                }
            }
        }
    }
    return n_off_match;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_coord_to_n_keyed(
    pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_coord_t ijk)
{
    pnanovdb_uint32_t node_n = pnanovdb_node2_coord_to_n(node_type, ijk);
    pnanovdb_uint32_t level;
    node_n = pnanovdb_node2_adaptive_n(buf, node, node_type, node_n, PNANOVDB_REF(level));
    node_n += pnanovdb_node2_find_n_offset_to_key(buf, node, node_type, 0u, node_n, ijk);
    return node_n;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_find_node_pass(
    pnanovdb_buf_t buf,
    pnanovdb_node2_handle_t node,
    pnanovdb_int32_t root_n_off,
    PNANOVDB_INOUT(pnanovdb_node2_handle_t) out_node,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_node_type,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_node_n,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_level,
    PNANOVDB_INOUT(pnanovdb_bool_t) key_is_match,
    pnanovdb_coord_t ijk)
{
    PNANOVDB_DEREF(key_is_match) = PNANOVDB_TRUE;
    // find finest node in tree
    pnanovdb_uint32_t node_type = 0u;
    pnanovdb_uint32_t node_n = 0u;
    for (; node_type < 4u; node_type++)
    {
        node_n = pnanovdb_node2_coord_to_n(node_type, ijk);
        node_n = pnanovdb_node2_adaptive_n(buf, node, node_type, node_n, out_level);

        if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
        {
            node_n += root_n_off;
            pnanovdb_uint64_t key =
                pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.z) >> 12) |
                (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.y) >> 12) << 21) |
                (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.x) >> 12) << 42);
            pnanovdb_uint32_t child_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type])) & 0xFFFF;
            pnanovdb_bool_t key_active;
            pnanovdb_uint32_t key_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(key_active));
            pnanovdb_uint64_t cmp_key = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_count + key_idx);
            PNANOVDB_DEREF(key_is_match) = !key_active || (key_active && (key == cmp_key || pnanovdb_node2_end_key == cmp_key));
        }
        if (node_type == PNANOVDB_NODE2_TYPE_LEAF)
        {
            break;
        }

        pnanovdb_bool_t is_active;
        pnanovdb_uint32_t child_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(is_active));
        pnanovdb_uint32_t child = 0u;
        if (is_active)
        {
            child = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_idx) >> 3u);
        }
        if (child == 0u)
        {
            break;
        }
        // advance to next node
        node.idx64 = node.idx64 + child;
    }

    PNANOVDB_DEREF(out_node) = node;
    PNANOVDB_DEREF(out_node_type) = node_type;
    PNANOVDB_DEREF(out_node_n) = node_n;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_find_node(
    pnanovdb_buf_t buf,
    pnanovdb_node2_handle_t root,
    PNANOVDB_INOUT(pnanovdb_node2_handle_t) out_node,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_node_type,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_node_n,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_level,
    pnanovdb_coord_t ijk)
{
    pnanovdb_bool_t key_is_match;
    pnanovdb_node2_find_node_pass(buf, root, 0, out_node, out_node_type, out_node_n, out_level, PNANOVDB_REF(key_is_match), ijk);
    if (!key_is_match)
    {
        pnanovdb_uint32_t node_n = pnanovdb_node2_coord_to_n(PNANOVDB_NODE2_TYPE_ROOT, ijk);
        node_n = pnanovdb_node2_adaptive_n(buf, root, PNANOVDB_NODE2_TYPE_ROOT, node_n, out_level);
        pnanovdb_int32_t n_off = pnanovdb_node2_find_n_offset_to_key(buf, root, PNANOVDB_NODE2_TYPE_ROOT, 1u, node_n, ijk);
        pnanovdb_node2_find_node_pass(buf, root, n_off, out_node, out_node_type, out_node_n, out_level, PNANOVDB_REF(key_is_match), ijk);
    }
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_node2_get_value_index(
    pnanovdb_buf_t buf,
    pnanovdb_node2_handle_t node,
    pnanovdb_uint32_t node_type,
    pnanovdb_uint32_t node_n,
    pnanovdb_bool_t is_levelset,
    pnanovdb_address_t node_inactive_idxs,
    pnanovdb_address_t inactive_value_idxs)
{
    // compute node local active value index
    pnanovdb_bool_t is_active;
    pnanovdb_uint32_t local_value_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(is_active));

    // compute global value index
    pnanovdb_uint64_t value_idx = 0u;
    if (is_active)
    {
        value_idx = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_idx) + local_value_idx;
    }
    else if (is_levelset)
    {
        // flip node_value_idx for inactive
        local_value_idx = node_n - local_value_idx;

        pnanovdb_uint64_t node_idx = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_node_idx);
        pnanovdb_uint64_t inactive_idx = pnanovdb_read_uint64(buf, pnanovdb_address_offset64_product(node_inactive_idxs, node_idx, 8u));

        // compute single bit address and load, assuming 1-bit inactive value
        pnanovdb_uint64_t bit_idx = (inactive_value_idxs.byte_offset << 3u) + inactive_idx + local_value_idx;
        pnanovdb_uint64_t val_raw = pnanovdb_node2_read(buf, pnanovdb_uint32_t(bit_idx >> 6u));
        value_idx = (val_raw >> (bit_idx & 63u)) & 1u;
    }
    return value_idx;
}

struct pnanovdb_node2_levelset_values_t
{
    pnanovdb_address_t values;
    pnanovdb_address_t node_inactive_idxs;
    pnanovdb_address_t inactive_value_idxs;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node2_levelset_values_t)

struct pnanovdb_node2_accessor_t
{
    pnanovdb_coord_t key;
    pnanovdb_uint32_t node_n;
    pnanovdb_uint32_t level;
    pnanovdb_uint32_t node_type;
    pnanovdb_node2_handle_t node;
    pnanovdb_node2_handle_t leaf;
    pnanovdb_node2_handle_t lower;
    pnanovdb_node2_handle_t upper;
    pnanovdb_node2_handle_t root;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_node2_accessor_t)

PNANOVDB_FORCE_INLINE void pnanovdb_node2_accessor_init(PNANOVDB_INOUT(pnanovdb_node2_accessor_t) acc, pnanovdb_node2_handle_t root)
{
    PNANOVDB_DEREF(acc).key.x = 0x7FFFFFFF;
    PNANOVDB_DEREF(acc).key.y = 0x7FFFFFFF;
    PNANOVDB_DEREF(acc).key.z = 0x7FFFFFFF;
    PNANOVDB_DEREF(acc).node_n = 0u;
    PNANOVDB_DEREF(acc).level = 0u;
    PNANOVDB_DEREF(acc).leaf.idx64 = 0u;
    PNANOVDB_DEREF(acc).lower.idx64 = 0u;
    PNANOVDB_DEREF(acc).upper.idx64 = 0u;
    PNANOVDB_DEREF(acc).root = root;
}

PNANOVDB_FORCE_INLINE void pnanovdb_accessor_node2_find_node_pass(
    pnanovdb_buf_t buf,
    PNANOVDB_INOUT(pnanovdb_node2_accessor_t) acc,
    pnanovdb_node2_handle_t node,
    pnanovdb_uint32_t node_type,
    pnanovdb_int32_t root_n_off,
    PNANOVDB_INOUT(pnanovdb_bool_t) key_is_match,
    pnanovdb_coord_t ijk)
{
    PNANOVDB_DEREF(key_is_match) = PNANOVDB_TRUE;
    // find finest node in tree
    pnanovdb_uint32_t node_n = 0u;
    for (; node_type < 4u; node_type++)
    {
        node_n = pnanovdb_node2_coord_to_n(node_type, ijk);
        node_n = pnanovdb_node2_adaptive_n(buf, node, node_type, node_n, PNANOVDB_REF(PNANOVDB_DEREF(acc).level));

        if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
        {
            node_n += root_n_off;
            pnanovdb_uint64_t key =
                pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.z) >> 12) |
                (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.y) >> 12) << 21) |
                (pnanovdb_uint64_t(pnanovdb_uint32_t(ijk.x) >> 12) << 42);
            pnanovdb_uint32_t child_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type])) & 0xFFFF;
            pnanovdb_bool_t key_active;
            pnanovdb_uint32_t key_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(key_active));
            pnanovdb_uint64_t cmp_key = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_count + key_idx);
            PNANOVDB_DEREF(key_is_match) = !key_active || (key_active && (key == cmp_key || pnanovdb_node2_end_key == cmp_key));
        }
        if (node_type == PNANOVDB_NODE2_TYPE_LEAF)
        {
            break;
        }

        pnanovdb_bool_t is_active;
        pnanovdb_uint32_t child_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(is_active));
        if (is_active)
        {
            // advance to next node
            node.idx64 = node.idx64 + pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_idx) >> 3u);
            if (node_type == 0u)
            {
                PNANOVDB_DEREF(acc).upper = node;
            }
            if (node_type == 1u)
            {
                PNANOVDB_DEREF(acc).lower = node;
            }
            if (node_type == 2u)
            {
                PNANOVDB_DEREF(acc).leaf = node;
            }
        }
        else
        {
            break;
        }
    }
    PNANOVDB_DEREF(acc).node = node;
    PNANOVDB_DEREF(acc).node_type = node_type;
    PNANOVDB_DEREF(acc).node_n = node_n;
}

PNANOVDB_FORCE_INLINE void pnanovdb_accessor_node2_find_node(
    pnanovdb_buf_t buf,
    PNANOVDB_INOUT(pnanovdb_node2_accessor_t) acc,
    pnanovdb_node2_handle_t node,
    pnanovdb_uint32_t node_type,
    pnanovdb_coord_t ijk)
{
    pnanovdb_bool_t key_is_match;
    pnanovdb_accessor_node2_find_node_pass(buf, acc, node, node_type, 0, PNANOVDB_REF(key_is_match), ijk);
    if (!key_is_match)
    {
        pnanovdb_uint32_t node_n = pnanovdb_node2_coord_to_n(PNANOVDB_NODE2_TYPE_ROOT, ijk);
        node_n = pnanovdb_node2_adaptive_n(buf, PNANOVDB_DEREF(acc).root, PNANOVDB_NODE2_TYPE_ROOT, node_n, PNANOVDB_REF(PNANOVDB_DEREF(acc).level));
        pnanovdb_int32_t n_off = pnanovdb_node2_find_n_offset_to_key(buf, PNANOVDB_DEREF(acc).root, PNANOVDB_NODE2_TYPE_ROOT, 1u, node_n, ijk);
        pnanovdb_accessor_node2_find_node_pass(buf, acc, node, node_type, n_off, PNANOVDB_REF(key_is_match), ijk);
    }
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_accessor_find_node(
    pnanovdb_buf_t buf,
    PNANOVDB_INOUT(pnanovdb_node2_accessor_t) acc,
    PNANOVDB_INOUT(pnanovdb_node2_handle_t) out_node,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_node_type,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_node_n,
    PNANOVDB_INOUT(pnanovdb_uint32_t) out_level,
    pnanovdb_coord_t ijk)
{
    pnanovdb_int32_t dirty = (ijk.x ^ PNANOVDB_DEREF(acc).key.x) | (ijk.y ^ PNANOVDB_DEREF(acc).key.y) | (ijk.z ^ PNANOVDB_DEREF(acc).key.z);
    pnanovdb_uint32_t dirty_mask = (1u << PNANOVDB_DEREF(acc).level) - 1u;
    dirty = dirty & ~dirty_mask;
    if (dirty != 0)
    {
        pnanovdb_node2_handle_t node = PNANOVDB_DEREF(acc).root;
        pnanovdb_uint32_t node_type = 0u;
        if (PNANOVDB_DEREF(acc).leaf.idx64 != 0u && (dirty & ~0x007) == 0)
        {
            node = PNANOVDB_DEREF(acc).leaf;
            node_type = 3u;
        }
        else
        {
            PNANOVDB_DEREF(acc).leaf.idx64 = 0u;
            if (PNANOVDB_DEREF(acc).lower.idx64 != 0u && (dirty & ~0x07F) == 0)
            {
                node = PNANOVDB_DEREF(acc).lower;
                node_type = 2u;
            }
            else
            {
                PNANOVDB_DEREF(acc).lower.idx64 = 0u;
                if (PNANOVDB_DEREF(acc).upper.idx64 != 0u && (dirty & ~0xFFF) == 0)
                {
                    node = PNANOVDB_DEREF(acc).upper;
                    node_type = 1u;
                }
                else
                {
                    PNANOVDB_DEREF(acc).upper.idx64 = 0u;
                }
            }
        }
        pnanovdb_accessor_node2_find_node(buf, acc, node, node_type, ijk);
    }
    PNANOVDB_DEREF(acc).key = ijk;
    PNANOVDB_DEREF(out_node_n) = PNANOVDB_DEREF(acc).node_n;
    PNANOVDB_DEREF(out_level) = PNANOVDB_DEREF(acc).level;
    PNANOVDB_DEREF(out_node_type) = PNANOVDB_DEREF(acc).node_type;
    PNANOVDB_DEREF(out_node) = PNANOVDB_DEREF(acc).node;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node2_get_child_mask_bit(pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_uint32_t node_n)
{
    pnanovdb_uint64_t mask = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask[node_type] + (node_n >> 6u));
    pnanovdb_uint64_t cmp_mask = 1llu << (node_n & 63u);
    return (mask & cmp_mask) != 0llu;
}

PNANOVDB_FORCE_INLINE pnanovdb_node2_handle_t pnanovdb_node2_get_child(pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_uint32_t node_n)
{
    pnanovdb_bool_t is_active;
    pnanovdb_uint32_t child_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(is_active));
    pnanovdb_uint32_t child = 0u;
    if (is_active)
    {
        child = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_idx) >> 3u);
    }
    pnanovdb_node2_handle_t child_handle = {};
    if (child != 0u)
    {
        // advance to next node
        child_handle.idx64 = node.idx64 + child;
    }
    return child_handle;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_node2_get_key(pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_uint32_t node_n)
{
    pnanovdb_uint64_t key = pnanovdb_node2_end_key;
    if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
    {
        pnanovdb_bool_t key_active;
        pnanovdb_uint32_t key_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(key_active));
        if (key_active)
        {
            pnanovdb_uint32_t child_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type])) & 0xFFFF;
            key = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_count + key_idx);
        }
    }
    return key;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_set_child(pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_uint32_t node_n, pnanovdb_node2_handle_t child_node)
{
    pnanovdb_bool_t is_active;
    pnanovdb_uint32_t child_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(is_active));
    if (is_active)
    {
        pnanovdb_node2_write(buf, node.idx64 + pnanovdb_node2_off_children[node_type] + child_idx, pnanovdb_uint64_t(child_node.idx64 - node.idx64) << 3u);
    }
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_node2_get_value_mask_bit(pnanovdb_buf_t buf, pnanovdb_node2_handle_t node, pnanovdb_uint32_t node_type, pnanovdb_uint32_t node_n)
{
    pnanovdb_uint64_t mask = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask[node_type] + (node_n >> 6u));
    pnanovdb_uint64_t cmp_mask = 1llu << (node_n & 63u);
    return (mask & cmp_mask) != 0llu;
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_key_to_coord(pnanovdb_uint64_t key)
{
    pnanovdb_coord_t ijk = {
        pnanovdb_uint32_as_int32(pnanovdb_uint64_to_uint32_lsr(key, 42u) << 12u),
        pnanovdb_uint32_as_int32(pnanovdb_uint64_to_uint32_lsr(key, 21u) << 12u),
        pnanovdb_uint32_as_int32(pnanovdb_uint64_to_uint32_lsr(key,  0u) << 12u)
    };
    return ijk;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_key_range_accum(
    PNANOVDB_INOUT(pnanovdb_uint64_t) key_min, PNANOVDB_INOUT(pnanovdb_uint64_t) key_max, pnanovdb_uint64_t key)
{
    pnanovdb_coord_t kmin = pnanovdb_node2_key_to_coord(PNANOVDB_DEREF(key_min));
    pnanovdb_coord_t kmax = pnanovdb_node2_key_to_coord(PNANOVDB_DEREF(key_max));
    pnanovdb_coord_t k = pnanovdb_node2_key_to_coord(key);
    pnanovdb_coord_t c3;
    kmin.x = k.x < kmin.x ? k.x : kmin.x;
    kmin.y = k.y < kmin.y ? k.y : kmin.y;
    kmin.z = k.z < kmin.z ? k.z : kmin.z;
    kmax.x = k.x > kmax.x ? k.x : kmax.x;
    kmax.y = k.y > kmax.y ? k.y : kmax.y;
    kmax.z = k.z > kmax.z ? k.z : kmax.z;
    PNANOVDB_DEREF(key_min) =
        pnanovdb_uint64_t(pnanovdb_uint32_t(kmin.z) >> 12) |
        (pnanovdb_uint64_t(pnanovdb_uint32_t(kmin.y) >> 12) << 21) |
        (pnanovdb_uint64_t(pnanovdb_uint32_t(kmin.x) >> 12) << 42);
    PNANOVDB_DEREF(key_max) =
        pnanovdb_uint64_t(pnanovdb_uint32_t(kmax.z) >> 12) |
        (pnanovdb_uint64_t(pnanovdb_uint32_t(kmax.y) >> 12) << 21) |
        (pnanovdb_uint64_t(pnanovdb_uint32_t(kmax.x) >> 12) << 42);
}

// ----------------------------- convert ------------------------------

#if defined(PNANOVDB_C)

PNANOVDB_FORCE_INLINE float pnanovdb_node2_read_float_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint64_t index)
{
    return pnanovdb_read_float(buf, pnanovdb_address_offset64_product(addr, index, 4u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_node2_write_float_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint64_t index, float val)
{
    pnanovdb_write_float(buf, pnanovdb_address_offset64_product(addr, index, 4u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_upper_coord_to_lower_coord(pnanovdb_coord_t upper_ijk, pnanovdb_uint32_t upper_n)
{
    pnanovdb_coord_t lower_ijk = {
        upper_ijk.x + (((pnanovdb_uint32_as_int32(upper_n) >> 10) & 31) << 7),
        upper_ijk.y + (((pnanovdb_uint32_as_int32(upper_n) >> 5) & 31) << 7),
        upper_ijk.z + (((pnanovdb_uint32_as_int32(upper_n)) & 31) << 7)
    };
    return lower_ijk;
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_node2_lower_coord_to_leaf_coord(pnanovdb_coord_t lower_ijk, pnanovdb_uint32_t lower_n)
{
    pnanovdb_coord_t leaf_ijk = {
        lower_ijk.x + (((pnanovdb_uint32_as_int32(lower_n) >> 8) & 15) << 3),
        lower_ijk.y + (((pnanovdb_uint32_as_int32(lower_n) >> 4) & 15) << 3),
        lower_ijk.z + (((pnanovdb_uint32_as_int32(lower_n)) & 15) << 3)
    };
    return leaf_ijk;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_compute_mask_prefix_sum(pnanovdb_buf_t buf, pnanovdb_uint32_t mask_idx64, pnanovdb_uint32_t node_type)
{
    pnanovdb_uint32_t mask_word_count = pnanovdb_node2_fanout_1d[node_type] >> 6u;
    pnanovdb_uint64_t value_mask_sum = 0u;
    pnanovdb_uint64_t bit_mask = 0xFFFF;
    for (pnanovdb_uint32_t mask_idx = 0u; mask_idx < mask_word_count; mask_idx++)
    {
        pnanovdb_uint32_t word_idx = (mask_idx + 1u) & (mask_word_count - 1u);

        pnanovdb_uint64_t value_mask = pnanovdb_node2_read(buf, mask_idx64 + pnanovdb_node2_prefix_sum_off_mask[node_type] + mask_idx);

        value_mask_sum += pnanovdb_uint64_countbits(value_mask);

        pnanovdb_uint64_t value_mask_prefix_sum =
            pnanovdb_node2_read(buf, mask_idx64 + (word_idx >> 2u));

        value_mask_prefix_sum &= ~(bit_mask << ((word_idx & 3) << 4u));

        value_mask_prefix_sum |= ((value_mask_sum & bit_mask) << ((word_idx & 3) << 4u));

        pnanovdb_node2_write(buf, mask_idx64 + (word_idx >> 2u), value_mask_prefix_sum);
    }
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_compute_prefix_sums(pnanovdb_buf_t dst_buf, pnanovdb_node2_handle_t dst_node, pnanovdb_uint32_t node_type)
{
    pnanovdb_node2_compute_mask_prefix_sum(dst_buf, dst_node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type], node_type);
    if (node_type != PNANOVDB_NODE2_TYPE_LEAF)
    {
        pnanovdb_node2_compute_mask_prefix_sum(dst_buf, dst_node.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[node_type], node_type);
    }
    if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
    {
        pnanovdb_node2_compute_mask_prefix_sum(dst_buf, dst_node.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[node_type], node_type);
    }
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_memcpy(
    pnanovdb_buf_t dst_buf, pnanovdb_address_t dst_addr,
    pnanovdb_buf_t src_buf, pnanovdb_address_t src_addr,
    pnanovdb_uint64_t num_bytes)
{
    pnanovdb_uint64_t num_words = (num_bytes >> 2u);
    for (pnanovdb_uint64_t word_idx = 0u; word_idx < num_words; word_idx++)
    {
        pnanovdb_uint32_t val = pnanovdb_read_uint32(src_buf, pnanovdb_address_offset_product(src_addr, word_idx, 4u));
        pnanovdb_write_uint32(dst_buf, pnanovdb_address_offset_product(dst_addr, word_idx, 4u), val);
    }
}

PNANOVDB_FORCE_INLINE void pnanovdb_convert_to_node2_value_copy(
    pnanovdb_buf_t buf,
    pnanovdb_node2_handle_t node,
    pnanovdb_uint32_t node_type,
    pnanovdb_coord_t base_ijk,
    pnanovdb_address_t values,
    pnanovdb_address_t node_inactive_idxs,
    pnanovdb_address_t inactive_value_idxs,
    pnanovdb_buf_t src_buf,
    pnanovdb_grid_type_t src_grid_type,
    pnanovdb_root_handle_t src_root
)
{
    pnanovdb_uint32_t dst_node_fanout = pnanovdb_node2_fanout_1d[node_type];
    for (pnanovdb_uint32_t dense_n_l = 0u; dense_n_l < dst_node_fanout; dense_n_l++)
    {
        pnanovdb_uint32_t node_n = dense_n_l;
        //if (pnanovdb_node2_type_is_adaptive(node_type))
        //{
        //    node_n = pnanovdb_node2_compute_adaptive_dense_n(buf, node, node_type, node_n);
        //}

        pnanovdb_bool_t has_child = PNANOVDB_FALSE;
        if (node_type != PNANOVDB_NODE2_TYPE_LEAF)
        {
            has_child = pnanovdb_node2_get_child_mask_bit(buf, node, node_type, node_n);
        }
        if (!has_child)
        {
            pnanovdb_coord_t ijk;
            if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
            {
                pnanovdb_uint64_t key = pnanovdb_node2_get_key(buf, node, node_type, node_n);
                ijk = pnanovdb_node2_key_to_coord(key);
            }
            else
            {
                const pnanovdb_uint32_t tiledim_bits = pnanovdb_node2_tiledim_bits[node_type];
                const pnanovdb_uint32_t fanout_bits = pnanovdb_node2_fanout_bits[node_type];
                ijk = {
                    base_ijk.x + pnanovdb_uint32_as_int32(((node_n >> (fanout_bits + fanout_bits)) & ((1u << fanout_bits) - 1u)) << tiledim_bits),
                    base_ijk.y + pnanovdb_uint32_as_int32(((node_n >> fanout_bits) & ((1u << fanout_bits) - 1u)) << tiledim_bits),
                    base_ijk.z + pnanovdb_uint32_as_int32((node_n & ((1u << fanout_bits) - 1u)) << tiledim_bits)
                };
            }

            pnanovdb_address_t val_addr = pnanovdb_root_get_value_address(src_grid_type, src_buf, src_root, PNANOVDB_REF(ijk));
            if (node_type == PNANOVDB_NODE2_TYPE_ROOT)
            {
                pnanovdb_uint64_t key = pnanovdb_node2_get_key(buf, node, node_type, node_n);
                if (pnanovdb_uint64_is_equal(key, pnanovdb_node2_end_key))
                {
                    val_addr = pnanovdb_root_get_background_address(src_grid_type, src_buf, src_root);
                }
            }
            float val = pnanovdb_read_float(src_buf, val_addr);

            pnanovdb_bool_t is_active;
            pnanovdb_uint32_t node_value_idx = pnanovdb_node2_mask_n_to_idx(buf, node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type], node_type, node_n, PNANOVDB_REF(is_active));

            pnanovdb_uint64_t value_idx = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_idx);
            if (is_active)
            {
                value_idx += node_value_idx;
                pnanovdb_node2_write_float_index(buf, values, value_idx, val);
            }
            else
            {
                // flip node_value_idx for inactive
                node_value_idx = node_n - node_value_idx;

                pnanovdb_uint64_t node_idx = pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_node_idx);
                pnanovdb_uint64_t inactive_idx = pnanovdb_read_uint64(buf, pnanovdb_address_offset64_product(node_inactive_idxs, node_idx, 8u));

                // compute single bit address and load, assuming 32-bit active value, 1-bit inactive value
                pnanovdb_uint64_t bit_idx = (inactive_value_idxs.byte_offset << 3u) + inactive_idx + node_value_idx;
                pnanovdb_uint64_t val_raw = pnanovdb_node2_read(buf, pnanovdb_uint32_t(bit_idx >> 6u));
                val_raw = val_raw & ~(1llu << (bit_idx & 63u));
                if (val < 0.f)
                {
                    val_raw = val_raw | (1llu << (bit_idx & 63u));
                }
                pnanovdb_node2_write(buf, bit_idx >> 6u, val_raw);
            }
        }
    }
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_node2_uint64_align32(pnanovdb_uint64_t v)
{
    return ((v + 31u) >> 5u) << 5u;
}

PNANOVDB_FORCE_INLINE void pnanovdb_node2_memclear(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t size)
{
    for (pnanovdb_uint32_t idx = 0u; idx < size; idx+=4u)
    {
        pnanovdb_write_uint32(buf, pnanovdb_address_offset(addr, idx), 0u);
    }
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_compute_values_word_count(
    pnanovdb_buf_t buf,
    pnanovdb_node2_handle_t node,
    pnanovdb_uint32_t node_type
)
{
    pnanovdb_uint32_t size = 0u;
    pnanovdb_uint32_t active_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type])) & 0xFFFF;
    size += active_count;
    return size;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_node2_compute_inactive_word_count(
    pnanovdb_buf_t buf,
    pnanovdb_node2_handle_t node,
    pnanovdb_uint32_t node_type
)
{
    pnanovdb_uint32_t size = 0u;
    pnanovdb_uint32_t active_count = pnanovdb_uint32_t(pnanovdb_node2_read(buf, node.idx64 + pnanovdb_node2_off_value_mask_prefix_sum[node_type])) & 0xFFFF;
    size += (pnanovdb_node2_fanout_1d[node_type] - active_count);
    return size;
}

PNANOVDB_FORCE_INLINE void pnanovdb_convert_to_node2(
    pnanovdb_buf_t dst_buf,
    pnanovdb_address_t dst_addr_max,
    pnanovdb_buf_t buf,
    pnanovdb_grid_handle_t grid
)
{
    pnanovdb_address_t dst_addr_end = pnanovdb_address_null();
    pnanovdb_uint64_t node_idx = 0u;

    pnanovdb_grid_type_t grid_type = pnanovdb_grid_get_grid_type(buf, grid);
    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);

    // allocate grid, tree
    pnanovdb_grid_handle_t dst_grid = {dst_addr_end};
    pnanovdb_uint32_t dst_grid_size = PNANOVDB_GRID_SIZE + PNANOVDB_TREE_SIZE;
    dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_grid_size);

    // copy grid header, but change type
    pnanovdb_node2_memcpy(dst_buf, dst_grid.address, buf, grid.address, PNANOVDB_GRID_SIZE);
    pnanovdb_grid_set_grid_type(dst_buf, dst_grid, PNANOVDB_GRID_TYPE_NODE2);

    // invalidate tree v1
    pnanovdb_tree_handle_t dst_tree = pnanovdb_grid_get_tree(dst_buf, dst_grid);
    pnanovdb_node2_memclear(dst_buf, dst_tree.address, PNANOVDB_TREE_SIZE);

    // convert root
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
    pnanovdb_uint32_t tile_count = pnanovdb_root_get_tile_count(buf, root);

    pnanovdb_uint64_t root_count = 0llu;
    pnanovdb_uint64_t upper_count = 0llu;
    pnanovdb_uint64_t lower_count = 0llu;
    pnanovdb_uint64_t leaf_count = 0llu;

    // allocate root v2
    pnanovdb_address_t dst_root_addr = dst_addr_end;
    pnanovdb_node2_handle_t dst_root = {pnanovdb_uint32_t(dst_root_addr.byte_offset >> 3u)};
    pnanovdb_uint32_t dst_root_size = pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_ROOT];
    dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_root_size);
    root_count++;
    pnanovdb_node2_memclear(dst_buf, dst_root_addr, dst_root_size);

    pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_node_idx, node_idx);
    node_idx++;

    // point tree at node2 root
    pnanovdb_tree_set_node_offset_root(dst_buf, dst_tree, pnanovdb_address_diff(dst_root_addr, dst_tree.address));

    // set keys to default
    pnanovdb_uint32_t node2_root_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT];
    for (pnanovdb_uint32_t root_n = 0u; root_n < 2u * node2_root_fanout; root_n++)
    {
        pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_children[PNANOVDB_NODE2_TYPE_ROOT] + root_n, pnanovdb_node2_end_key);
    }
    // set masks and keys
    pnanovdb_uint64_t root_key_range[2u];
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);
        pnanovdb_uint64_t key = pnanovdb_root_tile_get_key(buf, root_tile);
        pnanovdb_coord_t ijk = pnanovdb_node2_key_to_coord(key);
        pnanovdb_uint32_t root_n = pnanovdb_node2_coord_to_n_keyed(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, ijk);

        pnanovdb_uint32_t state = pnanovdb_root_tile_get_state(buf, root_tile);
        pnanovdb_int64_t child = pnanovdb_root_tile_get_child(buf, root_tile);
        if (state != 0u && pnanovdb_int64_is_zero(child))
        {
            pnanovdb_uint32_t word_idx = root_n >> 6u;
            pnanovdb_uint64_t mask_val = pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_ROOT] + word_idx);
            mask_val = mask_val | pnanovdb_uint64_bit_mask(root_n & 63u);
            pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_ROOT] + word_idx, mask_val);
        }
        pnanovdb_bool_t is_new_child = PNANOVDB_FALSE;
        if (!pnanovdb_int64_is_zero(child))
        {
            pnanovdb_uint32_t word_idx = root_n >> 6u;
            pnanovdb_uint64_t mask_val_old = pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_child_mask[PNANOVDB_NODE2_TYPE_ROOT] + word_idx);
            pnanovdb_uint64_t mask_val = mask_val_old | pnanovdb_uint64_bit_mask(root_n & 63u);
            pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_child_mask[PNANOVDB_NODE2_TYPE_ROOT] + word_idx, mask_val);
            is_new_child = mask_val != mask_val_old;
        }
        // always set key mask
        pnanovdb_bool_t is_key_child = PNANOVDB_FALSE;
        {
            pnanovdb_uint32_t word_idx = root_n >> 6u;
            pnanovdb_uint64_t mask_val_old = pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_key_mask[PNANOVDB_NODE2_TYPE_ROOT] + word_idx);
            pnanovdb_uint64_t mask_val = mask_val_old | pnanovdb_uint64_bit_mask(root_n & 63u);
            pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_key_mask[PNANOVDB_NODE2_TYPE_ROOT] + word_idx, mask_val);
            is_key_child = mask_val != mask_val_old;
        }
        pnanovdb_node2_compute_prefix_sums(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT);

        pnanovdb_bool_t child_active;
        pnanovdb_uint32_t child_idx = pnanovdb_node2_mask_n_to_idx(dst_buf,
            dst_root.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[PNANOVDB_NODE2_TYPE_ROOT], PNANOVDB_NODE2_TYPE_ROOT, root_n, PNANOVDB_REF(child_active));
        pnanovdb_bool_t key_active;
        pnanovdb_uint32_t key_idx = pnanovdb_node2_mask_n_to_idx(dst_buf,
            dst_root.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[PNANOVDB_NODE2_TYPE_ROOT], PNANOVDB_NODE2_TYPE_ROOT, root_n, PNANOVDB_REF(key_active));

        pnanovdb_uint32_t child_count = pnanovdb_uint32_t(
            pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[PNANOVDB_NODE2_TYPE_ROOT])) & 0xFFFF;
        pnanovdb_uint32_t child_count_old = is_new_child ? child_count - 1u : child_count;
        pnanovdb_uint32_t key_count = pnanovdb_uint32_t(
            pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[PNANOVDB_NODE2_TYPE_ROOT])) & 0xFFFF;

        for (pnanovdb_uint32_t idx = key_count - 1u; idx < key_count; idx--)
        {
            pnanovdb_uint32_t src_idx = idx > key_idx ? idx - 1u : idx;
            pnanovdb_uint64_t key_tmp = pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_children[PNANOVDB_NODE2_TYPE_ROOT] + child_count_old + src_idx);
            if (key_active && idx == key_idx)
            {
                key_tmp = key;
            }
            pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_children[PNANOVDB_NODE2_TYPE_ROOT] + child_count + idx, key_tmp);
        }
        for (pnanovdb_uint32_t idx = 0u; idx < child_count; idx++)
        {
            pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_children[PNANOVDB_NODE2_TYPE_ROOT] + idx, 0llu);
        }
        if (tile_idx == 0u)
        {
            root_key_range[0] = key;
            root_key_range[1] = key;
        }
        pnanovdb_node2_key_range_accum(&root_key_range[0], &root_key_range[1], key);
    }
    pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_key_range + 0u, root_key_range[0]);
    pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_key_range + 1u, root_key_range[1]);

    // return unused memory
    pnanovdb_uint32_t child_count = pnanovdb_uint32_t(
        pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[PNANOVDB_NODE2_TYPE_ROOT])) & 0xFFFF;
    pnanovdb_uint32_t key_count = pnanovdb_uint32_t(
        pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_key_mask_prefix_sum[PNANOVDB_NODE2_TYPE_ROOT])) & 0xFFFF;
    pnanovdb_uint32_t root_excess = 32u * ((2u * pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT] - child_count - key_count) / 4u);
    dst_addr_end = pnanovdb_address_offset_neg(dst_addr_end, root_excess);

    // set tree upper start
    pnanovdb_upper_handle_t first_upper = {dst_addr_end};
    pnanovdb_tree_set_first_upper(dst_buf, dst_tree, first_upper);

    // convert upper
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);

        if (pnanovdb_root_tile_get_child_mask(buf, root_tile))
        {
            pnanovdb_upper_handle_t upper = pnanovdb_root_get_child(grid_type, buf, root, root_tile);

            // allocate upper
            pnanovdb_address_t dst_upper_addr = dst_addr_end;
            pnanovdb_node2_handle_t dst_upper = {pnanovdb_uint32_t(dst_upper_addr.byte_offset >> 3u)};
            pnanovdb_uint32_t dst_upper_size = pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_UPPER];
            dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_upper_size);
            upper_count++;
            pnanovdb_node2_memclear(dst_buf, dst_upper_addr, dst_upper_size);

            pnanovdb_node2_write(dst_buf, dst_upper.idx64 + pnanovdb_node2_off_node_idx, node_idx);
            node_idx++;

            // copy child and value masks
            pnanovdb_uint32_t upper_mask_word_count = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER] >> 6u;
            for (pnanovdb_uint32_t word_idx = 0u; word_idx < upper_mask_word_count; word_idx++)
            {
                pnanovdb_uint64_t child_mask = pnanovdb_read_uint64(buf, pnanovdb_address_offset(upper.address, PNANOVDB_UPPER_OFF_CHILD_MASK + 8u * word_idx));
                pnanovdb_uint64_t value_mask = pnanovdb_read_uint64(buf, pnanovdb_address_offset(upper.address, PNANOVDB_UPPER_OFF_VALUE_MASK + 8u * word_idx));
                value_mask = value_mask & ~child_mask;
                pnanovdb_node2_write(dst_buf, dst_upper.idx64 + pnanovdb_node2_off_child_mask[PNANOVDB_NODE2_TYPE_UPPER] + word_idx, child_mask);
                pnanovdb_node2_write(dst_buf, dst_upper.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_UPPER] + word_idx, value_mask);
            }

            pnanovdb_node2_compute_prefix_sums(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER);

            // return unused memory
            pnanovdb_uint32_t upper_child_count = pnanovdb_uint32_t(
                pnanovdb_node2_read(dst_buf, dst_upper.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[PNANOVDB_NODE2_TYPE_UPPER])) & 0xFFFF;
            pnanovdb_uint32_t upper_excess = 32u * ((pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER] - upper_child_count) / 4u);
            dst_addr_end = pnanovdb_address_offset_neg(dst_addr_end, upper_excess);

            // set root child
            pnanovdb_coord_t ijk = pnanovdb_node2_key_to_coord(pnanovdb_root_tile_get_key(buf, root_tile));
            pnanovdb_uint32_t root_n = pnanovdb_node2_coord_to_n_keyed(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, ijk);
            pnanovdb_node2_set_child(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_n, dst_upper);
        }
    }

    // set tree lower start
    pnanovdb_lower_handle_t first_lower = {dst_addr_end};
    pnanovdb_tree_set_first_lower(dst_buf, dst_tree, first_lower);

    // convert lower
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);
        if (pnanovdb_root_tile_get_child_mask(buf, root_tile))
        {
            pnanovdb_upper_handle_t upper = pnanovdb_root_get_child(grid_type, buf, root, root_tile);
            // get dst_upper
            pnanovdb_coord_t ijk = pnanovdb_node2_key_to_coord(pnanovdb_root_tile_get_key(buf, root_tile));
            pnanovdb_uint32_t root_n = pnanovdb_node2_coord_to_n_keyed(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, ijk);
            pnanovdb_node2_handle_t dst_upper = pnanovdb_node2_get_child(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_n);

            for (pnanovdb_uint32_t upper_n = 0u; upper_n < PNANOVDB_UPPER_TABLE_COUNT; upper_n++)
            {
                if (pnanovdb_upper_get_child_mask(buf, upper, upper_n))
                {
                    pnanovdb_lower_handle_t lower = pnanovdb_upper_get_child(grid_type, buf, upper, upper_n);

                    // allocate lower
                    pnanovdb_address_t dst_lower_addr = dst_addr_end;
                    pnanovdb_node2_handle_t dst_lower = {pnanovdb_uint32_t(dst_lower_addr.byte_offset >> 3u)};
                    pnanovdb_uint32_t dst_lower_size = pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_LOWER];
                    dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_lower_size);
                    lower_count++;
                    pnanovdb_node2_memclear(dst_buf, dst_lower_addr, dst_lower_size);

                    pnanovdb_node2_write(dst_buf, dst_lower.idx64 + pnanovdb_node2_off_node_idx, node_idx);
                    node_idx++;

                    // copy child and value masks
                    pnanovdb_uint32_t lower_mask_word_count = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER] >> 6u;
                    for (pnanovdb_uint32_t word_idx = 0u; word_idx < lower_mask_word_count; word_idx++)
                    {
                        pnanovdb_uint64_t child_mask = pnanovdb_read_uint64(buf, pnanovdb_address_offset(lower.address, PNANOVDB_LOWER_OFF_CHILD_MASK + 8u * word_idx));
                        pnanovdb_uint64_t value_mask = pnanovdb_read_uint64(buf, pnanovdb_address_offset(lower.address, PNANOVDB_LOWER_OFF_VALUE_MASK + 8u * word_idx));
                        value_mask = value_mask & ~child_mask;
                        pnanovdb_node2_write(dst_buf, dst_lower.idx64 + pnanovdb_node2_off_child_mask[PNANOVDB_NODE2_TYPE_LOWER] + word_idx, child_mask);
                        pnanovdb_node2_write(dst_buf, dst_lower.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_LOWER] + word_idx, value_mask);
                    }
                    pnanovdb_node2_compute_prefix_sums(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER);

                    // return unused memory
                    pnanovdb_uint32_t lower_child_count = pnanovdb_uint32_t(
                        pnanovdb_node2_read(dst_buf, dst_lower.idx64 + pnanovdb_node2_off_child_mask_prefix_sum[PNANOVDB_NODE2_TYPE_LOWER])) & 0xFFFF;
                    pnanovdb_uint32_t lower_excess = 32u * ((pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER] - lower_child_count) / 4u);
                    dst_addr_end = pnanovdb_address_offset_neg(dst_addr_end, lower_excess);

                    // set upper child
                    pnanovdb_node2_set_child(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n, dst_lower);
                }
            }
        }
    }

    // set tree leaf start
    pnanovdb_leaf_handle_t first_leaf = {dst_addr_end};
    pnanovdb_tree_set_first_leaf(dst_buf, dst_tree, first_leaf);

    // convert leaf
    for (pnanovdb_uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++)
    {
        pnanovdb_root_tile_handle_t root_tile = pnanovdb_root_get_tile(grid_type, root, tile_idx);
        if (pnanovdb_root_tile_get_child_mask(buf, root_tile))
        {
            pnanovdb_upper_handle_t upper = pnanovdb_root_get_child(grid_type, buf, root, root_tile);
            // get dst_upper
            pnanovdb_coord_t ijk = pnanovdb_node2_key_to_coord(pnanovdb_root_tile_get_key(buf, root_tile));
            pnanovdb_uint32_t root_n = pnanovdb_node2_coord_to_n_keyed(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, ijk);
            pnanovdb_node2_handle_t dst_upper = pnanovdb_node2_get_child(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_n);

            pnanovdb_uint32_t active_upper_child_idx = 0u;
            for (pnanovdb_uint32_t upper_n = 0u; upper_n < PNANOVDB_UPPER_TABLE_COUNT; upper_n++)
            {
                if (pnanovdb_upper_get_child_mask(buf, upper, upper_n))
                {
                    pnanovdb_lower_handle_t lower = pnanovdb_upper_get_child(grid_type, buf, upper, upper_n);
                    // get dst_lower
                    pnanovdb_node2_handle_t dst_lower = pnanovdb_node2_get_child(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);

                    for (pnanovdb_uint32_t lower_n = 0u; lower_n < PNANOVDB_LOWER_TABLE_COUNT; lower_n++)
                    {
                        if (pnanovdb_lower_get_child_mask(buf, lower, lower_n))
                        {
                            pnanovdb_leaf_handle_t leaf = pnanovdb_lower_get_child(grid_type, buf, lower, lower_n);

                            // allocate leaf
                            pnanovdb_address_t dst_leaf_addr = dst_addr_end;
                            pnanovdb_node2_handle_t dst_leaf = {pnanovdb_uint32_t(dst_leaf_addr.byte_offset >> 3u)};
                            pnanovdb_uint32_t dst_leaf_size = pnanovdb_node2_max_size[PNANOVDB_NODE2_TYPE_LEAF];
                            dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_leaf_size);
                            leaf_count++;
                            pnanovdb_node2_memclear(dst_buf, dst_leaf_addr, dst_leaf_size);

                            pnanovdb_node2_write(dst_buf, dst_leaf.idx64 + pnanovdb_node2_off_node_idx, node_idx);
                            node_idx++;

                            // copy value masks
                            pnanovdb_uint32_t leaf_mask_word_count = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LEAF] >> 6u;
                            for (pnanovdb_uint32_t word_idx = 0u; word_idx < leaf_mask_word_count; word_idx++)
                            {
                                pnanovdb_uint64_t value_mask = pnanovdb_read_uint64(buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * word_idx));
                                pnanovdb_node2_write(dst_buf, dst_leaf.idx64 + pnanovdb_node2_off_value_mask[PNANOVDB_NODE2_TYPE_LEAF] + word_idx, value_mask);
                            }

                            pnanovdb_node2_compute_prefix_sums(dst_buf, dst_leaf, PNANOVDB_NODE2_TYPE_LEAF);

                            // set lower child
                            pnanovdb_node2_set_child(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n, dst_leaf);
                        }
                    }
                }
            }
        }
    }

    // finalize tree segment sizes
    pnanovdb_tree_set_node_count_upper(dst_buf, dst_tree, upper_count);
    pnanovdb_tree_set_node_count_lower(dst_buf, dst_tree, lower_count);
    pnanovdb_tree_set_node_count_leaf(dst_buf, dst_tree, leaf_count);
    // todo: set tile/voxel counts

    // generically walk tree, compute total active and inactive values
    pnanovdb_uint64_t value_count = pnanovdb_uint32_as_uint64_low(2u);     // default to 2 for inactive values
    pnanovdb_uint64_t inactive_count = pnanovdb_uint32_as_uint64_low(0u);

    pnanovdb_node2_write(dst_buf, dst_root.idx64 + pnanovdb_node2_off_value_idx, value_count);
    value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_values_word_count(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT));
    inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT));

    node2_root_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT];
    for (pnanovdb_uint32_t root_n = 0u; root_n < node2_root_fanout; root_n++)
    {
        pnanovdb_node2_handle_t dst_upper = pnanovdb_node2_get_child(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
        if (dst_upper.idx64 == 0u)
        {
            continue;
        }

        pnanovdb_node2_write(dst_buf, dst_upper.idx64 + pnanovdb_node2_off_value_idx, value_count);
        value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_values_word_count(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER));
        inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER));

        pnanovdb_uint32_t node2_upper_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER];
        for (pnanovdb_uint32_t upper_n = 0u; upper_n < node2_upper_fanout; upper_n++)
        {
            pnanovdb_node2_handle_t dst_lower = pnanovdb_node2_get_child(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);
            if (dst_lower.idx64 == 0u)
            {
                continue;
            }

            pnanovdb_node2_write(dst_buf, dst_lower.idx64 + pnanovdb_node2_off_value_idx, value_count);
            value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_values_word_count(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER));
            inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER));

            pnanovdb_uint32_t node2_lower_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER];
            for (pnanovdb_uint32_t lower_n = 0u; lower_n < node2_lower_fanout; lower_n++)
            {
                pnanovdb_node2_handle_t dst_leaf = pnanovdb_node2_get_child(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                if (dst_leaf.idx64 == 0u)
                {
                    continue;
                }

                pnanovdb_node2_write(dst_buf, dst_leaf.idx64 + pnanovdb_node2_off_value_idx, value_count);
                value_count = pnanovdb_uint64_offset(value_count, pnanovdb_node2_compute_values_word_count(dst_buf, dst_leaf, PNANOVDB_NODE2_TYPE_LEAF));
                inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_leaf, PNANOVDB_NODE2_TYPE_LEAF));
            }
        }
    }

    // allocate blindmetadata header for bbox and values
    pnanovdb_gridblindmetadata_handle_t dst_metadata = {dst_addr_end};
    pnanovdb_uint32_t dst_metadata_size = 4u * PNANOVDB_GRIDBLINDMETADATA_SIZE;
    dst_addr_end = pnanovdb_address_offset(dst_addr_end, dst_metadata_size);

    // link headers to grid
    pnanovdb_grid_set_first_gridblindmetadata(dst_buf, dst_grid, dst_metadata);
    pnanovdb_grid_set_blind_metadata_count(dst_buf, dst_grid, 4u);

    // allocate coords for root bbox
    pnanovdb_address_t dst_bboxes = {dst_addr_end};
    pnanovdb_uint64_t dst_bboxes_size = pnanovdb_node2_uint64_align32(6u * 4u);
    dst_addr_end = pnanovdb_address_offset64(dst_addr_end, dst_bboxes_size);

    // allocate floats for active values
    pnanovdb_address_t dst_values = {dst_addr_end};
    pnanovdb_uint64_t dst_values_size = pnanovdb_node2_uint64_align32(value_count * 4u);
    dst_addr_end = pnanovdb_address_offset64(dst_addr_end, dst_values_size);

    // allocate uint64 for inactive value indices
    pnanovdb_address_t dst_node_inactive_idxs = {dst_addr_end};
    pnanovdb_uint64_t dst_node_inactive_idxs_size = pnanovdb_node2_uint64_align32(node_idx * 8u);
    dst_addr_end = pnanovdb_address_offset64(dst_addr_end, dst_node_inactive_idxs_size);

    // allocate uint64 bitmasks for inactive values
    pnanovdb_address_t dst_inactive_value_idxs = {dst_addr_end};
    pnanovdb_uint64_t dst_inactive_value_idxs_size = pnanovdb_node2_uint64_align32(((inactive_count + 63u) / 64u) * 8u);
    dst_addr_end = pnanovdb_address_offset64(dst_addr_end, dst_inactive_value_idxs_size);

    // bbox header
    pnanovdb_gridblindmetadata_set_data_offset(dst_buf, dst_metadata, pnanovdb_address_diff(dst_bboxes, dst_metadata.address));
    pnanovdb_gridblindmetadata_set_value_count(dst_buf, dst_metadata, 2u);
    pnanovdb_gridblindmetadata_set_value_size(dst_buf, dst_metadata, 12u);
    pnanovdb_gridblindmetadata_set_semantic(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_class(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_type(dst_buf, dst_metadata, PNANOVDB_GRID_TYPE_VEC3F);
    pnanovdb_gridblindmetadata_set_name(dst_buf, dst_metadata, 0u, 0u);

    // values header
    dst_metadata.address = pnanovdb_address_offset(dst_metadata.address, PNANOVDB_GRIDBLINDMETADATA_SIZE);
    pnanovdb_gridblindmetadata_set_data_offset(dst_buf, dst_metadata, pnanovdb_address_diff(dst_values, dst_metadata.address));
    pnanovdb_gridblindmetadata_set_value_count(dst_buf, dst_metadata, value_count);
    pnanovdb_gridblindmetadata_set_value_size(dst_buf, dst_metadata, 4u);
    pnanovdb_gridblindmetadata_set_semantic(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_class(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_type(dst_buf, dst_metadata, PNANOVDB_GRID_TYPE_FLOAT);
    pnanovdb_gridblindmetadata_set_name(dst_buf, dst_metadata, 0u, 0u);

    // inactive value indices header
    dst_metadata.address = pnanovdb_address_offset(dst_metadata.address, PNANOVDB_GRIDBLINDMETADATA_SIZE);
    pnanovdb_gridblindmetadata_set_data_offset(dst_buf, dst_metadata, pnanovdb_address_diff(dst_node_inactive_idxs, dst_metadata.address));
    pnanovdb_gridblindmetadata_set_value_count(dst_buf, dst_metadata, node_idx);
    pnanovdb_gridblindmetadata_set_value_size(dst_buf, dst_metadata, 8u);
    pnanovdb_gridblindmetadata_set_semantic(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_class(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_type(dst_buf, dst_metadata, PNANOVDB_GRID_TYPE_INT64);
    pnanovdb_gridblindmetadata_set_name(dst_buf, dst_metadata, 0u, 0u);

    // inactive values header
    dst_metadata.address = pnanovdb_address_offset(dst_metadata.address, PNANOVDB_GRIDBLINDMETADATA_SIZE);
    pnanovdb_gridblindmetadata_set_data_offset(dst_buf, dst_metadata, pnanovdb_address_diff(dst_inactive_value_idxs, dst_metadata.address));
    pnanovdb_gridblindmetadata_set_value_count(dst_buf, dst_metadata, (inactive_count + 63u) / 64u);
    pnanovdb_gridblindmetadata_set_value_size(dst_buf, dst_metadata, 8u);
    pnanovdb_gridblindmetadata_set_semantic(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_class(dst_buf, dst_metadata, 0u);
    pnanovdb_gridblindmetadata_set_data_type(dst_buf, dst_metadata, PNANOVDB_GRID_TYPE_INT64);
    pnanovdb_gridblindmetadata_set_name(dst_buf, dst_metadata, 0u, 0u);

    // write root bbox
    pnanovdb_coord_t bbox_min = pnanovdb_root_get_bbox_min(buf, root);
    pnanovdb_coord_t bbox_max = pnanovdb_root_get_bbox_max(buf, root);
    pnanovdb_write_coord(dst_buf, pnanovdb_address_offset(dst_bboxes, 0u), PNANOVDB_REF(bbox_min));
    pnanovdb_write_coord(dst_buf, pnanovdb_address_offset(dst_bboxes, 12u), PNANOVDB_REF(bbox_max));

    // write inactive value indices
    inactive_count = pnanovdb_uint32_as_uint64_low(0u);

    pnanovdb_uint64_t node_idx_inactive = pnanovdb_node2_read(dst_buf, dst_root.idx64 + pnanovdb_node2_off_node_idx);
    pnanovdb_node2_write(dst_buf, (dst_node_inactive_idxs.byte_offset >> 3u) + node_idx_inactive, inactive_count);
    inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT));

    node2_root_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT];
    for (pnanovdb_uint32_t root_n = 0u; root_n < node2_root_fanout; root_n++)
    {
        pnanovdb_node2_handle_t dst_upper = pnanovdb_node2_get_child(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
        if (dst_upper.idx64 == 0u)
        {
            continue;
        }

        node_idx_inactive = pnanovdb_node2_read(dst_buf, dst_upper.idx64 + pnanovdb_node2_off_node_idx);
        pnanovdb_node2_write(dst_buf, (dst_node_inactive_idxs.byte_offset >> 3u) + node_idx_inactive, inactive_count);
        inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER));

        pnanovdb_uint32_t node2_upper_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER];
        for (pnanovdb_uint32_t upper_n = 0u; upper_n < node2_upper_fanout; upper_n++)
        {
            pnanovdb_node2_handle_t dst_lower = pnanovdb_node2_get_child(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);
            if (dst_lower.idx64 == 0u)
            {
                continue;
            }

            node_idx_inactive = pnanovdb_node2_read(dst_buf, dst_lower.idx64 + pnanovdb_node2_off_node_idx);
            pnanovdb_node2_write(dst_buf, (dst_node_inactive_idxs.byte_offset >> 3u) + node_idx_inactive, inactive_count);
            inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER));

            pnanovdb_uint32_t node2_lower_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER];
            for (pnanovdb_uint32_t lower_n = 0u; lower_n < node2_lower_fanout; lower_n++)
            {
                pnanovdb_node2_handle_t dst_leaf = pnanovdb_node2_get_child(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                if (dst_leaf.idx64 == 0u)
                {
                    continue;
                }

                node_idx_inactive = pnanovdb_node2_read(dst_buf, dst_leaf.idx64 + pnanovdb_node2_off_node_idx);
                pnanovdb_node2_write(dst_buf, (dst_node_inactive_idxs.byte_offset >> 3u) + node_idx_inactive, inactive_count);
                inactive_count = pnanovdb_uint64_offset(inactive_count, pnanovdb_node2_compute_inactive_word_count(dst_buf, dst_leaf, PNANOVDB_NODE2_TYPE_LEAF));
            }
        }
    }

    // capture background
    pnanovdb_address_t background_addr = pnanovdb_root_get_background_address(grid_type, buf, root);
    float background_val = pnanovdb_read_float(buf, background_addr);
    pnanovdb_node2_write_float_index(dst_buf, dst_values, 0u, background_val);
    pnanovdb_node2_write_float_index(dst_buf, dst_values, 1u, -background_val);

    // capture values
    pnanovdb_coord_t root_ijk = {};
    pnanovdb_convert_to_node2_value_copy(
        dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_ijk, dst_values, dst_node_inactive_idxs, dst_inactive_value_idxs, buf, grid_type, root);

    node2_root_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_ROOT];
    for (pnanovdb_uint32_t root_n = 0u; root_n < node2_root_fanout; root_n++)
    {
        pnanovdb_node2_handle_t dst_upper = pnanovdb_node2_get_child(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
        if (dst_upper.idx64 == 0u)
        {
            continue;
        }

        pnanovdb_uint64_t key = pnanovdb_node2_get_key(dst_buf, dst_root, PNANOVDB_NODE2_TYPE_ROOT, root_n);
        pnanovdb_coord_t upper_ijk = pnanovdb_node2_key_to_coord(key);

        pnanovdb_convert_to_node2_value_copy(
            dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER, upper_ijk, dst_values, dst_node_inactive_idxs, dst_inactive_value_idxs, buf, grid_type, root);

        pnanovdb_uint32_t node2_upper_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_UPPER];
        for (pnanovdb_uint32_t upper_n = 0u; upper_n < node2_upper_fanout; upper_n++)
        {
            pnanovdb_node2_handle_t dst_lower = pnanovdb_node2_get_child(dst_buf, dst_upper, PNANOVDB_NODE2_TYPE_UPPER, upper_n);
            if (dst_lower.idx64 == 0u)
            {
                continue;
            }

            pnanovdb_coord_t lower_ijk = pnanovdb_node2_upper_coord_to_lower_coord(upper_ijk, upper_n);

            pnanovdb_convert_to_node2_value_copy(
                dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER, lower_ijk, dst_values, dst_node_inactive_idxs, dst_inactive_value_idxs, buf, grid_type, root);

            pnanovdb_uint32_t node2_lower_fanout = pnanovdb_node2_fanout_1d[PNANOVDB_NODE2_TYPE_LOWER];
            for (pnanovdb_uint32_t lower_n = 0u; lower_n < node2_lower_fanout; lower_n++)
            {
                pnanovdb_node2_handle_t dst_leaf = pnanovdb_node2_get_child(dst_buf, dst_lower, PNANOVDB_NODE2_TYPE_LOWER, lower_n);
                if (dst_leaf.idx64 == 0u)
                {
                    continue;
                }

                pnanovdb_coord_t leaf_ijk = pnanovdb_node2_lower_coord_to_leaf_coord(lower_ijk, lower_n);

                pnanovdb_convert_to_node2_value_copy(
                    dst_buf, dst_leaf, PNANOVDB_NODE2_TYPE_LEAF, leaf_ijk, dst_values, dst_node_inactive_idxs, dst_inactive_value_idxs, buf, grid_type, root);
            }
        }
    }

    // finalize grid size
    pnanovdb_grid_set_grid_size(dst_buf, dst_grid, pnanovdb_address_diff(dst_addr_end, dst_grid.address));
}

#endif

#endif // end of NANOVDB_PNANOVDB2_H_HAS_BEEN_INCLUDED
