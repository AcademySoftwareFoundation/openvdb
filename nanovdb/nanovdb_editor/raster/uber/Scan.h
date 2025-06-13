
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/putil/Scan.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
            of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PUTILS_SCAN_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_SCAN_H_HAS_BEEN_INCLUDED

#include "nanovdb/PNanoVDB.h"
#include "nanovdb/putil/Reduce.h"

// ----------------------------- Workgroup Scan ---------------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_scan_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_scan(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT_ARRAY(pnanovdb_uvec4_t) result[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT(pnanovdb_uint32_t) total_count
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);

    pnanovdb_uint32_t local_val[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        local_val[vidx] = val[vidx].x +
            val[vidx].y +
            val[vidx].z +
            val[vidx].w;
        pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 1) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 1);
        if (thread_idx >= 2) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 2);
        if (thread_idx >= 3) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 3);
        pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 4) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 4);
        if (thread_idx >= 8) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 8);
        if (thread_idx >= 12) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 12);
        pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 16) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 16);
        if (thread_idx >= 32) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 32);
        if (thread_idx >= 48) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 48);
        pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 64) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 64);
        if (thread_idx >= 128) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 128);
        if (thread_idx >= 192) local_val[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 192);

        result[vidx].w = local_val[vidx];
        result[vidx].z = result[vidx].w - val[vidx].w;
        result[vidx].y = result[vidx].z - val[vidx].z;
        result[vidx].x = result[vidx].y - val[vidx].y;

        // compute totalCount
        PNANOVDB_DEREF(total_count) = pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 63) +
            pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 127) +
            pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 191) +
            pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 255);
    }
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_scan_max_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_scan_max(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT_ARRAY(pnanovdb_uvec4_t) result[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT(pnanovdb_uint32_t) total_count
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);

    pnanovdb_uint32_t local_val[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        local_val[vidx] = pnanovdb_uint32_max(
            pnanovdb_uint32_max(val[vidx].x, val[vidx].y),
            pnanovdb_uint32_max(val[vidx].z, val[vidx].w));
        pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 1) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 1));
        if (thread_idx >= 2) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 2));
        if (thread_idx >= 3) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 3));
        pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 4) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 4));
        if (thread_idx >= 8) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 8));
        if (thread_idx >= 12) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 12));
        pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 16) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 16));
        if (thread_idx >= 32) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 32));
        if (thread_idx >= 48) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 48));
        pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx >= 64) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 64));
        if (thread_idx >= 128) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 128));
        if (thread_idx >= 192) local_val[vidx] = pnanovdb_uint32_max(local_val[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx - 192));
        pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        pnanovdb_uint32_t scan_val = 0u;
        if (thread_idx > 0u)
        {
            scan_val = pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx - 1u);
        }

        result[vidx].x = pnanovdb_uint32_max(scan_val, val[vidx].x);
        result[vidx].y = pnanovdb_uint32_max(result[vidx].x, val[vidx].y);
        result[vidx].z = pnanovdb_uint32_max(result[vidx].y, val[vidx].z);
        result[vidx].w = pnanovdb_uint32_max(result[vidx].z, val[vidx].w);

        // compute totalCount
        PNANOVDB_DEREF(total_count) = pnanovdb_read_uint32_index(smem_buf, sdata0_addr, 255);
    }

    pnanovdb_workgroup_sync();
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_scan_keyed_float_smem_size_in_words = 256u + 256u + 256u;

PNANOVDB_FORCE_INLINE float pnanovdb_workgroup_scan_keyed_get_float_if_match(
    pnanovdb_uint32_t thread_idx,
    pnanovdb_buf_t smem_buf,
    pnanovdb_address_t sdata_addr,
    pnanovdb_address_t skey_addr,
    pnanovdb_uint32_t cmp_key,
    pnanovdb_uint32_t offset
)
{
    float val = 0.f;
    if (thread_idx >= offset)
    {
        pnanovdb_uint32_t key = pnanovdb_read_uint32_index(smem_buf, skey_addr, thread_idx - offset);
        if (key == cmp_key)
        {
            val = pnanovdb_read_float_index(smem_buf, sdata_addr, thread_idx - offset);
        }
    }
    return val;
}

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_scan_keyed_float(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_vec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    pnanovdb_uvec4_t key[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT_ARRAY(pnanovdb_vec4_t) result[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT(float) last_count,
    PNANOVDB_INOUT(pnanovdb_uint32_t) last_key
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);
    pnanovdb_address_t skey_addr = pnanovdb_address_offset(smem_addr, 256u * 4u + 256u * 4u);

    float local_val[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        local_val[vidx] = val[vidx].w;
        if (key[vidx].w == key[vidx].z) {local_val[vidx] += val[vidx].z;}
        if (key[vidx].w == key[vidx].y) {local_val[vidx] += val[vidx].y;}
        if (key[vidx].w == key[vidx].x) {local_val[vidx] += val[vidx].x;}
        pnanovdb_write_float_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
        pnanovdb_write_uint32_index(smem_buf, skey_addr, thread_idx, key[vidx].w);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata0_addr, skey_addr, key[vidx].w, 1u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata0_addr, skey_addr, key[vidx].w, 2u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata0_addr, skey_addr, key[vidx].w, 3u);
        pnanovdb_write_float_index(smem_buf, sdata1_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata1_addr, skey_addr, key[vidx].w, 4u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata1_addr, skey_addr, key[vidx].w, 8u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata1_addr, skey_addr, key[vidx].w, 12u);
        pnanovdb_write_float_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata0_addr, skey_addr, key[vidx].w, 16u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata0_addr, skey_addr, key[vidx].w, 32u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata0_addr, skey_addr, key[vidx].w, 48u);
        pnanovdb_write_float_index(smem_buf, sdata1_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata1_addr, skey_addr, key[vidx].w, 64u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata1_addr, skey_addr, key[vidx].w, 48u);
        local_val[vidx] += pnanovdb_workgroup_scan_keyed_get_float_if_match(thread_idx, smem_buf, sdata1_addr, skey_addr, key[vidx].w, 192u);
        pnanovdb_write_float_index(smem_buf, sdata0_addr, thread_idx, local_val[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        result[vidx] = val[vidx];
        // thread local scan
        if (key[vidx].x == key[vidx].y)
        {
            result[vidx].y += result[vidx].x;
        }
        if (key[vidx].y == key[vidx].z)
        {
            result[vidx].z += result[vidx].y;
        }
        if (key[vidx].z == key[vidx].w)
        {
            result[vidx].w += result[vidx].z;
        }
        // add exclusive scan result if key matches
        if (thread_idx > 0u)
        {
            float scan_val = pnanovdb_read_float_index(smem_buf, sdata0_addr, thread_idx - 1u);
            pnanovdb_uint32_t scan_key = pnanovdb_read_uint32_index(smem_buf, skey_addr, thread_idx - 1u);
            if (key[vidx].x == scan_key)
            {
                result[vidx].x += scan_val;
            }
            if (key[vidx].y == scan_key)
            {
                result[vidx].y += scan_val;
            }
            if (key[vidx].z == scan_key)
            {
                result[vidx].z += scan_val;
            }
            if (key[vidx].w == scan_key)
            {
                result[vidx].w += scan_val;
            }
        }

        PNANOVDB_DEREF(last_count) = pnanovdb_read_float_index(smem_buf, sdata0_addr, 255);
        PNANOVDB_DEREF(last_key) = pnanovdb_read_uint32_index(smem_buf, skey_addr, 255);
    }

    pnanovdb_workgroup_sync();
}

// ----------------------------- Global Scan ---------------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_scan_smem_size_in_words = 256u + 256u + 2u;

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_scan_compute_tmp_size(pnanovdb_uint32_t val_count)
{
    pnanovdb_uint32_t workgroup_count = (val_count + 1023u) / 1024u;

    pnanovdb_address_t tmp_addr = pnanovdb_address_null();
    pnanovdb_address_t reduce_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);
    pnanovdb_address_t reduce_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);
    return tmp_addr;
}

PNANOVDB_FORCE_INLINE void pnanovdb_scan(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t val_count,
    pnanovdb_buf_t val_buf, pnanovdb_address_t val_addr,
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr,
    pnanovdb_buf_t result_buf, pnanovdb_address_t result_addr,
    PNANOVDB_INOUT(pnanovdb_uint32_t) total_count
)
{
    pnanovdb_uint32_t workgroup_count = (val_count + 1023u) / 1024u;

    pnanovdb_address_t reduce_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);
    pnanovdb_address_t reduce_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);

    // reduce per workgroup
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

            if (4u * val4_idx < val_count)
            {
                val[vidx] = pnanovdb_read_uvec4_index(val_buf, val_addr, val4_idx);
            }
            if (4u * val4_idx + 0u >= val_count) { val[vidx].x = 0u; }
            if (4u * val4_idx + 1u >= val_count) { val[vidx].y = 0u; }
            if (4u * val4_idx + 2u >= val_count) { val[vidx].z = 0u; }
            if (4u * val4_idx + 3u >= val_count) { val[vidx].w = 0u; }
        }

        pnanovdb_uint32_t total_count = 0u;
        pnanovdb_workgroup_reduce(vidx_offset, smem_buf, smem_addr, val, PNANOVDB_REF(total_count));

        if (vidx_offset == 0u)
        {
            pnanovdb_uint32_t reduce_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;

            pnanovdb_write_uint32_index(tmp_buf, reduce_addr, reduce_idx, total_count);
        }
    }

    // scan across workgroup counts
    pnanovdb_sync(sync_state, 1u, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t scan_pass_count = (workgroup_count + 1023u) / 1024u;
        pnanovdb_uint32_t global_offset = 0u;
        for (pnanovdb_uint32_t scan_pass_idx = 0u; scan_pass_idx < scan_pass_count; scan_pass_idx++)
        {
            pnanovdb_uvec4_t reduce[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t reduce_idx = scan_pass_idx * 256u + thread_idx;

                reduce[vidx].x = 0u;
                reduce[vidx].y = 0u;
                reduce[vidx].z = 0u;
                reduce[vidx].w = 0u;
                if (4u * reduce_idx < workgroup_count)
                {
                    reduce[vidx] = pnanovdb_read_uvec4_index(tmp_buf, reduce_addr, reduce_idx);
                }
            }

            pnanovdb_uint32_t total_count = 0u;
            pnanovdb_uvec4_t reduce_scan[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_workgroup_scan(vidx_offset, smem_buf, smem_addr, reduce, reduce_scan, PNANOVDB_REF(total_count));

            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t reduce_idx = scan_pass_idx * 256u + thread_idx;

                reduce_scan[vidx].x += global_offset;
                reduce_scan[vidx].y += global_offset;
                reduce_scan[vidx].z += global_offset;
                reduce_scan[vidx].w += global_offset;

                if (4u * reduce_idx < workgroup_count)
                {
                    pnanovdb_write_uvec4_index(tmp_buf, reduce_scan_addr, reduce_idx, reduce_scan[vidx]);
                }
            }

            global_offset += total_count;
        }
    }

    // local scan + add workgroup count
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

            if (4u * val4_idx < val_count)
            {
                val[vidx] = pnanovdb_read_uvec4_index(val_buf, val_addr, val4_idx);
            }
            if (4u * val4_idx + 0u >= val_count) { val[vidx].x = 0u; }
            if (4u * val4_idx + 1u >= val_count) { val[vidx].y = 0u; }
            if (4u * val4_idx + 2u >= val_count) { val[vidx].z = 0u; }
            if (4u * val4_idx + 3u >= val_count) { val[vidx].w = 0u; }
        }

        pnanovdb_uint32_t total_count = 0u;
        pnanovdb_uvec4_t val_scan[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_workgroup_scan(vidx_offset, smem_buf, smem_addr, val, val_scan, PNANOVDB_REF(total_count));

        pnanovdb_uint32_t workgroup_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;
        pnanovdb_uint32_t global_offset = 0u;
        if (workgroup_idx > 0u)
        {
            global_offset = pnanovdb_read_uint32_index(tmp_buf, reduce_scan_addr, workgroup_idx - 1u);
        }

        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

            val_scan[vidx].x += global_offset;
            val_scan[vidx].y += global_offset;
            val_scan[vidx].z += global_offset;
            val_scan[vidx].w += global_offset;

            if (4u * val4_idx < val_count)
            {
                pnanovdb_write_uvec4_index(result_buf, result_addr, val4_idx, val_scan[vidx]);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    PNANOVDB_DEREF(total_count) = workgroup_count > 0u ? pnanovdb_read_uint32_index(tmp_buf, reduce_scan_addr, workgroup_count - 1u) : 0u;
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_scan_max_smem_size_in_words = 256u + 256u + 2u;

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_scan_max_compute_tmp_size(pnanovdb_uint32_t val_count)
{
    return pnanovdb_scan_compute_tmp_size(val_count);
}

PNANOVDB_FORCE_INLINE void pnanovdb_scan_max(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t val_count,
    pnanovdb_buf_t val_buf, pnanovdb_address_t val_addr,
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr,
    pnanovdb_buf_t result_buf, pnanovdb_address_t result_addr,
    PNANOVDB_INOUT(pnanovdb_uint32_t) total_count
)
{
    pnanovdb_uint32_t workgroup_count = (val_count + 1023u) / 1024u;

    pnanovdb_address_t reduce_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);
    pnanovdb_address_t reduce_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);

    // reduce per workgroup
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

            if (4u * val4_idx < val_count)
            {
                val[vidx] = pnanovdb_read_uvec4_index(val_buf, val_addr, val4_idx);
            }
            if (4u * val4_idx + 0u >= val_count) { val[vidx].x = 0u; }
            if (4u * val4_idx + 1u >= val_count) { val[vidx].y = 0u; }
            if (4u * val4_idx + 2u >= val_count) { val[vidx].z = 0u; }
            if (4u * val4_idx + 3u >= val_count) { val[vidx].w = 0u; }
        }

        pnanovdb_uint32_t total_max = 0u;
        pnanovdb_workgroup_reduce_max(vidx_offset, smem_buf, smem_addr, val, PNANOVDB_REF(total_max));

        if (vidx_offset == 0u)
        {
            pnanovdb_uint32_t reduce_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;

            pnanovdb_write_uint32_index(tmp_buf, reduce_addr, reduce_idx, total_max);
        }
    }

    // scan across workgroup counts
    pnanovdb_sync(sync_state, 1u, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t scan_pass_count = (workgroup_count + 1023u) / 1024u;
        pnanovdb_uint32_t global_max = 0u;
        for (pnanovdb_uint32_t scan_pass_idx = 0u; scan_pass_idx < scan_pass_count; scan_pass_idx++)
        {
            pnanovdb_uvec4_t reduce[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t reduce_idx = scan_pass_idx * 256u + thread_idx;

                reduce[vidx].x = 0u;
                reduce[vidx].y = 0u;
                reduce[vidx].z = 0u;
                reduce[vidx].w = 0u;
                if (4u * reduce_idx < workgroup_count)
                {
                    reduce[vidx] = pnanovdb_read_uvec4_index(tmp_buf, reduce_addr, reduce_idx);
                }
            }

            pnanovdb_uint32_t total_max = 0u;
            pnanovdb_uvec4_t reduce_scan[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_workgroup_scan_max(vidx_offset, smem_buf, smem_addr, reduce, reduce_scan, PNANOVDB_REF(total_max));

            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t reduce_idx = scan_pass_idx * 256u + thread_idx;

                reduce_scan[vidx].x = pnanovdb_uint32_max(reduce_scan[vidx].x, global_max);
                reduce_scan[vidx].y = pnanovdb_uint32_max(reduce_scan[vidx].y, global_max);
                reduce_scan[vidx].z = pnanovdb_uint32_max(reduce_scan[vidx].z, global_max);
                reduce_scan[vidx].w = pnanovdb_uint32_max(reduce_scan[vidx].w, global_max);

                if (4u * reduce_idx < workgroup_count)
                {
                    pnanovdb_write_uvec4_index(tmp_buf, reduce_scan_addr, reduce_idx, reduce_scan[vidx]);
                }
            }

            global_max = pnanovdb_uint32_max(global_max, total_max);
        }
    }

    // local scan + add workgroup count
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

            if (4u * val4_idx < val_count)
            {
                val[vidx] = pnanovdb_read_uvec4_index(val_buf, val_addr, val4_idx);
            }
            if (4u * val4_idx + 0u >= val_count) { val[vidx].x = 0u; }
            if (4u * val4_idx + 1u >= val_count) { val[vidx].y = 0u; }
            if (4u * val4_idx + 2u >= val_count) { val[vidx].z = 0u; }
            if (4u * val4_idx + 3u >= val_count) { val[vidx].w = 0u; }
        }

        pnanovdb_uint32_t total_max = 0u;
        pnanovdb_uvec4_t val_scan[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_workgroup_scan_max(vidx_offset, smem_buf, smem_addr, val, val_scan, PNANOVDB_REF(total_max));

        pnanovdb_uint32_t workgroup_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;
        pnanovdb_uint32_t global_max = 0u;
        if (workgroup_idx > 0u)
        {
            global_max = pnanovdb_read_uint32_index(tmp_buf, reduce_scan_addr, workgroup_idx - 1u);
        }

        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

            val_scan[vidx].x = pnanovdb_uint32_max(val_scan[vidx].x, global_max);
            val_scan[vidx].y = pnanovdb_uint32_max(val_scan[vidx].y, global_max);
            val_scan[vidx].z = pnanovdb_uint32_max(val_scan[vidx].z, global_max);
            val_scan[vidx].w = pnanovdb_uint32_max(val_scan[vidx].w, global_max);

            if (4u * val4_idx < val_count)
            {
                pnanovdb_write_uvec4_index(result_buf, result_addr, val4_idx, val_scan[vidx]);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    PNANOVDB_DEREF(total_count) = workgroup_count > 0u ? pnanovdb_read_uint32_index(tmp_buf, reduce_scan_addr, workgroup_count - 1u) : 0u;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_scan_keyed_float_compute_tmp_size(pnanovdb_uint32_t val_count, pnanovdb_uint32_t channel_count)
{
    pnanovdb_uint32_t workgroup_count = (val_count + 1023u) / 1024u;

    pnanovdb_address_t tmp_addr = pnanovdb_address_null();
    pnanovdb_address_t reduce_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);
    pnanovdb_address_t reduce_val_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), channel_count * workgroup_count * 4u);
    pnanovdb_address_t reduce_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), channel_count * workgroup_count * 4u);
    return tmp_addr;
}

PNANOVDB_FORCE_INLINE void pnanovdb_scan_keyed_float(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t val_count,
    pnanovdb_uint32_t channel_count,
    pnanovdb_buf_t val_buf, pnanovdb_address_t val_addr,
    pnanovdb_buf_t key_buf, pnanovdb_address_t key_addr,
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr,
    pnanovdb_buf_t result_buf, pnanovdb_address_t result_addr
)
{
    pnanovdb_uint32_t workgroup_count = (val_count + 1023u) / 1024u;

    pnanovdb_address_t reduce_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), workgroup_count * 4u);
    pnanovdb_address_t reduce_val_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), channel_count * workgroup_count * 4u);
    pnanovdb_address_t reduce_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), channel_count * workgroup_count * 4u);

    // reduce per workgroup
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t val_max_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 1024u + 1023u;
        pnanovdb_uint32_t key_max = 0xFFFFFFFF;
        if (val_max_idx < val_count)
        {
            key_max = pnanovdb_read_uint32_index(key_buf, key_addr, val_max_idx);
        }

        for (pnanovdb_uint32_t channel_idx = 0u; channel_idx < channel_count; channel_idx++)
        {
            pnanovdb_vec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

                pnanovdb_uvec4_t key;
                if (4u * val4_idx < val_count)
                {
                    key = pnanovdb_read_uvec4_index(key_buf, key_addr, val4_idx);
                }
                if (4u * val4_idx + 0u >= val_count) { key.x = 0xFFFFFFFF; }
                if (4u * val4_idx + 1u >= val_count) { key.y = 0xFFFFFFFF; }
                if (4u * val4_idx + 2u >= val_count) { key.z = 0xFFFFFFFF; }
                if (4u * val4_idx + 3u >= val_count) { key.w = 0xFFFFFFFF; }

                // fetch val only as needed and invalidate non matching keys
                pnanovdb_bool_t any_match = key.x == key_max || key.y == key_max || key.z == key_max || key.w == key_max;
                if (any_match && 4u * val4_idx < val_count)
                {
                    val[vidx].x = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 0u) + channel_idx);
                    val[vidx].y = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 1u) + channel_idx);
                    val[vidx].z = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 2u) + channel_idx);
                    val[vidx].w = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 3u) + channel_idx);
                }
                if (4u * val4_idx + 0u >= val_count || key.x != key_max) { val[vidx].x = 0u; }
                if (4u * val4_idx + 1u >= val_count || key.y != key_max) { val[vidx].y = 0u; }
                if (4u * val4_idx + 2u >= val_count || key.z != key_max) { val[vidx].z = 0u; }
                if (4u * val4_idx + 3u >= val_count || key.w != key_max) { val[vidx].w = 0u; }
            }

            float total_count = 0.f;
            pnanovdb_workgroup_reduce_float(vidx_offset, smem_buf, smem_addr, val, PNANOVDB_REF(total_count));

            if (vidx_offset == 0u)
            {
                pnanovdb_uint32_t reduce_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;

                pnanovdb_write_float_index(tmp_buf, reduce_val_addr, channel_count * reduce_idx + channel_idx, total_count);
                pnanovdb_write_uint32_index(tmp_buf, reduce_key_addr, reduce_idx, key_max);
            }
        }
    }

    // scan across workgroup counts
    pnanovdb_sync(sync_state, 1u, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t scan_pass_count = (workgroup_count + 1023u) / 1024u;
        for (pnanovdb_uint32_t channel_idx = 0u; channel_idx < channel_count; channel_idx++)
        {
            float global_offset = 0.f;
            pnanovdb_uint32_t global_key = 0xFFFFFFFF;
            for (pnanovdb_uint32_t scan_pass_idx = 0u; scan_pass_idx < scan_pass_count; scan_pass_idx++)
            {
                pnanovdb_vec4_t reduce[PNANOVDB_COMPUTE_VECTOR_WIDTH];
                pnanovdb_uvec4_t key[PNANOVDB_COMPUTE_VECTOR_WIDTH];
                pnanovdb_uint32_t vidx;
                for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
                {
                    pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                    pnanovdb_uint32_t reduce_idx = scan_pass_idx * 256u + thread_idx;

                    reduce[vidx].x = 0.f;
                    reduce[vidx].y = 0.f;
                    reduce[vidx].z = 0.f;
                    reduce[vidx].w = 0.f;
                    key[vidx].x = 0xFFFFFFFF;
                    key[vidx].y = 0xFFFFFFFF;
                    key[vidx].z = 0xFFFFFFFF;
                    key[vidx].w = 0xFFFFFFFF;
                    if (4u * reduce_idx < workgroup_count)
                    {
                        reduce[vidx].x = pnanovdb_read_float_index(tmp_buf, reduce_val_addr, channel_count * (4u * reduce_idx + 0u) + channel_idx);
                        reduce[vidx].y = pnanovdb_read_float_index(tmp_buf, reduce_val_addr, channel_count * (4u * reduce_idx + 1u) + channel_idx);
                        reduce[vidx].z = pnanovdb_read_float_index(tmp_buf, reduce_val_addr, channel_count * (4u * reduce_idx + 2u) + channel_idx);
                        reduce[vidx].w = pnanovdb_read_float_index(tmp_buf, reduce_val_addr, channel_count * (4u * reduce_idx + 3u) + channel_idx);
                        key[vidx] = pnanovdb_read_uvec4_index(tmp_buf, reduce_key_addr, reduce_idx);
                    }
                }

                float last_count = 0.f;
                pnanovdb_uint32_t last_key = 0u;
                pnanovdb_vec4_t reduce_scan[PNANOVDB_COMPUTE_VECTOR_WIDTH];
                pnanovdb_workgroup_scan_keyed_float(vidx_offset, smem_buf, smem_addr, reduce, key, reduce_scan, PNANOVDB_REF(last_count), PNANOVDB_REF(last_key));

                for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
                {
                    pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                    pnanovdb_uint32_t reduce_idx = scan_pass_idx * 256u + thread_idx;

                    if (key[vidx].x == global_key) { reduce_scan[vidx].x += global_offset; }
                    if (key[vidx].y == global_key) { reduce_scan[vidx].y += global_offset; };
                    if (key[vidx].z == global_key) { reduce_scan[vidx].z += global_offset; };
                    if (key[vidx].w == global_key) { reduce_scan[vidx].w += global_offset; };

                    if (4u * reduce_idx < workgroup_count)
                    {
                        pnanovdb_write_float_index(tmp_buf, reduce_scan_addr, channel_count * (4u * reduce_idx + 0u) + channel_idx, reduce_scan[vidx].x);
                        pnanovdb_write_float_index(tmp_buf, reduce_scan_addr, channel_count * (4u * reduce_idx + 1u) + channel_idx, reduce_scan[vidx].y);
                        pnanovdb_write_float_index(tmp_buf, reduce_scan_addr, channel_count * (4u * reduce_idx + 2u) + channel_idx, reduce_scan[vidx].z);
                        pnanovdb_write_float_index(tmp_buf, reduce_scan_addr, channel_count * (4u * reduce_idx + 3u) + channel_idx, reduce_scan[vidx].w);
                    }
                }

                if (last_key == global_key)
                {
                    global_offset += last_count;
                }
                else
                {
                    global_offset = last_count;
                    global_key = last_key;
                }
            }
        }
    }

    // local scan + add workgroup count
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        for (pnanovdb_uint32_t channel_idx = 0u; channel_idx < channel_count; channel_idx++)
        {
            pnanovdb_vec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uvec4_t key[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

                if (4u * val4_idx < val_count)
                {
                    key[vidx] = pnanovdb_read_uvec4_index(key_buf, key_addr, val4_idx);
                }
                if (4u * val4_idx + 0u >= val_count) { key[vidx].x = 0xFFFFFFFF; }
                if (4u * val4_idx + 1u >= val_count) { key[vidx].y = 0xFFFFFFFF; }
                if (4u * val4_idx + 2u >= val_count) { key[vidx].z = 0xFFFFFFFF; }
                if (4u * val4_idx + 3u >= val_count) { key[vidx].w = 0xFFFFFFFF; }

                if (4u * val4_idx < val_count)
                {
                    val[vidx].x = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 0u) + channel_idx);
                    val[vidx].y = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 1u) + channel_idx);
                    val[vidx].z = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 2u) + channel_idx);
                    val[vidx].w = pnanovdb_read_float_index(val_buf, val_addr, channel_count * (4u * val4_idx + 3u) + channel_idx);
                }
                if (4u * val4_idx + 0u >= val_count) { val[vidx].x = 0u; }
                if (4u * val4_idx + 1u >= val_count) { val[vidx].y = 0u; }
                if (4u * val4_idx + 2u >= val_count) { val[vidx].z = 0u; }
                if (4u * val4_idx + 3u >= val_count) { val[vidx].w = 0u; }
            }

            float last_count = 0u;
            pnanovdb_uint32_t last_key = 0u;
            pnanovdb_vec4_t val_scan[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_workgroup_scan_keyed_float(vidx_offset, smem_buf, smem_addr, val, key, val_scan, PNANOVDB_REF(last_count), PNANOVDB_REF(last_key));

            pnanovdb_uint32_t workgroup_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;
            float global_offset = 0.f;
            pnanovdb_uint32_t global_key = 0xFFFFFFFF;
            if (workgroup_idx > 0u)
            {
                global_offset = pnanovdb_read_float_index(tmp_buf, reduce_scan_addr, channel_count * (workgroup_idx - 1u) + channel_idx);
                global_key = pnanovdb_read_uint32_index(tmp_buf, reduce_key_addr, workgroup_idx - 1u);
            }

            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t val4_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

                if (key[vidx].x == global_key) { val_scan[vidx].x += global_offset; }
                if (key[vidx].y == global_key) { val_scan[vidx].y += global_offset; }
                if (key[vidx].z == global_key) { val_scan[vidx].z += global_offset; }
                if (key[vidx].w == global_key) { val_scan[vidx].w += global_offset; }

                if (4u * val4_idx < val_count)
                {
                    pnanovdb_write_float_index(result_buf, result_addr, channel_count * (4u * val4_idx + 0u) + channel_idx, val_scan[vidx].x);
                    pnanovdb_write_float_index(result_buf, result_addr, channel_count * (4u * val4_idx + 1u) + channel_idx, val_scan[vidx].y);
                    pnanovdb_write_float_index(result_buf, result_addr, channel_count * (4u * val4_idx + 2u) + channel_idx, val_scan[vidx].z);
                    pnanovdb_write_float_index(result_buf, result_addr, channel_count * (4u * val4_idx + 3u) + channel_idx, val_scan[vidx].w);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
}

#endif // end of NANOVDB_PUTILS_RASTER_H_HAS_BEEN_INCLUDED
