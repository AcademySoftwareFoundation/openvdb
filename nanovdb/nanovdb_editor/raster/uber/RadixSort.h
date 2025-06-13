
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/putil/RadixSort.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
            of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PUTILS_RADIX_SORT_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_RADIX_SORT_H_HAS_BEEN_INCLUDED

#include "nanovdb/PNanoVDB.h"
#include "nanovdb/putil/Workgroup.h"
#include "nanovdb/putil/Reduce.h"
#include "nanovdb/putil/Scan.h"

PNANOVDB_FORCE_INLINE void pnanovdb_radix_count(PNANOVDB_INOUT(pnanovdb_uvec4_t) counter, pnanovdb_uint32_t bucket)
{
    if (bucket == 0) PNANOVDB_DEREF(counter).x += (1 << 0);
    if (bucket == 1) PNANOVDB_DEREF(counter).x += (1 << 16);
    if (bucket == 2) PNANOVDB_DEREF(counter).y += (1 << 0);
    if (bucket == 3) PNANOVDB_DEREF(counter).y += (1 << 16);
    if (bucket == 4) PNANOVDB_DEREF(counter).z += (1 << 0);
    if (bucket == 5) PNANOVDB_DEREF(counter).z += (1 << 16);
    if (bucket == 6) PNANOVDB_DEREF(counter).w += (1 << 0);
    if (bucket == 7) PNANOVDB_DEREF(counter).w += (1 << 16);
    if (bucket == 8) PNANOVDB_DEREF(counter).x += (1 << 8);
    if (bucket == 9) PNANOVDB_DEREF(counter).x += (1 << 24);
    if (bucket == 10) PNANOVDB_DEREF(counter).y += (1 << 8);
    if (bucket == 11) PNANOVDB_DEREF(counter).y += (1 << 24);
    if (bucket == 12) PNANOVDB_DEREF(counter).z += (1 << 8);
    if (bucket == 13) PNANOVDB_DEREF(counter).z += (1 << 24);
    if (bucket == 14) PNANOVDB_DEREF(counter).w += (1 << 8);
    if (bucket == 15) PNANOVDB_DEREF(counter).w += (1 << 24);
}

PNANOVDB_FORCE_INLINE pnanovdb_uvec4_t pnanovdb_radix_expand8to16l(pnanovdb_uvec4_t counter)
{
    pnanovdb_uvec4_t counterL;
    counterL.x = counter.x & 0x00FF00FF;
    counterL.y = counter.y & 0x00FF00FF;
    counterL.z = counter.z & 0x00FF00FF;
    counterL.w = counter.w & 0x00FF00FF;
    return counterL;
}

PNANOVDB_FORCE_INLINE pnanovdb_uvec4_t pnanovdb_radix_expand8to16h(pnanovdb_uvec4_t counter)
{
    pnanovdb_uvec4_t counterH;
    counterH.x = (counter.x & 0xFF00FF00) >> 8;
    counterH.y = (counter.y & 0xFF00FF00) >> 8;
    counterH.z = (counter.z & 0xFF00FF00) >> 8;
    counterH.w = (counter.w & 0xFF00FF00) >> 8;
    return counterH;
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_radix_sort1_smem_size_in_words = 256u + 64u;

PNANOVDB_FORCE_INLINE void pnanovdb_radix_sort1(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_uint32_t workgroup_idx,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uint32_t pass_start,
    pnanovdb_uint32_t pass_num_bits,
    pnanovdb_uint32_t pass_mask,
    pnanovdb_uint32_t key_count,
    pnanovdb_buf_t key_in_buf, pnanovdb_address_t key_in_addr,
    pnanovdb_buf_t counters_out_buf, pnanovdb_address_t counters_out_addr
)
{
    pnanovdb_address_t scount0_addr = smem_addr;
    pnanovdb_address_t scount1_addr = pnanovdb_address_offset(scount0_addr, 256u * 4u);

    pnanovdb_uint32_t workgroup_count = (key_count + 1023u) / 1024u;

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        pnanovdb_uvec4_t local_count = {};

        pnanovdb_uint32_t key4_idx = workgroup_idx * 256u + thread_idx;

        pnanovdb_uvec4_t key_local;
        if (4u * key4_idx < key_count)
        {
            key_local = pnanovdb_read_uvec4_index(key_in_buf, key_in_addr, key4_idx);
        }
        if (4u * key4_idx + 0u >= key_count) { key_local.x = 0xFFFFFFFF; }
        if (4u * key4_idx + 1u >= key_count) { key_local.y = 0xFFFFFFFF; }
        if (4u * key4_idx + 2u >= key_count) { key_local.z = 0xFFFFFFFF; }
        if (4u * key4_idx + 3u >= key_count) { key_local.w = 0xFFFFFFFF; }
        key_local.x = (key_local.x >> pass_start) & pass_mask;
        key_local.y = (key_local.y >> pass_start) & pass_mask;
        key_local.z = (key_local.z >> pass_start) & pass_mask;
        key_local.w = (key_local.w >> pass_start) & pass_mask;
        pnanovdb_radix_count(PNANOVDB_REF(local_count), key_local.x);
        pnanovdb_radix_count(PNANOVDB_REF(local_count), key_local.y);
        pnanovdb_radix_count(PNANOVDB_REF(local_count), key_local.z);
        pnanovdb_radix_count(PNANOVDB_REF(local_count), key_local.w);

        pnanovdb_write_uvec4(smem_buf, pnanovdb_address_offset(scount0_addr, thread_idx * 16u), local_count);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 256u / 4u)
        {
            pnanovdb_uvec4_t local_count = pnanovdb_read_uvec4_index(smem_buf, scount0_addr, thread_idx);
            local_count = pnanovdb_uvec4_add(local_count, pnanovdb_read_uvec4_index(smem_buf, scount0_addr, thread_idx + 1 * 256u / 4u));
            local_count = pnanovdb_uvec4_add(local_count, pnanovdb_read_uvec4_index(smem_buf, scount0_addr, thread_idx + 2 * 256u / 4u));
            local_count = pnanovdb_uvec4_add(local_count, pnanovdb_read_uvec4_index(smem_buf, scount0_addr, thread_idx + 3 * 256u / 4u));
            pnanovdb_write_uvec4(smem_buf, pnanovdb_address_offset(scount1_addr, thread_idx * 16u), local_count);
        }
    }
    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        // expand to 16-bit from 8-bit
        if (thread_idx < 256u / 16u)
        {
            pnanovdb_uvec4_t local_count;
            pnanovdb_uvec4_t local_counth;
            pnanovdb_uvec4_t local_raw;

            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, thread_idx);
            local_count = pnanovdb_radix_expand8to16l(local_raw);
            local_counth = pnanovdb_radix_expand8to16h(local_raw);
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, thread_idx + 1 * 256u / 16);
            local_count = pnanovdb_uvec4_add(local_count, pnanovdb_radix_expand8to16l(local_raw));
            local_counth = pnanovdb_uvec4_add(local_counth, pnanovdb_radix_expand8to16h(local_raw));
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, thread_idx + 2 * 256u / 16);
            local_count = pnanovdb_uvec4_add(local_count, pnanovdb_radix_expand8to16l(local_raw));
            local_counth = pnanovdb_uvec4_add(local_counth, pnanovdb_radix_expand8to16h(local_raw));
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, thread_idx + 3 * 256u / 16);
            local_count = pnanovdb_uvec4_add(local_count, pnanovdb_radix_expand8to16l(local_raw));
            local_counth = pnanovdb_uvec4_add(local_counth, pnanovdb_radix_expand8to16h(local_raw));
            pnanovdb_write_uvec4(smem_buf, pnanovdb_address_offset(scount0_addr, thread_idx * 16u), local_count);
            pnanovdb_write_uvec4(smem_buf, pnanovdb_address_offset(scount0_addr, (thread_idx + 256u / 16u) * 16u), local_counth);
        }
    }
    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        // two sets of 16 uint4 left to be reduced
        pnanovdb_uint32_t set_id = thread_idx / (256u / 2u);
        pnanovdb_uint32_t set_lane_id = thread_idx & (256u / 2u - 1u);
        if (set_lane_id < 256u / 64u)
        {
            pnanovdb_uvec4_t local_count;
            pnanovdb_uvec4_t local_raw;

            pnanovdb_uint32_t offset = set_id * 256u / 16u;
            local_count = pnanovdb_read_uvec4_index(smem_buf, scount0_addr, set_lane_id + offset);
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount0_addr, set_lane_id + 1 * 256u / 64 + offset);
            local_count = pnanovdb_uvec4_add(local_count, local_raw);
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount0_addr, set_lane_id + 2 * 256u / 64 + offset);
            local_count = pnanovdb_uvec4_add(local_count, local_raw);
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount0_addr, set_lane_id + 3 * 256u / 64 + offset);
            local_count = pnanovdb_uvec4_add(local_count, local_raw);

            pnanovdb_write_uvec4(smem_buf, pnanovdb_address_offset(scount1_addr, (set_lane_id + set_id * 256u / 64) * 16u), local_count);
        }
    }
    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        pnanovdb_uint32_t set_id = thread_idx / (256u / 2u);
        pnanovdb_uint32_t set_lane_id = thread_idx & (256u / 2u - 1u);

        // two sets of 4 uint4 left to be reduced
        if (set_lane_id == 0)
        {
            pnanovdb_uvec4_t local_count;
            pnanovdb_uvec4_t local_raw;

            pnanovdb_uint32_t offset = set_id * 256u / 64u;
            local_count = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, 0 + offset);
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, 1 + offset);
            local_count = pnanovdb_uvec4_add(local_count, local_raw);
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, 2 + offset);
            local_count = pnanovdb_uvec4_add(local_count, local_raw);
            local_raw = pnanovdb_read_uvec4_index(smem_buf, scount1_addr, 3 + offset);
            local_count = pnanovdb_uvec4_add(local_count, local_raw);

            pnanovdb_uint32_t bucket_offset = 8 * set_id;

            // output counter values for global scan
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 0) * workgroup_count + workgroup_idx,
                (local_count.x & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 1) * workgroup_count + workgroup_idx,
                (local_count.x & 0xFFFF0000) >> 16);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 2) * workgroup_count + workgroup_idx,
                (local_count.y & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 3) * workgroup_count + workgroup_idx,
                (local_count.y & 0xFFFF0000) >> 16);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 4) * workgroup_count + workgroup_idx,
                (local_count.z & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 5) * workgroup_count + workgroup_idx,
                (local_count.z & 0xFFFF0000) >> 16);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 6) * workgroup_count + workgroup_idx,
                (local_count.w & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 7) * workgroup_count + workgroup_idx,
                (local_count.w & 0xFFFF0000) >> 16);

            // output counter values for local scan
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 0) + 16 * (workgroup_count + workgroup_idx),
                (local_count.x & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 1) + 16 * (workgroup_count + workgroup_idx),
                (local_count.x & 0xFFFF0000) >> 16);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 2) + 16 * (workgroup_count + workgroup_idx),
                (local_count.y & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 3) + 16 * (workgroup_count + workgroup_idx),
                (local_count.y & 0xFFFF0000) >> 16);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 4) + 16 * (workgroup_count + workgroup_idx),
                (local_count.z & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 5) + 16 * (workgroup_count + workgroup_idx),
                (local_count.z & 0xFFFF0000) >> 16);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 6) + 16 * (workgroup_count + workgroup_idx),
                (local_count.w & 0x0000FFFF) >> 0);
            pnanovdb_write_uint32_index(
                counters_out_buf, counters_out_addr, (bucket_offset + 7) + 16 * (workgroup_count + workgroup_idx),
                (local_count.w & 0xFFFF0000) >> 16);
        }
    }
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_radix_sort2_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_radix_sort2(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_uint32_t workgroup_idx,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uint32_t key_count,
    pnanovdb_buf_t counters_in_buf, pnanovdb_address_t counters_in_addr,
    pnanovdb_buf_t counters_out_buf, pnanovdb_address_t counters_out_addr
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);

    pnanovdb_uint32_t workgroup_count = (key_count + 1023u) / 1024u;

    if (workgroup_idx == 0u)
    {
        pnanovdb_uint32_t numPasses = ((16u * workgroup_count) / 4 + 256u - 1) / (256u);

        pnanovdb_uint32_t global_offset[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_uint32_t idx[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            idx[vidx] = thread_idx;
            global_offset[vidx] = 0u;
        }

        for (pnanovdb_uint32_t passID = 0; passID < numPasses; passID++)
        {
            pnanovdb_uint32_t workgroup_offset = 0u;

            pnanovdb_uvec4_t count_local[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                count_local[vidx].x = 0u;
                count_local[vidx].y = 0u;
                count_local[vidx].z = 0u;
                count_local[vidx].w = 0u;
                if (idx[vidx] < (16u * workgroup_count) / 4)
                {
                    count_local[vidx] = pnanovdb_read_uvec4_index(counters_in_buf, counters_in_addr, idx[vidx]);
                }
            }

            pnanovdb_uvec4_t count_global[PNANOVDB_COMPUTE_VECTOR_WIDTH];

            pnanovdb_workgroup_scan(vidx_offset, smem_buf, smem_addr, count_local, count_global, PNANOVDB_REF(workgroup_offset));

            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                // make exclusive
                count_global[vidx].x -= count_local[vidx].x;
                count_global[vidx].y -= count_local[vidx].y;
                count_global[vidx].z -= count_local[vidx].z;
                count_global[vidx].w -= count_local[vidx].w;

                count_global[vidx].x += global_offset[vidx];
                count_global[vidx].y += global_offset[vidx];
                count_global[vidx].z += global_offset[vidx];
                count_global[vidx].w += global_offset[vidx];

                if (idx[vidx] < (16u * workgroup_count) / 4)
                {
                    pnanovdb_write_uvec4_index(counters_out_buf, counters_out_addr, idx[vidx], count_global[vidx]);
                }

                global_offset[vidx] += workgroup_offset;

                idx[vidx] += 256u;
            }
        }
    }
    else if (workgroup_idx == 1u)
    {
        pnanovdb_uint32_t numPasses = ((16u * workgroup_count) / 4 + 256u - 1) / (256u);

        pnanovdb_uint32_t idx[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            idx[vidx] = thread_idx;
        }

        for (pnanovdb_uint32_t passID = 0; passID < numPasses; passID++)
        {

            pnanovdb_uvec4_t count_local[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                count_local[vidx].x = 0u;
                count_local[vidx].y = 0u;
                count_local[vidx].z = 0u;
                count_local[vidx].w = 0u;
                if (idx[vidx] < (16u * workgroup_count) / 4)
                {
                    count_local[vidx] = pnanovdb_read_uvec4_index(counters_in_buf, counters_in_addr, idx[vidx] + (16u * workgroup_count) / 4);
                }

                pnanovdb_uint32_t count_local_32 = count_local[vidx].x + count_local[vidx].y + count_local[vidx].z + count_local[vidx].w;
                pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, count_local_32);
            }

            pnanovdb_workgroup_sync();

            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

                pnanovdb_uint32_t scan_total = 0;
                if ((thread_idx & 3u) >= 1u)
                {
                    scan_total += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, 4u * (thread_idx / 4u) + 0u);
                }
                if ((thread_idx & 3u) >= 2u)
                {
                    scan_total += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, 4u * (thread_idx / 4u) + 1u);
                }
                if ((thread_idx & 3u) >= 3u)
                {
                    scan_total += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, 4u * (thread_idx / 4u) + 2u);
                }

                // make final scan exclusive
                count_local[vidx].w = count_local[vidx].z + count_local[vidx].y + count_local[vidx].x + scan_total;
                count_local[vidx].z = count_local[vidx].y + count_local[vidx].x + scan_total;
                count_local[vidx].y = count_local[vidx].x + scan_total;
                count_local[vidx].x = scan_total;

                if (idx[vidx] < (16u * workgroup_count) / 4)
                {
                    pnanovdb_write_uvec4_index(counters_out_buf, counters_out_addr, idx[vidx] + (16u * workgroup_count) / 4, count_local[vidx]);
                }

                idx[vidx] += 256u;
            }
        }
    }
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_radix_split4_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_radix_split4(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uvec4_t pred[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT_ARRAY(pnanovdb_uvec4_t) rank[PNANOVDB_COMPUTE_VECTOR_WIDTH]
)
{
    pnanovdb_uint32_t total_count;
    pnanovdb_uvec4_t scan_val[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_workgroup_scan(vidx_offset, smem_buf, smem_addr, pred, scan_val, PNANOVDB_REF(total_count));

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        rank[vidx].x = bool(pred[vidx].x) ? scan_val[vidx].x - 1 : 4 * thread_idx + 0 - scan_val[vidx].x + total_count;
        rank[vidx].y = bool(pred[vidx].y) ? scan_val[vidx].y - 1 : 4 * thread_idx + 1 - scan_val[vidx].y + total_count;
        rank[vidx].z = bool(pred[vidx].z) ? scan_val[vidx].z - 1 : 4 * thread_idx + 2 - scan_val[vidx].z + total_count;
        rank[vidx].w = bool(pred[vidx].w) ? scan_val[vidx].w - 1 : 4 * thread_idx + 3 - scan_val[vidx].w + total_count;
    }
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_radix_sort3_smem_size_in_words = 256u + 256u + 1024u + 1024u;

PNANOVDB_FORCE_INLINE void pnanovdb_radix_sort3(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_uint32_t workgroup_idx,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uint32_t pass_start,
    pnanovdb_uint32_t pass_num_bits,
    pnanovdb_uint32_t pass_mask,
    pnanovdb_uint32_t key_count,
    pnanovdb_buf_t key_in_buf, pnanovdb_address_t key_in_addr,
    pnanovdb_buf_t val_in_buf, pnanovdb_address_t val_in_addr,
    pnanovdb_buf_t counters_in_buf, pnanovdb_address_t counters_in_addr,
    pnanovdb_buf_t key_out_buf, pnanovdb_address_t key_out_addr,
    pnanovdb_buf_t val_out_buf, pnanovdb_address_t val_out_addr
)
{
    pnanovdb_address_t split4_smem_addr = smem_addr;
    pnanovdb_address_t skey_addr = pnanovdb_address_offset(split4_smem_addr, (256u + 256u) * 4u);
    pnanovdb_address_t sval_addr = pnanovdb_address_offset(skey_addr, (1024u) * 4u);

    pnanovdb_uint32_t workgroup_count = (key_count + 1023u) / 1024u;

    pnanovdb_uvec4_t key_local[PNANOVDB_COMPUTE_VECTOR_WIDTH];
    pnanovdb_uvec4_t val_local[PNANOVDB_COMPUTE_VECTOR_WIDTH];
    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        pnanovdb_uint32_t key4_idx = 256u * workgroup_idx + thread_idx;

        if (4u * key4_idx < key_count)
        {
            key_local[vidx] = pnanovdb_read_uvec4_index(key_in_buf, key_in_addr, key4_idx);
            val_local[vidx] = pnanovdb_read_uvec4_index(val_in_buf, val_in_addr, key4_idx);
        }
        if (4u * key4_idx + 0u >= key_count) { key_local[vidx].x = 0xFFFFFFFF; val_local[vidx].x = 0u; }
        if (4u * key4_idx + 1u >= key_count) { key_local[vidx].y = 0xFFFFFFFF; val_local[vidx].y = 0u; }
        if (4u * key4_idx + 2u >= key_count) { key_local[vidx].z = 0xFFFFFFFF; val_local[vidx].z = 0u; }
        if (4u * key4_idx + 3u >= key_count) { key_local[vidx].w = 0xFFFFFFFF; val_local[vidx].w = 0u; }
    }

    for (pnanovdb_uint32_t pass_id = pass_start; pass_id < pass_start + pass_num_bits; pass_id++)
    {
        pnanovdb_uvec4_t alloc_val[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            alloc_val[vidx].x = ((key_local[vidx].x >> pass_id) & 1) ^ 1u;
            alloc_val[vidx].y = ((key_local[vidx].y >> pass_id) & 1) ^ 1u;
            alloc_val[vidx].z = ((key_local[vidx].z >> pass_id) & 1) ^ 1u;
            alloc_val[vidx].w = ((key_local[vidx].w >> pass_id) & 1) ^ 1u;
        }

        pnanovdb_uvec4_t alloc_idx[PNANOVDB_COMPUTE_VECTOR_WIDTH];
        pnanovdb_radix_split4(vidx_offset, smem_buf, split4_smem_addr, alloc_val, alloc_idx);

        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_write_uint32_index(smem_buf, skey_addr, alloc_idx[vidx].x, key_local[vidx].x);
            pnanovdb_write_uint32_index(smem_buf, skey_addr, alloc_idx[vidx].y, key_local[vidx].y);
            pnanovdb_write_uint32_index(smem_buf, skey_addr, alloc_idx[vidx].z, key_local[vidx].z);
            pnanovdb_write_uint32_index(smem_buf, skey_addr, alloc_idx[vidx].w, key_local[vidx].w);

            pnanovdb_write_uint32_index(smem_buf, sval_addr, alloc_idx[vidx].x, val_local[vidx].x);
            pnanovdb_write_uint32_index(smem_buf, sval_addr, alloc_idx[vidx].y, val_local[vidx].y);
            pnanovdb_write_uint32_index(smem_buf, sval_addr, alloc_idx[vidx].z, val_local[vidx].z);
            pnanovdb_write_uint32_index(smem_buf, sval_addr, alloc_idx[vidx].w, val_local[vidx].w);
        }

        pnanovdb_workgroup_sync();

        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            key_local[vidx].x = pnanovdb_read_uint32_index(smem_buf, skey_addr, 4 * thread_idx + 0);
            key_local[vidx].y = pnanovdb_read_uint32_index(smem_buf, skey_addr, 4 * thread_idx + 1);
            key_local[vidx].z = pnanovdb_read_uint32_index(smem_buf, skey_addr, 4 * thread_idx + 2);
            key_local[vidx].w = pnanovdb_read_uint32_index(smem_buf, skey_addr, 4 * thread_idx + 3);

            val_local[vidx].x = pnanovdb_read_uint32_index(smem_buf, sval_addr, 4 * thread_idx + 0);
            val_local[vidx].y = pnanovdb_read_uint32_index(smem_buf, sval_addr, 4 * thread_idx + 1);
            val_local[vidx].z = pnanovdb_read_uint32_index(smem_buf, sval_addr, 4 * thread_idx + 2);
            val_local[vidx].w = pnanovdb_read_uint32_index(smem_buf, sval_addr, 4 * thread_idx + 3);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        for (pnanovdb_uint32_t shared_idx = thread_idx; shared_idx < 4 * 256u; shared_idx += 256u)
        {
            pnanovdb_uint32_t key = pnanovdb_read_uint32_index(smem_buf, skey_addr, shared_idx);
            pnanovdb_uint32_t bucket_idx = (key >> pass_start) & pass_mask;

            pnanovdb_uint32_t dst_idx = shared_idx -
                pnanovdb_read_uint32_index(counters_in_buf, counters_in_addr, bucket_idx + 16 * (workgroup_count + workgroup_idx)) +
                pnanovdb_read_uint32_index(counters_in_buf, counters_in_addr, bucket_idx * workgroup_count + workgroup_idx);

            pnanovdb_uint32_t val = pnanovdb_read_uint32_index(smem_buf, sval_addr, shared_idx);

            if (dst_idx < key_count)
            {
                pnanovdb_write_uint32_index(key_out_buf, key_out_addr, dst_idx, key);
                pnanovdb_write_uint32_index(val_out_buf, val_out_addr, dst_idx, val);
            }
        }
    }
}

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_radix_sort_smem_size_in_words = 256u + 256u + 1024u + 1024u + 2u;

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_radix_sort_compute_tmp_size(pnanovdb_uint32_t key_count)
{
    pnanovdb_uint32_t workgroup_count = (key_count + 1023u) / 1024u;
    pnanovdb_uint32_t total_counters = 2u * 2u * 16u * workgroup_count;

    pnanovdb_address_t tmp_addr = pnanovdb_address_null();
    pnanovdb_address_t counters_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), total_counters * 4u);
    pnanovdb_address_t key_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t val_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);

    return tmp_addr;
}

PNANOVDB_FORCE_INLINE void pnanovdb_radix_sort(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t num_key_bits,
    pnanovdb_uint32_t key_count,
    pnanovdb_buf_t key_buf, pnanovdb_address_t key_addr,
    pnanovdb_buf_t val_buf, pnanovdb_address_t val_addr,
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr
)
{
    pnanovdb_uint32_t workgroup_count = (key_count + 1023u) / 1024u;
    pnanovdb_uint32_t total_counters = 2u * 2u * 16u * workgroup_count;

    pnanovdb_address_t counters_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), total_counters * 4u);
    pnanovdb_address_t key_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t val_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);

    pnanovdb_address_t counters_a_addr = counters_addr;
    pnanovdb_address_t counters_b_addr = pnanovdb_address_offset(counters_addr, (2u * 16u * workgroup_count) * 4u);

    for (pnanovdb_uint32_t pass_id = 0; pass_id < num_key_bits; pass_id+=4u)
    {
        pnanovdb_uint32_t pass_start = pass_id;
        pnanovdb_uint32_t pass_num_bits = num_key_bits - pass_id;
        if (pass_num_bits > 4u)
        {
            pass_num_bits = 4u;
        }
        pnanovdb_uint32_t pass_mask = (1u << pass_num_bits) - 1u;

        pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_radix_sort1(
                PNANOVDB_DEREF(sync_state).vidx_offset, PNANOVDB_DEREF(sync_state).workgroup_idx,
                smem_buf, smem_addr,
                pass_start, pass_num_bits, pass_mask, key_count,
                key_buf, key_addr, tmp_buf, counters_a_addr
            );
        }

        pnanovdb_sync(sync_state, 2u, smem_buf, smem_sync_addr, sync_buf, sync_addr);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_radix_sort2(
                PNANOVDB_DEREF(sync_state).vidx_offset, PNANOVDB_DEREF(sync_state).workgroup_idx,
                smem_buf, smem_addr,
                key_count,
                tmp_buf, counters_a_addr, tmp_buf, counters_b_addr
            );
        }

        pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_radix_sort3(
                PNANOVDB_DEREF(sync_state).vidx_offset, PNANOVDB_DEREF(sync_state).workgroup_idx,
                smem_buf, smem_addr,
                pass_start, pass_num_bits, pass_mask, key_count,
                key_buf, key_addr, val_buf, val_addr,
                tmp_buf, counters_b_addr,
                tmp_buf, key_tmp_addr, tmp_buf, val_tmp_addr
            );
        }
        pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

        pass_id+=4u;

        pass_start = pass_id;
        pass_num_bits = 0u;
        if (pass_id < num_key_bits)
        {
            pass_num_bits = num_key_bits - pass_id;
        }
        if (pass_num_bits > 4u)
        {
            pass_num_bits = 4u;
        }
        pass_mask = (1u << pass_num_bits) - 1u;

        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_radix_sort1(
                PNANOVDB_DEREF(sync_state).vidx_offset, PNANOVDB_DEREF(sync_state).workgroup_idx,
                smem_buf, smem_addr,
                pass_start, pass_num_bits, pass_mask, key_count,
                tmp_buf, key_tmp_addr, tmp_buf, counters_a_addr
            );
        }

        pnanovdb_sync(sync_state, 2u, smem_buf, smem_sync_addr, sync_buf, sync_addr);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_radix_sort2(
                PNANOVDB_DEREF(sync_state).vidx_offset, PNANOVDB_DEREF(sync_state).workgroup_idx,
                smem_buf, smem_addr,
                key_count,
                tmp_buf, counters_a_addr, tmp_buf, counters_b_addr
            );
        }

        pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_radix_sort3(
                PNANOVDB_DEREF(sync_state).vidx_offset, PNANOVDB_DEREF(sync_state).workgroup_idx,
                smem_buf, smem_addr,
                pass_start, pass_num_bits, pass_mask, key_count,
                tmp_buf, key_tmp_addr, tmp_buf, val_tmp_addr,
                tmp_buf, counters_b_addr,
                key_buf, key_addr, val_buf, val_addr
            );
        }
        pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    }
}


PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_radix_sort_96_smem_size_in_words = 256u + 256u + 1024u + 1024u + 2u;

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_radix_sort_96_compute_tmp_size(pnanovdb_uint32_t key_count, pnanovdb_uint32_t element_size)
{
    pnanovdb_address_t tmp_addr = pnanovdb_address_null();
    pnanovdb_address_t radix_sort_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_radix_sort_compute_tmp_size(key_count));
    pnanovdb_address_t key_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t val_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t copy_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), 3u * key_count * 4u);
    pnanovdb_address_t copy_val_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * element_size);
    return tmp_addr;
}

PNANOVDB_FORCE_INLINE void pnanovdb_radix_sort_96(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t key_count,
    pnanovdb_uint32_t element_size,
    pnanovdb_buf_t key_buf, pnanovdb_address_t key_addr,
    pnanovdb_buf_t val_buf, pnanovdb_address_t val_addr,
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr
)
{
    pnanovdb_uint32_t workgroup_count_256 = (key_count + 255u) / 256u;

    pnanovdb_address_t radix_sort_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_radix_sort_compute_tmp_size(key_count));
    pnanovdb_address_t key_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t val_tmp_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t copy_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), 3u * key_count * 4u);
    pnanovdb_address_t copy_val_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * element_size);

    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count_256);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t global_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (global_idx < key_count)
            {
                pnanovdb_uint32_t key_a = pnanovdb_read_uint32_index(key_buf, key_addr, 3u * global_idx + 0u);

                pnanovdb_write_uint32_index(tmp_buf, key_tmp_addr, global_idx, key_a);
                pnanovdb_write_uint32_index(tmp_buf, val_tmp_addr, global_idx, global_idx);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count_256, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    pnanovdb_radix_sort(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        32u,
        key_count,
        tmp_buf, key_tmp_addr,
        tmp_buf, val_tmp_addr,
        tmp_buf, radix_sort_addr
    );

    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count_256);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t global_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (global_idx < key_count)
            {
                pnanovdb_uint32_t src_idx = pnanovdb_read_uint32_index(tmp_buf, val_tmp_addr, global_idx);
                pnanovdb_uint32_t key_b = pnanovdb_read_uint32_index(key_buf, key_addr, 3u * src_idx + 1u);

                pnanovdb_write_uint32_index(tmp_buf, key_tmp_addr, global_idx, key_b);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count_256, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    pnanovdb_radix_sort(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        32u,
        key_count,
        tmp_buf, key_tmp_addr,
        tmp_buf, val_tmp_addr,
        tmp_buf, radix_sort_addr
    );

    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count_256);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t global_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (global_idx < key_count)
            {
                pnanovdb_uint32_t src_idx = pnanovdb_read_uint32_index(tmp_buf, val_tmp_addr, global_idx);
                pnanovdb_uint32_t key_c = pnanovdb_read_uint32_index(key_buf, key_addr, 3u * src_idx + 2u);

                pnanovdb_write_uint32_index(tmp_buf, key_tmp_addr, global_idx, key_c);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count_256, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    pnanovdb_radix_sort(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        32u,
        key_count,
        tmp_buf, key_tmp_addr,
        tmp_buf, val_tmp_addr,
        tmp_buf, radix_sort_addr
    );

    // scatter original 96-bit keys and 32-bit values out to tmp memory
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count_256);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t dst_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (dst_idx < key_count)
            {
                pnanovdb_uint32_t src_idx = pnanovdb_read_uint32_index(tmp_buf, val_tmp_addr, dst_idx);

                pnanovdb_uint32_t sub_idx;
                for (sub_idx = 0u; sub_idx < 3u; sub_idx++)
                {
                    pnanovdb_uint32_t val = pnanovdb_read_uint32_index(key_buf, key_addr, 3u * src_idx + sub_idx);

                    pnanovdb_write_uint32_index(tmp_buf, copy_key_addr, 3u * dst_idx + sub_idx, val);
                }
                for (sub_idx = 0u; sub_idx < element_size; sub_idx+=4u)
                {
                    pnanovdb_uint32_t val = pnanovdb_read_uint32_index(val_buf, val_addr, (element_size / 4u) * src_idx + (sub_idx / 4u));

                    pnanovdb_write_uint32_index(tmp_buf, copy_val_addr, (element_size / 4u) * dst_idx + (sub_idx / 4u), val);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count_256, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // copy scattered keys back to input key buffer
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count_256);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t dst_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (dst_idx < key_count)
            {
                pnanovdb_uint32_t sub_idx;
                for (sub_idx = 0u; sub_idx < 3u; sub_idx++)
                {
                    pnanovdb_uint32_t val = pnanovdb_read_uint32_index(tmp_buf, copy_key_addr, 3u * dst_idx + sub_idx);

                    pnanovdb_write_uint32_index(key_buf, key_addr, 3u * dst_idx + sub_idx, val);
                }
                for (sub_idx = 0u; sub_idx < element_size; sub_idx+=4u)
                {
                    pnanovdb_uint32_t val = pnanovdb_read_uint32_index(tmp_buf, copy_val_addr, (element_size / 4u) * dst_idx + (sub_idx / 4u));

                    pnanovdb_write_uint32_index(val_buf, val_addr, (element_size / 4u) * dst_idx + (sub_idx / 4u), val);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count_256, smem_buf, smem_sync_addr, sync_buf, sync_addr);
}

#endif // end of NANOVDB_PUTILS_RADIX_SORT_H_HAS_BEEN_INCLUDED
