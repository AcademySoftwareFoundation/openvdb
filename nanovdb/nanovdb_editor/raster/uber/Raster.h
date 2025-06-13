
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/putil/Raster.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
            of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PUTILS_RASTER_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_RASTER_H_HAS_BEEN_INCLUDED

#include "nanovdb/PNanoVDB.h"
#include "nanovdb/putil/Workgroup.h"
#include "nanovdb/putil/Reduce.h"
#include "nanovdb/putil/Scan.h"
#include "nanovdb/putil/RadixSort.h"

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_coord_to_sort_key(pnanovdb_coord_t ijk)
{
    pnanovdb_coord_t key;
    // leaf[0, 9) lower[9 to 21) upper[21 to 36) root[36 to 96)
    key.x =
        (ijk.z & 7) | ((ijk.y & 7) << 3) | ((ijk.x & 7) << 6) |
        (((ijk.z >> 3) & 15) << 9) | (((ijk.y >> 3) & 15) << 13) | (((ijk.x >> 3) & 15) << 17) |
        (((ijk.z >> 7) & 31) << 21) | (((ijk.y >> 7) & 31) << 26) | (((ijk.x >> 7) & 31) << 31);
    key.y =
        (((ijk.x >> 7) & 31) >> 1) |
        (((ijk.z >> 12) & 0x000FFFFF) << 4) | (((ijk.y >> 12) & 0x000FFFFF) << 24);
    key.z =
        (((ijk.y >> 12) & 0x000FFFFF) >> 8) | (((ijk.x >> 12) & 0x000FFFFF) << 12);
    return key;
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_sort_key_to_coord(pnanovdb_coord_t key)
{
    pnanovdb_coord_t ijk;
    // leaf[0, 9) lower[9 to 21) upper[21 to 36) root[36 to 96)
    ijk.z =
        (key.x & 7) |
        (((key.x >> 9) & 15) << 3) |
        (((key.x >> 21) & 31) << 7) |
        (((key.y >> 4) & 0x000FFFFF) << 12);
    ijk.y =
        ((key.x >> 3) & 7) |
        (((key.x >> 13) & 15) << 3) |
        (((key.x >> 26) & 31) << 7) |
        (((key.y >> 24) & 0x000000FF) << 12) |
        (((key.z << 8) & 0x000FFF00) << 12);
    ijk.x =
        ((key.x >> 6) & 7) |
        (((key.x >> 17) & 15) << 3) |
        (((key.x >> 31) & 1) << 7) |
        (((key.y << 1) & 30) << 7) |
        (((key.z >> 12) & 0x000FFFFF) << 12);
    return ijk;
}

// ----------------------------- Fanout for Raster Footprint ---------------------------

struct pnanovdb_raster_fanout_state_t
{
    pnanovdb_uint32_t bbox_idx_offset;
    pnanovdb_uint32_t total_request_count;
    pnanovdb_uint32_t active_count;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_raster_fanout_state_t)

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_raster_fanout_compute_tmp_size(pnanovdb_uint32_t bbox_count)
{
    pnanovdb_address_t tmp_addr = pnanovdb_address_null();
    pnanovdb_address_t tmp_scan_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(bbox_count));
    pnanovdb_address_t tmp_voxel_count_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_voxel_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_scatter_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    return tmp_addr;
}

PNANOVDB_FORCE_INLINE void pnanovdb_raster_fanout_init(
    PNANOVDB_INOUT(pnanovdb_raster_fanout_state_t) state,
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t bbox_count,
    pnanovdb_buf_t bbox_buf, pnanovdb_address_t bbox_addr,  // 6u * bbox_count * 4u bytes
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr
)
{
    pnanovdb_uint32_t workgroup_count = (bbox_count + 255u) / 256u;

    pnanovdb_address_t tmp_scan_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(bbox_count));
    pnanovdb_address_t tmp_voxel_count_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_voxel_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_scatter_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);

    // compute leaf counts per bbox
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t bbox_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

            pnanovdb_coord_t ijk_min = pnanovdb_read_coord_index(bbox_buf, bbox_addr, 2u * bbox_idx + 0u);
            pnanovdb_coord_t ijk_max = pnanovdb_read_coord_index(bbox_buf, bbox_addr, 2u * bbox_idx + 1u);

            pnanovdb_uint32_t voxel_count = 0u;
            if (bbox_idx < bbox_count && ijk_min.x <= ijk_max.x && ijk_min.y <= ijk_max.y && ijk_min.z <= ijk_max.z)
            {
                // ijk_max is inclusive
                voxel_count =
                    (ijk_max.x + 1 - ijk_min.x) *
                    (ijk_max.y + 1 - ijk_min.y) *
                    (ijk_max.z + 1 - ijk_min.z);
            }

            pnanovdb_write_uint32_index(tmp_buf, tmp_voxel_count_addr, bbox_idx, voxel_count);
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // scan leaf counts to batch
    pnanovdb_uint32_t total_request_count = 0u;
    pnanovdb_scan(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        bbox_count,
        tmp_buf, tmp_voxel_count_addr,
        tmp_buf, tmp_scan_addr,
        tmp_buf, tmp_voxel_scan_addr,
        PNANOVDB_REF(total_request_count)
    );

    // update state
    PNANOVDB_DEREF(state).bbox_idx_offset = 0u;
    PNANOVDB_DEREF(state).total_request_count = total_request_count;
    PNANOVDB_DEREF(state).active_count = 0u;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_raster_fanout_update(
    PNANOVDB_INOUT(pnanovdb_raster_fanout_state_t) state,
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t bbox_count,
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr,
    pnanovdb_buf_t out_bbox_idx_buf, pnanovdb_address_t out_bbox_idx_addr,
    pnanovdb_buf_t out_sub_idx_buf, pnanovdb_address_t out_sub_idx_addr
)
{
    pnanovdb_uint32_t workgroup_count = (bbox_count + 255u) / 256u;

    if (PNANOVDB_DEREF(state).bbox_idx_offset >= PNANOVDB_DEREF(state).total_request_count)
    {
        PNANOVDB_DEREF(state).active_count = 0u;
        return PNANOVDB_FALSE;
    }

    pnanovdb_address_t tmp_scan_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(bbox_count));
    pnanovdb_address_t tmp_voxel_count_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_voxel_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_scatter_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);

    // clear scatter target
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t bbox_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (bbox_idx < bbox_count)
            {
                pnanovdb_write_uint32_index(tmp_buf, tmp_scatter_addr, bbox_idx, 0u);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // scatter start
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t bbox_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (bbox_idx < bbox_count)
            {
                pnanovdb_uint32_t bbox_voxel_count = pnanovdb_read_uint32_index(tmp_buf, tmp_voxel_count_addr, bbox_idx);
                pnanovdb_uint32_t bbox_voxel_scan = pnanovdb_read_uint32_index(tmp_buf, tmp_voxel_scan_addr, bbox_idx);

                pnanovdb_uint32_t range_min = PNANOVDB_DEREF(state).bbox_idx_offset;
                pnanovdb_uint32_t range_max = PNANOVDB_DEREF(state).bbox_idx_offset + bbox_count;

                pnanovdb_uint32_t bbox_range_min = bbox_voxel_scan - bbox_voxel_count; // scan was inclusive
                pnanovdb_uint32_t bbox_range_max = bbox_voxel_scan;

                // clip to range
                if (bbox_range_min < range_min && bbox_range_max > range_min)
                {
                    bbox_range_min = range_min;
                }
                // scatter to range
                if (bbox_range_min >= range_min && bbox_range_min < range_max)
                {
                    pnanovdb_uint32_t scatter_idx = bbox_range_min - range_min;
                    pnanovdb_write_uint32_index(tmp_buf, tmp_scatter_addr, scatter_idx, bbox_idx);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // max scan to fill in zeroes
    pnanovdb_uint32_t fill_max = 0u;
    pnanovdb_scan_max(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        bbox_count,
        tmp_buf, tmp_scatter_addr,
        tmp_buf, tmp_scan_addr,
        tmp_buf, out_bbox_idx_addr,
        PNANOVDB_REF(fill_max)
    );

    // compute sub idx
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t work_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (work_idx < bbox_count)
            {
                pnanovdb_uint32_t bbox_idx = pnanovdb_read_uint32_index(tmp_buf, out_bbox_idx_addr, work_idx);

                // use voxel_count and scan info to find sub index
                pnanovdb_uint32_t bbox_voxel_count = pnanovdb_read_uint32_index(tmp_buf, tmp_voxel_count_addr, bbox_idx);
                pnanovdb_uint32_t bbox_voxel_scan = pnanovdb_read_uint32_index(tmp_buf, tmp_voxel_scan_addr, bbox_idx);

                pnanovdb_uint32_t bbox_range_min = bbox_voxel_scan - bbox_voxel_count; // scan was inclusive
                pnanovdb_uint32_t bbox_range_max = bbox_voxel_scan;

                pnanovdb_uint32_t global_idx = work_idx + PNANOVDB_DEREF(state).bbox_idx_offset;

                pnanovdb_uint32_t sub_idx = ~0u;
                if (global_idx >= bbox_range_min && global_idx < bbox_range_max)
                {
                    sub_idx = global_idx - bbox_range_min;
                }

                pnanovdb_write_uint32_index(tmp_buf, out_sub_idx_addr, work_idx, sub_idx);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    pnanovdb_uint32_t remaining_work = PNANOVDB_DEREF(state).total_request_count - PNANOVDB_DEREF(state).bbox_idx_offset;
    // update state
    PNANOVDB_DEREF(state).bbox_idx_offset += bbox_count;
    PNANOVDB_DEREF(state).active_count = bbox_count < remaining_work ? bbox_count : remaining_work;

    return PNANOVDB_TRUE;
}

// ----------------------------- Raster and Reduce ---------------------------

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_raster_and_reduce_compute_tmp_size(pnanovdb_uint32_t bbox_count, pnanovdb_uint32_t key_max_count)
{
    pnanovdb_address_t tmp_addr = pnanovdb_address_null();
    pnanovdb_address_t tmp_raster_fanout_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_raster_fanout_compute_tmp_size(bbox_count));
    pnanovdb_address_t tmp_bbox_idx_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_sub_idx_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  3u * bbox_count * 4u);
    pnanovdb_address_t tmp_radix_sort_96_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_radix_sort_96_compute_tmp_size(bbox_count + key_max_count, 16u));
    pnanovdb_address_t tmp_compact_mask_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), (bbox_count + key_max_count) * 4u);
    pnanovdb_address_t tmp_compact_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), (bbox_count + key_max_count) * 4u);
    pnanovdb_address_t tmp_scan_keyed_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_keyed_float_compute_tmp_size(bbox_count + key_max_count, 4u));
    pnanovdb_address_t tmp_scan_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(bbox_count + key_max_count));
    pnanovdb_address_t tmp_val4_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), 4u * (bbox_count + key_max_count) * 4u);
    pnanovdb_address_t tmp_key_copy_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), 3u * (bbox_count + key_max_count) * 4u);
    return tmp_addr;
}

PNANOVDB_FORCE_INLINE void pnanovdb_raster_and_reduce(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_uint32_t bbox_count,
    pnanovdb_uint32_t key_max_count,
    pnanovdb_buf_t bbox_buf, pnanovdb_address_t bbox_addr,  // 6u * bbox_count * 4u bytes
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr,
    pnanovdb_buf_t out_key_buf, pnanovdb_address_t out_key_addr, // 3u * (bbox_count + key_max_count) * 4u bytes
    pnanovdb_buf_t out_val4_buf, pnanovdb_address_t out_val4_addr, // 4u * (bbox_count + key_max_count) * 4u bytes
    pnanovdb_buf_t out_key_count_buf, pnanovdb_address_t out_key_count_addr // 4u bytes
)
{
    pnanovdb_uint32_t workgroup_count = (bbox_count + 255u) / 256u;

    pnanovdb_address_t tmp_raster_fanout_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_raster_fanout_compute_tmp_size(bbox_count));
    pnanovdb_address_t tmp_bbox_idx_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_sub_idx_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  bbox_count * 4u);
    pnanovdb_address_t tmp_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr),  3u * bbox_count * 4u);
    pnanovdb_address_t tmp_radix_sort_96_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_radix_sort_96_compute_tmp_size(bbox_count + key_max_count, 16u));
    pnanovdb_address_t tmp_compact_mask_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), (bbox_count + key_max_count) * 4u);
    pnanovdb_address_t tmp_compact_key_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), (bbox_count + key_max_count) * 4u);
    pnanovdb_address_t tmp_scan_keyed_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_keyed_float_compute_tmp_size(bbox_count + key_max_count, 4u));
    pnanovdb_address_t tmp_scan_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(bbox_count + key_max_count));
    pnanovdb_address_t tmp_scan_result_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), (bbox_count + key_max_count) * 4u);
    pnanovdb_address_t tmp_val4_scan_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), 4u * (bbox_count + key_max_count) * 4u);
    pnanovdb_address_t tmp_key_copy_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), 3u * (bbox_count + key_max_count) * 4u);

    // clear output key count
    pnanovdb_sync_set_workgroup_count(sync_state, 1u);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        if (vidx_offset == 0u)
        {
            pnanovdb_write_uint32(out_key_count_buf, out_key_count_addr, 0u);
        }
    }
    pnanovdb_sync(sync_state, 1u, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    pnanovdb_raster_fanout_state_t fanout_state;
    pnanovdb_raster_fanout_init(
        PNANOVDB_REF(fanout_state),
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        bbox_count,
        bbox_buf, bbox_addr,
        tmp_buf, tmp_raster_fanout_addr
    );

    // execute batches
    pnanovdb_bool_t work_available = true;
    while (work_available)
    {
        work_available = pnanovdb_raster_fanout_update(
            PNANOVDB_REF(fanout_state),
            vidx_offset,
            smem_buf, smem_addr, smem_sync_addr,
            sync_state,
            sync_buf, sync_addr,
            bbox_count,
            tmp_buf, tmp_raster_fanout_addr,
            tmp_buf, tmp_bbox_idx_addr,
            tmp_buf, tmp_sub_idx_addr
        );

        if (work_available)
        {
            // compute sort key and val4 and merge to accumulation array
            while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
            {
                pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
                pnanovdb_uint32_t vidx;
                for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
                {
                    pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                    pnanovdb_uint32_t work_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
                    if (work_idx < bbox_count)
                    {
                        pnanovdb_uint32_t bbox_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_bbox_idx_addr, work_idx);
                        pnanovdb_uint32_t sub_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_sub_idx_addr, work_idx);

                        pnanovdb_coord_t ijk_min = pnanovdb_read_coord_index(bbox_buf, bbox_addr, 2u * bbox_idx + 0u);
                        pnanovdb_coord_t ijk_max = pnanovdb_read_coord_index(bbox_buf, bbox_addr, 2u * bbox_idx + 1u);

                        pnanovdb_coord_t key = { -1, -1, -1 };
                        pnanovdb_vec4_t val4 = { 1.f, 1.f, 1.f, 1.f };
                        if (ijk_min.x <= ijk_max.x && ijk_min.y <= ijk_max.y && ijk_min.z <= ijk_max.z)
                        {
                            pnanovdb_int32_t range_k = ijk_max.z + 1 - ijk_min.z;
                            pnanovdb_int32_t range_j = ijk_max.y + 1 - ijk_min.y;
                            pnanovdb_int32_t range_i = ijk_max.x + 1 - ijk_min.x;

                            pnanovdb_int32_t pnanovdb_voxel_k = sub_idx % range_k;
                            pnanovdb_int32_t pnanovdb_voxel_j = (sub_idx / range_k) % range_j;
                            pnanovdb_int32_t pnanovdb_voxel_i = sub_idx / (range_k * range_j);

                            pnanovdb_coord_t ijk;
                            ijk.z = pnanovdb_voxel_k + ijk_min.z;
                            ijk.y = pnanovdb_voxel_j + ijk_min.y;
                            ijk.x = pnanovdb_voxel_i + ijk_min.x;

                            if (pnanovdb_voxel_i < range_i)
                            {
                                key = pnanovdb_coord_to_sort_key(ijk);
                            }
                        }

                        pnanovdb_uint32_t merge_idx = pnanovdb_read_uint32(out_key_count_buf, out_key_count_addr);
                        pnanovdb_uint32_t dst_idx = work_idx + merge_idx;
                        if (dst_idx < bbox_count + key_max_count)
                        {
                            pnanovdb_write_coord_index(out_key_buf, out_key_addr, dst_idx, PNANOVDB_REF(key));
                            pnanovdb_write_vec4_index(out_val4_buf, out_val4_addr, dst_idx, val4);
                        }
                    }
                }
            }
            pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
        }

        pnanovdb_uint32_t merged_count = pnanovdb_read_uint32(out_key_count_buf, out_key_count_addr) + fanout_state.active_count;
        pnanovdb_uint32_t merged_workgroup_count = (merged_count + 255u) / 256u;

        // if next merge will fit, radix sort and compact can be deferred
        if (work_available && merged_count <= key_max_count)
        {
            pnanovdb_sync_set_workgroup_count(sync_state, 1u);
            while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
            {
                if (vidx_offset == 0u)
                {
                    pnanovdb_write_uint32(out_key_count_buf, out_key_count_addr, merged_count);
                }
            }
            pnanovdb_sync(sync_state, 1u, smem_buf, smem_sync_addr, sync_buf, sync_addr);

            continue;
        }

        // radix sort 96-bit
        pnanovdb_radix_sort_96(
            vidx_offset,
            smem_buf, smem_addr, smem_sync_addr,
            sync_state,
            sync_buf, sync_addr,
            merged_count,
            16u,
            out_key_buf, out_key_addr,
            out_val4_buf, out_val4_addr,
            tmp_buf, tmp_radix_sort_96_addr
        );

        // compute compaction mask
        pnanovdb_sync_set_workgroup_count(sync_state, merged_workgroup_count);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                pnanovdb_uint32_t work_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

                if (work_idx < merged_count)
                {
                    pnanovdb_coord_t key = pnanovdb_read_coord_index(out_key_buf, out_key_addr, work_idx);

                    pnanovdb_bool_t is_unique = PNANOVDB_TRUE;
                    if (work_idx > 0u)
                    {
                        pnanovdb_coord_t key_cmp = pnanovdb_read_coord_index(out_key_buf, out_key_addr, work_idx - 1u);

                        is_unique = key.x != key_cmp.x || key.y != key_cmp.y || key.z != key_cmp.z;
                    }

                    pnanovdb_write_uint32_index(tmp_buf, tmp_compact_mask_addr, work_idx, is_unique ? 1u : 0u);
                }
            }
        }
        pnanovdb_sync(sync_state, merged_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

        // scan compaction mask to produce 32-bit unique keys
        pnanovdb_uint32_t compacted_count = 0u;
        pnanovdb_scan(
            vidx_offset,
            smem_buf, smem_addr, smem_sync_addr,
            sync_state,
            sync_buf, sync_addr,
            merged_count,
            tmp_buf, tmp_compact_mask_addr,
            tmp_buf, tmp_scan_addr,
            tmp_buf, tmp_compact_key_addr,
            PNANOVDB_REF(compacted_count)
        );

        // keyed scan to merge values with matching keys
        pnanovdb_scan_keyed_float(
            vidx_offset,
            smem_buf, smem_addr, smem_sync_addr,
            sync_state,
            sync_buf, sync_addr,
            merged_count,
            4u,
            out_val4_buf, out_val4_addr,
            tmp_buf, tmp_compact_key_addr,
            tmp_buf, tmp_scan_keyed_addr,
            tmp_buf, tmp_val4_scan_addr
        );

        // compute compaction mask for upper range of segment
        pnanovdb_sync_set_workgroup_count(sync_state, merged_workgroup_count);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                pnanovdb_uint32_t work_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

                if (work_idx < merged_count)
                {
                    pnanovdb_coord_t key = pnanovdb_read_coord_index(out_key_buf, out_key_addr, work_idx);

                    pnanovdb_bool_t is_unique = PNANOVDB_TRUE;
                    if (work_idx < merged_count - 1u)
                    {
                        pnanovdb_coord_t key_cmp = pnanovdb_read_coord_index(out_key_buf, out_key_addr, work_idx + 1u);

                        is_unique = key.x != key_cmp.x || key.y != key_cmp.y || key.z != key_cmp.z;
                    }

                    pnanovdb_write_uint32_index(tmp_buf, tmp_compact_mask_addr, work_idx, is_unique ? 1u : 0u);
                }
            }
        }
        pnanovdb_sync(sync_state, merged_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

        // compute compaction dst indices
        pnanovdb_scan(
            vidx_offset,
            smem_buf, smem_addr, smem_sync_addr,
            sync_state,
            sync_buf, sync_addr,
            merged_count,
            tmp_buf, tmp_compact_mask_addr,
            tmp_buf, tmp_scan_addr,
            tmp_buf, tmp_scan_result_addr,
            PNANOVDB_REF(compacted_count)
        );

        // copy key to allow compaction
        pnanovdb_sync_set_workgroup_count(sync_state, merged_workgroup_count);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                pnanovdb_uint32_t work_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
                if (work_idx < merged_count)
                {
                    pnanovdb_coord_t key = pnanovdb_read_coord_index(out_key_buf, out_key_addr, work_idx);
                    pnanovdb_write_coord_index(tmp_buf, tmp_key_copy_addr, work_idx, PNANOVDB_REF(key));
                }
            }
        }
        pnanovdb_sync(sync_state, merged_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

        // compact scatter
        pnanovdb_sync_set_workgroup_count(sync_state, merged_workgroup_count);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                pnanovdb_uint32_t work_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;

                if (work_idx < merged_count)
                {
                    pnanovdb_uint32_t compact_mask = pnanovdb_read_uint32_index(tmp_buf, tmp_compact_mask_addr, work_idx);
                    if (compact_mask != 0u)
                    {
                        pnanovdb_uint32_t compact_scan = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_result_addr, work_idx);

                        pnanovdb_uint32_t dst_idx = compact_scan - compact_mask;

                        pnanovdb_coord_t key = pnanovdb_read_coord_index(tmp_buf, tmp_key_copy_addr, work_idx);
                        pnanovdb_write_coord_index(out_key_buf, out_key_addr, dst_idx, PNANOVDB_REF(key));

                        pnanovdb_vec4_t val = pnanovdb_read_vec4_index(tmp_buf, tmp_val4_scan_addr, work_idx);
                        pnanovdb_write_vec4_index(out_val4_buf, out_val4_addr, dst_idx, val);
                    }
                }
            }
        }
        pnanovdb_sync(sync_state, merged_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

        pnanovdb_sync_set_workgroup_count(sync_state, 1u);
        while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
        {
            if (vidx_offset == 0u)
            {
                pnanovdb_write_uint32(out_key_count_buf, out_key_count_addr, compacted_count);
            }
        }
        pnanovdb_sync(sync_state, 1u, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    }
}

// ----------------------------- Form NanoVDB from Sorted Keys ---------------------------

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_build_from_sorted_keys_compute_tmp_size(pnanovdb_uint32_t key_count)
{
    pnanovdb_address_t tmp_addr = pnanovdb_address_null();
    pnanovdb_address_t tmp_scan_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(key_count));
    pnanovdb_address_t tmp_scan_mask_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_scan_result_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_root_tile_list_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_upper_list_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_lower_list_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_leaf_list_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    return tmp_addr;
}

PNANOVDB_FORCE_INLINE void pnanovdb_build_from_sorted_keys(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr, pnanovdb_address_t smem_sync_addr,
    PNANOVDB_INOUT(pnanovdb_sync_state_t) sync_state,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr,
    pnanovdb_grid_type_t grid_type,
    pnanovdb_uint32_t key_count,
    pnanovdb_vec3_t voxel_size,
    pnanovdb_address_t nanovdb_max_size,
    pnanovdb_buf_t key_buf, pnanovdb_address_t key_addr,  // 3u * key_count * 4u bytes
    pnanovdb_buf_t tmp_buf, pnanovdb_address_t tmp_addr,
    pnanovdb_buf_t out_nanovdb_buf, pnanovdb_address_t out_nanovdb_addr
)
{
    pnanovdb_uint32_t workgroup_count = (key_count + 255u) / 256u;

    pnanovdb_address_t tmp_scan_addr = pnanovdb_alloc_address_aligned(PNANOVDB_REF(tmp_addr), pnanovdb_scan_compute_tmp_size(key_count));
    pnanovdb_address_t tmp_scan_mask_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_scan_result_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_root_tile_upper_range_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_upper_lower_range_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_lower_leaf_range_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);
    pnanovdb_address_t tmp_leaf_key_range_addr = pnanovdb_alloc_address_aligned32(PNANOVDB_REF(tmp_addr), key_count * 4u);

    // allocate leaves
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t key_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (key_idx < key_count)
            {
                pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                pnanovdb_bool_t is_unique = PNANOVDB_TRUE;
                if (key_idx > 0u)
                {
                    pnanovdb_coord_t key_cmp = pnanovdb_read_coord_index(key_buf, key_addr, key_idx - 1u);
                    key.x = key.x & 0xFFFFFE00;
                    key.y = key.y & 0xFFFFFFFF;
                    key_cmp.x = key_cmp.x & 0xFFFFFE00;
                    key_cmp.y = key_cmp.y & 0xFFFFFFFF;
                    is_unique = key.x != key_cmp.x || key.y != key_cmp.y || key.z != key_cmp.z;
                }
                pnanovdb_write_uint32_index(tmp_buf, tmp_scan_mask_addr, key_idx, is_unique ? 1u : 0u);
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    // scan to allocate
    pnanovdb_uint32_t leaf_count = 0u;
    pnanovdb_scan(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        key_count,
        tmp_buf, tmp_scan_mask_addr,
        tmp_buf, tmp_scan_addr,
        tmp_buf, tmp_scan_result_addr,
        PNANOVDB_REF(leaf_count)
    );
    // scatter to form lists
    pnanovdb_sync_set_workgroup_count(sync_state, workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t key_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (key_idx < key_count)
            {
                pnanovdb_uint32_t mask = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_mask_addr, key_idx);
                if (mask != 0u)
                {
                    pnanovdb_uint32_t dst_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_result_addr, key_idx) - mask;
                    pnanovdb_write_uint32_index(tmp_buf, tmp_leaf_key_range_addr, dst_idx, key_idx);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // allocate lowers
    pnanovdb_uint32_t leaf_workgroup_count = (leaf_count + 255u) / 256u;
    pnanovdb_sync_set_workgroup_count(sync_state, leaf_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t leaf_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (leaf_idx < leaf_count)
            {
                pnanovdb_uint32_t key_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx);

                pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                pnanovdb_bool_t is_unique = PNANOVDB_TRUE;
                if (key_idx > 0u)
                {
                    pnanovdb_coord_t key_cmp = pnanovdb_read_coord_index(key_buf, key_addr, key_idx - 1u);
                    key.x = key.x & 0xFFE00000;
                    key.y = key.y & 0xFFFFFFFF;
                    key_cmp.x = key_cmp.x & 0xFFE00000;
                    key_cmp.y = key_cmp.y & 0xFFFFFFFF;
                    is_unique = key.x != key_cmp.x || key.y != key_cmp.y || key.z != key_cmp.z;
                }
                pnanovdb_write_uint32_index(tmp_buf, tmp_scan_mask_addr, leaf_idx, is_unique ? 1u : 0u);
            }
        }
    }
    pnanovdb_sync(sync_state, leaf_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    // scan to allocate
    pnanovdb_uint32_t lower_count = 0u;
    pnanovdb_scan(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        leaf_count,
        tmp_buf, tmp_scan_mask_addr,
        tmp_buf, tmp_scan_addr,
        tmp_buf, tmp_scan_result_addr,
        PNANOVDB_REF(lower_count)
    );
    // scatter to form lists
    pnanovdb_sync_set_workgroup_count(sync_state, leaf_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t leaf_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (leaf_idx < leaf_count)
            {
                pnanovdb_uint32_t mask = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_mask_addr, leaf_idx);
                if (mask != 0u)
                {
                    pnanovdb_uint32_t dst_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_result_addr, leaf_idx) - mask;
                    pnanovdb_write_uint32_index(tmp_buf, tmp_lower_leaf_range_addr, dst_idx, leaf_idx);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, leaf_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // allocate uppers
    pnanovdb_uint32_t lower_workgroup_count = (lower_count + 255u) / 256u;
    pnanovdb_sync_set_workgroup_count(sync_state, lower_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t lower_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (lower_idx < lower_count)
            {
                pnanovdb_uint32_t leaf_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_lower_leaf_range_addr, lower_idx);
                pnanovdb_uint32_t key_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx);

                pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                pnanovdb_bool_t is_unique = PNANOVDB_TRUE;
                if (key_idx > 0u)
                {
                    pnanovdb_coord_t key_cmp = pnanovdb_read_coord_index(key_buf, key_addr, key_idx - 1u);
                    key.x = key.x & 0x00000000;
                    key.y = key.y & 0xFFFFFFF0;
                    key_cmp.x = key_cmp.x & 0x00000000;
                    key_cmp.y = key_cmp.y & 0xFFFFFFF0;
                    is_unique = key.x != key_cmp.x || key.y != key_cmp.y || key.z != key_cmp.z;
                }
                pnanovdb_write_uint32_index(tmp_buf, tmp_scan_mask_addr, lower_idx, is_unique ? 1u : 0u);
            }
        }
    }
    pnanovdb_sync(sync_state, lower_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    // scan to allocate
    pnanovdb_uint32_t upper_count = 0u;
    pnanovdb_scan(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        lower_count,
        tmp_buf, tmp_scan_mask_addr,
        tmp_buf, tmp_scan_addr,
        tmp_buf, tmp_scan_result_addr,
        PNANOVDB_REF(upper_count)
    );
    // scatter to form lists
    pnanovdb_sync_set_workgroup_count(sync_state, lower_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t lower_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (lower_idx < lower_count)
            {
                pnanovdb_uint32_t mask = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_mask_addr, lower_idx);
                if (mask != 0u)
                {
                    pnanovdb_uint32_t dst_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_result_addr, lower_idx) - mask;
                    pnanovdb_write_uint32_index(tmp_buf, tmp_upper_lower_range_addr, dst_idx, lower_idx);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, lower_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // allocate root tiles
    pnanovdb_uint32_t upper_workgroup_count = (upper_count + 255u) / 256u;
    pnanovdb_sync_set_workgroup_count(sync_state, upper_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t upper_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (upper_idx < upper_count)
            {
                pnanovdb_uint32_t lower_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_upper_lower_range_addr, upper_idx);
                pnanovdb_uint32_t leaf_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_lower_leaf_range_addr, lower_idx);
                pnanovdb_uint32_t key_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx);

                pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                pnanovdb_bool_t is_unique = PNANOVDB_TRUE;
                if (key_idx > 0u)
                {
                    pnanovdb_coord_t key_cmp = pnanovdb_read_coord_index(key_buf, key_addr, key_idx - 1u);
                    key.x = key.x & 0x00000000;
                    key.y = key.y & 0xFFFFFFF0;
                    key_cmp.x = key_cmp.x & 0x00000000;
                    key_cmp.y = key_cmp.y & 0xFFFFFFF0;
                    is_unique = key.x != key_cmp.x || key.y != key_cmp.y || key.z != key_cmp.z;
                }
                pnanovdb_write_uint32_index(tmp_buf, tmp_scan_mask_addr, upper_idx, is_unique ? 1u : 0u);
            }
        }
    }
    pnanovdb_sync(sync_state, upper_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);
    // scan to allocate
    pnanovdb_uint32_t root_tile_count = 0u;
    pnanovdb_scan(
        vidx_offset,
        smem_buf, smem_addr, smem_sync_addr,
        sync_state,
        sync_buf, sync_addr,
        upper_count,
        tmp_buf, tmp_scan_mask_addr,
        tmp_buf, tmp_scan_addr,
        tmp_buf, tmp_scan_result_addr,
        PNANOVDB_REF(root_tile_count)
    );
    // scatter to form lists
    pnanovdb_sync_set_workgroup_count(sync_state, upper_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t upper_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (upper_idx < upper_count)
            {
                pnanovdb_uint32_t mask = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_mask_addr, upper_idx);
                if (mask != 0u)
                {
                    pnanovdb_uint32_t dst_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_scan_result_addr, upper_idx) - mask;
                    pnanovdb_write_uint32_index(tmp_buf, tmp_root_tile_upper_range_addr, dst_idx, upper_idx);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, upper_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    pnanovdb_address_t alloc_addr = pnanovdb_address_null();

    pnanovdb_grid_handle_t grid = { pnanovdb_alloc_address_aligned32(PNANOVDB_REF(alloc_addr), PNANOVDB_GRID_SIZE) };
    pnanovdb_tree_handle_t tree = { pnanovdb_alloc_address_aligned32(PNANOVDB_REF(alloc_addr), PNANOVDB_TREE_SIZE) };
    pnanovdb_root_handle_t root = { pnanovdb_alloc_address_aligned32(PNANOVDB_REF(alloc_addr), PNANOVDB_GRID_TYPE_GET(grid_type, root_size)) };

    pnanovdb_root_tile_handle_t first_root_tile = { pnanovdb_alloc_address_aligned(PNANOVDB_REF(alloc_addr),
        pnanovdb_address_offset_product(pnanovdb_address_null(), PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size), root_tile_count))
    };
    pnanovdb_upper_handle_t first_upper = { pnanovdb_alloc_address_aligned(PNANOVDB_REF(alloc_addr),
        pnanovdb_address_offset_product(pnanovdb_address_null(), PNANOVDB_GRID_TYPE_GET(grid_type, upper_size), upper_count))
    };
    pnanovdb_lower_handle_t first_lower = { pnanovdb_alloc_address_aligned(PNANOVDB_REF(alloc_addr),
        pnanovdb_address_offset_product(pnanovdb_address_null(), PNANOVDB_GRID_TYPE_GET(grid_type, lower_size), lower_count))
    };
    pnanovdb_leaf_handle_t first_leaf = { pnanovdb_alloc_address_aligned(PNANOVDB_REF(alloc_addr),
        pnanovdb_address_offset_product(pnanovdb_address_null(), PNANOVDB_GRID_TYPE_GET(grid_type, leaf_size), leaf_count))
    };

    // clear memory
    pnanovdb_uint32_t clear_count = pnanovdb_uint64_low((pnanovdb_address_diff(alloc_addr, pnanovdb_address_null())));
    pnanovdb_uint32_t clear_workgroup_count = (clear_count + 255u) / 256u;
    pnanovdb_sync_set_workgroup_count(sync_state, clear_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            pnanovdb_uint32_t work_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (work_idx < clear_count)
            {
                pnanovdb_write_uint32_index(out_nanovdb_buf, out_nanovdb_addr, work_idx, 0u);
            }
        }
    }
    pnanovdb_sync(sync_state, clear_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // write leaf
    pnanovdb_sync_set_workgroup_count(sync_state, leaf_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t leaf_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;
        pnanovdb_uint32_t key_begin_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx);
        pnanovdb_uint32_t key_end_idx = key_count;
        if (leaf_idx + 1u < leaf_count)
        {
            key_end_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx + 1u);
        }
        pnanovdb_leaf_handle_t leaf = {
            pnanovdb_address_offset_product(first_leaf.address, leaf_idx, PNANOVDB_GRID_TYPE_GET(grid_type, leaf_size))
        };
        pnanovdb_coord_t ijk_min = {PNANOVDB_INT_MAX, PNANOVDB_INT_MAX, PNANOVDB_INT_MAX};
        pnanovdb_coord_t ijk_max = {PNANOVDB_INT_MIN, PNANOVDB_INT_MIN, PNANOVDB_INT_MIN};
        for (pnanovdb_uint32_t key_idx_base = key_begin_idx; key_idx_base < key_end_idx; key_idx_base += 256u)
        {
            pnanovdb_uint32_t write_word_idx[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t write_word_mask[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                pnanovdb_uint32_t key_idx = key_idx_base + thread_idx;

                write_word_idx[vidx] = ~0u;
                write_word_mask[vidx] = 0u;

                if (key_idx < key_end_idx)
                {
                    pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                    pnanovdb_coord_t ijk = pnanovdb_sort_key_to_coord(key);
                    pnanovdb_uint32_t leaf_n = pnanovdb_leaf_coord_to_offset(PNANOVDB_REF(ijk));

                    ijk_min = pnanovdb_coord_min(ijk_min, ijk);
                    ijk_max = pnanovdb_coord_max(ijk_max, ijk);

                    write_word_idx[vidx] = leaf_n >> 5u;
                    write_word_mask[vidx] = (1u << (leaf_n & 31u));
                }
            }
            pnanovdb_workgroup_reduce_and_write_mask(
                vidx_offset,
                smem_buf, smem_addr,
                write_word_idx,
                write_word_mask,
                out_nanovdb_buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK)
            );
        }
        pnanovdb_workgroup_reduce_min_coord(
            vidx_offset,
            smem_buf, smem_addr,
            ijk_min,
            PNANOVDB_REF(ijk_min)
        );
        pnanovdb_workgroup_reduce_max_coord(
            vidx_offset,
            smem_buf, smem_addr,
            ijk_max,
            PNANOVDB_REF(ijk_max)
        );
        if (vidx_offset == 0u)
        {
            pnanovdb_leaf_set_bbox_min(out_nanovdb_buf, leaf, PNANOVDB_REF(ijk_min));
            pnanovdb_uint32_t dif_and_flags =
                pnanovdb_int32_as_uint32((ijk_max.x - ijk_min.x) & 255) |
                pnanovdb_int32_as_uint32(((ijk_max.y - ijk_min.y) & 255) << 8) |
                pnanovdb_int32_as_uint32(((ijk_max.z - ijk_min.z) & 255) << 16);
            pnanovdb_leaf_set_bbox_dif_and_flags(out_nanovdb_buf, leaf, dif_and_flags);
        }
    }
    pnanovdb_sync(sync_state, leaf_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // write lower
    pnanovdb_sync_set_workgroup_count(sync_state, lower_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t lower_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;
        pnanovdb_uint32_t leaf_begin_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_lower_leaf_range_addr, lower_idx);
        pnanovdb_uint32_t leaf_end_idx = leaf_count;
        if (lower_idx + 1u < lower_count)
        {
            leaf_end_idx =  pnanovdb_read_uint32_index(tmp_buf, tmp_lower_leaf_range_addr, lower_idx + 1u);
        }
        pnanovdb_lower_handle_t lower = {
            pnanovdb_address_offset_product(first_lower.address, lower_idx, PNANOVDB_GRID_TYPE_GET(grid_type, lower_size))
        };
        pnanovdb_coord_t ijk_min = {PNANOVDB_INT_MAX, PNANOVDB_INT_MAX, PNANOVDB_INT_MAX};
        pnanovdb_coord_t ijk_max = {PNANOVDB_INT_MIN, PNANOVDB_INT_MIN, PNANOVDB_INT_MIN};
        for (pnanovdb_uint32_t leaf_idx_base = leaf_begin_idx; leaf_idx_base < leaf_end_idx; leaf_idx_base += 256u)
        {
            pnanovdb_uint32_t write_word_idx[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t write_word_mask[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                pnanovdb_uint32_t leaf_idx = leaf_idx_base + thread_idx;

                write_word_idx[vidx] = ~0u;
                write_word_mask[vidx] = 0u;

                if (leaf_idx < leaf_end_idx)
                {
                    pnanovdb_uint32_t key_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx);

                    pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                    pnanovdb_coord_t ijk = pnanovdb_sort_key_to_coord(key);
                    pnanovdb_uint32_t lower_n = pnanovdb_lower_coord_to_offset(PNANOVDB_REF(ijk));

                    pnanovdb_leaf_handle_t leaf = {
                        pnanovdb_address_offset_product(first_leaf.address, leaf_idx, PNANOVDB_GRID_TYPE_GET(grid_type, leaf_size))
                    };

                    pnanovdb_coord_t leaf_ijk_min = pnanovdb_leaf_get_bbox_min(out_nanovdb_buf, leaf);
                    pnanovdb_uint32_t dif_and_flags = pnanovdb_leaf_get_bbox_dif_and_flags(out_nanovdb_buf, leaf);
                    pnanovdb_coord_t leaf_ijk_max = {
                        pnanovdb_uint32_as_int32(dif_and_flags & 255u) + leaf_ijk_min.x,
                        pnanovdb_uint32_as_int32((dif_and_flags >> 8u) & 255u) + leaf_ijk_min.y,
                        pnanovdb_uint32_as_int32((dif_and_flags >> 16u) & 255u) + leaf_ijk_min.z
                    };

                    ijk_min = pnanovdb_coord_min(ijk_min, leaf_ijk_min);
                    ijk_max = pnanovdb_coord_max(ijk_max, leaf_ijk_max);

                    write_word_idx[vidx] = lower_n >> 5u;
                    write_word_mask[vidx] = (1u << (lower_n & 31u));

                    pnanovdb_lower_set_child(grid_type, out_nanovdb_buf, lower, lower_n, leaf);
                }
            }
            pnanovdb_workgroup_reduce_and_write_mask(
                vidx_offset,
                smem_buf, smem_addr,
                write_word_idx,
                write_word_mask,
                out_nanovdb_buf, pnanovdb_address_offset(lower.address, PNANOVDB_LOWER_OFF_VALUE_MASK)
            );
            pnanovdb_workgroup_reduce_and_write_mask(
                vidx_offset,
                smem_buf, smem_addr,
                write_word_idx,
                write_word_mask,
                out_nanovdb_buf, pnanovdb_address_offset(lower.address, PNANOVDB_LOWER_OFF_CHILD_MASK)
            );
        }
        pnanovdb_workgroup_reduce_min_coord(
            vidx_offset,
            smem_buf, smem_addr,
            ijk_min,
            PNANOVDB_REF(ijk_min)
        );
        pnanovdb_workgroup_reduce_max_coord(
            vidx_offset,
            smem_buf, smem_addr,
            ijk_max,
            PNANOVDB_REF(ijk_max)
        );
        if (vidx_offset == 0u)
        {
            pnanovdb_lower_set_bbox_min(out_nanovdb_buf, lower, PNANOVDB_REF(ijk_min));
            pnanovdb_lower_set_bbox_max(out_nanovdb_buf, lower, PNANOVDB_REF(ijk_max));
        }
    }
    pnanovdb_sync(sync_state, lower_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // write upper
    pnanovdb_sync_set_workgroup_count(sync_state, upper_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t upper_idx = PNANOVDB_DEREF(sync_state).workgroup_idx;
        pnanovdb_uint32_t lower_begin_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_upper_lower_range_addr, upper_idx);
        pnanovdb_uint32_t lower_end_idx = lower_count;
        if (upper_idx + 1u < upper_count)
        {
            lower_end_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_upper_lower_range_addr, upper_idx + 1u);
        }
        pnanovdb_upper_handle_t upper = {
            pnanovdb_address_offset_product(first_upper.address, upper_idx, PNANOVDB_GRID_TYPE_GET(grid_type, upper_size))
        };
        pnanovdb_coord_t ijk_min = {PNANOVDB_INT_MAX, PNANOVDB_INT_MAX, PNANOVDB_INT_MAX};
        pnanovdb_coord_t ijk_max = {PNANOVDB_INT_MIN, PNANOVDB_INT_MIN, PNANOVDB_INT_MIN};
        for (pnanovdb_uint32_t lower_idx_base = lower_begin_idx; lower_idx_base < lower_end_idx; lower_idx_base += 256u)
        {
            pnanovdb_uint32_t write_word_idx[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t write_word_mask[PNANOVDB_COMPUTE_VECTOR_WIDTH];
            pnanovdb_uint32_t vidx;
            for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
            {
                pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
                pnanovdb_uint32_t lower_idx = lower_idx_base + thread_idx;

                write_word_idx[vidx] = ~0u;
                write_word_mask[vidx] = 0u;

                if (lower_idx < lower_end_idx)
                {
                    pnanovdb_uint32_t leaf_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_lower_leaf_range_addr, lower_idx);
                    pnanovdb_uint32_t key_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx);

                    pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                    pnanovdb_coord_t ijk = pnanovdb_sort_key_to_coord(key);
                    pnanovdb_uint32_t upper_n = pnanovdb_upper_coord_to_offset(PNANOVDB_REF(ijk));

                    pnanovdb_lower_handle_t lower = {
                        pnanovdb_address_offset_product(first_lower.address, lower_idx, PNANOVDB_GRID_TYPE_GET(grid_type, lower_size))
                    };

                    ijk_min = pnanovdb_coord_min(ijk_min, pnanovdb_lower_get_bbox_min(out_nanovdb_buf, lower));
                    ijk_max = pnanovdb_coord_max(ijk_max, pnanovdb_lower_get_bbox_max(out_nanovdb_buf, lower));

                    write_word_idx[vidx] = upper_n >> 5u;
                    write_word_mask[vidx] = (1u << (upper_n & 31u));

                    pnanovdb_upper_set_child(grid_type, out_nanovdb_buf, upper, upper_n, lower);
                }
            }
            pnanovdb_workgroup_reduce_and_write_mask(
                vidx_offset,
                smem_buf, smem_addr,
                write_word_idx,
                write_word_mask,
                out_nanovdb_buf, pnanovdb_address_offset(upper.address, PNANOVDB_UPPER_OFF_VALUE_MASK)
            );
            pnanovdb_workgroup_reduce_and_write_mask(
                vidx_offset,
                smem_buf, smem_addr,
                write_word_idx,
                write_word_mask,
                out_nanovdb_buf, pnanovdb_address_offset(upper.address, PNANOVDB_UPPER_OFF_CHILD_MASK)
            );
        }
        pnanovdb_workgroup_reduce_min_coord(
            vidx_offset,
            smem_buf, smem_addr,
            ijk_min,
            PNANOVDB_REF(ijk_min)
        );
        pnanovdb_workgroup_reduce_max_coord(
            vidx_offset,
            smem_buf, smem_addr,
            ijk_max,
            PNANOVDB_REF(ijk_max)
        );
        if (vidx_offset == 0u)
        {
            pnanovdb_upper_set_bbox_min(out_nanovdb_buf, upper, PNANOVDB_REF(ijk_min));
            pnanovdb_upper_set_bbox_max(out_nanovdb_buf, upper, PNANOVDB_REF(ijk_max));
        }
    }
    pnanovdb_sync(sync_state, upper_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // write root tile
    pnanovdb_uint32_t root_tile_workgroup_count = (root_tile_count + 255u) / 256u;
    pnanovdb_sync_set_workgroup_count(sync_state, root_tile_workgroup_count);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

            pnanovdb_uint32_t root_tile_idx = PNANOVDB_DEREF(sync_state).workgroup_idx * 256u + thread_idx;
            if (root_tile_idx < root_tile_count)
            {
                pnanovdb_uint32_t upper_begin_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_root_tile_upper_range_addr, root_tile_idx);
                pnanovdb_uint32_t upper_end_idx = upper_count;
                if (root_tile_idx + 1u < root_tile_count)
                {
                    upper_end_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_root_tile_upper_range_addr, root_tile_idx + 1u);
                }
                pnanovdb_root_tile_handle_t root_tile = {
                    pnanovdb_address_offset_product(first_root_tile.address, root_tile_idx, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size))
                };
                for (pnanovdb_uint32_t upper_idx = upper_begin_idx; upper_idx < upper_end_idx; upper_idx++)
                {
                    pnanovdb_uint32_t lower_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_upper_lower_range_addr, upper_idx);
                    pnanovdb_uint32_t leaf_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_lower_leaf_range_addr, lower_idx);
                    pnanovdb_uint32_t key_idx = pnanovdb_read_uint32_index(tmp_buf, tmp_leaf_key_range_addr, leaf_idx);

                    pnanovdb_coord_t key = pnanovdb_read_coord_index(key_buf, key_addr, key_idx);
                    pnanovdb_coord_t ijk = pnanovdb_sort_key_to_coord(key);

                    pnanovdb_upper_handle_t upper = {
                        pnanovdb_address_offset_product(first_upper.address, upper_idx, PNANOVDB_GRID_TYPE_GET(grid_type, upper_size))
                    };

                    pnanovdb_root_tile_set_key(out_nanovdb_buf, root_tile, pnanovdb_coord_to_key(PNANOVDB_REF(ijk)));
                    pnanovdb_root_set_child(grid_type, out_nanovdb_buf, root, root_tile, upper);
                    pnanovdb_root_tile_set_state(out_nanovdb_buf, root_tile, 1u);
                }
            }
        }
    }
    pnanovdb_sync(sync_state, root_tile_workgroup_count, smem_buf, smem_sync_addr, sync_buf, sync_addr);

    // write grid, tree, root
    pnanovdb_sync_set_workgroup_count(sync_state, 1u);
    while (pnanovdb_sync_get_workgroup_idx(sync_state, smem_buf, smem_sync_addr, sync_buf, sync_addr))
    {
        pnanovdb_uint32_t vidx;
        for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
        {
            pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
            if (thread_idx == 0u)
            {
                pnanovdb_grid_set_grid_type(out_nanovdb_buf, grid, grid_type);

                pnanovdb_tree_set_first_root(out_nanovdb_buf, tree, root);
                pnanovdb_tree_set_first_upper(out_nanovdb_buf, tree, first_upper);
                pnanovdb_tree_set_first_lower(out_nanovdb_buf, tree, first_lower);
                pnanovdb_tree_set_first_leaf(out_nanovdb_buf, tree, first_leaf);

                pnanovdb_tree_set_node_count_upper(out_nanovdb_buf, tree, upper_count);
                pnanovdb_tree_set_tile_count_root(out_nanovdb_buf, tree, upper_count);
                pnanovdb_tree_set_node_count_lower(out_nanovdb_buf, tree, lower_count);
                pnanovdb_tree_set_tile_count_upper(out_nanovdb_buf, tree, lower_count);
                pnanovdb_tree_set_node_count_leaf(out_nanovdb_buf, tree, leaf_count);
                pnanovdb_tree_set_tile_count_lower(out_nanovdb_buf, tree, leaf_count);
                pnanovdb_tree_set_voxel_count(out_nanovdb_buf, tree, key_count);

                pnanovdb_root_set_tile_count(out_nanovdb_buf, root, root_tile_count);
            }
            if (thread_idx == 32)
            {
                // todo: factor in case with active root_tile but no upper
                pnanovdb_coord_t ijk_min = {PNANOVDB_INT_MAX, PNANOVDB_INT_MAX, PNANOVDB_INT_MAX};
                pnanovdb_coord_t ijk_max = {PNANOVDB_INT_MIN, PNANOVDB_INT_MIN, PNANOVDB_INT_MIN};
                for (pnanovdb_uint32_t upper_idx = 0u; upper_idx < upper_count; upper_idx++)
                {
                    pnanovdb_upper_handle_t upper = {
                        pnanovdb_address_offset_product(first_upper.address, upper_idx, PNANOVDB_GRID_TYPE_GET(grid_type, upper_size))
                    };
                    ijk_min = pnanovdb_coord_min(ijk_min, pnanovdb_upper_get_bbox_min(out_nanovdb_buf, upper));
                    ijk_max = pnanovdb_coord_max(ijk_max, pnanovdb_upper_get_bbox_max(out_nanovdb_buf, upper));
                }
                pnanovdb_root_set_bbox_min(out_nanovdb_buf, root, PNANOVDB_REF(ijk_min));
                pnanovdb_root_set_bbox_max(out_nanovdb_buf, root, PNANOVDB_REF(ijk_max));

                //pnanovdb_atlas_compute_bbox(grid_type, dst_buf, dst_root);

                // finalize grid header
                pnanovdb_uint32_t gridFlags = PNANOVDB_GRID_FLAGS_HAS_BBOX/* |
                    PNANOVDB_GRID_FLAGS_HAS_MIN_MAX |
                    PNANOVDB_GRID_FLAGS_HAS_AVERAGE |
                    PNANOVDB_GRID_FLAGS_HAS_STD_DEVIATION*/;

                pnanovdb_uint64_t grid_size = pnanovdb_int64_as_uint64(pnanovdb_address_diff(alloc_addr, pnanovdb_address_null()));

                pnanovdb_grid_set_magic(out_nanovdb_buf, grid, PNANOVDB_MAGIC_NUMBER);
                pnanovdb_grid_set_checksum(out_nanovdb_buf, grid, 0u);
                pnanovdb_grid_set_version(out_nanovdb_buf, grid, pnanovdb_make_version(PNANOVDB_MAJOR_VERSION_NUMBER, PNANOVDB_MINOR_VERSION_NUMBER, PNANOVDB_PATCH_VERSION_NUMBER));
                pnanovdb_grid_set_flags(out_nanovdb_buf, grid, gridFlags);
                pnanovdb_grid_set_grid_index(out_nanovdb_buf, grid, 0u);
                pnanovdb_grid_set_grid_count(out_nanovdb_buf, grid, 1u);
                pnanovdb_grid_set_grid_size(out_nanovdb_buf, grid, grid_size);

                // placeholder name
                pnanovdb_grid_set_grid_name(out_nanovdb_buf, grid, 0u, 0x616C7441);
                pnanovdb_grid_set_grid_name(out_nanovdb_buf, grid, 1u, 0x00000073);
            }
            if (thread_idx == 64)
            {
                pnanovdb_vec3_t voxel_size_inv = {
                    1.f / voxel_size.x,
                    1.f / voxel_size.y,
                    1.f / voxel_size.z
                };

                pnanovdb_map_handle_t map = pnanovdb_grid_get_map(out_nanovdb_buf, grid);
                pnanovdb_map_set_matf(out_nanovdb_buf, map, 0u, voxel_size.x);
                pnanovdb_map_set_matf(out_nanovdb_buf, map, 4u, voxel_size.y);
                pnanovdb_map_set_matf(out_nanovdb_buf, map, 8u, voxel_size.z);
                pnanovdb_map_set_invmatf(out_nanovdb_buf, map, 0u, voxel_size_inv.x);
                pnanovdb_map_set_invmatf(out_nanovdb_buf, map, 4u, voxel_size_inv.y);
                pnanovdb_map_set_invmatf(out_nanovdb_buf, map, 8u, voxel_size_inv.z);
                pnanovdb_map_set_matd(out_nanovdb_buf, map, 0u, voxel_size.x);
                pnanovdb_map_set_matd(out_nanovdb_buf, map, 4u, voxel_size.y);
                pnanovdb_map_set_matd(out_nanovdb_buf, map, 8u, voxel_size.z);
                pnanovdb_map_set_invmatd(out_nanovdb_buf, map, 0u, voxel_size_inv.x);
                pnanovdb_map_set_invmatd(out_nanovdb_buf, map, 4u, voxel_size_inv.y);
                pnanovdb_map_set_invmatd(out_nanovdb_buf, map, 8u, voxel_size_inv.z);
            }
            if (thread_idx == 96)
            {
                pnanovdb_map_handle_t map = pnanovdb_grid_get_map(out_nanovdb_buf, grid);
                pnanovdb_map_set_vecf(out_nanovdb_buf, map, 0u, 0.f);
                pnanovdb_map_set_vecf(out_nanovdb_buf, map, 1u, 0.f);
                pnanovdb_map_set_vecf(out_nanovdb_buf, map, 2u, 0.f);
                pnanovdb_map_set_vecd(out_nanovdb_buf, map, 0u, 0.0);
                pnanovdb_map_set_vecd(out_nanovdb_buf, map, 1u, 0.0);
                pnanovdb_map_set_vecd(out_nanovdb_buf, map, 2u, 0.0);

                pnanovdb_coord_t root_bbox_min = pnanovdb_root_get_bbox_min(out_nanovdb_buf, root);
                pnanovdb_coord_t root_bbox_max = pnanovdb_root_get_bbox_max(out_nanovdb_buf, root);

                pnanovdb_grid_set_world_bbox(out_nanovdb_buf, grid, 0u, root_bbox_min.x * voxel_size.x);
                pnanovdb_grid_set_world_bbox(out_nanovdb_buf, grid, 1u, root_bbox_min.y * voxel_size.y);
                pnanovdb_grid_set_world_bbox(out_nanovdb_buf, grid, 2u, root_bbox_min.z * voxel_size.z);
                pnanovdb_grid_set_world_bbox(out_nanovdb_buf, grid, 3u, root_bbox_max.x * voxel_size.x);
                pnanovdb_grid_set_world_bbox(out_nanovdb_buf, grid, 4u, root_bbox_max.y * voxel_size.y);
                pnanovdb_grid_set_world_bbox(out_nanovdb_buf, grid, 5u, root_bbox_max.z * voxel_size.z);
            }
            if (thread_idx == 128)
            {
                pnanovdb_grid_set_voxel_size(out_nanovdb_buf, grid, 0u, voxel_size.x);
                pnanovdb_grid_set_voxel_size(out_nanovdb_buf, grid, 1u, voxel_size.y);
                pnanovdb_grid_set_voxel_size(out_nanovdb_buf, grid, 2u, voxel_size.z);

                pnanovdb_uint32_t grid_class = PNANOVDB_GRID_CLASS_UNKNOWN;

                pnanovdb_grid_set_grid_class(out_nanovdb_buf, grid, grid_class);
                pnanovdb_grid_set_grid_type(out_nanovdb_buf, grid, grid_type);
                pnanovdb_grid_set_blind_metadata_offset(out_nanovdb_buf, grid, 0u);
                pnanovdb_grid_set_blind_metadata_count(out_nanovdb_buf, grid, 0u);
            }
        }
    }
    pnanovdb_sync(sync_state, 1u, smem_buf, smem_sync_addr, sync_buf, sync_addr);
}

#endif // end of NANOVDB_PUTILS_RASTER_H_HAS_BEEN_INCLUDED
