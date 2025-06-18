
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/putil/Reduce.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
            of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PUTILS_REDUCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_REDUCE_H_HAS_BEEN_INCLUDED

#include "nanovdb/PNanoVDB.h"
#include "nanovdb/putil/Workgroup.h"

// ----------------------------------- uint32 add reduce ---------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_reduce_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_reduce(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT(pnanovdb_uint32_t) total_count
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);

    pnanovdb_uint32_t localVal[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        localVal[vidx] = val[vidx].x +
            val[vidx].y +
            val[vidx].z +
            val[vidx].w;
        pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, localVal[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 64u)
        {
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 64u);
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 128u);
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 192u);
            pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 16u)
        {
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx + 16u);
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx + 32u);
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx + 48u);
            pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 4u)
        {
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 4u);
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 8u);
            localVal[vidx] += pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 12u);
            pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        // compute totalCount
        PNANOVDB_DEREF(total_count) = pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 0) +
            pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 1) +
            pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 2) +
            pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 3);
    }
}

// ----------------------------------- float add reduce ---------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_reduce_float_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_reduce_float(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_vec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT(float) total_count
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);

    float localVal[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        localVal[vidx] = val[vidx].x +
            val[vidx].y +
            val[vidx].z +
            val[vidx].w;
        pnanovdb_write_float_index(smem_buf, sdata0_addr, thread_idx, localVal[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 64u)
        {
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata0_addr, thread_idx + 64u);
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata0_addr, thread_idx + 128u);
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata0_addr, thread_idx + 192u);
            pnanovdb_write_float_index(smem_buf, sdata1_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 16u)
        {
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata1_addr, thread_idx + 16u);
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata1_addr, thread_idx + 32u);
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata1_addr, thread_idx + 48u);
            pnanovdb_write_float_index(smem_buf, sdata0_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 4u)
        {
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata0_addr, thread_idx + 4u);
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata0_addr, thread_idx + 8u);
            localVal[vidx] += pnanovdb_read_float_index(smem_buf, sdata0_addr, thread_idx + 12u);
            pnanovdb_write_float_index(smem_buf, sdata1_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        // compute totalCount
        PNANOVDB_DEREF(total_count) = pnanovdb_read_float_index(smem_buf, sdata1_addr, 0) +
            pnanovdb_read_float_index(smem_buf, sdata1_addr, 1) +
            pnanovdb_read_float_index(smem_buf, sdata1_addr, 2) +
            pnanovdb_read_float_index(smem_buf, sdata1_addr, 3);
    }
}

// ----------------------------------- uint32 max reduce ---------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_reduce_max_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_reduce_max(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uvec4_t val[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    PNANOVDB_INOUT(pnanovdb_uint32_t) total_max
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);

    pnanovdb_uint32_t localVal[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        localVal[vidx] = pnanovdb_uint32_max(
            pnanovdb_uint32_max(val[vidx].x, val[vidx].y),
            pnanovdb_uint32_max(val[vidx].z, val[vidx].w));
        pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, localVal[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 64u)
        {
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 64u));
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 128u));
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 192u));
            pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 16u)
        {
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx + 16u));
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx + 32u));
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata1_addr, thread_idx + 48u));
            pnanovdb_write_uint32_index(smem_buf, sdata0_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 4u)
        {
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 4u));
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 8u));
            localVal[vidx] = pnanovdb_uint32_max(localVal[vidx], pnanovdb_read_uint32_index(smem_buf, sdata0_addr, thread_idx + 12u));
            pnanovdb_write_uint32_index(smem_buf, sdata1_addr, thread_idx, localVal[vidx]);
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        // compute totalCount
        PNANOVDB_DEREF(total_max) = pnanovdb_uint32_max(
            pnanovdb_uint32_max(
                pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 0),
                pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 1)),
            pnanovdb_uint32_max(
                pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 2),
                pnanovdb_read_uint32_index(smem_buf, sdata1_addr, 3))
            );
    }
}

// ----------------------------------- coord max reduce ---------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_reduce_max_coord_smem_size_in_words = 3u * (256u + 256u);

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_reduce_max_coord(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_coord_t val,
    PNANOVDB_INOUT(pnanovdb_coord_t) total_max
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 3u * 256u * 4u);

    pnanovdb_coord_t localVal[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        localVal[vidx] = val;
        pnanovdb_write_coord_index(smem_buf, sdata0_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 64u)
        {
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 64u));
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 128u));
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 192u));
            pnanovdb_write_coord_index(smem_buf, sdata1_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 16u)
        {
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata1_addr, thread_idx + 16u));
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata1_addr, thread_idx + 32u));
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata1_addr, thread_idx + 48u));
            pnanovdb_write_coord_index(smem_buf, sdata0_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 4u)
        {
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 4u));
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 8u));
            localVal[vidx] = pnanovdb_coord_max(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 12u));
            pnanovdb_write_coord_index(smem_buf, sdata1_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        // compute totalCount
        PNANOVDB_DEREF(total_max) = pnanovdb_coord_max(
            pnanovdb_coord_max(
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 0),
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 1)),
            pnanovdb_coord_max(
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 2),
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 3))
            );
    }
}

// ----------------------------------- coord min reduce ---------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_reduce_min_coord_smem_size_in_words = 3u * (256u + 256u);

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_reduce_min_coord(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_coord_t val,
    PNANOVDB_INOUT(pnanovdb_coord_t) total_min
)
{
    pnanovdb_address_t sdata0_addr = smem_addr;
    pnanovdb_address_t sdata1_addr = pnanovdb_address_offset(smem_addr, 3u * 256u * 4u);

    pnanovdb_coord_t localVal[PNANOVDB_COMPUTE_VECTOR_WIDTH];

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        localVal[vidx] = val;
        pnanovdb_write_coord_index(smem_buf, sdata0_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 64u)
        {
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 64u));
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 128u));
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 192u));
            pnanovdb_write_coord_index(smem_buf, sdata1_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 16u)
        {
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata1_addr, thread_idx + 16u));
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata1_addr, thread_idx + 32u));
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata1_addr, thread_idx + 48u));
            pnanovdb_write_coord_index(smem_buf, sdata0_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        if (thread_idx < 4u)
        {
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 4u));
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 8u));
            localVal[vidx] = pnanovdb_coord_min(localVal[vidx], pnanovdb_read_coord_index(smem_buf, sdata0_addr, thread_idx + 12u));
            pnanovdb_write_coord_index(smem_buf, sdata1_addr, thread_idx, PNANOVDB_REF(localVal[vidx]));
        }
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;

        // compute totalCount
        PNANOVDB_DEREF(total_min) = pnanovdb_coord_min(
            pnanovdb_coord_min(
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 0),
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 1)),
            pnanovdb_coord_min(
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 2),
                pnanovdb_read_coord_index(smem_buf, sdata1_addr, 3))
            );
    }
}

// ----------------------------------- uint32 mask reduce and write ---------------------

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_workgroup_reduce_and_write_mask_max_smem_size_in_words = 256u + 256u;

PNANOVDB_FORCE_INLINE void pnanovdb_workgroup_reduce_and_write_mask(
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_uint32_t word_idx[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    pnanovdb_uint32_t word_mask[PNANOVDB_COMPUTE_VECTOR_WIDTH],
    pnanovdb_buf_t dst_buf, pnanovdb_address_t dst_addr
)
{
    pnanovdb_address_t s_word_idx_addr = smem_addr;
    pnanovdb_address_t s_word_mask_addr = pnanovdb_address_offset(smem_addr, 256u * 4u);

    pnanovdb_uint32_t vidx;
    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
        pnanovdb_write_uint32_index(smem_buf, s_word_idx_addr, thread_idx, word_idx[vidx]);
        pnanovdb_write_uint32_index(smem_buf, s_word_mask_addr, thread_idx, word_mask[vidx]);
    }

    pnanovdb_workgroup_sync();

    for (vidx = 0u; vidx < PNANOVDB_COMPUTE_VECTOR_WIDTH; vidx++)
    {
        pnanovdb_uint32_t thread_idx = vidx + vidx_offset;
        pnanovdb_uint32_t word_idx = pnanovdb_read_uint32_index(smem_buf, s_word_idx_addr, thread_idx);
        pnanovdb_bool_t is_unique = word_idx != ~0u;
        if (thread_idx > 0u)
        {
            pnanovdb_uint32_t cmp_word_idx = pnanovdb_read_uint32_index(smem_buf, s_word_idx_addr, thread_idx - 1u);
            is_unique = word_idx != cmp_word_idx;
        }
        if (is_unique)
        {
            pnanovdb_uint32_t mask_accum = pnanovdb_read_uint32_index(smem_buf, s_word_mask_addr, thread_idx);
            for (pnanovdb_uint32_t bit_idx = 1u; bit_idx < 32u && (thread_idx + bit_idx) < 256u; bit_idx++)
            {
                pnanovdb_uint32_t cmp_word_idx = pnanovdb_read_uint32_index(smem_buf, s_word_idx_addr, thread_idx + bit_idx);
                if (word_idx != cmp_word_idx)
                {
                    break;
                }
                mask_accum = mask_accum | pnanovdb_read_uint32_index(smem_buf, s_word_mask_addr, thread_idx + bit_idx);
            }
            // write out reduced mask
            pnanovdb_write_uint32_index(dst_buf, dst_addr, word_idx, mask_accum);
        }
    }

    pnanovdb_workgroup_sync();
}

#endif // end of NANOVDB_PUTILS_REDUCE_H_HAS_BEEN_INCLUDED
