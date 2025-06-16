
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/putil/Workgroup.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
            of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PUTILS_WORKGROUP_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_WORKGROUP_H_HAS_BEEN_INCLUDED

#include "nanovdb/PNanoVDB.h"

#if defined(PNANOVDB_BUF_C)

#if defined(__CUDACC__)
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_atomic_add_uint32_ptr(uint32_t* addr, uint32_t val)
{
    return atomicAdd(addr, val);
}
#elif defined(_WIN32)
#include <Windows.h>
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_atomic_add_uint32_ptr(uint32_t* addr, uint32_t val)
{
    return (uint32_t)InterlockedAdd((volatile LONG*)addr, val) - val;
}
#else
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_atomic_add_uint32_ptr(uint32_t* addr, uint32_t val)
{
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}
#endif

#if defined(PNANOVDB_ADDRESS_32)
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_atomic_add_uint32(pnanovdb_buf_t buf, uint32_t byte_offset, uint32_t value)
{
    uint32_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    if (wordaddress < buf.size_in_words)
    {
        return pnanovdb_buf_atomic_add_uint32_ptr(&buf.data[wordaddress], value);
    }
    return 0u;
#else
    return pnanovdb_buf_atomic_add_uint32_ptr(&buf.data[wordaddress], value);
#endif
}
#elif defined(PNANOVDB_ADDRESS_64)
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_atomic_add_uint32(pnanovdb_buf_t buf, uint64_t byte_offset, uint32_t value)
{
    uint64_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    if (wordaddress < buf.size_in_words)
    {
        return pnanovdb_buf_atomic_add_uint32_ptr(&buf.data[wordaddress], value);
    }
    return 0u;
#else
    return pnanovdb_buf_atomic_add_uint32_ptr(&buf.data[wordaddress], value);
#endif
}
#endif
#elif defined(PNANOVDB_BUF_HLSL)
#if defined(PNANOVDB_ADDRESS_32)
uint pnanovdb_buf_atomic_add_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value)
{
    uint old_value;
    InterlockedAdd(buf[(byte_offset >> 2u)], value, old_value);
    return old_value;
}
#elif defined(PNANOVDB_ADDRESS_64)
uint pnanovdb_buf_atomic_add_uint32(pnanovdb_buf_t buf, uint64_t byte_offset, uint value)
{
    uint old_value;
    InterlockedAdd(buf[uint(byte_offset >> 2u)], value, old_value);
    return old_value;
}
#endif
#elif defined(PNANOVDB_BUF_GLSL)
uint pnanovdb_buf_atomic_add_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value)
{
    return atomicAdd(pnanovdb_buf_data[(byte_offset >> 2u)], value);
}
#endif

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_atomic_add_uint32(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_uint32_t value)
{
    return pnanovdb_buf_atomic_add_uint32(buf, address.byte_offset, value);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint32_max(pnanovdb_uint32_t a, pnanovdb_uint32_t b)
{
    return a > b ? a : b;
}
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_int32_max(pnanovdb_int32_t a, pnanovdb_int32_t b)
{
    return a > b ? a : b;
}
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_int32_min(pnanovdb_int32_t a, pnanovdb_int32_t b)
{
    return a < b ? a : b;
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_coord_max(pnanovdb_coord_t a, pnanovdb_coord_t b)
{
    pnanovdb_coord_t ret = {
        pnanovdb_int32_max(a.x, b.x), pnanovdb_int32_max(a.y, b.y), pnanovdb_int32_max(a.z, b.z)
    };
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_coord_min(pnanovdb_coord_t a, pnanovdb_coord_t b)
{
    pnanovdb_coord_t ret = {
        pnanovdb_int32_min(a.x, b.x), pnanovdb_int32_min(a.y, b.y), pnanovdb_int32_min(a.z, b.z)
    };
    return ret;
}

#if defined(PNANOVDB_C)

#define PNANOVDB_INOUT_ARRAY(X) X

#if defined(__CUDA_ARCH__)
#define pnanovdb_workgroup_sync __syncthreads
#else
#define pnanovdb_workgroup_sync(...)
#endif

#elif defined(PNANOVDB_HLSL)

#define PNANOVDB_INOUT_ARRAY(X) inout X

#define pnanovdb_workgroup_sync DeviceMemoryBarrierWithGroupSync

#elif defined(PNANOVDB_GLSL)

#define PNANOVDB_INOUT_ARRAY(X) inout X

#define pnanovdb_workgroup_sync memoryBarrier() barrier

#endif

#define PNANOVDB_INT_MIN (-2147483647 - 1)
#define PNANOVDB_INT_MAX 2147483647

#if defined(PNANOVDB_GPU_THREADS)
#define PNANOVDB_COMPUTE_VECTOR_WIDTH 1u
#else
#define PNANOVDB_COMPUTE_VECTOR_WIDTH 256u
#endif

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_read_uint32_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index)
{
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset_product(addr, index, 4u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_uint32_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index, pnanovdb_uint32_t val)
{
    pnanovdb_write_uint32(buf, pnanovdb_address_offset_product(addr, index, 4u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_read_int32_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index)
{
    return pnanovdb_read_int32(buf, pnanovdb_address_offset_product(addr, index, 4u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_int32_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index, pnanovdb_int32_t val)
{
    pnanovdb_write_int32(buf, pnanovdb_address_offset_product(addr, index, 4u), val);
}

PNANOVDB_FORCE_INLINE float pnanovdb_read_float_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index)
{
    return pnanovdb_read_float(buf, pnanovdb_address_offset_product(addr, index, 4u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_float_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index, float val)
{
    pnanovdb_write_float(buf, pnanovdb_address_offset_product(addr, index, 4u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_uvec4_t pnanovdb_read_uvec4(pnanovdb_buf_t buf, pnanovdb_address_t addr)
{
    pnanovdb_uvec4_t ret;
    ret.x = pnanovdb_read_uint32(buf, pnanovdb_address_offset(addr, 0u));
    ret.y = pnanovdb_read_uint32(buf, pnanovdb_address_offset(addr, 4u));
    ret.z = pnanovdb_read_uint32(buf, pnanovdb_address_offset(addr, 8u));
    ret.w = pnanovdb_read_uint32(buf, pnanovdb_address_offset(addr, 12u));
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_uvec4_t pnanovdb_read_uvec4_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index)
{
    return pnanovdb_read_uvec4(buf, pnanovdb_address_offset_product(addr, index, 16u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_uvec4(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uvec4_t val)
{
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(addr, 0u), val.x);
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(addr, 4u), val.y);
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(addr, 8u), val.z);
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(addr, 12u), val.w);
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_uvec4_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index, pnanovdb_uvec4_t val)
{
    pnanovdb_write_uvec4(buf, pnanovdb_address_offset_product(addr, index, 16u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec4_t pnanovdb_read_vec4(pnanovdb_buf_t buf, pnanovdb_address_t addr)
{
    pnanovdb_vec4_t ret;
    ret.x = pnanovdb_read_float(buf, pnanovdb_address_offset(addr, 0u));
    ret.y = pnanovdb_read_float(buf, pnanovdb_address_offset(addr, 4u));
    ret.z = pnanovdb_read_float(buf, pnanovdb_address_offset(addr, 8u));
    ret.w = pnanovdb_read_float(buf, pnanovdb_address_offset(addr, 12u));
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec4_t pnanovdb_read_vec4_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index)
{
    return pnanovdb_read_vec4(buf, pnanovdb_address_offset_product(addr, index, 16u));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_vec4(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_vec4_t val)
{
    pnanovdb_write_float(buf, pnanovdb_address_offset(addr, 0u), val.x);
    pnanovdb_write_float(buf, pnanovdb_address_offset(addr, 4u), val.y);
    pnanovdb_write_float(buf, pnanovdb_address_offset(addr, 8u), val.z);
    pnanovdb_write_float(buf, pnanovdb_address_offset(addr, 12u), val.w);
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_vec4_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index, pnanovdb_vec4_t val)
{
    pnanovdb_write_vec4(buf, pnanovdb_address_offset_product(addr, index, 16u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_read_coord_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index)
{
    return pnanovdb_read_coord(buf, pnanovdb_address_offset_product(addr, index, 12u));
}

PNANOVDB_FORCE_INLINE void pnanovdb_write_coord_index(pnanovdb_buf_t buf, pnanovdb_address_t addr, pnanovdb_uint32_t index, PNANOVDB_INOUT(pnanovdb_coord_t) val)
{
    pnanovdb_write_coord(buf, pnanovdb_address_offset_product(addr, index, 12u), val);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_alloc_address_aligned(PNANOVDB_INOUT(pnanovdb_address_t) alloc_addr, pnanovdb_address_t alloc_size)
{
    pnanovdb_address_t ret;
    ret.byte_offset = PNANOVDB_DEREF(alloc_addr).byte_offset;
    PNANOVDB_DEREF(alloc_addr).byte_offset += 32u * ((alloc_size.byte_offset + 31u) / 32u);
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_alloc_address_aligned32(PNANOVDB_INOUT(pnanovdb_address_t) alloc_addr, pnanovdb_uint32_t alloc_size)
{
    pnanovdb_address_t ret;
    ret.byte_offset = PNANOVDB_DEREF(alloc_addr).byte_offset;
    PNANOVDB_DEREF(alloc_addr).byte_offset += 32u * ((alloc_size + 31u) / 32u);
    return ret;
}

struct pnanovdb_sync_state_t
{
    pnanovdb_uint32_t attempt_idx;
    pnanovdb_uint32_t dispatch_idx;
    pnanovdb_uint32_t workgroup_idx;
    pnanovdb_uint32_t vidx_offset;
    pnanovdb_uint32_t workgroup_count;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_sync_state_t)

PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_dispatch_sync_smem_size_in_words = 2u;

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_sync_init(
    PNANOVDB_INOUT(pnanovdb_sync_state_t) state,
    pnanovdb_uint32_t vidx_offset,
    pnanovdb_uint32_t workgroup_count,
    pnanovdb_buf_t workgroup_status_buf, pnanovdb_address_t workgroup_status_addr)
{
    PNANOVDB_DEREF(state).attempt_idx = 0u;
    PNANOVDB_DEREF(state).dispatch_idx = 1u;
    PNANOVDB_DEREF(state).workgroup_idx = ~0u;
    PNANOVDB_DEREF(state).vidx_offset = vidx_offset;
    PNANOVDB_DEREF(state).workgroup_count = workgroup_count;

    return 0u == pnanovdb_read_uint32(workgroup_status_buf, workgroup_status_addr);
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_sync_get_workgroup_idx(
    PNANOVDB_INOUT(pnanovdb_sync_state_t) state,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr)
{
    pnanovdb_address_t s_block_idx_addr = smem_addr;

    PNANOVDB_DEREF(state).attempt_idx++;
    if (PNANOVDB_DEREF(state).attempt_idx > PNANOVDB_DEREF(state).workgroup_count + 1) // + 1 to make sure last task gets reported complete
    {
        return false;
    }

    pnanovdb_workgroup_sync();
    if (PNANOVDB_DEREF(state).vidx_offset == 0u)
    {
        // report completed
        if (PNANOVDB_DEREF(state).workgroup_idx < PNANOVDB_DEREF(state).workgroup_count)
        {
            pnanovdb_atomic_add_uint32(sync_buf, pnanovdb_address_offset(sync_addr, (2u * PNANOVDB_DEREF(state).dispatch_idx + 1u) * 4u), 1u);
        }

        // get new work
        PNANOVDB_DEREF(state).workgroup_idx = pnanovdb_atomic_add_uint32(sync_buf,
            pnanovdb_address_offset(sync_addr, (2u * PNANOVDB_DEREF(state).dispatch_idx + 0u) * 4u), 1u);
        pnanovdb_write_uint32(smem_buf, s_block_idx_addr, PNANOVDB_DEREF(state).workgroup_idx);
    }
    pnanovdb_workgroup_sync();
    PNANOVDB_DEREF(state).workgroup_idx = pnanovdb_read_uint32(smem_buf, s_block_idx_addr);
    return PNANOVDB_DEREF(state).workgroup_idx < PNANOVDB_DEREF(state).workgroup_count;
}

PNANOVDB_FORCE_INLINE void pnanovdb_sync(PNANOVDB_INOUT(
    pnanovdb_sync_state_t) state,
    pnanovdb_uint32_t new_workgroup_count,
    pnanovdb_buf_t smem_buf, pnanovdb_address_t smem_addr,
    pnanovdb_buf_t sync_buf, pnanovdb_address_t sync_addr)
{
    pnanovdb_address_t s_is_finished_addr = pnanovdb_address_offset(smem_addr, 4u);

    if (PNANOVDB_DEREF(state).vidx_offset == 0u)
    {
        pnanovdb_write_uint32(smem_buf, s_is_finished_addr, 0u);
    }

    pnanovdb_workgroup_sync();

    while (0u == pnanovdb_read_uint32(smem_buf, s_is_finished_addr))
    {
        if (PNANOVDB_DEREF(state).vidx_offset == 0u)
        {
            pnanovdb_uint32_t oldValue;
            oldValue = pnanovdb_atomic_add_uint32(sync_buf,
                pnanovdb_address_offset(sync_addr, (2u * PNANOVDB_DEREF(state).dispatch_idx + 1u) * 4u), 0u);
            bool isFinished = oldValue >= PNANOVDB_DEREF(state).workgroup_count;
            if (isFinished)
            {
                pnanovdb_write_uint32(smem_buf, s_is_finished_addr, 1u);
            }
        }

        pnanovdb_workgroup_sync();
    }

    PNANOVDB_DEREF(state).attempt_idx = 0u;
    PNANOVDB_DEREF(state).dispatch_idx++;
    PNANOVDB_DEREF(state).workgroup_idx = ~0u;
    PNANOVDB_DEREF(state).workgroup_count = new_workgroup_count;
}

PNANOVDB_FORCE_INLINE void pnanovdb_sync_set_workgroup_count(PNANOVDB_INOUT(
    pnanovdb_sync_state_t) state,
    pnanovdb_uint32_t new_workgroup_count)
{
    PNANOVDB_DEREF(state).workgroup_count = new_workgroup_count;
}

PNANOVDB_FORCE_INLINE void pnanovdb_sync_final(
    PNANOVDB_INOUT(pnanovdb_sync_state_t) state,
    pnanovdb_buf_t workgroup_status_buf, pnanovdb_address_t workgroup_status_addr)
{
    pnanovdb_workgroup_sync();

    if (PNANOVDB_DEREF(state).vidx_offset == 0u)
    {
        pnanovdb_write_uint32(workgroup_status_buf, workgroup_status_addr, 1u);
    }
}

#endif // end of NANOVDB_PUTILS_WORKGROUP_H_HAS_BEEN_INCLUDED
