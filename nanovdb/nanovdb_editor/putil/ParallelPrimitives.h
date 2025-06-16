// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/putil/ParallelPrimitives.h

    \author Andrew Reidmeyer

    \brief
*/

#ifndef NANOVDB_PUTILS_PARALLEL_PRIMITIVES_H_HAS_BEEN_INCLUDED
#define NANOVDB_PUTILS_PARALLEL_PRIMITIVES_H_HAS_BEEN_INCLUDED

#include "nanovdb_editor/putil/Compute.h"

/// ********************************* ParallelPrimitives ***************************************

struct pnanovdb_parallel_primitives_context_t;
typedef struct pnanovdb_parallel_primitives_context_t pnanovdb_parallel_primitives_context_t;

typedef struct pnanovdb_parallel_primitives_t
{
    PNANOVDB_REFLECT_INTERFACE();

    pnanovdb_parallel_primitives_context_t* (PNANOVDB_ABI* create_context)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue);

    void (PNANOVDB_ABI* destroy_context)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_parallel_primitives_context_t* context);

    void (PNANOVDB_ABI* global_scan)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_parallel_primitives_context_t* context,
        pnanovdb_compute_buffer_t* val_in,
        pnanovdb_compute_buffer_t* val_out,
        pnanovdb_uint64_t val_count,
        pnanovdb_uint32_t dispatch_count);

    void (PNANOVDB_ABI* global_scan_uint64)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_parallel_primitives_context_t* context,
        pnanovdb_compute_buffer_t* val_in,
        pnanovdb_compute_buffer_t* val_out,
        pnanovdb_uint64_t val_count,
        pnanovdb_uint32_t dispatch_count);

    void (PNANOVDB_ABI* global_scan_max)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_parallel_primitives_context_t* context,
        pnanovdb_compute_buffer_t* val_in,
        pnanovdb_compute_buffer_t* val_out,
        pnanovdb_uint64_t val_count,
        pnanovdb_uint32_t dispatch_count);

    void (PNANOVDB_ABI* radix_sort)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_parallel_primitives_context_t* context_in,
        pnanovdb_compute_buffer_t* key_inout,
        pnanovdb_compute_buffer_t* val_inout,
        pnanovdb_uint64_t key_count,
        pnanovdb_uint32_t key_bit_count);

    void (PNANOVDB_ABI* radix_sort_dual_key)(
        const pnanovdb_compute_t* compute,
        pnanovdb_compute_queue_t* queue,
        pnanovdb_parallel_primitives_context_t* context_in,
        pnanovdb_compute_buffer_t* key_low_inout,
        pnanovdb_compute_buffer_t* key_high_inout,
        pnanovdb_compute_buffer_t* val_inout,
        pnanovdb_uint64_t key_count,
        pnanovdb_uint32_t key_low_bit_count,
        pnanovdb_uint32_t key_high_bit_count);

    const pnanovdb_compute_t* compute;

}pnanovdb_parallel_primitives_t;

#define PNANOVDB_REFLECT_TYPE pnanovdb_parallel_primitives_t
PNANOVDB_REFLECT_BEGIN()
PNANOVDB_REFLECT_FUNCTION_POINTER(create_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(destroy_context, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(global_scan, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(global_scan_uint64, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(global_scan_max, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(radix_sort, 0, 0)
PNANOVDB_REFLECT_FUNCTION_POINTER(radix_sort_dual_key, 0, 0)
PNANOVDB_REFLECT_POINTER(pnanovdb_compute_t, compute, 0, 0)
PNANOVDB_REFLECT_END(0)
PNANOVDB_REFLECT_INTERFACE_IMPL()
#undef PNANOVDB_REFLECT_TYPE

typedef pnanovdb_parallel_primitives_t* (PNANOVDB_ABI* PFN_pnanovdb_get_parallel_primitives)();

PNANOVDB_API pnanovdb_parallel_primitives_t* pnanovdb_get_parallel_primitives();

static void pnanovdb_parallel_primitives_load(
    pnanovdb_parallel_primitives_t* parallel_primitives, const pnanovdb_compute_t* compute)
{
    auto get_parallel_primitives = (PFN_pnanovdb_get_parallel_primitives)
        pnanovdb_get_proc_address(compute->module, "pnanovdb_get_parallel_primitives");
    if (!get_parallel_primitives)
    {
        printf("Error: Failed to acquire parallel primitives\n");
        return;
    }
    *parallel_primitives = *get_parallel_primitives();

    parallel_primitives->compute = compute;
}

static void pnanovdb_parallel_primitives_free(pnanovdb_parallel_primitives_t* parallel_primitives)
{
    // NOP for now
}

#endif
