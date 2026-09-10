// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NANOVDB_CUDA_MANAGEDRESOURCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_MANAGEDRESOURCE_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>
#include <nanovdb/util/cuda/Util.h>

#include <cstddef>

namespace nanovdb {

namespace cuda {

/// @brief Managed (unified) memory resource. Allocations
///        (cudaMallocManaged) are accessible from both the host and the
///        device, with the driver migrating pages on demand, so a container
///        over this resource serves grids that are read on both sides --
///        the replacement for the legacy UnifiedBuffer. A GridHandle over
///        this resource parses (and validates) its metadata through the
///        device like any device buffer -- a host-side parse could race
///        still-running producer kernels -- while the host accessors remain
///        available afterwards; ordering host reads after device writes is
///        the caller's responsibility, exactly as with UnifiedBuffer.
/// @note This resource is *synchronous*: cudaMallocManaged / cudaFree have
///       no stream-ordered form, so it models the synchronous Resource
///       concept (allocate / deallocate, no stream). The usual contract
///       applies: the caller ensures device work touching an allocation has
///       completed before it is freed.
class ManagedResource
{
public:
    // cudaMallocManaged aligns to at least 256 bytes.
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    /// @brief Managed allocations are mapped into the host address space
    ///        (detected by nanovdb::cuda::is_host_accessible_resource).
    static constexpr bool HOST_ACCESSIBLE = true;

    /// @brief Managed allocations are also valid device addresses (detected
    ///        by nanovdb::cuda::is_device_accessible_resource), so a handle
    ///        over this resource exposes the device accessors as well as the
    ///        host ones.
    static constexpr bool DEVICE_ACCESSIBLE = true;

    /// @brief Synchronous allocation of managed memory.
    /// @param bytes number of bytes to allocate
    /// @param alignment requested alignment (ignored; cudaMallocManaged
    ///        satisfies at least DEFAULT_ALIGNMENT)
    void* allocate(size_t bytes, size_t alignment) {
        (void)alignment;
        void* p = nullptr;
        cudaCheck(cudaMallocManaged(&p, bytes));
        return p;
    }

    /// @brief Synchronous deallocation.
    /// @param p pointer previously returned by allocate
    /// @param bytes size passed to the matching allocate (unused)
    /// @param alignment alignment passed to the matching allocate (unused)
    void deallocate(void* p, size_t bytes, size_t alignment) {
        (void)bytes;
        (void)alignment;
        cudaCheck(cudaFree(p));
    }
};

}// namespace cuda

}// namespace nanovdb

#endif // NANOVDB_CUDA_MANAGEDRESOURCE_H_HAS_BEEN_INCLUDED
