// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NANOVDB_CUDA_PINNEDRESOURCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_PINNEDRESOURCE_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>
#include <nanovdb/util/cuda/Util.h>

#include <cstddef>

namespace nanovdb {

namespace cuda {

/// @brief Default host-pinned memory resource. Allocations are page-locked
///        host memory (cudaMallocHost) that is both host-accessible and
///        device-accessible, so it can serve as the host side of an
///        asynchronous host<->device copy.
/// @note This resource is *synchronous*: cudaMallocHost / cudaFreeHost
///       synchronize the context, so it models the synchronous Resource
///       concept (allocate / deallocate, no stream) rather than the
///       stream-ordered AsyncResource concept. It is a seam so callers can
///       substitute a pooled / genuinely stream-ordered pinned-host resource
///       and avoid that synchronization.
class PinnedResource
{
public:
    // cudaMallocHost over-aligns (to at least the page size), so the requested
    // alignment is always satisfied; 256 is the nominal default advertised here.
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    /// @brief Synchronous allocation of page-locked host memory.
    /// @param bytes number of bytes to allocate
    /// @param alignment requested alignment (ignored; cudaMallocHost is page-aligned)
    void* allocate(size_t bytes, size_t alignment) {
        (void)alignment;
        void* p = nullptr;
        cudaCheck(cudaMallocHost(&p, bytes));
        return p;
    }

    /// @brief Synchronous deallocation.
    /// @param p pointer previously returned by allocate
    /// @param bytes size passed to the matching allocate (unused)
    /// @param alignment alignment passed to the matching allocate (unused)
    void deallocate(void* p, size_t bytes, size_t alignment) {
        (void)bytes;
        (void)alignment;
        cudaCheck(cudaFreeHost(p));
    }
};

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_PINNEDRESOURCE_H_HAS_BEEN_INCLUDED
