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
/// @note The default implementation is *synchronous*: cudaMallocHost /
///       cudaFreeHost ignore the supplied stream and synchronize the context.
///       The stream parameter is accepted purely so this type models the same
///       AsyncResource shape as cuda::DeviceResource; the reason the resource
///       is a seam at all is so callers can substitute a pooled / genuinely
///       stream-ordered pinned-host resource and avoid that synchronization.
class PinnedResource
{
public:
    // cudaMallocHost over-aligns (to at least the page size); 256 is the
    // nominal default this resource advertises through the AsyncResource API.
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    static void* allocateAsync(size_t bytes, size_t, cudaStream_t) {
        void* p = nullptr;
        cudaCheck(cudaMallocHost(&p, bytes));
        return p;
    }

    static void deallocateAsync(void* p, size_t, size_t, cudaStream_t) {
        cudaCheck(cudaFreeHost(p));
    }

    /// @brief Instance allocation entry point modelling the AsyncResource concept.
    /// @param bytes number of bytes to allocate
    /// @param alignment requested alignment (cudaMallocHost is page-aligned)
    /// @param stream cuda stream (ignored by the synchronous default impl)
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        return allocateAsync(bytes, alignment, stream);
    }

    /// @brief Instance deallocation entry point modelling the AsyncResource concept.
    /// @param p pointer previously returned by allocate_async
    /// @param bytes size passed to the matching allocate_async
    /// @param alignment alignment passed to the matching allocate_async
    /// @param stream cuda stream (ignored by the synchronous default impl)
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        deallocateAsync(p, bytes, alignment, stream);
    }
};

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_PINNEDRESOURCE_H_HAS_BEEN_INCLUDED
