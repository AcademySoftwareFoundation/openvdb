// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>
#include <nanovdb/util/cuda/Util.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace nanovdb {

namespace cuda {

/// @brief Default stream-ordered device memory resource. Allocations are made
///        with cudaMallocAsync and freed with cudaFreeAsync via the
///        util::cuda wrappers.
/// @note Exposes both the legacy static methods (allocateAsync /
///       deallocateAsync, retained for source compatibility) and the
///       instance methods (allocate_async / deallocate_async) that model
///       the AsyncResource concept. The instance methods forward to the
///       static ones, so a default-constructed instance is stateless and
///       adds no overhead.
class DeviceResource
{
public:
    // cudaMalloc aligns memory to 256 bytes by default
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    static void* allocateAsync(size_t bytes, size_t, cudaStream_t stream) {
        void* p = nullptr;
        cudaCheck(util::cuda::mallocAsync(&p, bytes, stream));
        return p;
    }

    static void deallocateAsync(void *p, size_t, size_t, cudaStream_t stream) {
        cudaCheck(util::cuda::freeAsync(p, stream));
    }

    /// @brief Instance allocation entry point modelling the AsyncResource concept.
    /// @param bytes number of bytes to allocate
    /// @param alignment requested alignment (cudaMallocAsync always 256B-aligns)
    /// @param stream cuda stream the allocation is ordered on
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        return allocateAsync(bytes, alignment, stream);
    }

    /// @brief Instance deallocation entry point modelling the AsyncResource concept.
    /// @param p pointer previously returned by allocate_async
    /// @param bytes size passed to the matching allocate_async
    /// @param alignment alignment passed to the matching allocate_async
    /// @param stream cuda stream the deallocation is ordered on
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        deallocateAsync(p, bytes, alignment, stream);
    }
};

/// @brief Returns a program-lifetime, address-stable reference to a default
///        instance of resource @c R.
/// @details The instance is a function-local static, so it outlives every
///          caller and is safe to bind through a default function/constructor
///          argument. @c R must be default-constructible.
template <class R>
inline R& default_resource()
{
    static R sResource;
    return sResource;
}

/// @brief Detection trait: @c is_async_resource<R>::value is true iff @c R
///        models the stream-ordered AsyncResource concept, i.e. exposes
///        allocate_async(size_t, size_t, cudaStream_t) and
///        deallocate_async(void*, size_t, size_t, cudaStream_t).
/// @details Use it to dispatch between a stream-ordered resource and a
///          synchronous one (which exposes allocate/deallocate without a
///          stream argument):
/// @code
/// template<typename R>
/// void* allocate(R& resource, size_t bytes, size_t alignment, cudaStream_t stream)
/// {
///     if constexpr (nanovdb::cuda::is_async_resource<R>::value)
///         return resource.allocate_async(bytes, alignment, stream); // stream-ordered
///     else
///         return resource.allocate(bytes, alignment);               // synchronous
/// }
/// @endcode
template <class R, class = void>
struct is_async_resource : std::false_type {};

template <class R>
struct is_async_resource<R, std::void_t<
    decltype(std::declval<R&>().allocate_async(size_t{0}, size_t{0}, cudaStream_t{0})),
    decltype(std::declval<R&>().deallocate_async(std::declval<void*>(), size_t{0}, size_t{0}, cudaStream_t{0}))>>
    : std::true_type {};

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
