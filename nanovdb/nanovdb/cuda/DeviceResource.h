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
/// @note Models the AsyncResource concept, which refines the synchronous
///       Resource concept (as in CCCL's cuda::mr): the instance methods
///       provide both the stream-ordered allocate_async / deallocate_async
///       and the synchronous allocate / deallocate. The type is stateless,
///       so a default-constructed instance adds no overhead. The static
///       allocateAsync / deallocateAsync methods are deprecated.
class DeviceResource
{
public:
    // cudaMalloc aligns memory to 256 bytes by default
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    /// @brief Stream-ordered allocation.
    /// @param bytes number of bytes to allocate
    /// @param stream cuda stream the allocation is ordered on
    /// @note the alignment parameter is unnamed: cudaMallocAsync always
    ///       256B-aligns
    void* allocate_async(size_t bytes, size_t, cudaStream_t stream) {
        void* p = nullptr;
        cudaCheck(util::cuda::mallocAsync(&p, bytes, stream));
        return p;
    }

    /// @brief Stream-ordered deallocation.
    /// @param p pointer previously returned by allocate_async
    /// @param stream cuda stream the deallocation is ordered on
    void deallocate_async(void* p, size_t, size_t, cudaStream_t stream) {
        cudaCheck(util::cuda::freeAsync(p, stream));
    }

    /// @brief Synchronous allocation; the returned memory is immediately
    ///        valid on every stream.
    /// @param bytes number of bytes to allocate
    /// @param alignment requested alignment
    void* allocate(size_t bytes, size_t alignment) {
        void* p = this->allocate_async(bytes, alignment, cudaStream_t(0));
        cudaCheck(cudaStreamSynchronize(cudaStream_t(0)));
        return p;
    }

    /// @brief Synchronous deallocation; the caller guarantees that device
    ///        work touching the memory has completed.
    /// @param p pointer previously returned by allocate or allocate_async
    void deallocate(void* p, size_t bytes, size_t alignment) {
        this->deallocate_async(p, bytes, alignment, cudaStream_t(0));
    }

    [[deprecated("use the instance method allocate_async")]]
    static void* allocateAsync(size_t bytes, size_t alignment, cudaStream_t stream) {
        return DeviceResource().allocate_async(bytes, alignment, stream);
    }

    [[deprecated("use the instance method deallocate_async")]]
    static void deallocateAsync(void *p, size_t bytes, size_t alignment, cudaStream_t stream) {
        DeviceResource().deallocate_async(p, bytes, alignment, stream);
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
/// @note AsyncResource refines the synchronous Resource concept, matching
///       CCCL's cuda::mr: an async resource must also provide the
///       synchronous allocate / deallocate, so is_async_resource<R> implies
///       is_resource<R>. A synchronous-only resource (e.g. PinnedResource)
///       models just is_resource. The synchronous methods of a stream-ordered
///       resource are typically thin delegates (allocate_async on the null
///       stream followed by a stream synchronize).
template <class R, class = void>
struct is_async_resource : std::false_type {};

template <class R>
struct is_async_resource<R, std::void_t<
    decltype(std::declval<R&>().allocate_async(size_t{0}, size_t{0}, cudaStream_t{0})),
    decltype(std::declval<R&>().deallocate_async(std::declval<void*>(), size_t{0}, size_t{0}, cudaStream_t{0})),
    decltype(std::declval<R&>().allocate(size_t{0}, size_t{0})),
    decltype(std::declval<R&>().deallocate(std::declval<void*>(), size_t{0}, size_t{0}))>>
    : std::true_type {};

/// @brief Detection trait: @c is_resource<R>::value is true iff @c R models
///        the synchronous Resource concept, i.e. exposes
///        allocate(size_t, size_t) and deallocate(void*, size_t, size_t).
/// @details Use it to constrain code paths that need a resource without a
///          stream argument, e.g. host-side allocations:
/// @code
/// template<typename R>
/// void* allocate(R& resource, size_t bytes, size_t alignment)
/// {
///     static_assert(nanovdb::cuda::is_resource<R>::value, "R must be a synchronous resource");
///     return resource.allocate(bytes, alignment);
/// }
/// @endcode
template <class R, class = void>
struct is_resource : std::false_type {};

template <class R>
struct is_resource<R, std::void_t<
    decltype(std::declval<R&>().allocate(size_t{0}, size_t{0})),
    decltype(std::declval<R&>().deallocate(std::declval<void*>(), size_t{0}, size_t{0}))>>
    : std::true_type {};

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
