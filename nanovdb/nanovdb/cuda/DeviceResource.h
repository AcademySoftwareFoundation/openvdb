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

/// @brief CRTP base supplying the synchronous half of the resource concept in
///        terms of the stream-ordered half, so a custom stream-ordered resource
///        only has to write allocate_async and deallocate_async.
/// @tparam Derived the resource deriving from this base
/// @details A stream-ordered resource must also model the synchronous concept
///          (is_async_resource implies is_resource), which means writing four
///          methods where two would do. The synchronous pair is not a bare
///          delegate: memory from allocate must be usable immediately on any
///          stream, so the null-stream allocation has to be synchronized before
///          it is returned. Omitting that synchronization yields memory that
///          satisfies the concept but is not actually synchronous -- a race
///          rather than a compile error -- so it lives here rather than being
///          rewritten per resource.
/// @code
/// struct MyResource : nanovdb::cuda::SyncFromAsync<MyResource> {
///     static constexpr size_t DEFAULT_ALIGNMENT = 256;
///     void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream);
///     void  deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream);
/// };
/// @endcode
template <class Derived>
struct SyncFromAsync
{
    /// @brief Allocates @c bytes usable on any stream when this returns.
    /// @param bytes number of bytes to allocate
    /// @param alignment requested alignment
    /// @note Every call synchronizes the null stream; on hot paths prefer the
    ///       stream-ordered pair.
    void* allocate(size_t bytes, size_t alignment)
    {
        void* p = static_cast<Derived&>(*this).allocate_async(bytes, alignment, cudaStream_t{0});
        cudaCheck(cudaStreamSynchronize(cudaStream_t{0}));
        return p;
    }

    /// @brief Frees @c p on the null stream.
    /// @param p pointer previously returned by allocate
    /// @param bytes size passed to the matching allocate
    /// @param alignment alignment passed to the matching allocate
    /// @note No synchronization here: the synchronous concept's contract is
    ///       that the memory is already quiescent when deallocate is called.
    void deallocate(void* p, size_t bytes, size_t alignment)
    {
        static_cast<Derived&>(*this).deallocate_async(p, bytes, alignment, cudaStream_t{0});
    }
};

/// @brief Synchronous device memory resource backed by cudaMalloc/cudaFree.
///        Models only the Resource concept: it never touches stream-ordered
///        allocation, so it works on devices without memory-pool support
///        (cudaDevAttrMemoryPoolsSupported == 0), where DeviceResource's
///        cudaMallocAsync path fails by design. Pair with AsyncFromSync to
///        drive the stream-ordered builders on such a device.
class MallocResource
{
public:
    // cudaMalloc aligns memory to 256 bytes by default
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    /// @brief Allocates @c bytes with cudaMalloc; valid on every stream when
    ///        this returns. A zero request returns nullptr.
    void* allocate(size_t bytes, size_t)
    {
        if (bytes == 0) return nullptr;
        void* p = nullptr;
        cudaCheck(cudaMalloc(&p, bytes));
        return p;
    }

    /// @brief Frees @c p with cudaFree; the caller guarantees that device work
    ///        touching the memory has completed.
    void deallocate(void* p, size_t, size_t) { cudaCheck(cudaFree(p)); }
};// MallocResource

/// @brief Wrapper presenting a synchronous resource as a stream-ordered one,
///        so it can drive components that require the AsyncResource concept
///        (TempPool and the GPU builders).
/// @tparam R the wrapped synchronous resource, held by value; wrap a
///         ResourceRef<R> to borrow a stateful instance instead.
/// @details The mirror of SyncFromAsync, and the analog of cuda::mr's
///          synchronous_resource_adapter. allocate_async forwards to
///          R::allocate, whose memory is immediately valid on every stream --
///          a stronger guarantee than stream-ordering requires.
///          deallocate_async synchronizes @c stream before R::deallocate,
///          establishing the quiescence the synchronous contract demands.
/// @warning Every deallocation synchronizes its stream, so expect
///          serialization relative to a genuinely stream-ordered resource.
///          That is the unavoidable cost of a synchronous backend under a
///          stream-ordered algorithm; this wrapper exists so the cost is
///          explicit and chosen by the caller -- e.g. on a device without
///          memory-pool support -- rather than silently substituted.
template<class R>
struct AsyncFromSync
{
    static_assert(is_resource<R>::value,
                  "AsyncFromSync requires R to model the synchronous Resource concept");

    static constexpr size_t DEFAULT_ALIGNMENT = R::DEFAULT_ALIGNMENT;

    R resource;

    /// @brief Allocates through the synchronous resource; the result is valid
    ///        on every stream, hence trivially valid on @c stream.
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t) { return resource.allocate(bytes, alignment); }

    /// @brief Synchronizes @c stream, then frees through the synchronous
    ///        resource -- the synchronize makes the quiescence contract hold.
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream)
    {
        cudaCheck(cudaStreamSynchronize(stream));
        resource.deallocate(p, bytes, alignment);
    }

    /// @brief Synchronous pair, forwarding to the wrapped resource.
    void* allocate(size_t bytes, size_t alignment) { return resource.allocate(bytes, alignment); }
    void  deallocate(void* p, size_t bytes, size_t alignment) { resource.deallocate(p, bytes, alignment); }
};// AsyncFromSync<R>

/// @brief Non-owning reference to a memory resource that is itself a resource:
///        copying the ref shares the underlying resource rather than copying it.
/// @tparam R the referenced resource type
/// @details Types that hold their resource by value -- cuda::Buffer, matching
///          cuda::buffer -- select their ownership semantics by what is placed
///          in that slot: a concrete resource is owned as a copy, while a
///          ResourceRef borrows. This is the same division cuda::mr draws
///          between any_resource (owning) and resource_ref (borrowing), and the
///          same shape as std::pmr::polymorphic_allocator over memory_resource*.
///          Use it when a resource is stateful or long-lived and a container
///          must allocate through *that* instance rather than a copy of it.
/// @warning The referenced resource must outlive every use of this ref and of
///          all copies of it, including any container holding one.
template <class R>
struct ResourceRef
{
    static_assert(is_async_resource<R>::value || is_resource<R>::value,
                  "ResourceRef requires R to model the AsyncResource or the Resource concept");

    static constexpr size_t DEFAULT_ALIGNMENT = R::DEFAULT_ALIGNMENT;

    /// @brief Constructs a ref borrowing @c resource.
    /// @param resource resource to allocate from; must outlive this ref
    ResourceRef(R& resource) : mResource(&resource) {}

    /// @{
    /// @brief Stream-ordered pair, present only when @c R models AsyncResource,
    ///        so a ref over a synchronous resource does not misreport its tier.
    template<class S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream)
    {
        return mResource->allocate_async(bytes, alignment, stream);
    }
    template<class S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream)
    {
        mResource->deallocate_async(p, bytes, alignment, stream);
    }
    /// @}

    /// @brief Synchronous pair, forwarding to the referenced resource.
    void* allocate(size_t bytes, size_t alignment) { return mResource->allocate(bytes, alignment); }
    void  deallocate(void* p, size_t bytes, size_t alignment) { mResource->deallocate(p, bytes, alignment); }

    /// @brief Two refs compare equal iff they reference the same resource, i.e.
    ///        memory allocated through one may be deallocated through the other.
    friend bool operator==(ResourceRef lhs, ResourceRef rhs) { return lhs.mResource == rhs.mResource; }
    friend bool operator!=(ResourceRef lhs, ResourceRef rhs) { return lhs.mResource != rhs.mResource; }

private:
    R* mResource;
};// ResourceRef<R>

}

} // namespace nanovdb::cuda

#endif // end of NANOVDB_CUDA_DEVICERESOURCE_H_HAS_BEEN_INCLUDED
