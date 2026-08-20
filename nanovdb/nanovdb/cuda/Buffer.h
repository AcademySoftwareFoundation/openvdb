// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file nanovdb/cuda/Buffer.h
///
/// @brief Typed containers for CUDA memory: the owning, resource-aware,
///        stream-ordered cuda::Buffer and the non-owning cuda::BufferView.

#ifndef NANOVDB_CUDA_BUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_BUFFER_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/util/cuda/Util.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace nanovdb {

namespace cuda {

/// @brief Tag type selecting the Buffer constructors that skip element
///        initialization, leaving the contents indeterminate.
struct NoInit {};
inline constexpr NoInit noInit{};

namespace detail {

/// @brief Conditional stream storage for Buffer. The async specialization
///        retains the stream of the most recent allocation; the synchronous
///        specialization is an empty base, so a Buffer over a synchronous
///        resource carries no stream state and exposes no stream API.
template<bool IsAsync>
struct StreamHolder {};

template<>
struct StreamHolder<true> { cudaStream_t mStream = 0; };

} // namespace detail

/// @brief Owning, typed container of @c T elements allocated from a memory
///        resource @c R held by value.
/// @tparam T element type; sizes are expressed in elements, not bytes
/// @tparam R memory resource, either stream-ordered (AsyncResource concept,
///         see is_async_resource) or synchronous (Resource concept, see
///         is_resource). When @c R provides both interfaces the stream-ordered
///         one is used.
/// @details With a stream-ordered resource the Buffer retains the stream of
///          the most recent allocation (or the one supplied via set_stream)
///          and orders its deallocation on that stream. Buffer is move-only.
template<typename T, typename R = DeviceResource>
class Buffer : private detail::StreamHolder<is_async_resource<R>::value>
{
    static_assert(is_async_resource<R>::value || is_resource<R>::value,
                  "Buffer requires R to model the AsyncResource or the Resource concept");
    static_assert(std::is_trivially_copyable<T>::value,
                  "Buffer requires a trivially copyable T: elements are copied bytewise and never constructed or destroyed");

    static constexpr bool IsAsync = is_async_resource<R>::value;

public:
    /// @brief Element and resource types, for generic code that rebinds one
    ///        or constructs sibling buffers over the same resource.
    using ElementType  = T;
    using ResourceType = R;

    /// @brief Alias for a sibling buffer over the same resource with a
    ///        different element type.
    template<typename U>
    using rebind = Buffer<U, R>;

private:
    R      mResource;
    T*     mData = nullptr;
    size_t mSize = 0; // element count

public:
    /// @brief Default c-tor of an empty buffer; performs no allocation.
    Buffer() = default;

    /// @brief C-tor allocating @c count uninitialized elements, stream-ordered
    ///        on @c stream. Parameter order follows cuda::buffer:
    ///        (stream, resource, count, no_init).
    /// @param stream cuda stream the allocation is ordered on
    /// @param resource resource instance the buffer takes ownership of
    /// @param count number of elements
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    explicit Buffer(cudaStream_t stream, R resource, size_t count, NoInit)
        : detail::StreamHolder<true>{stream}
        , mResource(std::move(resource))
    {
        this->allocate(count, stream);
    }

    /// @brief Convenience c-tor using a default-constructed resource.
    /// @param stream cuda stream the allocation is ordered on
    /// @param count number of elements
    /// @note There is deliberately no count c-tor without NoInit: implicit
    ///       initialization of freshly allocated memory costs a hidden fill
    ///       pass that the dominant allocate-then-overwrite pattern wastes,
    ///       so initialization is always explicit (matching cuda::buffer,
    ///       whose count c-tor likewise requires cuda::no_init).
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    Buffer(cudaStream_t stream, size_t count, NoInit) : Buffer(stream, R(), count, noInit) {}

    /// @brief C-tor allocating @c count uninitialized elements from a
    ///        synchronous resource: the stream-less analog of
    ///        (stream, resource, count, no_init).
    /// @param resource resource instance the buffer takes ownership of
    /// @param count number of elements
    template<typename S = R, std::enable_if_t<!is_async_resource<S>::value && is_resource<S>::value, int> = 0>
    explicit Buffer(R resource, size_t count, NoInit)
        : mResource(std::move(resource))
    {
        this->allocate(count, cudaStream_t{0});
    }

    /// @brief Convenience c-tor using a default-constructed resource.
    /// @param count number of elements
    template<typename S = R, std::enable_if_t<!is_async_resource<S>::value && is_resource<S>::value, int> = 0>
    Buffer(size_t count, NoInit) : Buffer(R(), count, noInit) {}

    /// @brief Explicitly disallow copy construction and assignment operation
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /// @brief Move c-tor; steals the allocation (and retained stream, if any)
    ///        and leaves @c other empty.
    Buffer(Buffer&& other) noexcept
        : detail::StreamHolder<IsAsync>(other)
        , mResource(std::move(other.mResource))
        , mData(other.mData)
        , mSize(other.mSize)
    {
        other.mData = nullptr;
        other.mSize = 0;
    }

    /// @brief Move assignment; frees the current allocation first, then steals
    ///        from @c other and leaves it empty. Self-move is a no-op.
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this != &other) {
            this->destroy();
            static_cast<detail::StreamHolder<IsAsync>&>(*this) = other;
            mResource = std::move(other.mResource);
            mData = other.mData;
            mSize = other.mSize;
            other.mData = nullptr;
            other.mSize = 0;
        }
        return *this;
    }

    /// @brief Returns a deep copy of this buffer, allocated from a copy of the
    ///        resource; the allocation and element copy are ordered on @c stream,
    ///        which becomes the copy's retained stream.
    /// @param stream cuda stream the allocation and element copy are ordered on
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    Buffer copy(cudaStream_t stream) const
    {
        Buffer out(stream, mResource, mSize, noInit);
        if (mData) cudaCheck(cudaMemcpyAsync(out.mData, mData, this->size_bytes(), cudaMemcpyDefault, stream));
        return out;
    }

    /// @brief Returns a deep copy of this buffer ordered on the retained
    ///        stream, i.e. copy(this->stream()).
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    Buffer copy() const { return this->copy(this->stream()); }

    /// @brief Returns a deep copy of this buffer, allocated from a copy of the
    ///        synchronous resource.
    template<typename S = R, std::enable_if_t<!is_async_resource<S>::value && is_resource<S>::value, int> = 0>
    Buffer copy() const
    {
        Buffer out(mResource, mSize, noInit);
        if (mData) cudaCheck(cudaMemcpy(out.mData, mData, this->size_bytes(), cudaMemcpyDefault));
        return out;
    }

    /// @brief D-tor. A stream-ordered resource frees on the retained stream;
    ///        a synchronous resource frees immediately.
    ~Buffer() { this->destroy(); }

    /// @brief Returns the retained stream, i.e. the stream the buffer's memory
    ///        will be freed on.
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    cudaStream_t stream() const { return this->mStream; }

    /// @brief Replaces the retained stream without synchronizing; subsequent
    ///        deallocation (and destruction) is ordered on @c stream instead.
    /// @param stream cuda stream subsequent deallocation is ordered on
    /// @warning The caller is responsible for ordering @c stream after any
    ///          in-flight work that uses the buffer's memory. This deliberately
    ///          does not synchronize, matching cuda::buffer's set_stream, which
    ///          avoids implicit synchronization in fundamental primitives.
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    void set_stream(cudaStream_t stream) { this->mStream = stream; }

    /// @brief Resizes the buffer to @c count elements, preserving the leading
    ///        min(old, new) elements. Every operation — the new allocation, the
    ///        prefix copy, and the free of the old block — is ordered on
    ///        @c stream, which becomes the retained stream: the prefix copy is
    ///        the old block's last use, so that is the stream its free must be
    ///        ordered on.
    /// @param count number of elements
    /// @param stream cuda stream the reallocation is ordered on
    /// @warning The caller is responsible for ordering @c stream after any
    ///          in-flight work on the previously retained stream that uses the
    ///          buffer's memory.
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    void resize(size_t count, cudaStream_t stream)
    {
        if (count != mSize) {
            // No member is mutated until every throwing operation has
            // succeeded, so a failed resize leaves the buffer untouched --
            // including its retained stream.
            T* newData = count ? static_cast<T*>(mResource.allocate_async(checkedBytes(count), R::DEFAULT_ALIGNMENT, stream))
                               : nullptr;
            if (newData && mData) {
                const size_t prefix = count < mSize ? count : mSize;
                try {
                    cudaCheck(cudaMemcpyAsync(newData, mData, prefix * sizeof(T), cudaMemcpyDefault, stream));
                }
                catch (...) {
                    mResource.deallocate_async(newData, checkedBytes(count), R::DEFAULT_ALIGNMENT, stream);
                    throw;
                }
            }
            this->mStream = stream;
            this->deallocate(mData, mSize); // ordered on stream: after the prefix copy
            mData = newData;
            mSize = count;
        }
        else {
            this->mStream = stream; // no reallocation: set_stream semantics
        }
    }

    /// @brief Resizes the buffer to @c count elements through the synchronous
    ///        resource, preserving the leading min(old, new) elements.
    /// @param count number of elements
    template<typename S = R, std::enable_if_t<!is_async_resource<S>::value && is_resource<S>::value, int> = 0>
    void resize(size_t count)
    {
        if (count == mSize) return;
        // No member is mutated until every throwing operation has succeeded,
        // so a failed resize leaves the buffer untouched.
        T* newData = count ? static_cast<T*>(mResource.allocate(checkedBytes(count), R::DEFAULT_ALIGNMENT))
                           : nullptr;
        if (newData && mData) {
            const size_t prefix = count < mSize ? count : mSize;
            try {
                cudaCheck(cudaMemcpy(newData, mData, prefix * sizeof(T), cudaMemcpyDefault));
            }
            catch (...) {
                mResource.deallocate(newData, checkedBytes(count), R::DEFAULT_ALIGNMENT);
                throw;
            }
        }
        this->deallocate(mData, mSize);
        mData = newData;
        mSize = count;
    }

    /// @brief Returns a pointer to the elements, or nullptr if empty.
    T*       data()       { return mData; }
    const T* data() const { return mData; }

    /// @brief Returns a copy of the resource; for a ResourceRef this refers
    ///        to the same underlying instance.
    /// @note Requires R to be copy-constructible (the cuda::mr convention:
    ///       resources are cheap handles). A resource that owns its pool by
    ///       value hands the caller an independent copy of that pool.
    R resource() const { return mResource; }

    /// @brief Returns the number of elements.
    size_t size() const { return mSize; }

    /// @brief Returns the size of the buffer's allocation in bytes.
    size_t size_bytes() const { return mSize * sizeof(T); }

    /// @brief Returns true if this buffer manages no memory.
    bool empty() const { return mSize == 0; }

    /// @brief Frees the buffer memory (if any) and resets to the empty state.
    ///        A stream-ordered resource frees on the retained stream.
    /// @note Spelled destroy to match cuda::buffer. This is the name to use.
    void destroy()
    {
        this->deallocate(mData, mSize);
        mData = nullptr;
        mSize = 0;
    }

    /// @brief Frees the buffer memory (if any) and resets to the empty state.
    /// @deprecated Use destroy(). Documentation-level only for now: the
    ///             [[deprecated]] attribute would fire from GridHandle::reset
    ///             and NodeManager::reset, which must keep calling clear()
    ///             until every buffer type provides destroy().
    /// @note Transitional, and not the name to use: it exists only because
    ///       GridHandle::reset still calls clear() on its buffer. It goes away
    ///       when the legacy dual buffers do and GridHandle moves to destroy().
    void clear() { this->destroy(); }

    /// @brief Frees the buffer memory (if any) on @c stream and resets to the
    ///        empty state. @c stream becomes the retained stream.
    /// @param stream cuda stream the deallocation is ordered on
    /// @warning The caller is responsible for ordering @c stream after any
    ///          in-flight work that uses the buffer's memory.
    template<typename S = R, std::enable_if_t<is_async_resource<S>::value, int> = 0>
    void destroy(cudaStream_t stream)
    {
        this->mStream = stream;
        this->destroy();
    }

    /// @brief Exchanges the contents of this buffer with @c other. Neither
    ///        buffer allocates, frees, or copies element data.
    /// @param other buffer to exchange contents with
    void swap(Buffer& other) noexcept
    {
        auto& lhs = static_cast<detail::StreamHolder<IsAsync>&>(*this);
        auto& rhs = static_cast<detail::StreamHolder<IsAsync>&>(other);
        std::swap(lhs, rhs);
        std::swap(mResource, other.mResource);
        std::swap(mData, other.mData);
        std::swap(mSize, other.mSize);
    }

private:
    /// @brief Returns @c count * sizeof(T), throwing std::runtime_error if the
    ///        byte size would overflow size_t instead of silently wrapping into
    ///        a tiny allocation.
    /// @param count number of elements
    static size_t checkedBytes(size_t count)
    {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("nanovdb::cuda::Buffer: element count overflows the byte size");
        return count * sizeof(T);
    }

    /// @brief Allocates @c count elements through the resource and records the
    ///        new extent. A zero count allocates nothing.
    /// @param count number of elements
    /// @param stream cuda stream the allocation is ordered on; used by the
    ///        stream-ordered form and ignored by the synchronous one
    void allocate(size_t count, cudaStream_t stream)
    {
        if (count) {
            if constexpr (IsAsync)
                mData = static_cast<T*>(mResource.allocate_async(checkedBytes(count), R::DEFAULT_ALIGNMENT, stream));
            else
                mData = static_cast<T*>(mResource.allocate(checkedBytes(count), R::DEFAULT_ALIGNMENT));
        }
        mSize = count;
    }

    /// @brief Frees @c count elements at @c p through the resource; the
    ///        stream-ordered form frees on the retained stream. Null is a no-op.
    /// @param p pointer to the elements to free
    /// @param count number of elements
    void deallocate(T* p, size_t count)
    {
        if (!p) return;
        if constexpr (IsAsync)
            mResource.deallocate_async(p, count * sizeof(T), R::DEFAULT_ALIGNMENT, this->mStream);
        else
            mResource.deallocate(p, count * sizeof(T), R::DEFAULT_ALIGNMENT);
    }
}; // Buffer<T, R> class

/// @brief Non-owning, trivially copyable view of a contiguous range of @c T
///        elements, with span semantics.
/// @tparam T element type; spell constness in the element type
///         (e.g. BufferView<const std::byte> is the read-only form), since
///         const on the view itself is shallow.
template<typename T>
class BufferView
{
    T*     mData = nullptr;
    size_t mSize = 0; // element count

public:
    /// @brief Default c-tor of an empty view.
    BufferView() = default;

    /// @brief C-tor viewing a contiguous range; the caller guarantees the
    ///        underlying storage outlives every use of the view.
    /// @param data pointer to the first element
    /// @param count number of elements
    /// @throw std::runtime_error if @c data is null while @c count is non-zero.
    BufferView(T* data, size_t count) : mData(data), mSize(count)
    {
        if (data == nullptr && count != 0)
            throw std::runtime_error("BufferView: null data with a non-zero element count");
    }

    /// @brief Returns a pointer to the viewed elements, or nullptr if empty.
    T* data() const { return mData; }

    /// @brief Returns the number of viewed elements.
    size_t size() const { return mSize; }

    /// @brief Returns the size of the viewed range in bytes.
    size_t size_bytes() const { return mSize * sizeof(T); }

    /// @brief Returns true if this view references no elements.
    bool empty() const { return mSize == 0; }

    /// @brief Detaches the view (nulls the pointer and zeroes the size)
    ///        without touching the underlying storage. This is the one
    ///        deliberate deviation from std::span, required by the buffer
    ///        static interface: GridHandle::reset() calls buffer.clear().
    void clear()
    {
        mData = nullptr;
        mSize = 0;
    }
}; // BufferView<T> class

} // namespace cuda

// Primary template defined in HostBuffer.h; declared here so this header
// stays self-contained without pulling in the host-buffer machinery.
template<typename BufferT>
struct BufferTraits;

/// @brief GridHandle support for the single-space cuda::Buffer: the buffer
///        owns exactly one allocation, resident on the device, so the handle
///        parses metadata through a device read and exposes only the device
///        accessors. Requires byte-addressed storage.
/// @note This trait doubles as the definition of the single-space
///       device-buffer concept: a buffer whose BufferTraits specialization
///       sets hasDeviceSingle guarantees ElementType and ResourceType
///       typedefs, data(), size() and size_bytes() (byte-addressed elements,
///       enforced by the consumer), resource(), copy(), clear(), and stream()
///       when the resource is stream-ordered. Any consumer of hasDeviceSingle
///       may rely on exactly this interface and nothing more; in particular,
///       scratch allocates through resource() as a cuda::Buffer, so a
///       conforming buffer never needs to be constructible by a consumer.
template<typename T, typename R>
struct BufferTraits<cuda::Buffer<T, R>>
{
    static constexpr bool hasDeviceDual   = false;
    // Device-resident storage; the byte-addressed requirement is enforced by
    // the single-space GridHandle constructor, so trait queries stay
    // answerable for any element type.
    static constexpr bool hasDeviceSingle = !cuda::is_host_accessible_resource<R>::value;
    // A buffer over a host-accessible resource (e.g. PinnedResource) is
    // host-readable, but GridHandle's host paths also require the create()
    // static interface and byte-count size semantics that cuda::Buffer does
    // not provide -- GridHandle rejects such buffers with a named error until
    // that adaptation lands.
    static constexpr bool hasHostSingle   = cuda::is_host_accessible_resource<R>::value;
};

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_BUFFER_H_HAS_BEEN_INCLUDED
