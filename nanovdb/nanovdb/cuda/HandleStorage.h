// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/cuda/HandleStorage.h

    \brief Allocates the device-resident storage behind a GridHandle or
           NodeManagerHandle for any buffer family: through the static
           create() interface for buffers that provide it (the dual-space
           DeviceBuffer family), and through the buffer's memory resource
           for a single-space cuda::Buffer. This is the bridge that lets
           every tool entry point accept either buffer family.

    \note This header is host-includable: it calls the CUDA runtime but
          launches no kernels, so a plain C++ translation unit (linked
          against the CUDA runtime) can allocate storage and transfer grid
          handles with cuda::copyTo.
*/

#ifndef NANOVDB_CUDA_HANDLESTORAGE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_HANDLESTORAGE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_HANDLESTORAGE_DEFERS_GRIDHANDLE_CUH // see the end of nanovdb/GridHandle.h

#include <nanovdb/GridHandle.h> // for the handle cuda::copyTo transfers
#include <nanovdb/HostBuffer.h> // for the BufferTraits detectors
#include <nanovdb/cuda/Buffer.h> // for noInit and the resource concepts
#include <nanovdb/util/cuda/Util.h> // for cudaCheck

#include <utility> // for std::move
#include <vector> // for the adopted metadata

#include <stdexcept> // for std::runtime_error
#include <type_traits> // for std::is_default_constructible, std::enable_if, std::is_same, std::is_convertible, std::is_null_pointer

namespace nanovdb {

namespace cuda {

namespace detail {

/// @brief Allocates @c bytes of device-resident storage of buffer type
///        @c BufferT: single-space buffers allocate through @c pool's
///        resource -- on @c stream when the resource is stream-ordered --
///        and every other buffer type goes through its static
///        create(bytes, pool, device, stream) interface.
/// @param bytes size of the allocation
/// @param proto prototype buffer or null: passed through as the pool for
///        create()-style buffers; the source of the resource for
///        single-space buffers, whose resource is default-constructed when
///        @c proto is null
/// @param device device the storage lives on; single-space buffers
///        allocate on the current device, which every call site has
///        already made current
/// @param stream stream the allocation is ordered on where supported
template<typename BufferT>
inline BufferT createDeviceStorage(uint64_t bytes, const BufferT* proto, int device, cudaStream_t stream)
{
    if constexpr (BufferHasDeviceSingle<BufferT>::value) {
        using ResourceT = typename BufferT::ResourceType;
        (void)device;
        if (!proto) {
            if constexpr (std::is_default_constructible<ResourceT>::value) {
                if constexpr (is_async_resource<ResourceT>::value) return BufferT(stream, bytes, noInit);
                else                                               return BufferT(bytes, noInit);
            } else {
                throw std::runtime_error("createDeviceStorage: a buffer over a non-default-constructible "
                                         "resource requires a prototype buffer to take the resource from");
            }
        }
        if constexpr (is_async_resource<ResourceT>::value) {
            return BufferT(stream, proto->resource(), bytes, noInit);
        } else {
            (void)stream;
            return BufferT(proto->resource(), bytes, noInit);
        }
    } else {
        return BufferT::create(bytes, proto, device, stream);
    }
}

/// @brief The device address of a storage buffer made by
///        createDeviceStorage: data() for a single-space buffer, whose one
///        allocation is the device allocation, and deviceData() for the
///        dual-space family.
template<typename BufferT>
inline void* deviceStorageData(BufferT& buffer)
{
    if constexpr (BufferHasDeviceSingle<BufferT>::value) return buffer.data();
    else return buffer.deviceData();
}

/// @brief True when @c BufferT is a stream rather than a buffer: cudaStream_t, or
///        another non-buffer type that converts to it. nullptr is not a stream here.
template<typename BufferT>
struct IsStreamArgument
{
    static constexpr bool value = std::is_convertible<BufferT, cudaStream_t>::value && !std::is_null_pointer<BufferT>::value
                                  && !BufferTraits<BufferT>::hasDeviceDual && !BufferHasDeviceSingle<BufferT>::value
                                  && !BufferHasHostSingle<BufferT>::value;
};
/// @brief Return type of a tool's prototype-buffer form: @c T, except when
///        @c BufferT deduced as a stream (cudaStream_t, or any other type that
///        converts to it and is not a buffer, such as a stream wrapper), i.e.
///        the call passed a stream where the prototype goes, which is the
///        resource form's signature. A literal 0 or nullptr keeps the prototype
///        form, whose static_assert asks for cudaStream_t(0).
///        Removing the prototype form there leaves the resource form as the
///        only candidate rather than relying on partial ordering, which
///        compilers resolve differently for these overloads.
template<typename BufferT, typename T>
using PrototypeFormResult = typename std::enable_if<!IsStreamArgument<BufferT>::value, T>::type;

/// @brief Orders the host after @c stream where a handle is about to be
///        constructed from bytes still being written on it: a single-space
///        buffer over a synchronous resource retains no stream, so the
///        constructor's metadata parse (which runs on the default stream) is
///        not otherwise ordered after the producer. A no-op for dual-space
///        buffers (their constructor path predates this bridge) and for
///        stream-ordered resources (the buffer retains the stream).
template<typename BufferT>
inline void orderBeforeHandleConstruction(cudaStream_t stream)
{
    if constexpr (BufferHasDeviceSingle<BufferT>::value) {
        if constexpr (!is_async_resource<typename BufferT::ResourceType>::value)
            cudaCheck(cudaStreamSynchronize(stream));
    }
    (void)stream;
}

/// @brief Allocates @c bytes of destination storage for a cross-space
///        transfer: single-space buffers allocate through @c proto's resource
///        (or a default-constructed resource without one), on @c stream when
///        the resource is stream-ordered; buffers providing create() go
///        through it.
template<typename DstBufferT>
inline DstBufferT makeTransferStorage(uint64_t bytes, cudaStream_t stream, const DstBufferT* proto)
{
    if constexpr (BufferHasDeviceSingle<DstBufferT>::value || BufferHasHostSingle<DstBufferT>::value) {
        using ResourceT = typename DstBufferT::ResourceType;
        if (!proto) {
            // both branches of a plain conditional would instantiate the
            // default-resource constructor, breaking non-default-constructible
            // resources (e.g. ResourceRef) even for callers that pass a proto
            if constexpr (std::is_default_constructible<ResourceT>::value) {
                if constexpr (is_async_resource<ResourceT>::value) return DstBufferT(stream, bytes, noInit);
                else                                               return DstBufferT(bytes, noInit);
            } else {
                throw std::runtime_error("cuda::copyTo: a destination buffer over a non-default-constructible "
                                         "resource requires a prototype buffer");
            }
        }
        if constexpr (is_async_resource<ResourceT>::value) {
            return DstBufferT(stream, proto->resource(), bytes, noInit);
        } else {
            (void)stream;
            return DstBufferT(proto->resource(), bytes, noInit);
        }
    } else {
        return DstBufferT::create(bytes, proto);
    }
}

/// @deprecated Use GridHandle::adopt and GridHandle::metaData.
struct [[deprecated("use GridHandle::adopt(buffer, meta) and GridHandle::metaData()")]] HandleFactory
{
    template<typename BufferT>
    static GridHandle<BufferT> make(BufferT&& buffer, std::vector<GridHandleMetaData> meta)
    {
        return GridHandle<BufferT>::adopt(std::move(buffer), std::move(meta));
    }

    template<typename BufferT>
    static const std::vector<GridHandleMetaData>& meta(const GridHandle<BufferT>& handle)
    {
        return handle.metaData();
    }
};

}// namespace detail

/// @brief Deep-copies a grid handle into a different address space: the
///        explicit, stream-carrying transfer between single-space device
///        handles and host-readable handles (HostBuffer or a host-accessible
///        single-space buffer such as a pinned-resource cuda::Buffer).
/// @tparam DstBufferT destination buffer type (specify explicitly)
/// @param src the handle to copy; must not use a deprecated dual-space
///        buffer (build device handles into cuda::Buffer<std::byte> instead)
/// @param stream stream the copy is issued on; a device destination buffer
///        with a stream-ordered resource retains it
/// @warning Passing a stream other than the source buffer's retained stream
///          makes the caller responsible for ordering: prior work on the
///          source (and the source's later destruction, which frees on its
///          own stream) must be ordered against @a stream by the caller,
///          e.g. with cudaStreamWaitEvent or a synchronization. The
///          stream-less overload below has no such requirement for a source
///          with a retained stream. A source WITHOUT one (a synchronous
///          resource, e.g. a pinned-resource buffer) is the caller's to keep
///          alive under either overload: its destruction frees host memory
///          immediately, unordered against the still-asynchronous copy, so
///          synchronize @a stream before destroying such a source. (A
///          pageable HostBuffer source is exempt: its copy degrades to
///          synchronous behavior.)
/// @param proto optional buffer whose resource (or pool, for buffers
///        providing create()) allocates the destination storage; without it
///        the destination resource is default-constructed
/// @return a handle of the destination buffer type with equal contents; an
///         empty @a src yields an empty handle, over @a proto's resource for
///         a single-space destination when given
/// @throw std::runtime_error if @a src is empty and DstBufferT is not
///        default-constructible, unless it is single-space and @a proto is given
/// @details A host-readable destination -- HostBuffer, pinned, or a
///          both-space managed buffer -- synchronizes @c stream before
///          returning, so its host accessors are immediately valid; a
///          device-only destination is stream-ordered, so use its
///          contents on @c stream or synchronize first. The destination
///          reuses the source handle's metadata, which was checked when the
///          source was made, so no kernel runs and this function is
///          callable from host-only translation units. A
///          pageable host source or destination (HostBuffer) degrades the
///          copy to synchronous behavior; pinned single-space handles keep
///          it asynchronous.
template<typename DstBufferT, typename SrcBufferT>
inline GridHandle<DstBufferT> copyTo(const GridHandle<SrcBufferT>& src, cudaStream_t stream, const DstBufferT* proto = nullptr)
{
    constexpr bool srcDev = BufferHasDeviceSingle<SrcBufferT>::value;
    constexpr bool dstDev = BufferHasDeviceSingle<DstBufferT>::value;
    static_assert(!BufferTraits<SrcBufferT>::hasDeviceDual && !BufferTraits<DstBufferT>::hasDeviceDual,
                  "cuda::copyTo does not support the deprecated dual-space buffers: build device handles into cuda::Buffer<std::byte> by passing a cuda::Buffer<std::byte>() prototype to the GPU tool");
    static_assert(srcDev || dstDev,
                  "cuda::copyTo is for cross-space transfers involving a device buffer: use GridHandle::copy for host-to-host");
    const uint64_t bytes = src.bufferSize();
    if (bytes == 0u) {// an empty handle copies to an empty handle, over the prototype's resource where given
        if constexpr (BufferHasDeviceSingle<DstBufferT>::value || BufferHasHostSingle<DstBufferT>::value) {
            if (proto) return detail::HandleFactory::make(detail::makeTransferStorage<DstBufferT>(0, stream, proto), {});
        }
        if constexpr (std::is_default_constructible<DstBufferT>::value) {
            return GridHandle<DstBufferT>();
        } else {
            throw std::runtime_error("cuda::copyTo: copying an empty handle into a buffer type that is not "
                                     "default-constructible requires a single-space buffer type and a prototype buffer");
        }
    }
    DstBufferT dst = detail::makeTransferStorage<DstBufferT>(bytes, stream, proto);
    const void* srcPtr;
    if constexpr (srcDev) srcPtr = src.deviceData();
    else                  srcPtr = src.data();
    cudaCheck(cudaMemcpyAsync(dst.data(), srcPtr, bytes, cudaMemcpyDefault, stream));
    if constexpr (!dstDev || BufferHasHostSingle<DstBufferT>::value)
        cudaCheck(cudaStreamSynchronize(stream)); // the host-readable result is the postcondition; covers the both-space (managed) destination, whose handle exposes host accessors immediately
    // The copy reuses the source's layout unchecked, so no kernel runs here,
    // which keeps this header host-includable. A device destination is
    // stream-ordered: use it on @a stream, or synchronize.
    return nanovdb::detail::LayoutCopy::make(std::move(dst), src);
}// cuda::copyTo

/// @brief Convenience overload issuing the copy on the source buffer's
///        retained stream when it has one (any single-space source over a
///        stream-ordered resource), the default stream otherwise.
template<typename DstBufferT, typename SrcBufferT>
inline GridHandle<DstBufferT> copyTo(const GridHandle<SrcBufferT>& src, const DstBufferT* proto = nullptr)
{
    cudaStream_t stream = 0;
    if constexpr (BufferHasStream<SrcBufferT>::value) stream = src.buffer().stream();
    return copyTo<DstBufferT>(src, stream, proto);
}// cuda::copyTo (retained stream)


}// namespace cuda

}// namespace nanovdb

#undef NANOVDB_CUDA_HANDLESTORAGE_DEFERS_GRIDHANDLE_CUH
#if defined(__CUDACC__) && !defined(NANOVDB_TOOLS_CUDA_GRIDCHECKSUM_DEFERS_GRIDHANDLE_CUH)
#include <nanovdb/cuda/GridHandle.cuh>
#endif

#endif // NANOVDB_CUDA_HANDLESTORAGE_H_HAS_BEEN_INCLUDED
