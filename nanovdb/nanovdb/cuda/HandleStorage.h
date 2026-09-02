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

#include <nanovdb/GridHandle.h> // for the handle cuda::copyTo transfers
#include <nanovdb/HostBuffer.h> // for the BufferTraits detectors
#include <nanovdb/cuda/Buffer.h> // for noInit and the resource concepts
#include <nanovdb/util/cuda/Util.h> // for cudaCheck

#include <utility> // for std::move
#include <vector> // for the adopted metadata

#include <stdexcept> // for std::runtime_error
#include <type_traits> // for std::is_default_constructible

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


/// @brief The one gateway for constructing a GridHandle from a buffer plus
///        metadata that is already known to be valid -- adopted from another
///        handle, whose own construction from raw bytes did the validation.
struct HandleFactory
{
    template<typename BufferT>
    static GridHandle<BufferT> make(BufferT&& buffer, std::vector<GridHandleMetaData> meta)
    {
        return GridHandle<BufferT>(std::move(buffer), std::move(meta));
    }

    template<typename BufferT>
    static const std::vector<GridHandleMetaData>& meta(const GridHandle<BufferT>& handle)
    {
        return handle.mMetaData;
    }
};

}// namespace detail
/// @brief Deep-copies a grid handle into a different address space: the
///        explicit, stream-carrying transfer between single-space device
///        handles and host-readable handles (HostBuffer or a host-accessible
///        single-space buffer such as a pinned-resource cuda::Buffer).
/// @tparam DstBufferT destination buffer type (specify explicitly)
/// @param src the handle to copy; must not be dual-space (use
///        deviceUpload/deviceDownload on those)
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
/// @return a handle of the destination buffer type with equal contents
/// @details A host-readable destination -- HostBuffer, pinned, or a
///          both-space managed buffer -- synchronizes @c stream before
///          returning, so its host accessors are immediately valid; a
///          device-only destination is stream-ordered, so use its
///          contents on @c stream or synchronize first. The metadata is
///          adopted from the source handle -- it was validated when that
///          handle was constructed from raw bytes -- so no kernel runs and
///          this function is callable from host-only translation units. A
///          pageable host source or destination (HostBuffer) degrades the
///          copy to synchronous behavior; pinned single-space handles keep
///          it asynchronous.
template<typename DstBufferT, typename SrcBufferT>
inline GridHandle<DstBufferT> copyTo(const GridHandle<SrcBufferT>& src, cudaStream_t stream, const DstBufferT* proto = nullptr)
{
    constexpr bool srcDev = BufferHasDeviceSingle<SrcBufferT>::value;
    constexpr bool dstDev = BufferHasDeviceSingle<DstBufferT>::value;
    static_assert(!BufferTraits<SrcBufferT>::hasDeviceDual && !BufferTraits<DstBufferT>::hasDeviceDual,
                  "cuda::copyTo does not support dual-space buffers: use deviceUpload/deviceDownload on the handle");
    static_assert(srcDev || dstDev,
                  "cuda::copyTo is for cross-space transfers involving a device buffer: use GridHandle::copy for host-to-host");
    const uint64_t bytes = src.bufferSize();
    if (bytes == 0u) {
        if constexpr (std::is_default_constructible<DstBufferT>::value) {
            return GridHandle<DstBufferT>();
        } else {
            throw std::runtime_error("cuda::copyTo: an empty handle cannot be copied to a buffer type "
                                     "that is not default-constructible");
        }
    }
    DstBufferT dst = detail::makeTransferStorage<DstBufferT>(bytes, stream, proto);
    const void* srcPtr;
    if constexpr (srcDev) srcPtr = src.deviceData();
    else                  srcPtr = src.data();
    cudaCheck(cudaMemcpyAsync(dst.data(), srcPtr, bytes, cudaMemcpyDefault, stream));
    if constexpr (!dstDev || BufferHasHostSingle<DstBufferT>::value)
        cudaCheck(cudaStreamSynchronize(stream)); // the host-readable result is the postcondition; covers the both-space (managed) destination, whose handle exposes host accessors immediately
    // A handle-to-handle copy adopts the source's metadata, which was
    // validated when that handle was constructed from raw bytes -- no kernel
    // runs here, which is what keeps this header host-includable. A device
    // destination is stream-ordered: use it on @a stream, or synchronize.
    return detail::HandleFactory::make(std::move(dst), detail::HandleFactory::meta(src));
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

#endif // NANOVDB_CUDA_HANDLESTORAGE_H_HAS_BEEN_INCLUDED
