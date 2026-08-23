// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/cuda/GridHandle.cuh

    \author Ken Museth, Doyub Kim

    \date August 3, 2023

    \brief Contains cuda kernels for GridHandle

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/cuda/Buffer.h>// for the resource-aware scratch buffers below
#include <nanovdb/cuda/DeviceBuffer.h>// required for instantiation of move c-tor of GridHandle
#include <nanovdb/tools/cuda/GridChecksum.cuh>// for cuda::updateChecksum
#include <nanovdb/GridHandle.h>

#include <string>// for the grid index in chain-validation error messages
#include <type_traits>// for std::is_default_constructible in cuda::copyTo

namespace nanovdb {

namespace cuda {

namespace detail {

static __global__ void updateGridCount(GridData *d_data, uint32_t gridIndex, uint32_t gridCount, bool *d_dirty)
{
    NANOVDB_ASSERT(gridIndex < gridCount);
    *d_dirty = (d_data->mGridIndex != gridIndex) || (d_data->mGridCount != gridCount);
    if (*d_dirty) {
        d_data->mGridIndex = gridIndex;
        d_data->mGridCount = gridCount;
        if (d_data->mChecksum.isEmpty()) *d_dirty = false;// no need to update checksum if it didn't already exist
    }
}

/// @brief Defect classes the device-side chain walk can report, mapped to
///        exceptions on the host by parseGridChain.
enum class ChainError : uint32_t { Ok = 0, Truncated, Invalid, Inconsistent, BadSize };

/// @brief Result of the device-side chain walk: the defect class and the
///        index of the grid the walk failed at.
struct ChainStatus { ChainError error; uint32_t gridIndex; };

/// @brief Validates every header of the grid chain and fills the metadata
///        scratch in a single pass. Bounds are checked before every header
///        read, so a truncated buffer or a forged header can never cause an
///        out-of-bounds access.
static __global__ void parseGridChainKernel(const GridData *d_head, uint64_t bytes, uint32_t count,
                                            GridHandleMetaData *d_meta, ChainStatus *d_status)
{
    *d_status = ChainStatus{ChainError::Ok, 0u};
    uint64_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (offset + sizeof(GridData) > bytes) { *d_status = {ChainError::Truncated, i}; return; }
        const GridData *data = util::PtrAdd<GridData>(d_head, offset);
        if (!data->isValid()) { *d_status = {ChainError::Invalid, i}; return; }
        if (data->mGridIndex != i || data->mGridCount != count) { *d_status = {ChainError::Inconsistent, i}; return; }
        if (data->mGridSize < sizeof(GridData) || data->mGridSize > bytes - offset) { *d_status = {ChainError::BadSize, i}; return; }
        d_meta[i] = GridHandleMetaData{offset, data->mGridSize, data->mGridType};
        offset += data->mGridSize;
    }
}

/// @brief Reads and validates the head of the grid chain, returning the grid
///        count the chain claims. This is the one synchronizing header read
///        that sizing the metadata scratch requires; the rest of the chain is
///        validated on the device by parseGridChain.
inline uint32_t validGridChainHead(const GridData *d_head, uint64_t bytes, cudaStream_t stream)
{
    if (bytes < sizeof(GridData))
        throw std::runtime_error("GridHandle: grid chain exceeds the device buffer (truncated or corrupt grid data)");
    GridData tmp;
    cudaCheck(cudaMemcpyAsync(&tmp, d_head, sizeof(GridData), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    if (!tmp.isValid()) throw std::runtime_error("GridHandle was constructed with an invalid device buffer");
    if (tmp.mGridCount == 0) throw std::runtime_error("GridHandle: device buffer contains no grids");
    if (tmp.mGridIndex != 0)
        throw std::runtime_error("GridHandle: inconsistent grid index/count in the device buffer's grid chain");
    if (uint64_t(tmp.mGridCount) > bytes / sizeof(GridData))// every grid is at least one full header
        throw std::runtime_error("GridHandle: grid chain exceeds the device buffer (truncated or corrupt grid data)");
    return tmp.mGridCount;
}

/// @brief Bytes of device scratch parseGridChain needs for @c count grids:
///        the metadata array followed by the status word.
inline constexpr uint64_t chainScratchSize(uint32_t count)
{
    return count * sizeof(GridHandleMetaData) + sizeof(ChainStatus);
}

/// @brief Launches the combined validate-and-parse walk over the grid chain
///        and fills @c meta from its scratch: one kernel, one readback and
///        one synchronization regardless of the grid count.
/// @param scratch device scratch of at least chainScratchSize(count) bytes
/// @throw std::runtime_error naming the defect when any header fails validation
template<typename ScratchBufferT>
inline void parseGridChain(const GridData *d_head, uint64_t bytes, uint32_t count,
                           ScratchBufferT &scratch, std::vector<GridHandleMetaData> &meta, cudaStream_t stream)
{
    auto *d_meta   = reinterpret_cast<GridHandleMetaData*>(scratch.data());
    auto *d_status = reinterpret_cast<ChainStatus*>(scratch.data() + count * sizeof(GridHandleMetaData));
    parseGridChainKernel<<<1, 1, 0, stream>>>(d_head, bytes, count, d_meta, d_status);
    cudaCheckError();
    meta.resize(count);
    ChainStatus status;
    cudaCheck(cudaMemcpyAsync(meta.data(), d_meta, count * sizeof(GridHandleMetaData), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(&status, d_status, sizeof(ChainStatus), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    if (status.error == ChainError::Ok) return;
    const std::string where = " (grid " + std::to_string(status.gridIndex) + " of " + std::to_string(count) + ")";
    switch (status.error) {
    case ChainError::Truncated:
        throw std::runtime_error("GridHandle: grid chain exceeds the device buffer (truncated or corrupt grid data)" + where);
    case ChainError::Inconsistent:
        throw std::runtime_error("GridHandle: inconsistent grid index/count in the device buffer's grid chain" + where);
    case ChainError::BadSize:
        throw std::runtime_error("GridHandle: grid size field exceeds the device buffer (truncated or corrupt grid data)" + where);
    default:
        throw std::runtime_error("GridHandle was constructed with an invalid device buffer" + where);
    }
}

/// @brief The buffer's retained stream when its resource is stream-ordered,
///        the default stream otherwise.
template<typename BufferT>
inline cudaStream_t retainedStreamOrDefault(const BufferT& buf)
{
    if constexpr (is_async_resource<typename BufferT::ResourceType>::value) return buf.stream();
    else { (void)buf; return cudaStream_t(0); }
}

/// @brief Constructs the metadata scratch through @c buf's resource, on
///        @c stream for a stream-ordered resource.
template<typename ScratchT, typename BufferT>
inline ScratchT makeMetaScratch(const BufferT& buf, uint64_t count, cudaStream_t stream)
{
    if constexpr (is_async_resource<typename BufferT::ResourceType>::value)
        return ScratchT(stream, buf.resource(), count, noInit);
    else { (void)stream; return ScratchT(buf.resource(), count, noInit); }
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

}// namespace detail

template<typename BufferT, template <class, class...> class VectorT = std::vector>
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
splitGridHandles(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    const void *ptr = handle.deviceData();
    if (ptr == nullptr) return VectorT<GridHandle<BufferT>>();
    VectorT<GridHandle<BufferT>> handles(handle.gridCount());
    Buffer<bool, DeviceResource> dirtyBuf(stream, 1, noInit);
    bool *d_dirty = dirtyBuf.data();
    int device = util::cuda::currentDevice();
    for (uint32_t n=0; n<handle.gridCount(); ++n) {
        bool dirty = false;// set when the checksum needs to be recomputed
        auto buffer = BufferT::create(handle.gridSize(n), other, device, stream);
        GridData *dst = reinterpret_cast<GridData*>(buffer.deviceData());
        const GridData *src = reinterpret_cast<const GridData*>(ptr);
        cudaCheck(cudaMemcpyAsync(dst, src, handle.gridSize(n), cudaMemcpyDeviceToDevice, stream));
        detail::updateGridCount<<<1, 1, 0, stream>>>(dst, 0u, 1u, d_dirty);
        cudaCheckError();
        cudaCheck(cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        if (dirty) tools::cuda::updateChecksum(dst, CheckMode::Partial, stream);
        handles[n] = nanovdb::GridHandle<BufferT>(std::move(buffer));
        ptr = util::PtrAdd(ptr, handle.gridSize(n));
    }
    return handles;
}// cuda::splitGridHandles

template<typename BufferT, template <class, class...> class VectorT>
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
mergeGridHandles(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    uint64_t size = 0u;
    uint32_t counter = 0u, gridCount = 0u;
    for (auto &h : handles) {
        gridCount += h.gridCount();
        for (uint32_t n=0; n<h.gridCount(); ++n) size += h.gridSize(n);
    }
    int device = util::cuda::currentDevice();
    auto buffer = BufferT::create(size, other, device, stream);
    void *dst = buffer.deviceData();
    Buffer<bool, DeviceResource> dirtyBuf(stream, 1, noInit);
    bool *d_dirty = dirtyBuf.data();
    for (auto &h : handles) {
        const void *src = h.deviceData();
        for (uint32_t n=0; n<h.gridCount(); ++n) {
            bool dirty = false;// set when the checksum needs to be recomputed
            cudaCheck(cudaMemcpyAsync(dst, src, h.gridSize(n), cudaMemcpyDeviceToDevice, stream));
            GridData *data = reinterpret_cast<GridData*>(dst);
            detail::updateGridCount<<<1, 1, 0, stream>>>(data, counter++, gridCount, d_dirty);
            cudaCheckError();
            cudaCheck(cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
            cudaCheck(cudaStreamSynchronize(stream));
            if (dirty) tools::cuda::updateChecksum(data, CheckMode::Partial, stream);
            dst = util::PtrAdd(dst, h.gridSize(n));
            src = util::PtrAdd(src, h.gridSize(n));
        }
    }
    return GridHandle<BufferT>(std::move(buffer));
}// cuda::mergeGridHandles

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
///          stream-less overload below has no such requirement.
/// @param proto optional buffer whose resource (or pool, for buffers
///        providing create()) allocates the destination storage; without it
///        the destination resource is default-constructed
/// @return a handle of the destination buffer type with equal contents
/// @details The returned handle is immediately usable: a host-readable
///          destination synchronizes @c stream before returning, and a device
///          destination parses (and validates) its metadata on the
///          transferred bytes, which synchronizes internally. A pageable host
///          source or destination (HostBuffer) degrades the copy to
///          synchronous behavior; pinned single-space handles keep it
///          asynchronous.
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
    constexpr cudaMemcpyKind kind = srcDev ? (dstDev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost)
                                           : cudaMemcpyHostToDevice;
    cudaCheck(cudaMemcpyAsync(dst.data(), srcPtr, bytes, kind, stream));
    if constexpr (dstDev) {
        // A synchronous destination resource retains no stream: order the
        // metadata parse (which runs on the default stream) after the copy.
        if constexpr (!is_async_resource<typename DstBufferT::ResourceType>::value)
            cudaCheck(cudaStreamSynchronize(stream));
    } else {
        cudaCheck(cudaStreamSynchronize(stream));// the host-readable result is the postcondition
    }
    return GridHandle<DstBufferT>(std::move(dst));// the constructor parses (and validates) the metadata
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

template<typename BufferT, template <class, class...> class VectorT = std::vector>
[[deprecated("Use nanovdb::cuda::splitGridHandles instead")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
splitDeviceGrids(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cuda::splitGridHandles(handle, other, stream); }

template<typename BufferT, template <class, class...> class VectorT>
[[deprecated("Use nanovdb::cuda::mergeGridHandles instead")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
mergeDeviceGrids(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cuda::mergeGridHandles<BufferT, VectorT>(handles, other, stream); }

template<typename BufferT>
template<typename T, typename util::enable_if<BufferTraits<T>::hasDeviceDual, int>::type>
GridHandle<BufferT>::GridHandle(T&& buffer)
    : mBuffer(std::move(buffer))
{
    static_assert(util::is_same<T,BufferT>::value, "Expected U==BufferT");
    if (auto *data = reinterpret_cast<const GridData*>(mBuffer.data())) {
        if (!data->isValid()) throw std::runtime_error("GridHandle was constructed with an invalid host buffer");
        mMetaData.resize(data->mGridCount);
        cpyGridHandleMeta(data, mMetaData.data());
    } else {
        if (auto *d_data = reinterpret_cast<const GridData*>(mBuffer.deviceData())) {
            const uint32_t count = cuda::detail::validGridChainHead(d_data, mBuffer.size(), cudaStream_t(0));
            // MallocResource: plain cudaMalloc, so this long-standing parse path
            // keeps working on devices without memory-pool support.
            cuda::Buffer<std::byte, cuda::MallocResource> scratch(cuda::detail::chainScratchSize(count), cuda::noInit);
            cuda::detail::parseGridChain(d_data, mBuffer.size(), count, scratch, mMetaData, cudaStream_t(0));
        }
    }
}// GridHandle(T&& buffer)

// move constructor from a single-space device buffer: all device work runs on
// the buffer's retained stream (or the default stream for a synchronous
// resource), and the metadata scratch allocates through the buffer's resource.
template<typename BufferT>
template<typename T, typename util::enable_if<BufferHasDeviceSingle<T>::value, int>::type, typename>
GridHandle<BufferT>::GridHandle(T&& buffer)
    : mBuffer(std::move(buffer))
{
    static_assert(util::is_same<T,BufferT>::value, "Expected U==BufferT");
    using ResourceT = typename BufferT::ResourceType;
    if (const GridData *d_data = reinterpret_cast<const GridData*>(mBuffer.data())) {
        const cudaStream_t stream = cuda::detail::retainedStreamOrDefault(mBuffer);
        const uint32_t count = cuda::detail::validGridChainHead(d_data, mBuffer.size_bytes(), stream);
        using ScratchT = cuda::Buffer<std::byte, ResourceT>;
        ScratchT scratch = cuda::detail::makeMetaScratch<ScratchT>(mBuffer, cuda::detail::chainScratchSize(count), stream);
        cuda::detail::parseGridChain(d_data, mBuffer.size_bytes(), count, scratch, mMetaData, stream);
    }
}// GridHandle(T&& buffer) for single-space device buffers

// Emit the dual-buffer move constructor from every CUDA translation unit, so
// host-only translation units that construct GridHandle<cuda::DeviceBuffer>
// (they see only the declaration) always find the symbol at link time. An
// unused private function is not enough: the optimizer may drop the
// complete-object constructor it instantiates.
template GridHandle<cuda::DeviceBuffer>::GridHandle(cuda::DeviceBuffer&&);

} // namespace nanovdb

#endif // NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED
