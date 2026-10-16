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
#include <nanovdb/cuda/HandleStorage.h> // cuda::copyTo and the storage helpers live there (host-includable)
#include <nanovdb/cuda/DeviceBuffer.h>// required for instantiation of move c-tor of GridHandle
#include <nanovdb/tools/cuda/GridChecksum.cuh>// for cuda::updateChecksum
#include <nanovdb/GridHandle.h>

#include <limits>// for the grid-count bound in cuda::mergeGridHandles
#include <memory>// for the host copy of the dirty flags
#include <stdexcept>// for std::runtime_error
#include <string>// for the grid index in chain-validation error messages
#include <type_traits>// for std::is_default_constructible in cuda::mergeGridHandles
#include <utility>// for std::move
#include <vector>// for the per-grid destinations and metadata

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
        // Read through a copy: a forged size in the preceding header can place this one at an
        // unaligned offset, where dereferencing a GridData pointer would fault.
        alignas(GridData) unsigned char raw[sizeof(GridData)];
        memcpy(raw, util::PtrAdd<const void>(d_head, offset), sizeof(GridData));
        const GridData *data = reinterpret_cast<const GridData*>(raw);
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

/// @brief Orders @c stream after all work already issued on @c buf's
///        retained stream, using @c event; a no-op for a buffer that retains
///        no stream or retains @c stream itself.
template<typename BufferT>
inline void orderAfterRetainedStream(const BufferT& buf, cudaStream_t stream, cudaEvent_t event)
{
    if constexpr (BufferHasStream<BufferT>::value) {
        if (buf.stream() == stream) return;
        cudaCheck(cudaEventRecord(event, buf.stream()));
        cudaCheck(cudaStreamWaitEvent(stream, event, 0));
    } else {
        (void)buf; (void)stream; (void)event;
    }
}

/// @brief Rewrites the grid index and count of the @c dst[i]'th grid header
///        and refreshes the checksum of every header that changed. One
///        readback and one synchronization of @c stream cover all grids, so
///        every copy issued on @c stream before the call has completed when
///        it returns.
/// @param dirty device scratch of at least dst.size() flags
inline void rewriteGridCounts(const std::vector<GridData*>& dst, const std::vector<uint32_t>& gridIndex,
                              uint32_t gridCount, bool* dirty, cudaStream_t stream)
{
    const size_t n = dst.size();
    if (gridIndex.size() != n)
        throw std::invalid_argument("rewriteGridCounts: expected one grid index per destination");
    for (size_t i = 0; i < n; ++i) {
        if (gridIndex[i] >= gridCount)
            throw std::invalid_argument("rewriteGridCounts: grid index out of range of the grid count");
    }
    for (size_t i = 0; i < n; ++i) {
        updateGridCount<<<1, 1, 0, stream>>>(dst[i], gridIndex[i], gridCount, dirty + i);
        cudaCheckError();
    }
    std::unique_ptr<bool[]> hostDirty(new bool[n]);
    cudaCheck(cudaMemcpyAsync(hostDirty.get(), dirty, n * sizeof(bool), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < n; ++i) {
        if (hostDirty[i]) tools::cuda::updateChecksum(dst[i], CheckMode::Partial, stream);
    }
}

}// namespace detail

namespace detail {

/// @brief Implementation behind the deprecated dual-space splitGridHandles,
///        kept warning-free for in-tree callers until the dual buffers are
///        removed.
template<typename BufferT, template <class, class...> class VectorT = std::vector>
inline VectorT<GridHandle<BufferT>>
splitDualGridHandles(const GridHandle<BufferT> &handle, const BufferT* other, cudaStream_t stream)
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
}// cuda::detail::splitDualGridHandles

/// @brief Implementation behind the deprecated dual-space mergeGridHandles,
///        kept warning-free for in-tree callers until the dual buffers are
///        removed.
template<typename BufferT, template <class, class...> class VectorT>
inline GridHandle<BufferT>
mergeDualGridHandles(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other, cudaStream_t stream)
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
}// cuda::detail::mergeDualGridHandles

}// namespace detail

/// @deprecated The dual-space buffers are removed in a future release: hold
///             grids in GridHandle<cuda::Buffer<std::byte>> and use the
///             single-space overload splitGridHandles(handle, stream, proto).
template<typename BufferT, template <class, class...> class VectorT = std::vector>
[[deprecated("dual-space buffers are being removed: use GridHandle<cuda::Buffer<std::byte>> with splitGridHandles(handle, stream, proto)")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
splitGridHandles(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    return detail::splitDualGridHandles<BufferT, VectorT>(handle, other, stream);
}

/// @deprecated The dual-space buffers are removed in a future release: hold
///             grids in GridHandle<cuda::Buffer<std::byte>> and use the
///             single-space overload mergeGridHandles(handles, stream, proto).
template<typename BufferT, template <class, class...> class VectorT>
[[deprecated("dual-space buffers are being removed: use GridHandle<cuda::Buffer<std::byte>> with mergeGridHandles(handles, stream, proto)")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
mergeGridHandles(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    return detail::mergeDualGridHandles<BufferT, VectorT>(handles, other, stream);
}

/// @brief Splits a single-space device handle into one handle per grid.
/// @param handle source handle containing one or more grids
/// @param stream stream every copy and allocation is ordered on; it is first
///        ordered after the source buffer's retained stream, if any
/// @warning A source buffer WITHOUT a retained stream (a synchronous
///          resource) is not ordered by this call: work that produced its
///          contents on another stream must be ordered before @a stream by
///          the caller, e.g. with cudaStreamWaitEvent or a synchronization.
/// @param proto optional buffer whose resource allocates each output buffer;
///        without it the resource is default-constructed
/// @return one handle per grid, each with grid index 0 and grid count 1
/// @details The copies have completed when this returns, so the caller may
///          release @a handle immediately. Each output handle adopts the
///          metadata of its grid in @a handle, so no device chain is parsed.
///          Outputs over a stream-ordered resource retain @a stream; any
///          remaining checksum updates are ordered on it. Outputs over a
///          synchronous resource are complete on return.
template<typename BufferT, template <class, class...> class VectorT = std::vector>
inline typename util::enable_if<BufferHasDeviceSingle<BufferT>::value, VectorT<GridHandle<BufferT>>>::type
splitGridHandles(const GridHandle<BufferT> &handle, cudaStream_t stream, const BufferT* proto = nullptr)
{
    VectorT<GridHandle<BufferT>> handles;
    const uint32_t gridCount = handle.gridCount();
    if (gridCount == 0u) return handles;
    const auto &meta = detail::HandleFactory::meta(handle);

    cudaEvent_t event;
    cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    detail::orderAfterRetainedStream(handle.buffer(), stream, event);
    cudaCheck(cudaEventDestroy(event));

    const int device = util::cuda::currentDevice();
    std::vector<BufferT> buffers;
    std::vector<GridData*> dst;
    buffers.reserve(gridCount);
    dst.reserve(gridCount);
    const void *src = handle.deviceData();
    for (uint32_t n = 0; n < gridCount; ++n) {
        buffers.push_back(detail::createDeviceStorage<BufferT>(meta[n].size, proto, device, stream));
        dst.push_back(reinterpret_cast<GridData*>(detail::deviceStorageData(buffers.back())));
        cudaCheck(cudaMemcpyAsync(dst.back(), util::PtrAdd(src, meta[n].offset), meta[n].size, cudaMemcpyDeviceToDevice, stream));
    }
    using DirtyT = Buffer<bool, typename BufferT::ResourceType>;
    {
        DirtyT dirty = detail::makeMetaScratch<DirtyT>(buffers.front(), gridCount, stream);
        detail::rewriteGridCounts(dst, std::vector<uint32_t>(gridCount, 0u), 1u, dirty.data(), stream);
    }
    if constexpr (!is_async_resource<typename BufferT::ResourceType>::value) cudaCheck(cudaStreamSynchronize(stream));

    handles.reserve(gridCount);
    for (uint32_t n = 0; n < gridCount; ++n) {
        std::vector<GridHandleMetaData> gridMeta{GridHandleMetaData{0u, meta[n].size, meta[n].gridType}};
        handles.push_back(detail::HandleFactory::make(std::move(buffers[n]), std::move(gridMeta)));
    }
    return handles;
}// cuda::splitGridHandles

/// @brief Merges single-space device handles into one handle containing all
///        of their grids, in order.
/// @param handles source handles, each containing one or more grids
/// @param stream stream every copy and the allocation are ordered on; it is
///        first ordered after each source buffer's retained stream, if any
/// @warning A source buffer WITHOUT a retained stream (a synchronous
///          resource) is not ordered by this call: work that produced its
///          contents on another stream must be ordered before @a stream by
///          the caller, e.g. with cudaStreamWaitEvent or a synchronization.
/// @param proto optional buffer whose resource allocates the output buffer;
///        without it the resource is default-constructed
/// @return a handle whose grids carry their merged index and total count
/// @details The copies have completed when this returns, so the caller may
///          release @a handles immediately. The output adopts the metadata
///          of its source grids at their merged offsets, so no device chain
///          is parsed. An output over a stream-ordered resource retains
///          @a stream; any remaining checksum updates are ordered on it. An
///          output over a synchronous resource is complete on return.
template<typename BufferT, template <class, class...> class VectorT>
inline typename util::enable_if<BufferHasDeviceSingle<BufferT>::value, GridHandle<BufferT>>::type
mergeGridHandles(const VectorT<GridHandle<BufferT>> &handles, cudaStream_t stream, const BufferT* proto = nullptr)
{
    std::vector<GridHandleMetaData> meta;
    uint64_t size = 0u;
    for (auto &h : handles) {
        for (auto &m : detail::HandleFactory::meta(h)) {
            meta.push_back(GridHandleMetaData{size, m.size, m.gridType});
            size += m.size;
        }
    }
    if (size == 0u) {
        if constexpr (std::is_default_constructible<BufferT>::value) {
            return GridHandle<BufferT>();
        } else {
            throw std::runtime_error("cuda::mergeGridHandles: merging empty handles requires a "
                                     "default-constructible buffer type");
        }
    }

    cudaEvent_t event;
    cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    for (auto &h : handles) detail::orderAfterRetainedStream(h.buffer(), stream, event);
    cudaCheck(cudaEventDestroy(event));

    if (meta.size() > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("cuda::mergeGridHandles: the merged handle would exceed the maximum grid count");
    const uint32_t gridCount = static_cast<uint32_t>(meta.size());
    BufferT buffer = detail::createDeviceStorage<BufferT>(size, proto, util::cuda::currentDevice(), stream);
    std::vector<GridData*> dst;
    std::vector<uint32_t> gridIndex;
    dst.reserve(gridCount);
    gridIndex.reserve(gridCount);
    uint32_t counter = 0u;
    for (auto &h : handles) {
        const void *src = h.deviceData();
        for (auto &m : detail::HandleFactory::meta(h)) {
            dst.push_back(util::PtrAdd<GridData>(detail::deviceStorageData(buffer), meta[counter].offset));
            gridIndex.push_back(counter++);
            cudaCheck(cudaMemcpyAsync(dst.back(), util::PtrAdd(src, m.offset), m.size, cudaMemcpyDeviceToDevice, stream));
        }
    }
    using DirtyT = Buffer<bool, typename BufferT::ResourceType>;
    {
        DirtyT dirty = detail::makeMetaScratch<DirtyT>(buffer, gridCount, stream);
        detail::rewriteGridCounts(dst, gridIndex, gridCount, dirty.data(), stream);
    }
    if constexpr (!is_async_resource<typename BufferT::ResourceType>::value) cudaCheck(cudaStreamSynchronize(stream));
    return detail::HandleFactory::make(std::move(buffer), std::move(meta));
}// cuda::mergeGridHandles

}// namespace cuda

template<typename BufferT, template <class, class...> class VectorT = std::vector>
[[deprecated("dual-space buffers are being removed: use GridHandle<cuda::Buffer<std::byte>> with nanovdb::cuda::splitGridHandles(handle, stream, proto)")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
splitDeviceGrids(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cuda::detail::splitDualGridHandles<BufferT, VectorT>(handle, other, stream); }

template<typename BufferT, template <class, class...> class VectorT>
[[deprecated("dual-space buffers are being removed: use GridHandle<cuda::Buffer<std::byte>> with nanovdb::cuda::mergeGridHandles(handles, stream, proto)")]]
inline typename util::enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
mergeDeviceGrids(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cuda::detail::mergeDualGridHandles<BufferT, VectorT>(handles, other, stream); }

template<typename BufferT>
template<typename T, typename util::enable_if<BufferTraits<T>::hasDeviceDual, int>::type>
GridHandle<BufferT>::GridHandle(T&& buffer)
    : mBuffer(std::move(buffer))
{
    static_assert(util::is_same<T,BufferT>::value, "Expected U==BufferT");
    if (auto *data = reinterpret_cast<const GridData*>(mBuffer.data())) {
        detail::parseHostGridChain(data, mBuffer.size(), mMetaData);
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
// host-only translation units that construct GridHandle<cuda::DualDeviceBuffer>
// (they see only the declaration) always find the symbol at link time. An
// unused private function is not enough: the optimizer may drop the
// complete-object constructor it instantiates.
template GridHandle<cuda::DualDeviceBuffer>::GridHandle(cuda::DualDeviceBuffer&&);

} // namespace nanovdb

#endif // NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED
