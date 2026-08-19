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

namespace nanovdb {

namespace cuda {

namespace detail {

static __global__ void cpyGridHandleMeta(const GridData *d_data, GridHandleMetaData *d_meta)
{
    nanovdb::cpyGridHandleMeta(d_data, d_meta);
}

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

/// @brief Walks the grid chain with per-header device-to-host reads, checking
///        that every header and every grid span lies inside the allocation, so
///        the device-side metadata walk that follows can never read out of
///        bounds from a truncated buffer or a forged header.
/// @return the validated grid count
inline uint32_t validGridChainCount(const GridData *d_head, uint64_t bytes, cudaStream_t stream)
{
    uint64_t offset = 0;
    uint32_t count = 0, expected = 0;
    GridData tmp;
    do {
        if (offset + sizeof(GridData) > bytes)
            throw std::runtime_error("GridHandle: grid chain exceeds the device buffer (truncated or corrupt grid data)");
        cudaCheck(cudaMemcpyAsync(&tmp, util::PtrAdd<GridData>(d_head, offset), sizeof(GridData), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        if (!tmp.isValid()) throw std::runtime_error("GridHandle was constructed with an invalid device buffer");
        if (count == 0) {
            expected = tmp.mGridCount;
            if (expected == 0) throw std::runtime_error("GridHandle: device buffer contains no grids");
        }
        if (tmp.mGridIndex != count || tmp.mGridCount != expected)
            throw std::runtime_error("GridHandle: inconsistent grid index/count in the device buffer's grid chain");
        if (tmp.mGridSize < sizeof(GridData) || tmp.mGridSize > bytes - offset)
            throw std::runtime_error("GridHandle: grid size field exceeds the device buffer (truncated or corrupt grid data)");
        offset += tmp.mGridSize;
    } while (++count < expected);
    return expected;
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
inline ScratchT makeMetaScratch(const BufferT& buf, uint32_t count, cudaStream_t stream)
{
    if constexpr (is_async_resource<typename BufferT::ResourceType>::value)
        return ScratchT(stream, buf.resource(), count, noInit);
    else { (void)stream; return ScratchT(buf.resource(), count, noInit); }
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
            const uint32_t count = cuda::detail::validGridChainCount(d_data, mBuffer.size(), cudaStream_t(0));
            // MallocResource: plain cudaMalloc, so this long-standing parse path
            // keeps working on devices without memory-pool support.
            cuda::Buffer<GridHandleMetaData, cuda::MallocResource> scratch(count, cuda::noInit);
            cuda::detail::cpyGridHandleMeta<<<1,1>>>(d_data, scratch.data());
            cudaCheckError();
            mMetaData.resize(count);
            cudaCheck(cudaMemcpy(mMetaData.data(), scratch.data(), count*sizeof(GridHandleMetaData), cudaMemcpyDeviceToHost));
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
    static_assert(sizeof(typename BufferT::ElementType) == 1,
                  "GridHandle requires byte-addressed single-space storage, e.g. cuda::Buffer<std::byte, R>");
    using ResourceT = typename BufferT::ResourceType;
    if (const GridData *d_data = reinterpret_cast<const GridData*>(mBuffer.data())) {
        const cudaStream_t stream = cuda::detail::retainedStreamOrDefault(mBuffer);
        const uint32_t count = cuda::detail::validGridChainCount(d_data, mBuffer.size_bytes(), stream);
        using ScratchT = cuda::Buffer<GridHandleMetaData, ResourceT>;
        ScratchT scratch = cuda::detail::makeMetaScratch<ScratchT>(mBuffer, count, stream);
        cuda::detail::cpyGridHandleMeta<<<1, 1, 0, stream>>>(d_data, scratch.data());
        cudaCheckError();
        mMetaData.resize(count);
        cudaCheck(cudaMemcpyAsync(mMetaData.data(), scratch.data(), count*sizeof(GridHandleMetaData), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }
}// GridHandle(T&& buffer) for single-space device buffers

// Dummy function that ensures instantiation of the move-constructor above when BufferT=cuda::DeviceBuffer
namespace {auto __dummy(){return GridHandle<cuda::DeviceBuffer>(std::move(cuda::DeviceBuffer()));}}

} // namespace nanovdb

#endif // NANOVDB_CUDA_GRIDHANDLE_CUH_HAS_BEEN_INCLUDED
