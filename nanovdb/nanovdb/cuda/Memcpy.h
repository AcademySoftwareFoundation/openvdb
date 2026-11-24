// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file nanovdb/cuda/Memcpy.h
///
/// @brief Copies between any two locations: device memory on any device,
///        managed memory, and pinned or pageable host memory, independent of
///        which device is current.
///
/// @details By default, stream-ordered (pool) allocations are accessible only
///          from their owning device: cudaDeviceEnablePeerAccess does not
///          change that, only cudaMemPoolSetAccess does. A plain cudaMemcpy
///          whose source or destination is another device's pool memory
///          therefore fails with cudaErrorInvalidValue. These functions use a
///          peer copy between device memory on two different devices, and
///          otherwise make the device that owns the memory current for the
///          copy. A peer copy works without pool access, and the driver stages
///          it through the host when the devices have no peer-to-peer path.
///          Streams are used only while their own device is current, so that
///          compute-sanitizer reports no cross-context stream use.

#ifndef NANOVDB_CUDA_MEMCPY_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_MEMCPY_H_HAS_BEEN_INCLUDED

#include <cuda_runtime_api.h>
#include <nanovdb/util/cuda/Util.h> // for cudaCheck and util::cuda::currentDevice

#include <cstddef> // for size_t

namespace nanovdb {

namespace cuda {

namespace detail {

/// @brief Returns the device that owns @c ptr if it points to device memory,
///        and -1 for managed, pinned or pageable host memory.
/// @throw std::runtime_error if cudaPointerGetAttributes fails, e.g. for a
///        pointer that CUDA cannot classify
inline int owningDevice(const void* ptr)
{
    cudaPointerAttributes attr;
    cudaCheck(cudaPointerGetAttributes(&attr, ptr));
    return attr.type == cudaMemoryTypeDevice ? attr.device : -1;
}

/// @brief Makes @c device current for the lifetime of this object, and
///        restores the previously current device on destruction. A negative
///        @c device, or the current device, leaves the current device as is.
class ScopedDevice
{
    int mPrevious = -1;
public:
    explicit ScopedDevice(int device)
    {
        if (device < 0) return;
        const int current = util::cuda::currentDevice();
        if (device == current) return;
        cudaCheck(cudaSetDevice(device));
        mPrevious = current;
    }
    ~ScopedDevice() { if (mPrevious >= 0) cudaSetDevice(mPrevious); }
    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
};

/// @brief Returns the device @c stream belongs to: the current device for the
///        default streams, and -1 if it cannot be queried (before CUDA 12.8).
inline int streamDevice(cudaStream_t stream)
{
    if (stream == 0 || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        return util::cuda::currentDevice();
#if CUDART_VERSION >= 12080
    int device = -1;
    cudaCheck(cudaStreamGetDevice(stream, &device));
    return device;
#else
    return -1;
#endif
}

/// @brief Copies between host or managed memory and @c owner's device memory,
///        ordered on @c stream of another device, @c streamDev. A stream is
///        used only while its own device is current: the copy runs on a
///        temporary stream of @c owner, between events on @c stream.
inline void bridgedMemcpyAsync(void* dst, const void* src, size_t bytes, int owner, cudaStream_t stream, int streamDev)
{
    ScopedDevice guard(streamDev);
    cudaEvent_t before, after;
    cudaCheck(cudaEventCreateWithFlags(&before, cudaEventDisableTiming));
    cudaCheck(cudaEventRecord(before, stream));
    cudaCheck(cudaSetDevice(owner));
    cudaStream_t ownerStream;
    cudaCheck(cudaStreamCreateWithFlags(&ownerStream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamWaitEvent(ownerStream, before, 0));
    cudaCheck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, ownerStream));
    cudaCheck(cudaEventCreateWithFlags(&after, cudaEventDisableTiming));
    cudaCheck(cudaEventRecord(after, ownerStream));
    cudaCheck(cudaStreamDestroy(ownerStream));// released once its copy completes
    cudaCheck(cudaSetDevice(streamDev));
    cudaCheck(cudaStreamWaitEvent(stream, after, 0));
    cudaCheck(cudaEventDestroy(before));
    cudaCheck(cudaEventDestroy(after));
}

}// namespace detail

/// @brief Copies @c bytes from @c src to @c dst, ordered on @c stream, where
///        either pointer may be device memory on any device, managed memory,
///        or pinned or pageable host memory. The current device is unchanged
///        on return.
/// @param dst destination of the copy
/// @param src source of the copy
/// @param bytes number of bytes to copy; zero issues no copy
/// @param stream cuda stream the copy is ordered on; it may belong to any device
/// @note As with cudaMemcpyAsync, a pageable host source or destination makes
///       the copy synchronous with respect to the host.
inline void memcpyAsync(void* dst, const void* src, size_t bytes, cudaStream_t stream)
{
    if (bytes == 0u) return;
    const int dstDevice = detail::owningDevice(dst), srcDevice = detail::owningDevice(src);
    const int streamDev = detail::streamDevice(stream);
    if (dstDevice >= 0 && srcDevice >= 0 && dstDevice != srcDevice) {
        detail::ScopedDevice guard(streamDev);
        cudaCheck(cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, bytes, stream));
        return;
    }
    const int owner = dstDevice >= 0 ? dstDevice : srcDevice;// -1 if neither is device memory
    if (owner >= 0 && streamDev >= 0 && owner != streamDev) {
        detail::bridgedMemcpyAsync(dst, src, bytes, owner, stream, streamDev);
    } else {
        detail::ScopedDevice guard(owner >= 0 ? owner : streamDev);
        cudaCheck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
    }
}

/// @brief Synchronous version of memcpyAsync: copies @c bytes from @c src to
///        @c dst and returns when the copy has completed.
inline void memcpySync(void* dst, const void* src, size_t bytes)
{
    if (bytes == 0u) return;
    const int dstDevice = detail::owningDevice(dst), srcDevice = detail::owningDevice(src);
    if (dstDevice >= 0 && srcDevice >= 0 && dstDevice != srcDevice) {
        cudaCheck(cudaMemcpyPeer(dst, dstDevice, src, srcDevice, bytes));
    } else {
        detail::ScopedDevice guard(dstDevice >= 0 ? dstDevice : srcDevice);
        cudaCheck(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
    }
}

}// namespace cuda

}// namespace nanovdb

#endif// NANOVDB_CUDA_MEMCPY_H_HAS_BEEN_INCLUDED
