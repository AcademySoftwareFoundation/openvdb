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

    \warning This header uses the CUDA runtime, so only include it from
             .cu files (or other CUDA headers).
*/

#ifndef NANOVDB_CUDA_HANDLESTORAGE_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_HANDLESTORAGE_H_HAS_BEEN_INCLUDED

#include <nanovdb/HostBuffer.h>// for the BufferTraits detectors
#include <nanovdb/cuda/Buffer.h>// for noInit and the resource concepts

#include <stdexcept>// for std::runtime_error
#include <type_traits>// for std::is_default_constructible

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

}// namespace detail

}// namespace cuda

}// namespace nanovdb

#endif // NANOVDB_CUDA_HANDLESTORAGE_H_HAS_BEEN_INCLUDED
