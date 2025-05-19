// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_RAIIRAWDEVICEBUFFER_H
#define FVDB_DETAIL_UTILS_CUDA_RAIIRAWDEVICEBUFFER_H

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

namespace fvdb {

/// @brief A wrapper around a raw device buffer that automatically frees the buffer when it goes out
/// of scope
/// @tparam T The type of data this buffer points to
template <typename T> struct RAIIRawDeviceBuffer {
    T           *devicePtr  = nullptr;
    size_t       bufferSize = 0;
    cudaStream_t stream     = 0;

    RAIIRawDeviceBuffer()                            = default;
    RAIIRawDeviceBuffer(const RAIIRawDeviceBuffer &) = delete;

    RAIIRawDeviceBuffer(RAIIRawDeviceBuffer &&other) {
        devicePtr        = other.devicePtr;
        bufferSize       = other.bufferSize;
        stream           = other.stream;
        other.devicePtr  = nullptr;
        other.bufferSize = 0;
        other.stream     = 0;
    }

    /// @brief Create a buffer containing size elements of type T on the specified device
    /// @param size The number of elements to allocate
    /// @param device The device to allocate the buffer on
    RAIIRawDeviceBuffer(size_t size, c10::Device device) {
        TORCH_CHECK(device.has_index(), "Device must specify an index");
        c10::cuda::CUDAGuard deviceGuard(device);
        stream     = at::cuda::getCurrentCUDAStream(device.index()).stream();
        bufferSize = size * sizeof(T);
        devicePtr  = reinterpret_cast<T *>(
            c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(bufferSize, stream));
        // devicePtr = reinterpret_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(bufferSize));
    }

    ~RAIIRawDeviceBuffer() {
        if (devicePtr == nullptr) {
            return;
        }
        c10::cuda::CUDACachingAllocator::raw_delete(devicePtr);
        devicePtr = nullptr;
    }

    /// @brief Copy data from the host to the device
    /// @param hostData The host data to copy (assumes it points to a buffer of size bufferSize)
    void
    setData(const T *hostData, bool blocking) {
        cudaMemcpyAsync((void *)devicePtr, (const void *)hostData, bufferSize,
                        cudaMemcpyHostToDevice, stream);
        if (blocking) {
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        // cudaMemcpy((void*) devicePtr, (const void*) hostData, bufferSize,
        // cudaMemcpyHostToDevice);
    }
};

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_RAIIRAWDEVICEBUFFER_H
