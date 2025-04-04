// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "TorchDeviceBuffer.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

namespace nanovdb {

// This is a dirty hack to clone the buffer accross devices
// guide is an empty buffer with the target device of the copy
// TODO: Pass in synchronous option
template <>
template <>
GridHandle<fvdb::detail::TorchDeviceBuffer>
GridHandle<fvdb::detail::TorchDeviceBuffer>::copy(
    const fvdb::detail::TorchDeviceBuffer &guide) const {
    if (mBuffer.isEmpty()) {
        fvdb::detail::TorchDeviceBuffer retbuf(0, nullptr, guide.device());
        return GridHandle<fvdb::detail::TorchDeviceBuffer>(
            std::move(retbuf)); // return an empty handle
    }

    const bool guideIsHost   = guide.device().is_cpu();
    const bool iAmHost       = mBuffer.device().is_cpu();
    const bool guideIsDevice = !guideIsHost;
    const bool iAmDevice     = !iAmHost;

    auto buffer = fvdb::detail::TorchDeviceBuffer::create(mBuffer.size(), &guide);

    if (iAmHost && guideIsHost) {
        std::memcpy(buffer.data(), mBuffer.data(),
                    mBuffer.size()); // deep copy of buffer in CPU RAM
        return GridHandle<fvdb::detail::TorchDeviceBuffer>(std::move(buffer));
    } else if (iAmHost && guideIsDevice) {
        const at::cuda::CUDAGuard device_guard{ guide.device() };
        cudaCheck(cudaMemcpy(buffer.deviceData(), mBuffer.data(), mBuffer.size(),
                             cudaMemcpyHostToDevice));
        return GridHandle<fvdb::detail::TorchDeviceBuffer>(std::move(buffer));
    } else if (iAmDevice && guideIsHost) {
        const at::cuda::CUDAGuard device_guard{ mBuffer.device() };
        cudaCheck(cudaMemcpy(buffer.data(), mBuffer.deviceData(), mBuffer.size(),
                             cudaMemcpyDeviceToHost));
        return GridHandle<fvdb::detail::TorchDeviceBuffer>(std::move(buffer));
    } else if (iAmDevice && guideIsDevice) {
        const at::cuda::CUDAGuard device_guard{ guide.device() };
        cudaCheck(cudaMemcpy(buffer.deviceData(), mBuffer.deviceData(), mBuffer.size(),
                             cudaMemcpyDeviceToDevice));
        return GridHandle<fvdb::detail::TorchDeviceBuffer>(std::move(buffer));
    } else {
        TORCH_CHECK(false, "All host/device combos exhausted. This should never happen.");
    }
}

} // namespace nanovdb

namespace fvdb {
namespace detail {

TorchDeviceBuffer::TorchDeviceBuffer(uint64_t size /* = 0*/, void *data /* = nullptr*/,
                                     const torch::Device &device /* = torch::kCPU*/)
    : mSize(0), mData(nullptr), mDevice(device) {
    if (size == 0)
        return;

    // Initalize on the host
    if (mDevice.is_cpu()) {
        if (data) {
            mData = reinterpret_cast<uint8_t *>(data);
        } else {
            mData = reinterpret_cast<uint8_t *>(malloc(size));
        }
        // Initalize on the device
    } else {
        if (data) {
            mData = reinterpret_cast<uint8_t *>(data);
        } else {
            c10::cuda::CUDAGuard deviceGuard(mDevice);
            mData = reinterpret_cast<uint8_t *>(c10::cuda::CUDACachingAllocator::raw_alloc(size));
            checkPtr(mData, "failed to allocate device data");
        }
    }

    mSize = size;
}

TorchDeviceBuffer::TorchDeviceBuffer(TorchDeviceBuffer &&other) noexcept
    : mSize(other.mSize), mData(other.mData), mDevice(other.mDevice) {
    other.mSize = 0;
    other.mData = nullptr;
}

TorchDeviceBuffer &
TorchDeviceBuffer::operator=(TorchDeviceBuffer &&other) noexcept {
    clear();
    mSize       = other.mSize;
    mData       = other.mData;
    mDevice     = other.mDevice;
    other.mSize = 0;
    other.mData = nullptr;
    return *this;
}

void
TorchDeviceBuffer::to(const torch::Device &device) {
    if (mDevice == device) {
        TORCH_CHECK(
            mData && (mSize > 0),
            "Source device matches destination device but existing data pointer is invalid");
        return;
    }

    TORCH_CHECK(!device.is_cuda() || device.has_index(), "CUDA devices must specify an index");

    uint8_t *data = nullptr;
    if (mDevice.is_cpu() && device.is_cuda()) { // CPU -> CUDA
        c10::cuda::CUDAGuard deviceGuard(device);
        data = reinterpret_cast<uint8_t *>(c10::cuda::CUDACachingAllocator::raw_alloc(mSize));
        cudaMemcpy(data, mData, mSize, cudaMemcpyHostToDevice);
        free(mData);
    } else if (mDevice.is_cuda() && device.is_cpu()) {
        data = reinterpret_cast<uint8_t *>(malloc(mSize));
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        cudaMemcpy(data, mData, mSize, cudaMemcpyDeviceToHost);
        c10::cuda::CUDACachingAllocator::raw_delete(mData);
    } else if (mDevice.is_cuda() && device.is_cuda()) {
        {
            c10::cuda::CUDAGuard deviceGuard(device);
            data = reinterpret_cast<uint8_t *>(c10::cuda::CUDACachingAllocator::raw_alloc(mSize));
            cudaMemcpy(data, mData, mSize, cudaMemcpyDeviceToDevice);
        }
        {
            c10::cuda::CUDAGuard deviceGuard(mDevice);
            c10::cuda::CUDACachingAllocator::raw_delete(mData);
        }

    } else {
        TORCH_CHECK(false, "Unsupported source and destination device combination");
    }

    mData   = data;
    mDevice = device;
}

void
TorchDeviceBuffer::clear() {
    if (mDevice.is_cuda() && mData) {
        c10::cuda::CUDAGuard deviceGuard(mDevice);
        c10::cuda::CUDACachingAllocator::raw_delete(mData);
    } else if (mDevice.is_cpu() && mData) {
        free(mData);
    }

    mData = nullptr;
    mSize = 0;
}

TorchDeviceBuffer
TorchDeviceBuffer::create(uint64_t size, const TorchDeviceBuffer *proto, int device, void *stream) {
    if (proto) {
        // This is a hack to pass in the device index when creating grids from nanovdb. Since we
        // can't pass arguments through nanovdb creation functions, we use a prototype grid to pass
        // in the device index.
        return TorchDeviceBuffer(size, nullptr, proto->device());
    } else if (device == cudaCpuDeviceId) {
        return TorchDeviceBuffer(size, nullptr, torch::kCPU);
    } else if (device > cudaCpuDeviceId) {
        return TorchDeviceBuffer(size, nullptr, torch::Device(torch::kCUDA, device));
    } else {
        TORCH_CHECK(false, "Invalid parameters specified for TorchDeviceBuffer::create");
    }
}

} // namespace detail
} // namespace fvdb
