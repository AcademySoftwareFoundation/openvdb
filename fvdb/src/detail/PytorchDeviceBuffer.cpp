#include "PytorchDeviceBuffer.h"

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include <cuda_runtime_api.h> // for cudaMalloc/cudaMallocManaged/cudaFree
#include <c10/cuda/CUDACachingAllocator.h>

namespace nanovdb {

// This is a dirty hack to clone the buffer accross devices
// guide is an empty buffer with the target device of the copy
// TODO: Pass in synchronous option
template <>
template <>
GridHandle<fvdb::detail::PytorchDeviceBuffer> GridHandle<fvdb::detail::PytorchDeviceBuffer>::copy(const fvdb::detail::PytorchDeviceBuffer& guide) const {
    if (mBuffer.isEmpty()) {
        fvdb::detail::PytorchDeviceBuffer retbuf(0, nullptr);
        retbuf.setDevice(guide.device(), false);
        return GridHandle<fvdb::detail::PytorchDeviceBuffer>(std::move(retbuf));// return an empty handle
    }

    const bool guideIsHost = guide.device().is_cpu();
    const bool iAmHost = mBuffer.device().is_cpu();
    const bool guideIsDevice = !guideIsHost;
    const bool iAmDevice = !iAmHost;

    auto buffer = fvdb::detail::PytorchDeviceBuffer::create(mBuffer.size(), &guide, guideIsHost);

    if (iAmHost && guideIsHost) {
        std::memcpy(buffer.data(), mBuffer.data(), mBuffer.size()); // deep copy of buffer in CPU RAM
    } else if (iAmHost && guideIsDevice) {
        at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(guide.device().index());
        cudaCheck(cudaMemcpyAsync(buffer.deviceData(), mBuffer.data(), mBuffer.size(), cudaMemcpyHostToDevice, defaultStream.stream()));
        cudaCheck(cudaStreamSynchronize(defaultStream.stream()));
    } else if (iAmDevice && guideIsHost) {
        at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(mBuffer.device().index());
        cudaCheck(cudaMemcpyAsync(buffer.data(), mBuffer.deviceData(), mBuffer.size(), cudaMemcpyDeviceToHost, defaultStream.stream()));
        cudaCheck(cudaStreamSynchronize(defaultStream.stream()));
    } else if (iAmDevice && guideIsDevice) {
        if (mBuffer.device() == guide.device()) {
            at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(mBuffer.device().index());
            cudaCheck(cudaMemcpyAsync(buffer.deviceData(), mBuffer.deviceData(), mBuffer.size(), cudaMemcpyDeviceToDevice, defaultStream.stream()));
            cudaCheck(cudaStreamSynchronize(defaultStream.stream()));
        } else {
            std::unique_ptr<uint8_t[]> buf(new uint8_t[mBuffer.size()]);
            at::cuda::CUDAStream mBufferStream = at::cuda::getCurrentCUDAStream(mBuffer.device().index());
            at::cuda::CUDAStream outBufferStream = at::cuda::getCurrentCUDAStream(buffer.device().index());
            cudaCheck(cudaMemcpyAsync(buf.get(), mBuffer.deviceData(), mBuffer.size(), cudaMemcpyDeviceToHost, mBufferStream.stream()));
            cudaCheck(cudaStreamSynchronize(mBufferStream.stream()));
            cudaCheck(cudaMemcpyAsync(buffer.deviceData(), buf.get(), mBuffer.size(), cudaMemcpyHostToDevice, outBufferStream.stream()));
            cudaCheck(cudaStreamSynchronize(outBufferStream.stream()));
        }
    }
    return GridHandle<fvdb::detail::PytorchDeviceBuffer>(std::move(buffer));
}

}

namespace fvdb {
namespace detail {

PytorchDeviceBuffer::PytorchDeviceBuffer(uint64_t size /* = 0*/, void* data /* = nullptr*/, bool host /* = true*/, int deviceIndex /* = -1*/)
    : mSize(0)
    , mCpuData(nullptr)
    , mGpuData(nullptr)
    , mDevice(host ? torch::kCPU : torch::kCUDA, deviceIndex) {

    TORCH_CHECK(host || (!host && deviceIndex >= 0), "You must set deviceIndex when setting host to false");
    this->init(size, data, host);
}


PytorchDeviceBuffer::PytorchDeviceBuffer(PytorchDeviceBuffer&& other) noexcept
    : mSize(other.mSize)
    , mCpuData(other.mCpuData)
    , mGpuData(other.mGpuData)
    , mDevice(other.mDevice) {
    other.mSize = 0;
    other.mCpuData = nullptr;
    other.mGpuData = nullptr;
}


PytorchDeviceBuffer& PytorchDeviceBuffer::operator=(PytorchDeviceBuffer&& other) noexcept {
    clear();
    mSize = other.mSize;
    mCpuData = other.mCpuData;
    mGpuData = other.mGpuData;
    mDevice = other.mDevice;
    other.mSize = 0;
    other.mCpuData = nullptr;
    other.mGpuData = nullptr;
    return *this;
}


void PytorchDeviceBuffer::setDevice(const torch::Device& toDevice, bool blocking) {
    // Same device, no-op
    if (toDevice == mDevice) {
        return;
    }

    // If this is an empty buffer, just set the device and return
    if (mCpuData == nullptr && mGpuData == nullptr) {
        if (toDevice.is_cuda()) {
            TORCH_CHECK(toDevice.has_index(), "Invalid CUDA device must specify device index");
        }
        mDevice = toDevice;
        return;
    }

    // Otherwise, we might need to shuffle data around, outsource that to toCuda/toCpu
    if (toDevice.is_cpu()) {
        this->toCpu(blocking);
    } else if (toDevice.is_cuda()) {
        this->toCuda(toDevice, blocking);
    } else {
        TORCH_CHECK(false, "Only CPU and CUDA devices are supported")
    }

}

void PytorchDeviceBuffer::toCpu(bool blocking) {

    // Empty buffer, set the device and return
    if (mGpuData == nullptr && mCpuData == nullptr) {
        mDevice = torch::kCPU;
        return;
    }

    // If this is a cuda device, copy the data to the CPU
    if (mDevice.is_cuda()) {
        at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(mDevice.index());
        copyDeviceToHostAndFreeDevice(defaultStream.stream(), blocking);
    }

    if (mGpuData != nullptr) {
        c10::cuda::CUDACachingAllocator::raw_delete(mGpuData);
        mGpuData = nullptr;
    }

    // Set the device
    mDevice = torch::kCPU;
}

void PytorchDeviceBuffer::toCuda(torch::Device toDevice, bool blocking) {
    TORCH_CHECK(toDevice.is_cuda(), "Invalid device must be a CUDA device");
    TORCH_CHECK(toDevice.has_index(), "Invalid device must specify device index");

    // Empty buffer, set the device and return
    if (mGpuData == nullptr && mCpuData == nullptr) {
        mDevice = toDevice;
        return;
    }

    // Same device, no-op
    if (toDevice == mDevice) {
        return;
    }

    if (mDevice.is_cuda() && toDevice != mDevice) { // CUDA -> CUDA accross different devices
        std::unique_ptr<uint8_t[]> buf(new uint8_t[mSize]);
        at::cuda::CUDAStream currentStream = at::cuda::getCurrentCUDAStream(mDevice.index());
        at::cuda::CUDAStream toStream = at::cuda::getCurrentCUDAStream(toDevice.index());
        cudaCheck(cudaMemcpyAsync(buf.get(), mGpuData, mSize, cudaMemcpyDeviceToHost, currentStream.stream()));
        cudaCheck(cudaStreamSynchronize(currentStream.stream()));
        c10::cuda::CUDACachingAllocator::raw_delete(mGpuData);
        mGpuData = reinterpret_cast<uint8_t*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(mSize, toStream.stream()));
        cudaCheck(cudaMemcpyAsync(mGpuData, buf.get(), mSize, cudaMemcpyHostToDevice, toStream.stream()));
        mDevice = toDevice;
    } else if (mDevice.is_cpu()) {  // CPU -> CUDA
        TORCH_CHECK(toDevice.has_index(), "Invalid device must specify device index");
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(toDevice.index());
        copyHostToDeviceAndFreeHost((void*) stream.stream(), blocking);
        mDevice = toDevice;
    } else {
        TORCH_CHECK(false, "This should never happen. File a bug.")
    }
}


void PytorchDeviceBuffer::init(uint64_t size, void* data /* = nullptr */, bool host /* = true */) {
    TORCH_CHECK((host && mDevice.is_cpu()) || (!host && mDevice.is_cuda()), "Invalid device for host argument to PytorchDeviceBuffer::init");
    if (size == mSize) { // If we already initialized the buffer with the same size, just return
        return;
    }
    if (mSize >= 0) { // If we're initializing to a different size, need to free the old buffer
        this->clear();
    }
    if (size == 0) { // If we're initializing to a size of 0, just return
        return;
    }

    // Set the new size
    mSize = size;

    // Initalize on the host
    if (host) {
        if (data) {
            mCpuData = (uint8_t*) data;
        } else {
            // cudaCheck(cudaMallocHost((void**)&mCpuData, size)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
            mCpuData = (uint8_t*) malloc(size);
        }
        // checkPtr(mCpuData, "failed to allocate host data");
    // Initalize on the device
    } else {
        if (data) {
            mGpuData = (uint8_t*) data;
        } else {
            at::cuda::CUDAStream defaultStream = at::cuda::getCurrentCUDAStream(mDevice.index());
            mGpuData = reinterpret_cast<uint8_t*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, defaultStream.stream()));
            checkPtr(mGpuData, "failed to allocate device data");
        }
    }
}

void PytorchDeviceBuffer::clear() {
    if (mGpuData) {
        c10::cuda::CUDACachingAllocator::raw_delete(mGpuData);
    }
    if (mCpuData) {
        // cudaCheck(cudaFreeHost(mCpuData));
        free(mCpuData);
    }
    mCpuData = mGpuData = nullptr;
    mSize = 0;
}

PytorchDeviceBuffer PytorchDeviceBuffer::create(uint64_t size, const PytorchDeviceBuffer* proto, bool host, void* stream) {
    // This is a hack to pass in the device index when creating grids from nanovdb. Since we can't pass arguments
    // through nanovdb creation functions, we use a prototype grid to pass in the device index.
    int deviceId = -1;
    if (proto != nullptr) {
        TORCH_CHECK((host && proto->device().is_cpu()) || (!host && proto->device().is_cuda()), "Invalid guide buffer device for host argument to PytorchDeviceBuffer::create");
        deviceId = proto->mDevice.index();
    }
    return PytorchDeviceBuffer(size, nullptr, host, host ? -1 : deviceId);
}

void PytorchDeviceBuffer::copyDeviceToHostAndFreeDevice(void* streamPtr /* = 0*/, bool blocking /* = true*/) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamPtr);

    TORCH_CHECK(mGpuData, "uninitialized cpu data, this should never happen");
    if (mCpuData == nullptr) { // Allocate CPU data if we upload to the device
        // cudaCheck(cudaMallocHost((void**)&mCpuData, mSize)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
        mCpuData = (uint8_t*) malloc(mSize);
    }
    // Copy to the host buffer
    cudaCheck(cudaMemcpyAsync(mCpuData, mGpuData, mSize, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    if (blocking) {
        cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
    }

    c10::cuda::CUDACachingAllocator::raw_delete(mGpuData);
}

void PytorchDeviceBuffer::copyHostToDeviceAndFreeHost(void* streamPtr /* = 0*/, bool blocking /* = true*/) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamPtr);

    TORCH_CHECK(mCpuData, "uninitialized cpu data, this should never happen");
    if (mGpuData == nullptr) {  // Allocate a new CUDA buffer
        mGpuData = reinterpret_cast<uint8_t*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(mSize, stream));
    }
    // Copy the data to the CUDA buffer
    cudaCheck(cudaMemcpyAsync(mGpuData, mCpuData, mSize, cudaMemcpyHostToDevice, stream));
    if (blocking) {
        cudaCheck(cudaStreamSynchronize(stream));
    }

    // cudaCheck(cudaFreeHost(mCpuData));
    free(mCpuData);
    mCpuData = nullptr;
}

} // namespace detail
} // namespace fvdb
