// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaDeviceBuffer.h

    \author Ken Museth

    \date January 8, 2020

    \brief Implements a simple dual (host/device) CUDA buffer.

    \note This file has no device-only (kernel) function calls,
          which explains why it's a .h and not .cuh file.
*/

#ifndef NANOVDB_CUDA_DEVICE_BUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICE_BUFFER_H_HAS_BEEN_INCLUDED

#include "../HostBuffer.h" // for BufferTraits
#include "CudaUtils.h"// for cudaMalloc/cudaMallocManaged/cudaFree

namespace nanovdb {

// ----------------------------> CudaDeviceBuffer <--------------------------------------

/// @brief Simple memory buffer using un-managed pinned host memory when compiled with NVCC.
///        Obviously this class is making explicit used of CUDA so replace it with your own memory
///        allocator if you are not using CUDA.
/// @note  While CUDA's pinned host memory allows for asynchronous memory copy between host and device
///        it is significantly slower then cached (un-pinned) memory on the host.
class CudaDeviceBuffer
{

    uint64_t mSize; // total number of bytes managed by this buffer (assumed to be identical for host and device)
    uint8_t *mCpuData, *mGpuData; // raw pointers to the host and device buffers

public:
    /// @brief Static factory method that return an instance of this buffer
    /// @param size byte size of buffer to be initialized
    /// @param dummy this argument is currently ignored but required to match the API of the HostBuffer
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @param stream optional stream argument (defaults to stream NULL)
    /// @return An instance of this class using move semantics
    static CudaDeviceBuffer create(uint64_t size, const CudaDeviceBuffer* dummy = nullptr, bool host = true, void* stream = nullptr);

    /// @brief Constructor
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @param stream optional stream argument (defaults to stream NULL)
    CudaDeviceBuffer(uint64_t size = 0, bool host = true, void* stream = nullptr)
        : mSize(0)
        , mCpuData(nullptr)
        , mGpuData(nullptr)
    {
        if (size > 0) this->init(size, host, stream);
    }

    /// @brief Disallow copy-construction
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;

    /// @brief Move copy-constructor
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
        : mSize(other.mSize)
        , mCpuData(other.mCpuData)
        , mGpuData(other.mGpuData)
    {
        other.mSize = 0;
        other.mCpuData = nullptr;
        other.mGpuData = nullptr;
    }

    /// @brief Disallow copy assignment operation
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    /// @brief Move copy assignment operation
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept
    {
        this->clear();
        mSize = other.mSize;
        mCpuData = other.mCpuData;
        mGpuData = other.mGpuData;
        other.mSize = 0;
        other.mCpuData = nullptr;
        other.mGpuData = nullptr;
        return *this;
    }

    /// @brief Destructor frees memory on both the host and device
    ~CudaDeviceBuffer() { this->clear(); };

    /// @brief Initialize buffer
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is initialized only on the host/CPU, else on the device/GPU
    /// @note All existing buffers are first cleared
    /// @warning size is expected to be non-zero. Use clear() clear buffer!
    void init(uint64_t size, bool host = true, void* stream = nullptr);

    /// @brief Retuns a raw pointer to the host/CPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    uint8_t* data() const { return mCpuData; }

    /// @brief Retuns a raw pointer to the device/GPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    uint8_t* deviceData() const { return mGpuData; }

    /// @brief  Upload this buffer from the host to the device, i.e. CPU -> GPU.
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param sync if false the memory copy is asynchronous
    /// @note If the device/GPU buffer does not exist it is first allocated
    /// @warning Assumes that the host/CPU buffer already exists
    void deviceUpload(void* stream = nullptr, bool sync = true) const;

    /// @brief Upload this buffer from the device to the host, i.e. GPU -> CPU.
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param sync if false the memory copy is asynchronous
    /// @note If the host/CPU buffer does not exist it is first allocated
    /// @warning Assumes that the device/GPU buffer already exists
    void deviceDownload(void* stream = nullptr, bool sync = true) const;

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const { return mSize; }

    //@{
    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const { return mSize == 0; }
    bool isEmpty() const { return mSize == 0; }
    //@}

    /// @brief De-allocate all memory managed by this allocator and set all pointers to NULL
    void clear(void* stream = nullptr);

}; // CudaDeviceBuffer class

template<>
struct BufferTraits<CudaDeviceBuffer>
{
    static constexpr bool hasDeviceDual = true;
};

// --------------------------> Implementations below <------------------------------------

inline CudaDeviceBuffer CudaDeviceBuffer::create(uint64_t size, const CudaDeviceBuffer*, bool host, void* stream)
{
    return CudaDeviceBuffer(size, host, stream);
}

inline void CudaDeviceBuffer::init(uint64_t size, bool host, void* stream)
{
    if (mSize>0) this->clear(stream);
    NANOVDB_ASSERT(size > 0);
    if (host) {
        cudaCheck(cudaMallocHost((void**)&mCpuData, size)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
        checkPtr(mCpuData, "CudaDeviceBuffer::init: failed to allocate host buffer");
    } else {
        cudaCheck(CUDA_MALLOC((void**)&mGpuData, size, reinterpret_cast<cudaStream_t>(stream))); // un-managed memory on the device, always 32B aligned!
        checkPtr(mGpuData, "CudaDeviceBuffer::init: failed to allocate device buffer");
    }
    mSize = size;
} // CudaDeviceBuffer::init

inline void CudaDeviceBuffer::deviceUpload(void* stream, bool sync) const
{
    checkPtr(mCpuData, "uninitialized cpu data");
    if (mGpuData == nullptr) {
        cudaCheck(CUDA_MALLOC((void**)&mGpuData, mSize, reinterpret_cast<cudaStream_t>(stream))); // un-managed memory on the device, always 32B aligned!
    }
    checkPtr(mGpuData, "uninitialized gpu data");
    cudaCheck(cudaMemcpyAsync(mGpuData, mCpuData, mSize, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    if (sync) cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // CudaDeviceBuffer::gpuUpload

inline void CudaDeviceBuffer::deviceDownload(void* stream, bool sync) const
{
    checkPtr(mGpuData, "uninitialized gpu data");
    if (mCpuData == nullptr) {
        cudaCheck(cudaMallocHost((void**)&mCpuData, mSize)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
    }
    checkPtr(mCpuData, "uninitialized cpu data");
    cudaCheck(cudaMemcpyAsync(mCpuData, mGpuData, mSize, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    if (sync) cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // CudaDeviceBuffer::gpuDownload

inline void CudaDeviceBuffer::clear(void *stream)
{
    if (mGpuData) cudaCheck(CUDA_FREE(mGpuData, reinterpret_cast<cudaStream_t>(stream)));
    if (mCpuData) cudaCheck(cudaFreeHost(mCpuData));
    mCpuData = mGpuData = nullptr;
    mSize = 0;
} // CudaDeviceBuffer::clear

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_DEVICE_BUFFER_H_HAS_BEEN_INCLUDED
