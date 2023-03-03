// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaDeviceBuffer.h

    \author Ken Museth

    \date January 8, 2020

    \brief Implements a simple CUDA allocator!

          CudaDeviceBuffer - a class for simple cuda buffer allocation and management
*/

#ifndef NANOVDB_CUDA_DEVICE_BUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICE_BUFFER_H_HAS_BEEN_INCLUDED

#include "HostBuffer.h" // for BufferTraits

#include <cuda_runtime_api.h> // for cudaMalloc/cudaMallocManaged/cudaFree

#if defined(DEBUG) || defined(_DEBUG)
    static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
    {
        if (code != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    static inline void ptrAssert(void* ptr, const char* msg, const char* file, int line, bool abort = true)
    {
        if (ptr == nullptr) {
            fprintf(stderr, "NULL pointer error: %s %s %d\n", msg, file, line);
            if (abort) exit(1);
        }
        if (uint64_t(ptr) % NANOVDB_DATA_ALIGNMENT) {
            fprintf(stderr, "Pointer misalignment error: %s %s %d\n", msg, file, line);
            if (abort) exit(1);
        }
    }
#else
    static inline void gpuAssert(cudaError_t, const char*, int, bool = true){}
    static inline void ptrAssert(void*, const char*, const char*, int, bool = true){}
#endif

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
#define cudaCheck(ans) \
    { \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

#define checkPtr(ptr, msg) \
    { \
        ptrAssert((ptr), (msg), __FILE__, __LINE__); \
    }

namespace nanovdb {

// ----------------------------> CudaDeviceBuffer <--------------------------------------

/// @brief Simple memory buffer using un-managed pinned host memory when compiled with NVCC.
///        Obviously this class is making explicit used of CUDA so replace it with your own memory
///        allocator if you are not using CUDA.
/// @note  While CUDA's pinned host memory allows for asynchronous memory copy between host and device
///        it is significantly slower then cached (un-pinned) memory on the host.
class CudaDeviceBuffer
{
    uint64_t mSize; // total number of bytes for the NanoVDB grid.
    uint8_t *mCpuData, *mGpuData; // raw buffer for the NanoVDB grid.

public:
    CudaDeviceBuffer(uint64_t size = 0)
        : mSize(0)
        , mCpuData(nullptr)
        , mGpuData(nullptr)
    {
        this->init(size);
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
        clear();
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

    void init(uint64_t size);

    // @brief Retuns a pointer to the raw memory buffer managed by this allocator.
    ///
    /// @warning Note that the pointer can be NULL is the allocator was not initialized!
    uint8_t* data() const { return mCpuData; }
    uint8_t* deviceData() const { return mGpuData; }

    /// @brief Copy grid from the CPU/host to the GPU/device. If @c sync is false the memory copy is asynchronous!
    ///
    /// @note This will allocate memory on the GPU/device if it is not already allocated
    void deviceUpload(void* stream = 0, bool sync = true) const;

    /// @brief Copy grid from the GPU/device to the CPU/host. If @c sync is false the memory copy is asynchronous!
    void deviceDownload(void* stream = 0, bool sync = true) const;

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const { return mSize; }

    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const { return mSize == 0; }

    /// @brief De-allocate all memory managed by this allocator and set all pointer to NULL
    void clear();

    static CudaDeviceBuffer create(uint64_t size, const CudaDeviceBuffer* context = nullptr);

}; // CudaDeviceBuffer class

template<>
struct BufferTraits<CudaDeviceBuffer>
{
    static const bool hasDeviceDual = true;
};

// --------------------------> Implementations below <------------------------------------

inline CudaDeviceBuffer CudaDeviceBuffer::create(uint64_t size, const CudaDeviceBuffer*)
{
    return CudaDeviceBuffer(size);
}

inline void CudaDeviceBuffer::init(uint64_t size)
{
    if (size == mSize)
        return;
    if (mSize > 0)
        this->clear();
    if (size == 0)
        return;
    mSize = size;
    cudaCheck(cudaMallocHost((void**)&mCpuData, size)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
    checkPtr(mCpuData, "failed to allocate host data");
} // CudaDeviceBuffer::init

inline void CudaDeviceBuffer::deviceUpload(void* stream, bool sync) const
{
    checkPtr(mCpuData, "uninitialized cpu data");
    if (mGpuData == nullptr)
        cudaCheck(cudaMalloc((void**)&mGpuData, mSize)); // un-managed memory on the device, always 32B aligned!
    checkPtr(mGpuData, "uninitialized gpu data");
    cudaCheck(cudaMemcpyAsync(mGpuData, mCpuData, mSize, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    if (sync)
        cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // CudaDeviceBuffer::gpuUpload

inline void CudaDeviceBuffer::deviceDownload(void* stream, bool sync) const
{
    checkPtr(mCpuData, "uninitialized cpu data");
    checkPtr(mGpuData, "uninitialized gpu data");
    cudaCheck(cudaMemcpyAsync(mCpuData, mGpuData, mSize, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    if (sync)
        cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
} // CudaDeviceBuffer::gpuDownload

inline void CudaDeviceBuffer::clear()
{
    if (mGpuData)
        cudaCheck(cudaFree(mGpuData));
    if (mCpuData)
        cudaCheck(cudaFreeHost(mCpuData));
    mCpuData = mGpuData = nullptr;
    mSize = 0;
} // CudaDeviceBuffer::clear

} // namespace nanovdb

#endif // end of NANOVDB_CUDA_DEVICE_BUFFER_H_HAS_BEEN_INCLUDED
