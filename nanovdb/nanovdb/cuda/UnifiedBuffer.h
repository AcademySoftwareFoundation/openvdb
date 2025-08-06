// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file UnifiedBuffer.h

    \author Ken Museth

    \date October 15, 2024

    \brief nanovdb::cuda::UnifiedBuffer that uses unified memory management

    \note This file has no device-only kernel functions,
          which explains why it's a .h and not .cuh file.
*/

#ifndef NANOVDB_CUDA_UNIFIEDBUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_UNIFIEDBUFFER_H_HAS_BEEN_INCLUDED

#include <cuda.h>
#include <memory>// for std::shared_ptr
#include <nanovdb/HostBuffer.h>// for BufferTraits
#include <nanovdb/util/cuda/Util.h>// for cudaCheck

namespace nanovdb {// ================================================================

namespace cuda {// ===================================================================

/// @brief  buffer, used for instance by the GridHandle, to allocate unified memory that
///         can be resized and shared between multiple devices and the host.
class UnifiedBuffer
{
    void *mPtr;
    size_t mSize, mCapacity;
public:

    using PtrT = std::shared_ptr<UnifiedBuffer>;

    /// @brief Default constructor of an empty buffer
    UnifiedBuffer() : mPtr(nullptr), mSize(0), mCapacity(0){}

    /// @brief Constructor that specifies both the size and capacity
    /// @param size size of the buffer in bytes, indication what is actually used
    /// @param capacity number of bytes in the virtual page table, i.e max size for growing
    /// @note Capacity can be over-estimated to allow for future growth. Memory is not allocated
    ///       with this constructor, only a page table. Allocation happens on usage or when calling prefetch
    UnifiedBuffer(size_t size, size_t capacity) : mPtr(nullptr), mSize(size), mCapacity(capacity)
    {
        assert(mSize <= mCapacity);
        cudaCheck(cudaMallocManaged(&mPtr, mCapacity, cudaMemAttachGlobal));
    }

     /// @brief Similar to the constructor above except the size and capacity are equal, so no future growth is supported
    UnifiedBuffer(size_t size) : UnifiedBuffer(size, size){}

    /// @brief Constructor that specifies the size, capacity, and device (for prefetching)
    /// @param size
    /// @param capacity
    /// @param device
    /// @param stream
    UnifiedBuffer(uint64_t size, uint64_t capacity, int device, cudaStream_t stream = 0) : mPtr(nullptr), mSize(size), mCapacity(size)
    {
        assert(mSize <= mCapacity);
        cudaCheck(cudaMallocManaged(&mPtr, mCapacity, cudaMemAttachGlobal));
        cudaCheck(util::cuda::memAdvise(mPtr, size, cudaMemAdviseSetPreferredLocation, device));
        cudaCheck(util::cuda::memPrefetchAsync(mPtr, size, device, stream));
    }

    /// @brief Constructor with a specified device
    /// @param size
    /// @param device
    /// @param stream
    UnifiedBuffer(uint64_t size, int device, cudaStream_t stream = 0) : UnifiedBuffer(size, size, device, stream){}

     /// @brief Disallow copy-construction
    UnifiedBuffer(const UnifiedBuffer&) = delete;

    /// @brief Move copy-constructor
    UnifiedBuffer(UnifiedBuffer&& other) noexcept
        : mPtr(other.mPtr)
        , mSize(other.mSize)
        , mCapacity(other.mCapacity)
    {
        other.mPtr = nullptr;
        other.mSize = other.mCapacity = 0;
    }

    /// @brief Destructor
    ~UnifiedBuffer(){cudaCheck(cudaFree(mPtr));}

    ///////////////////////////////////////////////////////////////////////

    //@{
    /// @brief Factory methods that create an UnifiedBuffer instance and returns it with move semantics
    static UnifiedBuffer create(size_t size, size_t capacity) {return UnifiedBuffer(size, capacity);}
    static UnifiedBuffer create(size_t size) {return UnifiedBuffer(size);}
    ///@}

    //@{
    /// @brief Factory methods that create a shared pointer to an UnifiedBuffer instance
    static PtrT createPtr(size_t size, size_t capacity) {return std::make_shared<UnifiedBuffer>(size, capacity);}
    static PtrT createPtr(size_t size) {return std::make_shared<UnifiedBuffer>(size);}
    ///@}

    /// @brief Legacy factory method that mirrors DeviceBuffer. It creates a UnifiedBuffer from a size and a reference buffer.
    ///        If a reference buffer is provided and its non-empty, it is used to defined the capacity of the new buffer
    /// @param size Size on bytes of the new buffer
    /// @param reference reference buffer optionally used to define the capacity
    /// @param host Ignored for now
    /// @param stream cuda stream
    /// @return An instance of a new UnifiedBuffer using move semantics
    static UnifiedBuffer create(size_t size, const UnifiedBuffer* reference, int device, cudaStream_t stream)
    {
        const size_t capacity = (reference && reference->capacity()) ? reference->capacity() : size;
        UnifiedBuffer buffer(size, capacity);
        cudaCheck(util::cuda::memAdvise(buffer.mPtr, size, cudaMemAdviseSetPreferredLocation, device));
        cudaCheck(util::cuda::memPrefetchAsync(buffer.mPtr, size, device, stream));
        return buffer;
    }

    /// @brief Factory method that created a buffer on the host of the specified size. If the
    ///        reference buffer has a capacity it is used. Also the buffer is prefetched to the host
    /// @param size byte size of buffer initiated on the host
    /// @param reference optional reference buffer from which the capacity is derived
    static UnifiedBuffer create(size_t size, const UnifiedBuffer* reference){return create(size, reference, cudaCpuDeviceId, (cudaStream_t)0);}

    /// @brief Factory method that created a buffer on the host or device of the specified size. If the
    ///        reference buffer has a capacity it is used. Also the buffer is prefetched to the host or (current) device
    /// @param size byte size of buffer initiated on the device or host
    /// @param reference optional reference buffer from which the capacity is derived
    /// @param host If true the buffer will be prefetched to the host, else to the current device
    /// @param stream optional cuda stream
    static UnifiedBuffer create(size_t size, const UnifiedBuffer* reference, bool host, void* stream = nullptr)
    {
        int device = cudaCpuDeviceId;
        if (!host) cudaGetDevice(&device);
        return create(size, reference, device, (cudaStream_t)stream);
    }

    /// @brief Free all memory and reset this instance to empty
    void clear()
    {
        cudaCheck(cudaFree(mPtr));
        mPtr = nullptr;
        mSize = mCapacity = 0;
    }

    /// @brief Disallow copy assignment operation
    UnifiedBuffer& operator=(const UnifiedBuffer&) = delete;

    /// @brief Allow move assignment operation
    UnifiedBuffer& operator=(UnifiedBuffer&& other)
    {
        cudaCheck(cudaFree(mPtr));
        mPtr = other.mPtr;
        mSize = other.mSize;
        mCapacity = other.mCapacity;
        other.mPtr = nullptr;
        other.mSize = other.mCapacity = 0;
        return *this;
    }

    /// @brief initialize buffer as a new with the specified size and capacity
    /// @param size size of memory block to be used in bytes
    /// @param capacity size of page table in bytes
    void init(size_t size, size_t capacity)
    {
        NANOVDB_ASSERT(size <= capacity);
        cudaCheck(cudaFree(mPtr));
        mSize = size;
        mCapacity = capacity;
        cudaCheck(cudaMallocManaged(&mPtr, capacity, cudaMemAttachGlobal));
    }

    /// @brief Resize the memory block managed by this buffer. If the current capacity is larger than the new size this method
    ///        simply redefines size. Otherwise a new page-table is defined, with the specified advice, and the old block is copied to the new block.
    /// @param size size of the new memory block
    /// @param dev the device ID on which to apply each advice provided in list, cudaCpuDeviceId = -1, 0, 1, ...
    /// @param list advices to be applied to the resized range
    void resize(size_t size, int dev = cudaCpuDeviceId, std::initializer_list<cudaMemoryAdvise> list = {cudaMemAdviseSetPreferredLocation})
    {
        if (size <= mCapacity) {
            mSize = size;
        } else {
            void *ptr = 0;
            cudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
            if (dev > -2) for (auto a : list) cudaCheck(util::cuda::memAdvise(ptr, size, a, dev));
            if (mSize > 0) {// copy over data from the old memory block
                cudaCheck(cudaMemcpy(ptr, mPtr, std::min(mSize, size), cudaMemcpyDefault));
                cudaCheck(cudaFree(mPtr));
            }
            mPtr = ptr;
            mSize = mCapacity = size;
        }
    }

    /// @brief Apply a single advise to a memory block
    /// @param byteOffset offset in bytes marking the beginning of the memory block to be advised
    /// @param size size in bytes of the memory block to be advised.
    /// @param dev the device ID on which to apply the advice provided in adv, cudaCpuDeviceId = -1, 0, 1, ...
    /// @param adv advice to be applied to the resized range
    void advise(ptrdiff_t byteOffset, size_t size, int dev, cudaMemoryAdvise adv) const
    {
        cudaCheck(util::cuda::memAdvise(util::PtrAdd(mPtr, byteOffset), size, adv, dev));
    }

    /// @brief Apply a list of advices to a memory block
    /// @param byteOffset offset in bytes marking the beginning of the memory block to be advised
    /// @param size size in bytes of the memory block to be advised.
    /// @param dev the device ID to prefetch to, cudaCpuDeviceId = -1, 0, 1, ...
    /// @param list list of cuda advises
    void advise(ptrdiff_t byteOffset, size_t size, int dev, std::initializer_list<cudaMemoryAdvise> list) const
    {
        void *ptr = util::PtrAdd(mPtr, byteOffset);
        for (auto a : list)  cudaCheck(util::cuda::memAdvise(ptr, size, a, dev));
    }

    /// @brief Prefetches data to the specified device, i.e. ensure the device has an up-to-date copy of the memory specified
    /// @param byteOffset offset in bytes marking the beginning of the memory block to be prefetched
    /// @param size size in bytes of the memory block to be prefetched. The default value of zero means copy all @c this->size() bytes.
    /// @param dev the device ID to prefetch to, cudaCpuDeviceId = -1, 0, 1, ...
    /// @param stream  cuda stream
    void prefetch(ptrdiff_t byteOffset = 0, size_t size = 0, int dev = cudaCpuDeviceId, cudaStream_t stream = cudaStreamPerThread) const
    {
        cudaCheck(util::cuda::memPrefetchAsync(util::PtrAdd(mPtr, byteOffset), size ? size : mSize, dev, stream));
    }

    ///////////////////////////////////////////////////////////////////////

    /// @brief Prefetches all data to the specified device
    /// @param device device ID, cudaCpuDeviceId = -1, 0, 1, ...
    /// @param stream cuda stream
    /// @param sync if false the memory copy is asynchronous
    /// @note Legacy method included for compatibility with DeviceBuffer
    void deviceUpload(int device = 0, cudaStream_t stream = cudaStreamPerThread, bool sync = false) const
    {
        cudaCheck(util::cuda::memPrefetchAsync(mPtr, mSize, device, stream));
        if (sync) cudaCheck(cudaStreamSynchronize(stream));
    }
    void deviceUpload(int device, void* stream, bool sync) const{this->deviceUpload(device, cudaStream_t(stream));}

    /// @brief Prefetches all data to the current device, as given by cudaGetDevice
    /// @param stream cuda stream
    /// @param sync if false the memory copy is asynchronous
    /// @note Legacy method included for compatibility with DeviceBuffer
    void deviceUpload(void* stream, bool sync) const{
        int device = 0;
        cudaCheck(cudaGetDevice(&device));
        this->deviceUpload(device, cudaStream_t(stream), sync);
    }

    ///////////////////////////////////////////////////////////////////////

    /// @brief Prefetches all data to the host
    /// @param stream cuda stream
    /// @param sync if false the memory copy is asynchronous
    void deviceDownload(cudaStream_t stream = 0, bool sync = false) const
    {
        cudaCheck(util::cuda::memPrefetchAsync(mPtr, mSize, cudaCpuDeviceId, stream));
        if (sync) cudaCheck(cudaStreamSynchronize(stream));
    }

    /// @brief Legacy
    /// @param stream
    /// @param sync
    void deviceDownload(void* stream, bool sync) const{this->deviceDownload(cudaStream_t(stream), sync);}

    // used by GridHandle
    void deviceDownload(int dummmy, void* stream, bool sync) const{this->deviceDownload(cudaStream_t(stream), sync);}

    ///////////////////////////////////////////////////////////////////////

    /// @brief Returns a raw pointer to the unified memory managed by this instance.
    /// @warning Note that the pointer can be NULL!
    void* data() const {return mPtr;}

    /// @brief Returns an offset pointer of a specific type from the allocated unified memory
    /// @tparam T Type of the pointer returned
    /// @param count Numbers of elements of @c parameter type T to skip (or offset) the return pointer
    /// @warning assumes that this instance is not empty!
    template <typename T>
    T* data(ptrdiff_t count = 0) const {
        NANOVDB_ASSERT(mPtr != nullptr || count == 0);
        return reinterpret_cast<T*>(mPtr) + count;
    }

    /// @brief Returns a byte offset void pointer from the unified memory
    /// @param byteOffset Number of bytes to skip (or offset) the return pointer
    /// @warning assumes that this instance is not empty!
    void* data(ptrdiff_t byteOffset) const {
        NANOVDB_ASSERT(mPtr != nullptr || byteOffset == 0);
        return util::PtrAdd(mPtr, byteOffset);
    }

    /// @brief Legacy
    /// @return
    void* deviceData()    const {return mPtr;}
    void* deviceData(int) const {return mPtr;}

    /// @brief Size of the allocated pages in this instance
    /// @return number bytes allocated by this instance
    size_t size() const {return mSize;}

    /// @brief Capacity of this instance, i.e. room in page table
    /// @return number of bytes reserved, but not necessarily allocated, by this instance
    size_t capacity() const {return mCapacity;}

    //@{
    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    inline bool empty() const { return mPtr == nullptr; }
    inline bool isEmpty() const { return this->empty(); }
    //@}

};// UnifiedBuffer

}// namespace cuda

template<>
struct BufferTraits<cuda::UnifiedBuffer>
{
    static constexpr bool hasDeviceDual = true;
};

}// namespace nanovdb

#endif // end of NANOVDB_CUDA_UNIFIEDBUFFER_H_HAS_BEEN_INCLUDED
