// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file DeviceBuffer.h

    \author Ken Museth

    \date January 8, 2020

    \brief DeviceBuffer has one pinned host buffer and multiple device CUDA buffers

    \note This file has no device-only kernel functions,
          which explains why it's a .h and not .cuh file.
*/

#ifndef NANOVDB_CUDA_DEVICEBUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_DEVICEBUFFER_H_HAS_BEEN_INCLUDED

#include <cuda.h>
#include <memory>// for std::shared_ptr
#include <nanovdb/HostBuffer.h>// for BufferTraits
#include <nanovdb/util/cuda/Util.h>// for cudaMalloc/cudaMallocManaged/cudaFree

namespace nanovdb {// ================================================================

namespace cuda {// ===================================================================

// ----------------------------> DeviceBuffer <--------------------------------------

/// @brief Simple memory buffer using un-managed pinned host memory when compiled with NVCC.
///        Obviously this class is making explicit used of CUDA so replace it with your own memory
///        allocator if you are not using CUDA.
/// @note  While CUDA's pinned host memory allows for asynchronous memory copy between host and device
///        it is significantly slower then cached (un-pinned) memory on the host.
class DeviceBuffer
{
    uint64_t mSize; // total number of bytes managed by this buffer (assumed to be identical for host and device)
    void *mCpuData, **mGpuData; // raw pointers to the host and device buffers
    int   mDeviceCount, mManaged;// if mManaged is non-zero this class is responsible for allocating and freeing memory buffers. Otherwise this is assumed to be handled externally
    cudaEvent_t *mEvents = nullptr;// per-device event marking the last use of each managed device buffer (parallel to mGpuData, length mDeviceCount). Every use waits on this event before issuing work and re-records it afterwards, so the single event transitively covers EVERY stream the buffer has been used on. Frees then wait on it, which orders them after all outstanding work: freeing on the default stream alone is only safe for blocking streams, and freeing on the last-used stream alone is only safe when just one stream was used.

    /// @brief Initialize buffer
    /// @param size byte size of buffer to be initialized
    /// @param device id of the device on which to initialize the buffer
    /// @note All existing buffers are first cleared
    /// @warning size is expected to be non-zero. Use clear() clear buffer!
    void init(uint64_t size, int device, cudaStream_t stream);

    /// @brief Free every managed device allocation, each ordered after all tracked uses of the
    ///        buffer, and destroy the tracking events.
    /// @param stream Stream the frees are issued on for allocations owned by the CURRENT device.
    ///        A stream belongs to a single device, so it cannot carry frees for other devices'
    ///        memory pools; allocations on other devices are freed on their own device's default
    ///        stream, after switching to that device.
    /// @note Destroying an event with a pending wait is safe: CUDA releases it once the device
    ///       has completed it.
    void freeDeviceBuffers(cudaStream_t stream)
    {
        int current = 0;
        cudaCheck(cudaGetDevice(&current));
        for (int i = 0; i < mDeviceCount; ++i) {
            if (mGpuData[i]) {
                const cudaStream_t freeStream = (i == current) ? stream : cudaStream_t{0};
                if (i != current) cudaCheck(cudaSetDevice(i));
                this->orderAfterPriorUses(i, freeStream);
                cudaCheck(util::cuda::freeAsync(mGpuData[i], freeStream));
                if (i != current) cudaCheck(cudaSetDevice(current));
            }
            if (mEvents && mEvents[i]) {
                cudaCheck(cudaEventDestroy(mEvents[i]));
                mEvents[i] = nullptr;
            }
        }
    }

public:

    using PtrT = std::shared_ptr<DeviceBuffer>;

    /// @brief Default constructor of an empty buffer
    DeviceBuffer() : mSize(0), mCpuData(nullptr), mGpuData(nullptr), mDeviceCount(0), mManaged(0){}

    /// @brief Constructor with a specified device and size
    /// @param size byte size of buffer to be initialized
    /// @param device id of the device on which to initialize the buffer
    /// @param stream cuda stream
    DeviceBuffer(uint64_t size, int device = cudaCpuDeviceId, cudaStream_t stream = 0) : DeviceBuffer()
    {
        this->init(size, device, stream);
    }

    /// @brief Constructor
    /// @param size byte size of buffer to be initialized
    /// @param host If true buffer is initialized only on the host/CPU, else on the current device/GPU
    /// @param stream optional stream argument (defaults to stream NULL)
    DeviceBuffer(uint64_t size, bool host, void* stream) : DeviceBuffer()
    {
        int device = cudaCpuDeviceId;
        if (!host) cudaCheck(cudaGetDevice(&device));
        this->init(size, device, reinterpret_cast<cudaStream_t>(stream));
    }

    /// @brief Constructor for externally managed host and device buffers
    /// @param size byte size of the two external buffers
    /// @param cpuData host buffer, assumed to NOT be NULL
    /// @param gpuData device buffer, assumed to NOT be NULL;
    /// @note The device buffer, @c gpuData, will be associated
    ///       with the current device ID given by cudaGetDevice
    DeviceBuffer(uint64_t size, void* cpuData, void* gpuData)
        : mSize(size)
        , mCpuData(cpuData)
        , mManaged(0)
    {
        cudaCheck(cudaGetDeviceCount(&mDeviceCount));
        mGpuData = new void*[mDeviceCount]();// NULL initialization
        NANOVDB_ASSERT(cpuData);
        NANOVDB_ASSERT(gpuData);
        int device = 0;
        cudaCheck(cudaGetDevice(&device));
        mGpuData[device] = gpuData;
    }

    /// @brief Constructor for externally managed host and multiple device buffers
    /// @param size byte size of the two external buffers
    /// @param cpuData host buffer, assumed to NOT be NULL
    /// @param list list of device IDs and external device buffers, all assumed to not be NULL
    DeviceBuffer(uint64_t size, void* cpuData, std::initializer_list<std::pair<int,void*>> list)
        : mSize(size)
        , mCpuData(cpuData)
        , mManaged(0)
    {
        NANOVDB_ASSERT(cpuData);
        cudaCheck(cudaGetDeviceCount(&mDeviceCount));
        mGpuData = new void*[mDeviceCount]();// NULL initialization
        for (auto &p : list) {
            NANOVDB_ASSERT(p.first>=0 && p.first<mDeviceCount);
            NANOVDB_ASSERT(p.second);
            mGpuData[p.first] = p.second;
        }
    }

    /// @brief Disallow copy-construction
    DeviceBuffer(const DeviceBuffer&) = delete;

    /// @brief Move copy-constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : mSize(other.mSize)
        , mCpuData(other.mCpuData)
        , mGpuData(other.mGpuData)
        , mDeviceCount(other.mDeviceCount)
        , mManaged(other.mManaged)
        , mEvents(other.mEvents)
    {
        other.mCpuData = other.mGpuData = nullptr;
        other.mEvents = nullptr;
        other.mSize = other.mDeviceCount = other.mManaged = 0;
    }

    /// @brief Copy-constructor from a HostBuffer
    /// @param buffer host buffer from which to copy data
    /// @param device id of the device on which to initialize the buffer
    /// @param stream cuda stream
    DeviceBuffer(const HostBuffer& buffer, int device = cudaCpuDeviceId, cudaStream_t stream = 0)
        : DeviceBuffer(buffer.size(), device, stream)
    {
        if (mCpuData) {
            cudaCheck(cudaMemcpy(mCpuData, buffer.data(), mSize, cudaMemcpyHostToHost));
        } else if (mGpuData[device]) {
            cudaCheck(cudaMemcpyAsync(mGpuData[device], buffer.data(), mSize, cudaMemcpyHostToDevice, stream));
        }
    }

     /// @brief Destructor frees memory on both the host and device
    /// @note Each managed device free waits on that device's tracking event first, so it is
    ///       ordered after every stream the buffer was used on, not just the most recent one.
    ~DeviceBuffer() { this->clear(); };

    /// @brief Static factory method that return an instance of this buffer
    /// @param size byte size of buffer to be initialized
    /// @param dummy this argument is currently ignored but required to match the API of the HostBuffer
    /// @param host If true buffer is initialized only on the host/CPU, else only on the device/GPU
    /// @param stream optional stream argument (defaults to stream NULL)
    /// @return An instance of this class using move semantics
    static DeviceBuffer create(uint64_t size, const DeviceBuffer* dummy, bool host, void* stream){return DeviceBuffer(size, host, stream);}

    /// @brief Static factory method that returns an instance of this buffer
    /// @param size byte size of buffer to be initialized
    /// @param dummy this argument is currently ignored but required to match the API of the HostBuffer
    /// @param device id of the device on which to initialize the buffer
    /// @param stream cuda stream
    static DeviceBuffer create(uint64_t size, const DeviceBuffer* dummy = nullptr, int device = cudaCpuDeviceId, cudaStream_t stream = 0){return DeviceBuffer(size, device, stream);}

    /// @brief Static factory method that returns an instance of this buffer that wraps externally managed memory
    /// @param size byte size of buffer specified by external memory
    /// @param cpuData pointer to externally managed host memory
    /// @param gpuData pointer to externally managed device memory
    /// @return An instance of this class using move semantics
    static DeviceBuffer create(uint64_t size, void* cpuData, void* gpuData) {return DeviceBuffer(size, cpuData, gpuData);}

    /// @brief  Static factory method that returns an instance of this buffer that wraps externally managed host and device memory
    /// @param size byte size of buffer to be initialized
    /// @param cpuData  pointer to externally managed host memory
    /// @param list list of device IDs and device memory pointers
    static DeviceBuffer create(uint64_t size, void* cpuData, std::initializer_list<std::pair<int,void*>> list) {return DeviceBuffer(size, cpuData, list);}

    /// @brief Static factory method that returns an instance of this buffer constructed from a HostBuffer
    /// @param buffer host buffer from which to copy data
    /// @param device id of the device on which to initialize the buffer
    /// @param stream cuda stream
    static DeviceBuffer create(const HostBuffer& buffer, int device = cudaCpuDeviceId, cudaStream_t stream = 0) {return DeviceBuffer(buffer, device, stream);}

    ///////////////////////////////////////////////////////////////////////

    /// @{
    /// @brief Factory methods that create a shared pointer to an DeviceBuffer instance
    static PtrT createPtr(uint64_t size, const DeviceBuffer* = nullptr, int device = cudaCpuDeviceId, cudaStream_t stream = 0) {return std::make_shared<DeviceBuffer>(size, device, stream);}
    static PtrT createPtr(uint64_t size, void* cpuData, void* gpuData) {return std::make_shared<DeviceBuffer>(size, cpuData, gpuData);}
    static PtrT createPtr(uint64_t size, void* cpuData, std::initializer_list<std::pair<int,void*>> list) {return std::make_shared<DeviceBuffer>(size, cpuData, list);}
    static PtrT createPtr(const HostBuffer& buffer, int device = cudaCpuDeviceId, cudaStream_t stream = 0) {return std::make_shared<DeviceBuffer>(buffer, device, stream);}
    /// @}

    ///////////////////////////////////////////////////////////////////////

    /// @brief Disallow copy assignment operation
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /// @brief Move copy assignment operation
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    ///////////////////////////////////////////////////////////////////////

    /// @brief Retuns a raw void pointer to the host/CPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    void* data() const { return mCpuData; }

    /// @brief Returns an offset pointer of a specific type from the allocated host memory
    /// @tparam T Type of the pointer returned
    /// @param count Numbers of elements of @c parameter type T to skip
    /// @warning might return NULL
    template <typename T>
    T* data(ptrdiff_t count = 0, int device = cudaCpuDeviceId) const
    {
        NANOVDB_ASSERT(device >= cudaCpuDeviceId && device < mDeviceCount);
        void *ptr = device == cudaCpuDeviceId ? mCpuData : mGpuData[device];
        return ptr ? reinterpret_cast<T*>(ptr) + count : nullptr;
    }

    /// @brief Returns a byte offset void pointer from the allocated host memory
    /// @param byteOffset offset of return pointer in units of bytes
    /// @warning assumes that this instance is not empty!
    void* data(ptrdiff_t byteOffset, int device = cudaCpuDeviceId) const
    {
        NANOVDB_ASSERT(device >= cudaCpuDeviceId && device < mDeviceCount);
        void *ptr = device == cudaCpuDeviceId ? mCpuData : mGpuData[device];
        return ptr ? reinterpret_cast<char*>(ptr) + byteOffset : nullptr;
    }

    ///////////////////////////////////////////////////////////////////////

    /// @brief Order work subsequently issued on @a stream after every prior use of this
    ///        device buffer, whichever stream those uses were issued on. The consume-side
    ///        companion of recordUse: an external consumer (e.g. a zero-copy array-interface
    ///        export) calls this with its own stream before reading, so it cannot observe a
    ///        partially-written buffer after asynchronous uploads or recorded kernels.
    /// @param device Device whose buffer is about to be read
    /// @param stream Stream the consumer's work will be issued on
    void orderAfterPriorUses(int device, cudaStream_t stream) const
    {
        if (mEvents && mEvents[device]) cudaCheck(cudaStreamWaitEvent(stream, mEvents[device], 0));
    }

    /// @brief Record that this buffer's device data was just used on @a stream, so that the
    ///        buffer's device frees (destructor, move-assignment, clear) are ordered after that
    ///        work. Uses issued through deviceUpload/deviceDownload are recorded automatically;
    ///        callers that enqueue their own kernels or copies against the raw pointer returned
    ///        by deviceData() should call this afterwards. Without it, such work is only safe if
    ///        it is on a blocking stream (which the free, issued on the default stream, waits on
    ///        implicitly) or if the caller synchronizes before the buffer is cleared/destroyed.
    /// @param device Device whose buffer was used
    /// @param stream Stream the work was issued on
    /// @note Recording chains across streams: @a stream is first ordered after the previously
    ///       recorded use (if any) so the single per-device event transitively covers every
    ///       recorded use, not just the last one. Without this, concurrent uses on streams A
    ///       then B would leave only B's event, and the device free could run while A's work
    ///       is still in flight. The side effect is that work subsequently issued on @a stream
    ///       also waits on the previously recorded use -- acceptable for a shared buffer, where
    ///       later-recorded consumers observing earlier writes is the expected ordering. Note
    ///       this also serializes CONCURRENT READERS that record uses (the single event cannot
    ///       distinguish read-read from write-read); if that ever matters in a profile, the
    ///       upgrade path is a read/write-separated or per-record event scheme, not a revert.
    void recordUse(int device, cudaStream_t stream)
    {
        if (!mEvents) return;
        if (mEvents[device] == nullptr) {// events are per-device, so create it on the right one
            int current = 0;
            cudaCheck(cudaGetDevice(&current));
            if (current != device) cudaCheck(cudaSetDevice(device));
            cudaCheck(cudaEventCreateWithFlags(&mEvents[device], cudaEventDisableTiming));
            if (current != device) cudaCheck(cudaSetDevice(current));
        } else {
            // Re-recording MOVES the event; chain first so the new capture also covers the
            // prior recorded use (waiting on a never-recorded or completed event is a no-op).
            cudaCheck(cudaStreamWaitEvent(stream, mEvents[device], 0));
        }
        cudaCheck(cudaEventRecord(mEvents[device], stream));
    }

    /// @brief Retuns a raw pointer to the specified device/GPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    /// @note Work enqueued against this raw pointer is invisible to the buffer's lifetime
    ///       tracking: on a non-blocking stream, call recordUse afterwards (or synchronize
    ///       before the buffer is cleared/destroyed) so the device free is ordered after it.
    void* deviceData(int device) const {
        NANOVDB_ASSERT(device >= 0 && device < mDeviceCount);
        return mGpuData[device];
    }

    /// @brief Retuns a raw pointer to the current device/GPU buffer managed by this allocator.
    /// @warning Note that the pointer can be NULL!
    void* deviceData() const {
        int device = cudaCpuDeviceId;
        cudaCheck(cudaGetDevice(&device));
        return this->deviceData(device);
    }

    ///////////////////////////////////////////////////////////////////////

    /// @brief Uploads buffer on the host to a specific device. If it doesn't exist it's created first.
    /// @param device Device ID that the data is copied to
    /// @param stream cuda stream
    /// @param sync if false the memory copy is asynchronous.
    /// @warning Assumes that the host buffer already exists!
    /// @note determine the current device with cudaGetDevice
    void deviceUpload(int device = 0, cudaStream_t stream = 0, bool sync = true);
    void deviceUpload(int device, void* stream, bool sync){this->deviceUpload(device, cudaStream_t(stream), sync);}

    /// @brief Upload buffer from the host to ALL the existing devices, i.e. CPU -> GPU.
    ///        If no device buffers exist one is created for the current device (typically 0)
    ///        and subsequently populated with the host data.
    /// @param stream CUDA stream.
    /// @param sync if false the memory copy is asynchronous.
    /// @warning Assumes that the host buffer already exists!
    void deviceUpload(cudaStream_t stream, bool sync);
    void deviceUpload(void* stream, bool sync) {this->deviceUpload(cudaStream_t(stream), sync);}

    ///////////////////////////////////////////////////////////////////////

    /// @brief Download data from a specified device to the host. If the host buffer des not exist it will first be allocated
    /// @param device device ID to download source data from
    /// @param stream cuda stream
    /// @param sync if false the memory copy is asynchronous.
    /// @warning Assumes that the specifed device buffer already exists!
    void deviceDownload(int device = 0, cudaStream_t stream = 0, bool sync = true);
    void deviceDownload(int device, void* stream , bool sync) {this->deviceDownload(device, cudaStream_t(stream), sync);}

    /// @brief Download the buffer from the current device to the host, i.e. GPU -> CPU.
    ///        If the host buffer des not exist it will first be allocated
    /// @param stream CUDA stream
    /// @param sync if false the memory copy is asynchronous
    /// @note If the host/CPU buffer does not exist it is first allocated
    /// @warning Assumes that the device/GPU buffer already exists
    void deviceDownload(void* stream, bool sync);

    ///////////////////////////////////////////////////////////////////////

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const { return mSize; }
    uint64_t capacity() const {return this->size();}

    /// @brief Returns the number of buffers that are not NULL
    int bufferCount() const {
        int count = mCpuData ? 1 : 0;
        for (int i=0; i<mDeviceCount; ++i) if (mGpuData[i]) ++count;
        return count;
    }

    int deviceCount() const {return mDeviceCount;}

    /// @{
    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const { return mSize == 0; }
    bool isEmpty() const { return this->empty(); }
    /// @}

    /// @brief De-allocate all memory managed by this allocator and set all pointers to NULL
    /// @param stream Stream the device frees are issued on. The frees are additionally ordered
    ///        after every stream the buffer was used on (via the per-device tracking event), so
    ///        @a stream selects where the free is enqueued, not what it is ordered against - any
    ///        stream is safe to pass here regardless of where the buffer was used.
    void clear(cudaStream_t stream = 0);
    void clear(void* stream){this->clear(cudaStream_t(stream));}

}; // DeviceBuffer class

// --------------------------> Implementations below <------------------------------------

inline DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
{
    if (this == &other) return *this;// self-move would free our buffers and then read them back
    if (mManaged) {// first free all the managed data buffers, ordered after every use of each
        cudaCheck(cudaFreeHost(mCpuData));
        this->freeDeviceBuffers(cudaStream_t{0});
    }
    delete [] mGpuData;
    delete [] mEvents;
    mSize    = other.mSize;
    mCpuData = other.mCpuData;
    mGpuData = other.mGpuData;
    mDeviceCount = other.mDeviceCount;
    mManaged = other.mManaged;
    mEvents = other.mEvents;
    other.mCpuData = nullptr;
    other.mGpuData = nullptr;
    other.mEvents = nullptr;
    other.mSize = 0;
    other.mDeviceCount = 0;
    other.mManaged = 0;
    return *this;
}

inline void DeviceBuffer::init(uint64_t size, int device, cudaStream_t stream)
{
    if (size==0) return;
    cudaCheck(cudaGetDeviceCount(&mDeviceCount));
    mGpuData = new void*[mDeviceCount]();// NULL initialization
    mEvents = new cudaEvent_t[mDeviceCount]();// NULL initialization; created lazily on first use
    NANOVDB_ASSERT(device >= cudaCpuDeviceId && device < mDeviceCount);
    if (device == cudaCpuDeviceId) {
        cudaCheck(cudaMallocHost((void**)&mCpuData, size)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
        checkPtr(mCpuData, "cuda::DeviceBuffer::init: failed to allocate host buffer");
    } else {
        cudaCheck(util::cuda::mallocAsync(mGpuData+device, size, stream)); // un-managed memory on the device, always 32B aligned!
        checkPtr(mGpuData[device], "cuda::DeviceBuffer::init: failed to allocate device buffer");
        this->recordUse(device, stream);// the free must be ordered after this allocation
    }
    mSize = size;
    mManaged = 1;// i.e. this instance is responsible for allocating and delete memory
} // DeviceBuffer::init

inline void DeviceBuffer::deviceUpload(int device, cudaStream_t stream, bool sync)
{
    NANOVDB_ASSERT(device >= 0 && device < mDeviceCount);// should be device and not the host
    checkPtr(mCpuData, "uninitialized cpu source data");
    if (mGpuData[device] == nullptr) {
        if (mManaged==0) throw std::runtime_error("DeviceBuffer::deviceUpload called on externally managed memory that wasn\'t allocated.");
        cudaCheck(util::cuda::mallocAsync(mGpuData+device, mSize, stream)); // un-managed memory on the device, always 32B aligned!
    }
    checkPtr(mGpuData[device], "uninitialized gpu destination data");
    // Order this transfer after any use of the buffer on another stream, then mark it as the
    // latest use, so the tracking event keeps covering every stream the buffer has seen.
    this->orderAfterPriorUses(device, stream);
    cudaCheck(cudaMemcpyAsync(mGpuData[device], mCpuData, mSize, cudaMemcpyHostToDevice, stream));
    this->recordUse(device, stream);
    if (sync) cudaCheck(cudaStreamSynchronize(stream));
} // DeviceBuffer::deviceUpload

inline void DeviceBuffer::deviceUpload(cudaStream_t stream, bool sync)
{
    int device = 0;
    cudaGetDevice(&device);
    this->deviceUpload(device, stream, sync);
} // DeviceBuffer::deviceUpload

inline void DeviceBuffer::deviceDownload(int device, cudaStream_t stream, bool sync)
{
    NANOVDB_ASSERT(device >= 0 && device < mDeviceCount);
    checkPtr(mGpuData[device], "uninitialized gpu source data");// no source data on the specified device
    if (mCpuData == nullptr) {
        if (mManaged==0) throw std::runtime_error("DeviceBuffer::deviceDownload called on uninitialized cpu destination memory that is externally managed.");
        cudaCheck(cudaMallocHost((void**)&mCpuData, mSize)); // un-managed pinned memory on the host (can be slow to access!). Always 32B aligned
    }
    checkPtr(mCpuData, "uninitialized cpu destination data");
    this->orderAfterPriorUses(device, stream);
    cudaCheck(cudaMemcpyAsync(mCpuData, mGpuData[device], mSize, cudaMemcpyDeviceToHost, stream));
    this->recordUse(device, stream);
    if (sync) cudaCheck(cudaStreamSynchronize(stream));
} // DeviceBuffer::deviceDownload

inline void DeviceBuffer::deviceDownload(void* stream, bool sync)
{
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    this->deviceDownload(device, cudaStream_t(stream), sync);
} // DeviceBuffer::deviceDownload

inline void DeviceBuffer::clear(cudaStream_t stream)
{
    if (mManaged) {// free all the managed data buffers, ordered after every use of each
        cudaCheck(cudaFreeHost(mCpuData));
        this->freeDeviceBuffers(stream);
    }
    delete [] mGpuData;
    delete [] mEvents;
    mCpuData = nullptr;
    mGpuData = nullptr;
    mEvents = nullptr;
    mSize = 0;
    mDeviceCount = 0;
    mManaged = 0;
} // DeviceBuffer::clear

}// namespace cuda

using CudaDeviceBuffer [[deprecated("Use nanovdb::cuda::DeviceBuffer instead")]] = cuda::DeviceBuffer;

template<>
struct BufferTraits<cuda::DeviceBuffer>
{
    static constexpr bool hasDeviceDual = true;
};

}// namespace nanovdb

#endif // end of NANOVDB_CUDA_DEVICEBUFFER_H_HAS_BEEN_INCLUDED
