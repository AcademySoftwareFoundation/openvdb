// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    @file HostBuffer.h

    @date April 20, 2021

    @brief HostBuffer - a buffer that contains a shared or private bump
           pool to either externally or internally managed host memory.

    @details This HostBuffer can be used in multiple ways, most of which are
             demonstrated in the examples below. Memory in the pool can
             be managed or unmanged (e.g. internal or external) and can
             be shared between multiple buffers or belong to a single buffer.

   Example that uses HostBuffer::create inside io::readGrids to create a
   full self-managed buffer, i.e. not shared and without padding, per grid in the file.
   @code
        auto handles = nanovdb::io::readGrids("file.nvdb");
   @endcode

   Example that uses HostBuffer::createFull. Assuming you have a raw pointer
   to a NanoVDB grid of unknown type, this examples shows how to create its
   GridHandle which can be used to enquire about the grid type and meta data.
   @code
        void    *data;// pointer to a NanoVDB grid of unknown type
        uint64_t size;// byte size of NanoVDB grid of unknown type
        auto buffer = nanovdb::HostBuffer::createFull(size, data);
        nanovdb::GridHandle<> gridHandle(std::move(buffer));
   @endcode

   Example that uses HostBuffer::createPool for internally managed host memory.
   Suppose you want to read multiple grids in multiple files, but reuse the same
   fixed sized memory buffer to both avoid memory fragmentation as well as
   exceeding the fixed memory ceiling!
   @code
        auto pool = nanovdb::HostBuffer::createPool(1 << 30);// 1 GB memory pool
        std::vector<std::string>> frames;// vector of grid names
        for (int i=0; i<frames.size(); ++i) {
            auto handles = nanovdb::io::readGrids(frames[i], 0, pool);// throws if grids in file exceed 1 GB
            ...
            pool.reset();// clears all handles and resets the memory pool for reuse
        }
   @endcode

   Example that uses HostBuffer::createPool for externally managed host memory.
   Note that in this example @c handles are allowed to outlive @c pool since
   they internally store a shared pointer to the memory pool. However @c data
   MUST outlive @c handles since the pool does not own its memory in this example.
   @code
        const size_t poolSize = 1 << 30;// 1 GB
        uint8_t *data = static_cast<uint8_t*>(std::malloc(size));// 1 GB buffer
        auto pool = nanovdb::HostBuffer::createPool(poolSize, data);
        auto handles1 = nanovdb::io::readGrids("file1.nvdb", 0, pool);
        auto handles2 = nanovdb::io::readGrids("file2.nvdb", 0, pool);
        ....
        std::free(data);
   @endcode

   Example that uses HostBuffer::createPool for externally managed host memory.
   Note that in this example @c handles are allowed to outlive @c pool since
   they internally store a shared pointer to the memory pool. However @c array
   MUST outlive @c handles since the pool does not own its memory in this example.
   @code
        const size_t poolSize = 1 << 30;// 1 GB
        std::unique_ptr<uint8_t[]> array(new uint8_t[size]);// scoped buffer of 1 GB
        auto pool = nanovdb::HostBuffer::createPool(poolSize, array.get());
        auto handles = nanovdb::io::readGrids("file.nvdb", 0, pool);
   @endcode
*/

#ifndef NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED

#include <stdint.h> //    for types like int32_t etc
#include <cstdio> //      for fprintf
#include <cstdlib> //     for std::malloc/std::reallow/std::free
#include <memory>//       for std::make_shared
#include <mutex>//        for std::mutex
#include <unordered_set>//for std::unordered_set
#include <cassert>//      for assert
#include <sstream>//      for std::stringstream
#include <cstring>//      for memcpy

#define checkPtr(ptr, msg) \
    { \
        ptrAssert((ptr), (msg), __FILE__, __LINE__); \
    }

namespace nanovdb {

template<typename BufferT>
struct BufferTraits
{
    static const bool hasDeviceDual = false;
};

// ----------------------------> HostBuffer <--------------------------------------

/// @brief This is a buffer that contains a shared or private pool
///        to either externally or internally managed host memory.
///
/// @note  Terminology:
///        Pool:   0 = buffer.size() < buffer.poolSize()
///        Buffer: 0 < buffer.size() < buffer.poolSize()
///        Full:   0 < buffer.size() = buffer.poolSize()
///        Empty:  0 = buffer.size() = buffer.poolSize()
class HostBuffer
{
    struct Pool;// forward declaration of private pool struct
    std::shared_ptr<Pool> mPool;
    uint64_t              mSize; // total number of bytes for the NanoVDB grid.
    uint8_t*              mData; // raw buffer for the NanoVDB grid.

#if defined(DEBUG) || defined(_DEBUG)
    static inline void ptrAssert(void* ptr, const char* msg, const char* file, int line, bool abort = true)
    {
        if (ptr == nullptr) {
            fprintf(stderr, "NULL pointer error: %s %s %d\n", msg, file, line);
            if (abort)
                exit(1);
        }
    }
#else
    static inline void ptrAssert(void*, const char*, const char*, int, bool = true)
    {
    }
#endif

public:
    /// @brief Return a full buffer or an empty buffer
    HostBuffer(uint64_t bufferSize = 0);

     /// @brief Move copy-constructor
    HostBuffer(HostBuffer&& other);

    /// @brief Custom descructor
    ~HostBuffer() { this->clear(); }

    /// @brief Move copy assignment operation
    HostBuffer& operator=(HostBuffer&& other);

    /// @brief Disallow copy-construction
    HostBuffer(const HostBuffer&) = delete;

    /// @brief Disallow copy assignment operation
    HostBuffer& operator=(const HostBuffer&) = delete;

    /// @brief Return a pool buffer which satisfies: buffer.size == 0,
    ///        buffer.poolSize() == poolSize, and buffer.data() == nullptr.
    ///        If data==nullptr, memory for the pool will be allocated.
    ///
    /// @throw If poolSize is zero.
    static HostBuffer createPool(uint64_t poolSize, void *data = nullptr);

    /// @brief Return a full buffer which satisfies: buffer.size == bufferSize,
    ///        buffer.poolSize() == bufferSize, and buffer.data() == data.
    ///        If data==nullptr, memory for the pool will be allocated.
    ///
    /// @throw If bufferSize is zero.
    static HostBuffer createFull(uint64_t bufferSize, void *data = nullptr);

    /// @brief Return a buffer with @c bufferSize bytes managed by
    ///        the specified memory @c pool. If none is provided, i.e.
    ///        @c pool == nullptr or @c pool->poolSize() == 0, one is
    ///        created with size @c bufferSize, i.e. a full buffer is returned.
    ///
    /// @throw If the specified @c pool has insufficient memory for
    ///        the requested buffer size.
    static HostBuffer create(uint64_t bufferSize, const HostBuffer* pool = nullptr);

    /// @brief Initialize as a full buffer with the specified size. If data is NULL
    ///        the memory is internally allocated.
    void init(uint64_t bufferSize, void *data = nullptr);

    //@{
    /// @brief Retuns a pointer to the raw memory buffer managed by this allocator.
    ///
    /// @warning Note that the pointer can be NULL if the allocator was not initialized!
    const uint8_t* data() const { return mData; }
    uint8_t* data() { return mData; }
    //@}

    //@{
    /// @brief Returns the size in bytes associated with this buffer.
    uint64_t bufferSize() const { return mSize; }
    uint64_t size() const { return this->bufferSize(); }
    //@}

    /// @brief Returns the size in bytes of the memory pool shared with this instance.
    uint64_t poolSize() const;

    /// @brief Return true if memory is managed (using std::malloc and std:free) by the
    ///        shared pool in this buffer. Else memory is assumed to be managed externally.
    bool isManaged() const;

    //@{
    /// @brief Returns true if this buffer has no memory associated with it
    bool isEmpty() const { return !mPool || mSize == 0 || mData == nullptr; }
    bool empty() const { return this->isEmpty(); }
    //@}

    /// @brief Return true if this is a pool, i.e. an empty buffer with a nonempty
    ///        internal pool, i.e. this->size() == 0 and this->poolSize() != 0
    bool isPool() const { return mSize == 0 && this->poolSize() > 0; }

    /// @brief Return true if the pool exists, is nonempty but has no more available memory
    bool isFull() const;

    /// @brief Clear this buffer so it is empty.
    void clear();

    /// @brief Clears all existing buffers that are registered against the memory pool
    ///        and resets the pool so it can be reused to create new buffers.
    ///
    /// @throw If this instance is not empty or contains no pool.
    ///
    /// @warning This method is not thread-safe!
    void reset();

    /// @brief Total number of bytes from the pool currently in use by buffers
    uint64_t poolUsage() const;

    /// @brief resize the pool size. It will attempt to resize the existing
    ///        memory block, but if that fails a deep copy is performed.
    ///        If @c data is not NULL it will be used as new externally
    ///        managed memory for the pool. All registered buffers are
    ///        updated so GridHandle::grid might return a new address (if
    ///        deep copy was performed).
    ///
    /// @note  This method can be use to resize the memory pool and even
    ///        change it from internally to externally managed memory or vice versa.
    ///
    /// @throw if @c poolSize is less than this->poolUsage() the used memory
    ///        or allocations fail.
    void resizePool(uint64_t poolSize, void *data = nullptr);

}; // HostBuffer class

// --------------------------> Implementation of HostBuffer::Pool <------------------------------------

// This is private struct of HostBuffer so you can safely ignore the API
struct HostBuffer::Pool
{
    using HashTableT = std::unordered_set<HostBuffer*>;
    std::mutex mMutex;// mutex for updating mRegister and mFree
    HashTableT mRegister;
    uint8_t*   mData;
    uint8_t*   mFree;
    uint64_t   mSize;
    bool       mManaged;

    /// @brief External memory ctor
    Pool(uint64_t size = 0, void *data = nullptr) : mData((uint8_t*)data), mFree(mData), mSize(size), mManaged(data==nullptr)
    {
        if (mManaged) {
            mData = mFree = static_cast<uint8_t*>(std::malloc(size));
            if (mData == nullptr) {
                throw std::runtime_error("Pool::Pool malloc failed");
            }
        }
    }

    /// @brief Custom destructor
    ~Pool()
    {
        assert(mRegister.empty());
        if (mManaged) {
            std::free(mData);
        }
    }

    /// @brief Disallow copy-construction
    Pool(const Pool&) = delete;

    /// @brief Disallow move-construction
    Pool(const Pool&&) = delete;

    /// @brief Disallow copy assignment operation
    Pool& operator=(const Pool&) = delete;

    /// @brief Disallow move assignment operation
    Pool& operator=(const Pool&&) = delete;

    /// @brief Return the total number of bytes used from this Pool by buffers
    uint64_t usage() const { return static_cast<uint64_t>(mFree - mData); }

    /// @brief Allocate a buffer of the specified size and add it to the register
    void add(HostBuffer *buffer, uint64_t size)
    {
        if (mFree + size > mData + mSize) {
            std::stringstream ss;
            ss << "HostBuffer::Pool: insufficient memory\n"
               << "\tA buffer requested " << size << " bytes from a pool with "
               << mSize <<" bytes of which\n\t" << (mFree-mData)
               << " bytes are used by " << mRegister.size() << " other buffer(s). "
               << "Pool is " << (mManaged ? "internally" : "externally") << " managed.\n";
            //std::cerr << ss.str();
            throw std::runtime_error(ss.str());
        }
        buffer->mSize = size;
        const std::lock_guard<std::mutex> lock(mMutex);
        mRegister.insert(buffer);
        buffer->mData = mFree;
        mFree += size;
    }

    /// @brief Remove the specified buffer from the register
    void remove(HostBuffer *buffer)
    {
        const std::lock_guard<std::mutex> lock(mMutex);
        mRegister.erase(buffer);
    }

    /// @brief Replaces buffer1 with buffer2 in the register
    void replace(HostBuffer *buffer1, HostBuffer *buffer2)
    {
        const std::lock_guard<std::mutex> lock(mMutex);
        mRegister.erase( buffer1);
        mRegister.insert(buffer2);
    }

    /// @brief Reset the register and all its buffers
    void reset()
    {
        for (HostBuffer *buffer : mRegister) {
            buffer->mPool.reset();
            buffer->mSize = 0;
            buffer->mData = nullptr;
        }
        mRegister.clear();
        mFree = mData;
    }

    /// @brief Resize this Pool and update registered buffers as needed. If data is no NULL
    ///        it is used as externally managed memory.
    void resize(uint64_t size, void *data = nullptr)
    {
        const uint64_t memUsage = this->usage();
        if (memUsage > size) {
            throw std::runtime_error("Pool::resize: insufficient memory");
        }
        const bool managed = (data == nullptr);
        if (mManaged && managed && size != mSize) {// managed -> manged
            data = std::realloc(mData, size);// performs both copy and free of mData
        } else if (!mManaged && managed) {// un-managed -> managed
            data = std::malloc(size);
        }
        if (data == nullptr) {
            throw std::runtime_error("Pool::resize: allocation failed");
        } else if (data != mData) {
            if (!(mManaged && managed)) {// no need to copy if managed -> managed
                memcpy(data, mData, memUsage);
            }
            for (HostBuffer *buffer : mRegister) {// update registered buffers
                buffer->mData = static_cast<uint8_t*>(data) + ptrdiff_t(buffer->mData - mData);
            }
            mFree = static_cast<uint8_t*>(data) + memUsage;// update the free pointer
            if (mManaged && !managed) {// only free if managed -> un-managed
                std::free(mData);
            }
            mData = static_cast<uint8_t*>(data);
        }
        mSize    = size;
        mManaged = managed;
    }
    /// @brief Return true is all the memory in this pool is in use.
    bool isFull() const
    {
        assert(mFree <= mData + mSize);
        return mSize > 0 ? mFree == mData + mSize : false;
    }
};// struct HostBuffer::Pool

// --------------------------> Implementation of HostBuffer <------------------------------------

inline HostBuffer::HostBuffer(uint64_t size) : mPool(nullptr), mSize(size), mData(nullptr)
{
    if (size>0) {
        mPool = std::make_shared<Pool>(size);
        mData = mPool->mFree;
        mPool->mRegister.insert(this);
        mPool->mFree += size;
    }
}

inline HostBuffer::HostBuffer(HostBuffer&& other) : mPool(other.mPool), mSize(other.mSize), mData(other.mData)
{
    if (mPool && mSize != 0) {
        mPool->replace(&other, this);
    }
    other.mPool.reset();
    other.mSize = 0;
    other.mData = nullptr;
}

inline void HostBuffer::init(uint64_t bufferSize, void *data)
{
    if (bufferSize == 0) {
        throw std::runtime_error("HostBuffer: invalid buffer size");
    }
    if (mPool) {
        mPool.reset();
    }
    if (!mPool || mPool->mSize != bufferSize) {
        mPool = std::make_shared<Pool>(bufferSize, data);
    }
    mPool->add(this, bufferSize);
}

inline HostBuffer& HostBuffer::operator=(HostBuffer&& other)
{
    if (mPool) {
        mPool->remove(this);
    }
    mPool = other.mPool;
    mSize = other.mSize;
    mData = other.mData;
    if (mPool && mSize != 0) {
        mPool->replace(&other, this);
    }
    other.mPool.reset();
    other.mSize = 0;
    other.mData = nullptr;
    return *this;
}

inline uint64_t HostBuffer::poolSize() const
{
    return mPool ? mPool->mSize : 0u;
}

inline uint64_t HostBuffer::poolUsage() const
{
    return mPool ? mPool->usage(): 0u;
}

inline bool HostBuffer::isManaged() const
{
    return mPool ? mPool->mManaged : false;
}

inline bool HostBuffer::isFull() const
{
    return mPool ? mPool->isFull() : false;
}

inline HostBuffer HostBuffer::createPool(uint64_t poolSize, void *data)
{
    if (poolSize == 0) {
        throw std::runtime_error("HostBuffer: invalid pool size");
    }
    HostBuffer buffer;
    buffer.mPool = std::make_shared<Pool>(poolSize, data);
    // note the buffer is NOT registered by its pool since it is not using its memory
    buffer.mSize = 0;
    buffer.mData = nullptr;
    return buffer;
}

inline HostBuffer HostBuffer::createFull(uint64_t bufferSize, void *data)
{
    if (bufferSize == 0) {
        throw std::runtime_error("HostBuffer: invalid buffer size");
    }
    HostBuffer buffer;
    buffer.mPool = std::make_shared<Pool>(bufferSize, data);
    buffer.mPool->add(&buffer, bufferSize);
    return buffer;
}

inline HostBuffer HostBuffer::create(uint64_t bufferSize, const HostBuffer* pool)
{
    HostBuffer buffer;
    if (pool == nullptr || !pool->mPool) {
        buffer.mPool = std::make_shared<Pool>(bufferSize);
    } else {
       buffer.mPool = pool->mPool;
    }
    buffer.mPool->add(&buffer, bufferSize);
    return buffer;
}

inline void HostBuffer::clear()
{
    if (mPool) {// remove self from the buffer register in the pool
        mPool->remove(this);
    }
    mPool.reset();
    mSize = 0;
    mData = nullptr;
}

inline void HostBuffer::reset()
{
    if (this->size()>0) {
        throw std::runtime_error("HostBuffer: only empty buffers can call reset");
    }
    if (!mPool) {
        throw std::runtime_error("HostBuffer: this buffer contains no pool to reset");
    }
    mPool->reset();
}

inline void HostBuffer::resizePool(uint64_t size, void *data)
{
    if (!mPool) {
        throw std::runtime_error("HostBuffer: this buffer contains no pool to resize");
    }
    mPool->resize(size, data);
}

} // namespace nanovdb

#endif // end of NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED
