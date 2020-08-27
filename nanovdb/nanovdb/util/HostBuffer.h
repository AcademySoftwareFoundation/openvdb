// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file HostBuffer.h

    \author Ken Museth

    \date January 8, 2020

    \brief HostBuffer - a class for simple buffer allocation and management
*/

#ifndef NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED
#define NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

#include <stdint.h> //    for types like int32_t etc
#include <cstdio> //      for fprintf
#include <cstdlib> //     for std::malloc/std::free

#define checkPtr(ptr, msg) \
    { \
        ptrAssert((ptr), (msg), __FILE__, __LINE__); \
    }

namespace nanovdb {

// ----------------------------> HostBuffer <--------------------------------------

/// @brief Simple memory allocator using host memory.
class HostBuffer
{
    uint64_t mSize; // total number of bytes for the NanoVDB grid.
    uint8_t* mData; // raw buffer for the NanoVDB grid.

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
    HostBuffer(uint64_t size = 0)
        : mSize(0)
        , mData(nullptr)
    {
        this->init(size);
    }
    /// @brief Disallow copy-construction
    HostBuffer(const HostBuffer&) = delete;
    /// @brief Move copy-constructor
    HostBuffer(HostBuffer&& other) noexcept
        : mSize(other.mSize)
        , mData(other.mData)
    {
        other.mSize = 0;
        other.mData = nullptr;
    }
    /// @brief Disallow copy assignment operation
    HostBuffer& operator=(const HostBuffer&) = delete;
    /// @brief Move copy assignment operation
    HostBuffer& operator=(HostBuffer&& other) noexcept
    {
        clear();
        mSize = other.mSize;
        mData = other.mData;
        other.mSize = 0;
        other.mData = nullptr;
        return *this;
    }
    /// @brief Destructor frees memory allocated on the heap
    ~HostBuffer() { this->clear(); };

    void init(uint64_t size);

    /// @brief Retuns a pointer to the raw memory buffer managed by this allocator.
    ///
    /// @warning Note that the pointer can be NULL if the allocator was not initialized!
    const uint8_t* data() const { return mData; }
    uint8_t* data() { return mData; }

    /// @brief Returns the size in bytes of the raw memory buffer managed by this allocator.
    uint64_t size() const { return mSize; }

    /// @brief Returns true if this allocator is empty, i.e. has no allocated memory
    bool empty() const { return mSize == 0; }

    /// @brief De-allocate all memory managed by this allocator and set all pointer to NULL
    void clear();

    static HostBuffer create(uint64_t size, const HostBuffer* context = nullptr);

}; // HostBuffer class

// --------------------------> Implementations below <------------------------------------

inline HostBuffer HostBuffer::create(uint64_t size, const HostBuffer*)
{
    return HostBuffer(size);
}

inline void HostBuffer::init(uint64_t size)
{
    if (size == mSize)
        return;
    if (mSize > 0)
        this->clear();
    if (size == 0)
        return;
    mSize = size;
    mData = static_cast<uint8_t*>(std::malloc(size));
    checkPtr(mData, "failed to allocate host data");
} // HostBuffer::init

inline void HostBuffer::clear()
{
    std::free(mData);
    mData = nullptr;
    mSize = 0;
} // HostBuffer::clear

} // namespace nanovdb

#endif // end of NANOVDB_HOSTBUFFER_H_HAS_BEEN_INCLUDED
