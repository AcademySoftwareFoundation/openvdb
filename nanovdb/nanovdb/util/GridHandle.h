// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridHandle.h

    \author Ken Museth

    \date January 8, 2020

    \brief Defines two classes, a GridRegister the defines the value type (e.g. Double, Float etc)
           of a NanoVDB grid, and a GridHandle and manages the memory of a NanoVDB grid.

    \note  This file has NO dependency on OpenVDB.
*/

#ifndef NANOVDB_GRID_HANDLE_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRID_HANDLE_H_HAS_BEEN_INCLUDED

#include "../NanoVDB.h"// for mapToGridType
#include "HostBuffer.h"

namespace nanovdb {

// --------------------------> GridHandleBase <------------------------------------

class GridHandleBase
{
public:
    virtual ~GridHandleBase() {}

    /// @brief Returns the size in bytes of the raw memory buffer managed by this GridHandle's allocator.
    virtual uint64_t size() const = 0;

    virtual uint8_t*       data() = 0;
    virtual const uint8_t* data() const = 0;

    /// @brief Return true if this handle is empty, i.e. has no allocated memory
    bool empty() const { return size() == 0; }

    /// @brief Return true if this handle contains a grid
    operator bool() const { return !this->empty(); }

    /// @brief Returns a const point to the grid meta data (see definition above).
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    const GridMetaData* gridMetaData() const { return reinterpret_cast<const GridMetaData*>(data()); }

    /// @brief Returns the GridType handled by this instance, and GridType::End if empty
    GridType gridType() const
    {
        const GridMetaData* ptr = this->gridMetaData();
        return ptr ? ptr->gridType() : GridType::End;
    }

    /// @brief Return the number of grids contained in this buffer
    uint32_t gridCount() const
    {
        auto *ptr = this->gridMetaData();
        return ptr ? ptr->gridCount() : 0;
    }
};// GridHandleBase

// --------------------------> GridHandle <------------------------------------

/// @brief This class serves to manage a raw memory buffer of a NanoVDB Grid.
///
/// @note  It is important to note that this class does NOT depend on OpenVDB.
template<typename BufferT = HostBuffer>
class GridHandle : public GridHandleBase
{
    BufferT mBuffer;

    template<typename ValueT>
    const NanoGrid<ValueT>* getGrid(uint32_t n = 0) const;

    template<typename ValueT, typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, const NanoGrid<ValueT>*>::type
    getDeviceGrid(uint32_t n = 0) const;

    template <typename T>
    static T* no_const(const T* ptr) { return const_cast<T*>(ptr); }

public:
    using BufferType = BufferT;

    /// @brief Move constructor from a buffer
    GridHandle(BufferT&& buffer) { mBuffer = std::move(buffer); }
    /// @brief Empty ctor
    GridHandle() = default;
    /// @brief Disallow copy-construction
    GridHandle(const GridHandle&) = delete;
    /// @brief Disallow copy assignment operation
    GridHandle& operator=(const GridHandle&) = delete;
    /// @brief Move copy assignment operation
    GridHandle& operator=(GridHandle&& other) noexcept
    {
        mBuffer = std::move(other.mBuffer);
        return *this;
    }
    /// @brief Move copy-constructor
    GridHandle(GridHandle&& other) noexcept { mBuffer = std::move(other.mBuffer); }
    /// @brief Default destructor
    ~GridHandle() override { reset(); }
    /// @brief clear the buffer
    void reset() { mBuffer.clear(); }

    /// @brief Return a reference to the buffer
    BufferT&       buffer() { return mBuffer; }

    /// @brief Return a const reference to the buffer
    const BufferT& buffer() const { return mBuffer; }

    /// @brief Returns a non-const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    uint8_t* data() override { return mBuffer.data(); }

    /// @brief Returns a const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    const uint8_t* data() const override { return mBuffer.data(); }

    /// @brief Returns the size in bytes of the raw memory buffer managed by this GridHandle's allocator.
    uint64_t size() const override { return mBuffer.size(); }

    /// @brief Returns a const pointer to the @a n'th NanoVDB grid encoded in this GridHandle.
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized, @a n is invalid
    ///          or if the template parameter does not match the specified grid!
    template<typename ValueT>
    const NanoGrid<ValueT>* grid(uint32_t n = 0) const { return this->template getGrid<ValueT>(n); }

    /// @brief Returns a pointer to the @a n'th  NanoVDB grid encoded in this GridHandle.
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized, @a n is invalid
    ///          or if the template parameter does not match the specified grid!
    template<typename ValueT>
    NanoGrid<ValueT>* grid(uint32_t n = 0) { return no_const(this->template getGrid<ValueT>(n)); }

    /// @brief Return a const pointer to the @a n'th grid encoded in this GridHandle on the device, e.g. GPU
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized, @a n is invalid
    ///          or if the template parameter does not match the specified grid!
    template<typename ValueT, typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, const NanoGrid<ValueT>*>::type
    deviceGrid(uint32_t n = 0) const { return this->template getDeviceGrid<ValueT>(n); }

    /// @brief Return a const pointer to the @a n'th grid encoded in this GridHandle on the device, e.g. GPU
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized, @a n is invalid
    ///          or if the template parameter does not match the specified grid!
    template<typename ValueT, typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, NanoGrid<ValueT>*>::type
    deviceGrid(uint32_t n = 0) { return no_const(this->template getDeviceGrid<ValueT>(n)); }

    /// @brief Upload the grid to the device, e.g. from CPU to GPU
    ///
    /// @note This method is only available if the buffer supports devices
    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* stream = nullptr, bool sync = true) { mBuffer.deviceUpload(stream, sync); }

    /// @brief Download the grid to from the device, e.g. from GPU to CPU
    ///
    /// @note This method is only available if the buffer supports devices
    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true) { mBuffer.deviceDownload(stream, sync); }
}; // GridHandle

// --------------------------> Implementation of private methods in GridHandle <------------------------------------

template<typename BufferT>
template<typename ValueT>
inline const NanoGrid<ValueT>* GridHandle<BufferT>::getGrid(uint32_t index) const
{
    using GridT = const NanoGrid<ValueT>;
    auto  *data = mBuffer.data();
    GridT *grid = reinterpret_cast<GridT*>(data);
    if (grid == nullptr || index >= grid->gridCount()) {// un-initialized or index is out of range
        return nullptr;
    }
    while(index != grid->gridIndex()) {
        data += grid->gridSize();
        grid  = reinterpret_cast<GridT*>(data);
    }
    return grid->gridType() == mapToGridType<ValueT>() ? grid : nullptr;
}

template<typename BufferT>
template<typename ValueT, typename U>
inline typename std::enable_if<BufferTraits<U>::hasDeviceDual, const NanoGrid<ValueT>*>::type
GridHandle<BufferT>::getDeviceGrid(uint32_t index) const
{
    using GridT = const NanoGrid<ValueT>;
    auto  *data = mBuffer.data();
    GridT *grid = reinterpret_cast<GridT*>(data);
    if (grid == nullptr || index >= grid->gridCount()) {// un-initialized or index is out of range
        return nullptr;
    }
    auto* dev = mBuffer.deviceData();
    while(index != grid->gridIndex()) {
        data += grid->gridSize();
        dev  += grid->gridSize();
        grid  = reinterpret_cast<GridT*>(data);
    }
    return grid->gridType() == mapToGridType<ValueT>() ? reinterpret_cast<GridT*>(dev) : nullptr;
}

} // namespace nanovdb

#endif // NANOVDB_GRID_HANDLE_H_HAS_BEEN_INCLUDED
