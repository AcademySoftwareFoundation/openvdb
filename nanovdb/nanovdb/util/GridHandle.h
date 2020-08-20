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

#include "HostBuffer.h"

#include <fstream> // for std::ifstream
#include <iostream> // for std::cerr/cout
#include <string> // for std::string
#include <type_traits> // for std::is_pod

namespace nanovdb {

template<typename BufferT>
struct BufferTraits
{
    static const bool hasDeviceDual = false;
};

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
    operator bool() const { return !empty(); }

    /// @brief Returns a const point to the grid meta data (see definition above).
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    const GridMetaData* gridMetaData() const { return reinterpret_cast<const GridMetaData*>(data()); }
};

// --------------------------> GridHandle <------------------------------------

/// @brief This class serves to manage a raw memory buffer of a NanoVDB Grid.
///
/// @note  It is important to note that is class does NOT depend on OpenVDB.
template<typename BufferT = HostBuffer>
class GridHandle : public GridHandleBase
{
    friend BufferT;

    BufferT mBuffer;

public:
    GridHandle(BufferT&& resources);

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
    GridHandle(GridHandle&& other) noexcept
    {
        mBuffer = std::move(other.mBuffer);
    }
    /// @brief Default destructor
    ~GridHandle() override { reset(); }

    void reset() { mBuffer.clear(); }

    /// @brief Maps from a NanoVDB grid type to GridType
    template<typename GridT>
    static GridType map();

    BufferT&       buffer() { return mBuffer; }
    const BufferT& buffer() const { return mBuffer; }

    /// @brief Returns a non-const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    uint8_t* data() override
    {
        return mBuffer.data();
    }

    /// @brief Returns a const pointer to the data.
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized
    const uint8_t* data() const override
    {
        return mBuffer.data();
    }

    /// @brief Returns the size in bytes of the raw memory buffer managed by this GridHandle's allocator.
    uint64_t size() const override
    {
        return mBuffer.size();
    }

    /// @brief Returns a const pointer to the NanoVDB grid encoded in the GridHandle.
    ///
    /// @warning Note that the return pointer can be NULL if the GridHandle was not initialized or the template
    ///          parameter does not match!
    template<typename ValueT>
    const NanoGrid<ValueT>* grid() const;

    template<typename ValueT, typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, const NanoGrid<ValueT>*>::type
    deviceGrid() const;

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceUpload(void* stream = nullptr, bool sync = true);

    template<typename U = BufferT>
    typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceDownload(void* stream = nullptr, bool sync = true);
}; // GridHandle

// --------------------------> Implementation of GridHandle <------------------------------------

template<typename BufferT>
template<typename GridT>
inline GridType GridHandle<BufferT>::map()
{
    using T = typename GridT::ValueType;
    if (std::is_same<T, float>::value) { // resolved at compiletime
        return GridType::Float;
    } else if (std::is_same<T, double>::value) {
        return GridType::Double;
    } else if (std::is_same<T, int16_t>::value) {
        return GridType::Int16;
    } else if (std::is_same<T, int32_t>::value) {
        return GridType::Int32;
    } else if (std::is_same<T, int64_t>::value) {
        return GridType::Int64;
    } else if (std::is_same<T, Vec3f>::value) {
        return GridType::Vec3f;
    } else if (std::is_same<T, Vec3d>::value) {
        return GridType::Vec3d;
    } else if (std::is_same<T, uint32_t>::value) {
        return GridType::UInt32;
    } else if (std::is_same<T, ValueMask>::value) {
        return GridType::Mask;
    }
    return GridType::Unknown;
}

template<typename BufferT>
GridHandle<BufferT>::GridHandle(BufferT&& resources)
{
    mBuffer = std::move(resources);
}

template<typename BufferT>
template<typename ValueT>
inline const NanoGrid<ValueT>* GridHandle<BufferT>::grid() const
{
    using GridT = const NanoGrid<ValueT>;
    GridT* grid = reinterpret_cast<GridT*>(mBuffer.data());
    return (grid && grid->gridType() == map<GridT>()) ? grid : nullptr;
}

template<typename BufferT>
template<typename ValueT, typename U>
inline typename std::enable_if<BufferTraits<U>::hasDeviceDual, const NanoGrid<ValueT>*>::type
GridHandle<BufferT>::deviceGrid() const
{
    using GridT = const NanoGrid<ValueT>;
    GridT* grid = reinterpret_cast<GridT*>(mBuffer.deviceData());
    return (grid && this->gridMetaData()->gridType() == map<GridT>()) ? grid : nullptr;
}

template<typename BufferT>
template<typename U>
inline typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type GridHandle<BufferT>::deviceUpload(void* stream, bool sync)
{
    mBuffer.deviceUpload(stream, sync);
}

template<typename BufferT>
template<typename U>
inline typename std::enable_if<BufferTraits<U>::hasDeviceDual, void>::type GridHandle<BufferT>::deviceDownload(void* stream, bool sync)
{
    mBuffer.deviceDownload(stream, sync);
}

} // namespace nanovdb

#endif // NANOVDB_GRID_HANDLE_H_HAS_BEEN_INCLUDED
