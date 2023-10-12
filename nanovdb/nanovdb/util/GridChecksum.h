// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridChecksum.h

    \author Ken Museth

    \brief Computes a pair of 32bit checksums, of a Grid, by means of Cyclic Redundancy Check (CRC)

    \details A CRC32 is the 32 bit remainder, or residue, of binary division of a message, by a polynomial.
*/

#ifndef NANOVDB_GRIDCHECKSUM_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRIDCHECKSUM_H_HAS_BEEN_INCLUDED

#include <algorithm>// for std::generate
#include <array>
#include <vector>
#include <cstdint>
#include <cstddef>// offsetof macro
#include <numeric>
#include <type_traits>
#include <memory>// for std::unique_ptr

#include <nanovdb/NanoVDB.h>
#include "ForEach.h"
#include "NodeManager.h"

// Define log of block size for FULL CRC32 computation.
// A value of 12 corresponds to a block size of 4KB (2^12 = 4096).
// Undefine to use old checksum computation
#define NANOVDB_CRC32_LOG2_BLOCK_SIZE 12

namespace nanovdb {

/// @brief List of different modes for computing for a checksum
enum class ChecksumMode : uint32_t { Disable = 0,// no computation
                                     Partial = 1,// fast but approximate
                                     Full = 2,// slow but accurate
                                     Default = 1,// defaults to Partial
                                     End = 3 };// marks the end of the enum list

/// @brief  Return the (2 x CRC32) checksum of the specified @a grid
/// @tparam BuildT Template parameter used to build NanoVDB grid.
/// @param grid Grid from which the checksum is computed.
/// @param mode Defines the mode of computation for the checksum.
/// @return Return the (2 x CRC32) checksum of the specified @a grid
template <typename BuildT>
uint64_t checksum(const NanoGrid<BuildT> &grid, ChecksumMode mode = ChecksumMode::Default);

/// @brief Return true if the checksum of the @a grid matches the expected
///        value already encoded into the grid's meta data.
/// @tparam BuildT Template parameter used to build NanoVDB grid.
/// @param grid Grid whose checksum is validated.
/// @param mode Defines the mode of computation for the checksum.
template <typename BuildT>
bool validateChecksum(const NanoGrid<BuildT> &grid, ChecksumMode mode = ChecksumMode::Default);

/// @brief Updates the checksum of a grid
///
/// @param grid Grid whose checksum will be updated.
/// @param mode Defines the mode of computation for the checksum.
template <typename BuildT>
void updateChecksum(NanoGrid<BuildT> &grid, ChecksumMode mode = ChecksumMode::Default);

namespace crc32 {

/// @brief Initiate single entry in look-up-table for CRC32 computations
/// @param lut pointer of size 256 for look-up-table
/// @param n entry in table (assumed n < 256)
inline __hostdev__ void initLut(uint32_t lut[256], uint32_t n)
{
    uint32_t &cs = lut[n] = n;
    for (int i = 0; i < 8; ++i) cs = (cs >> 1) ^ ((cs & 1) ? 0xEDB88320 : 0);
}

/// @brief Initiate entire look-up-table for CRC32 computations
/// @param lut pointer of size 256 for look-up-table
inline __hostdev__ void initLut(uint32_t lut[256]){for (uint32_t n = 0u; n < 256u; ++n) initLut(lut, n);}

/// @brief Create and initiate entire look-up-table for CRC32 computations
/// @return returns a unique pointer to the lookup table of size 256.
inline std::unique_ptr<uint32_t[]> createLut()
{
    std::unique_ptr<uint32_t[]> lut(new uint32_t[256]);
    initLut(lut.get());
    return lut;
}

/// @brief Compute crc32 checksum of @c data of @c size bytes (without a lookup table))
/// @param data pointer to beginning of data
/// @param size byte size of data
/// @param crc initial value of crc32 checksum
/// @return return crc32 checksum of @c data
inline __hostdev__ uint32_t checksum(const void* data, size_t size, uint32_t crc = 0)
{
    crc = ~crc;
    for (auto *p = (const uint8_t*)data, *q = p + size; p != q; ++p) {
        crc ^= *p;
        for (int j = 0; j < 8; ++j) crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

/// @brief Compute crc32 checksum of data between @c begin and @c end
/// @param begin points to beginning of data
/// @param end points to end of @data, (exclusive)
/// @param crc initial value of crc32 checksum
/// @return return crc32 checksum
inline __hostdev__ uint32_t checksum(const void *begin, const void *end, uint32_t crc = 0)
{
    NANOVDB_ASSERT(begin && end);
    NANOVDB_ASSERT(end >= begin);
    return checksum(begin, (const char*)end - (const char*)begin, crc);
}

/// @brief Compute crc32 checksum of @c data with @c size bytes using a lookup table
/// @param data pointer to begenning of data
/// @param size byte size
/// @param lut pointer to loopup table for accelerated crc32 computation
/// @param crc initial value of the checksum
/// @return crc32 checksum of @c data with @c size bytes
inline __hostdev__ uint32_t checksum(const void *data, size_t size, const uint32_t lut[256], uint32_t crc = 0)
{
    crc = ~crc;
    for (auto *p = (const uint8_t*)data, *q = p + size; p != q; ++p) crc = lut[(crc ^ *p) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

/// @brief Compute crc32 checksum of data between @c begin and @c end using a lookup table
/// @param begin points to beginning of data
/// @param end points to end of @data, (exclusive)
/// @param lut pointer to loopup table for accelerated crc32 computation
/// @param crc initial value of crc32 checksum
/// @return return crc32 checksum
inline __hostdev__ uint32_t checksum(const void *begin, const void *end, const uint32_t lut[256], uint32_t crc = 0)
{
    NANOVDB_ASSERT(begin && end);
    NANOVDB_ASSERT(end >= begin);
    return checksum(begin, (const char*)end - (const char*)begin, lut, crc);
}

}// namespace crc32

/// @brief Class that encapsulates two CRC32 checksums, one for the Grid, Tree and Root node meta data
///        and one for the remaining grid nodes.
class GridChecksum
{
    /// Three types of checksums:
    ///   1) Empty: all 64 bits are on (used to signify no checksum)
    ///   2) Partial: Upper 32 bits are on and not all of lower 32 bits are on (lower 32 bits checksum head of grid)
    ///   3) Full: Not all of the 64 bits are one (lower 32 bits checksum head of grid and upper 32 bits checksum tail of grid)
    union {uint32_t mCRC[2]; uint64_t mChecksum; };// mCRC[0] is checksum of Grid, Tree and Root, and mCRC[1] is checksum of nodes
    static constexpr uint32_t EMPTY32 = ~uint32_t{0};

public:

    static constexpr uint64_t EMPTY = ~uint64_t(0);

    /// @brief default constructor initiates checksum to EMPTY
    GridChecksum() : mCRC{EMPTY32, EMPTY32} {}

    /// @brief Constructor that allows the two 32bit checksums to be initiated explicitly
    /// @param head Initial 32bit CRC checksum of grid, tree and root data
    /// @param tail Initial 32bit CRC checksum of all the nodes and blind data
    GridChecksum(uint32_t head, uint32_t tail) :  mCRC{head, tail} {}

    /// @brief
    /// @param checksum
    /// @param mode
    GridChecksum(uint64_t checksum, ChecksumMode mode = ChecksumMode::Full) : mChecksum{mode == ChecksumMode::Disable ? EMPTY : checksum}
    {
        if (mode == ChecksumMode::Partial) mCRC[1] = EMPTY32;
    }

    /// @brief return the 64 bit checksum of this instance
    uint64_t checksum() const { return mChecksum; }

    /// @brief return 32 bit (crc32) checksum of this instance
    /// @param i index of value 0 or 1 indicated the 32 bit checksum of the head or nodes
    /// @return non-const reference of the i'th 32bit checksum
    uint32_t& checksum(int i) {NANOVDB_ASSERT(i==0 || i==1); return mCRC[i]; }

    /// @brief return 32 bit (crc32) checksum of this instance
    /// @param i index of value 0 or 1 indicated the 32 bit checksum of the head or nodes
    /// @return copy of the i'th 32bit checksum
    uint32_t checksum(int i) const {NANOVDB_ASSERT(i==0 || i==1); return mCRC[i]; }

    /// @brief return true if the 64 bit checksum is partial, i.e. of head only
    bool isPartial() const { return mCRC[0] != EMPTY32 && mCRC[1] == EMPTY32; }

    /// @brief return true if the 64 bit checksum is fill, i.e. of both had and nodes
    bool isFull() const { return mCRC[0] != EMPTY32 && mCRC[1] != EMPTY32; }

    /// @brief return true if the 64 bit checksum is disables (unset)
    bool isEmpty() const { return mChecksum == EMPTY; }

    /// @brief return the mode of the 64 bit checksum
    ChecksumMode mode() const
    {
        return mChecksum == EMPTY ? ChecksumMode::Disable :
               mCRC[1] == EMPTY32 ? ChecksumMode::Partial : ChecksumMode::Full;
    }
#ifdef NANOVDB_CRC32_LOG2_BLOCK_SIZE
    /// @brief compute checksum of @c gridData using a 4KB blocked approach
    /// @param gridData Reference to GridData
    /// @param mode Mode of the checksum computation
    ChecksumMode operator()(const GridData &gridData, ChecksumMode mode = ChecksumMode::Full);
#else
    /// @brief Compute checksum using old (node-based) approach
    /// @tparam ValueT Build type of the grid
    /// @param grid Reference to Grid
    /// @param mode Mode of the checksum computation
    template <typename ValueT>
    void operator()(const NanoGrid<ValueT> &grid, ChecksumMode mode = ChecksumMode::Full);
#endif
    /// @brief return true if the checksums are identical
    /// @param rhs other GridChecksum
    bool operator==(const GridChecksum &rhs) const {return mChecksum == rhs.mChecksum;}

    /// @brief return true if the checksums are not identical
    /// @param rhs other GridChecksum
    bool operator!=(const GridChecksum &rhs) const {return mChecksum != rhs.mChecksum;}
};// GridChecksum

// [GridData][TreeData]---[RootData][ROOT TILES...]---[NodeData<5>]---[NodeData<4>]---[LeafData<3>]---[BLINDMETA...]---[BLIND0]---[BLIND1]---etc.

#ifdef NANOVDB_CRC32_LOG2_BLOCK_SIZE

inline ChecksumMode GridChecksum::operator()(const GridData &gridData, ChecksumMode mode)
{
    mChecksum = EMPTY;

    if (mode == ChecksumMode::Disable) return ChecksumMode::Disable;

    auto lut = crc32::createLut();
    const uint8_t *begin = (const uint8_t*)(&gridData), *mid = gridData.template nodePtr<2>(), *end = begin + gridData.mGridSize;// what about empty grids?
    if (mid == nullptr) {// no (upper) nodes
        if (gridData.mBlindMetadataCount) {
            mid = begin + gridData.mBlindMetadataOffset;// exclude blind data from Partial checksum
        } else {
            mid = end;// no nodes or blind data, so Partial checksum is computed on the entire grid buffer
        }
    }
    mCRC[0] = crc32::checksum(begin + 16, mid, lut.get());// GridData, TreeData. RootData but exclude GridData::mMagic and GridData::mChecksum

    if (mode != ChecksumMode::Full || mid == end) return ChecksumMode::Partial;

    uint64_t size = end - mid;// includes blind data
    const uint64_t blockCount = size >> NANOVDB_CRC32_LOG2_BLOCK_SIZE;// number of 4 KB (4096 byte) blocks
    std::unique_ptr<uint32_t[]> checksums(new uint32_t[blockCount]);
    forEach(0, blockCount, 64, [&](const Range1D &r) {
        uint32_t blockSize = 1 << NANOVDB_CRC32_LOG2_BLOCK_SIZE;
        uint32_t *p = checksums.get() + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i) {
            if (i+1 == blockCount) blockSize += size - (blockCount<<NANOVDB_CRC32_LOG2_BLOCK_SIZE);
            *p++ = crc32::checksum(mid + (i<<NANOVDB_CRC32_LOG2_BLOCK_SIZE), blockSize, lut.get());
        }
    });
    mCRC[1] = crc32::checksum(checksums.get(), sizeof(uint32_t)*blockCount, lut.get());

    return ChecksumMode::Full;
}// GridChecksum::operator(const GridData&, ChecksumMode)

#else// NANOVDB_CRC32_LOG2_BLOCK_SIZE

template <typename ValueT>
void GridChecksum::operator()(const NanoGrid<ValueT> &grid, ChecksumMode mode)
{
    // Validate the assumed memory layout
    static_assert(offsetof(GridData, mMagic)    ==  0, "Unexpected offset to magic number");
    static_assert(offsetof(GridData, mChecksum) ==  8, "Unexpected offset to checksum");
    static_assert(offsetof(GridData, mVersion)  == 16, "Unexpected offset to version number");

    mChecksum = EMPTY;

    if (mode == ChecksumMode::Disable) return;

    auto lut = crc32::createLut();
    const uint8_t *begin = reinterpret_cast<const uint8_t*>(&grid), *mid = grid.template nodePtr<2>();

    mCRC[0] = crc32::checksum(begin + 16, mid, lut.get());// process Grid + Tree + Root but exclude mMagic and mChecksum

    if (mode != ChecksumMode::Full || grid.isEmpty()) return;

    const auto &tree = grid.tree();
    const auto &root = tree.root();
    auto nodeMgrHandle = createNodeManager(grid);
    auto *nodeMgr = nodeMgrHandle.template mgr<ValueT>();
    assert(isValid(nodeMgr));
    const auto nodeCount = tree.nodeCount(0) + tree.nodeCount(1) + tree.nodeCount(2);
    std::vector<uint32_t> checksums(nodeCount, 0);
    // process upper internal nodes
    auto kernel2 = [&](const Range1D &r) {
        uint32_t *p = checksums.data() + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &node = nodeMgr->upper(static_cast<uint32_t>(i));
            *p++ = crc32::checksum(&node, node.memUsage(), lut.get());
        }
    };
    // process lower internal nodes
    auto kernel1 = [&](const Range1D &r) {
        uint32_t *p = checksums.data() + r.begin() + tree.nodeCount(2);
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &node = nodeMgr->lower(static_cast<uint32_t>(i));
            *p++ = crc32::checksum(&node, node.memUsage(), lut.get());
        }
    };
    // process leaf nodes
    auto kernel0 = [&](const Range1D &r) {
        uint32_t *p = checksums.data() + r.begin() + tree.nodeCount(1) + tree.nodeCount(2);
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &leaf = nodeMgr->leaf(static_cast<uint32_t>(i));
            *p++ = crc32::checksum(&leaf, leaf.memUsage(), lut.get());
        }
    };
    forEach(0, tree.nodeCount(2), 1, kernel2);
    forEach(0, tree.nodeCount(1), 1, kernel1);
    forEach(0, tree.nodeCount(0), 8, kernel0);
    mCRC[1] = crc32::checksum(checksums.data(), sizeof(uint32_t)*checksums.size(), lut.get());
}// GridChecksum::operator()

#endif// NANOVDB_CRC32_LOG2_BLOCK_SIZE

template <typename ValueT>
uint64_t checksum(const NanoGrid<ValueT> &grid, ChecksumMode mode)
{
    GridChecksum cs;
    cs(grid, mode);
    return cs.checksum();
}

template <typename ValueT>
bool validateChecksum(const NanoGrid<ValueT> &grid, ChecksumMode mode)
{
    GridChecksum cs1(grid.checksum(), mode), cs2;
    cs2(grid, cs1.mode() );
    return cs1 == cs2;
}

template <typename ValueT>
void updateChecksum(NanoGrid<ValueT> &grid, ChecksumMode mode)
{
    GridChecksum cs;
    cs(grid, mode);
    grid.data()->mChecksum = cs.checksum();
}

inline bool updateChecksum(GridData &gridData, ChecksumMode mode)
{
#ifdef NANOVDB_CRC32_LOG2_BLOCK_SIZE
    GridChecksum cs;
    cs(gridData, mode);
    gridData.mChecksum = cs.checksum();
#else
    if (mode == ChecksumMode::Disable) return false;
    switch (data->mGridType){
        case GridType::Float:
            updateChecksum(*reinterpret_cast<NanoGrid<float>*>(data), mode);
            break;
        case GridType::Double:
            updateChecksum(*reinterpret_cast<NanoGrid<double>*>(data), mode);
            break;
        case GridType::Int16:
            updateChecksum(*reinterpret_cast<NanoGrid<int16_t>*>(data), mode);
            break;
        case GridType::Int32:
            updateChecksum(*reinterpret_cast<NanoGrid<int32_t>*>(data), mode);
            break;
        case GridType::Int64:
            updateChecksum(*reinterpret_cast<NanoGrid<int64_t>*>(data), mode);
            break;
        case GridType::Vec3f:
            updateChecksum(*reinterpret_cast<NanoGrid<Vec3f>*>(data), mode);
            break;
        case GridType::Vec3d:
            updateChecksum(*reinterpret_cast<NanoGrid<Vec3d>*>(data), mode);
            break;
        case GridType::UInt32:
            updateChecksum(*reinterpret_cast<NanoGrid<uint32_t>*>(data), mode);
            break;
        case GridType::Mask:
            updateChecksum(*reinterpret_cast<NanoGrid<ValueMask>*>(data), mode);
            break;
        case GridType::Index:
            updateChecksum(*reinterpret_cast<NanoGrid<ValueIndex>*>(data), mode);
            break;
        case GridType::OnIndex:
            updateChecksum(*reinterpret_cast<NanoGrid<ValueOnIndex>*>(data), mode);
            break;
        case GridType::IndexMask:
            updateChecksum(*reinterpret_cast<NanoGrid<ValueIndexMask>*>(data), mode);
            break;
        case GridType::OnIndexMask:
            updateChecksum(*reinterpret_cast<NanoGrid<ValueOnIndexMask>*>(data), mode);
            break;
        case GridType::Boolean:
            updateChecksum(*reinterpret_cast<NanoGrid<bool>*>(data), mode);
            break;
        case GridType::RGBA8:
            updateChecksum(*reinterpret_cast<NanoGrid<Rgba8>*>(data), mode);
            break;
        case GridType::Fp4:
            updateChecksum(*reinterpret_cast<NanoGrid<Fp4>*>(data), mode);
            break;
        case GridType::Fp8:
            updateChecksum(*reinterpret_cast<NanoGrid<Fp8>*>(data), mode);
            break;
        case GridType::Fp16:
            updateChecksum(*reinterpret_cast<NanoGrid<Fp16>*>(data), mode);
            break;
        case GridType::FpN:
            updateChecksum(*reinterpret_cast<NanoGrid<FpN>*>(data), mode);
            break;
        case GridType::Vec4f:
            updateChecksum(*reinterpret_cast<NanoGrid<Vec4f>*>(data), mode);
            break;
        case GridType::Vec4d:
            updateChecksum(*reinterpret_cast<NanoGrid<Vec4d>*>(data), mode);
            break;
        default: {
            std::stringstream ss;
            ss << "Cannot update checksum for grid of unknown type \"" << toStr(data->mGridType);
            throw std::runtime_error(ss.str() + "\"");
        }
    }// switch
#endif
    return true;
}// updateChecksum(GridData *data, ChecksumMode mode)

/// @brief Preserve the existing mode of the checksum and update it if it's not disabled
/// @param data
/// @return
inline bool updateChecksum(GridData *data)
{
    GridChecksum cs(data->mChecksum);
    const auto mode = cs.mode();
    return updateChecksum(*data, mode);
}// updateChecksum(GridData *data)

/// @brief Updates the ground index and count, as well as the partial checksum if needed
/// @param data Pointer to grid data
/// @param gridIndex New value of the index
/// @param gridCount New value of the grid count
/// @return returns true if the checksum was updated
inline bool updateGridCount(GridData *data, uint32_t gridIndex, uint32_t gridCount)
{
    NANOVDB_ASSERT(gridIndex < gridCount);
    if (data->mGridIndex == gridIndex && data->mGridCount == gridCount) return false;// nothing to update
    data->mGridIndex = gridIndex;
    data->mGridCount = gridCount;
    GridChecksum cs(data->mChecksum);
    if (cs.isEmpty()) return false;// no checksum to update
    updateChecksum(*data, ChecksumMode::Partial);// only update the checksum of the grid since we only modified the GridData
    reinterpret_cast<GridChecksum*>(&(data->mChecksum))->checksum(1) = cs.checksum(1);// copy the old checksum of the tree nodes since it was set to EMPTY during the update
    return true;
}

} // namespace nanovdb

#endif // NANOVDB_GRIDCHECKSUM_H_HAS_BEEN_INCLUDED
