// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridChecksum.h

    \author Ken Museth

    \brief Computes a pair of 32bit checksums, og a Grid, by means of Cyclic Redundancy Check (CRC)

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

#include "../NanoVDB.h"
#include "GridHandle.h"
#include "ForEach.h"
#include "NodeManager.h"

namespace nanovdb {

/// @brief List of different modes for computing for a checksum
enum class ChecksumMode : uint32_t { Disable = 0,// no computation
                                     Partial = 1,// fast but approximate
                                     Full = 2,// slow but accurate
                                     Default = 1,// defaults to Partial
                                     End = 3 };// marks the end of the enum list

/// @brief Return the (2 x CRC32) checksum of the specified @a grid
///
/// @param grid Grid from which the checksum is computed.
/// @param mode Defines the mode of computation for the checksum.
template <typename ValueT>
uint64_t checksum(const NanoGrid<ValueT> &grid, ChecksumMode mode = ChecksumMode::Default);

/// @brief Return true if the checksum of the @a grid matches the expected
///        value already encoded into the grid's meta data.
///
/// @param grid Grid whose checksum is validated.
/// @param mode Defines the mode of computation for the checksum.
template <typename ValueT>
bool validateChecksum(const NanoGrid<ValueT> &grid, ChecksumMode mode = ChecksumMode::Default);

/// @brief Updates the checksum of a grid
///
/// @param grid Grid whose checksum will be updated.
/// @param mode Defines the mode of computation for the checksum.
template <typename ValueT>
void updateChecksum(NanoGrid<ValueT> &grid, ChecksumMode mode = ChecksumMode::Default);

/// @brief Return the CRC32 checksum of the raw @a data of @a size
/// @param data The beginning of the raw data.
/// @param size Size of the data to bytes!
inline std::uint_fast32_t crc32(const void *data, size_t size);

/// @brief Return the CRC32 checksum of the content pointed to be the iterator
/// @param begin Beginning of the iterator range
/// @param end End of the iterator range (exclusive)
/// @warning The dereference of the iterator must be convertible to a uint8_t
template <typename IterT>
std::uint_fast32_t crc32(IterT begin, IterT end);

/// @brief Class that computes the Cyclic Redundancy Check (CRC)
class CRC32
{
    using ByteT = std::uint_fast8_t;
    using HashT = std::uint_fast32_t;
    HashT mChecksum;

    static std::array<HashT, 256> INIT()
    {
        HashT n = 0;
        auto kernel = [&n]()->HashT{
            HashT checksum = n++;
            for (int i = 0; i < 8; ++i) checksum = (checksum >> 1) ^ ((checksum & 0x1u) ? HashT{0xEDB88320uL} : 0);
            return checksum;
        };
        std::array<HashT, 256> LUT{};
        std::generate(LUT.begin(), LUT.end(), kernel);
        return LUT;// move semantic should prevent a deep copy
    }

public:

   static const HashT EMPTY = ~HashT{0} & HashT{0xFFFFFFFFuL};// All bits are on

    CRC32() : mChecksum(EMPTY) {}

    void reset() { mChecksum = EMPTY; }

    HashT checksum() const { return HashT{0xFFFFFFFFuL} & ~mChecksum; }

    template <typename IterT>
    void operator()(IterT begin, IterT end)
    {
        static const auto LUT = INIT();// scoped static initialization is thread-safe since C++11
        auto kernel = [](HashT checksum, ByteT value){return LUT[(checksum ^ value) & 0xFFu] ^ (checksum >> 8);};
        mChecksum = std::accumulate(begin, end, mChecksum, kernel);
    }

    void operator()(const void *data, size_t byteSize)
    {
        const ByteT *begin = static_cast<const ByteT*>(data);
        this->operator()<const ByteT*>(begin, begin + byteSize);
    }

    template <typename T>
    void operator()(const T &data) {(*this)(&data, sizeof(T));}
};// CRC32

inline std::uint_fast32_t crc32(const void *data, size_t byteSize)
{
    CRC32 crc;
    crc(data, byteSize);
    return crc.checksum();
}

template <typename IterT>
inline std::uint_fast32_t crc32(IterT begin, IterT end)
{
    CRC32 crc;
    crc(begin, end);
    return crc.checksum();
}

/// @brief Class that encapsulates two CRC32 checksums, one for the Grid, Tree and Root node meta data
///        and one for the remaining grid nodes.
class GridChecksum
{
    union {uint32_t mCRC[2]; uint64_t mChecksum; };

public:

    static const uint64_t EMPTY = (static_cast<uint64_t>(CRC32::EMPTY) << 32) | static_cast<uint64_t>(CRC32::EMPTY);

    GridChecksum() : mCRC{CRC32::EMPTY, CRC32::EMPTY} {}

    GridChecksum(uint32_t head, uint32_t tail) :  mCRC{head, tail} {}

    GridChecksum(uint64_t checksum, ChecksumMode mode = ChecksumMode::Full) : mChecksum{mode == ChecksumMode::Disable ? EMPTY : checksum}
    {
        if (mode == ChecksumMode::Partial) mCRC[1] = CRC32::EMPTY;
    }

    uint64_t checksum() const { return mChecksum; }

    uint32_t crc32(int i) const {assert(i==0 || i==1); return mCRC[i]; }

    bool isFull() const { return mCRC[0] != CRC32::EMPTY && mCRC[1] != CRC32::EMPTY; }

    bool isEmpty() const { return mChecksum == EMPTY; }

    ChecksumMode mode() const
    {
        return mChecksum == EMPTY ? ChecksumMode::Disable :
               mCRC[1] == CRC32::EMPTY ? ChecksumMode::Partial : ChecksumMode::Full;
    }

    template <typename ValueT>
    void operator()(const NanoGrid<ValueT> &grid, ChecksumMode mode = ChecksumMode::Full);

    bool operator==(const GridChecksum &rhs) const {return mChecksum == rhs.mChecksum;}
    bool operator!=(const GridChecksum &rhs) const {return mChecksum != rhs.mChecksum;}
};// GridChecksum

// [GridData][TreeData]---[RootData][ROOT TILES...]---[NodeData<5>]---[NodeData<4>]---[LeafData<3>]---[BLINDMETA...]---[BLIND0]---[BLIND1]---etc.
template <typename ValueT>
void GridChecksum::operator()(const NanoGrid<ValueT> &grid, ChecksumMode mode)
{
    // Validate the assumed memory layout
#if 1
    NANOVDB_ASSERT(NANOVDB_OFFSETOF(GridData, mMagic)    ==  0);
    NANOVDB_ASSERT(NANOVDB_OFFSETOF(GridData, mChecksum) ==  8);
    NANOVDB_ASSERT(NANOVDB_OFFSETOF(GridData, mVersion)  == 16);
#else// the static asserts below generate compiler warnings
    static_assert(offsetof(GridData, mMagic)    ==  0, "Unexpected offset to magic number");
    static_assert(offsetof(GridData, mChecksum) ==  8, "Unexpected offset to checksum");
    static_assert(offsetof(GridData, mVersion)  == 16, "Unexpected offset to version number");
#endif
    static const size_t offset = 16;

    mChecksum = EMPTY;

    if (mode == ChecksumMode::Disable) return;

    const auto &tree = grid.tree();
    const auto &root = tree.root();
    CRC32 crc;

    // process Grid + Tree + Root but exclude mMagic and mChecksum
    const uint8_t *begin = reinterpret_cast<const uint8_t*>(&grid);
    const uint8_t *end = begin + grid.memUsage() + tree.memUsage() + root.memUsage();
    crc(begin + offset, end);

    mCRC[0] = crc.checksum();

    if (mode == ChecksumMode::Partial || tree.isEmpty()) return;

    auto nodeMgrHandle = createNodeManager(grid);
    auto *nodeMgr = nodeMgrHandle.template mgr<ValueT>();
    assert(isValid(nodeMgr));
    const auto nodeCount = tree.nodeCount(0) + tree.nodeCount(1) + tree.nodeCount(2);
    std::vector<std::uint_fast32_t> checksums(nodeCount, 0);

    // process upper internal nodes
    auto kernel2 = [&](const Range1D &r) {
        CRC32 local;
        std::uint_fast32_t *p = checksums.data() + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &node = nodeMgr->upper(static_cast<uint32_t>(i));
            local(node);
            *p++ = local.checksum();
            local.reset();
        }
    };

    // process lower internal nodes
    auto kernel1 = [&](const Range1D &r) {
        CRC32 local;
        std::uint_fast32_t *p = checksums.data() + r.begin() + tree.nodeCount(2);
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &node = nodeMgr->lower(static_cast<uint32_t>(i));
            local(node);
            *p++ = local.checksum();
            local.reset();
        }
    };

    // process leaf nodes
    auto kernel0 = [&](const Range1D &r) {
        CRC32 local;
        std::uint_fast32_t *p = checksums.data() + r.begin() + tree.nodeCount(1) + tree.nodeCount(2);
        for (auto i = r.begin(); i != r.end(); ++i) {
            const auto &leaf = nodeMgr->leaf(static_cast<uint32_t>(i));
            local(leaf);
            *p++ = local.checksum();
            local.reset();
        }
    };

    forEach(0, tree.nodeCount(2), 1, kernel2);
    forEach(0, tree.nodeCount(1), 1, kernel1);
    forEach(0, tree.nodeCount(0), 8, kernel0);

    crc.reset();
    crc(checksums.data(), sizeof(std::uint_fast32_t)*checksums.size() );
    mCRC[1] = crc.checksum();
}// GridChecksum::operator()

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

} // namespace nanovdb

#endif // NANOVDB_GRIDCHECKSUM_H_HAS_BEEN_INCLUDED
