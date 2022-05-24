// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridBuilder.h

    \author Ken Museth

    \date June 26, 2020

    \brief Generates a NanoVDB grid from any volume or function.

    \note This is only intended as a simple tool to generate nanovdb grids without
          any dependency on openvdb.
*/

#ifndef NANOVDB_GRIDBUILDER_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRIDBUILDER_H_HAS_BEEN_INCLUDED

#include "GridHandle.h"
#include "GridStats.h"
#include "GridChecksum.h"
#include "Range.h"
#include "Invoke.h"
#include "ForEach.h"
#include "Reduce.h"
#include "DitherLUT.h"// for nanovdb::DitherLUT

#include <map>
#include <limits>
#include <sstream> // for stringstream
#include <vector>
#include <cstring> // for memcpy

namespace nanovdb {

/// @brief Compression oracle based on absolute difference
class AbsDiff
{
    float mTolerance;// absolute error tolerance
public:
    /// @note The default value of -1 means it's un-initialized!
    AbsDiff(float tolerance = -1.0f) : mTolerance(tolerance) {}
    AbsDiff(const AbsDiff&) = default;
    void  setTolerance(float tolerance) { mTolerance = tolerance; }
    float getTolerance() const { return mTolerance; }
    /// @brief Return true if the approximate value is within the accepted
    ///        absolute error bounds of the exact value.
    ///
    /// @details Required member method
    bool  operator()(float exact, float approx) const
    {
        return Abs(exact - approx) <= mTolerance;
    }
};// AbsDiff

inline std::ostream& operator<<(std::ostream& os, const AbsDiff& diff)
{
    os << "Absolute tolerance: " << diff.getTolerance();
    return os;
}

/// @brief Compression oracle based on relative difference
class RelDiff
{
    float mTolerance;// relative error tolerance
public:
    /// @note The default value of -1 means it's un-initialized!
    RelDiff(float tolerance = -1.0f) : mTolerance(tolerance) {}
    RelDiff(const RelDiff&) = default;
    void  setTolerance(float tolerance) { mTolerance = tolerance; }
    float getTolerance() const { return mTolerance; }
    /// @brief Return true if the approximate value is within the accepted
    ///        relative error bounds of the exact value.
    ///
    /// @details Required member method
    bool  operator()(float exact, float approx) const
    {
        return  Abs(exact - approx)/Max(Abs(exact), Abs(approx)) <= mTolerance;
    }
};// RelDiff

inline std::ostream& operator<<(std::ostream& os, const RelDiff& diff)
{
    os << "Relative tolerance: " << diff.getTolerance();
    return os;
}

/// @brief Allows for the construction of NanoVDB grids without any dependecy
template<typename ValueT, typename BuildT = ValueT, typename StatsT = Stats<ValueT>>
class GridBuilder
{
    struct BuildLeaf;
    template<typename ChildT>
    struct BuildNode;
    template<typename ChildT>
    struct BuildRoot;
    struct ValueAccessor;

    struct Codec {float min, max; uint16_t log2, size;};// used for adaptive bit-rate quantization

    using SrcNode0 = BuildLeaf;
    using SrcNode1 = BuildNode<SrcNode0>;
    using SrcNode2 = BuildNode<SrcNode1>;
    using SrcRootT = BuildRoot<SrcNode2>;

    using DstNode0 = NanoLeaf< BuildT>;// nanovdb::LeafNode<ValueT>; // leaf
    using DstNode1 = NanoLower<BuildT>;// nanovdb::InternalNode<DstNode0>; // lower
    using DstNode2 = NanoUpper<BuildT>;// nanovdb::InternalNode<DstNode1>; // upper
    using DstRootT = NanoRoot< BuildT>;// nanovdb::RootNode<DstNode2>;
    using DstTreeT = NanoTree< BuildT>;
    using DstGridT = NanoGrid< BuildT>;

    ValueT                   mDelta; // skip node if: node.max < -mDelta || node.min > mDelta
    uint8_t*                 mBufferPtr;// pointer to the beginning of the buffer
    uint64_t                 mBufferOffsets[9];//grid, tree, root, upper. lower, leafs, meta data, blind data, buffer size
    int                      mVerbose;
    uint64_t                 mBlindDataSize;
    SrcRootT                 mRoot;// this root supports random write
    std::vector<SrcNode0*>   mArray0; // leaf nodes
    std::vector<SrcNode1*>   mArray1; // lower internal nodes
    std::vector<SrcNode2*>   mArray2; // upper internal nodes
    std::unique_ptr<Codec[]> mCodec;// defines a codec per leaf node
    GridClass                mGridClass;
    StatsMode                mStats;
    ChecksumMode             mChecksum;
    bool                     mDitherOn;

    // Below are private methods use to serialize nodes into NanoVDB
    template< typename OracleT, typename BufferT>
    GridHandle<BufferT> initHandle(const OracleT &oracle, const BufferT& buffer);

    template <typename T, typename OracleT>
    inline typename std::enable_if<!is_same<T, FpN>::value>::type
    compression(uint64_t&, OracleT) {}// no-op

    template <typename T, typename OracleT>
    inline typename std::enable_if<is_same<T, FpN>::value>::type
    compression(uint64_t &offset, OracleT oracle);

    template<typename T>
    typename std::enable_if<!is_same<Fp4,       typename T::BuildType>::value &&
                            !is_same<Fp8,       typename T::BuildType>::value &&
                            !is_same<Fp16,      typename T::BuildType>::value &&
                            !is_same<FpN,       typename T::BuildType>::value>::type
    processLeafs(std::vector<T*>&);

    template<typename T>
    typename std::enable_if<is_same<Fp4,  typename T::BuildType>::value ||
                            is_same<Fp8,  typename T::BuildType>::value ||
                            is_same<Fp16, typename T::BuildType>::value>::type
    processLeafs(std::vector<T*>&);

    template<typename T>
    typename std::enable_if<is_same<FpN, typename T::BuildType>::value>::type
    processLeafs(std::vector<T*>&);

    template<typename SrcNodeT>
    void processNodes(std::vector<SrcNodeT*>&);

    DstRootT* processRoot();

    DstTreeT* processTree();

    DstGridT* processGrid(const Map&, const std::string&);

    template<typename T, typename FlagT>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    setFlag(const T&, const T&, FlagT& flag) const { flag &= ~FlagT(1); } // unset first bit

    template<typename T, typename FlagT>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    setFlag(const T& min, const T& max, FlagT& flag) const;

public:
    GridBuilder(ValueT background = ValueT(),
                GridClass gClass = GridClass::Unknown,
                uint64_t blindDataSize = 0);

    ValueAccessor getAccessor() { return ValueAccessor(mRoot); }

    /// @brief Performs multi-threaded bottum-up signed-distance flood-filling and changes GridClass to LevelSet
    ///
    /// @warning Only call this method once this GridBuilder contains a valid signed distance field
    void sdfToLevelSet();

    /// @brief Performs multi-threaded bottum-up signed-distance flood-filling followed by level-set -> FOG volume
    ///        conversion. It also changes the GridClass to FogVolume
    ///
    /// @warning Only call this method once this GridBuilder contains a valid signed distance field
    void sdfToFog();

    void setVerbose(int mode = 1) { mVerbose = mode; }

    void enableDithering(bool on = true) { mDitherOn = on; }

    void setStats(StatsMode mode = StatsMode::Default) { mStats = mode; }

    void setChecksum(ChecksumMode mode = ChecksumMode::Default) { mChecksum = mode; }

    void setGridClass(GridClass mode = GridClass::Unknown) { mGridClass = mode; }

    /// @brief Return an instance of a GridHandle (invoking move semantics)
    template<typename OracleT = AbsDiff, typename BufferT = HostBuffer>
    GridHandle<BufferT> getHandle(double             voxelSize = 1.0,
                                  const Vec3d&       gridOrigin = Vec3d(0),
                                  const std::string& name = "",
                                  const OracleT&     oracle = OracleT(),
                                  const BufferT&     buffer = BufferT());

    /// @brief Return an instance of a GridHandle (invoking move semantics)
    template<typename OracleT = AbsDiff, typename BufferT = HostBuffer>
    GridHandle<BufferT> getHandle(const Map&         map,
                                  const std::string& name = "",
                                  const OracleT&     oracle = OracleT(),
                                  const BufferT&     buffer = BufferT());

    /// @brief Sets grids values in domain of the @a bbox to those returned by the specified @a func with the
    ///        expected signature [](const Coord&)->ValueT.
    ///
    /// @note If @a func returns a value equal to the brackground value (specified in the constructor) at a
    ///       specific voxel coordinate, then the active state of that coordinate is left off! Else the value
    ///       value is set and the active state is on. This is done to allow for sparse grids to be generated.
    ///
    /// @param func  Functor used to evaluate the grid values in the @a bbox
    /// @param bbox  Coordinate bounding-box over which the grid values will be set.
    /// @param delta Specifies a lower threshold value for rendering (optional). Typically equals the voxel size
    ///              for level sets and otherwise it's zero.
    template<typename Func>
    void operator()(const Func& func, const CoordBBox& bbox, ValueT delta = ValueT(0));

}; // GridBuilder

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
GridBuilder<ValueT, BuildT, StatsT>::
GridBuilder(ValueT background, GridClass gClass, uint64_t blindDataSize)
    : mDelta(0)
    , mVerbose(0)
    , mBlindDataSize(blindDataSize)
    , mRoot(background)
    , mGridClass(gClass)
    , mStats(StatsMode::Default)
    , mChecksum(ChecksumMode::Default)
    , mDitherOn(false)
{
}

template<typename ValueT, typename BuildT, typename StatsT>
template<typename Func>
void GridBuilder<ValueT, BuildT, StatsT>::
operator()(const Func& func, const CoordBBox& voxelBBox, ValueT delta)
{
    static_assert(is_same<ValueT, typename std::result_of<Func(const Coord&)>::type>::value, "GridBuilder: mismatched ValueType");
    mDelta = delta; // delta = voxel size for level sets, else 0

    using LeafT = BuildLeaf;
    const CoordBBox leafBBox(voxelBBox[0] >> LeafT::TOTAL, voxelBBox[1] >> LeafT::TOTAL);
    std::mutex      mutex;
    auto            kernel = [&](const CoordBBox& b) {
        LeafT* leaf = nullptr;
        for (auto it = b.begin(); it; ++it) {
            Coord           min(*it << LeafT::TOTAL), max(min + Coord(LeafT::DIM - 1));
            const CoordBBox bbox(min.maxComponent(voxelBBox.min()),
                                 max.minComponent(voxelBBox.max()));// crop
            if (leaf == nullptr) {
                leaf = new LeafT(bbox[0], mRoot.mBackground, false);
            } else {
                leaf->mOrigin = bbox[0] & ~LeafT::MASK;
                NANOVDB_ASSERT(leaf->mValueMask.isOff());
            }
            leaf->mDstOffset = 0;// no prune
            for (auto ijk = bbox.begin(); ijk; ++ijk) {
                const auto v = func(*ijk);
                if (v == mRoot.mBackground) {// don't insert background values
                    continue;
                }
                leaf->setValue(*ijk, v);
            }
            if (!leaf->mValueMask.isOff()) {// has active values
                if (leaf->mValueMask.isOn()) {// only active values
                    const auto first = leaf->getFirstValue();
                    int n=1;
                    while (n<512) {// 8^3 = 512
                        if (leaf->mValues[n++] != first) break;
                    }
                    if (n == 512) leaf->mDstOffset = 1;// prune below
                }
                std::lock_guard<std::mutex> guard(mutex);
                NANOVDB_ASSERT(leaf != nullptr);
                mRoot.addNode(leaf);
                NANOVDB_ASSERT(leaf == nullptr);
            }
        }// loop over sub-part of leafBBox
        if (leaf) {
            delete leaf;
        }
    }; // kernel
    forEach(leafBBox, kernel);

    // Prune leaf and tile nodes
    for (auto it2 = mRoot.mTable.begin(); it2 != mRoot.mTable.end(); ++it2) {
        if (auto *upper = it2->second.child) {//upper level internal node
            for (auto it1 = upper->mChildMask.beginOn(); it1; ++it1) {
                auto *lower = upper->mTable[*it1].child;// lower level internal node
                for (auto it0 = lower->mChildMask.beginOn(); it0; ++it0) {
                    auto *leaf = lower->mTable[*it0].child;// leaf nodes
                    if (leaf->mDstOffset) {
                        lower->mTable[*it0].value = leaf->getFirstValue();
                        lower->mChildMask.setOff(*it0);
                        lower->mValueMask.setOn(*it0);
                        delete leaf;
                    }
                }// loop over leaf nodes
                if (lower->mChildMask.isOff()) {//only tiles
                    const auto first = lower->getFirstValue();
                    int n=1;
                    while (n < 4096) {// 16^3 = 4096
                        if (lower->mTable[n++].value != first) break;
                    }
                    if (n == 4096) {// identical tile values so prune
                        upper->mTable[*it1].value = first;
                        upper->mChildMask.setOff(*it1);
                        upper->mValueMask.setOn(*it1);
                        delete lower;
                    }
                }
            }// loop over lower internal nodes
            if (upper->mChildMask.isOff()) {//only tiles
                const auto first = upper->getFirstValue();
                int n=1;
                while (n < 32768) {// 32^3 = 32768
                    if (upper->mTable[n++].value != first) break;
                }
                if (n == 32768) {// identical tile values so prune
                    it2->second.value = first;
                    it2->second.state = upper->mValueMask.isOn();
                    it2->second.child = nullptr;
                    delete upper;
                }
            }
        }// is child node of the root
    }// loop over root table
}

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename OracleT, typename BufferT>
GridHandle<BufferT> GridBuilder<ValueT, BuildT, StatsT>::
initHandle(const OracleT &oracle, const BufferT& buffer)
{
    mArray0.clear();
    mArray1.clear();
    mArray2.clear();
    mArray0.reserve(mRoot.template nodeCount<SrcNode0>());
    mArray1.reserve(mRoot.template nodeCount<SrcNode1>());
    mArray2.reserve(mRoot.template nodeCount<SrcNode2>());

    uint64_t offset[3] = {0};
    for (auto it2 = mRoot.mTable.begin(); it2 != mRoot.mTable.end(); ++it2) {
        if (SrcNode2 *upper = it2->second.child) {
            upper->mDstOffset = offset[2];
            mArray2.emplace_back(upper);
            offset[2] += DstNode2::memUsage();
            for (auto it1 = upper->mChildMask.beginOn(); it1; ++it1) {
                SrcNode1 *lower = upper->mTable[*it1].child;
                lower->mDstOffset = offset[1];
                mArray1.emplace_back(lower);
                offset[1] += DstNode1::memUsage();
                for (auto it0 = lower->mChildMask.beginOn(); it0; ++it0) {
                    SrcNode0 *leaf = lower->mTable[*it0].child;
                    leaf->mDstOffset = offset[0];
                    mArray0.emplace_back(leaf);
                    offset[0] += DstNode0::memUsage();
                }// loop over leaf nodes
            }// loop over lower internal nodes
        }// is child node of the root
    }// loop over root table

    this->template compression<BuildT, OracleT>(offset[0], oracle);

    mBufferOffsets[0] = 0;// grid is always stored at the start of the buffer!
    mBufferOffsets[1] = DstGridT::memUsage(); // tree
    mBufferOffsets[2] = DstTreeT::memUsage(); // root
    mBufferOffsets[3] = DstRootT::memUsage(static_cast<uint32_t>(mRoot.mTable.size())); // upper internal nodes
    mBufferOffsets[4] = offset[2]; // lower internal nodes
    mBufferOffsets[5] = offset[1]; // leaf nodes
    mBufferOffsets[6] = offset[0]; // blind meta data
    mBufferOffsets[7] = GridBlindMetaData::memUsage(mBlindDataSize > 0 ? 1 : 0); // blind data
    mBufferOffsets[8] = mBlindDataSize;// end of buffer

    // Compute the prefixed sum
    for (int i = 2; i < 9; ++i) {
        mBufferOffsets[i] += mBufferOffsets[i - 1];
    }

    GridHandle<BufferT> handle(BufferT::create(mBufferOffsets[8], &buffer));
    mBufferPtr = handle.data();
    return handle;
} // GridBuilder::initHandle

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template <typename T, typename OracleT>
inline typename std::enable_if<is_same<T, FpN>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::compression(uint64_t &offset, OracleT oracle)
{
    static_assert(is_same<FpN  , BuildT>::value, "compression: expected BuildT == float");
    static_assert(is_same<float, ValueT>::value, "compression: expected ValueT == float");
    if (is_same<AbsDiff, OracleT>::value && oracle.getTolerance() < 0.0f) {// default tolerance for level set and fog volumes
        if (mGridClass == GridClass::LevelSet) {
            static const float halfWidth = 3.0f;
            oracle.setTolerance(0.1f * mRoot.mBackground / halfWidth);// range of ls: [-3dx; 3dx]
        } else if (mGridClass == GridClass::FogVolume) {
            oracle.setTolerance(0.01f);// range of FOG volumes: [0;1]
        } else {
            oracle.setTolerance(0.0f);
        }
    }

    const size_t size = mArray0.size();
    mCodec.reset(new Codec[size]);

    DitherLUT lut(mDitherOn);
    auto kernel = [&](const Range1D &r) {
        for (auto i=r.begin(); i!=r.end(); ++i) {
            const float *data = mArray0[i]->mValues;
            float min = std::numeric_limits<float>::max(), max = -min;
            for (int j=0; j<512; ++j) {
                float v = data[j];
                if (v<min) min = v;
                if (v>max) max = v;
            }
            mCodec[i].min = min;
            mCodec[i].max = max;
            const float range = max - min;
            uint16_t logBitWidth = 0;// 0,1,2,3,4 => 1,2,4,8,16 bits
            while (range > 0.0f && logBitWidth < 4u) {
                const uint32_t mask = (uint32_t(1) << (uint32_t(1) << logBitWidth)) - 1u;
                const float encode  = mask/range;
                const float decode  = range/mask;
                int j = 0;
                do {
                    const float exact  = data[j];// exact value
                    const uint32_t code = uint32_t(encode*(exact - min) + lut(j));
                    const float approx = code * decode + min;// approximate value
                    j += oracle(exact, approx) ? 1 : 513;
                } while(j < 512);
                if (j == 512) break;
                ++logBitWidth;
            }
            mCodec[i].log2 = logBitWidth;
            mCodec[i].size = DstNode0::DataType::memUsage(1u << logBitWidth);
        }
    };// kernel
    forEach(0, size, 4, kernel);

    if (mVerbose) {
        uint32_t counters[5+1] = {0};
        ++counters[mCodec[0].log2];
        for (size_t i=1; i<size; ++i) {
            ++counters[mCodec[i].log2];
            mArray0[i]->mDstOffset = mArray0[i-1]->mDstOffset + mCodec[i-1].size;
        }
        std::cout << "\n" << oracle << std::endl;
        std::cout << "Dithering: " << (mDitherOn ? "enabled" : "disabled") << std::endl;
        float avg = 0.0f;
        for (uint32_t i=0; i<=5; ++i) {
            if (uint32_t n = counters[i]) {
                avg += n * float(1 << i);
                printf("%2i bits: %6u leaf nodes, i.e. %4.1f%%\n",1<<i, n, 100.0f*n/float(size));
            }
        }
        printf("%4.1f bits per value on average\n", avg/float(size));
    } else {
        for (size_t i=1; i<size; ++i) {
            mArray0[i]->mDstOffset = mArray0[i-1]->mDstOffset + mCodec[i-1].size;
        }
    }
    offset = mArray0[size-1]->mDstOffset + mCodec[size-1].size;
}// GridBuilder::compression

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
void GridBuilder<ValueT, BuildT, StatsT>::
    sdfToLevelSet()
{
    mArray0.clear();
    mArray1.clear();
    mArray2.clear();
    mArray0.reserve(mRoot.template nodeCount<SrcNode0>());
    mArray1.reserve(mRoot.template nodeCount<SrcNode1>());
    mArray2.reserve(mRoot.template nodeCount<SrcNode2>());

    for (auto it2 = mRoot.mTable.begin(); it2 != mRoot.mTable.end(); ++it2) {
        if (SrcNode2 *upper = it2->second.child) {
            mArray2.emplace_back(upper);
            for (auto it1 = upper->mChildMask.beginOn(); it1; ++it1) {
                SrcNode1 *lower = upper->mTable[*it1].child;
                mArray1.emplace_back(lower);
                for (auto it0 = lower->mChildMask.beginOn(); it0; ++it0) {
                    mArray0.emplace_back(lower->mTable[*it0].child);
                }// loop over leaf nodes
            }// loop over lower internal nodes
        }// is child node of the root
    }// loop over root table

    // Note that the bottum-up flood filling is essential
    const ValueT outside = mRoot.mBackground;
    forEach(mArray0, 8, [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i)
            mArray0[i]->signedFloodFill(outside);
    });
    forEach(mArray1, 1, [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i)
            mArray1[i]->signedFloodFill(outside);
    });
    forEach(mArray2, 1, [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i)
            mArray2[i]->signedFloodFill(outside);
    });
    mRoot.signedFloodFill(outside);
    mGridClass = GridClass::LevelSet;
} // GridBuilder::sdfToLevelSet

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename OracleT, typename BufferT>
GridHandle<BufferT> GridBuilder<ValueT, BuildT, StatsT>::
    getHandle(double             dx, //voxel size
              const Vec3d&       p0, // origin
              const std::string& name,
              const OracleT&     oracle,
              const BufferT&     buffer)
{
    if (dx <= 0) {
        throw std::runtime_error("GridBuilder: voxel size is zero or negative");
    }
    Map          map; // affine map
    const double Tx = p0[0], Ty = p0[1], Tz = p0[2];
    const double mat[4][4] = {
        {dx, 0.0, 0.0, 0.0}, // row 0
        {0.0, dx, 0.0, 0.0}, // row 1
        {0.0, 0.0, dx, 0.0}, // row 2
        {Tx, Ty, Tz, 1.0}, // row 3
    };
    const double invMat[4][4] = {
        {1 / dx, 0.0, 0.0, 0.0}, // row 0
        {0.0, 1 / dx, 0.0, 0.0}, // row 1
        {0.0, 0.0, 1 / dx, 0.0}, // row 2
        {-Tx, -Ty, -Tz, 1.0}, // row 3
    };
    map.set(mat, invMat, 1.0);
    return this->getHandle(map, name, oracle, buffer);
} // GridBuilder::getHandle

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template< typename OracleT, typename BufferT>
GridHandle<BufferT> GridBuilder<ValueT, BuildT, StatsT>::
    getHandle(const Map&         map,
              const std::string& name,
              const OracleT&     oracle,
              const BufferT&     buffer)
{
    if (mGridClass == GridClass::LevelSet && !is_floating_point<ValueT>::value) {
        throw std::runtime_error("Level sets are expected to be floating point types");
    } else if (mGridClass == GridClass::FogVolume && !is_floating_point<ValueT>::value) {
        throw std::runtime_error("Fog volumes are expected to be floating point types");
    }

    auto handle = this->template initHandle<OracleT, BufferT>(oracle, buffer);// initialize the arrays of nodes

    this->processLeafs(mArray0);

    this->processNodes(mArray1);

    this->processNodes(mArray2);

    auto *grid = this->processGrid(map, name);

    gridStats(*grid, mStats);

    updateChecksum(*grid, mChecksum);

    return handle;
} // GridBuilder::getHandle

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename T, typename FlagT>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::
    setFlag(const T& min, const T& max, FlagT& flag) const
{
    if (mDelta > 0 && (min > mDelta || max < -mDelta)) {
        flag |= FlagT(1); // set first bit
    } else {
        flag &= ~FlagT(1); // unset first bit
    }
}

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
inline void GridBuilder<ValueT, BuildT, StatsT>::
    sdfToFog()
{
    this->sdfToLevelSet(); // performs signed flood fill

    const ValueT d = -mRoot.mBackground, w = 1.0f / d;
    auto        op = [&](ValueT& v) -> bool {
        if (v > ValueT(0)) {
            v = ValueT(0);
            return false;
        }
        v = v > d ? v * w : ValueT(1);
        return true;
    };
    auto kernel0 = [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            SrcNode0* node = mArray0[i];
            for (uint32_t i = 0; i < SrcNode0::SIZE; ++i)
                node->mValueMask.set(i, op(node->mValues[i]));
        }
    };
    auto kernel1 = [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            SrcNode1* node = mArray1[i];
            for (uint32_t i = 0; i < SrcNode1::SIZE; ++i) {
                if (node->mChildMask.isOn(i)) {
                    SrcNode0* leaf = node->mTable[i].child;
                    if (leaf->mValueMask.isOff()) {
                        node->mTable[i].value = leaf->getFirstValue();
                        node->mChildMask.setOff(i);
                        delete leaf;
                    }
                } else {
                    node->mValueMask.set(i, op(node->mTable[i].value));
                }
            }
        }
    };
    auto kernel2 = [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            SrcNode2* node = mArray2[i];
            for (uint32_t i = 0; i < SrcNode2::SIZE; ++i) {
                if (node->mChildMask.isOn(i)) {
                    SrcNode1* child = node->mTable[i].child;
                    if (child->mChildMask.isOff() && child->mValueMask.isOff()) {
                        node->mTable[i].value = child->getFirstValue();
                        node->mChildMask.setOff(i);
                        delete child;
                    }
                } else {
                    node->mValueMask.set(i, op(node->mTable[i].value));
                }
            }
        }
    };
    forEach(mArray0, 8, kernel0);
    forEach(mArray1, 1, kernel1);
    forEach(mArray2, 1, kernel2);

    for (auto it = mRoot.mTable.begin(); it != mRoot.mTable.end(); ++it) {
        SrcNode2* child = it->second.child;
        if (child == nullptr) {
            it->second.state = op(it->second.value);
        } else if (child->mChildMask.isOff() && child->mValueMask.isOff()) {
            it->second.value = child->getFirstValue();
            it->second.state = false;
            it->second.child = nullptr;
            delete child;
        }
    }
    mGridClass = GridClass::FogVolume;
} // GridBuilder::sdfToFog

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename T>
inline typename std::enable_if<!is_same<Fp4,       typename T::BuildType>::value &&
                               !is_same<Fp8,       typename T::BuildType>::value &&
                               !is_same<Fp16,      typename T::BuildType>::value &&
                               !is_same<FpN,       typename T::BuildType>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::
    processLeafs(std::vector<T*>& srcLeafs)
{
    static_assert(!is_same<bool, ValueT>::value, "Does not yet support bool leafs");
    static_assert(!is_same<ValueMask, ValueT>::value, "Does not yet support mask leafs");
    auto kernel = [&](const Range1D& r) {
        auto *ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *srcLeaf = srcLeafs[i];
            auto *dstLeaf = PtrAdd<DstNode0>(ptr, srcLeaf->mDstOffset);
            auto *data = dstLeaf->data();
            srcLeaf->mDstNode = dstLeaf;
            data->mBBoxMin = srcLeaf->mOrigin; // copy origin of node
            data->mBBoxDif[0] = 0u;
            data->mBBoxDif[1] = 0u;
            data->mBBoxDif[2] = 0u;
            data->mFlags = 0u;
            data->mValueMask = srcLeaf->mValueMask; // copy value mask
            const ValueT* src = srcLeaf->mValues;
            for (ValueT *dst = data->mValues, *end = dst + SrcNode0::SIZE; dst != end; dst += 4, src += 4) {
                dst[0] = src[0]; // copy *all* voxel values in sets of four, i.e. loop-unrolling
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
        }
    };
    forEach(srcLeafs, 8, kernel);
} // GridBuilder::processLeafs<T>

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename T>
inline typename std::enable_if<is_same<Fp4,  typename T::BuildType>::value ||
                               is_same<Fp8,  typename T::BuildType>::value ||
                               is_same<Fp16, typename T::BuildType>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::
    processLeafs(std::vector<T*>& srcLeafs)
{
    static_assert(is_same<float, ValueT>::value, "Expected ValueT == float");
    using ArrayT = typename DstNode0::DataType::ArrayType;
    using FloatT = typename std::conditional<DstNode0::DataType::bitWidth()>=16, double, float>::type;// 16 compression and higher requires double
    static constexpr FloatT UNITS = FloatT((1 << DstNode0::DataType::bitWidth()) - 1);// # of unique non-zero values
    DitherLUT lut(mDitherOn);

    auto kernel = [&](const Range1D& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *srcLeaf = srcLeafs[i];
            auto *dstLeaf = PtrAdd<DstNode0>(ptr, srcLeaf->mDstOffset);
            srcLeaf->mDstNode = dstLeaf;
            auto *data = dstLeaf->data();
            data->mBBoxMin = srcLeaf->mOrigin; // copy origin of node
            data->mBBoxDif[0] = 0u;
            data->mBBoxDif[1] = 0u;
            data->mBBoxDif[2] = 0u;
            data->mFlags = 0u;
            data->mValueMask = srcLeaf->mValueMask; // copy value mask
            const float* src = srcLeaf->mValues;
            // compute extrema values
            float min = std::numeric_limits<float>::max(), max = -min;
            for (int i=0; i<512; ++i) {
                const float v = src[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            data->init(min, max, DstNode0::DataType::bitWidth());
            // perform quantization relative to the values in the curret leaf node
            const FloatT encode = UNITS/(max-min);
            auto *code = reinterpret_cast<ArrayT*>(data->mCode);
            int offset = 0;
            if (is_same<Fp4, BuildT>::value) {// resolved at compile-time
                for (int j=0; j<128; ++j) {
                    auto tmp = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++  = ArrayT(encode * (*src++ - min) + lut(offset++)) << 4 | tmp;
                    tmp      = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++  = ArrayT(encode * (*src++ - min) + lut(offset++)) << 4 | tmp;
                }
            } else {
                for (int j=0; j<128; ++j) {
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                    *code++ = ArrayT(encode * (*src++ - min) + lut(offset++));
                }
            }
        }
    };
    forEach(srcLeafs, 8, kernel);
} // GridBuilder::processLeafs<Fp4, Fp8, Fp16>

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename T>
inline typename std::enable_if<is_same<FpN, typename T::BuildType>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::
    processLeafs(std::vector<T*>& srcLeafs)
{
    static_assert(is_same<float, ValueT>::value, "Expected ValueT == float");

    DitherLUT lut(mDitherOn);
    auto kernel = [&](const Range1D& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5];
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto *srcLeaf = srcLeafs[i];
            auto *dstLeaf = PtrAdd<DstNode0>(ptr, srcLeaf->mDstOffset);
            auto *data = dstLeaf->data();
            data->mBBoxMin = srcLeaf->mOrigin; // copy origin of node
            data->mBBoxDif[0] = 0u;
            data->mBBoxDif[1] = 0u;
            data->mBBoxDif[2] = 0u;
            srcLeaf->mDstNode = dstLeaf;
            const uint8_t logBitWidth = uint8_t(mCodec[i].log2);
            data->mFlags = logBitWidth << 5;// pack logBitWidth into 3 MSB of mFlag
            data->mValueMask = srcLeaf->mValueMask; // copy value mask
            const float* src = srcLeaf->mValues;
            const float min = mCodec[i].min, max = mCodec[i].max;
            data->init(min, max, uint8_t(1) << logBitWidth);
            // perform quantization relative to the values in the curret leaf node
            int offset = 0;
            switch (logBitWidth) {
                case 0u: {// 1 bit
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 1.0f/(max - min);
                    for (int j=0; j<64; ++j) {
                        uint8_t a = 0;
                        for (int k=0; k<8; ++k) {
                            a |= uint8_t(encode * (*src++ - min) + lut(offset++)) << k;
                        }
                        *dst++ = a;
                    }
                }
                break;
                case 1u: {// 2 bits
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 3.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        auto a = uint8_t(encode * (*src++ - min) + lut(offset++));
                        a     |= uint8_t(encode * (*src++ - min) + lut(offset++)) << 2;
                        a     |= uint8_t(encode * (*src++ - min) + lut(offset++)) << 4;
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++)) << 6 | a;
                    }
                }
                break;
                case 2u: {// 4 bits
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 15.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        auto a = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++)) << 4 | a;
                        a      = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++)) << 4 | a;
                    }
                }
                break;
                case 3u: {// 8 bits
                    auto *dst = reinterpret_cast<uint8_t*>(data+1);
                    const float encode = 255.0f/(max - min);
                    for (int j=0; j<128; ++j) {
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint8_t(encode * (*src++ - min) + lut(offset++));
                    }
                }
                break;
                default: {// 16 bits
                    auto *dst = reinterpret_cast<uint16_t*>(data+1);
                    const double encode = 65535.0/(max - min);// note that double is required!
                    for (int j=0; j<128; ++j) {
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                        *dst++ = uint16_t(encode * (*src++ - min) + lut(offset++));
                    }
                }
            }// end switch
        }
    };// kernel
    forEach(srcLeafs, 8, kernel);
} // GridBuilder::processLeafs<FpN>

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename SrcNodeT>
void GridBuilder<ValueT, BuildT, StatsT>::
    processNodes(std::vector<SrcNodeT*>& srcNodes)
{
    using DstNodeT = typename SrcNodeT::NanoNodeT;
    static_assert(DstNodeT::LEVEL == 1 || DstNodeT::LEVEL == 2, "Expected internal node");
    auto  kernel = [&](const Range1D& r) {
        uint8_t* ptr = mBufferPtr + mBufferOffsets[5 - DstNodeT::LEVEL];// 3 or 4
        for (auto i = r.begin(); i != r.end(); ++i) {
            SrcNodeT *srcNode = srcNodes[i];
            DstNodeT *dstNode = PtrAdd<DstNodeT>(ptr, srcNode->mDstOffset);
            auto     *data = dstNode->data();
            srcNode->mDstNode = dstNode;
            data->mBBox[0]   = srcNode->mOrigin; // copy origin of node
            data->mValueMask = srcNode->mValueMask; // copy value mask
            data->mChildMask = srcNode->mChildMask; // copy child mask
            for (uint32_t j = 0; j != SrcNodeT::SIZE; ++j) {
                if (data->mChildMask.isOn(j)) {
                    data->setChild(j, srcNode->mTable[j].child->mDstNode);
                } else
                    data->setValue(j, srcNode->mTable[j].value);
            }
        }
    };
    forEach(srcNodes, 4, kernel);
} // GridBuilder::processNodes

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
NanoRoot<BuildT>* GridBuilder<ValueT, BuildT, StatsT>::processRoot()
{
    auto *dstRoot = reinterpret_cast<DstRootT*>(mBufferPtr + mBufferOffsets[2]);
    auto *data = dstRoot->data();
    data->mTableSize = uint32_t(mRoot.mTable.size());
    data->mMinimum = data->mMaximum = data->mBackground = mRoot.mBackground;
    data->mBBox = CoordBBox(); // // set to an empty bounding box

    uint32_t tileID = 0;
    for (auto iter = mRoot.mTable.begin(); iter != mRoot.mTable.end(); ++iter) {
        auto *dstTile = data->tile(tileID++);
        if (auto* srcChild = iter->second.child) {
            dstTile->setChild(srcChild->mOrigin, srcChild->mDstNode, data);
        } else {
            dstTile->setValue(iter->first, iter->second.state, iter->second.value);
        }
    }
    return dstRoot;
} // GridBuilder::processRoot

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
NanoTree<BuildT>* GridBuilder<ValueT, BuildT, StatsT>::processTree()
{
    auto *dstTree = reinterpret_cast<DstTreeT*>(mBufferPtr + mBufferOffsets[1]);
    auto *data = dstTree->data();
    data->setRoot( this->processRoot() );

    DstNode2 *node2 = mArray2.empty() ? nullptr : reinterpret_cast<DstNode2*>(mBufferPtr + mBufferOffsets[3]);
    data->setFirstNode(node2);

    DstNode1 *node1 = mArray1.empty() ? nullptr : reinterpret_cast<DstNode1*>(mBufferPtr + mBufferOffsets[4]);
    data->setFirstNode(node1);

    DstNode0 *node0 = mArray0.empty() ? nullptr : reinterpret_cast<DstNode0*>(mBufferPtr + mBufferOffsets[5]);
    data->setFirstNode(node0);

    data->mNodeCount[0] = static_cast<uint32_t>(mArray0.size());
    data->mNodeCount[1] = static_cast<uint32_t>(mArray1.size());
    data->mNodeCount[2] = static_cast<uint32_t>(mArray2.size());

    // Count number of active leaf level tiles
    data->mTileCount[0] = reduce(mArray1, uint32_t(0), [&](Range1D &r, uint32_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mArray1[i]->mValueMask.countOn();
        return sum;}, std::plus<uint32_t>());

    // Count number of active lower internal node tiles
    data->mTileCount[1] = reduce(mArray2, uint32_t(0), [&](Range1D &r, uint32_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mArray2[i]->mValueMask.countOn();
        return sum;}, std::plus<uint32_t>());

    // Count number of active upper internal node tiles
    uint32_t sum = 0;
    for (auto &tile : mRoot.mTable) {
        if (tile.second.child==nullptr && tile.second.state) ++sum;
    }
    data->mTileCount[2] = sum;

    // Count number of active voxels
    data->mVoxelCount = reduce(mArray0, uint64_t(0), [&](Range1D &r, uint64_t sum){
        for (auto i=r.begin(); i!=r.end(); ++i) sum += mArray0[i]->mValueMask.countOn();
        return sum;}, std::plus<uint64_t>());

    data->mVoxelCount += data->mTileCount[0]*DstNode0::NUM_VALUES;
    data->mVoxelCount += data->mTileCount[1]*DstNode1::NUM_VALUES;
    data->mVoxelCount += data->mTileCount[2]*DstNode2::NUM_VALUES;

    return dstTree;
} // GridBuilder::processTree

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
NanoGrid<BuildT>* GridBuilder<ValueT, BuildT, StatsT>::
processGrid(const Map&         map,
            const std::string& name)
{
    auto *dstGrid = reinterpret_cast<DstGridT*>(mBufferPtr + mBufferOffsets[0]);
    this->processTree();
    auto* data = dstGrid->data();
    data->mMagic = NANOVDB_MAGIC_NUMBER;
    data->mChecksum = 0u;
    data->mVersion = Version();
    data->mFlags = static_cast<uint32_t>(GridFlags::IsBreadthFirst);
    data->mGridIndex = 0;
    data->mGridCount = 1;
    data->mGridSize = mBufferOffsets[8];
    data->mWorldBBox = BBox<Vec3R>();
    data->mBlindMetadataOffset = 0;
    data->mBlindMetadataCount = 0;
    data->mGridClass = mGridClass;
    data->mGridType = mapToGridType<BuildT>();

    if (!isValid(data->mGridType, data->mGridClass)) {
        std::stringstream ss;
        ss << "Invalid combination of GridType("<<int(data->mGridType)
           << ") and GridClass("<<int(data->mGridClass)<<"). See NanoVDB.h for details!";
        throw std::runtime_error(ss.str());
    }

    strncpy(data->mGridName, name.c_str(), GridData::MaxNameSize-1);
    if (name.length() >= GridData::MaxNameSize) {//  currenlty we don't support long grid names
        std::stringstream ss;
        ss << "Grid name \"" << name << "\" is more then " << GridData::MaxNameSize << " characters";
        throw std::runtime_error(ss.str());
    }

    data->mVoxelSize = map.applyMap(Vec3d(1)) - map.applyMap(Vec3d(0));
    data->mMap = map;

    if (mBlindDataSize>0) {
        auto *metaData = reinterpret_cast<GridBlindMetaData*>(mBufferPtr + mBufferOffsets[6]);
        data->mBlindMetadataOffset = PtrDiff(metaData, dstGrid);
        data->mBlindMetadataCount = 1u;// we currently support only 1 set of blind data
        auto *blindData = reinterpret_cast<char*>(mBufferPtr + mBufferOffsets[7]);
        metaData->setBlindData(blindData);
    }

    return dstGrid;
} // GridBuilder::processGrid

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename ChildT>
struct GridBuilder<ValueT, BuildT, StatsT>::BuildRoot
{
    using ValueType = typename ChildT::ValueType;
    using ChildType = ChildT;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    struct Tile
    {
        Tile(ChildT* c = nullptr)
            : child(c)
        {
        }
        Tile(const ValueT& v, bool s)
            : child(nullptr)
            , value(v)
            , state(s)
        {
        }
        ChildT* child;
        ValueT  value;
        bool    state;
    };
    using MapT = std::map<Coord, Tile>;
    MapT   mTable;
    ValueT mBackground;

    BuildRoot(const ValueT& background)
        : mBackground(background)
    {
    }
    BuildRoot(const BuildRoot&) = delete; // disallow copy-construction
    BuildRoot(BuildRoot&&) = default; // allow move construction
    BuildRoot& operator=(const BuildRoot&) = delete; // disallow copy assignment
    BuildRoot& operator=(BuildRoot&&) = default; // allow move assignment

    ~BuildRoot() { this->clear(); }

    bool empty() const { return mTable.empty(); }

    void clear()
    {
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter)
            delete iter->second.child;
        mTable.clear();
    }

    static Coord CoordToKey(const Coord& ijk) { return ijk & ~ChildT::MASK; }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, AccT& acc) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end())
            return false;
        if (iter->second.child) {
            acc.insert(ijk, iter->second.child);
            return iter->second.child->isActiveAndCache(ijk, acc);
        }
        return iter->second.state;
    }

    const ValueT& getValue(const Coord& ijk) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end()) {
            return mBackground;
        } else if (iter->second.child) {
            return iter->second.child->getValue(ijk);
        } else {
            return iter->second.value;
        }
    }

    template<typename AccT>
    const ValueT& getValueAndCache(const Coord& ijk, AccT& acc) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end())
            return mBackground;
        if (iter->second.child) {
            acc.insert(ijk, iter->second.child);
            return iter->second.child->getValueAndCache(ijk, acc);
        }
        return iter->second.value;
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueT& value, AccT& acc)
    {
        ChildT*     child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto        iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        if (child) {
            acc.insert(ijk, child);
            child->setValueAndCache(ijk, value, acc);
        }
    }

    template<typename NodeT>
    uint32_t nodeCount() const
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Root::getNodes: Invalid type");
        static_assert(NodeT::LEVEL < LEVEL, "Root::getNodes: LEVEL error");
        uint32_t sum = 0;
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
            if (iter->second.child == nullptr)
                continue; // skip tiles
            if (is_same<NodeT, ChildT>::value) { //resolved at compile-time
                ++sum;
            } else {
                sum += iter->second.child->template nodeCount<NodeT>();
            }
        }
        return sum;
    }

    template<typename NodeT>
    void getNodes(std::vector<NodeT*>& array)
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Root::getNodes: Invalid type");
        static_assert(NodeT::LEVEL < LEVEL, "Root::getNodes: LEVEL error");
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
            if (iter->second.child == nullptr)
                continue;
            if (is_same<NodeT, ChildT>::value) { //resolved at compile-time
                array.push_back(reinterpret_cast<NodeT*>(iter->second.child));
            } else {
                iter->second.child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        NANOVDB_ASSERT(child);
        const Coord key = CoordToKey(child->mOrigin);
        auto        iter = mTable.find(key);
        if (iter != mTable.end() && iter->second.child != nullptr) { // existing child node
            delete iter->second.child;
            iter->second.child = child;
        } else {
            mTable[key] = Tile(child);
        }
        child = nullptr;
    }

    template<typename NodeT>
    void addNode(NodeT*& node)
    {
        if (is_same<NodeT, ChildT>::value) { //resolved at compile-time
            this->addChild(reinterpret_cast<ChildT*&>(node));
        } else {
            ChildT*     child = nullptr;
            const Coord key = CoordToKey(node->mOrigin);
            auto        iter = mTable.find(key);
            if (iter == mTable.end()) {
                child = new ChildT(node->mOrigin, mBackground, false);
                mTable[key] = Tile(child);
            } else if (iter->second.child != nullptr) {
                child = iter->second.child;
            } else {
                child = new ChildT(node->mOrigin, iter->second.value, iter->second.state);
                iter->second.child = child;
            }
            child->addNode(node);
        }
    }

    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);

    template<typename T>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    signedFloodFill(T) {} // no-op for none floating point values
}; // GridBuilder::BuildRoot

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename ChildT>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::BuildRoot<ChildT>::
    signedFloodFill(T outside)
{
    std::map<Coord, ChildT*> nodeKeys;
    for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
        if (iter->second.child == nullptr)
            continue;
        nodeKeys.insert(std::pair<Coord, ChildT*>(iter->first, iter->second.child));
    }

    // We employ a simple z-scanline algorithm that inserts inactive tiles with
    // the inside value if they are sandwiched between inside child nodes only!
    auto b = nodeKeys.begin(), e = nodeKeys.end();
    if (b == e)
        return;
    for (auto a = b++; b != e; ++a, ++b) {
        Coord d = b->first - a->first; // delta of neighboring coordinates
        if (d[0] != 0 || d[1] != 0 || d[2] == int(ChildT::DIM))
            continue; // not same z-scanline or neighbors
        const ValueT fill[] = {a->second->getLastValue(), b->second->getFirstValue()};
        if (!(fill[0] < 0) || !(fill[1] < 0))
            continue; // scanline isn't inside
        Coord c = a->first + Coord(0u, 0u, ChildT::DIM);
        for (; c[2] != b->first[2]; c[2] += ChildT::DIM) {
            const Coord key = SrcRootT::CoordToKey(c);
            mTable[key] = typename SrcRootT::Tile(-outside, false); // inactive tile
        }
    }
} // Root::signedFloodFill

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename ChildT>
struct GridBuilder<ValueT, BuildT, StatsT>::
    BuildNode
{
    using ValueType = ValueT;
    using BuildType = BuildT;
    using ChildType = ChildT;
    static constexpr uint32_t LOG2DIM = ChildT::LOG2DIM + 1;
    static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; //dimension in index space
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); //number of tile values (or child pointers)
    static constexpr int32_t  MASK = DIM - 1;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using MaskT = Mask<LOG2DIM>;
    using NanoNodeT = typename NanoNode<BuildT, LEVEL>::Type;

    struct Tile
    {
        Tile(ChildT* c = nullptr)
            : child(c)
        {
        }
        union
        {
            ChildT* child;
            ValueT  value;
        };
    };
    Coord      mOrigin;
    MaskT      mValueMask;
    MaskT      mChildMask;
    Tile       mTable[SIZE];

    union {
        NanoNodeT *mDstNode;
        uint64_t   mDstOffset;
    };

    BuildNode(const Coord& origin, const ValueT& value, bool state)
        : mOrigin(origin & ~MASK)
        , mValueMask(state)
        , mChildMask()
        , mDstOffset(0)
    {
        for (uint32_t i = 0; i < SIZE; ++i) {
            mTable[i].value = value;
        }
    }
    BuildNode(const BuildNode&) = delete; // disallow copy-construction
    BuildNode(BuildNode&&) = delete; // disallow move construction
    BuildNode& operator=(const BuildNode&) = delete; // disallow copy assignment
    BuildNode& operator=(BuildNode&&) = delete; // disallow move assignment
    ~BuildNode()
    {
        for (auto iter = mChildMask.beginOn(); iter; ++iter) {
            delete mTable[*iter].child;
        }
    }

    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) +
               (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) +
                ((ijk[2] & MASK) >> ChildT::TOTAL);
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const uint32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & ((1 << LOG2DIM) - 1));
    }

    void localToGlobalCoord(Coord& ijk) const
    {
        ijk <<= ChildT::TOTAL;
        ijk += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = BuildNode::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            acc.insert(ijk, const_cast<ChildT*>(mTable[n].child));
            return mTable[n].child->isActiveAndCache(ijk, acc);
        }
        return mValueMask.isOn(n);
    }

    ValueT getFirstValue() const { return mChildMask.isOn(0) ? mTable[0].child->getFirstValue() : mTable[0].value; }
    ValueT getLastValue() const { return mChildMask.isOn(SIZE - 1) ? mTable[SIZE - 1].child->getLastValue() : mTable[SIZE - 1].value; }

    const ValueT& getValue(const Coord& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            return mTable[n].child->getValue(ijk);
        }
        return mTable[n].value;
    }

    template<typename AccT>
    const ValueT& getValueAndCache(const Coord& ijk, AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            acc.insert(ijk, const_cast<ChildT*>(mTable[n].child));
            return mTable[n].child->getValueAndCache(ijk, acc);
        }
        return mTable[n].value;
    }

    void setValue(const Coord& ijk, const ValueT& value)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT*        child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        child->setValue(ijk, value);
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueT& value, AccT& acc)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT*        child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        acc.insert(ijk, child);
        child->setValueAndCache(ijk, value, acc);
    }

    template<typename NodeT>
    uint32_t nodeCount() const
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Node::getNodes: Invalid type");
        NANOVDB_ASSERT(NodeT::LEVEL < LEVEL);
        uint32_t sum = 0;
        if (is_same<NodeT, ChildT>::value) { //resolved at compile-time
            sum += mChildMask.countOn();
        } else {
            for (auto iter = mChildMask.beginOn(); iter; ++iter) {
                sum += mTable[*iter].child->template nodeCount<NodeT>();
            }
        }
        return sum;
    }

    template<typename NodeT>
    void getNodes(std::vector<NodeT*>& array)
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Node::getNodes: Invalid type");
        NANOVDB_ASSERT(NodeT::LEVEL < LEVEL);
        for (auto iter = mChildMask.beginOn(); iter; ++iter) {
            if (is_same<NodeT, ChildT>::value) { //resolved at compile-time
                array.push_back(reinterpret_cast<NodeT*>(mTable[*iter].child));
            } else {
                mTable[*iter].child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        NANOVDB_ASSERT(child && (child->mOrigin & ~MASK) == this->mOrigin);
        const uint32_t n = CoordToOffset(child->mOrigin);
        if (mChildMask.isOn(n)) {
            delete mTable[n].child;
        } else {
            mChildMask.setOn(n);
        }
        mTable[n].child = child;
        child = nullptr;
    }

    template<typename NodeT>
    void addNode(NodeT*& node)
    {
        if (is_same<NodeT, ChildT>::value) { //resolved at compile-time
            this->addChild(reinterpret_cast<ChildT*&>(node));
        } else {
            const uint32_t n = CoordToOffset(node->mOrigin);
            ChildT*        child = nullptr;
            if (mChildMask.isOn(n)) {
                child = mTable[n].child;
            } else {
                child = new ChildT(node->mOrigin, mTable[n].value, mValueMask.isOn(n));
                mTable[n].child = child;
                mChildMask.setOn(n);
            }
            child->addNode(node);
        }
    }

    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);
    template<typename T>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
        signedFloodFill(T) {} // no-op for none floating point values
}; // GridBuilder::BuildNode

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename ChildT>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::BuildNode<ChildT>::
    signedFloodFill(T outside)
{
    const uint32_t first = *mChildMask.beginOn();
    if (first < NUM_VALUES) {
        bool xInside = mTable[first].child->getFirstValue() < 0;
        bool yInside = xInside, zInside = xInside;
        for (uint32_t x = 0; x != (1 << LOG2DIM); ++x) {
            const uint32_t x00 = x << (2 * LOG2DIM); // offset for block(x, 0, 0)
            if (mChildMask.isOn(x00)) {
                xInside = mTable[x00].child->getLastValue() < 0;
            }
            yInside = xInside;
            for (uint32_t y = 0; y != (1u << LOG2DIM); ++y) {
                const uint32_t xy0 = x00 + (y << LOG2DIM); // offset for block(x, y, 0)
                if (mChildMask.isOn(xy0))
                    yInside = mTable[xy0].child->getLastValue() < 0;
                zInside = yInside;
                for (uint32_t z = 0; z != (1 << LOG2DIM); ++z) {
                    const uint32_t xyz = xy0 + z; // offset for block(x, y, z)
                    if (mChildMask.isOn(xyz)) {
                        zInside = mTable[xyz].child->getLastValue() < 0;
                    } else {
                        mTable[xyz].value = zInside ? -outside : outside;
                    }
                }
            }
        }
    }
} // Node::signedFloodFill

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
struct GridBuilder<ValueT, BuildT, StatsT>::
    BuildLeaf
{
    using ValueType = ValueT;
    using BuildType = BuildT;
    static constexpr uint32_t LOG2DIM = 3;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr int32_t  MASK = DIM - 1; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using NodeMaskType = Mask<LOG2DIM>;
    using NanoLeafT = typename NanoNode<BuildT, 0>::Type;

    Coord         mOrigin;
    Mask<LOG2DIM> mValueMask;
    ValueT        mValues[SIZE];
    union {
        NanoLeafT *mDstNode;
        uint64_t   mDstOffset;
    };

    BuildLeaf(const Coord& ijk, const ValueT& value, bool state)
        : mOrigin(ijk & ~MASK)
        , mValueMask(state) //invalid
        , mDstOffset(0)
    {
        ValueT*  target = mValues;
        uint32_t n = SIZE;
        while (n--)
            *target++ = value;
    }
    BuildLeaf(const BuildLeaf&) = delete; // disallow copy-construction
    BuildLeaf(BuildLeaf&&) = delete; // disallow move construction
    BuildLeaf& operator=(const BuildLeaf&) = delete; // disallow copy assignment
    BuildLeaf& operator=(BuildLeaf&&) = delete; // disallow move assignment
    ~BuildLeaf() = default;

    /// @brief Return the linear offset corresponding to the given coordinate
    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return ((ijk[0] & MASK) << (2 * LOG2DIM)) + ((ijk[1] & MASK) << LOG2DIM) + (ijk[2] & MASK);
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const int32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & MASK);
    }

    void localToGlobalCoord(Coord& ijk) const
    {
        ijk += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = BuildLeaf::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, const AccT&) const
    {
        return mValueMask.isOn(CoordToOffset(ijk));
    }

    ValueT getFirstValue() const { return mValues[0]; }
    ValueT getLastValue() const { return mValues[SIZE - 1]; }

    const ValueT& getValue(const Coord& ijk) const
    {
        return mValues[CoordToOffset(ijk)];
    }

    template<typename AccT>
    const ValueT& getValueAndCache(const Coord& ijk, const AccT&) const
    {
        return mValues[CoordToOffset(ijk)];
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueT& value, const AccT&)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
        mValues[n] = value;
    }

    void setValue(const Coord& ijk, const ValueT& value)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
        mValues[n] = value;
    }

    template<typename NodeT>
    void getNodes(std::vector<NodeT*>&) { NANOVDB_ASSERT(false); }

    template<typename NodeT>
    void addNode(NodeT*&) {}

    template<typename NodeT>
    uint32_t nodeCount() const
    {
        NANOVDB_ASSERT(false);// should never get called
        return 1;
    }

    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);
    template<typename T>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
        signedFloodFill(T) {} // no-op for none floating point values
}; // BuildLeaf

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, BuildT, StatsT>::BuildLeaf::
    signedFloodFill(T outside)
{
    const uint32_t first = *mValueMask.beginOn();
    if (first < SIZE) {
        bool xInside = mValues[first] < 0, yInside = xInside, zInside = xInside;
        for (uint32_t x = 0; x != DIM; ++x) {
            const uint32_t x00 = x << (2 * LOG2DIM);
            if (mValueMask.isOn(x00))
                xInside = mValues[x00] < 0; // element(x, 0, 0)
            yInside = xInside;
            for (uint32_t y = 0; y != DIM; ++y) {
                const uint32_t xy0 = x00 + (y << LOG2DIM);
                if (mValueMask.isOn(xy0))
                    yInside = mValues[xy0] < 0; // element(x, y, 0)
                zInside = yInside;
                for (uint32_t z = 0; z != (1 << LOG2DIM); ++z) {
                    const uint32_t xyz = xy0 + z; // element(x, y, z)
                    if (mValueMask.isOn(xyz)) {
                        zInside = mValues[xyz] < 0;
                    } else {
                        mValues[xyz] = zInside ? -outside : outside;
                    }
                }
            }
        }
    }
} // BuildLeaf::signedFloodFill

//================================================================================================

template<typename ValueT, typename BuildT, typename StatsT>
struct GridBuilder<ValueT, BuildT, StatsT>::
    ValueAccessor
{
    ValueAccessor(SrcRootT& root)
        : mKeys{Coord(Maximum<int>::value()), Coord(Maximum<int>::value()), Coord(Maximum<int>::value())}
        , mNode{nullptr, nullptr, nullptr, &root}
    {
    }
    template<typename NodeT>
    bool isCached(const Coord& ijk) const
    {
        return (ijk[0] & ~NodeT::MASK) == mKeys[NodeT::LEVEL][0] &&
               (ijk[1] & ~NodeT::MASK) == mKeys[NodeT::LEVEL][1] &&
               (ijk[2] & ~NodeT::MASK) == mKeys[NodeT::LEVEL][2];
    }
    const ValueT& getValue(const Coord& ijk)
    {
        if (this->isCached<SrcNode0>(ijk)) {
            return ((SrcNode0*)mNode[0])->getValueAndCache(ijk, *this);
        } else if (this->isCached<SrcNode1>(ijk)) {
            return ((SrcNode1*)mNode[1])->getValueAndCache(ijk, *this);
        } else if (this->isCached<SrcNode2>(ijk)) {
            return ((SrcNode2*)mNode[2])->getValueAndCache(ijk, *this);
        }
        return ((SrcRootT*)mNode[3])->getValueAndCache(ijk, *this);
    }
    /// @brief Sets value in a leaf node and returns it.
    SrcNode0* setValue(const Coord& ijk, const ValueT& value)
    {
        if (this->isCached<SrcNode0>(ijk)) {
            ((SrcNode0*)mNode[0])->setValueAndCache(ijk, value, *this);
        } else if (this->isCached<SrcNode1>(ijk)) {
            ((SrcNode1*)mNode[1])->setValueAndCache(ijk, value, *this);
        } else if (this->isCached<SrcNode2>(ijk)) {
            ((SrcNode2*)mNode[2])->setValueAndCache(ijk, value, *this);
        } else {
            ((SrcRootT*)mNode[3])->setValueAndCache(ijk, value, *this);
        }
        NANOVDB_ASSERT(this->isCached<SrcNode0>(ijk));
        return (SrcNode0*)mNode[0];
    }
    bool isActive(const Coord& ijk)
    {
        if (this->isCached<SrcNode0>(ijk)) {
            return ((SrcNode0*)mNode[0])->isActiveAndCache(ijk, *this);
        } else if (this->isCached<SrcNode1>(ijk)) {
            return ((SrcNode1*)mNode[1])->isActiveAndCache(ijk, *this);
        } else if (this->isCached<SrcNode2>(ijk)) {
            return ((SrcNode2*)mNode[2])->isActiveAndCache(ijk, *this);
        }
        return ((SrcRootT*)mNode[3])->isActiveAndCache(ijk, *this);
    }
    bool isValueOn(const Coord& ijk) { return this->isActive(ijk); }
    template<typename NodeT>
    void insert(const Coord& ijk, NodeT* node)
    {
        mKeys[NodeT::LEVEL] = ijk & ~NodeT::MASK;
        mNode[NodeT::LEVEL] = node;
    }
    Coord mKeys[3];
    void* mNode[4];
}; // ValueAccessor

} // namespace nanovdb

#endif // NANOVDB_GRIDBUILDER_H_HAS_BEEN_INCLUDED
