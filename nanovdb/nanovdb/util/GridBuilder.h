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

#include <map>
#include <limits>
#include <atomic>
#include <sstream> // for stringstream
#include <vector>
#include <cstring> // for memcpy

namespace nanovdb {

/// @brief Allows for the construction of NanoVDB grids without any dependecy
template<typename ValueT, typename StatsT = Stats<ValueT>>
class GridBuilder
{
    struct Leaf;
    template<typename ChildT>
    struct Node;
    template<typename ChildT>
    struct Root;
    struct ValueAccessor;

    using SrcNode0 = Leaf;
    using SrcNode1 = Node<SrcNode0>;
    using SrcNode2 = Node<SrcNode1>;
    using SrcRootT = Root<SrcNode2>;

    using DstNode0 = nanovdb::LeafNode<ValueT>; // leaf
    using DstNode1 = nanovdb::InternalNode<DstNode0>; // lower
    using DstNode2 = nanovdb::InternalNode<DstNode1>; // upper
    using DstRootT = nanovdb::RootNode<DstNode2>;
    using DstTreeT = nanovdb::Tree<DstRootT>;
    using DstGridT = nanovdb::Grid<DstTreeT>;

    ValueT                 mDelta; // skip node if: node.max < -mDelta || node.min > mDelta
    SrcRootT               mRoot;
    uint8_t*               mData;
    uint64_t               mBytes[8]; // Byte offsets to from mData to: tree, blindmetadata, root, node2, node1, leafs, blinddata, (total size)
    std::vector<SrcNode0*> mArray0; // leaf nodes
    std::vector<SrcNode1*> mArray1; // lower internal nodes
    std::vector<SrcNode2*> mArray2; // upper internal nodes
    uint64_t               mBlindDataSize;

    template<typename DstNodeT>
    DstNodeT* node() const { return reinterpret_cast<DstNodeT*>(mData + mBytes[5 - DstNodeT::LEVEL]); }

    template<typename DstNodeT>
    typename DstNodeT::DataType* nodeData() const { return reinterpret_cast<typename DstNodeT::DataType*>(mData + mBytes[5 - DstNodeT::LEVEL]); }
    typename DstTreeT::DataType* treeData() const { return reinterpret_cast<typename DstTreeT::DataType*>(mData + mBytes[0]); }
    typename DstGridT::DataType* gridData() const { return reinterpret_cast<typename DstGridT::DataType*>(mData); }
    uint64_t                     gridSize() const { return mBytes[7]; }
    nanovdb::GridBlindMetaData*  blindMetaData() const { return reinterpret_cast<nanovdb::GridBlindMetaData*>(mData + mBytes[1]); }
    uint8_t*                     blindData() const { return reinterpret_cast<uint8_t*>(mData + mBytes[6]); }

    // Below are private methods use to serialize nodes into NanoVDB
    void processLeafs();
    template<typename SrcNodeT, typename DstNodeT>
    void processNodes(std::vector<SrcNodeT*>&);
    void processRoot();
    void processTree();
    void processGrid(const Map&, const std::string&, GridClass);

    template<typename SrcNodeT>
    void update(std::vector<SrcNodeT*>&);

    template<typename T, typename FlagT>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    setFlag(const T&, const T&, FlagT& flag) const { flag &= ~FlagT(1); } // unset first bit

    template<typename T, typename FlagT>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    setFlag(const T& min, const T& max, FlagT& flag) const;

public:
    GridBuilder(ValueT background = ValueT(), uint64_t blindDataSize = 0)
        : mDelta(0)
        , mRoot(background)
        , mData(nullptr)
        , mBlindDataSize(blindDataSize)
    {
    }

    ValueAccessor getAccessor() { return ValueAccessor(mRoot); }

    void sdfToLevelSet();

    void sdfToFog();

    template<typename BufferT = HostBuffer>
    GridHandle<BufferT> getHandle(double             voxelSize = 1.0,
                                  const Vec3d&       gridOrigin = Vec3d(0),
                                  const std::string& name = "",
                                  GridClass          gridClass = GridClass::Unknown,
                                  StatsMode          sMode = StatsMode::Default,
                                  ChecksumMode       mode = ChecksumMode::Default,
                                  const BufferT&     buffer = BufferT());

    template<typename BufferT = HostBuffer>
    GridHandle<BufferT> getHandle(const Map&         map,
                                  const std::string& name = "",
                                  GridClass          gridClass = GridClass::Unknown,
                                  StatsMode          sMode = StatsMode::Default,
                                  ChecksumMode       cMode = ChecksumMode::Default,
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
    /// @param delta Specifies a lower threshold value for rendering (optiona). Typically equals the voxel size
    ///              for level sets and otherwise it's zero.
    template<typename Func>
    void operator()(const Func& func, const CoordBBox& bbox, ValueT delta = ValueT(0));

}; // GridBuilder

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename Func>
void GridBuilder<ValueT, StatsT>::
operator()(const Func& func, const CoordBBox& voxelBBox, ValueT delta)
{
    static_assert(is_same<ValueT, typename std::result_of<Func(const Coord&)>::type>::value, "GridBuilder: mismatched ValueType");
    mDelta = delta; // delta = voxel size for level sets, else 0

    using NodeT = Leaf;
    const CoordBBox nodeBBox(voxelBBox[0] >> NodeT::TOTAL, voxelBBox[1] >> NodeT::TOTAL);
    std::mutex      mutex;
    auto            kernel = [&](const CoordBBox& b) {
        NodeT* node = nullptr;
        for (auto it = b.begin(); it; ++it) {
            Coord           min(*it << NodeT::TOTAL), max(min + Coord(NodeT::DIM - 1));
            const CoordBBox bbox(min.maxComponent(voxelBBox.min()),
                                 max.minComponent(voxelBBox.max()));
            if (node == nullptr) {
                node = new NodeT(bbox[0], mRoot.mBackground, false);
            } else {
                node->mOrigin = bbox[0] & ~NodeT::MASK;
            }
            uint64_t count = 0;
            for (auto ijk = bbox.begin(); ijk; ++ijk) {
                const auto v = func(*ijk);
                if (v == mRoot.mBackground)
                    continue;
                ++count;
                node->setValue(*ijk, v);
            }
            if (count > 0) {
                std::lock_guard<std::mutex> guard(mutex);
                assert(node != nullptr);
                mRoot.addNode(node);
                assert(node == nullptr);
            }
        }
        if (node)
            delete node;
    }; // kernel
    forEach(nodeBBox, kernel);
}

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename SrcNodeT>
void GridBuilder<ValueT, StatsT>::
    update(std::vector<SrcNodeT*>& array)
{
    const uint32_t nodeCount = mRoot.template nodeCount<SrcNodeT>();
    if (nodeCount != uint32_t(array.size())) {
        array.clear();
        array.reserve(nodeCount);
        mRoot.getNodes(array);
    }
} // GridBuilder::update

//================================================================================================

template<typename ValueT, typename StatsT>
void GridBuilder<ValueT, StatsT>::
    sdfToLevelSet()
{
    const ValueT outside = mRoot.mBackground;
    // Note that the bottum-up flood filling is essential
    invoke([&]() { this->update(mArray0); }, [&]() { this->update(mArray1); }, [&]() { this->update(mArray2); });
    forEach(0, mArray0.size(), 8, [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i)
            mArray0[i]->signedFloodFill(outside);
    });
    forEach(0, mArray1.size(), 1, [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i)
            mArray1[i]->signedFloodFill(outside);
    });
    forEach(0, mArray2.size(), 1, [&](const Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i)
            mArray2[i]->signedFloodFill(outside);
    });
    mRoot.signedFloodFill(outside);
} // GridBuilder::sdfToLevelSet

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename BufferT>
GridHandle<BufferT> GridBuilder<ValueT, StatsT>::
    getHandle(double             dx, //voxel size
              const Vec3d&       p0, // origin
              const std::string& name,
              GridClass          gridClass,
              StatsMode          sMode, // mode of computation for the statistics
              ChecksumMode       cMode, // mode of computation for the checksum
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
    return this->getHandle(map, name, gridClass, sMode, cMode, buffer);
} // GridBuilder::getHandle

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename BufferT>
GridHandle<BufferT> GridBuilder<ValueT, StatsT>::
    getHandle(const Map&         map,
              const std::string& name,
              GridClass          gridClass,
              StatsMode          sMode,
              ChecksumMode       cMode,
              const BufferT&     buffer)
{
    if (gridClass == GridClass::LevelSet && !is_floating_point<ValueT>::value)
        throw std::runtime_error("Level sets are expected to be floating point types");
    if (gridClass == GridClass::FogVolume && !is_floating_point<ValueT>::value)
        throw std::runtime_error("Fog volumes are expected to be floating point types");

    invoke([&]() { this->update(mArray0); }, [&]() { this->update(mArray1); }, [&]() { this->update(mArray2); });

    mBytes[0] = DstGridT::memUsage(); // grid
    mBytes[1] = DstTreeT::memUsage(); // tree
    mBytes[2] = nanovdb::GridBlindMetaData::memUsage(mBlindDataSize > 0 ? 1 : 0); // blind meta data
    mBytes[3] = DstRootT::memUsage(uint32_t(mRoot.mTable.size())); // root
    mBytes[4] = mArray2.size() * DstNode2::memUsage(); // upper internal nodes
    mBytes[5] = mArray1.size() * DstNode1::memUsage(); // lower internal nodes
    mBytes[6] = mArray0.size() * DstNode0::memUsage(); // leaf nodes
    mBytes[7] = mBlindDataSize;

    for (int i = 1; i < 8; ++i) {
        mBytes[i] += mBytes[i - 1]; // Byte offsets to: tree, blindmetadata, root, node2, node1, leafs, blinddata, total
    }

    GridHandle<BufferT> handle(BufferT::create(this->gridSize(), &buffer));
    mData = handle.data();
    auto* grid = reinterpret_cast<NanoGrid<ValueT>*>(mData);

    this->processLeafs();

    this->template processNodes<SrcNode1, DstNode1>(mArray1);

    this->template processNodes<SrcNode2, DstNode2>(mArray2);

    this->processRoot();

    this->processTree();

    this->processGrid(map, name, gridClass);

    gridStats(*grid, sMode);

    updateChecksum(*grid, cMode);

    return handle;
} // GridBuilder::getHandle

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename T, typename FlagT>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, StatsT>::
    setFlag(const T& min, const T& max, FlagT& flag) const
{
    if (mDelta > 0 && (min > mDelta || max < -mDelta)) {
        flag |= FlagT(1); // set first bit
    } else {
        flag &= ~FlagT(1); // unset first bit
    }
}

//================================================================================================

template<typename ValueT, typename StatsT>
inline void GridBuilder<ValueT, StatsT>::
    sdfToFog()
{
    this->sdfToLevelSet(); // performs signed flood fill

    const ValueT d = -mRoot.mBackground, w = 1.0f / d;
    auto         op = [&](ValueT& v) -> bool {
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
    forEach(0, mArray0.size(), 8, kernel0);
    forEach(0, mArray1.size(), 1, kernel1);
    forEach(0, mArray2.size(), 1, kernel2);

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
} // GridBuilder::sdfToFog

//================================================================================================

template<typename ValueT, typename StatsT>
void GridBuilder<ValueT, StatsT>::
    processLeafs()
{
    DstNode0* firstLeaf = this->template node<DstNode0>(); // address of first leaf node
    auto      kernel = [&](const Range1D& r) {
        auto* dstLeaf = firstLeaf + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstLeaf) {
            const SrcNode0& srcLeaf = *mArray0[i];
            auto*           data = dstLeaf->data();
            assert(size_t(srcLeaf.mID) == i);
            data->mValueMask = srcLeaf.mValueMask; // copy value mask
            data->mBBoxMin = srcLeaf.mOrigin; // copy origin of node
            data->mFlags = 0u;
            const ValueT* src = srcLeaf.mValues;
            for (ValueT *dst = data->mValues, *end = dst + SrcNode0::SIZE; dst != end; dst += 4, src += 4) {
                dst[0] = src[0]; // copy *all* voxel values in sets of four, i.e. loop-unrolling
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
        }
    };
    forEach(0, mArray0.size(), 8, kernel);
} // GridBuilder::processLeafs

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename SrcNodeT, typename DstNodeT>
void GridBuilder<ValueT, StatsT>::
    processNodes(std::vector<SrcNodeT*>& array)
{
    auto* start = this->template nodeData<DstNodeT>();
    auto  kernel = [&](const Range1D& r) {
        auto* data = start + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++data) {
            SrcNodeT& srcNode = *array[i];
            assert(srcNode.mID == i);
            data->mBBox[0] = srcNode.mOrigin; // copy origin of node
            data->mValueMask = srcNode.mValueMask; // copy value mask
            data->mChildMask = srcNode.mChildMask; // copy child mask
            data->mOffset = array.size() - i;
            auto noneChildMask = srcNode.mChildMask; //copy
            noneChildMask.toggle(); // bits are on for values vs child nodes
            for (auto iter = noneChildMask.beginOn(); iter; ++iter) {
                data->mTable[*iter].value = srcNode.mTable[*iter].value;
            }
            for (auto iter = srcNode.mChildMask.beginOn(); iter; ++iter) {
                data->mTable[*iter].childID = srcNode.mTable[*iter].child->mID;
            }
        }
    };
    forEach(0, array.size(), 4, kernel);
} // GridBuilder::processNodes

//================================================================================================

template<typename ValueT, typename StatsT>
void GridBuilder<ValueT, StatsT>::
    processRoot()
{
    auto& data = *(this->template nodeData<DstRootT>());
    data.mBackground = mRoot.mBackground;
    data.mTileCount = uint32_t(mRoot.mTable.size());
    data.mMinimum = data.mMaximum = data.mBackground;
    data.mBBox = CoordBBox(); // // set to an empty bounding box
    data.mActiveVoxelCount = 0;

    // since openvdb::RootNode internally uses a std::map for child nodes its iterator
    // visits elements in the stored order required by the nanovdb::RootNode
    if (data.mTileCount > 0) {
        uint32_t tileID = 0;
        for (auto iter = mRoot.mTable.begin(); iter != mRoot.mTable.end(); ++iter, ++tileID) {
            auto& dstTile = data.tile(tileID);
            if (auto* srcChild = iter->second.child) {
                dstTile.setChild(srcChild->mOrigin, srcChild->mID);
            } else {
                dstTile.setValue(iter->first, iter->second.state, iter->second.value);
            }
        }
    }
} // GridBuilder::processRoot

//================================================================================================

template<typename ValueT, typename StatsT>
void GridBuilder<ValueT, StatsT>::
    processTree()
{
    const uint64_t count[4] = {mArray0.size(), mArray1.size(), mArray2.size(), 1};
    auto&          data = *this->treeData(); // data for the tree
    for (int i = 0; i < 4; ++i) {
        if (count[i] > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Node count exceeds 32 bit range");
        data.mCount[i] = static_cast<uint32_t>(count[i]);
        data.mBytes[i] = mBytes[5 - i] - mBytes[0]; // offset from the tree to the first node at each tree level
    }
    data.mPFSum[3] = 0;
    for (int i = 2; i >= 0; --i)
        data.mPFSum[i] = data.mPFSum[i + 1] + data.mCount[i + 1]; // reverse prefix sum
} // GridBuilder::processTree

//================================================================================================

template<typename ValueT, typename StatsT>
void GridBuilder<ValueT, StatsT>::
    processGrid(const Map&         map,
                const std::string& name,
                GridClass          gridClass)
{
    auto& data = *this->gridData();
    data.mMagic = NANOVDB_MAGIC_NUMBER;
    data.mChecksum = 0u;
    data.mVersion = Version();
    data.mFlags = 0u;
    data.mGridSize = this->gridSize();
    data.mWorldBBox = BBox<Vec3R>();
    data.mBlindMetadataOffset = mBlindDataSize > 0 ? mBytes[1] : 0;
    data.mBlindMetadataCount = mBlindDataSize > 0 ? 1u : 0u;
    data.mGridClass = gridClass;
    data.mGridType = mapToGridType<ValueT>();
    
    if (!isValid(data.mGridType, data.mGridClass)) {
        std::stringstream ss;
        ss << "Invalid combination of GridType("<<int(data.mGridType)
           << ") and GridClass("<<int(data.mGridClass)<<"). See NanoVDB.h for details!";
        throw std::runtime_error(ss.str());
    }
    
    { // set grid name
        if (name.length() + 1 > GridData::MaxNameSize) {
            std::stringstream ss;
            ss << "Grid name \"" << name << "\" is more then " << nanovdb::GridData::MaxNameSize << " characters";
            throw std::runtime_error(ss.str());
        }
        memcpy(data.mGridName, name.c_str(), name.size() + 1);
    }
    data.mVoxelSize = map.applyMap(Vec3d(1)) - map.applyMap(Vec3d(0));
    data.mMap = map;
} // GridBuilder::processGrid

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename ChildT>
struct GridBuilder<ValueT, StatsT>::Root
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

    Root(const ValueT& background)
        : mBackground(background)
    {
    }
    Root(const Root&) = delete; // disallow copy-construction
    Root(Root&&) = default; // allow move construction
    Root& operator=(const Root&) = delete; // disallow copy assignment
    Root& operator=(Root&&) = default; // allow move assignment

    ~Root() { this->clear(); }

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
                iter->second.child->mID = static_cast<uint32_t>(array.size());
                array.push_back(reinterpret_cast<NodeT*>(iter->second.child));
            } else {
                iter->second.child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        assert(child);
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
}; // GridBuilder::Root

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename ChildT>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, StatsT>::Root<ChildT>::
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

template<typename ValueT, typename StatsT>
template<typename ChildT>
struct GridBuilder<ValueT, StatsT>::
    Node
{
    using ValueType = typename ChildT::ValueType;
    using ChildType = ChildT;
    static constexpr uint32_t LOG2DIM = ChildT::LOG2DIM + 1;
    static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; //dimension in index space
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); //number of tile values (or child pointers)
    static constexpr int32_t  MASK = DIM - 1;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using MaskT = Mask<LOG2DIM>;

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
    Coord    mOrigin;
    MaskT    mValueMask;
    MaskT    mChildMask;
    Tile     mTable[SIZE];
    uint32_t mID;

    Node(const Coord& origin, const ValueT& value, bool state)
        : mOrigin(origin & ~MASK)
        , mValueMask(state)
        , mChildMask()
    {
        for (uint32_t i = 0; i < SIZE; ++i)
            mTable[i].value = value;
    }
    Node(const Node&) = delete; // disallow copy-construction
    Node(Node&&) = delete; // disallow move construction
    Node& operator=(const Node&) = delete; // disallow copy assignment
    Node& operator=(Node&&) = delete; // disallow move assignment
    ~Node()
    {
        for (auto iter = mChildMask.beginOn(); iter; ++iter)
            delete mTable[*iter].child;
    }

    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) +
               (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) +
                ((ijk[2] & MASK) >> ChildT::TOTAL);
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        assert(n < SIZE);
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
        Coord ijk = Node::OffsetToLocalCoord(n);
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
        assert(NodeT::LEVEL < LEVEL);
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
        assert(NodeT::LEVEL < LEVEL);
        for (auto iter = mChildMask.beginOn(); iter; ++iter) {
            if (is_same<NodeT, ChildT>::value) { //resolved at compile-time
                mTable[*iter].child->mID = static_cast<uint32_t>(array.size());
                array.push_back(reinterpret_cast<NodeT*>(mTable[*iter].child));
            } else {
                mTable[*iter].child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        assert(child && (child->mOrigin & ~MASK) == this->mOrigin);
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
}; // GridBuilder::Node

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename ChildT>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, StatsT>::Node<ChildT>::
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

template<typename ValueT, typename StatsT>
struct GridBuilder<ValueT, StatsT>::
    Leaf
{
    using ValueType = ValueT;
    static constexpr uint32_t LOG2DIM = 3;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr int32_t  MASK = DIM - 1; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using NodeMaskType = Mask<LOG2DIM>;
    Coord         mOrigin;
    Mask<LOG2DIM> mValueMask;
    ValueT        mValues[SIZE];
    uint32_t      mID;

    Leaf(const Coord& ijk, const ValueT& value, bool state)
        : mOrigin(ijk & ~MASK)
        , mValueMask(state) //invalid
    {
        ValueT*  target = mValues;
        uint32_t n = SIZE;
        while (n--)
            *target++ = value;
    }
    Leaf(const Leaf&) = delete; // disallow copy-construction
    Leaf(Leaf&&) = delete; // disallow move construction
    Leaf& operator=(const Leaf&) = delete; // disallow copy assignment
    Leaf& operator=(Leaf&&) = delete; // disallow move assignment
    ~Leaf() = default;

    /// @brief Return the linear offset corresponding to the given coordinate
    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return ((ijk[0] & MASK) << (2 * LOG2DIM)) + ((ijk[1] & MASK) << LOG2DIM) + (ijk[2] & MASK);
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        assert(n < SIZE);
        const int32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & MASK);
    }

    void localToGlobalCoord(Coord& ijk) const
    {
        ijk += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = Leaf::OffsetToLocalCoord(n);
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
    void getNodes(std::vector<NodeT*>&) { assert(false); }

    template<typename NodeT>
    void addNode(NodeT*&) {}

    template<typename NodeT>
    uint32_t nodeCount() const
    {
        assert(false);
        return 1;
    }

    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);
    template<typename T>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
        signedFloodFill(T) {} // no-op for none floating point values
}; // Leaf

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, StatsT>::Leaf::
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
} // Leaf::signedFloodFill

//================================================================================================
template<typename ValueT, typename StatsT>
struct GridBuilder<ValueT, StatsT>::
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
        assert(this->isCached<SrcNode0>(ijk));
        return (SrcNode0*)mNode[0];
    }
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
