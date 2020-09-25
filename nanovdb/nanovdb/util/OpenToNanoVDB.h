// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file OpenToNanoVDB.h

    \author Ken Museth

    \date January 8, 2020

    \brief This class will serialize an OpenVDB grid into a NanoVDB grid.
*/

#include <nanovdb/util/GridHandle.h> // manages and streams the raw memory buffer of a NanoVDB grid.
#include <nanovdb/util/GridChecksum.h>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/util/CpuTimer.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>
#include <tbb/atomic.h>

#include <type_traits>

#ifndef NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED

namespace nanovdb {

/// @brief Forward declaration of free-standing function that converts an OpenVDB GridBase into a NanoVDB GridHandle
template<typename BufferT = HostBuffer>
GridHandle<BufferT>
openToNanoVDB(const openvdb::GridBase::Ptr& base, bool mortonSort = false, int verbose = 0, ChecksumMode mode = ChecksumMode::Default);

/// @brief Forward declaration of free-standing function that converts a typed OpenVDB Grid into a NanoVDB GridHandle
template<typename BufferT = HostBuffer, typename SrcTreeT = openvdb::FloatTree>
GridHandle<BufferT>
openToNanoVDB(const openvdb::Grid<SrcTreeT>& grid, bool mortonSort = false, int verbose = 0, ChecksumMode mode = ChecksumMode::Default);

namespace { // unnamed namespace

/// @brief This class will openToNanoVDB an OpenVDB grid into a NanoVDB grid managed by a GridHandle.
template<typename SrcTreeT, typename BufferT = HostBuffer>
class OpenToNanoVDB
{
    struct BlindMetaData; // forward decleration
    using ValueT = typename SrcTreeT::ValueType;
    ValueT                  mDelta; // skip node if: node.max < -mDelta || node.min > mDelta
    uint8_t*                mData;
    uint64_t                mBytes[8]; // Byte offsets to from mData to: tree, blindmetadata, root, node2, node1, leafs, blinddata, (total size)
    tbb::atomic<uint64_t>   mActiveVoxelCount;
    std::set<BlindMetaData> mBlindMetaData; // sorted accoring to index

public:
    /// @brief Construction from an existing const OpenVDB Grid.
    OpenToNanoVDB() : mData(nullptr)
    {
    }

    /// @brief Return a shared pointer to a NanoVDB grid constructed from the specified OpneVDB grid
    GridHandle<BufferT> operator()(const openvdb::Grid<SrcTreeT>& grid, bool fullChecksum = false, int verbose = 0, ChecksumMode mode = ChecksumMode::Default, const BufferT& allocator = BufferT());

private:
    static_assert(SrcTreeT::DEPTH == 4, "Converter assumes an OpenVDB tree of depth 4 (which is the default configuration)");
    using CoordT = openvdb::Coord;
    using BBoxT = nanovdb::BBox<CoordT>;

    using SrcGridT = openvdb::Grid<SrcTreeT>;
    using SrcRootT = typename SrcTreeT::RootNodeType; // OpenVDB root node
    using SrcNode2 = typename SrcRootT::ChildNodeType; // upper OpenVDB internal node
    using SrcNode1 = typename SrcNode2::ChildNodeType; // lower OpenVDB internal node
    using SrcNode0 = typename SrcNode1::ChildNodeType; // OpenVDB leaf node

    using DstNode0 = nanovdb::LeafNode<ValueT, CoordT, openvdb::util::NodeMask, SrcNode0::LOG2DIM>; // leaf
    using DstNode1 = nanovdb::InternalNode<DstNode0, SrcNode1::LOG2DIM>; // lower
    using DstNode2 = nanovdb::InternalNode<DstNode1, SrcNode2::LOG2DIM>; // upper
    using DstRootT = nanovdb::RootNode<DstNode2>;
    using DstTreeT = nanovdb::Tree<DstRootT>;
    using DstGridT = nanovdb::Grid<DstTreeT>;

    /// @brief Private method to concurrently process the leaf nodes
    void processLeafs(std::vector<const SrcNode0*>& array, int32_t* x0);

    /// @brief Private method to concurrently process the internal nodes
    template<typename SrcNode, typename DstNode>
    void processInternals(std::vector<const SrcNode*>& array, int32_t* x0, const int32_t* childX);

    /// @brief Private method to process the root node
    void processRoot(const SrcRootT& srcRoot, std::vector<const SrcNode2*>& array, const int32_t* childX);

    // @brief Private method to process the tree
    void processTree(const uint64_t* count);

    /// @brief Private method to process the grid
    void processGrid(const SrcGridT& srcGrid);

    /// @brief Private method to post-process the grid
    void postProcessGrid(ChecksumMode mode);

    template<typename LeafT>
    uint64_t pointCount(std::vector<const LeafT*>&);

    template<typename LeafT>
    typename std::enable_if<!(std::is_same<typename LeafT::ValueType, openvdb::PointIndex32>::value ||
                              std::is_same<typename LeafT::ValueType, openvdb::PointDataIndex32>::value)>::type
    preProcessPoints(std::vector<const LeafT*>&) {}

    template<typename LeafT>
    typename std::enable_if<!(std::is_same<typename LeafT::ValueType, openvdb::PointIndex32>::value ||
                              std::is_same<typename LeafT::ValueType, openvdb::PointDataIndex32>::value)>::type
    postProcessPoints(std::vector<const LeafT*>&) {}

    template<typename LeafT>
    typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointIndex32>::value>::type
    preProcessPoints(std::vector<const LeafT*>& array);

    template<typename LeafT>
    typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointDataIndex32>::value>::type
    preProcessPoints(std::vector<const LeafT*>& array);

    template<typename LeafT>
    typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointIndex32>::value>::type
    postProcessPoints(std::vector<const LeafT*>& array);

    template<typename LeafT>
    typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointDataIndex32>::value>::type
    postProcessPoints(std::vector<const LeafT*>& array);

    /// @brief Private methods to access points to data
    template<typename DstNodeT>
    typename DstNodeT::DataType* nodeData() const { return reinterpret_cast<typename DstNodeT::DataType*>(mData + mBytes[5 - DstNodeT::LEVEL]); }
    typename DstTreeT::DataType* treeData() const { return reinterpret_cast<typename DstTreeT::DataType*>(mData + mBytes[0]); }
    typename DstGridT::DataType* gridData() const { return reinterpret_cast<typename DstGridT::DataType*>(mData); }
    uint64_t gridSize() const { return mBytes[7]; }
    nanovdb::GridBlindMetaData* blindMetaData() const { return reinterpret_cast<nanovdb::GridBlindMetaData*>(mData + mBytes[1]); }
    uint8_t* blindData() const { return reinterpret_cast<uint8_t*>(mData + mBytes[6]); }

    /// @brief Private method used to cache the x compoment of a Coord into x and
    //         encode uint32_t id into the x component despite it being of type const int32_t.
    static void cache(int32_t& x, const CoordT& ijk, uint32_t id)
    {
        x = ijk[0];
        reinterpret_cast<uint32_t&>(const_cast<CoordT&>(ijk)[0]) = id;
    }

    template<typename T, typename FlagT>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    setFlag(const T&, const T&, FlagT& flag) const { flag &= ~FlagT(1); } // unset first bit

    template<typename T, typename FlagT>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    setFlag(const T& min, const T& max, FlagT& flag) const;

    /// @brief Private method to sorts the nodes in the specified array using the provided key function
    template<typename NodeT, typename KeyT>
    void sortNodes(std::vector<const NodeT*>& array, KeyT key);

}; // OpenToNanoVDB class

template<typename SrcTreeT, typename BufferT>
template<typename LeafT>
uint64_t OpenToNanoVDB<SrcTreeT, BufferT>::pointCount(std::vector<const LeafT*>& array)
{
    tbb::atomic<uint64_t> pointCount = 0;
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, array.size(), 16),
                      [&](const tbb::blocked_range<uint64_t>& r) {
                          uint64_t sum = 0;
                          for (auto i = r.begin(); i != r.end(); ++i)
                              sum += array[i]->getLastValue();
                          pointCount += sum;
                      });
    return pointCount;
}

template<typename SrcTreeT, typename BufferT>
struct OpenToNanoVDB<SrcTreeT, BufferT>::BlindMetaData
{
    BlindMetaData(const std::string& n, const std::string& t, size_t i, size_t c, size_t s)
        : name(n)
        , typeName(t)
        , index(i)
        , count(c)
        , size(nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(c * s))
    {
    }
    const std::string name, typeName;
    const size_t      index, count, size;
    bool              operator<(const BlindMetaData& other) const { return index < other.index; } // required by std::set
}; // OpenToNanoVDB::BlindMetaData

template<typename SrcTreeT, typename BufferT>
template<typename LeafT>
inline typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::preProcessPoints(std::vector<const LeafT*>& array)
{
    const uint64_t count = this->pointCount(array);
    mBlindMetaData.clear();
    if (count == 0)
        return;
    mBlindMetaData.emplace("index", "uint32", 0, count, sizeof(uint32_t));
}

template<typename SrcTreeT, typename BufferT>
template<typename LeafT>
inline typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointDataIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::preProcessPoints(std::vector<const LeafT*>& array)
{
    const uint64_t count = this->pointCount(array);
    mBlindMetaData.clear();
    if (count == 0)
        return;
    const auto& attributeSet = array.front()->attributeSet();
    const auto& descriptor = attributeSet.descriptor();
    const auto& nameMap = descriptor.map();
    for (auto it = nameMap.begin(); it != nameMap.end(); ++it) {
        const size_t index = it->second;
        auto&        attArray = array.front()->constAttributeArray(index);
        mBlindMetaData.emplace(it->first, descriptor.valueType(index), index, count, attArray.valueTypeSize());
    }
}

template<typename SrcTreeT, typename BufferT>
inline void
OpenToNanoVDB<SrcTreeT, BufferT>::postProcessGrid(ChecksumMode mode)
{
    auto& data = *this->gridData();
    data.mChecksum = nanovdb::checksum( *reinterpret_cast<const NanoGrid<ValueT>*>(mData), mode );
}

template<typename SrcTreeT, typename BufferT>
template<typename LeafT>
inline typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::postProcessPoints(std::vector<const LeafT*>& array)
{
    if (mBlindMetaData.empty())
        return;
    assert(mBlindMetaData.size() == 1);

    const uint32_t leafCount = static_cast<uint32_t>(array.size());
    auto*          data = this->template nodeData<DstNode0>();

    data[0].mValueMin = 0; // start of prefix sum
    for (uint32_t i = 1; i < leafCount; ++i)
        data[i].mValueMin = data[i - 1].mValueMin + data[i - 1].mValueMax;

    // write point offsets as blind meta data
    auto b = *mBlindMetaData.cbegin();
    assert(b.count == data[leafCount - 1].mValueMin + data[leafCount - 1].mValueMax);
    assert(b.name == "index" && b.typeName == "uint32");
    auto& meta = const_cast<nanovdb::GridBlindMetaData&>(this->gridData()->blindMetaData(0));
    meta.mByteOffset = uintptr_t(this->blindData()) - uintptr_t(this->gridData()); // offset from Grid to blind data;
    meta.mElementCount = b.count;
    meta.mFlags = 0;
    meta.mSemantic = GridBlindDataSemantic::Unknown;
    meta.mDataClass = GridBlindDataClass::IndexArray;
    meta.mDataType = GridType::UInt32;
    if (b.name.length() + 1 > nanovdb::GridBlindMetaData::MaxNameSize) {
        std::stringstream ss;
        ss << "Point attribute name \"" << b.name << "\" is more then " << nanovdb::GridBlindMetaData::MaxNameSize << " characters";
        OPENVDB_THROW(openvdb::ValueError, ss.str());
    }
    memcpy(meta.mName, b.name.c_str(), b.name.size() + 1);

    uint32_t* points = reinterpret_cast<uint32_t*>(this->blindData());
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, array.size(), 16),
                      [&](const tbb::blocked_range<uint32_t>& r) {
                          for (auto i = r.begin(); i != r.end(); ++i) {
                              uint32_t* p = points + data[i].mValueMin;
                              for (uint32_t idx : array[i]->indices())
                                  *p++ = idx;
                          }
                      });
}

template<typename SrcTreeT, typename BufferT>
template<typename LeafT>
inline typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointDataIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::postProcessPoints(std::vector<const LeafT*>& array)
{
    if (mBlindMetaData.empty())
        return;
    const uint32_t leafCount = array.size();
    auto*          data = this->template nodeData<DstNode0>();

    data[0].mValueMin = 0; // start of prefix sum
    for (uint32_t i = 1; i < leafCount; ++i)
        data[i].mValueMin = data[i - 1].mValueMin + data[i - 1].mValueMax;

    // write point coordinates as blind meta data
    size_t byteOffset = uintptr_t(this->blindData()) - uintptr_t(this->gridData()); // offset from Grid to blind data;
    for (auto& b : mBlindMetaData) {
        assert(b.count == data[leafCount - 1].mValueMin + data[leafCount - 1].mValueMax);
        auto& meta = const_cast<nanovdb::GridBlindMetaData&>(this->gridData()->blindMetaData(static_cast<uint32_t>(b.index)));
        meta.mByteOffset = byteOffset; // offset from Grid to blind data
        byteOffset += b.size;
        meta.mElementCount = b.count;
        meta.mFlags = 0;
        meta.mDataClass = GridBlindDataClass::AttributeArray;
        if (b.name.length() + 1 > nanovdb::GridBlindMetaData::MaxNameSize) {
            std::stringstream ss;
            ss << "Point attribute name \"" << b.name << "\" is more then " << nanovdb::GridBlindMetaData::MaxNameSize << " characters";
            OPENVDB_THROW(openvdb::ValueError, ss.str());
        }
        memcpy(meta.mName, b.name.c_str(), b.name.size() + 1);
        if (b.typeName == "vec3s") {
            meta.mDataType = GridType::Vec3f;
            using T = openvdb::Vec3f;
            T* ptr = reinterpret_cast<T*>(mData + meta.mByteOffset);
            if (b.name == "P") {
                meta.mSemantic = GridBlindDataSemantic::PointPosition;
                tbb::parallel_for(tbb::blocked_range<uint64_t>(0, array.size(), 16),
                                  [&](const tbb::blocked_range<uint64_t>& r) {
                                      for (auto i = r.begin(); i != r.end(); ++i) {
                                          auto*                               leaf = array[i];
                                          openvdb::points::AttributeHandle<T> posHandle(leaf->constAttributeArray(b.index));
                                          T*                                  p = ptr + data[i].mValueMin;
                                          for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                                              const auto ijk = iter.getCoord();
                                              assert(leaf->isValueOn(ijk));
                                              *p++ = ijk.asVec3s() + posHandle.get(*iter);
                                          }
                                      }
                                  });
            } else {
                if (b.name == "V") {
                    meta.mSemantic = GridBlindDataSemantic::PointVelocity;
                } else if (b.name == "Cd") {
                    meta.mSemantic = GridBlindDataSemantic::PointColor;
                } else if (b.name == "N") {
                    meta.mSemantic = GridBlindDataSemantic::PointNormal;
                } else {
                    meta.mSemantic = GridBlindDataSemantic::Unknown;
                }
                tbb::parallel_for(tbb::blocked_range<uint64_t>(0, array.size(), 16),
                                  [&](const tbb::blocked_range<uint64_t>& r) {
                                      for (auto i = r.begin(); i != r.end(); ++i) {
                                          auto*                               leaf = array[i];
                                          openvdb::points::AttributeHandle<T> posHandle(leaf->constAttributeArray(b.index));
                                          T*                                  p = ptr + data[i].mValueMin;
                                          for (auto iter = leaf->beginIndexOn(); iter; ++iter)
                                              *p++ = posHandle.get(*iter);
                                      }
                                  });
            }
        } else if (b.typeName == "int32") {
            meta.mDataType = GridType::Int32;
            if (b.name == "id") {
                meta.mSemantic = GridBlindDataSemantic::PointId;
            } else {
                meta.mSemantic = GridBlindDataSemantic::Unknown;
            }
            using T = int32_t;
            T* ptr = reinterpret_cast<T*>(mData + meta.mByteOffset);
            tbb::parallel_for(tbb::blocked_range<uint64_t>(0, array.size(), 16),
                              [&](const tbb::blocked_range<uint64_t>& r) {
                                  for (auto i = r.begin(); i != r.end(); ++i) {
                                      auto*                               leaf = array[i];
                                      openvdb::points::AttributeHandle<T> posHandle(leaf->constAttributeArray(b.index));
                                      T*                                  p = ptr + data[i].mValueMin;
                                      for (auto iter = leaf->beginIndexOn(); iter; ++iter)
                                          *p++ = posHandle.get(*iter);
                                  }
                              });
        } else {
            std::stringstream ss;
            ss << "Unsupported point attribute type: \"" << b.typeName << "\"";
            OPENVDB_THROW(openvdb::ValueError, ss.str());
        }
    } // loop over bind data
}

template<typename SrcTreeT, typename BufferT>
template<typename T, typename FlagT>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::setFlag(const T& min, const T& max, FlagT& flag) const
{
    if (mDelta > 0 && (min > mDelta || max < -mDelta)) {
        flag |= FlagT(1); // set first bit
    } else {
        flag &= ~FlagT(1); // unset first bit
    }
}

template<typename SrcTreeT, typename BufferT>
GridHandle<BufferT>
OpenToNanoVDB<SrcTreeT, BufferT>::operator()(const openvdb::Grid<SrcTreeT>& srcGrid, bool mortonSort, int verbose, ChecksumMode mode, const BufferT& allocator)
{
    openvdb::util::CpuTimer timer;

    const SrcTreeT& srcTree = srcGrid.tree();
    const SrcRootT& srcRoot = srcTree.root();

    if (verbose > 1)
        timer.start("Extracting nodes from openvdb grid");
    std::vector<const SrcNode2*> array2; // upper OpenVDB internal nodes
    std::vector<const SrcNode1*> array1; // lower OpenVDB internal nodes
    std::vector<const SrcNode0*> array0; // OpenVDB leaf nodes
    array0.reserve(srcTree.leafCount()); // fast pre-allocation of OpenVDB leaf nodes (of which there are many)
    tbb::parallel_invoke([&]() { srcTree.getNodes(array0); }, // multi-threaded population of node arrays from OpenVDB tree
                         [&]() { srcTree.getNodes(array1); },
                         [&]() { srcTree.getNodes(array2); });
    if (verbose > 1)
        timer.stop();

    if (srcRoot.getTableSize() != array2.size())
        OPENVDB_THROW(openvdb::RuntimeError, "Tiles at the root level are not supported yet!");

    auto key = [](const CoordT& p) { return DstRootT::DataType::CoordToKey(p); };
#ifdef USE_SINGLE_ROOT_KEY
    if (verbose > 1)
        timer.start("Sorting " + std::to_string(array2.size()) + " child nodes of the root node");
    this->sortNodes(array2, key);
    if (verbose > 1)
        timer.stop();
#endif
    assert(std::is_sorted(array2.begin(), array2.end(), [&key](const SrcNode2* a, const SrcNode2* b) { return key(a->origin()) < key(b->origin()); }));

    if (verbose > 1)
        timer.start("Pre-processing points");
    this->preProcessPoints(array0);
    if (verbose > 1)
        timer.stop();

    mBytes[0] = DstGridT::memUsage(); // grid + blind meta data
    mBytes[1] = DstTreeT::memUsage(); // tree
    mBytes[2] = nanovdb::GridBlindMetaData::memUsage(mBlindMetaData.size()); // blind meta data
    mBytes[3] = DstRootT::memUsage(srcRoot.getTableSize()); // root
    mBytes[4] = array2.size() * DstNode2::memUsage(); // upper internal nodes
    mBytes[5] = array1.size() * DstNode1::memUsage(); // lower internal nodes
    mBytes[6] = array0.size() * DstNode0::memUsage(); // leaf nodes
    mBytes[7] = 0;
    for (auto& i : mBlindMetaData)
        mBytes[7] += i.size; // blind meta data

    for (int i = 1; i < 8; ++i)
        mBytes[i] += mBytes[i - 1]; // Byte offsets to: tree, blindmetadata, root, node2, node1, leafs, blinddata, total

    if (mortonSort) { // Morton sorting of the leaf nodes is disabled by default since it has not been performance tested yet
        if (verbose > 1)
            timer.start("Morton sorting of " + std::to_string(array0.size()) + " leaf nodes");
        auto splitBy3 = [](int32_t i) {
            uint64_t x = uint32_t(i + 1000000000) & 0x1fffff; // offset to positive int and extract the first 21 bits
            x = (x | x << 32) & 0x1f00000000ffff;
            x = (x | x << 16) & 0x1f0000ff0000ff;
            x = (x | x << 8) & 0x100f00f00f00f00f;
            x = (x | x << 4) & 0x10c30c30c30c30c3;
            x = (x | x << 2) & 0x1249249249249249;
            return x;
        };
        auto key = [&splitBy3](const CoordT& p) { return splitBy3(p[0]) | splitBy3(p[1]) << 1 | splitBy3(p[2]) << 2; };
        this->sortNodes(array0, key);
        if (verbose > 1)
            timer.stop();
    }

    if (verbose > 1)
        timer.start("Allocating memory for the NanoVDB");
    GridHandle<BufferT> handle(allocator.create(this->gridSize()));
    mData = handle.data();
    if (verbose > 1)
        timer.stop();

    if (verbose)
        openvdb::util::printBytes(std::cerr, this->gridSize(), "Allocated", " for the NanoVDB\n");

    if (srcGrid.getGridClass() == openvdb::GRID_LEVEL_SET) {
        mDelta = ValueT(srcGrid.voxelSize()[0]); // skip a node if max < -mDelta || min > mDelta
    } else {
        mDelta = ValueT(0); // dummy value
    }

    std::unique_ptr<int32_t[]> cache0(new int32_t[array0.size()]); // cache for byte offsets to leaf nodes

    if (verbose > 1)
        timer.start("Processing leaf nodes");
    this->processLeafs(array0, cache0.get());
    if (verbose > 1)
        timer.stop();

    std::unique_ptr<int32_t[]> cache1(new int32_t[array1.size()]); // cache for byte offsets to lower internal nodes

    if (verbose > 1)
        timer.start("Processing lower internal nodes");
    this->processInternals<SrcNode1, DstNode1>(array1, cache1.get(), cache0.get());
    if (verbose > 1)
        timer.stop();

    cache0.reset();
    std::unique_ptr<int32_t[]> cache2(new int32_t[array2.size()]); // cache for byte offsets to upper internal nodes

    if (verbose > 1)
        timer.start("Processing upper internal nodes");
    this->processInternals<SrcNode2, DstNode2>(array2, cache2.get(), cache1.get());
    if (verbose > 1)
        timer.stop();

    cache1.reset();

    if (verbose > 1)
        timer.start("Processing Root node");
    this->processRoot(srcTree.root(), array2, cache2.get());
    if (verbose > 1)
        timer.stop();

    cache2.reset();

    if (verbose > 1)
        timer.start("Processing Tree");
    const uint64_t nodeCount[4] = {array0.size(), array1.size(), array2.size(), 1};
    this->processTree(nodeCount);
    if (verbose > 1)
        timer.stop();

    if (verbose > 1)
        timer.start("Processing Grid");
    this->processGrid(srcGrid);
    if (verbose > 1)
        timer.stop();

    if (verbose > 1)
        timer.start("Post-processing points");
    this->postProcessPoints(array0);
    if (verbose > 1)
        timer.stop();

    if (verbose > 1)
        timer.start("Post-processing Grid");
    this->postProcessGrid(mode);
    if (verbose > 1)
        timer.stop();

    mData = nullptr;
    return handle;// envokes mode constructor
} // operator()

template<typename SrcTreeT, typename BufferT>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    processLeafs(std::vector<const SrcNode0*>& array, int32_t* x0)
{
    mActiveVoxelCount = 0;
    auto* start = this->template nodeData<DstNode0>(); // address of first leaf node
    auto  op = [&](const tbb::blocked_range<uint32_t>& r) {
        int32_t* x = x0 + r.begin();
        uint64_t sum = 0;
        auto*    data = start + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++data) {
            const SrcNode0* srcLeaf = array[i];
            sum += srcLeaf->onVoxelCount();
            data->mValueMask = srcLeaf->valueMask();
            const ValueT* src = srcLeaf->buffer().data();
            for (ValueT *dst = data->mValues, *end = dst + SrcNode0::size(); dst != end; dst += 4, src += 4) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
            auto iter = srcLeaf->cbeginValueOn();
            // Since min, max and bbox are derived from active values they are required at the leaf level!
            if (!iter) {
                OPENVDB_THROW(openvdb::RuntimeError, "Expected at least one active voxel in every leaf node! Hint: try pruneInactive.");
            }
            data->mValueMin = *iter, data->mValueMax = data->mValueMin;
            openvdb::CoordBBox bbox;
            bbox.expand(srcLeaf->offsetToLocalCoord(iter.pos()));
            for (++iter; iter; ++iter) {
                bbox.expand(srcLeaf->offsetToLocalCoord(iter.pos()));
                const ValueT& v = *iter;
                if (v < data->mValueMin) {
                    data->mValueMin = v;
                } else if (v > data->mValueMax) {
                    data->mValueMax = v;
                }
            }
            this->setFlag(data->mValueMin, data->mValueMax, data->mFlags);
            bbox.translate(srcLeaf->origin());
            data->mBBoxMin = bbox.min();
            data->mBBoxDif[0] = uint8_t(bbox.max()[0] - bbox.min()[0]);
            data->mBBoxDif[1] = uint8_t(bbox.max()[1] - bbox.min()[1]);
            data->mBBoxDif[2] = uint8_t(bbox.max()[2] - bbox.min()[2]);
            cache(*x++, srcLeaf->origin(), i);
        }
        mActiveVoxelCount += sum;
    };
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(array.size()), 8), op);
} // processLeafs

template<typename SrcTreeT, typename BufferT>
template<typename SrcNode, typename DstNode>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    processInternals(std::vector<const SrcNode*>& array, int32_t* x0, const int32_t* childX)
{
    using SrcChildT = typename SrcNode::ChildNodeType;
    const uint32_t size = static_cast<uint32_t>(array.size());
    auto*          start = this->template nodeData<DstNode>();
    auto           op = [&](const tbb::blocked_range<size_t>& r) {
        int32_t* x = x0 + r.begin();
        auto*    data = start + r.begin();
        uint64_t sum = 0;
        for (auto i = r.begin(); i != r.end(); ++i, ++data) {
            const SrcNode* srcNode = array[i];
            sum += SrcChildT::NUM_VOXELS * srcNode->getValueMask().countOn();
            cache(*x++, srcNode->origin(), i);
            data->mOffset = size - i;
            data->mValueMask = srcNode->getValueMask();
            data->mChildMask = srcNode->getChildMask();
            for (auto iter = srcNode->cbeginValueAll(); iter; ++iter) {
                data->mTable[iter.pos()].value = *iter;
            }
            auto onValIter = srcNode->cbeginValueOn();
            auto childIter = srcNode->cbeginChildOn();
            if (onValIter) {
                data->mValueMin = *onValIter;
                data->mValueMax = data->mValueMin;
                const CoordT &ijk = onValIter.getCoord();
                data->mBBox[0] = ijk;
                data->mBBox[1] = ijk.offsetBy(SrcChildT::DIM - 1);
                ++onValIter;
            } else if (childIter) {
                const auto childID = static_cast<uint32_t>(childIter->origin()[0]);
                data->mTable[childIter.pos()].childID = childID;
                const_cast<CoordT&>(childIter->origin())[0] = childX[childID];
                auto* dstChild = data->child(childIter.pos());
                data->mValueMin = dstChild->valueMin();
                data->mValueMax = dstChild->valueMax();
                data->mBBox = dstChild->bbox();
                ++childIter;
            } else {
                OPENVDB_THROW(openvdb::RuntimeError, "Internal node with no children or active values! Hint: try pruneInactive.");
            }
            for (; onValIter; ++onValIter) { // typically there are few active tiles
                const auto& value = *onValIter;
                if (value < data->mValueMin) {
                    data->mValueMin = value;
                } else if (value > data->mValueMax) {
                    data->mValueMax = value;
                }
                const CoordT &ijk = onValIter.getCoord();
                data->mBBox.min().minComponent(ijk);
                data->mBBox.max().maxComponent(ijk.offsetBy(SrcChildT::DIM - 1));
            }
            for (; childIter; ++childIter) {
                const auto n = childIter.pos();
                const auto childID = static_cast<uint32_t>(childIter->origin()[0]);
                data->mTable[n].childID = childID;
                const_cast<CoordT&>(childIter->origin())[0] = childX[childID];
                auto* dstChild = data->child(n);
                if (dstChild->valueMin() < data->mValueMin)
                    data->mValueMin = dstChild->valueMin();
                if (dstChild->valueMax() > data->mValueMax)
                    data->mValueMax = dstChild->valueMax();
                const auto& bbox = dstChild->bbox();
                data->mBBox.min().minComponent(bbox.min());
                data->mBBox.max().maxComponent(bbox.max());
            }
            this->setFlag(data->mValueMin, data->mValueMax, data->mFlags);
        }
        mActiveVoxelCount += sum;
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, array.size(), 4), op);
} // processInternals

template<typename SrcTreeT, typename BufferT>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    processRoot(const SrcRootT& srcRoot, std::vector<const SrcNode2*>& array, const int32_t* childX)
{
    using SrcChildT = typename SrcRootT::ChildNodeType;
    auto& data = *(this->template nodeData<DstRootT>());
    data.mBackground = srcRoot.background();
    data.mTileCount = srcRoot.getTableSize();
    // since openvdb::RootNode internally uses a std::map for child nodes its iterator
    // visits elements in the stored order required by the nanovdb::RootNode
    if (data.mTileCount == 0) { // empty root node
        data.mValueMin = data.mValueMax = data.mBackground;
        data.mBBox.min() = openvdb::Coord::max(); // set to an empty bounding box
        data.mBBox.max() = openvdb::Coord::min();
        data.mActiveVoxelCount = 0;
    } else {
        auto*      node = array[0];
        auto&      tile = data.tile(0);
        const auto childID = static_cast<uint32_t>(node->origin()[0]);
        const_cast<CoordT&>(node->origin())[0] = childX[childID]; // restore cached coordinate
        tile.setChild(node->origin(), childID);
        auto& dstChild = data.child(tile);
        data.mValueMin = dstChild.valueMin();
        data.mValueMax = dstChild.valueMax();
        data.mBBox = dstChild.bbox();
        for (uint32_t i = 1, n = static_cast<uint32_t>(array.size()); i < n; ++i) {
            node = array[i];
            auto&      tile = data.tile(i);
            const auto childID = static_cast<uint32_t>(node->origin()[0]);
            const_cast<openvdb::Coord&>(node->origin())[0] = childX[childID]; // restore cached coordinate
            tile.setChild(node->origin(), childID);
            auto& dstChild = data.child(tile);
            if (dstChild.valueMin() < data.mValueMin)
                data.mValueMin = dstChild.valueMin();
            if (dstChild.valueMax() > data.mValueMax)
                data.mValueMax = dstChild.valueMax();
            data.mBBox.min().minComponent(dstChild.bbox().min());
            data.mBBox.max().maxComponent(dstChild.bbox().max());
        }
        for (auto iter = srcRoot.cbeginValueAll(); iter; ++iter) {
            OPENVDB_THROW(openvdb::RuntimeError, "Tiles at the root node is broken and need to be fixed!");
            auto& tile = data.tile(iter.pos());
            tile.setValue(iter.getCoord(), iter.isValueOn(), *iter);
            if (iter.isValueOn()) {
                if (tile.value < data.mValueMin) {
                    data.mValueMin = tile.value;
                } else if (tile.value > data.mValueMax) {
                    data.mValueMax = tile.value;
                }
                mActiveVoxelCount += SrcChildT::NUM_VOXELS;
                data.mBBox.min().minComponent(iter.getCoord());
                data.mBBox.max().maxComponent(iter.getCoord().offsetBy(SrcNode2::DIM - 1));
            }
        }
        data.mActiveVoxelCount = mActiveVoxelCount;
    }
}// processRoot

template<typename SrcTreeT, typename BufferT>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    processTree(const uint64_t* count)
{
    auto& data = *this->treeData(); // data for the tree
    for (int i = 0; i < 4; ++i) {
        if (count[i] > std::numeric_limits<uint32_t>::max())
            OPENVDB_THROW(openvdb::ValueError, "Node count exceeds 32 bit range");
        data.mCount[i] = static_cast<uint32_t>(count[i]);
        data.mBytes[i] = mBytes[5 - i] - mBytes[0]; // offset from the tree to the first node at each tree level
    }
}

template<typename SrcTreeT, typename BufferT>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    processGrid(const SrcGridT& srcGrid)
{
    if (!srcGrid.transform().baseMap()->isLinear())
        OPENVDB_THROW(openvdb::ValueError, "OpenToNanoVDB only supports grids with affine transforms");
    auto  affineMap = srcGrid.transform().baseMap()->getAffineMap();
    auto& data = *this->gridData();
    data.mMagic = NANOVDB_MAGIC_NUMBER;
    data.mMajor = NANOVDB_MAJOR_VERSION_NUMBER;
    data.mGridSize = this->gridSize();
    data.setFlagsOff();
    data.setMinMax(true);
    data.setBBox(true);
    data.mBlindMetadataOffset = mBlindMetaData.size()?mBytes[1]:0;
    data.mBlindMetadataCount = static_cast<uint32_t>(mBlindMetaData.size());
    { // set grid name
        const std::string name = srcGrid.getName();
        if (name.length() + 1 > nanovdb::GridData::MaxNameSize) {
            std::stringstream ss;
            ss << "Grid name \"" << name << "\" is more then " << nanovdb::GridData::MaxNameSize << " characters";
            OPENVDB_THROW(openvdb::ValueError, ss.str());
        }
        memcpy(data.mGridName, name.c_str(), name.size() + 1);
    }
    switch (srcGrid.getGridClass()) { // set grid class
    case openvdb::GRID_LEVEL_SET:
        if (!is_floating_point<ValueT>::value)
            OPENVDB_THROW(openvdb::ValueError, "Level sets are expected to be floating point types");
        data.mGridClass = GridClass::LevelSet;
        break;
    case openvdb::GRID_FOG_VOLUME:
        data.mGridClass = GridClass::FogVolume;
        break;
    case openvdb::GRID_STAGGERED:
        data.mGridClass = GridClass::Staggered;
        break;
    default:
        data.mGridClass = GridClass::Unknown;
    }
    if (std::is_same<ValueT, float>::value) { // resolved at compiletime
        data.mGridType = GridType::Float;
    } else if (std::is_same<ValueT, double>::value) {
        data.mGridType = GridType::Double;
    } else if (std::is_same<ValueT, int16_t>::value) {
        data.mGridType = GridType::Int16;
    } else if (std::is_same<ValueT, int32_t>::value) {
        data.mGridType = GridType::Int32;
    } else if (std::is_same<ValueT, int64_t>::value) {
        data.mGridType = GridType::Int64;
    } else if (std::is_same<ValueT, openvdb::Vec3f>::value) {
        data.mGridType = GridType::Vec3f;
    } else if (std::is_same<ValueT, openvdb::Index32>::value) {
        data.mGridType = GridType::UInt32;
    } else if (std::is_same<ValueT, openvdb::PointIndex32>::value) {
        data.mGridType = GridType::UInt32;
        data.mGridClass = GridClass::PointIndex;
    } else if (std::is_same<ValueT, openvdb::PointDataIndex32>::value) {
        data.mGridType = GridType::UInt32;
        data.mGridClass = GridClass::PointData;
    } else {
        OPENVDB_THROW(openvdb::ValueError, "Unsupported value type");
    }
    { // set affine map
        if (srcGrid.hasUniformVoxels())
            data.mVoxelSize = nanovdb::Vec3R(affineMap->voxelSize()[0]);
        else
            data.mVoxelSize = affineMap->voxelSize();
        const auto mat = affineMap->getMat4();
        // Only support non-tapered at the moment:
        data.mMap.set(mat, mat.inverse(), 1.0);
    }
    { // set world space AABB
        auto& rootData = *(this->template nodeData<DstRootT>()); 
        const openvdb::Vec3R                      min(&(rootData.mBBox.min()[0]));
        const openvdb::Vec3R                      max(&(rootData.mBBox.max()[0]));
        const openvdb::math::BBox<openvdb::Vec3R> bboxIndex(min, max);
        const auto                                bboxWorld = bboxIndex.applyMap(*srcGrid.transform().baseMap());
        data.mWorldBBox.min() = bboxWorld.min();
        data.mWorldBBox.max() = bboxWorld.max();
    }
}

template<typename SrcTreeT, typename BufferT>
template<typename NodeT, typename KeyT>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    sortNodes(std::vector<const NodeT*>& array, KeyT key)
{
    using RangeT = tbb::blocked_range<uint32_t>;
    struct Pair
    {
        const NodeT*                                node;
        typename std::result_of<KeyT(CoordT)>::type code;
        bool                                        operator<(const Pair& rhs) const { return code < rhs.code; }
    };

    // Construct a temporary vector of key,pointer pairs to be sorted by the key value
    const uint32_t          size = static_cast<uint32_t>(array.size());
    std::unique_ptr<Pair[]> tmp(new Pair[size]);
    tbb::parallel_for(RangeT(0, size), [&](const RangeT& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            tmp[i].node = array[i];
            tmp[i].code = key(array[i]->origin());
        }
    });

    tbb::parallel_sort(&tmp[0], &tmp[size]);

    // Copy back sorted points into the array
    tbb::parallel_for(RangeT(0, size), [&](const RangeT& r) {
    for (auto i=r.begin(); i!=r.end(); ++i) array[i] = tmp[i].node; });
} // sortNodes

} // unnamed namespace

template<typename BufferT, typename SrcTreeT>
GridHandle<BufferT>
openToNanoVDB(const openvdb::Grid<SrcTreeT>& grid, bool mortonSort, int verbose, ChecksumMode mode)
{
    OpenToNanoVDB<SrcTreeT, BufferT> s;
    return s(grid, mortonSort, verbose, mode);
}

template<typename BufferT>
GridHandle<BufferT>
openToNanoVDB(const openvdb::GridBase::Ptr& base, bool mortonSort, int verbose, ChecksumMode mode)
{
    if (auto grid = openvdb::GridBase::grid<openvdb::FloatGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::FloatTree>(*grid, mortonSort, verbose, mode);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::DoubleGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::DoubleTree>(*grid, mortonSort, verbose, mode);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int32Grid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Int32Tree>(*grid, mortonSort, verbose, mode);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int64Grid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Int64Tree>(*grid, mortonSort, verbose, mode);
     } else if (auto grid = openvdb::GridBase::grid<openvdb::Grid<openvdb::UInt32Tree>>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::UInt32Tree>(*grid, mortonSort, verbose, mode);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3fGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Vec3fTree>(*grid, mortonSort, verbose, mode);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3dGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Vec3dTree>(*grid, mortonSort, verbose, mode);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::points::PointDataGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::points::PointDataTree>(*grid, mortonSort, verbose, mode);
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unrecognized OpenVDB grid type");
    }
}

} // namespace nanovdb

#endif // NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED
