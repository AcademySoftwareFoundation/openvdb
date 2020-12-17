// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file OpenToNanoVDB.h

    \author Ken Museth

    \date January 8, 2020

    \brief This class will serialize an OpenVDB grid into a NanoVDB grid.
*/

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/util/CpuTimer.h>

#include "GridHandle.h" // manages and streams the raw memory buffer of a NanoVDB grid.
#include "GridChecksum.h" // for checksum
#include "GridStats.h" // for Extrema

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>

#include <atomic>
#include <type_traits>

#ifndef NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED

namespace nanovdb {

/// @brief Forward declaration of free-standing function that converts an OpenVDB GridBase into a NanoVDB GridHandle
template<typename BufferT = HostBuffer>
GridHandle<BufferT>
openToNanoVDB(const openvdb::GridBase::Ptr& base,
              StatsMode                     sMode = StatsMode::Default,
              ChecksumMode                  cMode = ChecksumMode::Default,
              bool                          mortonSort = false,
              int                           verbose = 0);

/// @brief Forward declaration of free-standing function that converts a typed OpenVDB Grid into a NanoVDB GridHandle
template<typename BufferT = HostBuffer, typename SrcTreeT = openvdb::FloatTree>
GridHandle<BufferT>
openToNanoVDB(const openvdb::Grid<SrcTreeT>& grid,
              StatsMode                      sMode = StatsMode::Default,
              ChecksumMode                   cMode = ChecksumMode::Default,
              bool                           mortonSort = false,
              int                            verbose = 0);

/// @brief Converts OpenVDB types to NanoVDB types
template<typename T>
struct TypeConverter
{
    using Type = T;
};

template<>
struct TypeConverter<openvdb::Coord>
{
    using Type = nanovdb::Coord;
    static_assert(sizeof(Type) == sizeof(openvdb::Coord), "Mismatching sizeof");
};

template<>
struct TypeConverter<openvdb::math::CoordBBox>
{
    using Type = nanovdb::CoordBBox;
    static_assert(sizeof(Type) == sizeof(openvdb::CoordBBox), "Mismatching sizeof");
};

template<typename T>
struct TypeConverter<openvdb::math::BBox<T>>
{
    using Type = nanovdb::BBox<T>;
    static_assert(sizeof(Type) == sizeof(openvdb::math::BBox<T>), "Mismatching sizeof");
};

template<typename T>
struct TypeConverter<openvdb::math::Vec3<T>>
{
    using Type = nanovdb::Vec3<T>;
    static_assert(sizeof(Type) == sizeof(openvdb::math::Vec3<T>), "Mismatching sizeof");
};

template<typename T>
struct TypeConverter<openvdb::math::Vec4<T>>
{
    using Type = nanovdb::Vec4<T>;
    static_assert(sizeof(Type) == sizeof(openvdb::math::Vec4<T>), "Mismatching sizeof");
};

template<>
struct TypeConverter<openvdb::ValueMask>
{
    using Type = nanovdb::ValueMask;
    static_assert(sizeof(Type) == sizeof(openvdb::ValueMask), "Mismatching sizeof");
};

namespace { // unnamed namespace

/// @brief This class will openToNanoVDB an OpenVDB grid into a NanoVDB grid managed by a GridHandle.
template<typename SrcTreeT, typename BufferT = HostBuffer>
class OpenToNanoVDB
{
    struct BlindMetaData; // forward decleration
    using SrcBuildT = typename SrcTreeT::BuildType;
    using BuildT = typename TypeConverter<typename SrcTreeT::BuildType>::Type;// openvdb to nanovdb type conversion 
    using ValueT = typename TypeConverter<typename SrcTreeT::ValueType>::Type;// openvdb to nanovdb type conversion
    ValueT                  mDelta; // skip node if: node.max < -mDelta || node.min > mDelta
    uint8_t*                mData; // raw pointer to memory buffer of the NanoVDB grid
    uint64_t                mBytes[8]; // Byte offsets to from mData to: tree, blindmetadata, root, node2, node1, leafs, blinddata, (total size)
    std::set<BlindMetaData> mBlindMetaData; // sorted according to index

public:
    /// @brief Construction from an existing const OpenVDB Grid.
    OpenToNanoVDB()
        : mData(nullptr)
    {
    }

    /// @brief Return a shared pointer to a NanoVDB grid constructed from the specified OpenVDB grid
    GridHandle<BufferT> operator()(const openvdb::Grid<SrcTreeT>& grid,
                                   StatsMode                      sMode = StatsMode::Default,
                                   ChecksumMode                   mode = ChecksumMode::Default,
                                   bool                           mortonSort = false,
                                   int                            verbose = 0,
                                   const BufferT&                 allocator = BufferT());

private:
    static_assert(SrcTreeT::DEPTH == 4, "Converter assumes an OpenVDB tree of depth 4 (which is the default configuration)");
    using CoordT = openvdb::Coord;
    using BBoxT = nanovdb::BBox<CoordT>;

    using SrcGridT = openvdb::Grid<SrcTreeT>;
    using SrcRootT = typename SrcTreeT::RootNodeType; // OpenVDB root node
    using SrcNode2 = typename SrcRootT::ChildNodeType; // upper OpenVDB internal node
    using SrcNode1 = typename SrcNode2::ChildNodeType; // lower OpenVDB internal node
    using SrcNode0 = typename SrcNode1::ChildNodeType; // OpenVDB leaf node

    static_assert(SrcNode0::LOG2DIM == 3, "Expected 8^3 leaf nodes");
    static_assert(SrcNode1::LOG2DIM == 4, "Expected 16^3 lower internal nodes");
    static_assert(SrcNode2::LOG2DIM == 5, "Expected 32^3 upper internal nodes");

    using DstNode0 = nanovdb::LeafNode<BuildT>;//, CoordT, openvdb::util::NodeMask, SrcNode0::LOG2DIM>; // leaf
    using DstNode1 = nanovdb::InternalNode<DstNode0>;//, SrcNode1::LOG2DIM>; // lower
    using DstNode2 = nanovdb::InternalNode<DstNode1>;//, SrcNode2::LOG2DIM>; // upper
    using DstRootT = nanovdb::RootNode<DstNode2>;
    using DstTreeT = nanovdb::Tree<DstRootT>;
    using DstGridT = nanovdb::Grid<DstTreeT>;

    /// @brief Private methods to pre-process the bind metadata
    template<typename SrcLeafT>
    typename std::enable_if<!(std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value ||
                              std::is_same<typename SrcLeafT::ValueType, openvdb::PointDataIndex32>::value)>::type
    preProcessMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>&);
    template<typename SrcLeafT>
    typename std::enable_if<std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value>::type
    preProcessMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>& array);
    template<typename SrcLeafT>
    typename std::enable_if<std::is_same<typename SrcLeafT::ValueType, openvdb::PointDataIndex32>::value>::type
    preProcessMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>& array);

    /// @brief Private methods for concurrent processing the leaf nodes
    template<typename SrcLeafT>
    typename std::enable_if<std::is_same<typename SrcLeafT::BuildType, bool>::value>::type
    processLeafs(std::vector<const SrcLeafT*>& array, int32_t* x0);
    template<typename SrcLeafT>
    typename std::enable_if<std::is_same<typename SrcLeafT::BuildType, openvdb::ValueMask>::value>::type
    processLeafs(std::vector<const SrcLeafT*>& array, int32_t* x0);
    template<typename SrcLeafT>
    typename std::enable_if<!std::is_same<typename SrcLeafT::BuildType, openvdb::ValueMask>::value && !std::is_same<typename SrcLeafT::BuildType, bool>::value>::type
    processLeafs(std::vector<const SrcLeafT*>& array, int32_t* x0);

    /// @brief Private method for concurrent processing the internal nodes
    template<typename SrcNode, typename DstNode>
    void processInternals(std::vector<const SrcNode*>& array, int32_t* x0, const int32_t* childX);
    
    /// @brief Private method to process the root node
    void processRoot(const SrcRootT& srcRoot, std::vector<const SrcNode2*>& array, const int32_t* childX);

    // @brief Private method to process the tree
    void processTree(const uint64_t* count);

    /// @brief Private method to process the grid
    void processGrid(const SrcGridT& srcGrid);

    /// @brief Private methods to process the blind metadata
    template<typename SrcLeafT>
    typename std::enable_if<!(std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value ||
                              std::is_same<typename SrcLeafT::ValueType, openvdb::PointDataIndex32>::value)>::type
    processMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>&);
    template<typename SrcLeafT>
    typename std::enable_if<std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value>::type
    processMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>& array);
    template<typename SrcLeafT>
    typename std::enable_if<std::is_same<typename SrcLeafT::ValueType, openvdb::PointDataIndex32>::value>::type
    processMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>& array);

    template<typename LeafT>
    uint64_t pointCount(std::vector<const LeafT*>&);
    
    //template<typename AttT>
    //void copyPointAttribute(std::vector<const SrcNode0*>& array, size_t attIdx, const openvdb::Name &codec);
    template<typename AttT, typename CodecT = openvdb::points::UnknownCodec>
    void copyPointAttribute(std::vector<const SrcNode0*>& array, size_t attIdx);

    /// @brief Private methods to access points to data
    template<typename DstNodeT>
    DstNodeT* node() const { return reinterpret_cast<DstNodeT*>(mData + mBytes[5 - DstNodeT::LEVEL]); }

    template<typename DstNodeT>
    typename DstNodeT::DataType* nodeData() const { return reinterpret_cast<typename DstNodeT::DataType*>(mData + mBytes[5 - DstNodeT::LEVEL]); }
    typename DstTreeT::DataType* treeData() const { return reinterpret_cast<typename DstTreeT::DataType*>(mData + mBytes[0]); }
    typename DstGridT::DataType* gridData() const { return reinterpret_cast<typename DstGridT::DataType*>(mData); }
    uint64_t                     gridSize() const { return mBytes[7]; }
    nanovdb::GridBlindMetaData*  blindMetaData() const { return reinterpret_cast<nanovdb::GridBlindMetaData*>(mData + mBytes[1]); }
    uint8_t*                     blindData() const { return reinterpret_cast<uint8_t*>(mData + mBytes[6]); }

    /// @brief Private method used to cache the x component of a Coord into x and
    //         encode uint32_t id into the x component despite it being of type const int32_t.
    static void cache(int32_t& x, const CoordT& ijk, uint32_t id)
    {
        x = ijk[0];
        reinterpret_cast<uint32_t&>(const_cast<CoordT&>(ijk)[0]) = id;
    }

    /// @brief Private method to sorts the nodes in the specified array using the provided key function
    template<typename NodeT, typename KeyT>
    void sortNodes(std::vector<const NodeT*>& array, KeyT key);

}; // OpenToNanoVDB class

template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
uint64_t OpenToNanoVDB<SrcTreeT, BufferT>::pointCount(std::vector<const SrcLeafT*>& array)
{
    std::atomic<uint64_t> pointCount{0};
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, array.size(), 16),
                      [&](const tbb::blocked_range<uint64_t>& r) {
                          uint64_t sum = 0;
                          for (auto i = r.begin(); i != r.end(); ++i) {
                              sum += array[i]->getLastValue();
                          }
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
template<typename SrcLeafT>
inline typename std::enable_if<!(std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value ||
                                 std::is_same<typename SrcLeafT::ValueType, openvdb::PointDataIndex32>::value)>::type
OpenToNanoVDB<SrcTreeT, BufferT>::preProcessMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>&) 
{
    const size_t length = srcGrid.getName().length();
    if (length >= nanovdb::GridData::MaxNameSize) {
        mBlindMetaData.emplace("grid name", "uint8_t", 0, 1, length + 1);// Null-terminated byte strings
    }
}

template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
inline typename std::enable_if<std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::preProcessMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>& array)
{
    const uint64_t count = this->pointCount(array);
    mBlindMetaData.clear();
    if (count != 0) {
        mBlindMetaData.emplace("index", "uint32", 0, count, sizeof(uint32_t));
    }
    const size_t length = srcGrid.getName().length();
    if (length >= nanovdb::GridData::MaxNameSize) {
        mBlindMetaData.emplace("grid name", "uint8_t", 1, 1, length + 1);// Null-terminated byte strings
    }
}

template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
inline typename std::enable_if<std::is_same<typename SrcLeafT::ValueType, openvdb::PointDataIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::preProcessMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>& array)
{
    const uint64_t count = this->pointCount(array);
    mBlindMetaData.clear();
    if (count == 0) {
        return;
    }
    const auto& attributeSet = array.front()->attributeSet();
    const auto& descriptor = attributeSet.descriptor();
    const auto& nameMap = descriptor.map();
    for (auto it = nameMap.begin(); it != nameMap.end(); ++it) {
        const size_t index = it->second;
        auto&        attArray = array.front()->constAttributeArray(index);
        mBlindMetaData.emplace(it->first, descriptor.valueType(index), index, count, attArray.valueTypeSize());
    }
    const size_t length = srcGrid.getName().length();
    if (length >= nanovdb::GridData::MaxNameSize) {
        mBlindMetaData.emplace("grid name", "uint8_t", nameMap.size(), 1, length + 1);// Null-terminated byte strings
    }
}

template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
inline typename std::enable_if<!(std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value ||
                                 std::is_same<typename SrcLeafT::ValueType, openvdb::PointDataIndex32>::value)>::type
OpenToNanoVDB<SrcTreeT, BufferT>::processMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>&) 
{
    if (mBlindMetaData.empty()) {
        return;
    }
    assert(mBlindMetaData.size() == 1);// only the grid name is expected
    size_t byteOffset = uintptr_t(this->blindData()) - uintptr_t(this->gridData()); // offset from Grid to blind data;
    auto it = mBlindMetaData.cbegin();
    assert(it->name == "grid name" && it->typeName == "uint8_t" && it->index == 0);
    assert(srcGrid.getName().length() >= nanovdb::GridData::MaxNameSize);
    auto& meta = const_cast<nanovdb::GridBlindMetaData&>(this->gridData()->blindMetaData(0));
    meta.mByteOffset = byteOffset; // offset from Grid to blind data
    meta.mElementCount = it->count;
    meta.mFlags = 0;
    meta.mSemantic = GridBlindDataSemantic::Unknown;
    meta.mDataClass = GridBlindDataClass::GridName;
    meta.mDataType = GridType::Unknown;
    strcpy(reinterpret_cast<char*>(mData + meta.mByteOffset), srcGrid.getName().c_str());
}

template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
inline typename std::enable_if<std::is_same<typename SrcLeafT::ValueType, openvdb::PointIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::processMetadata(const SrcGridT& srcGrid, std::vector<const SrcLeafT*>& array)
{
    if (mBlindMetaData.empty())
        return;
    assert(mBlindMetaData.size() == 1 || mBlindMetaData.size() == 2);

    const uint32_t leafCount = static_cast<uint32_t>(array.size());
    auto*          data = this->template nodeData<DstNode0>();

    data[0].mMinimum = 0; // start of prefix sum
    data[0].mMaximum = data[0].mValues[DstNode0::SIZE - 1u];
    for (uint32_t i = 1; i < leafCount; ++i) {
        data[i].mMinimum = data[i - 1].mMinimum + data[i - 1].mMaximum;
        data[i].mMaximum = data[i].mValues[DstNode0::SIZE - 1u];
    }

    // write point offsets as blind meta data
    size_t byteOffset = uintptr_t(this->blindData()) - uintptr_t(this->gridData()); // offset from Grid to blind data;
    auto it = mBlindMetaData.cbegin();
    assert(it->count == data[leafCount - 1].mMinimum + data[leafCount - 1].mMaximum);
    assert(it->name == "index" && it->typeName == "uint32" && it->index == 0);
    auto& meta = const_cast<nanovdb::GridBlindMetaData&>(this->gridData()->blindMetaData(0));
    meta.mByteOffset = byteOffset;
    byteOffset += it->size;
    meta.mElementCount = it->count;
    meta.mFlags = 0;
    meta.mSemantic = GridBlindDataSemantic::Unknown;
    meta.mDataClass = GridBlindDataClass::IndexArray;
    meta.mDataType = GridType::UInt32;
    if (it->name.length() >= nanovdb::GridBlindMetaData::MaxNameSize) {
        std::stringstream ss;
        ss << "Point attribute name \"" << it->name << "\" is more than " << (nanovdb::GridBlindMetaData::MaxNameSize-1) << " characters";
        OPENVDB_THROW(openvdb::ValueError, ss.str());
    }
    memcpy(meta.mName, it->name.c_str(), it->name.size() + 1);

    uint32_t* points = reinterpret_cast<uint32_t*>(this->blindData());
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, array.size(), 16),
                      [&](const tbb::blocked_range<uint32_t>& r) {
                          for (auto i = r.begin(); i != r.end(); ++i) {
                              uint32_t* p = points + data[i].mMinimum;
                              for (uint32_t idx : array[i]->indices())
                                  *p++ = idx;
                          }
                      });
    ++it;
    if (it != mBlindMetaData.cend()) {
        assert(it->name == "grid name" && it->typeName == "uint8_t" && it->index == 1);
        assert(srcGrid.getName().length() >= nanovdb::GridData::MaxNameSize);
        auto& meta = const_cast<nanovdb::GridBlindMetaData&>(this->gridData()->blindMetaData(1));
        meta.mByteOffset = byteOffset; // offset from Grid to blind data
        meta.mElementCount = it->count;
        meta.mFlags = 0;
        meta.mSemantic = GridBlindDataSemantic::Unknown;
        meta.mDataClass = GridBlindDataClass::GridName;
        meta.mDataType = GridType::Unknown;
        strcpy(reinterpret_cast<char*>(mData + meta.mByteOffset), srcGrid.getName().c_str());
    }   
}

template<typename SrcTreeT, typename BufferT>
template<typename AttT, typename CodecT>
inline void
OpenToNanoVDB<SrcTreeT, BufferT>::copyPointAttribute(std::vector<const SrcNode0*>& array, size_t attIdx)
{
    static_assert(std::is_same<typename SrcNode0::ValueType, openvdb::PointDataIndex32>::value, "Expected value to openvdb::PointData");
    using RangeT  = tbb::blocked_range<uint64_t>;
    using HandleT = openvdb::points::AttributeHandle<AttT, CodecT>;
    AttT* attPtr  = reinterpret_cast<AttT*>(mData + this->gridData()->blindMetaData(static_cast<uint32_t>(attIdx)).mByteOffset);
    auto* dstLeaf = this->template nodeData<DstNode0>(); 
    tbb::parallel_for(RangeT(0, array.size(), 16), [&](const RangeT& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto* srcLeaf = array[i];
            HandleT handle(srcLeaf->constAttributeArray(attIdx));
            AttT* p = attPtr + dstLeaf[i].mMinimum;
            for (auto iter = srcLeaf->beginIndexOn(); iter; ++iter) {
                *p++ = handle.get(*iter);
            }
        }
    });
}

/*
template<typename SrcTreeT, typename BufferT>
template<typename AttT>
inline void
OpenToNanoVDB<SrcTreeT, BufferT>::copyPointAttribute(std::vector<const SrcNode0*>& array, size_t attIdx, const openvdb::Name &codec)
{
    if (codec == openvdb::points::FixedPointCodec<false>::name()) {
        this->template copyPointAttribute<openvdb::Vec3f, openvdb::points::FixedPointCodec<false>>(array, attIdx);
    } else if (codec == openvdb::points::FixedPointCodec<true>::name()) {
        this->template copyPointAttribute<openvdb::Vec3f, openvdb::points::FixedPointCodec<true>>(array, attIdx);
    } else if (codec == openvdb::points::NullCodec::name()) {
        this->template copyPointAttribute<openvdb::Vec3f, openvdb::points::NullCodec>(array, attIdx);
    } else {
        this->template copyPointAttribute<openvdb::Vec3f, openvdb::points::UnknownCodec>(array, attIdx);
    }
}
*/
template<typename SrcTreeT, typename BufferT>
template<typename LeafT>
inline typename std::enable_if<std::is_same<typename LeafT::ValueType, openvdb::PointDataIndex32>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::processMetadata(const SrcGridT& srcGrid, std::vector<const LeafT*>& array)
{
    if (mBlindMetaData.empty()) {
        return;
    }
    const uint32_t leafCount = array.size();
    auto*          data = this->template nodeData<DstNode0>();

    data[0].mMinimum = 0; // start of prefix sum
    data[0].mMaximum = data[0].mValues[LeafT::SIZE - 1];
    for (uint32_t i = 1; i < leafCount; ++i) {
        data[i].mMinimum = data[i - 1].mMinimum + data[i - 1].mMaximum;
        data[i].mMaximum = data[i].mValues[LeafT::SIZE - 1];
    }

    // write point coordinates as blind meta data
    size_t byteOffset = uintptr_t(this->blindData()) - uintptr_t(this->gridData()); // offset from Grid to blind data;
    for (auto& b : mBlindMetaData) {
        auto& meta = const_cast<nanovdb::GridBlindMetaData&>(this->gridData()->blindMetaData(static_cast<uint32_t>(b.index)));
        meta.mByteOffset = byteOffset; // offset from Grid to blind data
        byteOffset += b.size;
        meta.mElementCount = b.count;
        meta.mFlags = 0;
        if (b.name == "grid name") {
            meta.mSemantic = GridBlindDataSemantic::Unknown;
            meta.mDataClass = GridBlindDataClass::GridName;
            meta.mDataType = GridType::Unknown;
            assert(srcGrid.getName().length() >= nanovdb::GridData::MaxNameSize);
            strcpy(reinterpret_cast<char*>(mData + meta.mByteOffset), srcGrid.getName().c_str());
        } else {
            assert(b.count == data[leafCount - 1].mMinimum + data[leafCount - 1].mMaximum);
            meta.mDataClass = GridBlindDataClass::AttributeArray;
            if (b.name.length()>= nanovdb::GridBlindMetaData::MaxNameSize) {
                std::stringstream ss;
                ss << "Point attribute name \"" << b.name << "\" is more than " << (nanovdb::GridBlindMetaData::MaxNameSize-1) << " characters";
                OPENVDB_THROW(openvdb::ValueError, ss.str());
            }
        
            //const openvdb::Name codec = array.empty() ? "" : array[0]->constAttributeArray(b.index).codecType();
            memcpy(meta.mName, b.name.c_str(), b.name.size() + 1);
            if (b.typeName == "vec3s") {
                meta.mDataType = GridType::Vec3f;
                this->template copyPointAttribute<openvdb::Vec3f>(array, b.index);
                if (b.name == "P") {
                    meta.mSemantic = GridBlindDataSemantic::PointPosition;  
                } else if (b.name == "V") {
                    meta.mSemantic = GridBlindDataSemantic::PointVelocity;
                } else if (b.name == "Cd") {
                    meta.mSemantic = GridBlindDataSemantic::PointColor;
                } else if (b.name == "N") {
                    meta.mSemantic = GridBlindDataSemantic::PointNormal;
                } else {
                    meta.mSemantic = GridBlindDataSemantic::Unknown;
                }
            } else if (b.typeName == "int32") {
                meta.mDataType = GridType::Int32;
                this->template copyPointAttribute<int32_t>(array, b.index);
                if (b.name == "id") {
                    meta.mSemantic = GridBlindDataSemantic::PointId;
                } else {
                    meta.mSemantic = GridBlindDataSemantic::Unknown;
                }
            } else if (b.typeName == "int64") {
                meta.mDataType = GridType::Int64;
                this->template copyPointAttribute<int64_t>(array, b.index);
                if (b.name == "id") {
                    meta.mSemantic = GridBlindDataSemantic::PointId;
                } else {
                    meta.mSemantic = GridBlindDataSemantic::Unknown;
                }
            } else if (b.typeName == "float") {
                meta.mDataType = GridType::Float;
                meta.mSemantic = GridBlindDataSemantic::Unknown;
                this->template copyPointAttribute<float>(array, b.index);
            } else {
                std::stringstream ss;
                ss << "Unsupported point attribute type: \"" << b.typeName << "\"";
                OPENVDB_THROW(openvdb::ValueError, ss.str());
            }
        }
    } // loop over bind data
}

template<typename SrcTreeT, typename BufferT>
GridHandle<BufferT>
OpenToNanoVDB<SrcTreeT, BufferT>::operator()(const openvdb::Grid<SrcTreeT>& srcGrid,
                                             StatsMode                      sMode,
                                             ChecksumMode                   cMode,
                                             bool                           mortonSort,
                                             int                            verbose,
                                             const BufferT&                 allocator)
{
    openvdb::util::CpuTimer timer;
    auto                    startTimer = [&](const std::string& s) {
        if (verbose > 1) {
            timer.start(s);
        }
    };
    auto stopTimer = [&]() {
        if (verbose > 1) {
            timer.stop();
        }
    };

    const SrcTreeT& srcTree = srcGrid.tree();
    const SrcRootT& srcRoot = srcTree.root();

    startTimer("Extracting nodes from openvdb grid");
    std::vector<const SrcNode2*> array2; // upper OpenVDB internal nodes
    std::vector<const SrcNode1*> array1; // lower OpenVDB internal nodes
    std::vector<const SrcNode0*> array0; // OpenVDB leaf nodes
    array0.reserve(srcTree.leafCount()); // fast pre-allocation of OpenVDB leaf nodes (of which there are many)
    tbb::parallel_invoke([&]() { srcTree.getNodes(array0); }, // multi-threaded population of node arrays from OpenVDB tree
                         [&]() { srcTree.getNodes(array1); },
                         [&]() { srcTree.getNodes(array2); });
    stopTimer();

    if (srcRoot.getTableSize() != array2.size())
        OPENVDB_THROW(openvdb::RuntimeError, "Tiles at the root level are not supported yet!");

    auto key = [](const CoordT& p) { return DstRootT::DataType::CoordToKey(p); };
#ifdef USE_SINGLE_ROOT_KEY
    startTimer("Sorting " + std::to_string(array2.size()) + " child nodes of the root node");
    this->sortNodes(array2, key);
    stopTimer();
#endif
    assert(std::is_sorted(array2.begin(), array2.end(), [&key](const SrcNode2* a, const SrcNode2* b) { return key(a->origin()) < key(b->origin()); }));

    startTimer("Pre-processing metadata");
    this->preProcessMetadata(srcGrid, array0);
    stopTimer();

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
        startTimer("Morton sorting of " + std::to_string(array0.size()) + " leaf nodes");
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
        stopTimer();
    }

    startTimer("Allocating memory for the NanoVDB");
    GridHandle<BufferT> handle(allocator.create(this->gridSize()));
    mData = handle.data();
    stopTimer();

    if (verbose)
        openvdb::util::printBytes(std::cerr, this->gridSize(), "Allocated", " for the NanoVDB grid\n");

    if (srcGrid.getGridClass() == openvdb::GRID_LEVEL_SET) {
        mDelta = ValueT(srcGrid.voxelSize()[0]); // skip a node if max < -mDelta || min > mDelta
    } else {
        mDelta = ValueT(0); // dummy value
    }

    std::unique_ptr<int32_t[]> cache0(new int32_t[array0.size()]); // cache for byte offsets to leaf nodes

    startTimer("Processing leaf nodes");
    this->processLeafs(array0, cache0.get());
    stopTimer();

    std::unique_ptr<int32_t[]> cache1(new int32_t[array1.size()]); // cache for byte offsets to lower internal nodes

    startTimer("Processing lower internal nodes");
    this->processInternals<SrcNode1, DstNode1>(array1, cache1.get(), cache0.get());
    stopTimer();

    cache0.reset();
    std::unique_ptr<int32_t[]> cache2(new int32_t[array2.size()]); // cache for byte offsets to upper internal nodes

    startTimer("Processing upper internal nodes");
    this->processInternals<SrcNode2, DstNode2>(array2, cache2.get(), cache1.get());
    stopTimer();

    cache1.reset();

    startTimer("Processing Root node");
    this->processRoot(srcTree.root(), array2, cache2.get());
    stopTimer();

    cache2.reset();

    startTimer("Processing Tree");
    const uint64_t nodeCount[4] = {array0.size(), array1.size(), array2.size(), 1};
    this->processTree(nodeCount);
    stopTimer();

    startTimer("Processing Grid");
    this->processGrid(srcGrid);
    stopTimer();

    startTimer("Process metadata");
    this->processMetadata(srcGrid, array0);
    stopTimer();

    // Since point grids already mde use of min/max we should not re-compute them
    if (std::is_same<SrcBuildT, openvdb::PointIndex32>::value ||
        std::is_same<SrcBuildT, openvdb::PointDataIndex32>::value) {
        sMode = StatsMode::BBox;
    }

    auto* nanoGrid = reinterpret_cast<DstGridT*>(mData);

    startTimer("GridStats");
    gridStats(*nanoGrid, sMode);
    stopTimer();

    startTimer("Checksum");
    updateChecksum(*nanoGrid, cMode);
    stopTimer();

    mData = nullptr;
    return handle; // invokes move constructor
} // operator()

template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
inline typename std::enable_if<!std::is_same<typename SrcLeafT::BuildType, openvdb::ValueMask>::value && 
                               !std::is_same<typename SrcLeafT::BuildType, bool>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::processLeafs(std::vector<const SrcLeafT*>& array, int32_t* x0)
{
    static_assert(sizeof(ValueT) == sizeof(typename SrcLeafT::ValueType), "Mismatching sizeof");
    DstNode0* firstLeaf = this->template node<DstNode0>(); // address of first leaf node
    auto      op = [&](const tbb::blocked_range<uint32_t>& r) {
        int32_t* x = x0 + r.begin();
        auto*    dstLeaf = firstLeaf + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstLeaf) {
            const SrcLeafT* srcLeaf = array[i];
            auto*              data = dstLeaf->data();
            data->mValueMask = srcLeaf->valueMask(); // copy value mask
            data->mBBoxMin = srcLeaf->origin(); // copy origin of node
            data->mFlags = 0u;
            auto *src = reinterpret_cast<const ValueT*>(srcLeaf->buffer().data());
            for (ValueT *dst = data->mValues, *end = dst + SrcLeafT::size(); dst != end; dst += 4, src += 4) {
                dst[0] = src[0]; // copy *all* voxel values in sets of four, i.e. loop-unrolling
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
            cache(*x++, srcLeaf->origin(), i);
        }
    };
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(array.size()), 8), op);
} // processLeafs<T>

template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
inline typename std::enable_if<std::is_same<typename SrcLeafT::BuildType, bool>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::processLeafs(std::vector<const SrcLeafT*>& array, int32_t* x0)
{
    DstNode0* firstLeaf = this->template node<DstNode0>(); // address of first leaf node
    auto      op = [&](const tbb::blocked_range<uint32_t>& r) {
        int32_t* x = x0 + r.begin();
        auto*    dstLeaf = firstLeaf + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstLeaf) {
            const SrcLeafT* srcLeaf = array[i];
            auto*              data = dstLeaf->data();
            data->mValueMask = srcLeaf->valueMask(); // copy value mask
            data->mValues = *reinterpret_cast<const nanovdb::Mask<3>*>(srcLeaf->buffer().data()); // copy values
            data->mBBoxMin = srcLeaf->origin(); // copy origin of node
            data->mFlags = 0u;
            cache(*x++, srcLeaf->origin(), i);
        }
    };
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(array.size()), 8), op);
} // processLeafs<bool>

// Partial template specialization on ValueMask
template<typename SrcTreeT, typename BufferT>
template<typename SrcLeafT>
inline typename std::enable_if<std::is_same<typename SrcLeafT::BuildType, openvdb::ValueMask>::value>::type
OpenToNanoVDB<SrcTreeT, BufferT>::processLeafs(std::vector<const SrcLeafT*>& array, int32_t* x0)
{
    DstNode0* firstLeaf = this->template node<DstNode0>(); // address of first leaf node
    auto      op = [&](const tbb::blocked_range<uint32_t>& r) {
        int32_t* x = x0 + r.begin();
        auto*    dstLeaf = firstLeaf + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++dstLeaf) {
            const SrcLeafT* srcLeaf = array[i];
            auto*              data = dstLeaf->data();
            data->mValueMask = srcLeaf->valueMask(); // copy value mask
            data->mBBoxMin = srcLeaf->origin(); // copy origin of node
            data->mFlags = 0u;
            cache(*x++, srcLeaf->origin(), i);
        }
    };
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, static_cast<uint32_t>(array.size()), 8), op);
} // processLeafs<ValueMask>

// Partial template specialization on bool value type
template<typename SrcTreeT, typename BufferT>
template<typename SrcNode, typename DstNode>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    processInternals(std::vector<const SrcNode*>& array, int32_t* x0, const int32_t* childX)
{
    const uint32_t size = static_cast<uint32_t>(array.size());
    auto*          start = this->template nodeData<DstNode>();
    auto           op = [&](const tbb::blocked_range<size_t>& r) {
        int32_t* x = x0 + r.begin();
        auto*    data = start + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++data) {
            const SrcNode* srcNode = array[i];
            data->mBBox.min() = srcNode->origin();
            cache(*x++, srcNode->origin(), i);
            data->mOffset = size - i;
            data->mValueMask = srcNode->getValueMask();
            data->mChildMask = srcNode->getChildMask();
            for (auto tileIter = srcNode->cbeginValueAll(); tileIter; ++tileIter) {
                data->mTable[tileIter.pos()].value = *tileIter;
            }
            for (auto childIter = srcNode->cbeginChildOn(); childIter; ++childIter) {
                const auto childID = static_cast<uint32_t>(childIter->origin()[0]);
                data->mTable[childIter.pos()].childID = childID;
                const_cast<CoordT&>(childIter->origin())[0] = childX[childID]; // restore origin
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, array.size(), 4), op);
} // processInternals

template<typename SrcTreeT, typename BufferT>
void OpenToNanoVDB<SrcTreeT, BufferT>::
    processRoot(const SrcRootT& srcRoot, std::vector<const SrcNode2*>& array, const int32_t* childX)
{
    auto& data = *(this->template nodeData<DstRootT>());
    data.mBackground = srcRoot.background();
    data.mTileCount = srcRoot.getTableSize();
    data.mMinimum = data.mMaximum = data.mBackground;
    data.mBBox.min() = openvdb::Coord::max(); // set to an empty bounding box
    data.mBBox.max() = openvdb::Coord::min();
    data.mActiveVoxelCount = 0;

    // since openvdb::RootNode internally uses a std::map for child nodes its iterator
    // visits elements in the stored order required by the nanovdb::RootNode
    if (data.mTileCount > 0) {
        auto*      node = array[0];
        auto&      tile = data.tile(0);
        const auto childID = static_cast<uint32_t>(node->origin()[0]);
        const_cast<CoordT&>(node->origin())[0] = childX[childID]; // restore cached coordinate
        tile.setChild(node->origin(), childID);
        for (uint32_t i = 1, n = static_cast<uint32_t>(array.size()); i < n; ++i) {
            node = array[i];
            auto&      tile = data.tile(i);
            const auto childID = static_cast<uint32_t>(node->origin()[0]);
            const_cast<openvdb::Coord&>(node->origin())[0] = childX[childID]; // restore cached coordinate
            tile.setChild(node->origin(), childID);
        }
        for (auto iter = srcRoot.cbeginValueAll(); iter; ++iter) {
            OPENVDB_THROW(openvdb::RuntimeError, "Tiles at the root node is broken and needs to be fixed!");
            auto& tile = data.tile(iter.pos());
            tile.setValue(iter.getCoord(), iter.isValueOn(), *iter);
        }
    }
} // processRoot

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
    data.mPFSum[3] = 0;
    for (int i = 2; i >= 0; --i)
        data.mPFSum[i] = data.mPFSum[i + 1] + data.mCount[i + 1]; // reverse prefix sum
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
    data.mChecksum = 0u;
    data.mVersion = Version();
    data.mFlags = 0u;// disables all flags!
    data.mGridSize = this->gridSize();
    data.mWorldBBox = BBox<Vec3R>();
    data.mBlindMetadataOffset = mBlindMetaData.size() ? mBytes[1] : 0;
    data.mBlindMetadataCount = static_cast<uint32_t>(mBlindMetaData.size());

    const std::string gridName = srcGrid.getName();
    if (gridName.length() >= nanovdb::GridData::MaxNameSize) {
        data.setLongGridNameOn();
    } else {
        strcpy(data.mGridName, gridName.c_str());
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
    
    if (std::is_same<SrcBuildT, float>::value) { // resolved at compiletime
        data.mGridType = GridType::Float;
    } else if (std::is_same<SrcBuildT, double>::value) {
        data.mGridType = GridType::Double;
    } else if (std::is_same<SrcBuildT, int16_t>::value) {
        data.mGridType = GridType::Int16;
    } else if (std::is_same<SrcBuildT, int32_t>::value) {
        data.mGridType = GridType::Int32;
    } else if (std::is_same<SrcBuildT, int64_t>::value) {
        data.mGridType = GridType::Int64;
    } else if (std::is_same<SrcBuildT, openvdb::Vec3f>::value) {
        data.mGridType = GridType::Vec3f;
    } else if (std::is_same<SrcBuildT, openvdb::Index32>::value) {
        data.mGridType = GridType::UInt32;
    } else if (std::is_same<SrcBuildT, openvdb::PointIndex32>::value) {
        data.mGridType = GridType::UInt32;
        data.mGridClass = GridClass::PointIndex;
    } else if (std::is_same<SrcBuildT, openvdb::PointDataIndex32>::value) {
        data.mGridType = GridType::UInt32;
        data.mGridClass = GridClass::PointData;
     } else if (std::is_same<SrcBuildT, openvdb::ValueMask>::value) {
        data.mGridType = GridType::Mask;
        data.mGridClass = GridClass::Topology;
     } else if (std::is_same<SrcBuildT, bool>::value) {
        data.mGridType = GridType::Boolean;
    } else {
        OPENVDB_THROW(openvdb::ValueError, "Unsupported value type");
    }
    { // set affine map
        if (srcGrid.hasUniformVoxels()) {
            data.mVoxelSize = nanovdb::Vec3R(affineMap->voxelSize()[0]);
        } else {
            data.mVoxelSize = affineMap->voxelSize();
        }
        const auto mat = affineMap->getMat4();
        // Only support non-tapered at the moment:
        data.mMap.set(mat, mat.inverse(), 1.0);
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
openToNanoVDB(const openvdb::Grid<SrcTreeT>& grid,
              StatsMode                      sMode,
              ChecksumMode                   cMode,
              bool                           mortonSort,
              int                            verbose)
{
    OpenToNanoVDB<SrcTreeT, BufferT> s;
    return s(grid, sMode, cMode, mortonSort, verbose);
}

template<typename BufferT>
GridHandle<BufferT>
openToNanoVDB(const openvdb::GridBase::Ptr& base,
              StatsMode                     sMode,
              ChecksumMode                  cMode,
              bool                          mortonSort,
              int                           verbose)
{
    if (auto grid = openvdb::GridBase::grid<openvdb::FloatGrid>(base)) {
        return openToNanoVDB<BufferT, openvdb::FloatTree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::DoubleGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::DoubleTree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int32Grid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Int32Tree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Int64Grid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Int64Tree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Grid<openvdb::UInt32Tree>>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::UInt32Tree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3fGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Vec3fTree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::Vec3dGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::Vec3dTree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::points::PointDataGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::points::PointDataTree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::MaskGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::MaskTree>(*grid, sMode, cMode, mortonSort, verbose);
    } else if (auto grid = openvdb::GridBase::grid<openvdb::BoolGrid>(base)) {
        return nanovdb::openToNanoVDB<BufferT, openvdb::BoolTree>(*grid, sMode, cMode, mortonSort, verbose);
    } else {
        OPENVDB_THROW(openvdb::RuntimeError, "Unrecognized OpenVDB grid type");
    }
}

} // namespace nanovdb

#endif // NANOVDB_OPENTONANOVDB_H_HAS_BEEN_INCLUDED
