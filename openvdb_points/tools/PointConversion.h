///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @author Dan Bailey
///
/// @file PointConversion.h
///
/// @brief  Convert points and attributes to and from VDB Point Data grids.
///


#ifndef OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED

#include <openvdb/math/Transform.h>

#include <openvdb/tools/PointIndexGrid.h>

#include <openvdb_points/tools/AttributeArrayString.h>
#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb_points/tools/IndexFilter.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointGroup.h>

#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief  Localises points with position into a @c PointDataGrid into two stages:
///         allocation of the leaf attribute data and population of the positions.
///
/// @param  pointIndexGrid  a PointIndexGrid into the points.
/// @param  positions       list of world space point positions.
/// @param  positionType    the type of the position (includes compression info).
/// @param  xform           world to index space transform.
/// @param  positionDefaultValue metadata default position value
///
/// @note   The position data must be supplied in a Point-Partitioner compatible
///         data structure. A convenience PointAttributeVector class is offered.
///
/// @note   The position data is populated separately to perform world space to
///         voxel space conversion and apply quantisation.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. Typically this is built implicitly by the PointDataGrid constructor.

template <typename CompressionT, typename PointDataGridT, typename PositionArrayT, typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid, const PositionArrayT& positions,
                    const math::Transform& xform, Metadata::Ptr positionDefaultValue = Metadata::Ptr());


/// @brief  Convenience method to create a @c PointDataGrid from a std::vector of
///         point positions.
///
/// @param  positions     list of world space point positions.
/// @param  positionType  the type of the position (includes compression info).
/// @param  xform         world to index space transform.
/// @param  positionDefaultValue metadata default position value
///
/// @note   This method implicitly wraps the std::vector for a Point-Partitioner compatible
///         data structure and creates the required @c PointIndexGrid to the points.

template <typename CompressionT, typename PointDataGridT, typename ValueT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const std::vector<ValueT>& positions, const math::Transform& xform,
                    Metadata::Ptr positionDefaultValue = Metadata::Ptr());


/// @brief  Stores point attribute data in an existing @c PointDataGrid attribute.
///
/// @param  tree            the PointDataGrid to be populated.
/// @param  pointIndexTree  a PointIndexTree into the points.
/// @param  attributeName   the name of the VDB Points attribute to be populated.
/// @param  data            a wrapper to the attribute data.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. This is required to ensure the same point index ordering.

template <typename PointDataTreeT, typename PointIndexTreeT, typename PointArrayT, bool Strided>
inline void
populateAttribute(  PointDataTreeT& tree, const PointIndexTreeT& pointIndexTree,
                    const openvdb::Name& attributeName, const PointArrayT& data,
                    const Index stride = 1);


/// @brief Convert the position attribute from a Point Data Grid
///
/// @param positionAttribute    the position attribute to be populated.
/// @param grid                 the PointDataGrid to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf
/// @param startOffset          a value to shift all the point offsets by
/// @param includeGroups        a vector of VDB Points groups to be included (default is all)
/// @param excludeGroups        a vector of VDB Points groups to be excluded (default is none)
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
///

template <typename PositionAttribute, typename PointDataGridT>
inline void
convertPointDataGridPosition(   PositionAttribute& positionAttribute,
                                const PointDataGridT& grid,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const std::vector<Name>& includeGroups = std::vector<Name>(),
                                const std::vector<Name>& excludeGroups = std::vector<Name>(),
                                const bool inCoreOnly = true);


/// @brief Convert the attribute from a PointDataGrid
///
/// @param attribute            the attribute to be populated.
/// @param tree                 the PointDataTree to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf.
/// @param startOffset          a value to shift all the point offsets by
/// @param arrayIndex           the index in the Descriptor of the array to be converted.
/// @param includeGroups        a vector of VDB Points groups to be included (default is all)
/// @param excludeGroups        a vector of VDB Points groups to be excluded (default is none)
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
///

template <typename TypedAttribute, typename PointDataTreeT>
inline void
convertPointDataGridAttribute(  TypedAttribute& attribute,
                                const PointDataTreeT& tree,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const unsigned arrayIndex,
                                const Index stride = 1,
                                const std::vector<Name>& includeGroups = std::vector<Name>(),
                                const std::vector<Name>& excludeGroups = std::vector<Name>(),
                                const bool inCoreOnly = true);


/// @brief Convert the group from a PointDataGrid
///
/// @param group                the group to be populated.
/// @param tree                 the PointDataTree to be converted.
/// @param pointOffsets         a vector of cumulative point offsets for each leaf
/// @param startOffset          a value to shift all the point offsets by
/// @param index                the group index to be converted.
/// @param includeGroups        a vector of VDB Points groups to be included (default is all)
/// @param excludeGroups        a vector of VDB Points groups to be excluded (default is none)
/// @param inCoreOnly           true if out-of-core leaf nodes are to be ignored
///

template <typename Group, typename PointDataTreeT>
inline void
convertPointDataGridGroup(  Group& group,
                            const PointDataTreeT& tree,
                            const std::vector<Index64>& pointOffsets,
                            const Index64 startOffset,
                            const AttributeSet::Descriptor::GroupIndex index,
                            const std::vector<Name>& includeGroups = std::vector<Name>(),
                            const std::vector<Name>& excludeGroups = std::vector<Name>(),
                            const bool inCoreOnly = true);


////////////////////////////////////////


/// @brief Point-partitioner compatible STL vector attribute wrapper for convenience
template<typename ValueType>
class PointAttributeVector {
public:
    using PosType = ValueType;
    using value_type= ValueType;

    PointAttributeVector(const std::vector<value_type>& data,
                         const Index stride = 1)
        : mData(data)
        , mStride(stride) { }

    size_t size() const { return mData.size(); }
    void getPos(size_t n, ValueType& xyz) const { xyz = mData[n]; }

    template <typename T>
    void get(T& value, size_t n) const { value = mData[n]; }

    template <typename T>
    void get(T& value, size_t n, openvdb::Index m) const { value = mData[n * mStride + m]; }

private:
    const std::vector<value_type>& mData;
    const Index mStride;
}; // PointAttributeVector


////////////////////////////////////////


namespace point_conversion_internal {


// ConversionTraits to create the relevant Attribute Handles from a LeafNode
template <bool Stride, typename T> struct ConversionTraits
{
    using Handle = AttributeHandle<T, UnknownCodec, Stride>;
    using WriteHandle = AttributeWriteHandle<T, UnknownCodec, Stride>;
    static T zero() { return T(0); }
    template <typename LeafT>
    static typename Handle::Ptr handleFromLeaf(LeafT& leaf, Index index) {
        const AttributeArray& array = leaf.constAttributeArray(index);
        return Handle::create(array);
    }
    template <typename LeafT>
    static typename WriteHandle::Ptr writeHandleFromLeaf(LeafT& leaf, Index index) {
        AttributeArray& array = leaf.attributeArray(index);
        return WriteHandle::create(array);
    }
}; // ConversionTraits
template <> struct ConversionTraits<false, openvdb::Name>
{
    using Handle = StringAttributeHandle;
    using WriteHandle = StringAttributeWriteHandle;
    static openvdb::Name zero() { return ""; }
    template <typename LeafT>
    static typename Handle::Ptr handleFromLeaf(LeafT& leaf, Index index) {
        const AttributeArray& array = leaf.constAttributeArray(index);
        const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();
        return Handle::create(array, descriptor.getMetadata());
    }
    template <typename LeafT>
    static typename WriteHandle::Ptr writeHandleFromLeaf(LeafT& leaf, Index index) {
        AttributeArray& array = leaf.attributeArray(index);
        const AttributeSet::Descriptor& descriptor = leaf.attributeSet().descriptor();
        return WriteHandle::create(array, descriptor.getMetadata());
    }
}; // ConversionTraits<openvdb::Name>


template<typename PointDataTreeType, typename PointIndexTreeType>
struct InitialiseAttributesOp {

    using LeafManagerT          = typename tree::LeafManager<PointDataTreeType>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;

    using PointIndexLeafNode    = typename PointIndexTreeType::LeafNodeType;
    using IndexArray            = typename PointIndexLeafNode::IndexArray;

    InitialiseAttributesOp( const PointIndexTreeType& pointIndexTree,
                            const AttributeSet::Descriptor::Ptr& attributeDescriptor)
        : mPointIndexTree(pointIndexTree)
        , mAttributeDescriptor(attributeDescriptor) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {
        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            // initialise the attribute storage

            const IndexArray& indices = pointIndexLeaf->indices();

            Index64 pointCount = indices.size();

            leaf->initializeAttributes(mAttributeDescriptor, pointCount);
        }
    }

    //////////

    const PointIndexTreeType&               mPointIndexTree;
    const AttributeSet::Descriptor::Ptr&    mAttributeDescriptor;
};

template<   typename PointDataTreeType,
            typename PointIndexTreeType,
            typename PositionListType>
struct PopulatePositionAttributeOp {

    using LeafManagerT          = typename tree::LeafManager<PointDataTreeType>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;

    using PointIndexLeafNode    = typename PointIndexTreeType::LeafNodeType;
    using IndexArray            = typename PointIndexLeafNode::IndexArray;

    using ValueType             = typename PositionListType::value_type;

    PopulatePositionAttributeOp(const PointIndexTreeType& pointIndexTree,
                                const math::Transform& transform,
                                const PositionListType& positions)
        : mPointIndexTree(pointIndexTree)
        , mTransform(transform)
        , mPositions(positions) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            auto attributeWriteHandle = AttributeWriteHandle<Vec3f>::create(leaf->attributeArray("P"));

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (const Index64& i : indices)
            {
                ValueType positionWorldSpace;
                mPositions.getPos(i, positionWorldSpace);

                const ValueType positionIndexSpace = mTransform.worldToIndex(positionWorldSpace);

                const ValueType positionVoxelSpace = ValueType(
                            positionIndexSpace.x() - math::Round(positionIndexSpace.x()),
                            positionIndexSpace.y() - math::Round(positionIndexSpace.y()),
                            positionIndexSpace.z() - math::Round(positionIndexSpace.z()));

                attributeWriteHandle->set(index, Vec3f(positionVoxelSpace));

                index++;
            }
        }
    }

    //////////

    const PointIndexTreeType&   mPointIndexTree;
    const math::Transform&      mTransform;
    const PositionListType&     mPositions;
};

template<   typename PointDataTreeType,
            typename PointIndexTreeType,
            typename AttributeListType,
            bool Stride = false>
struct PopulateAttributeOp {

    using LeafManagerT          = typename tree::LeafManager<PointDataTreeType>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;
    using PointIndexLeafNode    = typename PointIndexTreeType::LeafNodeType;
    using IndexArray            = typename PointIndexLeafNode::IndexArray;
    using ValueType             = typename AttributeListType::value_type;
    using HandleT               = typename ConversionTraits<Stride, ValueType>::WriteHandle;

    PopulateAttributeOp(const PointIndexTreeType& pointIndexTree,
                        const AttributeListType& data,
                        const size_t index,
                        const Index stride = 1)
        : mPointIndexTree(pointIndexTree)
        , mData(data)
        , mIndex(index)
        , mStride(stride) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            typename HandleT::Ptr attributeWriteHandle = ConversionTraits<Stride, ValueType>::writeHandleFromLeaf(*leaf, mIndex);

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (const Index64& leafIndex : indices)
            {
                ValueType value;
                if (Stride) {
                    for (Index i = 0; i < mStride; i++) {
                        mData.template get<ValueType>(value, leafIndex, i);
                        attributeWriteHandle->set(index, i, value);
                    }
                }
                else {
                    mData.template get<ValueType>(value, leafIndex);
                    attributeWriteHandle->set(index, 0, value);
                }
                index++;
            }

            // attempt to compact the array

            attributeWriteHandle->compact();
        }
    }

    //////////

    const PointIndexTreeType&   mPointIndexTree;
    const AttributeListType&    mData;
    const size_t                mIndex;
    const Index                 mStride;
};

template<typename PointDataTreeType, typename Attribute>
struct ConvertPointDataGridPositionOp {

    using LeafNode      = typename PointDataTreeType::LeafNodeType;
    using ValueType     = typename Attribute::ValueType;
    using LeafManagerT  = typename tree::LeafManager<const PointDataTreeType>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    ConvertPointDataGridPositionOp( Attribute& attribute,
                                    const std::vector<Index64>& pointOffsets,
                                    const Index64 startOffset,
                                    const math::Transform& transform,
                                    const size_t index,
                                    const std::vector<Name>& includeGroups = std::vector<Name>(),
                                    const std::vector<Name>& excludeGroups = std::vector<Name>(),
                                    const bool inCoreOnly = true)
        : mAttribute(attribute)
        , mPointOffsets(pointOffsets)
        , mStartOffset(startOffset)
        , mTransform(transform)
        , mIndex(index)
        , mIncludeGroups(includeGroups)
        , mExcludeGroups(excludeGroups)
        , mInCoreOnly(inCoreOnly)
    {
        // only accept Vec3f as ValueType
        static_assert(VecTraits<ValueType>::Size == 3 &&
                      std::is_floating_point<typename ValueType::ValueType>::value,
                      "ValueType is not Vec3f");
    }

    void operator()(const LeafRangeT& range) const {

        const bool useGroups = !mIncludeGroups.empty() || !mExcludeGroups.empty();

        typename Attribute::Handle pHandle(mAttribute);

        for (auto leaf = range.begin(); leaf; ++leaf) {

            assert(leaf.pos() < mPointOffsets.size());

#ifndef OPENVDB_2_ABI_COMPATIBLE
            if (mInCoreOnly && leaf->buffer().isOutOfCore())    continue;
#endif

            Index64 offset = mStartOffset;

            if (leaf.pos() > 0)     offset += mPointOffsets[leaf.pos() - 1];

            auto handle = AttributeHandle<ValueType>::create(leaf->constAttributeArray(mIndex));

            if (useGroups) {
                auto iter = leaf->beginIndexOn(MultiGroupFilter(mIncludeGroups, mExcludeGroups));

                for (; iter; ++iter) {
                    const Vec3d xyz = iter.getCoord().asVec3d();
                    const Vec3d pos = handle->get(Index64(*iter));
                    pHandle.set(offset++, /*stride=*/ 0, mTransform.indexToWorld(pos + xyz));
                }
            }
            else {
                auto iter = leaf->beginIndexOn();

                for (; iter; ++iter) {
                    const Vec3d xyz = iter.getCoord().asVec3d();
                    const Vec3d pos = handle->get(Index64(*iter));
                    pHandle.set(offset++, /*stride=*/ 0, mTransform.indexToWorld(pos + xyz));
                }
            }
        }
    }

    //////////

    Attribute&                              mAttribute;
    const std::vector<Index64>&             mPointOffsets;
    const Index64                           mStartOffset;
    const math::Transform&                  mTransform;
    const size_t                            mIndex;
    const std::vector<std::string>&         mIncludeGroups;
    const std::vector<std::string>&         mExcludeGroups;
    const bool                              mInCoreOnly;
}; // ConvertPointDataGridPositionOp


template<typename PointDataTreeType, typename Attribute, bool Stride = false>
struct ConvertPointDataGridAttributeOp {

    using LeafNode      = typename PointDataTreeType::LeafNodeType;
    using ValueType     = typename Attribute::ValueType;
    using HandleT       = typename ConversionTraits<Stride, ValueType>::Handle;
    using LeafManagerT  = typename tree::LeafManager<const PointDataTreeType>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    ConvertPointDataGridAttributeOp(Attribute& attribute,
                                    const std::vector<Index64>& pointOffsets,
                                    const Index64 startOffset,
                                    const size_t index,
                                    const Index stride = 1,
                                    const std::vector<Name>& includeGroups = std::vector<Name>(),
                                    const std::vector<Name>& excludeGroups = std::vector<Name>(),
                                    const bool inCoreOnly = true)
        : mAttribute(attribute)
        , mPointOffsets(pointOffsets)
        , mStartOffset(startOffset)
        , mIndex(index)
        , mStride(stride)
        , mIncludeGroups(includeGroups)
        , mExcludeGroups(excludeGroups)
        , mInCoreOnly(inCoreOnly) { }

    void operator()(const LeafRangeT& range) const {

        const bool useGroups = !mIncludeGroups.empty() || !mExcludeGroups.empty();

        typename Attribute::Handle pHandle(mAttribute);

        for (auto leaf = range.begin(); leaf; ++leaf) {

            assert(leaf.pos() < mPointOffsets.size());

#ifndef OPENVDB_2_ABI_COMPATIBLE
            if (mInCoreOnly && leaf->buffer().isOutOfCore())    continue;
#endif

            Index64 offset = mStartOffset;

            if (leaf.pos() > 0)     offset += mPointOffsets[leaf.pos() - 1];

            typename HandleT::Ptr handle = ConversionTraits<Stride, ValueType>::handleFromLeaf(*leaf, mIndex);

            const bool uniform = handle->isUniform();

            ValueType uniformValue = ConversionTraits<Stride, ValueType>::zero();
            if (uniform)    uniformValue = ValueType(handle->get(0));

            if (useGroups) {
                auto iter = leaf->beginIndexOn(MultiGroupFilter(mIncludeGroups, mExcludeGroups));

                if (uniform) {
                    for (; iter; ++iter) {
                        for (Index i = 0; i < mStride; i++) {
                            pHandle.set(offset, i, uniformValue);
                        }
                        offset++;
                    }
                }
                else {
                    for (; iter; ++iter) {
                        for (Index i = 0; i < mStride; i++) {
                            pHandle.set(offset, i, handle->get(Index64(*iter)));
                        }
                        offset++;
                    }
                }
            }
            else {
                auto iter = leaf->beginIndexOn();

                if (uniform) {
                    for (; iter; ++iter) {
                        for (Index i = 0; i < mStride; i++) {
                            pHandle.set(offset, i, uniformValue);
                        }
                        offset++;
                    }
                }
                else {
                    for (; iter; ++iter) {
                        for (Index i = 0; i < mStride; i++) {
                            pHandle.set(offset, i, handle->get(Index64(*iter), /*stride=*/i));
                        }
                        offset++;
                    }
                }
            }
        }
    }

    //////////

    Attribute&                              mAttribute;
    const std::vector<Index64>&             mPointOffsets;
    const Index64                           mStartOffset;
    const size_t                            mIndex;
    const Index                             mStride;
    const std::vector<std::string>&         mIncludeGroups;
    const std::vector<std::string>&         mExcludeGroups;
    const bool                              mInCoreOnly;
}; // ConvertPointDataGridAttributeOp

template<typename PointDataTreeType, typename Group>
struct ConvertPointDataGridGroupOp {

    using LeafNode      = typename PointDataTreeType::LeafNodeType;
    using GroupIndex    = AttributeSet::Descriptor::GroupIndex;
    using LeafManagerT  = typename tree::LeafManager<const PointDataTreeType>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    ConvertPointDataGridGroupOp(Group& group,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const AttributeSet::Descriptor::GroupIndex index,
                                const std::vector<Name>& includeGroups = std::vector<Name>(),
                                const std::vector<Name>& excludeGroups = std::vector<Name>(),
                                const bool inCoreOnly = true)
        : mGroup(group)
        , mPointOffsets(pointOffsets)
        , mStartOffset(startOffset)
        , mIndex(index)
        , mIncludeGroups(includeGroups)
        , mExcludeGroups(excludeGroups)
        , mInCoreOnly(inCoreOnly) { }

    void operator()(const LeafRangeT& range) const {

        const bool useGroups = !mIncludeGroups.empty() || !mExcludeGroups.empty();

        for (auto leaf = range.begin(); leaf; ++leaf) {

            assert(leaf.pos() < mPointOffsets.size());

#ifndef OPENVDB_2_ABI_COMPATIBLE
            if (mInCoreOnly && leaf->buffer().isOutOfCore())    continue;
#endif

            Index64 offset = mStartOffset;

            if (leaf.pos() > 0)     offset += mPointOffsets[leaf.pos() - 1];

            const AttributeArray& array = leaf->constAttributeArray(mIndex.first);
            const GroupType bitmask = GroupType(1) << mIndex.second;

            assert(isGroup(array));

            const GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

            const bool uniform = groupArray.isUniform();

            if (uniform) {
                if (!(groupArray.get(0) & bitmask))     continue;
            }

            if (useGroups) {
                auto iter = leaf->beginIndexOn(MultiGroupFilter(mIncludeGroups, mExcludeGroups));

                if (uniform) {
                    for (; iter; ++iter) {
                        mGroup.setOffsetOn(offset);
                        offset++;
                    }
                }
                else {
                    for (; iter; ++iter) {
                        if (groupArray.get(*iter) & bitmask) {
                            mGroup.setOffsetOn(offset);
                        }
                        offset++;
                    }
                }
            }
            else {
                auto iter = leaf->beginIndexOn();

                if (uniform) {
                    for (; iter; ++iter) {
                        mGroup.setOffsetOn(offset);
                        offset++;
                    }
                }
                else {
                    for (; iter; ++iter) {
                        if (groupArray.get(*iter) & bitmask) {
                            mGroup.setOffsetOn(offset);
                        }
                        offset++;
                    }
                }
            }
        }
    }

    //////////

    Group&                                  mGroup;
    const std::vector<Index64>&             mPointOffsets;
    const Index64                           mStartOffset;
    const GroupIndex                        mIndex;
    const std::vector<std::string>&         mIncludeGroups;
    const std::vector<std::string>&         mExcludeGroups;
    const bool                              mInCoreOnly;
}; // ConvertPointDataGridGroupOp


} // namespace point_conversion_internal


////////////////////////////////////////


template<typename CompressionT, typename PointDataGridT, typename PositionArrayT, typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid, const PositionArrayT& positions,
                    const math::Transform& xform, Metadata::Ptr positionDefaultValue)
{
    using PointDataTreeT        = typename PointDataGridT::TreeType;
    using PointIndexTreeT       = typename PointIndexGridT::TreeType;
    using LeafManagerT          = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT            = typename LeafManagerT::LeafRange;
    using PositionAttributeT    = TypedAttributeArray<Vec3f, CompressionT>;

    using point_conversion_internal::InitialiseAttributesOp;
    using point_conversion_internal::PopulatePositionAttributeOp;

    const NamePair positionType = PositionAttributeT::attributeType();

    // construct the Tree using a topology copy of the PointIndexGrid

    const PointIndexTreeT& pointIndexTree(pointIndexGrid.tree());
    typename PointDataTreeT::Ptr treePtr(new PointDataTreeT(pointIndexTree));

    LeafManagerT leafManager = LeafManagerT(*treePtr);
    LeafRangeT leafRange = leafManager.leafRange();

    // create attribute descriptor from position type

    auto descriptor = AttributeSet::Descriptor::create(positionType);

    // add default value for position if provided

    if (positionDefaultValue)   descriptor->setDefaultValue("P", *positionDefaultValue);

    // create point attribute storage on each leaf

    InitialiseAttributesOp<PointDataTreeT, PointIndexTreeT> initialise(
                                pointIndexGrid.tree(), descriptor);
    tbb::parallel_for(leafRange, initialise);

    // populate position attribute

    PopulatePositionAttributeOp<PointDataTreeT,
                                PointIndexTreeT,
                                PositionArrayT> populate(pointIndexTree,
                                                        xform,
                                                        positions);

    tbb::parallel_for(leafRange, populate);

    auto grid = PointDataGridT::create(treePtr);
    grid->setTransform(xform.copy());
    return grid;
}


////////////////////////////////////////


template <typename CompressionT, typename PointDataGridT, typename ValueT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const std::vector<ValueT>& positions,
                    const math::Transform& xform,
                    Metadata::Ptr positionDefaultValue)
{
    const PointAttributeVector<ValueT> pointList(positions);

    PointIndexGrid::Ptr pointIndexGrid = createPointIndexGrid<PointIndexGrid>(pointList, xform);
    return createPointDataGrid<CompressionT, PointDataGridT>(*pointIndexGrid, pointList, xform, positionDefaultValue);
}


////////////////////////////////////////


template <typename PointDataTreeT, typename PointIndexTreeT, typename PointArrayT, bool Strided>
inline void
populateAttribute(  PointDataTreeT& tree, const PointIndexTreeT& pointIndexTree,
                    const openvdb::Name& attributeName, const PointArrayT& data, const Index stride)
{
    using point_conversion_internal::PopulateAttributeOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const size_t index = iter->attributeSet().find(attributeName);

    if (index == AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError, "Attribute not found to populate - " << attributeName << ".");
    }

    // populate attribute

    typename tree::LeafManager<PointDataTreeT> leafManager(tree);

    PopulateAttributeOp<PointDataTreeT,
                        PointIndexTreeT,
                        PointArrayT,
                        Strided> populate(pointIndexTree, data, index, stride);
    tbb::parallel_for(leafManager.leafRange(), populate);
}


////////////////////////////////////////


template <typename PositionAttribute, typename PointDataGridT>
inline void
convertPointDataGridPosition(   PositionAttribute& positionAttribute,
                                const PointDataGridT& grid,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const std::vector<Name>& includeGroups,
                                const std::vector<Name>& excludeGroups,
                                const bool inCoreOnly)
{
    using TreeType      = typename PointDataGridT::TreeType;
    using LeafManagerT  = typename tree::LeafManager<const TreeType>;

    using point_conversion_internal::ConvertPointDataGridPositionOp;

    const TreeType& tree = grid.tree();
    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    // for efficiency, keep only groups that are present in the Descriptor

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    std::vector<Name> newIncludeGroups(includeGroups);
    std::vector<Name> newExcludeGroups(excludeGroups);

    deleteMissingPointGroups(newIncludeGroups, descriptor);
    deleteMissingPointGroups(newExcludeGroups, descriptor);

    LeafManagerT leafManager(tree);

    const size_t positionIndex = iter->attributeSet().find("P");

    positionAttribute.expand();
    ConvertPointDataGridPositionOp<TreeType, PositionAttribute> convert(
                    positionAttribute, pointOffsets, startOffset, grid.transform(), positionIndex,
                    newIncludeGroups, newExcludeGroups, inCoreOnly);
    tbb::parallel_for(leafManager.leafRange(), convert);
    positionAttribute.compact();
}


////////////////////////////////////////


template <typename TypedAttribute, typename PointDataTreeT>
inline void
convertPointDataGridAttribute(  TypedAttribute& attribute,
                                const PointDataTreeT& tree,
                                const std::vector<Index64>& pointOffsets,
                                const Index64 startOffset,
                                const unsigned arrayIndex,
                                const Index stride,
                                const std::vector<Name>& includeGroups,
                                const std::vector<Name>& excludeGroups,
                                const bool inCoreOnly)
{
    using LeafManagerT = typename tree::LeafManager<const PointDataTreeT>;

    using point_conversion_internal::ConvertPointDataGridAttributeOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    // for efficiency, keep only groups that are present in the Descriptor

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    std::vector<Name> newIncludeGroups(includeGroups);
    std::vector<Name> newExcludeGroups(excludeGroups);

    deleteMissingPointGroups(newIncludeGroups, descriptor);
    deleteMissingPointGroups(newExcludeGroups, descriptor);

    LeafManagerT leafManager(tree);

    attribute.expand();
    if (stride == 1) {
        ConvertPointDataGridAttributeOp<PointDataTreeT, TypedAttribute> convert(
                        attribute, pointOffsets, startOffset, arrayIndex, stride,
                        newIncludeGroups, newExcludeGroups, inCoreOnly);
        tbb::parallel_for(leafManager.leafRange(), convert);
    }
    else {
        ConvertPointDataGridAttributeOp<PointDataTreeT, TypedAttribute, /*Stride=*/true> convert(
                        attribute, pointOffsets, startOffset, arrayIndex, stride,
                        newIncludeGroups, newExcludeGroups, inCoreOnly);
        tbb::parallel_for(leafManager.leafRange(), convert);
    }
    attribute.compact();
}


////////////////////////////////////////


template <typename Group, typename PointDataTreeT>
inline void
convertPointDataGridGroup(  Group& group,
                            const PointDataTreeT& tree,
                            const std::vector<Index64>& pointOffsets,
                            const Index64 startOffset,
                            const AttributeSet::Descriptor::GroupIndex index,
                            const std::vector<Name>& includeGroups,
                            const std::vector<Name>& excludeGroups,
                            const bool inCoreOnly)
{
    using LeafManagerT= typename tree::LeafManager<const PointDataTreeT>;

    using point_conversion_internal::ConvertPointDataGridGroupOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    // for efficiency, keep only groups that are present in the Descriptor

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();

    std::vector<Name> newIncludeGroups(includeGroups);
    std::vector<Name> newExcludeGroups(excludeGroups);

    deleteMissingPointGroups(newIncludeGroups, descriptor);
    deleteMissingPointGroups(newExcludeGroups, descriptor);

    LeafManagerT leafManager(tree);

    ConvertPointDataGridGroupOp<PointDataTree, Group> convert(
                    group, pointOffsets, startOffset, index,
                    newIncludeGroups, newExcludeGroups, inCoreOnly);
    tbb::parallel_for(leafManager.leafRange(), convert);

    // must call this after modifying point groups in parallel

    group.finalize();
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
