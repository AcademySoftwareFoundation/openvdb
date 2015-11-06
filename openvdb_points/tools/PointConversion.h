///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
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
/// @brief  Convert existing points and attributes into VDB Point Data grids and attributes.
///


#ifndef OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/math/Transform.h>

#include <openvdb/tools/PointIndexGrid.h>

#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb_points/tools/PointDataGrid.h>

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
///
/// @note   The position data must be supplied in a Point-Partitioner compatible
///         data structure. A convenience PointAttributeVector class is offered.
///
/// @note   The position data is populated separately to perform world space to
///         voxel space conversion and apply quantisation.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. Typically this is built implicitly by the PointDataGrid constructor.

template<typename PointDataGridT, typename PositionArrayT, typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid, const PositionArrayT& positions,
                    const openvdb::NamePair& positionType, const math::Transform& xform);


/// @brief  Convenience method to create a @c PointDataGrid from a std::vector of
///         point positions.
///
/// @param  positions     list of world space point positions.
/// @param  positionType  the type of the position (includes compression info).
/// @param  xform         world to index space transform.
///
/// @note   This method implicitly wraps the std::vector for a Point-Partitioner compatible
///         data structure and creates the required @c PointIndexGrid to the points.

template <typename PointDataGridT, typename ValueT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const std::vector<ValueT>& positions,
                    const openvdb::NamePair& positionType, const math::Transform& xform);


/// @brief  Stores point attribute data in an existing @c PointDataGrid attribute.
///
/// @param  tree            the PointDataGrid to be populated.
/// @param  pointIndexTree  a PointIndexTree into the points.
/// @param  attributeName   the name of the VDB Points attribute to be populated.
/// @param  data            a wrapper to the attribute data.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. This is required to ensure the same point index ordering.

template <typename PointDataTreeT, typename PointIndexTreeT, typename PointArrayT>
inline void
populateAttribute(  PointDataTreeT& tree, const PointIndexTreeT& pointIndexTree,
                    const openvdb::Name& attributeName, const PointArrayT& data);


////////////////////////////////////////


/// @brief Point-partitioner compatible STL vector attribute wrapper for convenience
template<typename ValueType>
class PointAttributeVector {
public:
    typedef ValueType value_type;

    PointAttributeVector(const std::vector<value_type>& data)
        : mData(data) { }

    size_t size() const { return mData.size(); }
    void getPos(size_t n, ValueType& xyz) const { xyz = mData[n]; }

    template <typename T>
    void get(size_t n, T& value) const { value = mData[n]; }

private:
    const std::vector<value_type>& mData;
}; // PointAttributeVector


////////////////////////////////////////


namespace point_conversion_internal {

template<typename PointDataTreeType, typename PointIndexTreeType>
struct InitialiseAttributesOp {

    typedef typename tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef typename PointIndexTreeType::LeafNodeType PointIndexLeafNode;
    typedef typename PointIndexLeafNode::IndexArray IndexArray;

    InitialiseAttributesOp( PointDataTreeType& tree,
                            const PointIndexTreeType& pointIndexTree,
                            const AttributeSet::Descriptor::Ptr& attributeDescriptor)
        : mTree(tree)
        , mPointIndexTree(pointIndexTree)
        , mAttributeDescriptor(attributeDescriptor) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {
        for (typename LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            // initialise the attribute storage

            pointIndexLeaf->indices();

            const IndexArray& indices = pointIndexLeaf->indices();

            Index64 pointCount = indices.size();

            leaf->initializeAttributes(mAttributeDescriptor, pointCount);
        }
    }

    //////////

    const PointDataTreeType&                mTree;
    const PointIndexTreeType&               mPointIndexTree;
    const AttributeSet::Descriptor::Ptr&    mAttributeDescriptor;
};

template<   typename PointDataTreeType,
            typename PointIndexTreeType,
            typename PositionListType>
struct PopulatePositionAttributeOp {

    typedef typename tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef typename PointIndexTreeType::LeafNodeType PointIndexLeafNode;
    typedef typename PointIndexLeafNode::IndexArray IndexArray;

    typedef typename PositionListType::value_type ValueType;

    PopulatePositionAttributeOp(const PointIndexTreeType& pointIndexTree,
                                const math::Transform& transform,
                                const PositionListType& positions)
        : mPointIndexTree(pointIndexTree)
        , mTransform(transform)
        , mPositions(positions) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (typename LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            typename AttributeWriteHandle<ValueType>::Ptr attributeWriteHandle =
                AttributeWriteHandle<ValueType>::create(leaf->template attributeArray("P"));

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (typename IndexArray::const_iterator it = indices.begin(), it_end = indices.end(); it != it_end; ++it)
            {
                ValueType positionWorldSpace;
                mPositions.getPos(*it, positionWorldSpace);

                const ValueType positionIndexSpace = mTransform.worldToIndex(positionWorldSpace);

                const ValueType positionVoxelSpace = ValueType(
                            positionIndexSpace.x() - math::Round(positionIndexSpace.x()),
                            positionIndexSpace.y() - math::Round(positionIndexSpace.y()),
                            positionIndexSpace.z() - math::Round(positionIndexSpace.z()));

                attributeWriteHandle->set(index, positionVoxelSpace);

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
            typename AttributeListType>
struct PopulateAttributeOp {

    typedef typename tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef typename PointIndexTreeType::LeafNodeType PointIndexLeafNode;
    typedef typename PointIndexLeafNode::IndexArray IndexArray;

    typedef typename AttributeListType::value_type ValueType;

    PopulateAttributeOp(const PointIndexTreeType& pointIndexTree,
                        const AttributeListType& data,
                        const openvdb::Name& attributeName)
        : mPointIndexTree(pointIndexTree)
        , mData(data)
        , mAttributeName(attributeName) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (typename LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            typename AttributeWriteHandle<ValueType>::Ptr attributeWriteHandle =
                AttributeWriteHandle<ValueType>::create(leaf->attributeArray(mAttributeName));

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (typename IndexArray::const_iterator it = indices.begin(), it_end = indices.end(); it != it_end; ++it)
            {
                ValueType value;
                mData.template get<ValueType>(*it, value);

                attributeWriteHandle->set(index, value);

                index++;
            }
        }
    }

    //////////

    const PointIndexTreeType&   mPointIndexTree;
    const AttributeListType&    mData;
    const openvdb::Name&        mAttributeName;
};

} // namespace point_conversion_internal


////////////////////////////////////////


template<typename PointDataGridT, typename PositionArrayT, typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid, const PositionArrayT& positions,
                    const openvdb::NamePair& positionType, const math::Transform& xform)
{
    typedef typename PointDataGridT::TreeType                       PointDataTreeT;
    typedef typename PointIndexGridT::TreeType                      PointIndexTreeT;
    typedef typename tree::template LeafManager<PointDataTreeT>     LeafManagerT;
    typedef typename LeafManagerT::LeafRange                        LeafRangeT;

    using point_conversion_internal::InitialiseAttributesOp;
    using point_conversion_internal::PopulatePositionAttributeOp;

    // construct the Tree using a topology copy of the PointIndexGrid

    const PointIndexTreeT& pointIndexTree(pointIndexGrid.tree());
    typename PointDataTreeT::Ptr treePtr(new PointDataTreeT(pointIndexTree));

    LeafManagerT leafManager = LeafManagerT(*treePtr);
    LeafRangeT leafRange = leafManager.leafRange();

    // create attribute descriptor from position type

    AttributeSet::Descriptor::Ptr descriptor = AttributeSet::Descriptor::create(positionType);

    // create point attribute storage on each leaf

    InitialiseAttributesOp<PointDataTreeT, PointIndexTreeT> initialise(
                                *treePtr, pointIndexGrid.tree(), descriptor);
    tbb::parallel_for(leafRange, initialise);

    // populate position attribute

    PopulatePositionAttributeOp<PointDataTreeT,
                                PointIndexTreeT,
                                PositionArrayT> populate(pointIndexTree,
                                                        xform,
                                                        positions);

    tbb::parallel_for(leafRange, populate);

    typename PointDataGridT::Ptr grid = PointDataGridT::create(treePtr);
    grid->setTransform(xform.copy());
    return grid;
}


////////////////////////////////////////


template <typename PointDataGridT, typename ValueT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const std::vector<ValueT>& positions,
                    const openvdb::NamePair& positionType,
                    const math::Transform& xform)
{
    const PointAttributeVector<ValueT> pointList(positions);

    PointIndexGrid::Ptr pointIndexGrid = createPointIndexGrid<PointIndexGrid>(pointList, xform);
    return createPointDataGrid<PointDataGridT>(*pointIndexGrid, pointList, positionType, xform);
}


////////////////////////////////////////


template <typename PointDataTreeT, typename PointIndexTreeT, typename PointArrayT>
inline void
populateAttribute(  PointDataTreeT& tree, const PointIndexTreeT& pointIndexTree,
                    const openvdb::Name& attributeName, const PointArrayT& data)
{
    using point_conversion_internal::PopulateAttributeOp;

    // populate attribute

    PopulateAttributeOp<PointDataTreeT,
                        PointIndexTreeT,
                        PointArrayT> populate(pointIndexTree, data, attributeName);

    tbb::parallel_for(typename tree::template LeafManager<PointDataTree>(tree).leafRange(), populate);
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
