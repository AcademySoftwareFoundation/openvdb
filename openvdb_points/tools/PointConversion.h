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
/// @brief  Convert a generic point data structure into a VDB Point Data Grid.
///


#ifndef OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tree/LeafNodeBool.h>
#include <openvdb/util/Name.h>

#include <openvdb/tools/PointIndexGrid.h>

#include <utility> // std::pair, std::make_pair
#include <map>
#include <limits> // for std::numeric_limit

#include <boost/scoped_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/utility/enable_if.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief  Stores points within leaf nodes of a point data grid.
///
/// @param points   list of world space points.
/// @param xform    world to index space transform.
template<typename PointDataGridT, typename PointArrayT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointArrayT& points, const math::Transform& xform);

/// @brief  Stores points within leaf nodes of a point data grid.
///
/// @param pointIndexGrid   a PointIndexGrid into the points
/// @param points   list of world space points.
/// @param xform    world to index space transform.
template<typename PointDataGridT, typename PointArrayT, typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid,
    const PointArrayT& points, const math::Transform& xform);

////////////////////////////////////////

/// @brief A list of wrapped point attributes
/// @param PositionT vector type used to represent the position attribute
/// @param PointAttributeT PointAttribute class
///
/// @note The point attributes are application-specific
/// and must match the following interface
/// @code
/// struct PointAttribute {
///   typedef shared_ptr<PointAttribute> Ptr;
///   typedef VectorType position_type;
///   PointAttrPair info() const;
///   const Name& name() const;
///   AttrType type() const;
///   Index64 size() const;
///
///   template <typename T>
///   struct Accessor {
///   typedef shared_ptr<Accessor<T> > Ptr;
///   void getValue(Index64, T& xyz) const; };
///
///   template <typename T> Accessor<T>::Ptr getAccessor() const;
/// };
/// @endcode


template <typename PositionT, typename PointAttributeT>
class PointAttributeList
{
public:
    typedef PositionT                                                   value_type;
    typedef PositionT                                                   position_type;
    typedef PointAttributeT                                             PointAttributeType;
    typedef typename PointAttributeType::Ptr                            PointAttributePtr;
    typedef typename PointAttributeType::template Accessor<PositionT>   PositionAccessor;
    typedef typename PositionAccessor::Ptr                              PositionAccessorPtr;
    typedef boost::shared_ptr<PointAttributeList>                       Ptr;
    typedef typename std::vector<PointAttributePtr>::const_iterator     const_iterator;
    typedef AttributeSet::Descriptor                                    Descriptor;

    explicit PointAttributeList(PointAttributePtr positionAttribute)
        : mPositionAttribute(positionAttribute)
        , mPositionAccessor(mPositionAttribute->template getAccessor<PositionT>())
        , mSize(mPositionAttribute->size())
    {
        // implicitly insert position into attribute list

        mAttributes.push_back(positionAttribute);
    }

    Index64 attributes() const { return mAttributes.size(); }

    PointAttributePtr attribute(const unsigned index) const { return mAttributes[index]; }

    Index64 size() const { return mSize; }

    void getPos(Index64 n, PositionT& xyz) const {
        mPositionAccessor->getValue(n, xyz);
    }

    void addAttribute(PointAttributePtr attribute)
    {
        for (const_iterator it = cbegin(), itEnd = cend(); it != itEnd; ++it)
        {
            if (attribute->name() == (*it)->name()) {
                throw std::runtime_error("Duplicate attribute: " + attribute->name());
            }
        }

        mAttributes.push_back(attribute);

        // sort the elements after every insertion

        std::sort(mAttributes.begin(), mAttributes.end());
    }

    const_iterator cbegin() const { return mAttributes.begin(); }
    const_iterator cend() const { return mAttributes.end(); }

    Descriptor::Ptr attributeDescriptor() const {
        Descriptor::Ptr descriptor;

        std::vector<Descriptor::NameAndType> vec;

        for (const_iterator it = cbegin(), it_end = cend(); it != it_end; ++it) {
            vec.push_back(Descriptor::NameAndType((*it)->name(), (*it)->type()));
        }

        return Descriptor::create(vec);
    }

private:
    PointAttributePtr mPositionAttribute;
    PositionAccessorPtr mPositionAccessor;
    Index64 mSize;

    std::vector<PointAttributePtr> mAttributes;
}; // class PointAttributeList

// Internal operators

namespace point_data_grid_internal {

template<typename PointDataTreeType, typename PointIndexTreeType, typename PointArray>
struct InitialiseAttributesOp {

    typedef typename tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef typename PointIndexTreeType::LeafNodeType PointIndexLeafNode;
    typedef typename PointIndexLeafNode::IndexArray IndexArray;

    InitialiseAttributesOp( PointDataTreeType& tree,
                            const PointIndexTreeType& pointIndexTree,
                            const PointArray& points,
                            const AttributeSet::Descriptor::Ptr& attributeDescriptor)
        : mTree(tree)
        , mPointIndexTree(pointIndexTree)
        , mPoints(&points)
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
    PointArray      const * const           mPoints;
    const AttributeSet::Descriptor::Ptr&    mAttributeDescriptor;
};

template<   typename PointDataTreeType,
            typename PointIndexTreeType,
            typename HoudiniAttributeType,
            typename AccessorType>
struct PopulateAttributeOp {

    typedef typename tree::LeafManager<PointDataTreeType>       LeafManagerT;
    typedef typename LeafManagerT::LeafRange                    LeafRangeT;
    typedef typename PointDataTreeType::LeafNodeType            PointDataLeafNode;

    typedef typename PointIndexTreeType::LeafNodeType           PointIndexLeafNode;
    typedef typename PointIndexLeafNode::IndexArray             IndexArray;

    PopulateAttributeOp(const PointIndexTreeType& pointIndexTree,
                        const AccessorType& accessor,
                        const unsigned attributeIndex)
        : mPointIndexTree(pointIndexTree)
        , mAccessor(accessor)
        , mAttributeIndex(attributeIndex) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        HoudiniAttributeType value;

        for (typename LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            AttributeWriteHandle<HoudiniAttributeType>* attributeWriteHandle = leaf->template attributeWriteHandle<HoudiniAttributeType>(mAttributeIndex);

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (typename IndexArray::const_iterator it = indices.begin(), it_end = indices.end(); it != it_end; ++it)
            {
                mAccessor.getValue(*it, value);

                attributeWriteHandle->set(index, value);

                index++;
            }
        }
    }

    //////////

    const PointIndexTreeType&   mPointIndexTree;
    const AccessorType          mAccessor;
    const unsigned              mAttributeIndex;
};

template<   typename PointDataTreeType,
            typename PointIndexTreeType,
            typename HoudiniAttributeType,
            typename AccessorType>
struct PopulatePositionAttributeOp {

    typedef typename tree::LeafManager<PointDataTreeType> LeafManagerT;
    typedef typename LeafManagerT::LeafRange LeafRangeT;

    typedef typename PointIndexTreeType::LeafNodeType PointIndexLeafNode;
    typedef typename PointIndexLeafNode::IndexArray IndexArray;

    PopulatePositionAttributeOp(const PointIndexTreeType& pointIndexTree,
                                const math::Transform& transform,
                                const AccessorType& accessor,
                                const unsigned attributeIndex)
        : mPointIndexTree(pointIndexTree)
        , mTransform(transform)
        , mAccessor(accessor)
        , mAttributeIndex(attributeIndex) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        HoudiniAttributeType positionWorldSpace;

        for (typename LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            // obtain the PointIndexLeafNode (using the origin of the current leaf)

            const PointIndexLeafNode* pointIndexLeaf = mPointIndexTree.probeConstLeaf(leaf->origin());

            if (!pointIndexLeaf)    continue;

            AttributeWriteHandle<Vec3f>* attributeWriteHandle = leaf->template attributeWriteHandle<Vec3f>(mAttributeIndex);

            Index64 index = 0;

            const IndexArray& indices = pointIndexLeaf->indices();

            for (typename IndexArray::const_iterator it = indices.begin(), it_end = indices.end(); it != it_end; ++it)
            {
                mAccessor.getValue(*it, positionWorldSpace);

                const HoudiniAttributeType positionIndexSpace = mTransform.worldToIndex(positionWorldSpace);

                const HoudiniAttributeType positionVoxelSpace = HoudiniAttributeType(
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
    const AccessorType          mAccessor;
    const unsigned              mAttributeIndex;
};

template <typename PointAttributeT, typename AttributeT, typename PointDataTreeT, typename PointIndexTreeT>
void populatePositionAttribute(PointDataTreeT& tree, const PointIndexTreeT& pointIndexTree, const math::Transform& xform, PointAttributeT& attribute, const unsigned index)
{
    typedef typename PointAttributeT::template Accessor<AttributeT> AttributeAccessor;

    typename AttributeAccessor::Ptr accessor = attribute.template getAccessor<AttributeT>();

    PopulatePositionAttributeOp<PointDataTreeT,
                                PointIndexTreeT,
                                AttributeT,
                                AttributeAccessor> populate(pointIndexTree,
                                                            xform,
                                                            *accessor,
                                                            index);

    typename tree::template LeafManager<PointDataTreeT> leafManager(tree);
    tbb::parallel_for(leafManager.leafRange(), populate);
}

template <typename PointAttributeT, typename AttributeT, typename PointDataTreeT, typename PointIndexTreeT>
void populateAttribute(PointDataTreeT& tree, const PointIndexTreeT& pointIndexTree, PointAttributeT& attribute, const unsigned index)
{
    typedef typename PointAttributeT::template Accessor<AttributeT> AttributeAccessor;

    typename AttributeAccessor::Ptr accessor = attribute.template getAccessor<AttributeT>();

    PopulateAttributeOp<PointDataTreeT,
                        PointIndexTreeT,
                        AttributeT,
                        AttributeAccessor> populate(pointIndexTree,
                                                    *accessor,
                                                    index);

    typename tree::template LeafManager<PointDataTreeT> leafManager(tree);
    tbb::parallel_for(leafManager.leafRange(), populate);
}

} // namespace point_data_grid_internal


////////////////////////////////////////

/// @brief  Localises point attributes into a @c PointDataGrid into two stages:
///         allocation of the leaf attribute data and population of this data.
///
/// @note   The position data is populated separately to perform world space to
///         voxel space conversion and apply quantisation.
///
/// @note   A @c PointIndexGrid to the points must be supplied to perform this
///         operation. Typically this is built implicitly by the PointDataGrid constructor.

template<typename PointDataGridT, typename PointArrayT, typename PointIndexGridT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointIndexGridT& pointIndexGrid,
    const PointArrayT& points, const math::Transform& xform)
{
    typedef typename PointDataGridT::TreeType                       PointDataTreeT;
    typedef typename PointIndexGridT::TreeType                      PointIndexTreeT;
    typedef typename tree::template LeafManager<PointDataTreeT>     LeafManagerT;
    typedef typename LeafManagerT::LeafRange                        LeafRangeT;
    typedef typename PointArrayT::PointAttributeType                PointAttributeT;
    typedef typename PointArrayT::position_type                     PositionT;
    typedef typename PointArrayT::PositionAccessor                  PositionAccessorT;

    using point_data_grid_internal::InitialiseAttributesOp;
    using point_data_grid_internal::populateAttribute;
    using point_data_grid_internal::populatePositionAttribute;

    using math::Vec3;

    // construct the Tree using a topology copy of the PointIndexGrid

    const PointIndexTreeT& pointIndexTree(pointIndexGrid.tree());
    typename PointDataTreeT::Ptr treePtr(new PointDataTreeT(pointIndexTree));

    LeafManagerT leafManager = LeafManagerT(*treePtr);
    LeafRangeT leafRange = leafManager.leafRange();

    // create point attribute map

    AttributeSet::Descriptor::Ptr attributeDescriptor(points.attributeDescriptor());

    // create point attribute storage on each leaf

    InitialiseAttributesOp<PointDataTreeT, PointIndexTreeT, PointArrayT> initialise(
                                *treePtr, pointIndexGrid.tree(), points, attributeDescriptor);
    tbb::parallel_for(leafRange, initialise);

    // populate attributes

    for (typename PointArrayT::const_iterator it = points.cbegin(), it_end = points.cend();
            it != it_end; ++it) {

        const Name& name = (*it)->name();
        const unsigned index = attributeDescriptor->find(name);

        const openvdb::tools::NamePair& attrType = (*it)->type();

        typename PointAttributeT::Ptr attribute = points.attribute(index);

        if (name == "P") {
            // "P" (i.e. position) attributes are constructed differently since
            // they can be stored in quantized values

            if (attrType.first == "vec3s") {
                populatePositionAttribute<PointAttributeT, openvdb::Vec3f>(*treePtr, pointIndexTree, xform, *attribute, index);
            }
            else {
                throw std::runtime_error("Unknown Position Attribute Type for Conversion: " + attrType.first);
            }
        }
        else {

            if (attrType.first == "bool") {
                populateAttribute<PointAttributeT, bool>(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "int16") {
                populateAttribute<PointAttributeT, short>(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "int32") {
                populateAttribute<PointAttributeT, int>(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "int64") {
                populateAttribute<PointAttributeT, long>(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "half") {
                populateAttribute<PointAttributeT, half>(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "float") {
                populateAttribute<PointAttributeT, float>(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "double") {
                populateAttribute<PointAttributeT, double>(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "vec3h") {
                populateAttribute<PointAttributeT, openvdb::math::Vec3<half> >(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "vec3s") {
                populateAttribute<PointAttributeT, openvdb::math::Vec3<float> >(*treePtr, pointIndexTree, *attribute, index);
            }
            else if (attrType.first == "vec3d") {
                populateAttribute<PointAttributeT, openvdb::math::Vec3<double> >(*treePtr, pointIndexTree, *attribute, index);
            }
            else {
                throw std::runtime_error("Unknown Attribute Type for Conversion: " + attrType.first);
            }
        }
    }

    typename PointDataGridT::Ptr grid = PointDataGridT::create(treePtr);
    grid->setTransform(xform.copy());
    return grid;
}

template <typename PointDataGridT, typename PointArrayT>
inline typename PointDataGridT::Ptr
createPointDataGrid(const PointArrayT& points, const math::Transform& xform)
{
    PointIndexGrid::Ptr pointGridPtr = createPointIndexGrid<PointIndexGrid>(points, xform);
    return createPointDataGrid<PointDataGridT>(*pointGridPtr, points, xform);
}

////////////////////////////////////////

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_CONVERSION_HAS_BEEN_INCLUDED


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
