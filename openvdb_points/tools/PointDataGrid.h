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
/// @author Dan Bailey, Nick Avramoussis, Matt Warner
///
/// @file PointDataGrid.h
///
/// @brief  Attribute-owned data structure for points. Point attributes are
///         stored in leaf nodes and ordered by voxel for fast random and
///         sequential access.
///


#ifndef OPENVDB_TOOLS_POINT_DATA_GRID_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_DATA_GRID_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>

#include <openvdb/tools/PointIndexGrid.h>

#include <openvdb_points/tools/AttributeSet.h>

#include <utility> // std::pair, std::make_pair


class TestPointDataLeaf;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

// forward declaration
namespace tree {
    template<Index, typename> struct SameLeafConfig;
}


////////////////////////////////////////


namespace tools {


// forward declaration
template<typename T, Index Log2Dim> class PointDataLeafNode;

/// @brief Point index tree configured to match the default VDB configurations.
typedef tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode
    <PointDataLeafNode<PointDataIndex32, 3>, 4>, 5> > > PointDataTree;


/// @brief Point data grid.
typedef Grid<PointDataTree> PointDataGrid;


////////////////////////////////////////

namespace point_masks { enum PointCountMask { Active = 0, Inactive, All }; }

template<typename TreeT>
struct PointDataAccessor {
    typedef typename TreeT::ValueType       ValueType;
    typedef typename TreeT::LeafNodeType    LeafNode;
    typedef typename TreeT::LeafNodeType::ValueType PointIndexT;
    typedef typename TreeT::LeafNodeType::ValueTypePair PointDataIndex;

    explicit PointDataAccessor(const TreeT& tree)
        : mTree(&tree)
        , mAccessor(tree) { }

    /// @{
    /// @brief Individual voxel access.

    PointDataIndex get(const Coord& ijk) const;

    /// Returns the number of points at a particular leaf/voxel coordinate. Does
    /// not take into account the active state of the leaf/voxel
    Index64 pointCount(const Coord& ijk) const;

    /// Returns the total number of points in the entire tree pointed to by
    /// this accessor.
    /// @param mask   a PointCountMask which can be used to only increment the total
    /// count returned based on the active state of the leaves/voxels. By default,
    /// returns the total number of points stored in active voxels
    Index64 totalPointCount(const point_masks::PointCountMask mask = point_masks::Active) const;

    tree::ValueAccessor<const TreeT>& valueAccessor() const { return mAccessor; }

    /// @}

    /// Return a reference to the tree associated with this accessor.
    const TreeT& tree() const { assert(mTree); return *mTree; }

private:
    typedef tree::ValueAccessor<const TreeT>  ConstAccessor;
    const TreeT* mTree;
    mutable ConstAccessor mAccessor;
}; // struct PointDataAccessor

// PointDataAccessor implementation

template<typename TreeT>
inline typename PointDataAccessor<TreeT>::PointDataIndex
PointDataAccessor<TreeT>::get(const Coord& ijk) const
{
    typedef typename TreeT::LeafNodeType::ValueType PointIndexT;

    const LeafNode* leaf = mAccessor.probeConstLeaf(ijk);

    // leaf not active - no particles

    if (!leaf) return std::make_pair(0, 0);

    const unsigned index = LeafNode::coordToOffset(ijk);

    return leaf->pointIndex(index);
}


template<typename TreeT>
inline Index64
PointDataAccessor<TreeT>::pointCount(const Coord& ijk) const
{
    PointDataIndex index = get(ijk);

    return index.second - index.first;
}

template<typename TreeT>
inline Index64
PointDataAccessor<TreeT>::totalPointCount(const point_masks::PointCountMask mask) const
{
    typedef typename TreeT::LeafCIter LeafCIter;

    Index64 count = 0;

    for (LeafCIter iter = mTree->cbeginLeaf(); iter; ++iter) {
        const LeafNode* leaf = iter.getLeaf();

        Index64 size = leaf->pointCount(mask);

        count += size;
    }

    return count;
}

////////////////////////////////////////

// Internal utility methods
namespace point_data_grid_internal {

template<typename T>
struct UniquePtr
{
#ifdef OPENVDB_HAS_CXX11
    typedef std::unique_ptr<T>  type;
#else
    typedef std::auto_ptr<T>    type;
#endif
};
}


template <typename T, Index Log2Dim>
class PointDataLeafNode : public tree::LeafNode<T, Log2Dim> {

public:
    typedef PointDataLeafNode<T, Log2Dim>           LeafNodeType;
    typedef boost::shared_ptr<PointDataLeafNode>    Ptr;

    typedef T                                       ValueType;
    typedef std::pair<ValueType, ValueType>         ValueTypePair;
    typedef std::vector<ValueType>                  IndexArray;

    typedef AttributeSet::Descriptor                Descriptor;

    ////////////////////////////////////////

    // The following methods had to be copied from the LeafNode class
    // to make the derived PointDataLeafNode class compatible with the tree structure.

    typedef tree::LeafNode<T, Log2Dim>    BaseLeaf;
    typedef util::NodeMask<Log2Dim> NodeMaskType;

    using BaseLeaf::LOG2DIM;
    using BaseLeaf::TOTAL;
    using BaseLeaf::DIM;
    using BaseLeaf::NUM_VALUES;
    using BaseLeaf::NUM_VOXELS;
    using BaseLeaf::SIZE;
    using BaseLeaf::LEVEL;

    /// Default constructor
    PointDataLeafNode()
        : BaseLeaf()
        , mAttributeSet(new AttributeSet) { }

    ~PointDataLeafNode() { }

    /// Construct using deep copy of other PointDataLeafNode
    explicit PointDataLeafNode(const PointDataLeafNode& other)
        : BaseLeaf(other)
        , mAttributeSet(new AttributeSet(*other.mAttributeSet)) { }

    /// Construct using supplied origin, value and active status
    explicit
    PointDataLeafNode(const Coord& coords, const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(coords, value, active)
        , mAttributeSet(new AttributeSet) { }

    /// Construct using supplied origin, value and active status
    /// use attribute map from another PointDataLeafNode
    PointDataLeafNode(const PointDataLeafNode& other, const Coord& coords, const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(coords, value, active)
        , mAttributeSet(new AttributeSet(*other.mAttributeSet)) { }

    // Copy-construct from a PointIndexLeafNode with the same configuration but a different ValueType.
    template<typename OtherValueType>
    PointDataLeafNode(const tools::PointIndexLeafNode<OtherValueType, Log2Dim>& other)
        : BaseLeaf(other)
        , mAttributeSet(new AttributeSet) { }

    // Copy-construct from a LeafNode with the same configuration but a different ValueType.
    // Used for topology copies - explicitly sets the value (background) to zeroVal
    template <typename ValueType>
    PointDataLeafNode(const tree::LeafNode<ValueType, Log2Dim>& other, const T& value, TopologyCopy)
        : BaseLeaf(other, zeroVal<T>(), TopologyCopy())
        , mAttributeSet(new AttributeSet) { }

    // Copy-construct from a LeafNode with the same configuration but a different ValueType.
    // Used for topology copies - explicitly sets the on and off value (background) to zeroVal
    template <typename ValueType>
    PointDataLeafNode(const tree::LeafNode<ValueType, Log2Dim>& other, const T& offValue, const T& onValue, TopologyCopy)
        : BaseLeaf(other, zeroVal<T>(), zeroVal<T>(), TopologyCopy())
        , mAttributeSet(new AttributeSet) { }

#ifndef OPENVDB_2_ABI_COMPATIBLE
    PointDataLeafNode(PartialCreate, const Coord& coords,
        const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(PartialCreate(), coords, value, active)
        , mAttributeSet(new AttributeSet) { }
#endif

public:

    const AttributeSet& attributeSet() const
    {
        return *mAttributeSet;
    }

    /// @brief Create a new attribute set. Any existing attributes will be
    /// cleared.
    void initializeAttributes(const Descriptor::Ptr& descriptor, const size_t arrayLength)
    {
        mAttributeSet.reset(new AttributeSet(descriptor, arrayLength));
    }

    /// @brief Free all attribute storage
    void clearAttributes()
    {
        mAttributeSet.reset(new AttributeSet(mAttributeSet->descriptorPtr(), 0));

        for (Index n = 0; n < LeafNodeType::NUM_VALUES; n++) {
            this->setOffsetOn(n, 0);
        }
    }

    /// @brief Returns whether an attribute exists. This method is faster
    /// than hasAttribute(const Name&) as it avoids a map lookup.
    /// @param pos    Index of the attribute
    template <typename TypedAttributeArrayType>
    bool hasAttribute(const size_t pos) const
    {
        if (pos >= mAttributeSet->size())     return false;

        const AttributeArray* array = mAttributeSet->getConst(pos);

        return array->isType<TypedAttributeArrayType>();
    }

    /// @brief Returns whether an attribute exists.
    /// @param attributeName    Name of the attribute
    template <typename TypedAttributeArrayType>
    bool hasAttribute(const Name& attributeName) const
    {
        const size_t pos = mAttributeSet->find(attributeName);

        if (pos == AttributeSet::INVALID_POS)   return false;

        return hasAttribute<TypedAttributeArrayType>(pos);
    }

    /// @brief Append an attribute to the leaf.
    void appendAttribute(const AttributeSet::Util::NameAndType& attribute,
                         const Descriptor& expected, Descriptor::Ptr& replacement)
    {
        mAttributeSet->appendAttribute(attribute, expected, replacement);
    }

    template <typename AttributeType>
    typename AttributeWriteHandle<AttributeType>::Ptr attributeWriteHandle(const size_t pos)
    {
        if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range");

        AttributeArray* array = mAttributeSet->get(pos);

        return AttributeWriteHandle<AttributeType>::create(*array);
    }

    template <typename AttributeType>
    typename AttributeWriteHandle<AttributeType>::Ptr attributeWriteHandle(const Name& attributeName)
    {
        const size_t pos = mAttributeSet->find(attributeName);

        if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found");

        AttributeArray* array = mAttributeSet->get(pos);

        return AttributeWriteHandle<AttributeType>::create(*array);
    }

    template <typename AttributeType>
    typename AttributeHandle<AttributeType>::Ptr attributeHandle(const size_t pos) const
    {
        if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range");

        AttributeArray* array = mAttributeSet->get(pos);

        return AttributeHandle<AttributeType>::create(*array);
    }

    template <typename AttributeType>
    typename AttributeHandle<AttributeType>::Ptr attributeHandle(const Name& attributeName) const
    {
        const size_t pos = mAttributeSet->find(attributeName);

        if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found");

        AttributeArray* array = mAttributeSet->get(pos);

        return AttributeHandle<AttributeType>::create(*array);
    }

    template <typename TypedAttributeType>
    TypedAttributeType& typedAttributeArray(const size_t pos)
    {
        if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range");

        AttributeArray* array = mAttributeSet->get(pos);

        if (!array->isType<TypedAttributeType>())     OPENVDB_THROW(LookupError, "Invalid Attribute Type");

        return static_cast<TypedAttributeType& >(*array);
    }

    template <typename TypedAttributeType>
    TypedAttributeType& typedAttributeArray(const Name& attributeName)
    {
        const size_t pos = mAttributeSet->find(attributeName);

        if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found");

        AttributeArray* array = mAttributeSet->get(pos);

        if (!array->isType<TypedAttributeType>())     OPENVDB_THROW(LookupError, "Invalid Attribute Type");

        return static_cast<TypedAttributeType& >(*array);
    }

    template <typename TypedAttributeType>
    const TypedAttributeType& typedAttributeArray(const size_t pos) const
    {
        if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range");

        const AttributeArray* array = mAttributeSet->getConst(pos);

        if (!array->isType<TypedAttributeType>())     OPENVDB_THROW(LookupError, "Invalid Attribute Type");

        return static_cast<const TypedAttributeType& >(*array);
    }

    template <typename TypedAttributeType>
    const TypedAttributeType& typedAttributeArray(const Name& attributeName) const
    {
        const size_t pos = mAttributeSet->find(attributeName);

        if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found");

        const AttributeArray* array = mAttributeSet->getConst(pos);

        if (!array->isType<TypedAttributeType>())     OPENVDB_THROW(LookupError, "Invalid Attribute Type");

        return static_cast<const TypedAttributeType& >(*array);
    }

    ValueTypePair pointIndex(const unsigned index) const
    {
        assert(index < BaseLeaf::SIZE);

        const ValueType end = this->getValue(index);

        const ValueType start = (index == 0) ? ValueType(0) : this->getValue(index - 1);

        return std::make_pair(start, end);
    }

    Index64 pointCount(const unsigned n) const
    {
        ValueTypePair index = this->pointIndex(n);

        return index.second - index.first;
    }

    Index64 pointCount(const point_masks::PointCountMask mask = point_masks::Active) const
    {
        const util::NodeMask<Log2Dim>& valueMask = this->getValueMask();

        // return the number of points in every voxel if the mask matches the ValueMask
        // or mask == All
        if ((mask == point_masks::All)                              ||
            (mask == point_masks::Active && valueMask.isOn())       ||
            (mask == point_masks::Inactive && valueMask.isOff()))   return this->getValue(NUM_VOXELS - 1);

        // if the leaf is off and we are querying active points, or
        // if the leaf is on and we are querying inactive points, return 0
        if ((mask == point_masks::Active && valueMask.isOff())          ||
            (mask == point_masks::Inactive && valueMask.isOn()))    return 0;

        // otherwise portions of the leaf are inactive/active. loop to find these
        // depending on the mask
        Index64 totalPointCount = 0;

        const bool active = mask == point_masks::Active;
        const bool inactive = mask == point_masks::Inactive;

        for (unsigned i = 0; i < NUM_VOXELS; i++)
        {
            bool valueOn = this->isValueOn(i);
            if((valueOn && active) || (!valueOn && inactive))
            {
                ValueTypePair index = this->pointIndex(i);
                totalPointCount += (index.second - index.first);
            }
        }

        return totalPointCount;
    }

    void deactivateEmptyVoxels()
    {
        for (Index n = 0; n < LeafNodeType::NUM_VALUES; n++) {
            if (this->pointCount(n) > 0)         continue;

            this->setValueMaskOff(n);
        }
    }

    ////////////////////////////////////////

    void setOffsetOn(Index offset, const ValueType& val)
    {
        this->buffer().setValue(offset, val);
        this->setValueMaskOn(offset);
    }

    void setOffsetOnly(Index offset, const ValueType& val)
    {
        this->buffer().setValue(offset, val);
    }


    /// @brief Return @c true if the given node (which may have a different @c ValueType
    /// than this node) has the same active value topology as this node.
    template<typename OtherType, Index OtherLog2Dim>
    bool hasSameTopology(const PointDataLeafNode<OtherType, OtherLog2Dim>* other) const {
        return BaseLeaf::hasSameTopology(other);
    }

    /// Check for buffer, state and origin equivalence first.
    /// If this returns true, do a deeper comparison on the attribute set to check
    bool operator==(const PointDataLeafNode& other) const {
        if(BaseLeaf::operator==(other) != true) return false;
        return (*this->mAttributeSet == *other.mAttributeSet);
    }

    bool operator!=(const PointDataLeafNode& other) const { return !(other == *this); }

    void addLeaf(PointDataLeafNode*) {}
    template<typename AccessorT>
    void addLeafAndCache(PointDataLeafNode*, AccessorT&) {}

    //@{
    /// @brief Return a pointer to this node.
    PointDataLeafNode* touchLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointDataLeafNode* touchLeafAndCache(const Coord&, AccessorT&) { return this; }

    template<typename NodeT, typename AccessorT>
    NodeT* probeNodeAndCache(const Coord&, AccessorT&)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(boost::is_same<NodeT,PointDataLeafNode>::value)) return NULL;
        return reinterpret_cast<NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    PointDataLeafNode* probeLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointDataLeafNode* probeLeafAndCache(const Coord&, AccessorT&) { return this; }
    //@}

    //@{
    /// @brief Return a @const pointer to this node.
    const PointDataLeafNode* probeConstLeaf(const Coord&) const { return this; }
    template<typename AccessorT>
    const PointDataLeafNode* probeConstLeafAndCache(const Coord&, AccessorT&) const { return this; }
    template<typename AccessorT>
    const PointDataLeafNode* probeLeafAndCache(const Coord&, AccessorT&) const { return this; }
    const PointDataLeafNode* probeLeaf(const Coord&) const { return this; }
    template<typename NodeT, typename AccessorT>
    const NodeT* probeConstNodeAndCache(const Coord&, AccessorT&) const
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(boost::is_same<NodeT,PointDataLeafNode>::value)) return NULL;
        return reinterpret_cast<const NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    //@}

    // I/O methods

    void readTopology(std::istream& is, bool fromHalf = false);
    void writeTopology(std::ostream& os, bool toHalf = false) const;

    void readBuffers(std::istream& is, bool fromHalf = false);
    void readBuffers(std::istream& is, const CoordBBox&, bool fromHalf = false);
    void writeBuffers(std::ostream& os, bool toHalf = false) const;


    Index64 memUsage() const;

    ////////////////////////////////////////

    // Disable all write methods to avoid unintentional changes
    // to the point-array offsets.

    void assertNonmodifiable() {
        assert(false && "Cannot modify voxel values in a PointDataTree.");
    }

    void setActiveState(const Coord& xyz, bool on) { BaseLeaf::setActiveState(xyz, on); }
    void setActiveState(Index offset, bool on) { BaseLeaf::setActiveState(offset, on); }

    void setValueOnly(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOnly(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOff(const Coord& xyz) { BaseLeaf::setValueOff(xyz); }
    void setValueOff(Index offset) { BaseLeaf::setValueOff(offset); }

    void setValueOff(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOff(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOn(const Coord& xyz) { BaseLeaf::setValueOn(xyz); }
    void setValueOn(Index offset) {  BaseLeaf::setValueOn(offset); }

    void setValueOn(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOn(Index, const ValueType&) { assertNonmodifiable(); }

    void setValue(const Coord&, const ValueType&) { assertNonmodifiable(); }

    void setValuesOn() { BaseLeaf::setValuesOn(); }
    void setValuesOff() { BaseLeaf::setValuesOff(); }

    template<typename ModifyOp>
    void modifyValue(Index, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValue(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    void clip(const CoordBBox&, const ValueType&) { assertNonmodifiable(); }

    void fill(const CoordBBox&, const ValueType&, bool) { assertNonmodifiable(); }
    void fill(const ValueType&) {}
    void fill(const ValueType&, bool) { assertNonmodifiable(); }

    template<typename AccessorT>
    void setValueOnlyAndCache(const Coord&, const ValueType&, AccessorT&) {assertNonmodifiable();}

    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndActiveStateAndCache(const Coord&, const ModifyOp&, AccessorT&) {
        assertNonmodifiable();
    }

    template<typename AccessorT>
    void setValueOffAndCache(const Coord&, const ValueType&, AccessorT&) { assertNonmodifiable(); }

    template<typename AccessorT>
    void setActiveStateAndCache(const Coord& xyz, bool on, AccessorT& parent) { BaseLeaf::setActiveStateAndCache(xyz, on, parent); }

    void resetBackground(const ValueType&, const ValueType&) { assertNonmodifiable(); }

    void signedFloodFill(const ValueType&) { assertNonmodifiable(); }
    void signedFloodFill(const ValueType&, const ValueType&) { assertNonmodifiable(); }

    void negate() { assertNonmodifiable(); }

    friend class ::TestPointDataLeaf;

private:
    point_data_grid_internal::UniquePtr<AttributeSet>::type mAttributeSet;

protected:
    typedef typename BaseLeaf::ValueOn ValueOn;
    typedef typename BaseLeaf::ValueOff ValueOff;
    typedef typename BaseLeaf::ValueAll ValueAll;
    typedef typename BaseLeaf::ChildOn ChildOn;
    typedef typename BaseLeaf::ChildOff ChildOff;
    typedef typename BaseLeaf::ChildAll ChildAll;

    typedef typename NodeMaskType::OnIterator    MaskOnIterator;
    typedef typename NodeMaskType::OffIterator   MaskOffIterator;
    typedef typename NodeMaskType::DenseIterator MaskDenseIterator;

    // During topology-only construction, access is needed
    // to protected/private members of other template instances.
    template<typename, Index> friend class PointDataLeafNode;

    friend class tree::IteratorBase<MaskOnIterator, PointDataLeafNode>;
    friend class tree::IteratorBase<MaskOffIterator, PointDataLeafNode>;
    friend class tree::IteratorBase<MaskDenseIterator, PointDataLeafNode>;

public:

    typedef typename BaseLeaf::template ValueIter<
        MaskOnIterator, PointDataLeafNode, const ValueType, ValueOn> ValueOnIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskOnIterator, const PointDataLeafNode, const ValueType, ValueOn> ValueOnCIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskOffIterator, PointDataLeafNode, const ValueType, ValueOff> ValueOffIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskOffIterator,const PointDataLeafNode,const ValueType,ValueOff> ValueOffCIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskDenseIterator, PointDataLeafNode, const ValueType, ValueAll> ValueAllIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskDenseIterator,const PointDataLeafNode,const ValueType,ValueAll> ValueAllCIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOnIterator, PointDataLeafNode, ChildOn> ChildOnIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOnIterator, const PointDataLeafNode, ChildOn> ChildOnCIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOffIterator, PointDataLeafNode, ChildOff> ChildOffIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOffIterator, const PointDataLeafNode, ChildOff> ChildOffCIter;
    typedef typename BaseLeaf::template DenseIter<
        PointDataLeafNode, ValueType, ChildAll> ChildAllIter;
    typedef typename BaseLeaf::template DenseIter<
        const PointDataLeafNode, const ValueType, ChildAll> ChildAllCIter;

#define VMASK_ this->getValueMask()
    ValueOnCIter  cbeginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnCIter   beginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnIter    beginValueOn()        { return ValueOnIter(VMASK_.beginOn(), this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffIter   beginValueOff()       { return ValueOffIter(VMASK_.beginOff(), this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllIter   beginValueAll()       { return ValueAllIter(VMASK_.beginDense(), this); }

    ValueOnCIter  cendValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnCIter   endValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnIter    endValueOn()          { return ValueOnIter(VMASK_.endOn(), this); }
    ValueOffCIter cendValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffCIter  endValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffIter   endValueOff()         { return ValueOffIter(VMASK_.endOff(), this); }
    ValueAllCIter cendValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllCIter  endValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllIter   endValueAll()         { return ValueAllIter(VMASK_.endDense(), this); }

    ChildOnCIter  cbeginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   beginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    beginChildOn()        { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cbeginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  beginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   beginChildOff()       { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cbeginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllCIter  beginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllIter   beginChildAll()       { return ChildAllIter(VMASK_.beginDense(), this); }

    ChildOnCIter  cendChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   endChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    endChildOn()          { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cendChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  endChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   endChildOff()         { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cendChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllCIter  endChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllIter   endChildAll()         { return ChildAllIter(VMASK_.endDense(), this); }
#undef VMASK_
}; // struct PointDataLeafNode

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readTopology(std::istream& is, bool fromHalf)
{
    BaseLeaf::readTopology(is, fromHalf);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::writeTopology(std::ostream& os, bool toHalf) const
{
    BaseLeaf::writeTopology(os, toHalf);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    BaseLeaf::readBuffers(is, fromHalf);

    mAttributeSet->read(is);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& bbox, bool fromHalf)
{
    // Read and clip voxel values (no clipping yet).
    BaseLeaf::readBuffers(is, bbox, fromHalf);

    mAttributeSet->read(is);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    BaseLeaf::writeBuffers(os, toHalf);

    mAttributeSet->write(os);
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::memUsage() const
{
    return BaseLeaf::memUsage() + mAttributeSet->memUsage();
}

} // namespace tools

////////////////////////////////////////

namespace tree
{

/// Helper metafunction used to implement LeafNode::SameConfiguration
/// (which, as an inner class, can't be independently specialized)
template<Index Dim1, typename T2>
struct SameLeafConfig<Dim1, tools::PointDataLeafNode<T2, Dim1> > { static const bool value = true; };

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_DATA_GRID_HAS_BEEN_INCLUDED


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
