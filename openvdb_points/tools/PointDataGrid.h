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

#include <openvdb/Grid.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>

#include <openvdb/tools/PointIndexGrid.h>

#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb_points/tools/AttributeGroup.h>

#include <type_traits> // std::is_same
#include <utility> // std::pair, std::make_pair

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

class TestPointDataLeaf;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace io
{

/// @brief openvdb::io::readCompressedValues specialized on PointDataIndex32 arrays to
/// ignore the value mask, use a larger block size and use 16-bit size instead of 64-bit
template<>
inline void
readCompressedValues(   std::istream& is, PointDataIndex32* destBuf, Index destCount,
                        const util::NodeMask<3>& /*valueMask*/, bool /*fromHalf*/)
{
    static_assert(sizeof(PointDataIndex32) == sizeof(Index32),
                    "Size of PointDataIndex32 expected to match size of Index32");

    const int destBytes = destCount*sizeof(Index32);
    const int maximumBytes = std::numeric_limits<uint16_t>::max();
    if (destBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot read more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    uint16_t bytes16;
    is.read(reinterpret_cast<char*>(&bytes16), sizeof(uint16_t));

    if (bytes16 == std::numeric_limits<uint16_t>::max()) { // uncompressed
        if (destBuf == nullptr) {
            is.seekg(destBytes, std::ios_base::cur);
        }
        else {
            is.read(reinterpret_cast<char*>(destBuf), destBytes);
        }
    }
    else {
        if (destBuf == nullptr) {
            is.seekg(int(bytes16), std::ios_base::cur);
        }
        else {
#ifndef OPENVDB_USE_BLOSC
            OPENVDB_THROW(IoError, "Blosc decoding is not supported");
#else
            std::unique_ptr<char[]> bloscBuffer(new char[int(bytes16)]);
            is.read(bloscBuffer.get(), bytes16);
            const int numUncompressedBytes = blosc_decompress_ctx(  /*src=*/bloscBuffer.get(),
                                                                    /*dest=*/destBuf,
                                                                    /*destsize=*/destBytes,
                                                                    /*numthreads=*/1);
            if (numUncompressedBytes < 1) {
                OPENVDB_THROW(IoError, "blosc_decompress() returned error code "
                                            << numUncompressedBytes);
            }
            if (numUncompressedBytes != destBytes) {
                OPENVDB_THROW(IoError, "Expected to decompress " << destBytes
                    << " byte" << (destBytes == 1 ? "" : "s") << ", got "
                    << numUncompressedBytes << " byte" << (numUncompressedBytes == 1 ? "" : "s"));
            }
#endif
        }
    }
}

/// @brief openvdb::io::writeCompressedValues specialized on PointDataIndex32 arrays to
/// ignore the value mask, use a larger block size and use 16-bit size instead of 64-bit
template<>
inline void
writeCompressedValues(  std::ostream& os, PointDataIndex32* srcBuf, Index srcCount,
                        const util::NodeMask<3>& /*valueMask*/,
                        const util::NodeMask<3>& /*childMask*/, bool /*toHalf*/)
{
    static_assert(sizeof(PointDataIndex32) == sizeof(Index32),
                    "Size of PointDataIndex32 expected to match size of Index32");

    const int srcBytes = srcCount*sizeof(Index32);
    const int maximumBytes = std::numeric_limits<uint16_t>::max();
    if (srcBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot write more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    int bloscBytes = 0;
    std::unique_ptr<char[]> bloscBuffer;
#ifdef OPENVDB_USE_BLOSC
    size_t tempBytes = srcBytes + BLOSC_MAX_OVERHEAD;
    if (tempBytes < BLOSC_MAX_BUFFERSIZE) {
        bloscBuffer.reset(new char[tempBytes]);

        bloscBytes = blosc_compress_ctx(/*clevel=*/9, // 0 (no compression) to 9 (maximum compression)
                                        /*doshuffle=*/true,
                                        /*typesize=*/sizeof(Index32),
                                        /*srcsize=*/srcBytes,
                                        /*src=*/reinterpret_cast<const char*>(srcBuf),
                                        /*dest=*/bloscBuffer.get(),
                                        /*destsize=*/tempBytes,
                                        BLOSC_LZ4_COMPNAME,
                                        /*blocksize=*/srcBytes,
                                        /*numthreads=*/1);
    }
#endif

    if (bloscBytes > 0) {
        uint16_t bytes16(bloscBytes); // clamp to 16-bit unsigned integer
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
        os.write(reinterpret_cast<const char*>(bloscBuffer.get()), bloscBytes);
    }
    else {
        uint16_t bytes16(maximumBytes); // max value indicates uncompressed
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
        os.write(reinterpret_cast<const char*>(srcBuf), srcBytes);
    }
}

} // namespace io


// forward declaration
namespace tree {
    template<Index, typename> struct SameLeafConfig;
}


////////////////////////////////////////


namespace tools {


// forward declaration
template<typename T, Index Log2Dim> class PointDataLeafNode;

/// @brief Point index tree configured to match the default VDB configurations.
using PointDataTree = tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode
    <PointDataLeafNode<PointDataIndex32, 3>, 4>, 5>>>;


/// @brief Point data grid.
using PointDataGrid = Grid<PointDataTree>;


/// @brief  Deep copy the descriptor across all leaf nodes.
///
/// @param  tree the PointDataTree.
///
/// @return the new descriptor.
///
/// @note This method will fail if the Descriptors in the tree are not all identical.
template <typename PointDataTreeT>
inline AttributeSet::Descriptor::Ptr
makeDescriptorUnique(PointDataTreeT& tree);


////////////////////////////////////////

// Internal utility methods
namespace point_data_grid_internal {

template<typename T>
struct UniquePtr
{
    using type = std::unique_ptr<T>;
};

}


template <typename T, Index Log2Dim>
class PointDataLeafNode : public tree::LeafNode<T, Log2Dim> {

public:
    using LeafNodeType  = PointDataLeafNode<T, Log2Dim>;
    using Ptr           = std::shared_ptr<PointDataLeafNode>;

    using ValueType     = T;
    using ValueTypePair = std::pair<ValueType, ValueType>;
    using IndexArray    = std::vector<ValueType>;

    using Descriptor    = AttributeSet::Descriptor;

    ////////////////////////////////////////

    // The following methods had to be copied from the LeafNode class
    // to make the derived PointDataLeafNode class compatible with the tree structure.

    using BaseLeaf      = tree::LeafNode<T, Log2Dim>;
    using NodeMaskType  = util::NodeMask<Log2Dim>;

    using BaseLeaf::LOG2DIM;
    using BaseLeaf::TOTAL;
    using BaseLeaf::DIM;
    using BaseLeaf::NUM_VALUES;
    using BaseLeaf::NUM_VOXELS;
    using BaseLeaf::SIZE;
    using BaseLeaf::LEVEL;

    /// Default constructor
    PointDataLeafNode()
        : mAttributeSet(new AttributeSet) { }

    ~PointDataLeafNode() = default;

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
    PointDataLeafNode(const tree::LeafNode<ValueType, Log2Dim>& other, const T& /*value*/, TopologyCopy)
        : BaseLeaf(other, zeroVal<T>(), TopologyCopy())
        , mAttributeSet(new AttributeSet) { }

    // Copy-construct from a LeafNode with the same configuration but a different ValueType.
    // Used for topology copies - explicitly sets the on and off value (background) to zeroVal
    template <typename ValueType>
    PointDataLeafNode(const tree::LeafNode<ValueType, Log2Dim>& other, const T& /*offValue*/, const T& /*onValue*/, TopologyCopy)
        : BaseLeaf(other, zeroVal<T>(), zeroVal<T>(), TopologyCopy())
        , mAttributeSet(new AttributeSet) { }

#ifndef OPENVDB_2_ABI_COMPATIBLE
    PointDataLeafNode(PartialCreate, const Coord& coords,
        const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(PartialCreate(), coords, value, active)
        , mAttributeSet(new AttributeSet) { }
#endif

public:

    /// Retrieve the attribute set.
    const AttributeSet& attributeSet() const { return *mAttributeSet; }

    /// @brief Create a new attribute set. Existing attributes will be removed.
    void initializeAttributes(const Descriptor::Ptr& descriptor, const size_t arrayLength);
    /// @brief Clear the attribute set.
    void clearAttributes(const bool updateValueMask = true);

    /// @brief Returns @c true if an attribute with this index exists.
    /// @param pos Index of the attribute
    bool hasAttribute(const size_t pos) const;
    /// @brief Returns @c true if an attribute with this name exists.
    /// @param attributeName    Name of the attribute
    bool hasAttribute(const Name& attributeName) const;

    /// @brief Append an attribute to the leaf.
    /// @param attribute Name and type of the attribute to append.
    /// @param expected Existing descriptor is expected to match this parameter.
    /// @param replacement New descriptor to replace the existing one.
    /// @param pos Index of the new attribute in the descriptor replacement.
    /// @param stride Stride of the attribute array.
    AttributeArray::Ptr appendAttribute(const Descriptor& expected, Descriptor::Ptr& replacement,
                                        const size_t pos, const Index stride = 1);

    /// @brief Drop list of attributes.
    /// @param pos vector of attribute indices to drop
    /// @param expected Existing descriptor is expected to match this parameter.
    /// @param replacement New descriptor to replace the existing one.
    void dropAttributes(const std::vector<size_t>& pos,
                        const Descriptor& expected, Descriptor::Ptr& replacement);
    /// @brief Reorder attribute set.
    /// @param replacement New descriptor to replace the existing one.
    void reorderAttributes(const Descriptor::Ptr& replacement);
    /// @brief Rename attributes in attribute set (order must remain the same).
    /// @param expected Existing descriptor is expected to match this parameter.
    /// @param replacement New descriptor to replace the existing one.
    void renameAttributes(const Descriptor& expected, Descriptor::Ptr& replacement);
    /// @brief Compact all attributes in attribute set.
    void compactAttributes();

    /// @brief Swap the underlying attribute set with the given @a attributeSet.
    /// This leaf will assume ownership of the given attribute set. The descriptors must
    /// match and the voxel offsets values will need updating if the point order is different.
    void swap(AttributeSet* attributeSet);

    /// @brief Replace the descriptor with a new one
    /// The new Descriptor must exactly match the old one
    void resetDescriptor(const Descriptor::Ptr& replacement);

    /// @brief Sets all of the voxel offset values on this leaf, from the given vector
    /// of @a offsets. If @a updateValueMask is true, then the active value mask will
    /// be updated so voxels with points are active and empty voxels are inactive.
    void setOffsets(const std::vector<ValueType>& offsets, const bool updateValueMask = true);

    /// @brief Throws an error if the voxel values on this leaf are not monotonically
    /// increasing or within the bounds of the attribute arrays
    void validateOffsets() const;

    /// @brief Read-write attribute array reference from index
    /// {
    AttributeArray& attributeArray(const size_t pos);
    const AttributeArray& attributeArray(const size_t pos) const;
    const AttributeArray& constAttributeArray(const size_t pos) const;
    /// }
    /// @brief Read-write attribute array reference from name
    /// {
    AttributeArray& attributeArray(const Name& attributeName);
    const AttributeArray& attributeArray(const Name& attributeName) const;
    const AttributeArray& constAttributeArray(const Name& attributeName) const;
    /// }

    /// @brief Read-only group handle from group index
    GroupHandle groupHandle(const AttributeSet::Descriptor::GroupIndex& index) const;
    /// @brief Read-only group handle from group name
    GroupHandle groupHandle(const Name& group) const;
    /// @brief Read-write group handle from group index
    GroupWriteHandle groupWriteHandle(const AttributeSet::Descriptor::GroupIndex& index);
    /// @brief Read-write group handle from group name
    GroupWriteHandle groupWriteHandle(const Name& name);

    /// @brief Compute the total point count for the leaf
    Index64 pointCount() const;
    /// @brief Compute the total active (on) point count for the leaf
    Index64 onPointCount() const;
    /// @brief Compute the total inactive (off) point count for the leaf
    Index64 offPointCount() const;
    /// @brief Compute the point count in a specific group for the leaf
    Index64 groupPointCount(const Name& groupName) const;

    /// @brief Activate voxels with non-zero points, deactivate voxels with zero points.
    void updateValueMask();

    ////////////////////////////////////////

    void setOffsetOn(Index offset, const ValueType& val);
    void setOffsetOnly(Index offset, const ValueType& val);

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
        if (!(std::is_same<NodeT,PointDataLeafNode>::value)) return nullptr;
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
        if (!(std::is_same<NodeT,PointDataLeafNode>::value)) return nullptr;
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

    using ValueOn   = typename BaseLeaf::ValueOn;
    using ValueOff  = typename BaseLeaf::ValueOff;
    using ValueAll  = typename BaseLeaf::ValueAll;

private:
    point_data_grid_internal::UniquePtr<AttributeSet>::type mAttributeSet;

protected:
    using ChildOn           = typename BaseLeaf::ChildOn;
    using ChildOff          = typename BaseLeaf::ChildOff;
    using ChildAll          = typename BaseLeaf::ChildAll;

    using MaskOnIterator    = typename NodeMaskType::OnIterator;
    using MaskOffIterator   = typename NodeMaskType::OffIterator;
    using MaskDenseIterator = typename NodeMaskType::DenseIterator;

    // During topology-only construction, access is needed
    // to protected/private members of other template instances.
    template<typename, Index> friend class PointDataLeafNode;

    friend class tree::IteratorBase<MaskOnIterator, PointDataLeafNode>;
    friend class tree::IteratorBase<MaskOffIterator, PointDataLeafNode>;
    friend class tree::IteratorBase<MaskDenseIterator, PointDataLeafNode>;

public:
    /// @brief Leaf value voxel iterator
    ValueVoxelCIter beginValueVoxel(const Coord& ijk) const;

public:

#ifdef _MSC_VER
    using ValueOnIter = typename BaseLeaf::ValueIter<
        MaskOnIterator, PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOnCIter = typename BaseLeaf::ValueIter<
        MaskOnIterator, const PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOffIter = typename BaseLeaf::ValueIter<
        MaskOffIterator, PointDataLeafNode, const ValueType, ValueOff>;
    using ValueOffCIter = typename BaseLeaf::ValueIter<
        MaskOffIterator,const PointDataLeafNode,const ValueType,ValueOff>;
    using ValueAllIter = typename BaseLeaf::ValueIter<
        MaskDenseIterator, PointDataLeafNode, const ValueType, ValueAll>;
    using ValueAllCIter = typename BaseLeaf::ValueIter<
        MaskDenseIterator,const PointDataLeafNode,const ValueType,ValueAll>;
    using ChildOnIter = typename BaseLeaf::ChildIter<
        MaskOnIterator, PointDataLeafNode, ChildOn>;
    using ChildOnCIter = typename BaseLeaf::ChildIter<
        MaskOnIterator, const PointDataLeafNode, ChildOn>;
    using ChildOffIter = typename BaseLeaf::ChildIter<
        MaskOffIterator, PointDataLeafNode, ChildOff>;
    using ChildOffCIter = typename BaseLeaf::ChildIter<
        MaskOffIterator, const PointDataLeafNode, ChildOff>;
    using ChildAllIter = typename BaseLeaf::DenseIter<
        PointDataLeafNode, ValueType, ChildAll>;
    using ChildAllCIter = typename BaseLeaf::DenseIter<
        const PointDataLeafNode, const ValueType, ChildAll>;
#else
    using ValueOnIter = typename BaseLeaf::template ValueIter<
        MaskOnIterator, PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOnCIter = typename BaseLeaf::template ValueIter<
        MaskOnIterator, const PointDataLeafNode, const ValueType, ValueOn>;
    using ValueOffIter = typename BaseLeaf::template ValueIter<
        MaskOffIterator, PointDataLeafNode, const ValueType, ValueOff>;
    using ValueOffCIter = typename BaseLeaf::template ValueIter<
        MaskOffIterator,const PointDataLeafNode,const ValueType,ValueOff>;
    using ValueAllIter = typename BaseLeaf::template ValueIter<
        MaskDenseIterator, PointDataLeafNode, const ValueType, ValueAll>;
    using ValueAllCIter = typename BaseLeaf::template ValueIter<
        MaskDenseIterator,const PointDataLeafNode,const ValueType,ValueAll>;
    using ChildOnIter = typename BaseLeaf::template ChildIter<
        MaskOnIterator, PointDataLeafNode, ChildOn>;
    using ChildOnCIter = typename BaseLeaf::template ChildIter<
        MaskOnIterator, const PointDataLeafNode, ChildOn>;
    using ChildOffIter = typename BaseLeaf::template ChildIter<
        MaskOffIterator, PointDataLeafNode, ChildOff>;
    using ChildOffCIter = typename BaseLeaf::template ChildIter<
        MaskOffIterator, const PointDataLeafNode, ChildOff>;
    using ChildAllIter = typename BaseLeaf::template DenseIter<
        PointDataLeafNode, ValueType, ChildAll>;
    using ChildAllCIter = typename BaseLeaf::template DenseIter<
        const PointDataLeafNode, const ValueType, ChildAll>;
#endif

    using IndexVoxelIter    = IndexIter<ValueVoxelCIter, NullFilter>;
    using IndexAllIter      = IndexIter<ValueAllCIter, NullFilter>;
    using IndexOnIter       = IndexIter<ValueOnCIter, NullFilter>;
    using IndexOffIter      = IndexIter<ValueOffCIter, NullFilter>;

    /// @brief Leaf index iterator
    IndexAllIter beginIndexAll() const;
    IndexOnIter beginIndexOn() const;
    IndexOffIter beginIndexOff() const;

    template<typename IterT, typename FilterT>
    IndexIter<IterT, FilterT> beginIndex(const FilterT& filter) const;

    /// @brief Filtered leaf index iterator
    template<typename FilterT>
    IndexIter<ValueAllCIter, FilterT> beginIndexAll(const FilterT& filter) const;
    template<typename FilterT>
    IndexIter<ValueOnCIter, FilterT> beginIndexOn(const FilterT& filter) const;
    template<typename FilterT>
    IndexIter<ValueOffCIter, FilterT> beginIndexOff(const FilterT& filter) const;

    /// @brief Leaf index iterator from voxel
    IndexVoxelIter beginIndexVoxel(const Coord& ijk) const;

    /// @brief Filtered leaf index iterator from voxel
    template<typename FilterT>
    IndexIter<ValueVoxelCIter, FilterT> beginIndexVoxel(const Coord& ijk, const FilterT& filter) const;

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

////////////////////////////////////////

// PointDataLeafNode implementation

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::initializeAttributes(const Descriptor::Ptr& descriptor, const size_t arrayLength)
{
    mAttributeSet.reset(new AttributeSet(descriptor, arrayLength));
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::clearAttributes(const bool updateValueMask)
{
    mAttributeSet.reset(new AttributeSet(*mAttributeSet, 0));

    // zero voxel values

    for (Index n = 0; n < LeafNodeType::NUM_VALUES; n++) {
        this->setOffsetOnly(n, 0);
    }

    // if updateValueMask, also de-activate all voxels

    if (updateValueMask)    this->setValuesOff();
}

template<typename T, Index Log2Dim>
inline bool
PointDataLeafNode<T, Log2Dim>::hasAttribute(const size_t pos) const
{
    return pos < mAttributeSet->size();
}

template<typename T, Index Log2Dim>
inline bool
PointDataLeafNode<T, Log2Dim>::hasAttribute(const Name& attributeName) const
{
    const size_t pos = mAttributeSet->find(attributeName);
    return pos != AttributeSet::INVALID_POS;
}

template<typename T, Index Log2Dim>
inline AttributeArray::Ptr
PointDataLeafNode<T, Log2Dim>::appendAttribute( const Descriptor& expected, Descriptor::Ptr& replacement,
                                                const size_t pos, const Index stride)
{
    return mAttributeSet->appendAttribute(expected, replacement, pos, stride);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::dropAttributes(const std::vector<size_t>& pos,
                    const Descriptor& expected, Descriptor::Ptr& replacement)
{
    mAttributeSet->dropAttributes(pos, expected, replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::reorderAttributes(const Descriptor::Ptr& replacement)
{
    mAttributeSet->reorderAttributes(replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::renameAttributes(const Descriptor& expected, Descriptor::Ptr& replacement)
{
    mAttributeSet->renameAttributes(expected, replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::compactAttributes()
{
    for (size_t i = 0; i < mAttributeSet->size(); i++) {
        AttributeArray* array = mAttributeSet->get(i);
        array->compact();
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::swap(AttributeSet* attributeSet)
{
    if (!attributeSet) {
        OPENVDB_THROW(ValueError, "Cannot swap with a null attribute set");
    }

    if (mAttributeSet->descriptor() != attributeSet->descriptor()) {
        OPENVDB_THROW(ValueError, "Attribute set descriptors are not equal.");
    }

    mAttributeSet.reset(attributeSet);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::resetDescriptor(const Descriptor::Ptr& replacement)
{
    mAttributeSet->resetDescriptor(replacement);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::setOffsets(const std::vector<ValueType>& offsets, const bool updateValueMask)
{
    if (offsets.size() != LeafNodeType::NUM_VALUES) {
        OPENVDB_THROW(ValueError, "Offset vector size doesn't match number of voxels.")
    }

    for (size_t index = 0; index < offsets.size(); ++index) {
        setOffsetOnly(index, offsets[index]);
    }

    if (updateValueMask) this->updateValueMask();
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::validateOffsets() const
{
    // Ensure all of the offset values are monotonically increasing
    for (size_t index = 1; index < BaseLeaf::SIZE; ++index) {
        if (this->getValue(index-1) > this->getValue(index)) {
            OPENVDB_THROW(ValueError, "Voxel offset values are not monotonically increasing");
        }
    }

    // Ensure all attribute arrays are of equal length
    for (size_t attributeIndex = 1; attributeIndex < mAttributeSet->size(); ++attributeIndex ) {
        if (mAttributeSet->getConst(attributeIndex-1)->size() != mAttributeSet->getConst(attributeIndex)->size()) {
            OPENVDB_THROW(ValueError, "Attribute arrays have inconsistent length");
        }
    }

    // Ensure the last voxel's offset value matches the size of each attribute array
    if (mAttributeSet->size() > 0 && this->getValue(BaseLeaf::SIZE-1) != mAttributeSet->getConst(0)->size()) {
        OPENVDB_THROW(ValueError, "Last voxel offset value does not match attribute array length");
    }
}

template<typename T, Index Log2Dim>
inline AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const size_t pos)
{
    if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range - " << pos);
    return *mAttributeSet->get(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const size_t pos) const
{
    if (pos >= mAttributeSet->size())             OPENVDB_THROW(LookupError, "Attribute Out Of Range - " << pos);
    return *mAttributeSet->getConst(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::constAttributeArray(const size_t pos) const
{
    return this->attributeArray(pos);
}

template<typename T, Index Log2Dim>
inline AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const Name& attributeName)
{
    const size_t pos = mAttributeSet->find(attributeName);
    if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found - " << attributeName);
    return *mAttributeSet->get(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::attributeArray(const Name& attributeName) const
{
    const size_t pos = mAttributeSet->find(attributeName);
    if (pos == AttributeSet::INVALID_POS)         OPENVDB_THROW(LookupError, "Attribute Not Found - " << attributeName);
    return *mAttributeSet->getConst(pos);
}

template<typename T, Index Log2Dim>
inline const AttributeArray&
PointDataLeafNode<T, Log2Dim>::constAttributeArray(const Name& attributeName) const
{
    return this->attributeArray(attributeName);
}

template<typename T, Index Log2Dim>
inline GroupHandle
PointDataLeafNode<T, Log2Dim>::groupHandle(const AttributeSet::Descriptor::GroupIndex& index) const
{
    const AttributeArray& array = this->attributeArray(index.first);
    assert(isGroup(array));

    const GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

    return GroupHandle(groupArray, index.second);
}

template<typename T, Index Log2Dim>
inline GroupHandle
PointDataLeafNode<T, Log2Dim>::groupHandle(const Name& name) const
{
    const AttributeSet::Descriptor::GroupIndex index = this->attributeSet().groupIndex(name);
    return this->groupHandle(index);
}

template<typename T, Index Log2Dim>
inline GroupWriteHandle
PointDataLeafNode<T, Log2Dim>::groupWriteHandle(const AttributeSet::Descriptor::GroupIndex& index)
{
    AttributeArray& array = this->attributeArray(index.first);
    assert(isGroup(array));

    GroupAttributeArray& groupArray = GroupAttributeArray::cast(array);

    return GroupWriteHandle(groupArray, index.second);
}

template<typename T, Index Log2Dim>
inline GroupWriteHandle
PointDataLeafNode<T, Log2Dim>::groupWriteHandle(const Name& name)
{
    const AttributeSet::Descriptor::GroupIndex index = this->attributeSet().groupIndex(name);
    return this->groupWriteHandle(index);
}

template<typename T, Index Log2Dim>
template<typename ValueIterT, typename FilterT>
inline IndexIter<ValueIterT, FilterT>
PointDataLeafNode<T, Log2Dim>::beginIndex(const FilterT& filter) const
{
    using IterTraitsT = tree::IterTraits<LeafNodeType, ValueIterT>;

    // construct the value iterator and reset the filter to use this leaf

    ValueIterT valueIter = IterTraitsT::begin(*this);
    FilterT newFilter(filter);
    newFilter.reset(*this);

    return IndexIter<ValueIterT, FilterT>(valueIter, newFilter);
}

template<typename T, Index Log2Dim>
template<typename FilterT>
inline IndexIter<typename PointDataLeafNode<T, Log2Dim>::ValueAllCIter, FilterT>
PointDataLeafNode<T, Log2Dim>::beginIndexAll(const FilterT& filter) const
{
    return this->beginIndex<ValueAllCIter, FilterT>(filter);
}

template<typename T, Index Log2Dim>
template<typename FilterT>
inline IndexIter<typename PointDataLeafNode<T, Log2Dim>::ValueOnCIter, FilterT>
PointDataLeafNode<T, Log2Dim>::beginIndexOn(const FilterT& filter) const
{
    return this->beginIndex<ValueOnCIter, FilterT>(filter);
}

template<typename T, Index Log2Dim>
template<typename FilterT>
inline IndexIter<typename PointDataLeafNode<T, Log2Dim>::ValueOffCIter, FilterT>
PointDataLeafNode<T, Log2Dim>::beginIndexOff(const FilterT& filter) const
{
    return this->beginIndex<ValueOffCIter, FilterT>(filter);
}

template<typename T, Index Log2Dim>
inline IndexIter<typename PointDataLeafNode<T, Log2Dim>::ValueAllCIter, NullFilter>
PointDataLeafNode<T, Log2Dim>::beginIndexAll() const
{
    NullFilter filter;
    return this->beginIndex<ValueAllCIter, NullFilter>(filter);
}

template<typename T, Index Log2Dim>
inline typename PointDataLeafNode<T, Log2Dim>::IndexOnIter
PointDataLeafNode<T, Log2Dim>::beginIndexOn() const
{
    NullFilter filter;
    return this->beginIndex<ValueOnCIter, NullFilter>(filter);
}

template<typename T, Index Log2Dim>
inline typename PointDataLeafNode<T, Log2Dim>::IndexOffIter
PointDataLeafNode<T, Log2Dim>::beginIndexOff() const
{
    NullFilter filter;
    return this->beginIndex<ValueOffCIter, NullFilter>(filter);
}

template<typename T, Index Log2Dim>
inline ValueVoxelCIter
PointDataLeafNode<T, Log2Dim>::beginValueVoxel(const Coord& ijk) const
{
    const Index index = LeafNodeType::coordToOffset(ijk);
    assert(index < BaseLeaf::SIZE);
    const ValueType end = this->getValue(index);
    const ValueType start = (index == 0) ? ValueType(0) : this->getValue(index - 1);
    return ValueVoxelCIter(start, end);
}

template<typename T, Index Log2Dim>
inline typename PointDataLeafNode<T, Log2Dim>::IndexVoxelIter
PointDataLeafNode<T, Log2Dim>::beginIndexVoxel(const Coord& ijk) const
{
    ValueVoxelCIter iter = this->beginValueVoxel(ijk);
    return IndexVoxelIter(iter, NullFilter());
}

template<typename T, Index Log2Dim>
template<typename FilterT>
inline IndexIter<ValueVoxelCIter, FilterT>
PointDataLeafNode<T, Log2Dim>::beginIndexVoxel(const Coord& ijk, const FilterT& filter) const
{
    ValueVoxelCIter iter = this->beginValueVoxel(ijk);
    FilterT newFilter(filter);
    newFilter.reset(*this);
    return IndexIter<ValueVoxelCIter, FilterT>(iter, newFilter);
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::pointCount() const
{
    return iterCount(this->beginIndexAll());
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::onPointCount() const
{
    if (this->isEmpty())        return 0;
    else if (this->isDense())   return this->pointCount();
    return iterCount(this->beginIndexOn());
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::offPointCount() const
{
    if (this->isEmpty())        return this->pointCount();
    else if (this->isDense())   return 0;
    return iterCount(this->beginIndexOff());
}

template<typename T, Index Log2Dim>
inline Index64
PointDataLeafNode<T, Log2Dim>::groupPointCount(const Name& groupName) const
{
    GroupFilter filter(groupName);
    return iterCount(this->beginIndexAll(filter));
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::updateValueMask()
{
    ValueType start = 0, end = 0;
    for (Index n = 0; n < LeafNodeType::NUM_VALUES; n++) {
        end = this->getValue(n);
        this->setValueMask(n, (end - start) > 0);
        start = end;
    }
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::setOffsetOn(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
    this->setValueMaskOn(offset);
}

template<typename T, Index Log2Dim>
inline void
PointDataLeafNode<T, Log2Dim>::setOffsetOnly(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
}

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


////////////////////////////////////////


template <typename PointDataTreeT>
inline AttributeSet::Descriptor::Ptr
makeDescriptorUnique(PointDataTreeT& tree)
{
    auto leafIter = tree.beginLeaf();
    if (!leafIter)  return nullptr;

    const AttributeSet::Descriptor& descriptor = leafIter->attributeSet().descriptor();
    auto newDescriptor = std::make_shared<AttributeSet::Descriptor>(descriptor);
    for (; leafIter; ++leafIter) {
        leafIter->resetDescriptor(newDescriptor);
    }

    return newDescriptor;
}


} // namespace tools

////////////////////////////////////////

namespace tree
{

/// Helper metafunction used to implement LeafNode::SameConfiguration
/// (which, as an inner class, can't be independently specialized)
template<Index Dim1, typename T2>
struct SameLeafConfig<Dim1, tools::PointDataLeafNode<T2, Dim1>> { static const bool value = true; };

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_DATA_GRID_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
