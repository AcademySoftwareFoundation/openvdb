// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_ADAPTIVE_ADAPTIVE_GRID_HAS_BEEN_INCLUDED
#define OPENVDB_ADAPTIVE_ADAPTIVE_GRID_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/Tree.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {


////////////////////////////////////////


namespace adaptive {

template <typename TreeT>
class AdaptiveAccessor
{
public:
    using ValueType = typename TreeT::ValueType;

    explicit AdaptiveAccessor(TreeT& tree):
        mTree(tree) { }

    TreeT& tree() const { return mTree; }

private:
    TreeT& mTree;
};


template<typename _ValueType>
class AdaptiveTree final: public TreeBase
{
public:
    using Ptr = SharedPtr<AdaptiveTree>;
    using ConstPtr = SharedPtr<const AdaptiveTree>;

    using ValueType = _ValueType;
    using BuildType = _ValueType;

    static const Index DEPTH = 1;

    using Accessor            = AdaptiveAccessor<AdaptiveTree>;
    using ConstAccessor       = AdaptiveAccessor<const AdaptiveTree>;
    using UnsafeAccessor      = Accessor;
    using ConstUnsafeAccessor = ConstAccessor;

    AdaptiveTree() = default;

    AdaptiveTree& operator=(const AdaptiveTree&) = delete; // disallow assignment

    /// Deep copy constructor
    AdaptiveTree(const AdaptiveTree& other): TreeBase(other), mBackground(other.mBackground) { }

    /// Empty tree constructor
    AdaptiveTree(const ValueType& background): mBackground(background) { }

    ~AdaptiveTree() override = default;

    /// Return the name of this type of tree.
    static const Name& treeType();

    /// Return the name of this tree's type.
    const Name& type() const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// Return the name of the type of a voxel's value (e.g., "float" or "vec3d").
    Name valueType() const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// Return @c true if this tree is of the same type as the template parameter.
    template<typename TreeType>
    bool isType() const { return (this->type() == TreeType::treeType()); }

    /// Return a pointer to a deep copy of this tree
    TreeBase::Ptr copy() const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Return this tree's background value.
    const ValueType& background() const { return mBackground; }

    /// @brief Return @c true if this tree contains no nodes.
    bool empty() const { return true; }

    /// Remove all nodes.
    void clear() { }

    /// @brief Not implemented.
    void prune(const ValueType& /*tolerance*/ = zeroVal<ValueType>()) { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Not implemented.
    void clip(const CoordBBox&) { OPENVDB_THROW(NotImplementedError, ""); }

    //
    // Tree methods
    //

    /// @brief Return in @a bbox the axis-aligned bounding box of all
    /// active tiles and leaf nodes with active values.
    /// @details This is faster than calling evalActiveVoxelBoundingBox,
    /// which visits the individual active voxels, and hence
    /// evalLeafBoundingBox produces a less tight, i.e. approximate, bbox.
    /// @return @c false if the bounding box is empty (in which case
    /// the bbox is set to its default value).
    bool evalLeafBoundingBox(CoordBBox& /*bbox*/) const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Return in @a dim the dimensions of the axis-aligned bounding box
    /// of all leaf nodes.
    /// @return @c false if the bounding box is empty.
    bool evalLeafDim(Coord& /*dim*/) const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Return in @a bbox the axis-aligned bounding box of all
    /// active voxels and tiles.
    /// @details This method produces a more accurate, i.e. tighter,
    /// bounding box than evalLeafBoundingBox which is approximate but
    /// faster.
    /// @return @c false if the bounding box is empty (in which case
    /// the bbox is set to its default value).
    bool evalActiveVoxelBoundingBox(CoordBBox& /*bbox*/) const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Return in @a dim the dimensions of the axis-aligned bounding box of all
    /// active voxels.  This is a tighter bounding box than the leaf node bounding box.
    /// @return @c false if the bounding box is empty.
    bool evalActiveVoxelDim(Coord& /*dim*/) const override { OPENVDB_THROW(NotImplementedError, ""); }

    void getIndexRange(CoordBBox& /*bbox*/) const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Replace with background tiles any nodes whose voxel buffers
    /// have not yet been allocated.
    /// @details Typically, unallocated nodes are leaf nodes whose voxel buffers
    /// are not yet resident in memory because delayed loading is in effect.
    /// @sa readNonresidentBuffers, io::File::open
    void clipUnallocatedNodes() override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the total number of unallocated leaf nodes residing in this tree.
    Index32 unallocatedLeafCount() const override { OPENVDB_THROW(NotImplementedError, ""); }


    //
    // Statistics
    //
    /// @brief Return the depth of this tree.
    ///
    /// A tree with only a root node and leaf nodes has depth 2, for example.
    Index treeDepth() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the number of leaf nodes.
    Index32 leafCount() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return a vector with node counts. The number of nodes of type NodeType
    /// is given as element NodeType::LEVEL in the return vector. Thus, the size
    /// of this vector corresponds to the height (or depth) of this tree.
    std::vector<Index32> nodeCount() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the number of non-leaf nodes.
    Index32 nonLeafCount() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the number of active voxels stored in leaf nodes.
    Index64 activeLeafVoxelCount() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the number of inactive voxels stored in leaf nodes.
    Index64 inactiveLeafVoxelCount() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the total number of active voxels.
    Index64 activeVoxelCount() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the number of inactive voxels within the bounding box of all active voxels.
    Index64 inactiveVoxelCount() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Return the total number of active tiles.
    Index64 activeTileCount() const override { OPENVDB_THROW(NotImplementedError, ""); }

    //
    // Accessor methods
    //
    Accessor getAccessor() { return Accessor(*this); }
    UnsafeAccessor getUnsafeAccessor() { return UnsafeAccessor(*this); }
    ConstAccessor getAccessor() const { return ConstAccessor(*this); }
    ConstAccessor getConstAccessor() const { return ConstAccessor(*this); }
    ConstUnsafeAccessor getConstUnsafeAccessor() const { return ConstUnsafeAccessor(*this); }

    //
    // I/O methods
    //
    /// Read all data buffers for this tree.
    void readBuffers(std::istream&, bool /*saveFloatAsHalf*/ = false) override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Read all of this tree's data buffers that intersect the given bounding box.
    void readBuffers(std::istream&, const CoordBBox&, bool /*saveFloatAsHalf*/ = false) override { OPENVDB_THROW(NotImplementedError, ""); }
    /// @brief Read all of this tree's data buffers that are not yet resident in memory
    /// (because delayed loading is in effect).
    /// @details If this tree was read from a memory-mapped file, this operation
    /// disconnects the tree from the file.
    /// @sa clipUnallocatedNodes, io::File::open, io::MappedFile
    void readNonresidentBuffers() const override { OPENVDB_THROW(NotImplementedError, ""); }
    /// Write out all the data buffers for this tree.
    void writeBuffers(std::ostream&, bool /*saveFloatAsHalf*/ = false) const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Print statistics, memory usage and other information about this tree.
    /// @param os            a stream to which to write textual information
    /// @param verboseLevel  1: print tree configuration only;
    ///                      2: include node and voxel statistics;
    ///                      3: include memory usage;
    ///                      4: include minimum and maximum voxel values
    /// @warning @a verboseLevel 4 forces loading of any unallocated nodes.
    void print(std::ostream& /*os*/ = std::cout, int /*verboseLevel*/ = 1) const override { OPENVDB_THROW(NotImplementedError, ""); }

    /// @brief Dummy declarations to keep Grid class happy
    using LeafNodeType = void;
    using ValueAllIter = void;
    using ValueAllCIter = void;
    using ValueOnIter = void;
    using ValueOnCIter = void;
    using ValueOffIter = void;
    using ValueOffCIter = void;

private:
    ValueType mBackground = zeroVal<ValueType>();
}; // class AdaptiveTree



////////////////////////////////////////


template<typename ValueType>
inline const Name&
AdaptiveTree<ValueType>::treeType()
{
    static std::string sTreeTypeName = []()
    {
        std::ostringstream ostr;
        ostr << "Adaptive_Tree_" << typeNameAsString<BuildType>();
        return ostr.str();
    }();
    return sTreeTypeName;
}


/// @brief Adaptive grid.
template <typename T>
using AdaptiveGrid      = Grid<AdaptiveTree<T>>;

using FloatAdaptiveGrid = AdaptiveGrid<float>;

using AdaptiveGridTypes = TypeList<FloatAdaptiveGrid>;

} // namespace adaptive


////////////////////////////////////////

/// Partial specialization for AdaptiveAccessor types
template<typename _TreeType>
struct TreeAdapter<adaptive::AdaptiveAccessor<_TreeType> >
{
    using TreeType             = _TreeType;
    using NonConstTreeType     = typename std::remove_const<TreeType>::type;
    using TreePtrType          = typename TreeType::Ptr;
    using ConstTreePtrType     = typename TreeType::ConstPtr;
    using NonConstTreePtrType  = typename NonConstTreeType::Ptr;
    using GridType             = Grid<TreeType>;
    using NonConstGridType     = Grid<NonConstTreeType>;
    using GridPtrType          = typename GridType::Ptr;
    using NonConstGridPtrType  = typename NonConstGridType::Ptr;
    using ConstGridPtrType     = typename GridType::ConstPtr;
    using ValueType            = typename TreeType::ValueType;
    using AccessorType         = typename adaptive::AdaptiveAccessor<TreeType>;
    using ConstAccessorType    = typename adaptive::AdaptiveAccessor<const NonConstTreeType>;
    using NonConstAccessorType = typename adaptive::AdaptiveAccessor<NonConstTreeType>;

    static NonConstTreeType& tree(NonConstTreeType& t) { return t; }
    static NonConstTreeType& tree(NonConstGridType& g) { return g.tree(); }
    static NonConstTreeType& tree(NonConstAccessorType& a) { return a.tree(); }
    static const NonConstTreeType& tree(ConstAccessorType& a) { return a.tree(); }
    static const NonConstTreeType& tree(const NonConstTreeType& t) { return t; }
    static const NonConstTreeType& tree(const NonConstGridType& g) { return g.tree(); }
    static const NonConstTreeType& tree(const NonConstAccessorType& a) { return a.tree(); }
    static const NonConstTreeType& tree(const ConstAccessorType& a) { return a.tree(); }
    static const NonConstTreeType& constTree(NonConstTreeType& t) { return t; }
    static const NonConstTreeType& constTree(NonConstGridType& g) { return g.constTree(); }
    static const NonConstTreeType& constTree(NonConstAccessorType& a) { return a.tree(); }
    static const NonConstTreeType& constTree(ConstAccessorType& a) { return a.tree(); }
    static const NonConstTreeType& constTree(const NonConstTreeType& t) { return t; }
    static const NonConstTreeType& constTree(const NonConstGridType& g) { return g.constTree(); }
    static const NonConstTreeType& constTree(const NonConstAccessorType& a) { return a.tree(); }
    static const NonConstTreeType& constTree(const ConstAccessorType& a) { return a.tree(); }
};


////////////////////////////////////////

// Overload the TreeTraits struct to declare a const/non-const AdaptiveTree as adaptive

template<typename ValueT>
struct TreeTraits<adaptive::AdaptiveTree<ValueT>>
{
    constexpr static TreeRepresentation Representation = TreeRepresentation::Adaptive;
};

template<typename ValueT>
struct TreeTraits<const adaptive::AdaptiveTree<ValueT>>
{
    constexpr static TreeRepresentation Representation = TreeRepresentation::Adaptive;
};

// Overload the TreeTraits struct to declare an AdaptiveAccessor as adaptive

template<typename TreeT>
struct TreeTraits<adaptive::AdaptiveAccessor<TreeT>>
{
    constexpr static TreeRepresentation Representation = TreeRepresentation::Adaptive;
};


////////////////////////////////////////


} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_ADAPTIVE_ADAPTIVE_GRID_HAS_BEEN_INCLUDED
