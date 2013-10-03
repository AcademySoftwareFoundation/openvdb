///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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
/// @file tree/Tree.h

#ifndef OPENVDB_TREE_TREE_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_TREE_HAS_BEEN_INCLUDED

#include <iostream>
#include <sstream>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>
#include <tbb/atomic.h>
#include <tbb/concurrent_hash_map.h>
#include <openvdb/Types.h>
#include <openvdb/metadata/Metadata.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/BBox.h>
#include <openvdb/util/Formats.h>
#include <openvdb/util/logging.h>
#include <openvdb/Platform.h>
#include "RootNode.h"
#include "InternalNode.h"
#include "LeafNode.h"
#include "TreeIterator.h"
#include "ValueAccessor.h"
#include "Util.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

/// @brief Base class for typed trees
class OPENVDB_API TreeBase
{
public:
    typedef boost::shared_ptr<TreeBase> Ptr;
    typedef boost::shared_ptr<const TreeBase> ConstPtr;

    TreeBase() {}
    virtual ~TreeBase() {}

    /// Return the name of this tree's type.
    virtual const Name& type() const = 0;
    
    /// Return the name of the type of a voxel's value (e.g., "float" or "vec3d").
    virtual Name valueType() const = 0;

    /// Return a pointer to a deep copy of this tree
    virtual TreeBase::Ptr copy() const = 0;

    //
    // Tree methods
    //
    /// @brief Return this tree's background value wrapped as metadata.
    /// @note Query the metadata object for the value's type.
    virtual Metadata::Ptr getBackgroundValue() const { return Metadata::Ptr(); }

    /// @brief Return in @a bbox the axis-aligned bounding box of all
    /// leaf nodes and active tiles.
    /// @details This is faster then calling evalActiveVoxelBoundingBox, 
    /// which visits the individual active voxels, and hence
    /// evalLeafBoundingBox produces a less tight, i.e. approximate, bbox.
    /// @return @c false if the bounding box is empty (in which case
    /// the bbox is set to its default value).
    virtual bool evalLeafBoundingBox(CoordBBox& bbox) const = 0;
    
    /// @brief Return in @a dim the dimensions of the axis-aligned bounding box
    /// of all leaf nodes.
    /// @return @c false if the bounding box is empty.
    virtual bool evalLeafDim(Coord& dim) const = 0;

    /// @brief Return in @a bbox the axis-aligned bounding box of all
    /// active voxels and tiles.
    /// @details This method produces a more accurate, i.e. tighter,
    /// bounding box than evalLeafBoundingBox which is approximate but
    /// faster.
    /// @return @c false if the bounding box is empty (in which case
    /// the bbox is set to its default value).
    virtual bool evalActiveVoxelBoundingBox(CoordBBox& bbox) const = 0;
    
    /// @brief Return in @a dim the dimensions of the axis-aligned bounding box of all
    /// active voxels.  This is a tighter bounding box than the leaf node bounding box.
    /// @return @c false if the bounding box is empty.
    virtual bool evalActiveVoxelDim(Coord& dim) const = 0;

    virtual void getIndexRange(CoordBBox& bbox) const = 0;


    //
    // Statistics
    //
    /// @brief Return the depth of this tree.
    ///
    /// A tree with only a root node and leaf nodes has depth 2, for example.
    virtual Index treeDepth() const = 0;
    /// Return the number of leaf nodes.
    virtual Index32 leafCount() const = 0;
    /// Return the number of non-leaf nodes.
    virtual Index32 nonLeafCount() const = 0;
    /// Return the number of active voxels stored in leaf nodes.
    virtual Index64 activeLeafVoxelCount() const = 0;
    /// Return the number of inactive voxels stored in leaf nodes.
    virtual Index64 inactiveLeafVoxelCount() const = 0;
    /// Return the total number of active voxels.
    virtual Index64 activeVoxelCount() const = 0;
    /// Return the number of inactive voxels within the bounding box of all active voxels.
    virtual Index64 inactiveVoxelCount() const = 0;

    /// Return the total amount of memory in bytes occupied by this tree.
    virtual Index64 memUsage() const { return 0; }


    //
    // I/O methods
    //
    /// @brief Read the tree topology from a stream.
    ///
    /// This will read the tree structure and tile values, but not voxel data.
    virtual void readTopology(std::istream&, bool saveFloatAsHalf = false);
    /// @brief Write the tree topology to a stream.
    ///
    /// This will write the tree structure and tile values, but not voxel data.
    virtual void writeTopology(std::ostream&, bool saveFloatAsHalf = false) const;

    /// Read all data buffers for this tree.
    virtual void readBuffers(std::istream&, bool saveFloatAsHalf = false) = 0;
    /// Write out all the data buffers for this tree.
    virtual void writeBuffers(std::ostream&, bool saveFloatAsHalf = false) const = 0;

    /// @brief Print statistics, memory usage and other information about this tree.
    /// @param os            a stream to which to write textual information
    /// @param verboseLevel  1: print tree configuration only; 2: include node and
    ///                      voxel statistics; 3: include memory usage
    virtual void print(std::ostream& os = std::cout, int verboseLevel = 1) const;

private:
    // Disallow copying of instances of this class.
    //TreeBase(const TreeBase& other);
    TreeBase& operator=(const TreeBase& other);
};


////////////////////////////////////////


template<typename _RootNodeType>
class Tree: public TreeBase
{
public:
    typedef boost::shared_ptr<Tree> Ptr;
    typedef boost::shared_ptr<const Tree> ConstPtr;

    typedef _RootNodeType                        RootNodeType;
    typedef typename RootNodeType::ValueType     ValueType;
    typedef typename RootNodeType::LeafNodeType  LeafNodeType;

    static const Index DEPTH = RootNodeType::LEVEL + 1;

    /// @brief ValueConverter<T>::Type is the type of a tree having the same
    /// hierarchy as this tree but a different value type, T.
    ///
    /// For example, FloatTree::ValueConverter<double>::Type is equivalent to DoubleTree.
    /// @note If the source tree type is a template argument, it might be necessary
    /// to write "typename SourceTree::template ValueConverter<T>::Type".
    template<typename OtherValueType>
    struct ValueConverter {
        typedef Tree<typename RootNodeType::template ValueConverter<OtherValueType>::Type> Type;
    };


    Tree(){}

    /// Deep copy constructor
    Tree(const Tree& other): TreeBase(other), mRoot(other.mRoot)
    {
    }

    /// @brief Topology copy constructor from a tree of a different type
    ///
    /// Copy the structure, i.e., the active states of tiles and voxels, of another
    /// tree of a possibly different type, but don't copy any tile or voxel values.
    /// Instead, initialize tiles and voxels with the given active and inactive values.
    /// @param other          a tree having (possibly) a different ValueType
    /// @param inactiveValue  background value for this tree, and the value to which
    ///                       all inactive tiles and voxels are initialized
    /// @param activeValue    value to which active tiles and voxels are initialized
    template<typename OtherTreeType>
    Tree(const OtherTreeType& other,
        const ValueType& inactiveValue,
        const ValueType& activeValue,
        TopologyCopy):
        TreeBase(other),
        mRoot(other.getRootNode(), inactiveValue, activeValue, TopologyCopy())
    {
    }

    /// @brief Topology copy constructor from a tree of a different type
    ///
    /// @note This topology copy constructor is generally faster than
    /// the one that takes both a foreground and a background value.
    ///
    /// Copy the structure, i.e., the active states of tiles and voxels, of another
    /// tree of a possibly different type, but don't copy any tile or voxel values.
    /// Instead, initialize tiles and voxels with the given background value.
    /// @param other        a tree having (possibly) a different ValueType
    /// @param background   the value to which tiles and voxels are initialized
    template<typename OtherTreeType>
    Tree(const OtherTreeType& other, const ValueType& background, TopologyCopy):
        TreeBase(other),
        mRoot(other.getRootNode(), background, TopologyCopy())
    {
    }

    /// Empty tree constructor
    Tree(const ValueType& background): mRoot(background) {}

    virtual ~Tree() { releaseAllAccessors(); }

    /// Return a pointer to a deep copy of this tree
    virtual TreeBase::Ptr copy() const { return TreeBase::Ptr(new Tree(*this)); }

    /// Return the name of the type of a voxel's value (e.g., "float" or "vec3d")
    virtual Name valueType() const { return typeNameAsString<ValueType>(); }

    /// Return the name of this type of tree.
    static const Name& treeType();
    /// Return the name of this type of tree.
    virtual const Name& type() const { return this->treeType(); }

    bool operator==(const Tree&) const { OPENVDB_THROW(NotImplementedError, ""); }
    bool operator!=(const Tree&) const { OPENVDB_THROW(NotImplementedError, ""); }

    //@{
    /// Return this tree's root node.
    RootNodeType& root() { return mRoot; }
    const RootNodeType& root() const { return mRoot; }
    // Deprecate the methods below
    RootNodeType& getRootNode() { return mRoot; }
    const RootNodeType& getRootNode() const { return mRoot; }
    //@}


    //
    // Tree methods
    //
    /// @brief Return @c true if the given tree has the same node and active value
    /// topology as this tree, whether or not it has the same @c ValueType.
    template<typename OtherRootNodeType>
    bool hasSameTopology(const Tree<OtherRootNodeType>& other) const;

    virtual bool evalLeafBoundingBox(CoordBBox& bbox) const;
    virtual bool evalActiveVoxelBoundingBox(CoordBBox& bbox) const;
    virtual bool evalActiveVoxelDim(Coord& dim) const;
    virtual bool evalLeafDim(Coord& dim) const;

    /// @brief Traverse the type hierarchy of nodes, and return, in @a dims, a list
    /// of the Log2Dims of nodes in order from RootNode to LeafNode.
    /// @note Because RootNodes are resizable, the RootNode Log2Dim is 0 for all trees.
    static void getNodeLog2Dims(std::vector<Index>& dims);


    //
    // I/O methods
    //
    /// @brief Read the tree topology from a stream.
    ///
    /// This will read the tree structure and tile values, but not voxel data.
    virtual void readTopology(std::istream&, bool saveFloatAsHalf = false);
    /// @brief Write the tree topology to a stream.
    ///
    /// This will write the tree structure and tile values, but not voxel data.
    virtual void writeTopology(std::ostream&, bool saveFloatAsHalf = false) const;
    /// Read all data buffers for this tree.
    virtual void readBuffers(std::istream&, bool saveFloatAsHalf = false);
    /// Write out all data buffers for this tree.
    virtual void writeBuffers(std::ostream&, bool saveFloatAsHalf = false) const;

    virtual void print(std::ostream& os = std::cout, int verboseLevel = 1) const;


    //
    // Statistics
    //
    /// @brief Return the depth of this tree.
    ///
    /// A tree with only a root node and leaf nodes has depth 2, for example.
    virtual Index treeDepth() const { return DEPTH; }
    /// Return the number of leaf nodes.
    virtual Index32 leafCount() const { return mRoot.leafCount(); }
    /// Return the number of non-leaf nodes.
    virtual Index32 nonLeafCount() const { return mRoot.nonLeafCount(); }
    /// Return the number of active voxels stored in leaf nodes.
    virtual Index64 activeLeafVoxelCount() const { return mRoot.onLeafVoxelCount(); }
    /// Return the number of inactive voxels stored in leaf nodes.
    virtual Index64 inactiveLeafVoxelCount() const { return mRoot.offLeafVoxelCount(); }
    /// Return the total number of active voxels.
    virtual Index64 activeVoxelCount() const { return mRoot.onVoxelCount(); }
    /// Return the number of inactive voxels within the bounding box of all active voxels.
    virtual Index64 inactiveVoxelCount() const;

    /// Return the total number of active tiles.
    /// @note This method is not virtual so as to not change the ABI.
    Index64 activeTileCount() const { return mRoot.onTileCount(); }
    
    /// Return the minimum and maximum active values in this tree.
    void evalMinMax(ValueType &min, ValueType &max) const;

    virtual Index64 memUsage() const { return sizeof(*this) + mRoot.memUsage(); }


    //
    // Voxel access methods (using signed indexing)
    //
    /// Return the value of the voxel at the given coordinates.
    const ValueType& getValue(const Coord& xyz) const;
    /// @brief Return the value of the voxel at the given coordinates
    /// and update the given accessor's node cache.
    template<typename AccessT> const ValueType& getValue(const Coord& xyz, AccessT&) const;

    /// @brief Return the tree depth (0 = root) at which the value of voxel (x, y, z) resides.
    /// @details If (x, y, z) isn't explicitly represented in the tree (i.e., it is
    /// implicitly a background voxel), return -1.
    int getValueDepth(const Coord& xyz) const;

    /// Set the active state of the voxel at the given coordinates but don't change its value.
    void setActiveState(const Coord& xyz, bool on);
    /// Set the value of the voxel at the given coordinates but don't change its active state.
    void setValueOnly(const Coord& xyz, const ValueType& value);
    /// Mark the voxel at the given coordinates as active but don't change its value.
    void setValueOn(const Coord& xyz);
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValueOn(const Coord& xyz, const ValueType& value);
    /// Set the value of the voxel at the given coordinates and mark the voxel as active.
    void setValue(const Coord& xyz, const ValueType& value);
    /// @brief Set the value of the voxel at the given coordinates, mark the voxel as active,
    /// and update the given accessor's node cache.
    template<typename AccessT> void setValue(const Coord& xyz, const ValueType& value, AccessT&);
    /// Mark the voxel at the given coordinates as inactive but don't change its value.
    void setValueOff(const Coord& xyz);
    /// Set the value of the voxel at the given coordinates and mark the voxel as inactive.
    void setValueOff(const Coord& xyz, const ValueType& value);

    /// @brief Apply a functor to the value of the voxel at the given coordinates
    /// and mark the voxel as active.
    /// @details Provided that the functor can be inlined, this is typically
    /// significantly faster than calling getValue() followed by setValueOn().
    /// @param xyz  the coordinates of a voxel whose value is to be modified
    /// @param op   a functor of the form <tt>void op(ValueType&) const</tt> that modifies
    ///             its argument in place
    /// @par Example:
    /// @code
    /// Coord xyz(1, 0, -2);
    /// // Multiply the value of a voxel by a constant and mark the voxel as active.
    /// floatTree.modifyValue(xyz, [](float& f) { f *= 0.25; }); // C++11
    /// // Set the value of a voxel to the maximum of its current value and 0.25,
    /// // and mark the voxel as active.
    /// floatTree.modifyValue(xyz, [](float& f) { f = std::max(f, 0.25f); }); // C++11
    /// @endcode
    /// @note The functor is not guaranteed to be called only once.
    /// @see tools::foreach()
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op);

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details Provided that the functor can be inlined, this is typically
    /// significantly faster than calling getValue() followed by setValue().
    /// @param xyz  the coordinates of a voxel to be modified
    /// @param op   a functor of the form <tt>void op(ValueType&, bool&) const</tt> that
    ///             modifies its arguments, a voxel's value and active state, in place
    /// @par Example:
    /// @code
    /// Coord xyz(1, 0, -2);
    /// // Multiply the value of a voxel by a constant and mark the voxel as inactive.
    /// floatTree.modifyValueAndActiveState(xyz,
    ///     [](float& f, bool& b) { f *= 0.25; b = false; }); // C++11
    /// // Set the value of a voxel to the maximum of its current value and 0.25,
    /// // but don't change the voxel's active state.
    /// floatTree.modifyValueAndActiveState(xyz,
    ///     [](float& f, bool&) { f = std::max(f, 0.25f); }); // C++11
    /// @endcode
    /// @note The functor is not guaranteed to be called only once.
    /// @see tools::foreach()
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op);

    /// @brief Get the value of the voxel at the given coordinates.
    /// @return @c true if the value is active.
    bool probeValue(const Coord& xyz, ValueType& value) const;

    /// Return @c true if the value at the given coordinates is active.
    bool isValueOn(const Coord& xyz) const { return mRoot.isValueOn(xyz); }
    /// Return @c true if the value at the given coordinates is inactive.
    bool isValueOff(const Coord& xyz) const { return !this->isValueOn(xyz); }
    /// Return @c true if this tree has any active tiles.
    bool hasActiveTiles() const { return mRoot.hasActiveTiles(); }

    /// @brief Set all voxels within a given axis-aligned box to a constant value.
    /// If necessary, subdivide tiles that intersect the box.
    /// @param bbox           inclusive coordinates of opposite corners of an axis-aligned box
    /// @param value          the value to which to set voxels within the box
    /// @param active         if true, mark voxels within the box as active,
    ///                       otherwise mark them as inactive
    /// @note This operation generates a sparse, but not always optimally sparse,
    /// representation of the filled box.  Follow fill operations with a prune()
    /// operation for optimal sparseness.
    void fill(const CoordBBox& bbox, const ValueType& value, bool active = true);

    /// Call the @c PruneOp functor for each non-root node in the tree.
    /// If the functor returns @c true, prune the node and replace it with a tile.
    ///
    /// This method is used to implement all of the various pruning algorithms
    /// (prune(), pruneInactive(), etc.).  It should rarely be called directly.
    /// @see openvdb/tree/Util.h for the definition of the @c PruneOp functor
    template<typename PruneOp> void pruneOp(PruneOp&);

    /// @brief Reduce the memory footprint of this tree by replacing with tiles
    /// any nodes whose values are all the same (optionally to within a tolerance)
    /// and have the same active state.
    void prune(const ValueType& tolerance = zeroVal<ValueType>());

    /// @brief Reduce the memory footprint of this tree by replacing with
    /// tiles of the given value any nodes whose values are all inactive.
    void pruneInactive(const ValueType&);

    /// @brief Reduce the memory footprint of this tree by replacing with
    /// background tiles any nodes whose values are all inactive.
    void pruneInactive();

    /// @brief Reduce the memory footprint of this tree by replacing nodes
    /// whose values are all inactive with inactive tiles having a value equal to
    /// the first value encountered in the (inactive) child.
    /// @details This method is faster than tolerance-based prune and
    /// useful for narrow-band level set applications where inactive
    /// values are limited to either an inside or an outside value.
    void pruneLevelSet();

    /// @brief Add the given leaf node to this tree, creating a new branch if necessary.
    /// If a leaf node with the same origin already exists, replace it.
    void addLeaf(LeafNodeType& leaf) { mRoot.addLeaf(&leaf); }

    /// @brief Add a tile containing voxel (x, y, z) at the specified tree level,
    /// creating a new branch if necessary.  Delete any existing lower-level nodes
    /// that contain (x, y, z).
    /// @note @a level must be less than this tree's depth.
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool active);

    /// @brief Return a pointer to the node of type @c NodeT that contains voxel (x, y, z)
    /// and replace it with a tile of the specified value and state.
    /// If no such node exists, leave the tree unchanged and return @c NULL.
    /// @note The caller takes ownership of the node and is responsible for deleting it.
    template<typename NodeT>
    NodeT* stealNode(const Coord& xyz, const ValueType& value, bool active);

    /// @brief Return a pointer to the leaf node that contains voxel (x, y, z).
    /// If no such node exists, create one that preserves the values and
    /// active states of all voxels.
    /// @details Use this method to preallocate a static tree topology over which to
    /// safely perform multithreaded processing.
    LeafNodeType* touchLeaf(const Coord& xyz);

    //@{
    /// @brief Return a pointer to the node of type @c NodeType that contains
    /// voxel (x, y, z).  If no such node exists, return NULL.
    template<typename NodeType> NodeType* probeNode(const Coord& xyz);
    template<typename NodeType> const NodeType* probeConstNode(const Coord& xyz) const;
    template<typename NodeType> const NodeType* probeNode(const Coord& xyz) const;
    //@}

    //@{
    /// @brief Return a pointer to the leaf node that contains voxel (x, y, z).
    /// If no such node exists, return NULL.
    LeafNodeType* probeLeaf(const Coord& xyz);
    const LeafNodeType* probeConstLeaf(const Coord& xyz) const;
    const LeafNodeType* probeLeaf(const Coord& xyz) const { return this->probeConstLeaf(xyz); }
    //@}


    //
    // Aux methods
    //
    /// @brief Return @c true if this tree contains no nodes other than
    /// the root node and no tiles other than background tiles.
    bool empty() const { return mRoot.empty(); }

    /// Remove all tiles from this tree and all nodes other than the root node.
    void clear() { this->clearAllAccessors(); mRoot.clear(); }

    /// Clear all registered accessors.
    void clearAllAccessors();

    //@{
    /// @brief Register an accessor for this tree.  Registered accessors are
    /// automatically cleared whenever one of this tree's nodes is deleted.
    void attachAccessor(ValueAccessorBase<Tree>&) const;
    void attachAccessor(ValueAccessorBase<const Tree>&) const;
    //@}
    //@{
    /// Deregister an accessor so that it is no longer automatically cleared.
    void releaseAccessor(ValueAccessorBase<Tree>&) const;
    void releaseAccessor(ValueAccessorBase<const Tree>&) const;
    //@}

    /// @brief Return this tree's background value wrapped as metadata.
    /// @note Query the metadata object for the value's type.
    virtual Metadata::Ptr getBackgroundValue() const;

    /// Return this tree's background value.
    const ValueType& background() const { return mRoot.background(); }
    /// Replace this tree's background value.
    void setBackground(const ValueType& background) { mRoot.setBackground(background); }

    /// Min and max are both inclusive.
    virtual void getIndexRange(CoordBBox& bbox) const { mRoot.getIndexRange(bbox); }

    /// @brief Set the values of all inactive voxels and tiles of a narrow-band
    /// level set from the signs of the active voxels, setting outside values to
    /// +background and inside values to -background.
    /// @warning This method should only be used on closed, narrow-band level sets.
    void signedFloodFill() { mRoot.signedFloodFill(); }

    /// @brief Set the values of all inactive voxels and tiles of a narrow-band
    /// level set from the signs of the active voxels, setting exterior values to
    /// @a outside and interior values to @a inside.  Set the background value
    /// of this tree to @a outside.
    /// @warning This method should only be used on closed, narrow-band level sets.
    void signedFloodFill(const ValueType& outside, const ValueType& inside);

    /// Densify active tiles, i.e., replace them with leaf-level active voxels.
    void voxelizeActiveTiles();

    /// @brief Efficiently merge another tree into this tree using one of several schemes.
    /// @details This operation is primarily intended to combine trees that are mostly
    /// non-overlapping (for example, intermediate trees from computations that are
    /// parallelized across disjoint regions of space).
    /// @note This operation is not guaranteed to produce an optimally sparse tree.
    /// Follow merge() with prune() for optimal sparseness.
    /// @warning This operation always empties the other tree.
    void merge(Tree& other, MergePolicy = MERGE_ACTIVE_STATES);

    /// @brief Union this tree's set of active values with the active values
    /// of the other tree, whose @c ValueType may be different.
    /// @details The resulting state of a value is active if the corresponding value
    /// was already active OR if it is active in the other tree.  Also, a resulting
    /// value maps to a voxel if the corresponding value already mapped to a voxel
    /// OR if it is a voxel in the other tree.  Thus, a resulting value can only
    /// map to a tile if the corresponding value already mapped to a tile
    /// AND if it is a tile value in other tree.
    ///
    /// @note This operation modifies only active states, not values.
    /// Specifically, active tiles and voxels in this tree are not changed, and
    /// tiles or voxels that were inactive in this tree but active in the other tree
    /// are marked as active in this tree but left with their original values.
    template<typename OtherRootNodeType>
    void topologyUnion(const Tree<OtherRootNodeType>& other);
   
    /// @brief Intersects this tree's set of active values with the active values
    /// of the other tree, whose @c ValueType may be different.
    /// @details The resulting state of a value is active only if the corresponding 
    /// value was already active AND if it is active in the other tree. Also, a
    /// resulting value maps to a voxel if the corresponding value
    /// already mapped to an active voxel in either of the two grids
    /// and it maps to an active tile or voxel in the other grid.
    ///
    /// @note This operation can delete branches in this grid if they
    /// overlap with inactive tiles in the other grid. Likewise active
    /// voxels can be turned into unactive voxels resulting in leaf
    /// nodes with no active values. Thus, it is recommended to
    /// subsequently call prune. 
    template<typename OtherRootNodeType>
    void topologyIntersection(const Tree<OtherRootNodeType>& other);

    /// @brief Difference this tree's set of active values with the active values
    /// of the other tree, whose @c ValueType may be different. So a
    /// resulting voxel will be active only if the original voxel is
    /// active in this tree and inactive in the other tree.
    ///
    /// @note This operation can delete branches in this grid if they
    /// overlap with active tiles in the other grid. Likewise active
    /// voxels can be turned into inactive voxels resulting in leaf
    /// nodes with no active values. Thus, it is recommended to
    /// subsequently call prune.
    template<typename OtherRootNodeType>
    void topologyDifference(const Tree<OtherRootNodeType>& other);

    /*! For a given function @c f, use sparse traversal to compute <tt>f(this, other)</tt>
     *  over all corresponding pairs of values (tile or voxel) of this tree and the other tree
     *  and store the result in this tree.
     *  This method is typically more space-efficient than the two-tree combine2(),
     *  since it moves rather than copies nodes from the other tree into this tree.
     *  @note This operation always empties the other tree.
     *  @param other  a tree of the same type as this tree
     *  @param op     a functor of the form <tt>void op(const T& a, const T& b, T& result)</tt>,
     *                where @c T is this tree's @c ValueType, that computes
     *                <tt>result = f(a, b)</tt>
     *  @param prune  if true, prune the resulting tree one branch at a time (this is usually
     *                more space-efficient than pruning the entire tree in one pass)
     *
     *  @par Example:
     *      Compute the per-voxel difference between two floating-point trees,
     *      @c aTree and @c bTree, and store the result in @c aTree (leaving @c bTree empty).
     *  @code
     *  {
     *      struct Local {
     *          static inline void diff(const float& a, const float& b, float& result) {
     *              result = a - b;
     *          }
     *      };
     *      aTree.combine(bTree, Local::diff);
     *  }
     *  @endcode
     *
     *  @par Example:
     *      Compute <tt>f * a + (1 - f) * b</tt> over all voxels of two floating-point trees,
     *      @c aTree and @c bTree, and store the result in @c aTree (leaving @c bTree empty).
     *  @code
     *  namespace {
     *      struct Blend {
     *          Blend(float f): frac(f) {}
     *          inline void operator()(const float& a, const float& b, float& result) const {
     *              result = frac * a + (1.0 - frac) * b;
     *          }
     *          float frac;
     *      };
     *  }
     *  {
     *      aTree.combine(bTree, Blend(0.25)); // 0.25 * a + 0.75 * b
     *  }
     *  @endcode
     */
    template<typename CombineOp>
    void combine(Tree& other, CombineOp& op, bool prune = false);
#ifndef _MSC_VER
    template<typename CombineOp>
    void combine(Tree& other, const CombineOp& op, bool prune = false);
#endif

    /*! Like combine(), but with
     *  @param other  a tree of the same type as this tree
     *  @param op     a functor of the form <tt>void op(CombineArgs<ValueType>& args)</tt> that
     *                computes <tt>args.setResult(f(args.a(), args.b()))</tt> and, optionally,
     *                <tt>args.setResultIsActive(g(args.aIsActive(), args.bIsActive()))</tt>
     *                for some functions @c f and @c g
     *  @param prune  if true, prune the resulting tree one branch at a time (this is usually
     *                more space-efficient than pruning the entire tree in one pass)
     *
     *  This variant passes not only the @em a and @em b values but also the active states
     *  of the @em a and @em b values to the functor, which may then return, by calling
     *  @c args.setResultIsActive(), a computed active state for the result value.
     *  By default, the result is active if either the @em a or the @em b value is active.
     *
     *  @see openvdb/Types.h for the definition of the CombineArgs struct.
     *
     *  @par Example:
     *      Replace voxel values in floating-point @c aTree with corresponding values
     *      from floating-point @c bTree (leaving @c bTree empty) wherever the @c bTree
     *      values are larger.  Also, preserve the active states of any transferred values.
     *  @code
     *  {
     *      struct Local {
     *          static inline void max(CombineArgs<float>& args) {
     *              if (args.b() > args.a()) {
     *                  // Transfer the B value and its active state.
     *                  args.setResult(args.b());
     *                  args.setResultIsActive(args.bIsActive());
     *              } else {
     *                  // Preserve the A value and its active state.
     *                  args.setResult(args.a());
     *                  args.setResultIsActive(args.aIsActive());
     *              }
     *          }
     *      };
     *      aTree.combineExtended(bTree, Local::max);
     *  }
     *  @endcode
     */
    template<typename ExtendedCombineOp>
    void combineExtended(Tree& other, ExtendedCombineOp& op, bool prune = false);
#ifndef _MSC_VER
    template<typename ExtendedCombineOp>
    void combineExtended(Tree& other, const ExtendedCombineOp& op, bool prune = false);
#endif

    /*! For a given function @c f, use sparse traversal to compute <tt>f(a, b)</tt> over all
     *  corresponding pairs of values (tile or voxel) of trees A and B and store the result
     *  in this tree.
     *  @param a,b    two trees of the same type
     *  @param op     a functor of the form <tt>void op(const T& a, const T& b, T& result)</tt>,
     *                where @c T is this tree's @c ValueType, that computes
     *                <tt>result = f(a, b)</tt>
     *  @param prune  if true, prune the resulting tree one branch at a time (this is usually
     *                more space-efficient than pruning the entire tree in one pass)
     *
     *  @par Example:
     *      Compute the per-voxel difference between two floating-point trees,
     *      @c aTree and @c bTree, and store the result in a third tree.
     *  @code
     *  {
     *      struct Local {
     *          static inline void diff(const float& a, const float& b, float& result) {
     *              result = a - b;
     *          }
     *      };
     *      FloatTree resultTree;
     *      resultTree.combine2(aTree, bTree, Local::diff);
     *  }
     *  @endcode
     */
    template<typename CombineOp>
    void combine2(const Tree& a, const Tree& b, CombineOp& op, bool prune = false);
#ifndef _MSC_VER
    template<typename CombineOp>
    void combine2(const Tree& a, const Tree& b, const CombineOp& op, bool prune = false);
#endif

    /*! Like combine2(), but with
     *  @param a,b    two trees of the same type
     *  @param op     a functor of the form <tt>void op(CombineArgs<ValueType>& args)</tt> that
     *                computes <tt>args.setResult(f(args.a(), args.b()))</tt> and, optionally,
     *                <tt>args.setResultIsActive(g(args.aIsActive(), args.bIsActive()))</tt>
     *                for some functions @c f and @c g
     *  @param prune  if true, prune the resulting tree one branch at a time (this is usually
     *                more space-efficient than pruning the entire tree in one pass)
     *  This variant passes not only the @em a and @em b values but also the active states
     *  of the @em a and @em b values to the functor, which may then return, by calling
     *  <tt>args.setResultIsActive()</tt>, a computed active state for the result value.
     *  By default, the result is active if either the @em a or the @em b value is active.
     *
     *  @see openvdb/Types.h for the definition of the CombineArgs struct.
     *
     *  @par Example:
     *      Compute the per-voxel maximum values of two floating-point trees, @c aTree
     *      and @c bTree, and store the result in a third tree.  Set the active state
     *      of each output value to that of the larger of the two input values.
     *  @code
     *  {
     *      struct Local {
     *          static inline void max(CombineArgs<float>& args) {
     *              if (args.b() > args.a()) {
     *                  // Transfer the B value and its active state.
     *                  args.setResult(args.b());
     *                  args.setResultIsActive(args.bIsActive());
     *              } else {
     *                  // Preserve the A value and its active state.
     *                  args.setResult(args.a());
     *                  args.setResultIsActive(args.aIsActive());
     *              }
     *          }
     *      };
     *      FloatTree resultTree;
     *      resultTree.combine2Extended(aTree, bTree, Local::max);
     *  }
     *  @endcode
     */
    template<typename ExtendedCombineOp>
    void combine2Extended(const Tree& a, const Tree& b, ExtendedCombineOp& op,
        bool prune = false);
#ifndef _MSC_VER
    template<typename ExtendedCombineOp>
    void combine2Extended(const Tree& a, const Tree& b, const ExtendedCombineOp&,
        bool prune = false);
#endif

    /*! For a given function use sparse traversal to call it with
     *  bounding box information for all active tiles and leaf nodes
     *  or active voxels in the tree.
     *
     *  @note The bounding boxes are guarenteed to be non-overlapping.
     *  @param op     a template functor of the form
     *                <tt>template<Index LEVEL> void op(const
     *                CoordBBox&  bbox)</tt>, where <tt>bbox</tt>
     *                defines the bbox of an active tile if <tt>LEVEL>0</tt>,
     *                and else a LeafNode or active voxel. The functor
     *                must also provide a template method of the form
     *                <tt>template<Index LEVEL> bool descent()</tt>
     *                that returns false if no bboxes
     *                are to be derived below the templated tree level. In
     *                such cases of early tree termination a bbox is
     *                instead derived from each terminating child node.
     *
     *
     *  @par Example:
     *      Render all active tiles and leaf nodes in a tree. Note in
     *      this example descent returns false if LEVEL==0 which means
     *      the functor will never descent to the active voxels. In
     *      other words the smallest BBoxes correspond to LeafNodes or
     *      active tiles at LEVEL=1!
     *  @code
     *  {
     *      struct RenderTilesAndLeafs {
     *          template<Index LEVEL>
     *          inline bool descent() { return LEVEL>0; }//only descent to leaf nodes
     *          //inline bool descent() { return true; }//use this to decent to voxels
     *
     *          template<Index LEVEL>
     *          inline void operator()(const CoordBBox &bbox) {
     *            if (LEVEL>0) {
     *               // code to render active tile
     *            } else {
     *               // code to render leaf node
     *            }
     *          }
     *      };
     *      RenderTilesAndLeafs op;
     *      aTree.visitActiveBBox(op);
     *  }
     *  @endcode
     *  @see openvdb/unittest/TestTree.cc for another example.
     */
    template<typename BBoxOp> void visitActiveBBox(BBoxOp& op) const { mRoot.visitActiveBBox(op); }

    /*! Traverse this tree in depth-first order, and at each node call the given functor
     *  with a @c DenseIterator (see Iterator.h) that points to either a child node or a
     *  tile value.  If the iterator points to a child node and the functor returns true,
     *  do not descend to the child node; instead, continue the traversal at the next
     *  iterator position.
     *  @param op  a functor of the form <tt>template<typename IterT> bool op(IterT&)</tt>,
     *             where @c IterT is either a RootNode::ChildAllIter,
     *             an InternalNode::ChildAllIter or a LeafNode::ChildAllIter
     *
     *  @note There is no iterator that points to a RootNode, so to visit the root node,
     *  retrieve the @c parent() of a RootNode::ChildAllIter.
     *
     *  @par Example:
     *      Print information about the nodes and tiles of a tree, but not individual voxels.
     *  @code
     *  namespace {
     *      template<typename TreeT>
     *      struct PrintTreeVisitor
     *      {
     *          typedef typename TreeT::RootNodeType RootT;
     *          bool visitedRoot;
     *
     *          PrintTreeVisitor(): visitedRoot(false) {}
     *
     *          template<typename IterT>
     *          inline bool operator()(IterT& iter)
     *          {
     *              if (!visitedRoot && iter.parent().getLevel() == RootT::LEVEL) {
     *                  visitedRoot = true;
     *                  std::cout << "Level-" << RootT::LEVEL << " node" << std::endl;
     *              }
     *              typename IterT::NonConstValueType value;
     *              typename IterT::ChildNodeType* child = iter.probeChild(value);
     *              if (child == NULL) {
     *                  std::cout << "Tile with value " << value << std::endl;
     *                  return true; // no child to visit, so stop descending
     *              }
     *              std::cout << "Level-" << child->getLevel() << " node" << std::endl;
     *              return (child->getLevel() == 0); // don't visit leaf nodes
     *          }
     *
     *          // The generic method, above, calls iter.probeChild(), which is not defined
     *          // for LeafNode::ChildAllIter.  These overloads ensure that the generic
     *          // method template doesn't get instantiated for LeafNode iterators.
     *          bool operator()(typename TreeT::LeafNodeType::ChildAllIter&) { return true; }
     *          bool operator()(typename TreeT::LeafNodeType::ChildAllCIter&) { return true; }
     *      };
     *  }
     *  {
     *      PrintTreeVisitor visitor;
     *      tree.visit(visitor);
     *  }
     *  @endcode
     */
    template<typename VisitorOp> void visit(VisitorOp& op);
    template<typename VisitorOp> void visit(const VisitorOp& op);

    /// Like visit(), but using @c const iterators, i.e., with
    /// @param op  a functor of the form <tt>template<typename IterT> bool op(IterT&)</tt>,
    ///            where @c IterT is either a RootNode::ChildAllCIter,
    ///            an InternalNode::ChildAllCIter or a LeafNode::ChildAllCIter
    template<typename VisitorOp> void visit(VisitorOp& op) const;
    template<typename VisitorOp> void visit(const VisitorOp& op) const;

    /*! Traverse this tree and another tree in depth-first order, and for corresponding
     *  subregions of index space call the given functor with two @c DenseIterators
     *  (see Iterator.h), each of which points to either a child node or a tile value
     *  of this tree and the other tree.  If the A iterator points to a child node
     *  and the functor returns a nonzero value with bit 0 set (e.g., 1), do not descend
     *  to the child node; instead, continue the traversal at the next A iterator position.
     *  Similarly, if the B iterator points to a child node and the functor returns a value
     *  with bit 1 set (e.g., 2), continue the traversal at the next B iterator position.
     *  @note The other tree must have the same index space and fan-out factors as
     *  this tree, but it may have a different @c ValueType and a different topology.
     *  @param other  a tree of the same type as this tree
     *  @param op     a functor of the form
     *                <tt>template<class AIterT, class BIterT> int op(AIterT&, BIterT&)</tt>,
     *                where @c AIterT and @c BIterT are any combination of a
     *                RootNode::ChildAllIter, an InternalNode::ChildAllIter or a
     *                LeafNode::ChildAllIter with an @c OtherTreeType::RootNode::ChildAllIter,
     *                an @c OtherTreeType::InternalNode::ChildAllIter
     *                or an @c OtherTreeType::LeafNode::ChildAllIter
     *
     *  @par Example:
     *      Given two trees of the same type, @c aTree and @c bTree, replace leaf nodes of
     *      @c aTree with corresponding leaf nodes of @c bTree, leaving @c bTree partially empty.
     *  @code
     *  namespace {
     *      template<typename AIterT, typename BIterT>
     *      inline int stealLeafNodes(AIterT& aIter, BIterT& bIter)
     *      {
     *          typename AIterT::NonConstValueType aValue;
     *          typename AIterT::ChildNodeType* aChild = aIter.probeChild(aValue);
     *          typename BIterT::NonConstValueType bValue;
     *          typename BIterT::ChildNodeType* bChild = bIter.probeChild(bValue);
     *
     *          const Index aLevel = aChild->getLevel(), bLevel = bChild->getLevel();
     *          if (aChild && bChild && aLevel == 0 && bLevel == 0) { // both are leaf nodes
     *              aIter.setChild(bChild); // give B's child to A
     *              bIter.setValue(bValue); // replace B's child with a constant tile value
     *          }
     *          // Don't iterate over leaf node voxels of either A or B.
     *          int skipBranch = (aLevel == 0) ? 1 : 0;
     *          if (bLevel == 0) skipBranch = skipBranch | 2;
     *          return skipBranch;
     *      }
     *  }
     *  {
     *      aTree.visit2(bTree, stealLeafNodes);
     *  }
     *  @endcode
     */
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, VisitorOp& op);
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, const VisitorOp& op);

    /// Like visit2(), but using @c const iterators, i.e., with
    /// @param other  a tree of the same type as this tree
    /// @param op     a functor of the form
    ///               <tt>template<class AIterT, class BIterT> int op(AIterT&, BIterT&)</tt>,
    ///               where @c AIterT and @c BIterT are any combination of a
    ///               RootNode::ChildAllCIter, an InternalNode::ChildAllCIter
    ///               or a LeafNode::ChildAllCIter with an
    ///               @c OtherTreeType::RootNode::ChildAllCIter,
    ///               an @c OtherTreeType::InternalNode::ChildAllCIter
    ///               or an @c OtherTreeType::LeafNode::ChildAllCIter
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, VisitorOp& op) const;
    template<typename OtherTreeType, typename VisitorOp>
    void visit2(OtherTreeType& other, const VisitorOp& op) const;


    //
    // Iteration
    //
    //@{
    /// Return an iterator over children of the root node.
    typename RootNodeType::ChildOnCIter  beginRootChildren() const { return mRoot.cbeginChildOn(); }
    typename RootNodeType::ChildOnCIter cbeginRootChildren() const { return mRoot.cbeginChildOn(); }
    typename RootNodeType::ChildOnIter   beginRootChildren() { return mRoot.beginChildOn(); }
    //@}

    //@{
    /// Return an iterator over non-child entries of the root node's table.
    typename RootNodeType::ChildOffCIter  beginRootTiles() const { return mRoot.cbeginChildOff(); }
    typename RootNodeType::ChildOffCIter cbeginRootTiles() const { return mRoot.cbeginChildOff(); }
    typename RootNodeType::ChildOffIter   beginRootTiles() { return mRoot.beginChildOff(); }
    //@}

    //@{
    /// Return an iterator over all entries of the root node's table.
    typename RootNodeType::ChildAllCIter  beginRootDense() const { return mRoot.cbeginChildAll(); }
    typename RootNodeType::ChildAllCIter cbeginRootDense() const { return mRoot.cbeginChildAll(); }
    typename RootNodeType::ChildAllIter   beginRootDense() { return mRoot.beginChildAll(); }
    //@}


    //@{
    /// Iterator over all nodes in this tree
    typedef NodeIteratorBase<Tree, typename RootNodeType::ChildOnIter>        NodeIter;
    typedef NodeIteratorBase<const Tree, typename RootNodeType::ChildOnCIter> NodeCIter;
    //@}

    //@{
    /// Iterator over all leaf nodes in this tree
    typedef LeafIteratorBase<Tree, typename RootNodeType::ChildOnIter>        LeafIter;
    typedef LeafIteratorBase<const Tree, typename RootNodeType::ChildOnCIter> LeafCIter;
    //@}

    //@{
    /// Return an iterator over all nodes in this tree.
    NodeIter   beginNode() { return NodeIter(*this); }
    NodeCIter  beginNode() const { return NodeCIter(*this); }
    NodeCIter cbeginNode() const { return NodeCIter(*this); }
    //@}

    //@{
    /// Return an iterator over all leaf nodes in this tree.
    LeafIter   beginLeaf() { return LeafIter(*this); }
    LeafCIter  beginLeaf() const { return LeafCIter(*this); }
    LeafCIter cbeginLeaf() const { return LeafCIter(*this); }
    //@}

    typedef TreeValueIteratorBase<Tree, typename RootNodeType::ValueAllIter> ValueAllIter;
    typedef TreeValueIteratorBase<const Tree, typename RootNodeType::ValueAllCIter> ValueAllCIter;
    typedef TreeValueIteratorBase<Tree, typename RootNodeType::ValueOnIter> ValueOnIter;
    typedef TreeValueIteratorBase<const Tree, typename RootNodeType::ValueOnCIter> ValueOnCIter;
    typedef TreeValueIteratorBase<Tree, typename RootNodeType::ValueOffIter> ValueOffIter;
    typedef TreeValueIteratorBase<const Tree, typename RootNodeType::ValueOffCIter> ValueOffCIter;

    //@{
    /// Return an iterator over all values (tile and voxel) across all nodes.
    ValueAllIter   beginValueAll() { return ValueAllIter(*this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(*this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(*this); }
    //@}
    //@{
    /// Return an iterator over active values (tile and voxel) across all nodes.
    ValueOnIter   beginValueOn() { return ValueOnIter(*this); }
    ValueOnCIter  beginValueOn() const { return ValueOnCIter(*this); }
    ValueOnCIter cbeginValueOn() const { return ValueOnCIter(*this); }
    //@}
    //@{
    /// Return an iterator over inactive values (tile and voxel) across all nodes.
    ValueOffIter   beginValueOff() { return ValueOffIter(*this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(*this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(*this); }
    //@}

    /// @brief Return an iterator of type @c IterT (for example, begin<ValueOnIter>() is
    /// equivalent to beginValueOn()).
    template<typename IterT> IterT begin();
    /// @brief Return a const iterator of type CIterT (for example, cbegin<ValueOnCIter>()
    /// is equivalent to cbeginValueOn()).
    template<typename CIterT> CIterT cbegin() const;


protected:
    typedef tbb::concurrent_hash_map<ValueAccessorBase<Tree>*, bool> AccessorRegistry;
    typedef tbb::concurrent_hash_map<ValueAccessorBase<const Tree>*, bool> ConstAccessorRegistry;

    // Disallow assignment of instances of this class.
    Tree& operator=(const Tree&);

    /// @brief Notify all registered accessors, by calling ValueAccessor::release(),
    /// that this tree is about to be deleted.
    void releaseAllAccessors();


    //
    // Data members
    //
    RootNodeType mRoot; // root node of the tree
    mutable AccessorRegistry mAccessorRegistry;
    mutable ConstAccessorRegistry mConstAccessorRegistry;
}; // end of Tree class


/// @brief Tree3<T, N1, N2>::Type is the type of a three-level tree
/// (Root, Internal, Leaf) with value type T and
/// internal and leaf node log dimensions N1 and N2, respectively.
/// @note This is NOT the standard tree configuration (Tree4 is).
template<typename T, Index N1, Index N2>
struct Tree3 {
    typedef Tree<RootNode<InternalNode<LeafNode<T, N2>, N1> > > Type;
};


/// @brief Tree4<T, N1, N2, N3>::Type is the type of a four-level tree
/// (Root, Internal, Internal, Leaf) with value type T and
/// internal and leaf node log dimensions N1, N2 and N3, respectively.
/// @note This is the standard tree configuration.
template<typename T, Index N1, Index N2, Index N3>
struct Tree4 {
    typedef Tree<RootNode<InternalNode<InternalNode<LeafNode<T, N3>, N2>, N1> > > Type;
};


/// @brief Tree5<T, N1, N2, N3, N4>::Type is the type of a five-level tree
/// (Root, Internal, Internal, Internal, Leaf) with value type T and
/// internal and leaf node log dimensions N1, N2, N3 and N4, respectively.
/// @note This is NOT the standard tree configuration (Tree4 is).
template<typename T, Index N1, Index N2, Index N3, Index N4>
struct Tree5 {
    typedef Tree<RootNode<InternalNode<InternalNode<InternalNode<LeafNode<T, N4>, N3>, N2>, N1> > >
        Type;
};


////////////////////////////////////////


inline void
TreeBase::readTopology(std::istream& is, bool /*saveFloatAsHalf*/)
{
    int32_t bufferCount;
    is.read(reinterpret_cast<char*>(&bufferCount), sizeof(int32_t));
    if (bufferCount != 1) OPENVDB_LOG_WARN("multi-buffer trees are no longer supported");
}


inline void
TreeBase::writeTopology(std::ostream& os, bool /*saveFloatAsHalf*/) const
{
    int32_t bufferCount = 1;
    os.write(reinterpret_cast<char*>(&bufferCount), sizeof(int32_t));
}


inline void
TreeBase::print(std::ostream& os, int /*verboseLevel*/) const
{
    os << "    Tree Type: " << type()
       << "    Active Voxel Count: " << activeVoxelCount() << std::endl
       << "    Inactive Voxel Count: " << inactiveVoxelCount() << std::endl
       << "    Leaf Node Count: " << leafCount() << std::endl
       << "    Non-leaf Node Count: " << nonLeafCount() << std::endl;
}


////////////////////////////////////////


//
// Type traits for tree iterators
//

/// @brief TreeIterTraits provides, for all tree iterators, a begin(tree) function
/// that returns an iterator over a tree of arbitrary type.
template<typename TreeT, typename IterT> struct TreeIterTraits;

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOnIter> {
    static typename TreeT::RootNodeType::ChildOnIter begin(TreeT& tree) {
        return tree.beginRootChildren();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOnCIter> {
    static typename TreeT::RootNodeType::ChildOnCIter begin(const TreeT& tree) {
        return tree.cbeginRootChildren();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOffIter> {
    static typename TreeT::RootNodeType::ChildOffIter begin(TreeT& tree) {
        return tree.beginRootTiles();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildOffCIter> {
    static typename TreeT::RootNodeType::ChildOffCIter begin(const TreeT& tree) {
        return tree.cbeginRootTiles();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildAllIter> {
    static typename TreeT::RootNodeType::ChildAllIter begin(TreeT& tree) {
        return tree.beginRootDense();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::RootNodeType::ChildAllCIter> {
    static typename TreeT::RootNodeType::ChildAllCIter begin(const TreeT& tree) {
        return tree.cbeginRootDense();
    }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::NodeIter> {
    static typename TreeT::NodeIter begin(TreeT& tree) { return tree.beginNode(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::NodeCIter> {
    static typename TreeT::NodeCIter begin(const TreeT& tree) { return tree.cbeginNode(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::LeafIter> {
    static typename TreeT::LeafIter begin(TreeT& tree) { return tree.beginLeaf(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::LeafCIter> {
    static typename TreeT::LeafCIter begin(const TreeT& tree) { return tree.cbeginLeaf(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOnIter> {
    static typename TreeT::ValueOnIter begin(TreeT& tree) { return tree.beginValueOn(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOnCIter> {
    static typename TreeT::ValueOnCIter begin(const TreeT& tree) { return tree.cbeginValueOn(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOffIter> {
    static typename TreeT::ValueOffIter begin(TreeT& tree) { return tree.beginValueOff(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueOffCIter> {
    static typename TreeT::ValueOffCIter begin(const TreeT& tree) { return tree.cbeginValueOff(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueAllIter> {
    static typename TreeT::ValueAllIter begin(TreeT& tree) { return tree.beginValueAll(); }
};

template<typename TreeT> struct TreeIterTraits<TreeT, typename TreeT::ValueAllCIter> {
    static typename TreeT::ValueAllCIter begin(const TreeT& tree) { return tree.cbeginValueAll(); }
};


template<typename RootNodeType>
template<typename IterT>
inline IterT
Tree<RootNodeType>::begin()
{
    return TreeIterTraits<Tree, IterT>::begin(*this);
}


template<typename RootNodeType>
template<typename IterT>
inline IterT
Tree<RootNodeType>::cbegin() const
{
    return TreeIterTraits<Tree, IterT>::begin(*this);
}


////////////////////////////////////////


template<typename RootNodeType>
void
Tree<RootNodeType>::readTopology(std::istream& is, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    TreeBase::readTopology(is, saveFloatAsHalf);
    mRoot.readTopology(is, saveFloatAsHalf);
}


template<typename RootNodeType>
void
Tree<RootNodeType>::writeTopology(std::ostream& os, bool saveFloatAsHalf) const
{
    TreeBase::writeTopology(os, saveFloatAsHalf);
    mRoot.writeTopology(os, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::readBuffers(std::istream &is, bool saveFloatAsHalf)
{
    this->clearAllAccessors();
    mRoot.readBuffers(is, saveFloatAsHalf);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::writeBuffers(std::ostream &os, bool saveFloatAsHalf) const
{
    mRoot.writeBuffers(os, saveFloatAsHalf);
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::attachAccessor(ValueAccessorBase<Tree>& accessor) const
{
    typename AccessorRegistry::accessor a;
    mAccessorRegistry.insert(a, &accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::attachAccessor(ValueAccessorBase<const Tree>& accessor) const
{
    typename ConstAccessorRegistry::accessor a;
    mConstAccessorRegistry.insert(a, &accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAccessor(ValueAccessorBase<Tree>& accessor) const
{
    mAccessorRegistry.erase(&accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAccessor(ValueAccessorBase<const Tree>& accessor) const
{
    mConstAccessorRegistry.erase(&accessor);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::clearAllAccessors()
{
    for (typename AccessorRegistry::iterator it = mAccessorRegistry.begin();
        it != mAccessorRegistry.end(); ++it)
    {
        if (it->first) it->first->clear();
    }

    for (typename ConstAccessorRegistry::iterator it = mConstAccessorRegistry.begin();
        it != mConstAccessorRegistry.end(); ++it)
    {
        if (it->first) it->first->clear();
    }
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::releaseAllAccessors()
{
    mAccessorRegistry.erase(NULL);
    for (typename AccessorRegistry::iterator it = mAccessorRegistry.begin();
        it != mAccessorRegistry.end(); ++it)
    {
        it->first->release();
    }
    mAccessorRegistry.clear();

    mAccessorRegistry.erase(NULL);
    for (typename ConstAccessorRegistry::iterator it = mConstAccessorRegistry.begin();
        it != mConstAccessorRegistry.end(); ++it)
    {
        it->first->release();
    }
    mConstAccessorRegistry.clear();
}


////////////////////////////////////////


template<typename RootNodeType>
inline const typename RootNodeType::ValueType&
Tree<RootNodeType>::getValue(const Coord& xyz) const
{
    return mRoot.getValue(xyz);
}


template<typename RootNodeType>
template<typename AccessT>
inline const typename RootNodeType::ValueType&
Tree<RootNodeType>::getValue(const Coord& xyz, AccessT& accessor) const
{
    return accessor.getValue(xyz);
}


template<typename RootNodeType>
inline int
Tree<RootNodeType>::getValueDepth(const Coord& xyz) const
{
    return mRoot.getValueDepth(xyz);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOff(const Coord& xyz)
{
    mRoot.setValueOff(xyz);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOff(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOff(xyz, value);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setActiveState(const Coord& xyz, bool on)
{
    mRoot.setActiveState(xyz, on);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValue(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOn(xyz, value);
}

template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOnly(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOnly(xyz, value);
}

template<typename RootNodeType>
template<typename AccessT>
inline void
Tree<RootNodeType>::setValue(const Coord& xyz, const ValueType& value, AccessT& accessor)
{
    accessor.setValue(xyz, value);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOn(const Coord& xyz)
{
    mRoot.setActiveState(xyz, true);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::setValueOn(const Coord& xyz, const ValueType& value)
{
    mRoot.setValueOn(xyz, value);
}


template<typename RootNodeType>
template<typename ModifyOp>
inline void
Tree<RootNodeType>::modifyValue(const Coord& xyz, const ModifyOp& op)
{
    mRoot.modifyValue(xyz, op);
}


template<typename RootNodeType>
template<typename ModifyOp>
inline void
Tree<RootNodeType>::modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
{
    mRoot.modifyValueAndActiveState(xyz, op);
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::probeValue(const Coord& xyz, ValueType& value) const
{
    return mRoot.probeValue(xyz, value);
}


////////////////////////////////////////


template<typename RootNodeType>
template<typename PruneOp>
inline void
Tree<RootNodeType>::pruneOp(PruneOp& op)
{
    this->clearAllAccessors();
    mRoot.pruneOp(op);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::prune(const ValueType& tolerance)
{
    TolerancePrune<ValueType> op(tolerance);
    this->pruneOp(op);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::pruneInactive(const ValueType& bg)
{
    InactivePrune<ValueType> op(bg);
    this->pruneOp(op);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::pruneInactive()
{
    this->pruneInactive(this->background());
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::pruneLevelSet()
{
    LevelSetPrune<ValueType> op(this->background());
    this->pruneOp(op);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::addTile(Index level, const Coord& xyz,
                            const ValueType& value, bool active)
{
    mRoot.addTile(level, xyz, value, active);
}


template<typename RootNodeType>
template<typename NodeT>
inline NodeT*
Tree<RootNodeType>::stealNode(const Coord& xyz, const ValueType& value, bool active)
{
    this->clearAllAccessors();
    return mRoot.template stealNode<NodeT>(xyz, value, active);
}


template<typename RootNodeType>
inline typename RootNodeType::LeafNodeType*
Tree<RootNodeType>::touchLeaf(const Coord& xyz)
{
    return mRoot.touchLeaf(xyz);
}


template<typename RootNodeType>
inline typename RootNodeType::LeafNodeType*
Tree<RootNodeType>::probeLeaf(const Coord& xyz)
{
    return mRoot.probeLeaf(xyz);
}


template<typename RootNodeType>
inline const typename RootNodeType::LeafNodeType*
Tree<RootNodeType>::probeConstLeaf(const Coord& xyz) const
{
    return mRoot.probeConstLeaf(xyz);
}


template<typename RootNodeType>
template<typename NodeType>
inline NodeType*
Tree<RootNodeType>::probeNode(const Coord& xyz)
{
    return mRoot.template probeNode<NodeType>(xyz);
}


template<typename RootNodeType>
template<typename NodeType>
inline const NodeType*
Tree<RootNodeType>::probeNode(const Coord& xyz) const
{
    return this->template probeConstNode<NodeType>(xyz);
}


template<typename RootNodeType>
template<typename NodeType>
inline const NodeType*
Tree<RootNodeType>::probeConstNode(const Coord& xyz) const
{
    return mRoot.template probeConstNode<NodeType>(xyz);
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    this->clearAllAccessors();
    return mRoot.fill(bbox, value, active);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::signedFloodFill(const ValueType& outside, const ValueType& inside)
{
    mRoot.signedFloodFill(outside, inside);
}


template<typename RootNodeType>
Metadata::Ptr
Tree<RootNodeType>::getBackgroundValue() const
{
    Metadata::Ptr result;
    if (Metadata::isRegisteredType(valueType())) {
        typedef TypedMetadata<ValueType> MetadataT;
        result = Metadata::createMetadata(valueType());
        if (MetadataT* m = dynamic_cast<MetadataT*>(result.get())) {
            m->value() = mRoot.background();
        }
    }
    return result;
}


////////////////////////////////////////


template<typename RootNodeType>
inline void
Tree<RootNodeType>::voxelizeActiveTiles()
{
    this->clearAllAccessors();
    mRoot.voxelizeActiveTiles();
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::merge(Tree& other, MergePolicy policy)
{
    this->clearAllAccessors();
    other.clearAllAccessors();
    switch (policy) {
        case MERGE_ACTIVE_STATES:
            mRoot.template merge<MERGE_ACTIVE_STATES>(other.mRoot); break;
        case MERGE_NODES:
            mRoot.template merge<MERGE_NODES>(other.mRoot); break;
        case MERGE_ACTIVE_STATES_AND_NODES:
            mRoot.template merge<MERGE_ACTIVE_STATES_AND_NODES>(other.mRoot); break;
    }
}


template<typename RootNodeType>
template<typename OtherRootNodeType>
inline void
Tree<RootNodeType>::topologyUnion(const Tree<OtherRootNodeType>& other)
{
    this->clearAllAccessors();
    mRoot.topologyUnion(other.getRootNode());
}

template<typename RootNodeType>
template<typename OtherRootNodeType>
inline void
Tree<RootNodeType>::topologyIntersection(const Tree<OtherRootNodeType>& other)
{
    this->clearAllAccessors();
    mRoot.topologyIntersection(other.getRootNode());
}

template<typename RootNodeType>
template<typename OtherRootNodeType>
inline void
Tree<RootNodeType>::topologyDifference(const Tree<OtherRootNodeType>& other)
{
    this->clearAllAccessors();
    mRoot.topologyDifference(other.getRootNode());
}

////////////////////////////////////////


/// @brief Helper class to adapt a three-argument (a, b, result) CombineOp functor
/// into a single-argument functor that accepts a CombineArgs struct
template<typename ValueT, typename CombineOp>
struct CombineOpAdapter
{
    CombineOpAdapter(CombineOp& op): op(op) {}

    void operator()(CombineArgs<ValueT>& args) const {
        op(args.a(), args.b(), args.result());
    }

    CombineOp& op;
};


template<typename RootNodeType>
template<typename CombineOp>
inline void
Tree<RootNodeType>::combine(Tree& other, CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, CombineOp> extendedOp(op);
    this->combineExtended(other, extendedOp, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.combine(bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename CombineOp>
inline void
Tree<RootNodeType>::combine(Tree& other, const CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, const CombineOp> extendedOp(op);
    this->combineExtended(other, extendedOp, prune);
}
#endif


template<typename RootNodeType>
template<typename ExtendedCombineOp>
inline void
Tree<RootNodeType>::combineExtended(Tree& other, ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.combine(other.getRootNode(), op, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.combineExtended(bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename ExtendedCombineOp>
inline void
Tree<RootNodeType>::combineExtended(Tree& other, const ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.template combine<const ExtendedCombineOp>(other.mRoot, op, prune);
}
#endif


template<typename RootNodeType>
template<typename CombineOp>
inline void
Tree<RootNodeType>::combine2(const Tree& a, const Tree& b, CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, CombineOp> extendedOp(op);
    this->combine2Extended(a, b, extendedOp, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>tree.combine2(aTree, bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename CombineOp>
inline void
Tree<RootNodeType>::combine2(const Tree& a, const Tree& b, const CombineOp& op, bool prune)
{
    CombineOpAdapter<ValueType, const CombineOp> extendedOp(op);
    this->combine2Extended(a, b, extendedOp, prune);
}
#endif


template<typename RootNodeType>
template<typename ExtendedCombineOp>
inline void
Tree<RootNodeType>::combine2Extended(const Tree& a, const Tree& b,
    ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.combine2(a.mRoot, b.mRoot, op, prune);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>tree.combine2Extended(aTree, bTree, MyCombineOp(...))</tt>.
#ifndef _MSC_VER
template<typename RootNodeType>
template<typename ExtendedCombineOp>
inline void
Tree<RootNodeType>::combine2Extended(const Tree& a, const Tree& b,
    const ExtendedCombineOp& op, bool prune)
{
    this->clearAllAccessors();
    mRoot.template combine2<const ExtendedCombineOp>(a.mRoot, b.mRoot, op, prune);
}
#endif


////////////////////////////////////////


template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(VisitorOp& op)
{
    this->clearAllAccessors();
    mRoot.template visit<VisitorOp>(op);
}


template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(VisitorOp& op) const
{
    mRoot.template visit<VisitorOp>(op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>tree.visit(MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(const VisitorOp& op)
{
    this->clearAllAccessors();
    mRoot.template visit<const VisitorOp>(op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>tree.visit(MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename VisitorOp>
inline void
Tree<RootNodeType>::visit(const VisitorOp& op) const
{
    mRoot.template visit<const VisitorOp>(op);
}


////////////////////////////////////////


template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, VisitorOp& op)
{
    this->clearAllAccessors();
    typedef typename OtherTreeType::RootNodeType OtherRootNodeType;
    mRoot.template visit2<OtherRootNodeType, VisitorOp>(other.getRootNode(), op);
}


template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, VisitorOp& op) const
{
    typedef typename OtherTreeType::RootNodeType OtherRootNodeType;
    mRoot.template visit2<OtherRootNodeType, VisitorOp>(other.getRootNode(), op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.visit2(bTree, MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, const VisitorOp& op)
{
    this->clearAllAccessors();
    typedef typename OtherTreeType::RootNodeType OtherRootNodeType;
    mRoot.template visit2<OtherRootNodeType, const VisitorOp>(other.getRootNode(), op);
}


/// @internal This overload is needed (for ICC and GCC, but not for VC) to disambiguate
/// code like this: <tt>aTree.visit2(bTree, MyVisitorOp(...))</tt>.
template<typename RootNodeType>
template<typename OtherTreeType, typename VisitorOp>
inline void
Tree<RootNodeType>::visit2(OtherTreeType& other, const VisitorOp& op) const
{
    typedef typename OtherTreeType::RootNodeType OtherRootNodeType;
    mRoot.template visit2<OtherRootNodeType, const VisitorOp>(other.getRootNode(), op);
}


////////////////////////////////////////


template<typename RootNodeType>
inline const Name&
Tree<RootNodeType>::treeType()
{
    static tbb::atomic<const Name*> sTypeName;
    if (sTypeName == NULL) {
        std::vector<Index> dims;
        Tree::getNodeLog2Dims(dims);
        std::ostringstream ostr;
        ostr << "Tree_" << typeNameAsString<ValueType>();
        for (size_t i = 1, N = dims.size(); i < N; ++i) { // start from 1 to skip the RootNode
            ostr << "_" << dims[i];
        }
        Name* s = new Name(ostr.str());
        if (sTypeName.compare_and_swap(s, NULL) != NULL) delete s;
    }
    return *sTypeName;
}


template<typename RootNodeType>
template<typename OtherRootNodeType>
inline bool
Tree<RootNodeType>::hasSameTopology(const Tree<OtherRootNodeType>& other) const
{
    return mRoot.hasSameTopology(other.getRootNode());
}


template<typename RootNodeType>
Index64
Tree<RootNodeType>::inactiveVoxelCount() const
{
    Coord dim(0, 0, 0);
    this->evalActiveVoxelDim(dim);
    const Index64
        totalVoxels = dim.x() * dim.y() * dim.z(),
        activeVoxels = this->activeVoxelCount();
    assert(totalVoxels >= activeVoxels);
    return totalVoxels - activeVoxels;
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalLeafBoundingBox(CoordBBox& bbox) const
{
    bbox.reset(); // default invalid bbox
    
    if (this->empty()) return false;  // empty

    mRoot.evalActiveBoundingBox(bbox, false);

    return true;// not empty
}

template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalActiveVoxelBoundingBox(CoordBBox& bbox) const
{
    
    bbox.reset(); // default invalid bbox
    
    if (this->empty()) return false;  // empty

    mRoot.evalActiveBoundingBox(bbox, true);

    return true;// not empty
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalActiveVoxelDim(Coord& dim) const
{
    CoordBBox bbox;
    bool notEmpty = this->evalActiveVoxelBoundingBox(bbox);
    dim = bbox.extents();
    return notEmpty;
}


template<typename RootNodeType>
inline bool
Tree<RootNodeType>::evalLeafDim(Coord& dim) const
{
    CoordBBox bbox;
    bool notEmpty = this->evalLeafBoundingBox(bbox);
    dim = bbox.extents();
    return notEmpty;
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::evalMinMax(ValueType& minVal, ValueType& maxVal) const
{
    minVal = maxVal = zeroVal<ValueType>();
    if (ValueOnCIter iter = this->cbeginValueOn()) {
        minVal = maxVal = *iter;
        for (++iter; iter; ++iter) {
            const ValueType& val = *iter;
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::getNodeLog2Dims(std::vector<Index>& dims)
{
    dims.clear();
    RootNodeType::getNodeLog2Dims(dims);
}


template<typename RootNodeType>
inline void
Tree<RootNodeType>::print(std::ostream& os, int verboseLevel) const
{
    if (verboseLevel <= 0) return;

    struct OnExit {
        std::ostream& os;
        std::streamsize savedPrecision;
        OnExit(std::ostream& os): os(os), savedPrecision(os.precision()) {}
        ~OnExit() { os.precision(savedPrecision); }
    };
    OnExit restorePrecision(os);

    std::vector<Index> dims;
    Tree::getNodeLog2Dims(dims);

    std::vector<Index64> nodeCount;

    os << "Information about Tree:\n"
        << "  Type: " << this->type() << "\n";

    os << "  Configuration:\n";
    if (verboseLevel <= 1) {
        // Print node types and sizes.
        os << "    Root(" << mRoot.getTableSize() << ")";
        if (dims.size() > 1) {
            for (size_t i = 1, N = dims.size() - 1; i < N; ++i) {
                os << ", Internal(" << (1 << dims[i]) << "^3)";
            }
            os << ", Leaf(" << (1 << *dims.rbegin()) << "^3)\n";
        }
    } else {
        // Print node types, counts and sizes.
        nodeCount.resize(dims.size());
        for (NodeCIter it = cbeginNode(); it; ++it) {
            ++(nodeCount[it.getDepth()]);
        }
        os << "    Root(1 x " << mRoot.getTableSize() << ")";
        if (dims.size() > 1) {
            for (size_t i = 1, N = dims.size() - 1; i < N; ++i) {
                os << ", Internal(" << util::formattedInt(nodeCount[i]);
                os << " x " << (1 << dims[i]) << "^3)";
            }
            os << ", Leaf(" << util::formattedInt(*nodeCount.rbegin());
            os << " x " << (1 << *dims.rbegin()) << "^3)\n";
        }
    }
    os << "  Background value: " << mRoot.background() << "\n";

    if (verboseLevel == 1) return;

    // The following is tree information that is expensive to extract.

    if (nodeCount.empty()) {
        nodeCount.resize(dims.size());
        for (NodeCIter it = cbeginNode(); it; ++it) {
            ++(nodeCount[it.getDepth()]);
        }
    }

    // Statistics of topology and values
    ValueType minVal, maxVal;
    this->evalMinMax(minVal, maxVal);
    os << "  Min value: " << minVal << "\n";
    os << "  Max value: " << maxVal << "\n";

    const uint64_t
        leafCount = *nodeCount.rbegin(),
        numActiveVoxels = this->activeVoxelCount(),
        numActiveLeafVoxels = this->activeLeafVoxelCount();

    os << "  Number of active voxels:       " << util::formattedInt(numActiveVoxels) << "\n";

    Coord dim(0, 0, 0);
    uint64_t totalVoxels = 0;
    if (numActiveVoxels) { // nonempty
        CoordBBox bbox;
        this->evalActiveVoxelBoundingBox(bbox);
        dim = bbox.extents();
        totalVoxels = dim.x() * uint64_t(dim.y()) * dim.z();

        os << "  Bounding box of active voxels: " << bbox << "\n";
        os << "  Dimensions of active voxels:   "
            << dim[0] << " x " << dim[1] << " x " << dim[2] << "\n";

        const double activeRatio = (100.0 * numActiveVoxels) / totalVoxels;
        os << "  Percentage of active voxels:   " << std::setprecision(3) << activeRatio << "%\n";

        if (leafCount>0) {
            const double fillRatio =
                (100.0 * numActiveLeafVoxels) / (leafCount * LeafNodeType::NUM_VOXELS);
            os << "  Average leaf node fill ratio:  " << fillRatio << "%\n";
        }
    } else {
        os << "  Tree is empty!\n";
    }
    os << std::flush;

    if (verboseLevel == 2) return;

    // Memory footprint in bytes
    const uint64_t
        actualMem = this->memUsage(),
        denseMem = sizeof(ValueType) * totalVoxels,
        voxelsMem = sizeof(ValueType) * numActiveLeafVoxels;
            ///< @todo not accurate for BoolTree (and probably should count tile values)

    os << "Memory footprint:\n";
    util::printBytes(os, actualMem, "  Actual footprint: ");
    util::printBytes(os, voxelsMem, "  Voxel footprint:  ");

    if (numActiveVoxels) {
        util::printBytes(os, denseMem, "  Dense* footprint: ");
        os << "  Actual footprint is " << (100.0 * actualMem / denseMem)
            << "% of dense* footprint\n";
        os << "  Leaf voxel footprint is " << (100.0 * voxelsMem / actualMem)
           << "% of actual footprint\n";
        os << " *Dense refers to the smallest equivalent non-sparse volume" << std::endl;
    }
}

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_TREE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
