// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

///////////////////////////////////////////////////////////////////////////
//
/// @file FindActiveValues.h
///
/// @author Ken Museth
///
/// @brief Finds the active values and tiles in a tree that intersects a bounding box.
///        Methods are provided that count the number of active values and tiles,
///        test for the existence of active values and tiles, and return a list of
///        the active tiles that intersect a bbox.
///
/// @warning For repeated calls to the free-standing functions defined below
///          consider instead creating an instance of FindActiveValues
///          and then repeatedly call its member methods. This assumes the tree
///          to be constant between calls but is sightly faster.
///
///////////////////////////////////////////////////////////////////////////

#ifndef OPENVDB_TOOLS_FINDACTIVEVALUES_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_FINDACTIVEVALUES_HAS_BEEN_INCLUDED

#include <vector>
#include <openvdb/version.h> // for OPENVDB_VERSION_NAME
#include <openvdb/Types.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/openvdb.h>

#include "Count.h" // tools::countActiveVoxels()

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Struct that encodes a bounding box, value and level of a tile
///
/// @details The bbox of a tiles is trimmed to the bounding box that probed it.
///          The level is typically defined as: 1 is 8^3, 2 is 128^3, and 3 is 4096^3.
template<typename ValueType>
struct TileData;

/// @brief Returns true if the bounding box intersects any of the active
///        values in a tree, i.e. either active voxels or active tiles.
///
/// @warning For repeated calls to this method consider instead creating an instance of
///          FindActiveValues and then repeatedly call anyActiveValues(). This assumes the tree
///          to be constant between calls but is slightly faster.
///
/// @param tree   const tree to be tested for active values.
/// @param bbox   index bounding box which is intersected against the active values.
template<typename TreeT>
bool
anyActiveValues(const TreeT& tree, const CoordBBox &bbox);

/// @brief Returns true if the bounding box intersects any of the active
///        voxels in a tree, i.e. ignores active tile values.
///
/// @note In VDB voxels by definition reside in the leaf nodes ONLY. So this method
///       ignores active tile values that reside higher up in the VDB tree structure.
///
/// @warning For repeated calls to this method consider instead creating an instance of
///          FindActiveValues and then repeatedly call anyActiveVoxels(). This assumes the tree
///          to be constant between calls but is slightly faster.
///
/// @param tree   const tree to be tested for active voxels.
/// @param bbox   index bounding box which is intersected against the active voxels.
template<typename TreeT>
bool
anyActiveVoxels(const TreeT& tree, const CoordBBox &bbox);

/// @brief Returns true if the bounding box intersects any of the active
///        tiles in a tree, i.e. ignores active leaf values.
///
/// @warning For repeated calls to this method consider instead creating an instance of
///          FindActiveValues and then repeatedly call anyActiveTiles(). This assumes the tree
///          to be constant between calls but is slightly faster.
///
/// @param tree   const tree to be tested for active tiles.
/// @param bbox   index bounding box which is intersected against the active tiles.
template<typename TreeT>
bool
anyActiveTiles(const TreeT& tree, const CoordBBox &bbox);

/// @brief Returns true if the bounding box intersects none of the active
///        values in a tree, i.e. neither active voxels or active tiles.
///
/// @warning For repeated calls to this method consider instead creating an instance of
///          FindActiveValues and then repeatedly call noActiveValues(). This assumes the tree
///          to be constant between calls but is slightly faster.
///
/// @param tree   const tree to be tested for active values.
/// @param bbox   index bounding box which is intersected against the active values.
template<typename TreeT>
bool
noActiveValues(const TreeT& tree, const CoordBBox &bbox);

/// @brief Returns the number of active values that intersects a bounding box intersects,
///        i.e. the count includes both active voxels and virtual voxels in active tiles.
///
/// @warning For repeated calls to this method consider instead creating an instance of
///          FindActiveValues and then repeatedly call count(). This assumes the tree
///          to be constant between calls but is slightly faster.
///
/// @param tree   const tree to be tested for active values.
/// @param bbox   index bounding box which is intersected against the active values.
template<typename TreeT>
Index64
countActiveValues(const TreeT& tree, const CoordBBox &bbox);

/// @brief Return a vector with bounding boxes that represents all the intersections
///        between active tiles in the tree and the specified bounding box.
///
/// @warning For repeated calls to this method consider instead creating an instance of
///          FindActiveValues and then repeatedly call count(). This assumes the tree
///          to be constant between calls but is slightly faster.
///
/// @param tree   const tree to be tested for active tiles.
/// @param bbox   index bounding box which is intersected against the active tiles.
template<typename TreeT>
std::vector<TileData<typename TreeT::ValueType>>
activeTiles(const TreeT& tree, const CoordBBox &bbox);

//////////////////////////////////////////////////////////////////////////////////////////

/// @brief   Finds the active values in a tree which intersects a bounding box.
///
/// @details Two methods are provided, one that count the number of active values
///          and one that simply tests if any active values intersect the bbox.
///
/// @warning Tree nodes are cached by this class so it's important that the tree is not
///          modified after this class is instantiated and before its methods are called.
template<typename TreeT>
class FindActiveValues
{
public:

    using TileDataT = TileData<typename TreeT::ValueType>;

    /// @brief Constructor from a const tree, which is assumed not to be modified after construction.
    FindActiveValues(const TreeT& tree);

    /// @brief Default destructor
    ~FindActiveValues();

    /// @brief Initiate this class with a new (or modified) tree.
    void update(const TreeT& tree);

    /// @brief Returns true if the specified bounding box intersects any active values.
    ///
    /// @warning Using a ValueAccessor (i.e. useAccessor = true) can improve performance for especially
    ///          small bounding boxes, but at the cost of no thread-safety. So if multiple threads are
    ///          calling this method concurrently use the default setting, useAccessor = false.
    bool anyActiveValues(const CoordBBox &bbox, bool useAccessor = false) const;

    /// @brief Returns true if the specified bounding box intersects any active tiles only.
    bool anyActiveVoxels(const CoordBBox &bbox) const;

    /// @brief Returns true if the specified bounding box intersects any active tiles only.
    bool anyActiveTiles(const CoordBBox &bbox) const;

    /// @brief Returns true if the specified bounding box does not intersect any active values.
    ///
    /// @warning Using a ValueAccessor (i.e. useAccessor = true) can improve performance for especially
    ///          small bounding boxes, but at the cost of no thread-safety. So if multiple threads are
    ///          calling this method concurrently use the default setting, useAccessor = false.
    bool noActiveValues(const CoordBBox &bbox, bool useAccessor = false) const { return !this->anyActiveValues(bbox, useAccessor); }

    /// @brief Returns the number of active voxels intersected by the specified bounding box.
    Index64 count(const CoordBBox &bbox) const;

    /// @brief Return a vector with bounding boxes that represents all the intersections
    ///        between active tiles in the tree and the specified bounding box.
    std::vector<TileDataT> activeTiles(const CoordBBox &bbox) const;

private:

    // Cleans up internal data structures
    void clear();

    // builds internal data structures
    void init(const TreeT &tree);

    template<typename NodeT>
    typename NodeT::NodeMaskType getBBoxMask(const CoordBBox &bbox, const NodeT* node) const;

    // process leaf nodes
    inline bool anyActiveValues(const typename TreeT::LeafNodeType* leaf, const CoordBBox &bbox ) const { return this->anyActiveVoxels(leaf, bbox); }
    inline bool anyActiveVoxels(const typename TreeT::LeafNodeType* leaf, const CoordBBox &bbox ) const;
    static bool anyActiveTiles( const typename TreeT::LeafNodeType*, const CoordBBox& ) {return false;}
    void activeTiles(const typename TreeT::LeafNodeType*, const CoordBBox&, std::vector<TileDataT>&) const {;}// no-op
    inline Index64 count(const typename TreeT::LeafNodeType* leaf, const CoordBBox &bbox ) const;

    // process internal nodes
    template<typename NodeT>
    bool anyActiveValues(const NodeT* node, const CoordBBox &bbox) const;
    template<typename NodeT>
    bool anyActiveVoxels(const NodeT* node, const CoordBBox &bbox) const;
    template<typename NodeT>
    bool anyActiveTiles(const NodeT* node, const CoordBBox &bbox) const;
    template<typename NodeT>
    void activeTiles(const NodeT* node, const CoordBBox &bbox, std::vector<TileDataT> &tiles) const;
    template<typename NodeT>
    Index64 count(const NodeT* node, const CoordBBox &bbox) const;

    using AccT = tree::ValueAccessor<const TreeT, false/* IsSafe */>;
    using RootChildType = typename TreeT::RootNodeType::ChildNodeType;

    struct RootChild;

    AccT mAcc;
    std::vector<TileDataT> mRootTiles;// cache bbox of child nodes (faster to cache than access RootNode)
    std::vector<RootChild> mRootNodes;// cache bbox of active tiles (faster to cache than access RootNode)

};// FindActiveValues class

//////////////////////////////////////////////////////////////////////////////////////////

template<typename TreeT>
FindActiveValues<TreeT>::FindActiveValues(const TreeT& tree) : mAcc(tree), mRootTiles(), mRootNodes()
{
    this->init(tree);
}

template<typename TreeT>
FindActiveValues<TreeT>::~FindActiveValues()
{
    this->clear();
}

template<typename TreeT>
void FindActiveValues<TreeT>::update(const TreeT& tree)
{
    this->clear();
    mAcc = AccT(tree);
    this->init(tree);
}

template<typename TreeT>
void FindActiveValues<TreeT>::clear()
{
    mRootNodes.clear();
    mRootTiles.clear();
}

template<typename TreeT>
void FindActiveValues<TreeT>::init(const TreeT& tree)
{
    const auto &root = tree.root();
    for (auto i = root.cbeginChildOn(); i; ++i) {
        mRootNodes.emplace_back(i.getCoord(), &*i);
    }
    for (auto i = root.cbeginValueOn(); i; ++i) {
        mRootTiles.emplace_back(root, i.getCoord(), *i);
    }
}

template<typename TreeT>
bool FindActiveValues<TreeT>::anyActiveValues(const CoordBBox &bbox, bool useAccessor) const
{
    // test early-out: the center of the bbox is active
    if (useAccessor) {
        if (mAcc.isValueOn( (bbox.min() + bbox.max())>>1 )) return true;
    } else {
        if (mAcc.tree().isValueOn( (bbox.min() + bbox.max())>>1 )) return true;
    }

    for (auto& tile : mRootTiles) {
        if (tile.bbox.hasOverlap(bbox)) return true;
    }
    for (auto& node : mRootNodes) {
        if (!node.bbox.hasOverlap(bbox)) {// no overlap
            continue;
        } else if (node.bbox.isInside(bbox)) {// bbox is inside the child node
            return this->anyActiveValues(node.child, bbox);
        } else if (this->anyActiveValues(node.child, bbox)) {// bbox overlaps the child node
            return true;
        }
    }
    return false;
}

template<typename TreeT>
bool FindActiveValues<TreeT>::anyActiveVoxels(const CoordBBox &bbox) const
{
    for (auto& node : mRootNodes) {
        if (!node.bbox.hasOverlap(bbox)) {// no overlap
            continue;
        } else if (node.bbox.isInside(bbox)) {// bbox is inside the child node
            return this->anyActiveVoxels(node.child, bbox);
        } else if (this->anyActiveVoxels(node.child, bbox)) {// bbox overlaps the child node
            return true;
        }
    }
    return false;
}

template<typename TreeT>
bool FindActiveValues<TreeT>::anyActiveTiles(const CoordBBox &bbox) const
{
    for (auto& tile : mRootTiles) {
        if (tile.bbox.hasOverlap(bbox)) return true;
    }
    for (auto& node : mRootNodes) {
        if (!node.bbox.hasOverlap(bbox)) {// no overlap
            continue;
        } else if (node.bbox.isInside(bbox)) {// bbox is inside the child node
            return this->anyActiveTiles(node.child, bbox);
        } else if (this->anyActiveTiles(node.child, bbox)) {// bbox overlaps the child node
            return true;
        }
    }
    return false;
}

template<typename TreeT>
Index64 FindActiveValues<TreeT>::count(const CoordBBox &bbox) const
{
    Index64 count = 0;
    for (auto& tile : mRootTiles) {//loop over active tiles only
        if (!tile.bbox.hasOverlap(bbox)) {
            continue;//ignore non-overlapping tiles
        } else if (tile.bbox.isInside(bbox)) {
            return bbox.volume();// bbox is completely inside the active tile
        } else if (bbox.isInside(tile.bbox)) {
            count += RootChildType::NUM_VOXELS;
        } else {// partial overlap between tile and bbox
            auto tmp = tile.bbox;
            tmp.intersect(bbox);
            count += tmp.volume();
        }
    }
    for (auto &node : mRootNodes) {//loop over child nodes of the root node only
        if ( !node.bbox.hasOverlap(bbox) ) {
            continue;//ignore non-overlapping child nodes
        } else if ( node.bbox.isInside(bbox) ) {
            return this->count(node.child, bbox);// bbox is completely inside the child node
        } else {
            count += this->count(node.child, bbox);
        }
    }
    return count;
}

template<typename TreeT>
std::vector<TileData<typename TreeT::ValueType> >
FindActiveValues<TreeT>::activeTiles(const CoordBBox &bbox) const
{
    std::vector<TileDataT> tiles;
    for (auto& tile : mRootTiles) {//loop over active tiles only
        if (!tile.bbox.hasOverlap(bbox)) {
            continue;//ignore non-overlapping tiles
        } else if (tile.bbox.isInside(bbox)) {// bbox is completely inside the active tile
            tiles.emplace_back(bbox, tile.value, tile.level);
            return tiles;
        } else if (bbox.isInside(tile.bbox)) {// active tile is completely inside the bbox
            tiles.push_back(tile);
        } else {// partial overlap between tile and bbox
            auto tmp = tile.bbox;
            tmp.intersect(bbox);
            tiles.emplace_back(tmp, tile.value, tile.level);
        }
    }
    for (auto &node : mRootNodes) {//loop over child nodes of the root node only
        if ( !node.bbox.hasOverlap(bbox) ) {
            continue;//ignore non-overlapping child nodes
        } else if ( node.bbox.isInside(bbox) ) {// bbox is completely inside the child node
            this->activeTiles(node.child, bbox, tiles);
            return tiles;
        } else {// partial overlap between tile and child node
            this->activeTiles(node.child, bbox, tiles);
        }
    }
    return tiles;
}

template<typename TreeT>
template<typename NodeT>
typename NodeT::NodeMaskType FindActiveValues<TreeT>::getBBoxMask(const CoordBBox &bbox, const NodeT* node) const
{
    typename NodeT::NodeMaskType mask;// typically 32^3 or 16^3 bit mask
    auto b = node->getNodeBoundingBox();
    OPENVDB_ASSERT( bbox.hasOverlap(b) );
    if ( bbox.isInside(b) ) {
        mask.setOn();//node is completely inside the bbox so early out
    } else {
        b.intersect(bbox);// trim bounding box
        // transform bounding box from global to local coordinates
        b.min() &=  NodeT::DIM-1u;
        b.min() >>= NodeT::ChildNodeType::TOTAL;
        b.max() &=  NodeT::DIM-1u;
        b.max() >>= NodeT::ChildNodeType::TOTAL;
        OPENVDB_ASSERT( b.hasVolume() );
        auto it = b.begin();// iterates over all the child nodes or tiles that intersects bbox
        for (const Coord& ijk = *it; it; ++it) {
            mask.setOn(ijk[2] + (ijk[1] << NodeT::LOG2DIM) + (ijk[0] << 2*NodeT::LOG2DIM));
        }
    }
    return mask;
}

template<typename TreeT>
template<typename NodeT>
bool FindActiveValues<TreeT>::anyActiveValues(const NodeT* node, const CoordBBox &bbox) const
{
    // Generate a bit mask of the bbox coverage
    auto mask = this->getBBoxMask(bbox, node);

    // Check active tiles
    const auto tmp = mask & node->getValueMask();// prune active the tile mask with the bbox mask
    if (!tmp.isOff()) return true;

    // Check child nodes
    mask &= node->getChildMask();// prune the child mask with the bbox mask
    const auto* table = node->getTable();
    bool active = false;
    for (auto i = mask.beginOn(); !active && i; ++i) {
        active = this->anyActiveValues(table[i.pos()].getChild(), bbox);
    }
    return active;
}

template<typename TreeT>
template<typename NodeT>
bool FindActiveValues<TreeT>::anyActiveVoxels(const NodeT* node, const CoordBBox &bbox) const
{
    // Generate a bit mask of the bbox coverage
    auto mask = this->getBBoxMask(bbox, node);

    // Check child nodes
    mask &= node->getChildMask();// prune the child mask with the bbox mask
    const auto* table = node->getTable();
    bool active = false;
    for (auto i = mask.beginOn(); !active && i; ++i) {
        active = this->anyActiveVoxels(table[i.pos()].getChild(), bbox);
    }
    return active;
}

template<typename TreeT>
inline bool FindActiveValues<TreeT>::anyActiveVoxels(const typename TreeT::LeafNodeType* leaf, const CoordBBox &bbox ) const
{
    const auto &mask = leaf->getValueMask();

    // check for two common cases that leads to early-out
    if (bbox.isInside(leaf->getNodeBoundingBox())) return !mask.isOff();// leaf in inside the bbox
    if (mask.isOn()) return true;// all values are active

    bool active = false;
    for (auto i = leaf->cbeginValueOn(); !active && i; ++i) {
        active = bbox.isInside(i.getCoord());
    }
    return active;
}

template<typename TreeT>
template<typename NodeT>
bool FindActiveValues<TreeT>::anyActiveTiles(const NodeT* node, const CoordBBox &bbox) const
{
    // Generate a bit mask of the bbox coverage
    auto mask = this->getBBoxMask(bbox, node);

    // Check active tiles
    const auto tmp = mask & node->getValueMask();// prune active the tile mask with the bbox mask
    if (!tmp.isOff()) return true;

    bool active = false;
    if (NodeT::LEVEL>1) {// Only check child nodes if they are NOT leaf nodes
        mask &= node->getChildMask();// prune the child mask with the bbox mask
        const auto* table = node->getTable();
        for (auto i = mask.beginOn(); !active && i; ++i) {
            active = this->anyActiveTiles(table[i.pos()].getChild(), bbox);
        }
    }
    return active;
}

template<typename TreeT>
inline Index64 FindActiveValues<TreeT>::count(const typename TreeT::LeafNodeType* leaf, const CoordBBox &bbox ) const
{
    Index64 count = 0;
    auto b = leaf->getNodeBoundingBox();
    if (b.isInside(bbox)) { // leaf node is completely inside bbox
        count = leaf->onVoxelCount();
    } else if (leaf->isDense()) {
        b.intersect(bbox);
        count = b.volume();
    } else if (b.hasOverlap(bbox)) {
        for (auto i = leaf->cbeginValueOn(); i; ++i) {
            if (bbox.isInside(i.getCoord())) ++count;
        }
    }
    return count;
}

template<typename TreeT>
template<typename NodeT>
Index64 FindActiveValues<TreeT>::count(const NodeT* node, const CoordBBox &bbox) const
{
    Index64 count = 0;

    // Generate a bit masks
    auto mask = this->getBBoxMask(bbox, node);
    const auto childMask = mask & node->getChildMask();// prune the child mask with the bbox mask
    mask &= node->getValueMask();// prune active tile mask with the bbox mask
    const auto* table = node->getTable();

    {// Check child nodes
        using ChildT = typename NodeT::ChildNodeType;
        using RangeT = tbb::blocked_range<typename std::vector<const ChildT*>::iterator>;
        std::vector<const ChildT*> childNodes(childMask.countOn());
        int j=0;
        for (auto i = childMask.beginOn(); i; ++i, ++j) childNodes[j] = table[i.pos()].getChild();
        count += tbb::parallel_reduce( RangeT(childNodes.begin(), childNodes.end()), 0,
            [&](const RangeT& r, Index64 sum)->Index64 {
                for ( auto i = r.begin(); i != r.end(); ++i ) sum += this->count(*i, bbox);
                return sum;
            }, []( Index64 a, Index64 b )->Index64 { return a+b; }
        );
    }

    {// Check active tiles
        std::vector<Coord> coords(mask.countOn());
        using RangeT = tbb::blocked_range<typename std::vector<Coord>::iterator>;
        int j=0;
        for (auto i = mask.beginOn(); i; ++i, ++j) coords[j] = node->offsetToGlobalCoord(i.pos());
        count += tbb::parallel_reduce( RangeT(coords.begin(), coords.end()), 0,
            [&bbox](const RangeT& r, Index64 sum)->Index64 {
                for ( auto i = r.begin(); i != r.end(); ++i ) {
                    auto b = CoordBBox::createCube(*i, NodeT::ChildNodeType::DIM);
                    b.intersect(bbox);
                    sum += b.volume();
                }
                return sum;
            }, []( Index64 a, Index64 b )->Index64 { return a+b; }
        );
    }

    return count;
}

// process internal node
template<typename TreeT>
template<typename NodeT>
void FindActiveValues<TreeT>::activeTiles(const NodeT* node, const CoordBBox &bbox, std::vector<TileDataT> &tiles) const
{
    // Generate a bit masks
    auto mask = this->getBBoxMask(bbox, node);
    const auto childMask = mask & node->getChildMask();// prune the child mask with the bbox mask
    mask &= node->getValueMask();// prune active tile mask with the bbox mask

    if (NodeT::LEVEL > 1) {// Only check child nodes if they are NOT leaf nodes
        const auto* table = node->getTable();
        for (auto i = childMask.beginOn(); i; ++i) this->activeTiles(table[i.pos()].getChild(), bbox, tiles);
    }

    const size_t tileCount = mask.countOn();
    if (tileCount < 8) {// Serial processing of active tiles
        for (auto iter = mask.beginOn(); iter; ++iter) {
            tiles.emplace_back(*node, iter.pos());
            tiles.back().bbox.intersect(bbox);
        }
    } else {// Parallel processing of active tiles
        std::vector<TileDataT> tmp( tileCount );// for temporary thread-safe processing
        int n = 0;
        for (auto iter = mask.beginOn(); iter; ++iter) tmp[n++].level = iter.pos();// placeholder to support multi-threading
        tbb::parallel_for(tbb::blocked_range<size_t>(0, tileCount, 8), [&](const tbb::blocked_range<size_t>& r) {
            for ( size_t i = r.begin(); i != r.end(); ++i ) {
                tmp[i] = TileDataT(*node, tmp[i].level);
                tmp[i].bbox.intersect(bbox);
            }
        });
        tiles.insert(tiles.end(), tmp.begin(), tmp.end());
    }
}

template<typename TreeT>
struct FindActiveValues<TreeT>::RootChild
{
    const CoordBBox      bbox;
    const RootChildType* child;
    RootChild(const Coord& ijk = Coord(), const RootChildType* ptr = nullptr)
        : bbox(CoordBBox::createCube(ijk, RootChildType::DIM)), child(ptr)
    {
    }
};// RootChild struct

//////////////////////////////////////////////////////////////////////////////////////////

template<typename ValueType>
struct TileData
{
    CoordBBox bbox;
    ValueType value;
    Index     level;
    bool      state;

    /// @brief Default constructor
    TileData() = default;

    /// @brief Member data constructor
    TileData(const CoordBBox &b, const ValueType &v, Index l, bool active = true)
        : bbox(b), value(v), level(l), state(active) {}

    /// @brief Constructor from a parent node and the linear offset to one of its tiles
    ///
    /// @warning This is an expert-only method since it assumes the linear offset to be valid,
    ///          i.e. within the rand of the dimension of the parent node and NOT corresponding
    ///          to a child node.
    template <typename ParentNodeT>
    TileData(const ParentNodeT &parent, Index childIdx)
        : bbox(CoordBBox::createCube(parent.offsetToGlobalCoord(childIdx), parent.getChildDim()))
        , level(parent.getLevel())
        , state(true)
    {
        OPENVDB_ASSERT(childIdx < ParentNodeT::NUM_VALUES);
        OPENVDB_ASSERT(parent.isChildMaskOff(childIdx));
        OPENVDB_ASSERT(parent.isValueMaskOn(childIdx));
        value = parent.getTable()[childIdx].getValue();
    }

    /// @brief Constructor form a parent node, the coordinate of the origin of one of its tiles,
    ///        and said tiles value.
    template <typename ParentNodeT>
    TileData(const ParentNodeT &parent, const Coord &ijk, const ValueType &v)
        : bbox(CoordBBox::createCube(ijk, parent.getChildDim()))
        , value(v)
        , level(parent.getLevel())
        , state(true)
    {
    }
};// TileData struct

//////////////////////////////////////////////////////////////////////////////////////////

// Implementation of stand-alone function
template<typename TreeT>
bool
anyActiveValues(const TreeT& tree, const CoordBBox &bbox)
{
    FindActiveValues<TreeT> op(tree);
    return op.anyActiveValues(bbox);
}

// Implementation of stand-alone function
template<typename TreeT>
bool
anyActiveVoxels(const TreeT& tree, const CoordBBox &bbox)
{
    FindActiveValues<TreeT> op(tree);
    return op.anyActiveVoxels(bbox);
}

// Implementation of stand-alone function
template<typename TreeT>
bool
anyActiveTiles(const TreeT& tree, const CoordBBox &bbox)
{
    FindActiveValues<TreeT> op(tree);
    return op.anyActiveTiles(bbox);
}

// Implementation of stand-alone function
template<typename TreeT>
bool
noActiveValues(const TreeT& tree, const CoordBBox &bbox)
{
    FindActiveValues<TreeT> op(tree);
    return op.noActiveValues(bbox);
}

// Implementation of stand-alone function
template<typename TreeT>
Index64
countActiveValues(const TreeT& tree, const CoordBBox &bbox)
{
    return tools::countActiveVoxels(tree, bbox);
}

// Implementation of stand-alone function
template<typename TreeT>
std::vector<TileData<typename TreeT::ValueType>>
activeTiles(const TreeT& tree, const CoordBBox &bbox)
{
    FindActiveValues<TreeT> op(tree);
    return op.activeTiles(bbox);
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_FINDACTIVEVALUES
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    bool anyActiveValues(const TreeT&, const CoordBBox&)
OPENVDB_VOLUME_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    bool anyActiveVoxels(const TreeT&, const CoordBBox&)
OPENVDB_VOLUME_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    bool anyActiveTiles(const TreeT&, const CoordBBox&)
OPENVDB_VOLUME_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    bool noActiveValues(const TreeT&, const CoordBBox&)
OPENVDB_VOLUME_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Index64 countActiveValues(const TreeT&, const CoordBBox&)
OPENVDB_VOLUME_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    std::vector<TileData<TreeT::ValueType>> activeTiles(const TreeT&, const CoordBBox&)
OPENVDB_VOLUME_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_FINDACTIVEVALUES_HAS_BEEN_INCLUDED
