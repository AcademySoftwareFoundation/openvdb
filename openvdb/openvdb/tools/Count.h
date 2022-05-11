// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file   Count.h
///
/// @brief Functions to count tiles, nodes or voxels in a grid
///
/// @author Dan Bailey
///

#ifndef OPENVDB_TOOLS_COUNT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_COUNT_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/math/Stats.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tree/NodeManager.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Return the total number of active voxels in the tree.
template <typename TreeT>
Index64 countActiveVoxels(const TreeT& tree, bool threaded = true);


/// @brief Return the total number of active voxels in the tree that intersects
/// a bounding box.
template <typename TreeT>
Index64 countActiveVoxels(const TreeT& tree, const CoordBBox& bbox, bool threaded = true);


/// @brief Return the total number of active voxels stored in leaf nodes.
template <typename TreeT>
Index64 countActiveLeafVoxels(const TreeT& tree, bool threaded = true);


/// @brief Return the total number of active voxels stored in leaf nodes that intersects
/// a bounding box.
template <typename TreeT>
Index64 countActiveLeafVoxels(const TreeT& tree, const CoordBBox& bbox, bool threaded = true);


/// @brief Return the total number of inactive voxels in the tree.
template <typename TreeT>
Index64 countInactiveVoxels(const TreeT& tree, bool threaded = true);


/// @brief Return the total number of inactive voxels stored in leaf nodes.
template <typename TreeT>
Index64 countInactiveLeafVoxels(const TreeT& tree, bool threaded = true);


/// @brief Return the total number of active tiles in the tree.
template <typename TreeT>
Index64 countActiveTiles(const TreeT& tree, bool threaded = true);


/// @brief Return the total amount of memory in bytes occupied by this tree.
/// @details  This method returns the total in-core memory usage which can be
///   different to the maximum possible memory usage for trees which have not
///   been fully deserialized (via delay-loading). Thus, this is the current
///   true memory consumption.
template <typename TreeT>
Index64 memUsage(const TreeT& tree, bool threaded = true);


/// @brief Return the deserialized memory usage of this tree. This is not
///   necessarily equal to the current memory usage (returned by tools::memUsage)
///   if delay-loading is enabled. See File::open.
template <typename TreeT>
Index64 memUsageIfLoaded(const TreeT& tree, bool threaded = true);


/// @brief Return the minimum and maximum active values in this tree.
/// @note  Returns zeroVal<ValueType> for empty trees.
template <typename TreeT>
math::MinMax<typename TreeT::ValueType> minMax(const TreeT& tree, bool threaded = true);


////////////////////////////////////////

/// @cond OPENVDB_DOCS_INTERNAL

namespace count_internal {

/// @brief A DynamicNodeManager operator to count active voxels in a tree
template<typename TreeType>
struct ActiveVoxelCountOp
{
    using LeafT = typename TreeType::LeafNodeType;

    ActiveVoxelCountOp() = default;
    ActiveVoxelCountOp(const ActiveVoxelCountOp&, tbb::split) { }

    //  accumulate all voxels in active tile children
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            count += NodeT::ChildNodeType::NUM_VOXELS;
        }
        return true;
    }

    // accumulate all active voxels in the leaf
    bool operator()(const LeafT& leaf, size_t)
    {
        count += leaf.onVoxelCount();
        return false;
    }

    void join(const ActiveVoxelCountOp& other)
    {
        count += other.count;
    }

    openvdb::Index64 count{0};
}; // struct ActiveVoxelCountOp

/// @brief A DynamicNodeManager operator to count active voxels in a tree
/// that fall within a provided bounding box
template<typename TreeType>
struct ActiveVoxelCountBBoxOp
{
    using LeafT = typename TreeType::LeafNodeType;

    explicit ActiveVoxelCountBBoxOp(const CoordBBox& bbox)
        : mBBox(bbox) { }
    ActiveVoxelCountBBoxOp(const ActiveVoxelCountBBoxOp& other, tbb::split)
        : mBBox(other.mBBox) { }

    // accumulate all voxels in active tile children bounded by the bbox
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if (!mBBox.hasOverlap(node.getNodeBoundingBox()))   return false;

        // count any overlapping regions in active tiles
        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            CoordBBox bbox(CoordBBox::createCube(iter.getCoord(), NodeT::ChildNodeType::DIM));

            if (!bbox.hasOverlap(mBBox)) {
                // box is completely outside the active tile
                continue;
            } else if (bbox.isInside(mBBox)) {
                // bbox is completely inside the active tile
                count += mBBox.volume();
            } else if (mBBox.isInside(bbox)) {
                // active tile is completely inside bbox
                count += bbox.volume();
            } else {
                // partial overlap between tile and bbox
                bbox.intersect(mBBox);
                count += bbox.volume();
            }
        }

        // return true if any child nodes overlap with the bounding box
        for (auto iter = node.cbeginChildOn(); iter; ++iter) {
            if (mBBox.hasOverlap(iter->getNodeBoundingBox()))    return true;
        }

        // otherwise return false to prevent recursion along this branch
        return false;
    }

    // accumulate all active voxels in the leaf bounded by the bbox
    inline bool operator()(const LeafT& leaf, size_t)
    {
        // note: the true/false return value does nothing

        CoordBBox bbox = leaf.getNodeBoundingBox();

        if (mBBox.isInside(bbox)) {
            // leaf node is completely inside bbox
            count += leaf.onVoxelCount();
        } else if (!bbox.hasOverlap(mBBox)) {
            // bbox is completely outside the leaf node
            return false;
        } else if (leaf.isDense()) {
            // partial overlap between dense leaf node and bbox
            bbox.intersect(mBBox);
            count += bbox.volume();
        } else {
            // partial overlap between sparse leaf node and bbox
            for (auto i = leaf.cbeginValueOn(); i; ++i) {
                if (mBBox.isInside(i.getCoord())) ++count;
            }
        }
        return false;
    }

    void join(const ActiveVoxelCountBBoxOp& other)
    {
        count += other.count;
    }

    openvdb::Index64 count{0};
private:
    CoordBBox mBBox;
}; // struct ActiveVoxelCountBBoxOp

/// @brief A DynamicNodeManager operator to count inactive voxels in a tree
template<typename TreeType>
struct InactiveVoxelCountOp
{
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    InactiveVoxelCountOp() = default;
    InactiveVoxelCountOp(const InactiveVoxelCountOp&, tbb::split) { }

    // accumulate all inactive voxels in the root node
    bool operator()(const RootT& root, size_t)
    {
        for (auto iter = root.cbeginValueOff(); iter; ++iter) {
            // background tiles are not considered to contain inactive voxels
            if (!math::isApproxEqual(*iter, root.background())) {
                count += RootT::ChildNodeType::NUM_VOXELS;
            }
        }
        return true;
    }

    // accumulate all voxels in inactive tile children
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        for (auto iter = node.cbeginValueOff(); iter; ++iter) {
            if (node.isChildMaskOff(iter.pos())) {
                count += NodeT::ChildNodeType::NUM_VOXELS;
            }
        }
        return true;
    }

    // accumulate all inactive voxels in the leaf
    bool operator()(const LeafT& leaf, size_t)
    {
        count += leaf.offVoxelCount();
        return false;
    }

    void join(const InactiveVoxelCountOp& other)
    {
        count += other.count;
    }

    openvdb::Index64 count{0};
}; // struct InactiveVoxelCountOp

/// @brief A DynamicNodeManager operator to count active tiles in a tree
template<typename TreeType>
struct ActiveTileCountOp
{
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    ActiveTileCountOp() = default;
    ActiveTileCountOp(const ActiveTileCountOp&, tbb::split) { }

    // accumulate all active tiles in root node
    bool operator()(const RootT& root, size_t)
    {
        for (auto iter = root.cbeginValueOn(); iter; ++iter)    count++;
        return true;
    }

    // accumulate all active tiles in internal node
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        count += node.getValueMask().countOn();
        return true;
    }

    // do nothing (leaf nodes cannot contain tiles)
    bool operator()(const LeafT&, size_t)
    {
        return false;
    }

    void join(const ActiveTileCountOp& other)
    {
        count += other.count;
    }

    openvdb::Index64 count{0};
}; // struct ActiveTileCountOp

/// @brief A DynamicNodeManager operator to sum the number of bytes of memory used
template<typename TreeType>
struct MemUsageOp
{
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    MemUsageOp(const bool inCoreOnly) : mInCoreOnly(inCoreOnly) {}
    MemUsageOp(const MemUsageOp& other) : mCount(0), mInCoreOnly(other.mInCoreOnly) {}
    MemUsageOp(const MemUsageOp& other, tbb::split) : MemUsageOp(other) {}

    // accumulate size of the root node in bytes
    bool operator()(const RootT& root, size_t)
    {
        mCount += sizeof(root);
        return true;
    }

    // accumulate size of all child nodes in bytes
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        mCount += NodeT::NUM_VALUES * sizeof(typename NodeT::UnionType) +
            node.getChildMask().memUsage() + node.getValueMask().memUsage() +
            sizeof(Coord);
        return true;
    }

    // accumulate size of leaf node in bytes
    bool operator()(const LeafT& leaf, size_t)
    {
        if (mInCoreOnly) mCount += leaf.memUsage();
        else             mCount += leaf.memUsageIfLoaded();
        return false;
    }

    void join(const MemUsageOp& other)
    {
        mCount += other.mCount;
    }

    openvdb::Index64 mCount{0};
    const bool mInCoreOnly;
}; // struct MemUsageOp

/// @brief A DynamicNodeManager operator to find the minimum and maximum active values in this tree.
template<typename TreeType>
struct MinMaxValuesOp
{
    using ValueT = typename TreeType::ValueType;

    explicit MinMaxValuesOp()
        : min(zeroVal<ValueT>())
        , max(zeroVal<ValueT>())
        , seen_value(false) {}

    MinMaxValuesOp(const MinMaxValuesOp&, tbb::split)
        : MinMaxValuesOp() {}

    template <typename NodeType>
    bool operator()(NodeType& node, size_t)
    {
        if (auto iter = node.cbeginValueOn()) {
            if (!seen_value) {
                seen_value = true;
                min = max = *iter;
                ++iter;
            }

            for (; iter; ++iter) {
                const ValueT val = *iter;

                if (math::cwiseLessThan(val, min))
                    min = val;

                if (math::cwiseGreaterThan(val, max))
                    max = val;
            }
        }

        return true;
    }

    bool join(const MinMaxValuesOp& other)
    {
        if (!other.seen_value) return true;

        if (!seen_value) {
            min = other.min;
            max = other.max;
        }
        else {
            if (math::cwiseLessThan(other.min, min))
                min = other.min;
            if (math::cwiseGreaterThan(other.max, max))
                max = other.max;
        }

        seen_value = true;
        return true;
    }

    ValueT min, max;

private:

    bool seen_value;
}; // struct MinMaxValuesOp

} // namespace count_internal

/// @endcond


////////////////////////////////////////


template <typename TreeT>
Index64 countActiveVoxels(const TreeT& tree, bool threaded)
{
    count_internal::ActiveVoxelCountOp<TreeT> op;
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);
    return op.count;
}


template <typename TreeT>
Index64 countActiveVoxels(const TreeT& tree, const CoordBBox& bbox, bool threaded)
{
    if (bbox.empty())                   return Index64(0);
    else if (bbox == CoordBBox::inf())  return countActiveVoxels(tree, threaded);

    count_internal::ActiveVoxelCountBBoxOp<TreeT> op(bbox);
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);
    return op.count;
}


template <typename TreeT>
Index64 countActiveLeafVoxels(const TreeT& tree, bool threaded)
{
    count_internal::ActiveVoxelCountOp<TreeT> op;
    // use a leaf manager instead of a node manager
    tree::LeafManager<const TreeT> leafManager(tree);
    leafManager.reduce(op, threaded);
    return op.count;
}


template <typename TreeT>
Index64 countActiveLeafVoxels(const TreeT& tree, const CoordBBox& bbox, bool threaded)
{
    if (bbox.empty())                   return Index64(0);
    else if (bbox == CoordBBox::inf())  return countActiveLeafVoxels(tree, threaded);

    count_internal::ActiveVoxelCountBBoxOp<TreeT> op(bbox);
    // use a leaf manager instead of a node manager
    tree::LeafManager<const TreeT> leafManager(tree);
    leafManager.reduce(op, threaded);
    return op.count;
}


template <typename TreeT>
Index64 countInactiveVoxels(const TreeT& tree, bool threaded)
{
    count_internal::InactiveVoxelCountOp<TreeT> op;
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);
    return op.count;
}


template <typename TreeT>
Index64 countInactiveLeafVoxels(const TreeT& tree, bool threaded)
{
    count_internal::InactiveVoxelCountOp<TreeT> op;
    // use a leaf manager instead of a node manager
    tree::LeafManager<const TreeT> leafManager(tree);
    leafManager.reduce(op, threaded);
    return op.count;
}


template <typename TreeT>
Index64 countActiveTiles(const TreeT& tree, bool threaded)
{
    count_internal::ActiveTileCountOp<TreeT> op;
    // exclude leaf nodes as they cannot contain tiles
    tree::DynamicNodeManager<const TreeT, TreeT::DEPTH-2> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);
    return op.count;
}


template <typename TreeT>
Index64 memUsage(const TreeT& tree, bool threaded)
{
    count_internal::MemUsageOp<TreeT> op(true);
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);
    return op.mCount + sizeof(tree);
}

template <typename TreeT>
Index64 memUsageIfLoaded(const TreeT& tree, bool threaded)
{
    /// @note  For numeric (non-point) grids this really doesn't need to
    ///   traverse the tree and could instead be computed from the node counts.
    ///   We do so anyway as it ties this method into the tree data structure
    ///   which makes sure that changes to the tree/nodes are reflected/kept in
    ///   sync here.
    count_internal::MemUsageOp<TreeT> op(false);
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);
    return op.mCount + sizeof(tree);
}

template <typename TreeT>
math::MinMax<typename TreeT::ValueType> minMax(const TreeT& tree, bool threaded)
{
    using ValueT = typename TreeT::ValueType;

    count_internal::MinMaxValuesOp<TreeT> op;
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);

    return math::MinMax<ValueT>(op.min, op.max);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_COUNT_HAS_BEEN_INCLUDED
