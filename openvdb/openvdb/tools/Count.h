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


/// @brief Return the total amount of memory in bytes occupied by this tree.
template <typename TreeT>
Index64 memUsage(const TreeT& tree, bool threaded = true);


////////////////////////////////////////


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

/// @brief A DynamicNodeManager operator to sum the number of bytes of memory used
template<typename TreeType>
struct MemUsageOp
{
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    MemUsageOp() = default;
    MemUsageOp(const MemUsageOp&, tbb::split) { }

    // accumulate size of the root node in bytes
    bool operator()(const RootT& root, size_t)
    {
        count += sizeof(root);
        return true;
    }

    // accumulate size of all child nodes in bytes
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        count += NodeT::NUM_VALUES * sizeof(typename NodeT::UnionType) +
            node.getChildMask().memUsage() + node.getValueMask().memUsage() +
            sizeof(Coord);
        return true;
    }

    // accumulate size of leaf node in bytes
    bool operator()(const LeafT& leaf, size_t)
    {
        count += leaf.memUsage();
        return false;
    }

    void join(const MemUsageOp& other)
    {
        count += other.count;
    }

    openvdb::Index64 count{0};
}; // struct MemUsageOp

} // namespace count_internal


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
Index64 memUsage(const TreeT& tree, bool threaded)
{
    count_internal::MemUsageOp<TreeT> op;
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, threaded);
    return op.count + sizeof(tree);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_COUNT_HAS_BEEN_INCLUDED
