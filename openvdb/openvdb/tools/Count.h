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


/// @brief Return the total amount of memory in bytes occupied by this tree.
template <typename TreeT>
Index64 memUsage(const TreeT& tree, bool threaded = true);


////////////////////////////////////////


namespace count_internal {

template<typename TreeType>
struct ActiveVoxelCountOp
{
    using LeafT = typename TreeType::LeafNodeType;

    ActiveVoxelCountOp() = default;
    ActiveVoxelCountOp(const ActiveVoxelCountOp&, tbb::split) { }

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            count += NodeT::ChildNodeType::NUM_VOXELS;
        }
        return true;
    }

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
};

template<typename TreeType>
struct MemUsageOp
{
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    MemUsageOp() = default;
    MemUsageOp(const MemUsageOp&, tbb::split) { }

    bool operator()(const RootT& root, size_t)
    {
        count += sizeof(root);
        return true;
    }

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        count += NodeT::NUM_VALUES * sizeof(typename NodeT::UnionType) +
            node.getChildMask().memUsage() + node.getValueMask().memUsage() +
            sizeof(Coord);
        return true;
    }

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
};

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
