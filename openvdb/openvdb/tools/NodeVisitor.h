// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file NodeVisitor.h
///
/// @author Dan Bailey
///
/// @brief  Implementation of a depth-first node visitor.
///
/// @note   This algorithm is single-threaded by design and intended for rare
///         use cases where this is desirable.  It is highly recommended to use
///         the NodeManager or DynamicNodeManager for much greater threaded
///         performance.

#ifndef OPENVDB_TOOLS_NODE_VISITOR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_NODE_VISITOR_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Visit all nodes in the tree depth-first and apply a user-supplied
///  functor to each node.
///
/// @param tree      tree to be visited.
/// @param op        user-supplied functor, see examples for interface details.
/// @param idx       optional offset to start sequential node indexing from a
///                  non-zero index.
///
/// @warning This method is single-threaded. Use the NodeManager or
///  DynamicNodeManager for much greater threaded performance.
///
/// @par Example:
/// @code
/// Functor to offset all the active values of a tree.
/// struct OffsetOp
/// {
///     OffsetOp(float v): mOffset(v) { }
///
///     template<typename NodeT>
///     void operator()(NodeT& node, size_t) const
///     {
///         for (auto iter = node.beginValueOn(); iter; ++iter) {
///             iter.setValue(iter.getValue() + mOffset);
///         }
///     }
/// private:
///     const float mOffset;
/// };
///
/// // usage:
/// OffsetOp op(3.0f);
/// visitNodesDepthFirst(tree, op);
///
/// // Functor to offset all the active values of a tree. Note
/// // this implementation also illustrates how different
/// // computation can be applied to the different node types.
/// template<typename TreeT>
/// struct OffsetByLevelOp
/// {
///     using ValueT = typename TreeT::ValueType;
///     using RootT = typename TreeT::RootNodeType;
///     using LeafT = typename TreeT::LeafNodeType;
///     OffsetByLevelOp(const ValueT& v) : mOffset(v) {}
///     // Processes the root node.
///     void operator()(RootT& root, size_t) const
///     {
///         for (auto iter = root.beginValueOn(); iter; ++iter) {
///             iter.setValue(iter.getValue() + mOffset);
///         }
///     }
///     // Processes the leaf nodes.
///     void operator()(LeafT& leaf, size_t) const
///     {
///         for (auto iter = leaf.beginValueOn(); iter; ++iter) {
///             iter.setValue(iter.getValue() + mOffset);
///         }
///     }
///     // Processes the internal nodes.
///     template<typename NodeT>
///     void operator()(NodeT& node, size_t) const
///     {
///         for (auto iter = node.beginValueOn(); iter; ++iter) {
///             iter.setValue(iter.getValue() + mOffset);
///         }
///     }
/// private:
///     const ValueT mOffset;
/// };
///
/// // usage:
/// OffsetByLevelOp<FloatTree> op(3.0f);
/// visitNodesDepthFirst(tree, op);
///
/// @endcode
///
/// @par Here is an example when migrating from using the deprecated Tree::visit()
/// method. The main difference between the Tree::visit() method and this new
/// method is that the Tree::visit() method expected an object that can visit
/// tiles, values and nodes. In contrast, the visitNodesDepthFirst() method
/// visits only nodes and expects you to provide iteration over tiles and
/// voxels.
///
/// @par Tree::visit() operator methods:
///
/// @code
/// template <typename IterT>
/// bool operator()(IterT &iter)
/// {
///     typename IterT::NonConstValueType value;
///     typename IterT::ChildNodeType *child = iter.probeChild(value);
///
///     if (child)
///     {
///         // If it is a leaf, process it now
///         if (child->getLevel() == 0)
///         {
///             processNode(*child);
///             return true;
///         }
///         // Otherwise, we want to keep digging down
///         return false;
///     }
///
///     // No child, this is a constant node!
///     if (iter.isValueOn())
///     {
///         openvdb::CoordBBox  b;
///         b.min() = iter.getCoord();
///         b.max() = b.min().offsetBy(IterT::ChildNodeType::DIM);
///
///         processNodeBlock(b);
///     }
///
///     // No need to dig further as there is no child.
///     return true;
/// }
///
/// bool operator()(typename GridType::TreeType::LeafNodeType::ChildAllIter &)
/// { return true; }
/// bool operator()(typename GridType::TreeType::LeafNodeType::ChildAllCIter &)
/// { return true; }
///
/// @endcode
///
/// @par tools::visitNodesDepthFirst() operator methods:
///
/// @code
/// using LeafT = typename GridType::TreeType::LeafNodeType;
///
/// template <typename NodeT>
/// void operator()(const NodeT &node, size_t)
/// {
///     // iterate over active tiles
///     for (auto iter = node.beginValueOn(); iter; ++iter)
///     {
///         openvdb::CoordBBox  b;
///         b.min() = iter.getCoord();
///         b.max() = b.min().offsetBy(NodeT::ChildNodeType::DIM);
///
///         processNodeBlock(b);
///     }
/// }
///
/// void operator()(const LeafT &leaf, size_t)
/// {
///     processNode(leaf);
/// }
///
/// @endcode
template <typename TreeT, typename OpT>
size_t visitNodesDepthFirst(TreeT& tree, OpT& op, size_t idx = 0);


/// @brief Visit all nodes that are downstream of a specific node in
///  depth-first order and apply a user-supplied functor to each node.
///
/// @note This uses the same operator interface as documented in
///  visitNodesDepthFirst().
///
/// @note The LEVEL template argument can be used to reduce the traversal
///  depth. For example, calling visit() with a RootNode and using
///  NodeT::LEVEL-1 would not visit leaf nodes.
template <typename NodeT, Index LEVEL = NodeT::LEVEL>
struct DepthFirstNodeVisitor;


////////////////////////////////////////


template <typename NodeT, Index LEVEL>
struct DepthFirstNodeVisitor
{
    using NonConstChildType = typename NodeT::ChildNodeType;
    using ChildNodeType = typename CopyConstness<NodeT, NonConstChildType>::Type;

    template <typename OpT>
    static size_t visit(NodeT& node, OpT& op, size_t idx = 0)
    {
        size_t offset = 0;
        op(node, idx + offset++);
        for (auto iter = node.beginChildOn(); iter; ++iter) {
            offset += DepthFirstNodeVisitor<ChildNodeType>::visit(
                *iter, op, idx + offset);
        }
        return offset;
    }
};


// terminate recursion
template <typename NodeT>
struct DepthFirstNodeVisitor<NodeT, 0>
{
    template <typename OpT>
    static size_t visit(NodeT& node, OpT& op, size_t idx = 0)
    {
        op(node, idx);
        return size_t(1);
    }
};


template <typename TreeT, typename OpT>
size_t visitNodesDepthFirst(TreeT& tree, OpT& op, size_t idx)
{
    using NonConstRootNodeType = typename TreeT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeT, NonConstRootNodeType>::Type;

    return DepthFirstNodeVisitor<RootNodeType>::visit(tree.root(), op, idx);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_NODE_VISITOR_HAS_BEEN_INCLUDED
