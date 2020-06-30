// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file Merge.h
///
/// @brief Functions to efficiently merge grids
///
/// @author Dan Bailey

#ifndef OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/NodeManager.h>

#include <unordered_map>
#include <unordered_set>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Convenience class that contains a pointer to a const or non-const tree
/// and a subset of methods to retrieve data from the tree.
///
/// @details This class has methods to extract data from the tree that will either
/// result in stealing the data or deep-copying it depending on whether the tree
/// is mutable or not.
template <typename TreeT>
struct TreeToMerge
{
    using TreeType = std::remove_const_t<TreeT>;
    using RootNodeType = typename TreeType::RootNodeType;
    using ValueType = typename TreeType::ValueType;

    TreeToMerge() = delete;

    /// @brief Non-const tree constructor.
    explicit TreeToMerge(TreeType* tree) : mTree(tree) { }
    /// @brief Const tree constructor. As the tree is not mutable and thus cannot be pruned, a lightweight
    /// mask tree with the same topology is created that can be pruned to use as a reference.
    explicit TreeToMerge(const TreeType* constTree) : mConstTree(constTree)
    {
        if (mConstTree)    this->initializeMask();
    }

    /// @brief Retrieve a const pointer to the root node.
    const RootNodeType* rootPtr() const;

    /// @brief Return a pointer to the node of type @c NodeT that contains
    /// voxel (x, y, z).  If no such node exists, return @c nullptr.
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& ijk) const;

    /// @brief Return a pointer to the node of type @c NodeT that contains voxel (x, y, z).
    /// If the tree is non-const, steal the node and replace it with an inactive
    /// background-value tile.
    /// If the tree is const, deep-copy the node and modify the mask tree to prune the node.
    template <typename NodeT>
    std::unique_ptr<NodeT> stealOrDeepCopyNode(const Coord& ijk);

    /// @brief Add a tile containing voxel (x, y, z) at the level of NodeT,
    /// deleting the existing branch if necessary.
    template <typename NodeT>
    void addTile(const Coord& ijk, const ValueType& value, bool active);

private:
    // build the mask using a topology union of the const tree
    void initializeMask();

    TreeType* const mTree = nullptr;
    const TreeType* const mConstTree = nullptr;
    MaskTree mMaskTree;
}; // struct TreeToMerge


////////////////////////////////////////


/// @brief NodeManager operator to merge multiple trees using a CSG union.
/// @note This class modifies the topology of the tree so is designed to be used
/// from DynamicNodeManager::foreachTopDown().
template<typename TreeT>
struct CsgUnionMergeOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    /// @brief Templated constructor. This can be used to pass in a container of trees
    /// such as vector<TreeT*> or vector<const TreeT*>. However it will also accept a
    /// container of mixed const/non-const trees by wrapping them in TreeToMerge objects
    /// such as vector<TreeToMerge<TreeT>>. Merge order is preserved in this case.
    template <typename TreesT>
    CsgUnionMergeOp(const TreesT& trees)
        : mTreesToMerge(trees.cbegin(), trees.cend()) { }

    /// @brief Initializer list constructor. This is convenient for small numbers of
    /// trees that are all const or all non-const.
    CsgUnionMergeOp(std::initializer_list<TreeT*> init)
        : mTreesToMerge(init.begin(), init.end()) { }

    /// @brief Return the number of trees being merged
    size_t size() const { return mTreesToMerge.size(); }

    // Processes the root node. Required by the NodeManager
    void operator()(RootT& root) const;

    // Processes the internal nodes. Required by the NodeManager
    template<typename NodeT>
    void operator()(NodeT& node) const;

    // Processes the leaf nodes. Required by the NodeManager
    void operator()(LeafT& leaf) const;

private:
    // on processing the root node, the background value is stored, retrieve it
    // and check that the root node has already been processed
    const ValueT& background() const;

    // note that this vector is copied in NodeTransformer every time a foreach call is made,
    // however in typical use cases this cost will be dwarfed by the actual merge algorithm
    mutable std::vector<TreeToMerge<TreeT>> mTreesToMerge;
    mutable const ValueT* mBackground = nullptr;
}; // struct CsgUnionMergeOp


////////////////////////////////////////


template<typename TreeT>
void TreeToMerge<TreeT>::initializeMask()
{
    mMaskTree.topologyUnion(*mConstTree);
}

template<typename TreeT>
const typename TreeToMerge<TreeT>::RootNodeType*
TreeToMerge<TreeT>::rootPtr() const
{
    if (mTree)               return &mTree->root();
    else if (mConstTree)     return &mConstTree->root();
    return nullptr;
}

template<typename TreeT>
template<typename NodeT>
const NodeT*
TreeToMerge<TreeT>::probeConstNode(const Coord& ijk) const
{
    if (mTree)               return mTree->template probeConstNode<NodeT>(ijk);
    else if (mConstTree) {
        // test mutable mask first, node may have already been pruned
        using MaskNodeT = typename NodeT::template ValueConverter<ValueMask>::Type;
        if (!mMaskTree.probeConstNode<MaskNodeT>(ijk))    return nullptr;
        return mConstTree->template probeConstNode<NodeT>(ijk);
    }
    return nullptr;
}

template<typename TreeT>
template<typename NodeT>
std::unique_ptr<NodeT>
TreeToMerge<TreeT>::stealOrDeepCopyNode(const Coord& ijk)
{
    if (mTree) {
        return std::unique_ptr<NodeT>(
            mTree->root().template stealNode<NodeT>(ijk, mTree->root().background(), false)
        );
    } else if (auto* child = this->probeConstNode<NodeT>(ijk)) {
        auto result = std::make_unique<NodeT>(*child);
        // prune mask tree
        mMaskTree.addTile(NodeT::LEVEL, ijk, false, false);
        return result;
    }
    return std::unique_ptr<NodeT>();
}

template<typename TreeT>
template<typename NodeT>
void
TreeToMerge<TreeT>::addTile(const Coord& ijk, const ValueType& value, bool active)
{
    if (mTree) {
        auto* node = mTree->template probeNode<NodeT>(ijk);
        const Index pos = NodeT::coordToOffset(ijk);
        node->addTile(pos, value, active);
    } else {
        // prune mask tree
        mMaskTree.addTile(NodeT::LEVEL, ijk, false, false);
    }
}


////////////////////////////////////////


template <typename TreeT>
void CsgUnionMergeOp<TreeT>::operator()(RootT& root) const
{
    if (mTreesToMerge.empty())     return;

    // store the background value
    if (!mBackground)   mBackground = &root.background();

    // find all tile values in this root and track inside/outside and active state
    // note that level sets should never contain active tiles, but we handle them anyway

    constexpr uint8_t ACTIVE_TILE = 0x1;
    constexpr uint8_t INSIDE_TILE = 0x2;
    constexpr uint8_t OUTSIDE_TILE = 0x4;

    auto getTileFlag = [](auto valueIter) -> uint8_t
    {
        uint8_t flag(0);
        const ValueT& value = valueIter.getValue();
        if (value < zeroVal<ValueT>())          flag |= INSIDE_TILE;
        else if (value > zeroVal<ValueT>())     flag |= OUTSIDE_TILE;
        if (valueIter.isValueOn())              flag |= ACTIVE_TILE;
        return flag;
    };

    std::unordered_map<Coord, /*flags*/uint8_t> tiles;

    if (root.getTableSize() > 0) {
        for (auto valueIter = root.cbeginValueAll(); valueIter; ++valueIter) {
            const Coord& key = valueIter.getCoord();
            tiles.insert({key, getTileFlag(valueIter)});
        }
    }

    // find all tiles values in other roots and replace outside tiles with inside tiles

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const auto* mergeRoot = mergeTree.rootPtr();
        if (!mergeRoot)     continue;
        for (auto valueIter = mergeRoot->cbeginValueAll(); valueIter; ++valueIter) {
            const Coord& key = valueIter.getCoord();
            auto it = tiles.find(key);
            if (it == tiles.end()) {
                // if no tile with this key, insert it
                tiles.insert({key, getTileFlag(valueIter)});
            } else {
                // replace an outside tile with an inside tile
                const uint8_t flag = it->second;
                if (flag & OUTSIDE_TILE) {
                    const uint8_t newFlag = getTileFlag(valueIter);
                    if (newFlag & INSIDE_TILE) {
                        it->second = newFlag;
                    }
                }
            }
        }
    }

    // insert all inside tiles

    for (auto it : tiles) {
        const uint8_t flag = it.second;
        if (flag & INSIDE_TILE) {
            const Coord& key = it.first;
            const bool state = flag & ACTIVE_TILE;
            root.addTile(key, -this->background(), state);
        }
    }

    std::unordered_set<Coord> children;

    if (root.getTableSize() > 0) {
        for (auto childIter = root.cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();
            children.insert(key);
        }
    }

    // find all children in other roots and insert them if a child or tile with this key
    // does not already exist or if the child will replace an outside tile

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const auto* mergeRoot = mergeTree.rootPtr();
        if (!mergeRoot)     continue;
        for (auto childIter = mergeRoot->cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();

            // if child already exists, do nothing
            if (children.count(key))    continue;

            // if an inside tile exists, do nothing
            auto it = tiles.find(key);
            if (it != tiles.end() && it->second == INSIDE_TILE)     continue;

            auto childPtr = mergeTree.template stealOrDeepCopyNode<typename RootT::ChildNodeType>(key);
            if (childPtr)   root.addChild(childPtr.release());

            children.insert(key);
        }
    }

    // insert all outside tiles that don't replace an inside tile or a child node

    for (auto it : tiles) {
        const uint8_t flag = it.second;
        if (flag & OUTSIDE_TILE) {
            const Coord& key = it.first;
            if (!children.count(key)) {
                const bool state = flag & ACTIVE_TILE;
                root.addTile(key, this->background(), state);
            }
        }
    }
}

template<typename TreeT>
template<typename NodeT>
void CsgUnionMergeOp<TreeT>::operator()(NodeT& node) const
{
    using NonConstNodeT = typename std::remove_const<NodeT>::type;

    if (mTreesToMerge.empty())     return;

    const ValueT outsideBackground = this->background();
    const ValueT insideBackground = -outsideBackground;

    using NodeMaskT = typename NodeT::NodeMaskType;

    // store temporary masks to track inside and outside tile states
    NodeMaskT insideTile;
    NodeMaskT outsideTile;

    for (auto iter = node.beginChildOff(); iter; ++iter) {
        if (iter.getValue() < zeroVal<ValueT>()) {
            insideTile.setOn(iter.pos());
        } else if (iter.getValue() > zeroVal<ValueT>()) {
            outsideTile.setOn(iter.pos());
        }
    }

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {

        auto* mergeNode = mergeTree.template probeConstNode<NonConstNodeT>(node.origin());

        if (!mergeNode)     continue;

        // iterate over all tiles

        for (auto iter = mergeNode->cbeginChildOff(); iter; ++iter) {
            Index pos = iter.pos();
            // source node contains an inside tile, so ignore
            if (insideTile.isOn(pos))   continue;
            // this node contains an inside tile, so turn into an inside tile
            if (iter.getValue() < zeroVal<ValueT>()) {
                node.addTile(pos, insideBackground, iter.isValueOn());
                insideTile.setOn(pos);
            }
        }

        // iterate over all child nodes

        for (auto iter = mergeNode->cbeginChildOn(); iter; ++iter) {
            Index pos = iter.pos();
            const Coord& ijk = iter.getCoord();
            // source node contains an inside tile, so ensure other node has no child
            if (insideTile.isOn(pos)) {
                mergeTree.template addTile<NonConstNodeT>(ijk, outsideBackground, false);
            } else if (outsideTile.isOn(pos)) {
                auto childPtr = mergeTree.template stealOrDeepCopyNode<typename NodeT::ChildNodeType>(ijk);
                if (childPtr)   node.addChild(childPtr.release());
                outsideTile.setOff(pos);
            }
        }
    }
}

template <typename TreeT>
void CsgUnionMergeOp<TreeT>::operator()(LeafT& leaf) const
{
    if (mTreesToMerge.empty())     return;

    if (!leaf.allocate())   return;

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const LeafT* mergeLeaf = mergeTree.template probeConstNode<LeafT>(leaf.origin());
        if (!mergeLeaf)     continue;

        for (Index i = 0 ; i < LeafT::SIZE; i++) {
            const ValueT& newValue = mergeLeaf->getValue(i);
            if (newValue < leaf.getValue(i)) {
                leaf.setValueOnly(i, newValue);
                leaf.setActiveState(i, mergeLeaf->isValueOn(i));
            }
        }
    }
}

template <typename TreeT>
const typename CsgUnionMergeOp<TreeT>::ValueT&
CsgUnionMergeOp<TreeT>::background() const
{
    if (!mBackground) {
        OPENVDB_THROW(RuntimeError, "Background value not set. "
            "This operator is only intended to be used with foreachTopDown().");
    }
    return *mBackground;
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED
