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
    using MaskTreeType = typename TreeT::template ValueConverter<ValueMask>::Type;

    TreeToMerge() = delete;

    /// @brief Non-const tree constructor.
    explicit TreeToMerge(TreeType* tree) : mTree(tree) { }
    /// @brief Const tree constructor. As the tree is not mutable and thus cannot be pruned, a lightweight
    /// mask tree with the same topology is created that can be pruned to use as a reference.
    explicit TreeToMerge(const TreeType* constTree) : mConstTree(constTree)
    {
        if (mConstTree)     this->initializeMask();
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
    // DynamicNodeManager operator for mask copying
    struct MaskUnionOp;

    // build a lightweight mask using a union of the const tree where leaf nodes
    // are converted into active tiles
    void initializeMask();

    TreeType* const mTree = nullptr;
    const TreeType* const mConstTree = nullptr;
    typename MaskTreeType::Ptr mMaskTree;
}; // struct TreeToMerge


template <typename TreeT>
struct TreeToMerge<TreeT>::MaskUnionOp
{
    using MaskT = MaskTreeType;
    using RootT = typename MaskT::RootNodeType;
    using LeafT = typename MaskT::LeafNodeType;

    explicit MaskUnionOp(const TreeT& tree) : mTree(tree) { }
    bool operator()(RootT& root, size_t) const;
    template<typename NodeT>
    bool operator()(NodeT& node, size_t) const;
    bool operator()(LeafT&, size_t) const { return false; }
private:
    const TreeT& mTree;
}; // struct TreeToMerge<TreeT>::MaskUnionOp


////////////////////////////////////////


/// @brief DynamicNodeManager operator to merge trees using a CSG union or intersection.
/// @note This class modifies the topology of the tree so is designed to be used
/// from DynamicNodeManager::foreachTopDown().
/// @details A union and an intersection are opposite operations to each other so
/// implemented in a combined class. Use the CsgUnionOp and CsgIntersectionOp aliases
/// for convenience.
template<typename TreeT, bool Union>
struct CsgUnionOrIntersectionOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    /// @brief Templated constructor. This can be used to pass in a container of trees
    /// such as vector<TreeT*> or vector<const TreeT*>. However it will also accept a
    /// container of mixed const/non-const trees by wrapping them in TreeToMerge objects
    /// such as vector<TreeToMerge<TreeT>>. Merge order is preserved in this case.
    template <typename TreesT>
    CsgUnionOrIntersectionOp(const TreesT& trees)
        : mTreesToMerge(trees.cbegin(), trees.cend()) { }

    /// @brief Initializer list constructor. This is convenient for small numbers of
    /// trees that are all const or all non-const.
    CsgUnionOrIntersectionOp(std::initializer_list<TreeT*> init)
        : mTreesToMerge(init.begin(), init.end()) { }

    /// @brief Return true if no trees being merged
    bool empty() const { return mTreesToMerge.empty(); }

    /// @brief Return the number of trees being merged
    size_t size() const { return mTreesToMerge.size(); }

    // Processes the root node. Required by the NodeManager
    bool operator()(RootT& root, size_t idx) const;

    // Processes the internal nodes. Required by the NodeManager
    template<typename NodeT>
    bool operator()(NodeT& node, size_t idx) const;

    // Processes the leaf nodes. Required by the NodeManager
    bool operator()(LeafT& leaf, size_t idx) const;

private:
    // on processing the root node, the background value is stored, retrieve it
    // and check that the root node has already been processed
    const ValueT& background() const;

    mutable std::vector<TreeToMerge<TreeT>> mTreesToMerge;
    mutable const ValueT* mBackground = nullptr;
}; // struct CsgUnionOrIntersectionOp


template <typename TreeT>
using CsgUnionOp = CsgUnionOrIntersectionOp<TreeT, /*Union=*/true>;

template <typename TreeT>
using CsgIntersectionOp = CsgUnionOrIntersectionOp<TreeT, /*Union=*/false>;


/// @brief DynamicNodeManager operator to merge two trees using a CSG difference.
/// @note This class modifies the topology of the tree so is designed to be used
/// from DynamicNodeManager::foreachTopDown().
template<typename TreeT>
struct CsgDifferenceOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    /// @brief Non-const tree constructor.
    explicit CsgDifferenceOp(TreeT& tree) : mTree(&tree) { }

    /// @brief Const tree constructor.
    explicit CsgDifferenceOp(const TreeT& tree) : mTree(&tree) { }

    /// @brief TreeToMerge<TreeT> constructor.
    explicit CsgDifferenceOp(TreeToMerge<TreeT>& tree) : mTree(tree) { }

    /// @brief Return the number of trees being merged (only ever 1)
    size_t size() const { return 1; }

    // Processes the root node. Required by the NodeManager
    bool operator()(RootT& root, size_t idx) const;

    // Processes the internal nodes. Required by the NodeManager
    template<typename NodeT>
    bool operator()(NodeT& node, size_t idx) const;

    // Processes the leaf nodes. Required by the NodeManager
    bool operator()(LeafT& leaf, size_t idx) const;

private:
    // on processing the root node, the background values are stored, retrieve them
    // and check that the root nodes have already been processed
    const ValueT& background() const;
    const ValueT& otherBackground() const;

    // note that this vector is copied in NodeTransformer every time a foreach call is made,
    // however in typical use cases this cost will be dwarfed by the actual merge algorithm
    mutable TreeToMerge<TreeT> mTree;
    mutable const ValueT* mBackground = nullptr;
    mutable const ValueT* mOtherBackground = nullptr;
}; // struct CsgDifferenceOp


////////////////////////////////////////


template<typename TreeT>
void TreeToMerge<TreeT>::initializeMask()
{
    mMaskTree.reset(new MaskTreeType);
    MaskUnionOp op(*mConstTree);
    tree::DynamicNodeManager<MaskTreeType, MaskTreeType::RootNodeType::LEVEL-1> manager(*mMaskTree);
    manager.foreachTopDown(op);
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
        if (!mMaskTree->isValueOn(ijk))    return nullptr;
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
    } else if (mConstTree) {
        auto* child = this->probeConstNode<NodeT>(ijk);
        if (child) {
            auto result = std::make_unique<NodeT>(*child);
            // prune mask tree
            mMaskTree->addTile(NodeT::LEVEL, ijk, false, false);
            return result;
        }
    }
    return std::unique_ptr<NodeT>();
}

template<typename TreeT>
template<typename NodeT>
void
TreeToMerge<TreeT>::addTile(const Coord& ijk, const ValueType& value, bool active)
{
    // ignore leaf node tiles (values)
    if (NodeT::LEVEL == 0)  return;

    if (mTree) {
        auto* node = mTree->template probeNode<NodeT>(ijk);
        if (node) {
            const Index pos = NodeT::coordToOffset(ijk);
            node->addTile(pos, value, active);
        }
    } else if (mConstTree) {
        auto* node = mConstTree->template probeConstNode<NodeT>(ijk);
        // prune mask tree
        if (node) {
            mMaskTree->addTile(NodeT::LEVEL, ijk, false, false);
        }
    }
}


////////////////////////////////////////


template <typename TreeT>
bool TreeToMerge<TreeT>::MaskUnionOp::operator()(RootT& root, size_t /*idx*/) const
{
    using ChildT = typename RootT::ChildNodeType;

    const Index count = mTree.root().childCount();

    std::vector<std::unique_ptr<ChildT>> children(count);

    // allocate new root children

    tbb::parallel_for(
        tbb::blocked_range<Index>(0, count),
        [&](tbb::blocked_range<Index>& range)
        {
            for (Index i = range.begin(); i < range.end(); i++) {
                children[i] = std::make_unique<ChildT>(Coord::max(), true, true);
            }
        }
    );

    // apply origins and add root children to new root node

    size_t i = 0;
    for (auto iter = mTree.root().cbeginChildOn(); iter; ++iter) {
        children[i]->setOrigin(iter->origin());
        root.addChild(children[i].release());
        i++;
    }

    return true;
}

template <typename TreeT>
template <typename NodeT>
bool TreeToMerge<TreeT>::MaskUnionOp::operator()(NodeT& node, size_t /*idx*/) const
{
    using ChildT = typename NodeT::ChildNodeType;

    const auto* otherNode = mTree.template probeConstNode<NodeT>(node.origin());
    if (!otherNode) return false;

    // this mask tree stores active tiles in place of leaf nodes for compactness

    if (NodeT::LEVEL == 1) {
        for (auto iter = otherNode->cbeginChildOn(); iter; ++iter) {
            node.addTile(iter.pos(), true, true);
        }
    } else {
        for (auto iter = otherNode->cbeginChildOn(); iter; ++iter) {
            auto* child = new ChildT(iter->origin(), true, true);
            node.addChild(child);
        }
    }

    return true;
}


////////////////////////////////////////


namespace merge_internal {


template <typename BufferT, typename ValueT>
struct UnallocatedBuffer
{
    static void allocateAndFill(BufferT& buffer, const ValueT& background)
    {
        if (!buffer.isOutOfCore() && buffer.empty()) {
            buffer.allocate();
            buffer.fill(background);
        }
    }

    static bool isPartiallyConstructed(const BufferT& buffer)
    {
        return !buffer.isOutOfCore() && buffer.empty();
    }
}; // struct AllocateAndFillBuffer

template <typename BufferT>
struct UnallocatedBuffer<BufferT, bool>
{
    // do nothing for bool buffers as they cannot be unallocated
    static void allocateAndFill(BufferT&, const bool&) { }
    static bool isPartiallyConstructed(const BufferT&) { return false; }
}; // struct AllocateAndFillBuffer


} // namespace merge_internal


////////////////////////////////////////


template <typename TreeT, bool Union>
bool CsgUnionOrIntersectionOp<TreeT, Union>::operator()(RootT& root, size_t) const
{
    if (this->empty())     return false;

    // store the background value
    if (!mBackground)   mBackground = &root.background();

    // find all tile values in this root and track inside/outside and active state
    // note that level sets should never contain active tiles, but we handle them anyway

    constexpr uint8_t ACTIVE_TILE = 0x1;
    constexpr uint8_t INSIDE_TILE = 0x2;
    constexpr uint8_t OUTSIDE_TILE = 0x4;

    constexpr uint8_t INSIDE_STATE = Union ? INSIDE_TILE : OUTSIDE_TILE;
    constexpr uint8_t OUTSIDE_STATE = Union ? OUTSIDE_TILE : INSIDE_TILE;

    const ValueT insideBackground = Union ? -this->background() : this->background();
    const ValueT outsideBackground = -insideBackground;

    auto getTileFlag = [&](auto valueIter) -> uint8_t
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
                if (flag & OUTSIDE_STATE) {
                    const uint8_t newFlag = getTileFlag(valueIter);
                    if (newFlag & INSIDE_STATE) {
                        it->second = newFlag;
                    }
                }
            }
        }
    }

    // insert all inside tiles

    for (auto it : tiles) {
        const uint8_t flag = it.second;
        if (flag & INSIDE_STATE) {
            const Coord& key = it.first;
            const bool state = flag & ACTIVE_TILE;
            root.addTile(key, insideBackground, state);
        }
    }

    std::unordered_set<Coord> children;

    if (root.getTableSize() > 0) {
        for (auto childIter = root.cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();
            children.insert(key);
        }
    }

    bool continueRecurse = false;

    // find all children in other roots and insert them if a child or tile with this key
    // does not already exist or if the child will replace an outside tile

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const auto* mergeRoot = mergeTree.rootPtr();
        if (!mergeRoot)     continue;
        for (auto childIter = mergeRoot->cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();

            // if child already exists, merge recursion will need to continue to resolve conflict
            if (children.count(key)) {
                continueRecurse = true;
                continue;
            }

            // if an inside tile exists, do nothing
            auto it = tiles.find(key);
            if (it != tiles.end() && it->second == INSIDE_STATE)     continue;

            auto childPtr = mergeTree.template stealOrDeepCopyNode<typename RootT::ChildNodeType>(key);
            if (childPtr)   root.addChild(childPtr.release());

            children.insert(key);
        }
    }

    // insert all outside tiles that don't replace an inside tile or a child node

    for (auto it : tiles) {
        const uint8_t flag = it.second;
        if (flag & OUTSIDE_STATE) {
            const Coord& key = it.first;
            if (!children.count(key)) {
                const bool state = flag & ACTIVE_TILE;
                root.addTile(key, outsideBackground, state);
            }
        }
    }

    return continueRecurse;
}

template<typename TreeT, bool Union>
template<typename NodeT>
bool CsgUnionOrIntersectionOp<TreeT, Union>::operator()(NodeT& node, size_t) const
{
    using NonConstNodeT = typename std::remove_const<NodeT>::type;

    if (this->empty())     return false;

    const ValueT insideBackground = Union ? -this->background() : this->background();
    const ValueT outsideBackground = -insideBackground;

    using NodeMaskT = typename NodeT::NodeMaskType;

    // store temporary masks to track inside and outside tile states
    NodeMaskT validTile;
    NodeMaskT invalidTile;

    auto isValid = [](const ValueT& value)
    {
        return Union ? value < zeroVal<ValueT>() : value > zeroVal<ValueT>();
    };

    auto isInvalid = [](const ValueT& value)
    {
        return Union ? value > zeroVal<ValueT>() : value < zeroVal<ValueT>();
    };

    for (auto iter = node.cbeginValueAll(); iter; ++iter) {
        if (isValid(iter.getValue())) {
            validTile.setOn(iter.pos());
        } else if (isInvalid(iter.getValue())) {
            invalidTile.setOn(iter.pos());
        }
    }

    bool continueRecurse = false;

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {

        auto* mergeNode = mergeTree.template probeConstNode<NonConstNodeT>(node.origin());

        if (!mergeNode)     continue;

        // iterate over all tiles

        for (auto iter = mergeNode->cbeginValueAll(); iter; ++iter) {
            Index pos = iter.pos();
            // source node contains an inside tile, so ignore
            if (validTile.isOn(pos))   continue;
            // this node contains an inside tile, so turn into an inside tile
            if (isValid(iter.getValue())) {
                node.addTile(pos, insideBackground, iter.isValueOn());
                validTile.setOn(pos);
            }
        }

        // iterate over all child nodes

        for (auto iter = mergeNode->cbeginChildOn(); iter; ++iter) {
            Index pos = iter.pos();
            const Coord& ijk = iter.getCoord();
            // source node contains an inside tile, so ensure other node has no child
            if (validTile.isOn(pos)) {
                mergeTree.template addTile<NonConstNodeT>(ijk, outsideBackground, false);
            } else if (invalidTile.isOn(pos)) {
                auto childPtr = mergeTree.template stealOrDeepCopyNode<typename NodeT::ChildNodeType>(ijk);
                if (childPtr)   node.addChild(childPtr.release());
                invalidTile.setOff(pos);
            } else {
                // if both source and target are child nodes, merge recursion needs to continue
                // along this branch to resolve the conflict
                continueRecurse = true;
            }
        }
    }

    return continueRecurse;
}

template <typename TreeT, bool Union>
bool CsgUnionOrIntersectionOp<TreeT, Union>::operator()(LeafT& leaf, size_t) const
{
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename LeafT::ValueType;
    using BufferT = typename LeafT::Buffer;

    if (this->empty())      return false;

    const ValueT background = Union ? this->background() : -this->background();

    // if buffer is not out-of-core and empty, leaf node must have only been
    // partially constructed, so allocate and fill with background value

    merge_internal::UnallocatedBuffer<BufferT, ValueT>::allocateAndFill(
        leaf.buffer(), background);

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const LeafT* mergeLeaf = mergeTree.template probeConstNode<LeafT>(leaf.origin());
        if (!mergeLeaf)     continue;
        // if buffer is not out-of-core yet empty, leaf node must have only been
        // partially constructed, so skip merge
        if (merge_internal::UnallocatedBuffer<BufferT, ValueT>::isPartiallyConstructed(
            mergeLeaf->buffer())) {
            continue;
        }

        for (Index i = 0 ; i < LeafT::SIZE; i++) {
            const ValueT& newValue = mergeLeaf->getValue(i);
            const bool doMerge = Union ? newValue < leaf.getValue(i) : newValue > leaf.getValue(i);
            if (doMerge) {
                leaf.setValueOnly(i, newValue);
                leaf.setActiveState(i, mergeLeaf->isValueOn(i));
            }
        }
    }

    return false;
}

template <typename TreeT, bool Union>
const typename CsgUnionOrIntersectionOp<TreeT, Union>::ValueT&
CsgUnionOrIntersectionOp<TreeT, Union>::background() const
{
    if (!mBackground) {
        OPENVDB_THROW(RuntimeError, "Background value not set. "
            "This operator is only intended to be used with foreachTopDown().");
    }
    return *mBackground;
}


////////////////////////////////////////


template <typename TreeT>
bool CsgDifferenceOp<TreeT>::operator()(RootT& root, size_t) const
{
    // store the background values
    if (!mBackground)       mBackground = &root.background();
    if (!mOtherBackground)  mOtherBackground = &mTree.rootPtr()->background();

    // find all tile values in this root and track inside/outside and active state
    // note that level sets should never contain active tiles, but we handle them anyway

    constexpr uint8_t ACTIVE_TILE = 0x1;
    constexpr uint8_t INSIDE_TILE = 0x2;
    constexpr uint8_t CHILD = 0x4;

    auto getTileFlag = [&](auto valueIter) -> uint8_t
    {
        uint8_t flag(0);
        const ValueT& value = valueIter.getValue();
        if (value < zeroVal<ValueT>())          flag |= INSIDE_TILE;
        if (valueIter.isValueOn())              flag |= ACTIVE_TILE;
        return flag;
    };

    std::unordered_map<Coord, /*flags*/uint8_t> flags;

    if (root.getTableSize() > 0) {
        for (auto valueIter = root.cbeginValueAll(); valueIter; ++valueIter) {
            const Coord& key = valueIter.getCoord();
            const uint8_t flag = getTileFlag(valueIter);
            if (flag & INSIDE_TILE) {
                flags.insert({key, getTileFlag(valueIter)});
            }
        }

        for (auto childIter = root.cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();
            flags.insert({key, CHILD});
        }
    }

    bool continueRecurse = false;

    const auto* mergeRoot = mTree.rootPtr();

    if (mergeRoot) {
        for (auto valueIter = mergeRoot->cbeginValueAll(); valueIter; ++valueIter) {
            const Coord& key = valueIter.getCoord();
            const uint8_t flag = getTileFlag(valueIter);
            if (flag & INSIDE_TILE) {
                auto it = flags.find(key);
                if (it != flags.end()) {
                    const bool state = flag & ACTIVE_TILE;
                    root.addTile(key, this->background(), state);
                }
            }
        }

        for (auto childIter = mergeRoot->cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();
            auto it = flags.find(key);
            if (it != flags.end()) {
                const uint8_t otherFlag = it->second;
                if (otherFlag & CHILD) {
                    // if child already exists, merge recursion will need to continue to resolve conflict
                    continueRecurse = true;
                } else if (otherFlag & INSIDE_TILE) {
                    auto childPtr = mTree.template stealOrDeepCopyNode<typename RootT::ChildNodeType>(key);
                    if (childPtr) {
                        childPtr->resetBackground(this->otherBackground(), this->background());
                        childPtr->negate();
                        root.addChild(childPtr.release());
                    }
                }
            }
        }
    }

    return continueRecurse;
}

template<typename TreeT>
template<typename NodeT>
bool CsgDifferenceOp<TreeT>::operator()(NodeT& node, size_t) const
{
    using NonConstNodeT = typename std::remove_const<NodeT>::type;

    using NodeMaskT = typename NodeT::NodeMaskType;

    // store temporary mask to track inside tile state

    NodeMaskT insideTile;
    for (auto iter = node.cbeginValueAll(); iter; ++iter) {
        if (iter.getValue() < zeroVal<ValueT>()) {
            insideTile.setOn(iter.pos());
        }
    }

    bool continueRecurse = false;

    auto* mergeNode = mTree.template probeConstNode<NonConstNodeT>(node.origin());

    if (!mergeNode)     return continueRecurse;

    // iterate over all tiles

    for (auto iter = mergeNode->cbeginValueAll(); iter; ++iter) {
        Index pos = iter.pos();
        if (iter.getValue() < zeroVal<ValueT>()) {
            if (insideTile.isOn(pos) || node.isChildMaskOn(pos))   {
                node.addTile(pos, this->background(), iter.isValueOn());
            }
        }
    }

    // iterate over all children

    for (auto iter = mergeNode->cbeginChildOn(); iter; ++iter) {
        Index pos = iter.pos();
        const Coord& ijk = iter.getCoord();
        if (insideTile.isOn(pos)) {
            auto childPtr = mTree.template stealOrDeepCopyNode<typename NodeT::ChildNodeType>(ijk);
            if (childPtr) {
                childPtr->resetBackground(this->otherBackground(), this->background());
                childPtr->negate();
                node.addChild(childPtr.release());
            }
        } else if (node.isChildMaskOn(pos)) {
            // if both source and target are child nodes, merge recursion needs to continue
            // along this branch to resolve the conflict
            continueRecurse = true;
        }
    }

    return continueRecurse;
}

template <typename TreeT>
bool CsgDifferenceOp<TreeT>::operator()(LeafT& leaf, size_t) const
{
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename LeafT::ValueType;
    using BufferT = typename LeafT::Buffer;

    // if buffer is not out-of-core and empty, leaf node must have only been
    // partially constructed, so allocate and fill with background value

    merge_internal::UnallocatedBuffer<BufferT, ValueT>::allocateAndFill(
        leaf.buffer(), this->background());

    const LeafT* mergeLeaf = mTree.template probeConstNode<LeafT>(leaf.origin());
    if (!mergeLeaf)                 return false;

    // if buffer is not out-of-core yet empty, leaf node must have only been
    // partially constructed, so skip merge

    if (merge_internal::UnallocatedBuffer<BufferT, ValueT>::isPartiallyConstructed(
        mergeLeaf->buffer())) {
        return false;
    }

    for (Index i = 0 ; i < LeafT::SIZE; i++) {
        const ValueT& aValue = leaf.getValue(i);
        ValueT bValue = math::negative(mergeLeaf->getValue(i));
        if (aValue < bValue) { // a = max(a, -b)
            leaf.setValueOnly(i, bValue);
            leaf.setActiveState(i, mergeLeaf->isValueOn(i));
        }
    }

    return false;
}

template <typename TreeT>
const typename CsgDifferenceOp<TreeT>::ValueT&
CsgDifferenceOp<TreeT>::background() const
{
    if (!mBackground) {
        OPENVDB_THROW(RuntimeError, "Background value not set. "
            "This operator is only intended to be used with foreachTopDown().");
    }
    return *mBackground;
}

template <typename TreeT>
const typename CsgDifferenceOp<TreeT>::ValueT&
CsgDifferenceOp<TreeT>::otherBackground() const
{
    if (!mOtherBackground) {
        OPENVDB_THROW(RuntimeError, "Other background value not set. "
            "This operator is only intended to be used with foreachTopDown().");
    }
    return *mOtherBackground;
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED
