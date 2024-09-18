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
#include <openvdb/openvdb.h>
#include <openvdb/util/Assert.h>

#include "NodeVisitor.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Convenience class that contains a pointer to a tree to be stolen or
/// deep copied depending on the tag dispatch class used and a subset of
/// methods to retrieve data from the tree.
///
/// @details The primary purpose of this class is to be able to create an array
/// of TreeToMerge objects that each store a tree to be stolen or a tree to be
/// deep-copied in an arbitrary order. Certain operations such as floating-point
/// addition are non-associative so the order in which they are merged is
/// important for the operation to remain deterministic regardless of how the
/// data is being extracted from the tree.
///
/// @note Stealing data requires a non-const tree pointer. There is a constructor
/// to pass in a tree shared pointer for cases where it is desirable for this class
/// to maintain shared ownership.
template <typename TreeT>
struct TreeToMerge
{
    using TreeType = std::remove_const_t<TreeT>;
    using RootNodeType = typename TreeType::RootNodeType;
    using ValueType = typename TreeType::ValueType;
    using MaskTreeType = typename TreeT::template ValueConverter<ValueMask>::Type;

    TreeToMerge() = delete;

    /// @brief Non-const pointer tree constructor for stealing data.
    TreeToMerge(TreeType& tree, Steal)
        : mTree(&tree), mSteal(true) { }
    /// @brief Non-const shared pointer tree constructor for stealing data.
    TreeToMerge(typename TreeType::Ptr treePtr, Steal)
        : mTreePtr(treePtr), mTree(mTreePtr.get()), mSteal(true) { }

    /// @brief Const tree pointer constructor for deep-copying data. As the
    /// tree is not mutable and thus cannot be pruned, a lightweight mask tree
    /// with the same topology is created that can be pruned to use as a
    /// reference. Initialization of this mask tree can optionally be disabled
    /// for delayed construction.
    TreeToMerge(const TreeType& tree, DeepCopy, bool initialize = true)
        : mTree(&tree), mSteal(false)
    {
        if (mTree && initialize)     this->initializeMask();
    }

    /// @brief Non-const tree pointer constructor for deep-copying data. The
    /// tree is not intended to be modified so is not pruned, instead a
    /// lightweight mask tree with the same topology is created that can be
    /// pruned to use as a reference. Initialization of this mask tree can
    /// optionally be disabled for delayed construction.
    TreeToMerge(TreeType& tree, DeepCopy tag, bool initialize = true)
        : TreeToMerge(static_cast<const TreeType&>(tree), tag, initialize) { }

    /// @brief Reset the non-const tree shared pointer. This is primarily
    /// used to preserve the order of trees to merge in a container but have
    /// the data in the tree be lazily loaded or resampled.
    void reset(typename TreeType::Ptr treePtr, Steal);

    /// @brief Return a pointer to the tree to be stolen.
    TreeType* treeToSteal() { return mSteal ? const_cast<TreeType*>(mTree) : nullptr; }
    /// @brief Return a pointer to the tree to be deep-copied.
    const TreeType* treeToDeepCopy() { return mSteal ? nullptr : mTree; }

    /// @brief Retrieve a const pointer to the root node.
    const RootNodeType* rootPtr() const;

    /// @brief Return a pointer to the node of type @c NodeT that contains
    /// voxel (x, y, z).  If no such node exists, return @c nullptr.
    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& ijk) const;

    /// @brief Prune the mask and remove the node associated with this coord.
    void pruneMask(Index level, const Coord& ijk);

    /// @brief Return a pointer to the node of type @c NodeT that contains voxel (x, y, z).
    /// If the tree is non-const, steal the node and replace it with the value provided.
    /// If the tree is const, deep-copy the node and modify the mask tree to prune the node.
    template <typename NodeT>
    std::unique_ptr<NodeT> stealOrDeepCopyNode(const Coord& ijk, const ValueType& value);

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

    // build a lightweight mask using a union of the const tree where leaf nodes
    // are converted into active tiles
    void initializeMask();

    // returns true if mask has been initialized
    bool hasMask() const;

    // returns MaskTree pointer or nullptr
    MaskTreeType* mask() { return mMaskTree.ptr.get(); }
    const MaskTreeType* mask() const { return mMaskTree.ptr.get(); }

private:
    struct MaskPtr;
    struct MaskUnionOp;

    typename TreeType::Ptr mTreePtr;
    const TreeType* mTree;
    MaskPtr mMaskTree;
    bool mSteal;
}; // struct TreeToMerge


/// @brief Wrapper around unique_ptr that deep-copies mask on copy construction
template <typename TreeT>
struct TreeToMerge<TreeT>::MaskPtr
{
    std::unique_ptr<MaskTreeType> ptr;

    MaskPtr() = default;
    ~MaskPtr() = default;
    MaskPtr(MaskPtr&& other) = default;
    MaskPtr& operator=(MaskPtr&& other) = default;
    MaskPtr(const MaskPtr& other)
        : ptr(bool(other.ptr) ? std::make_unique<MaskTreeType>(*other.ptr) : nullptr) { }
    MaskPtr& operator=(const MaskPtr& other)
    {
        if (bool(other.ptr))    ptr = std::make_unique<MaskTreeType>(*other.ptr);
        else                    ptr.reset();
        return *this;
    }
};

/// @brief DynamicNodeManager operator used to generate a mask of the input
/// tree, but with dense leaf nodes replaced with active tiles for compactness
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

    /// @brief Convenience constructor to CSG union or intersect a single
    /// non-const tree with another. This constructor takes a Steal or DeepCopy
    /// tag dispatch class.
    template <typename TagT>
    CsgUnionOrIntersectionOp(TreeT& tree, TagT tag) { mTreesToMerge.emplace_back(tree, tag); }

    /// @brief Convenience constructor to CSG union or intersect a single
    /// const tree with another. This constructor requires a DeepCopy tag
    /// dispatch class.
    CsgUnionOrIntersectionOp(const TreeT& tree, DeepCopy tag) { mTreesToMerge.emplace_back(tree, tag); }

    /// @brief Constructor to CSG union or intersect a container of multiple
    /// const or non-const tree pointers. A Steal tag requires a container of
    /// non-const trees, a DeepCopy tag will accept either const or non-const
    /// trees.
    template <typename TreesT, typename TagT>
    CsgUnionOrIntersectionOp(TreesT& trees, TagT tag)
    {
        for (auto* tree : trees) {
            if (tree) {
                mTreesToMerge.emplace_back(*tree, tag);
            }
        }
    }

    /// @brief Constructor to accept a vector of TreeToMerge objects, primarily
    /// used when mixing const/non-const trees.
    /// @note Union/intersection order is preserved.
    explicit CsgUnionOrIntersectionOp(const std::vector<TreeToMerge<TreeT>>& trees)
        : mTreesToMerge(trees) { }

    /// @brief Constructor to accept a deque of TreeToMerge objects, primarily
    /// used when mixing const/non-const trees.
    /// @note Union/intersection order is preserved.
    explicit CsgUnionOrIntersectionOp(const std::deque<TreeToMerge<TreeT>>& trees)
        : mTreesToMerge(trees.cbegin(), trees.cend()) { }

    /// @brief Return true if no trees being merged
    bool empty() const { return mTreesToMerge.empty(); }

    /// @brief Return the number of trees being merged
    size_t size() const { return mTreesToMerge.size(); }

    /// Enables immediate pruning of tiles that cancel each other out.
    void setPruneCancelledTiles(bool doprune) { mPruneCancelledTiles = doprune; }

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
    bool mPruneCancelledTiles = false;
}; // struct CsgUnionOrIntersectionOp


template <typename TreeT>
using CsgUnionOp = CsgUnionOrIntersectionOp<TreeT, /*Union=*/true>;

template <typename TreeT>
using CsgIntersectionOp = CsgUnionOrIntersectionOp<TreeT, /*Union=*/false>;


/// @brief DynamicNodeManager operator to merge two trees using a CSG difference.
/// @note This class modifies the topology of the tree so is designed to be used
/// from DynamicNodeManager::foreachTopDown().
/// PruneCancelledTiles will set to background any leaf tile that matches
/// in the two trees, thus minimizing ghost banding when common borders
/// are differenced.
template<typename TreeT>
struct CsgDifferenceOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    /// @brief Convenience constructor to CSG difference a single non-const
    /// tree from another. This constructor takes a Steal or DeepCopy tag
    /// dispatch class.
    template <typename TagT>
    CsgDifferenceOp(TreeT& tree, TagT tag) : mTree(tree, tag) { }
    /// @brief Convenience constructor to CSG difference a single const
    /// tree from another. This constructor requires an explicit DeepCopy tag
    /// dispatch class.
    CsgDifferenceOp(const TreeT& tree, DeepCopy tag) : mTree(tree, tag) { }

    /// @brief Constructor to CSG difference the tree in a TreeToMerge object
    /// from another.
    explicit CsgDifferenceOp(TreeToMerge<TreeT>& tree) : mTree(tree) { }

    /// Enables immediate pruning of tiles that cancel each other out.
    void setPruneCancelledTiles(bool doprune) { mPruneCancelledTiles = doprune; }

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
    bool mPruneCancelledTiles = false;
}; // struct CsgDifferenceOp


/// @brief DynamicNodeManager operator to merge trees using a sum operation.
/// @note This class modifies the topology of the tree so is designed to be used
/// from DynamicNodeManager::foreachTopDown().
template<typename TreeT>
struct SumMergeOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    /// @brief Convenience constructor to sum a single non-const tree with another.
    /// This constructor takes a Steal or DeepCopy tag dispatch class.
    template <typename TagT>
    SumMergeOp(TreeT& tree, TagT tag) { mTreesToMerge.emplace_back(tree, tag); }

    /// @brief Convenience constructor to sum a single const tree with another.
    /// This constructor requires a DeepCopy tag dispatch class.
    SumMergeOp(const TreeT& tree, DeepCopy tag) { mTreesToMerge.emplace_back(tree, tag); }

    /// @brief Constructor to sum a container of multiple const or non-const tree pointers.
    /// A Steal tag requires a container of non-const trees, a DeepCopy tag will accept
    /// either const or non-const trees.
    template <typename TreesT, typename TagT>
    SumMergeOp(TreesT& trees, TagT tag)
    {
        for (auto* tree : trees) {
            if (tree) {
                mTreesToMerge.emplace_back(*tree, tag);
            }
        }
    }

    /// @brief Constructor to accept a vector of TreeToMerge objects, primarily
    /// used when mixing const/non-const trees.
    /// @note Sum order is preserved.
    explicit SumMergeOp(const std::vector<TreeToMerge<TreeT>>& trees)
        : mTreesToMerge(trees) { }

    /// @brief Constructor to accept a deque of TreeToMerge objects, primarily
    /// used when mixing const/non-const trees.
    /// @note Sum order is preserved.
    explicit SumMergeOp(const std::deque<TreeToMerge<TreeT>>& trees)
        : mTreesToMerge(trees.cbegin(), trees.cend()) { }

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
}; // struct SumMergeOp


////////////////////////////////////////


template<typename TreeT>
void TreeToMerge<TreeT>::initializeMask()
{
    if (mSteal)    return;
    mMaskTree.ptr.reset(new MaskTreeType);
    MaskUnionOp op(*mTree);
    tree::DynamicNodeManager<MaskTreeType, MaskTreeType::RootNodeType::LEVEL-1> manager(*this->mask());
    manager.foreachTopDown(op);
}

template<typename TreeT>
bool TreeToMerge<TreeT>::hasMask() const
{
    return bool(mMaskTree.ptr);
}

template<typename TreeT>
void TreeToMerge<TreeT>::reset(typename TreeType::Ptr treePtr, Steal)
{
    if (!treePtr) {
        OPENVDB_THROW(RuntimeError, "Cannot reset with empty Tree shared pointer.");
    }
    mSteal = true;
    mTreePtr = treePtr;
    mTree = mTreePtr.get();
}

template<typename TreeT>
const typename TreeToMerge<TreeT>::RootNodeType*
TreeToMerge<TreeT>::rootPtr() const
{
    return &mTree->root();
}

template<typename TreeT>
template<typename NodeT>
const NodeT*
TreeToMerge<TreeT>::probeConstNode(const Coord& ijk) const
{
    // test mutable mask first, node may have already been pruned
    if (!mSteal && !this->mask()->isValueOn(ijk))    return nullptr;
    return mTree->template probeConstNode<NodeT>(ijk);
}

template<typename TreeT>
void
TreeToMerge<TreeT>::pruneMask(Index level, const Coord& ijk)
{
    if (!mSteal) {
        OPENVDB_ASSERT(this->hasMask());
        this->mask()->addTile(level, ijk, false, false);
    }
}

template<typename TreeT>
template<typename NodeT>
std::unique_ptr<NodeT>
TreeToMerge<TreeT>::stealOrDeepCopyNode(const Coord& ijk, const ValueType& value)
{
    if (mSteal) {
        TreeType* tree = const_cast<TreeType*>(mTree);
        return std::unique_ptr<NodeT>(
            tree->root().template stealNode<NodeT>(ijk, value, false)
        );
    } else {
        auto* child = this->probeConstNode<NodeT>(ijk);
        if (child) {
            auto result = std::make_unique<NodeT>(*child);
            this->pruneMask(NodeT::LEVEL+1, ijk);
            return result;
        }
    }
    return std::unique_ptr<NodeT>();
}

template<typename TreeT>
template<typename NodeT>
std::unique_ptr<NodeT>
TreeToMerge<TreeT>::stealOrDeepCopyNode(const Coord& ijk)
{
    return this->stealOrDeepCopyNode<NodeT>(ijk, mTree->root().background());
}

template<typename TreeT>
template<typename NodeT>
void
TreeToMerge<TreeT>::addTile(const Coord& ijk, const ValueType& value, bool active)
{
    // ignore leaf node tiles (values)
    if (NodeT::LEVEL == 0)  return;

    if (mSteal) {
        TreeType* tree = const_cast<TreeType*>(mTree);
        auto* node = tree->template probeNode<NodeT>(ijk);
        if (node) {
            const Index pos = NodeT::coordToOffset(ijk);
            node->addTile(pos, value, active);
        }
    } else {
        auto* node = mTree->template probeConstNode<NodeT>(ijk);
        if (node)   this->pruneMask(NodeT::LEVEL, ijk);
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

/// @cond OPENVDB_DOCS_INTERNAL

namespace merge_internal {


template <typename BufferT, typename ValueT>
struct UnallocatedBuffer
{
    static void allocateAndFill(BufferT& buffer, const ValueT& background)
    {
        if (buffer.empty()) {
            if (!buffer.isOutOfCore()) {
                buffer.allocate();
                buffer.fill(background);
            }
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


// a convenience class that combines nested parallelism with the depth-visit
// node visitor which can result in increased performance with large sub-trees
template <Index LEVEL>
struct Dispatch
{
    template <typename NodeT, typename OpT>
    static void run(NodeT& node, OpT& op)
    {
        using NonConstChildT = typename NodeT::ChildNodeType;
        using ChildT = typename CopyConstness<NodeT, NonConstChildT>::Type;

        // use nested parallelism if there is more than one child

        Index32 childCount = node.childCount();
        if (childCount > 0) {
            op(node, size_t(0));

            // build linear list of child pointers
            std::vector<ChildT*> children;
            children.reserve(childCount);
            for (auto iter = node.beginChildOn(); iter; ++iter) {
                children.push_back(&(*iter));
            }

            // parallelize across children
            tbb::parallel_for(
                tbb::blocked_range<Index32>(0, childCount),
                [&](tbb::blocked_range<Index32>& range) {
                    for (Index32 n = range.begin(); n < range.end(); n++) {
                        DepthFirstNodeVisitor<ChildT>::visit(*children[n], op);
                    }
                }
            );
        } else {
            DepthFirstNodeVisitor<NodeT>::visit(node, op);
        }
    }
}; // struct Dispatch

// when LEVEL = 0, do not attempt nested parallelism
template <>
struct Dispatch<0>
{
    template <typename NodeT, typename OpT>
    static void run(NodeT& node, OpT& op)
    {
        DepthFirstNodeVisitor<NodeT>::visit(node, op);
    }
};


// an DynamicNodeManager operator to add a value and modify active state
// for every tile and voxel in a given subtree
template <typename TreeT>
struct ApplyTileSumToNodeOp
{
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    ApplyTileSumToNodeOp(const ValueT& value, const bool active):
        mValue(value), mActive(active) { }

    template <typename NodeT>
    void operator()(NodeT& node, size_t) const
    {
        // TODO: Need to add an InternalNode::setValue(Index offset, ...) to
        // avoid the cost of using a value iterator or coordToOffset() in the case
        // where node.isChildMaskOff() is true

        for (auto iter = node.beginValueAll(); iter; ++iter) {
            iter.setValue(mValue + *iter);
        }
        if (mActive)     node.setValuesOn();
    }

    void operator()(LeafT& leaf, size_t) const
    {
        auto* data = leaf.buffer().data();

        if (mValue != zeroVal<ValueT>()) {
            for (Index i = 0; i < LeafT::SIZE; ++i) {
                data[i] += mValue;
            }
        }
        if (mActive)    leaf.setValuesOn();
    }

    template <typename NodeT>
    void run(NodeT& node)
    {
        Dispatch<NodeT::LEVEL>::run(node, *this);
    }

private:
    ValueT mValue;
    bool mActive;
}; // struct ApplyTileSumToNodeOp



} // namespace merge_internal


/// @endcond

////////////////////////////////////////


template <typename TreeT, bool Union>
bool CsgUnionOrIntersectionOp<TreeT, Union>::operator()(RootT& root, size_t) const
{
    const bool Intersect = !Union;

    if (this->empty())     return false;

    // store the background value
    if (!mBackground)   mBackground = &root.background();

    // does the key exist in the root node?
    auto keyExistsInRoot = [&](const Coord& key) -> bool
    {
        return root.getValueDepth(key) > -1;
    };

    // does the key exist in all merge tree root nodes?
    auto keyExistsInAllTrees = [&](const Coord& key) -> bool
    {
        for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
            const auto* mergeRoot = mergeTree.rootPtr();
            if (!mergeRoot)                             return false;
            if (mergeRoot->getValueDepth(key) == -1)    return false;
        }
        return true;
    };

    // delete any background tiles
    root.eraseBackgroundTiles();

    // for intersection, delete any root node keys that are not present in all trees
    if (Intersect) {
        // find all tile coordinates to delete
        std::vector<Coord> toDelete;
        for (auto valueIter = root.cbeginValueAll(); valueIter; ++valueIter) {
            const Coord& key = valueIter.getCoord();
            if (!keyExistsInAllTrees(key))   toDelete.push_back(key);
        }
        // find all child coordinates to delete
        for (auto childIter = root.cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();
            if (!keyExistsInAllTrees(key))   toDelete.push_back(key);
        }
        // only mechanism to delete elements in root node is to delete background tiles,
        // so insert background tiles (which will replace any child nodes) and then delete
        for (Coord& key : toDelete)     root.addTile(key, *mBackground, false);
        root.eraseBackgroundTiles();
    }

    // find all tile values in this root and track inside/outside and active state
    // note that level sets should never contain active tiles, but we handle them anyway

    constexpr uint8_t ACTIVE_TILE = 0x1;
    constexpr uint8_t INSIDE_TILE = 0x2;
    constexpr uint8_t OUTSIDE_TILE = 0x4;

    constexpr uint8_t INSIDE_STATE = Union ? INSIDE_TILE : OUTSIDE_TILE;
    constexpr uint8_t OUTSIDE_STATE = Union ? OUTSIDE_TILE : INSIDE_TILE;

    const ValueT insideBackground = Union ? -this->background() : this->background();
    const ValueT outsideBackground = -insideBackground;

    auto getTileFlag = [&](auto& valueIter) -> uint8_t
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
            // for intersection, only add the tile if the key already exists in the tree
            if (Union || keyExistsInRoot(key)) {
                root.addTile(key, insideBackground, state);
            }
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

            // for intersection, only add child nodes if the key already exists in the tree
            if (Intersect && !keyExistsInRoot(key))        continue;

            // if child already exists, merge recursion will need to continue to resolve conflict
            if (children.count(key)) {
                continueRecurse = true;
                continue;
            }

            // if an inside tile exists, do nothing
            auto it = tiles.find(key);
            if (it != tiles.end() && it->second == INSIDE_STATE)     continue;

            auto childPtr = mergeTree.template stealOrDeepCopyNode<typename RootT::ChildNodeType>(key);
            childPtr->resetBackground(mergeRoot->background(), root.background());
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
                // for intersection, only add the tile if the key already exists in the tree
                if (Union || keyExistsInRoot(key)) {
                    root.addTile(key, outsideBackground, state);
                }
            }
        }
    }

    // finish by removing any background tiles
    root.eraseBackgroundTiles();

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
                if (childPtr) {
                    childPtr->resetBackground(mergeTree.rootPtr()->background(), this->background());
                    node.addChild(childPtr.release());
                }
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

        if (mPruneCancelledTiles) {
            bool allnegequal = true;
            for (Index i = 0 ; i < LeafT::SIZE; i++) {
                const ValueT& newValue = mergeLeaf->getValue(i);
                const ValueT& oldValue = leaf.getValue(i);
                allnegequal &= oldValue == math::negative(newValue);
                const bool doMerge = Union ? newValue < oldValue : newValue > oldValue;
                if (doMerge) {
                    leaf.setValueOnly(i, newValue);
                    leaf.setActiveState(i, mergeLeaf->isValueOn(i));
                }
            }
            if (allnegequal) {
                // If two diffed tiles have the same values of opposite signs,
                // we know they have both the same distances and gradients.
                // Thus they will cancel out.
                if (Union) { leaf.fill(math::negative(this->background()), false); }
                else { leaf.fill(this->background(), false); }
            }

        } else {
            for (Index i = 0 ; i < LeafT::SIZE; i++) {
                const ValueT& newValue = mergeLeaf->getValue(i);
                const ValueT& oldValue = leaf.getValue(i);
                const bool doMerge = Union ? newValue < oldValue : newValue > oldValue;
                if (doMerge) {
                    leaf.setValueOnly(i, newValue);
                    leaf.setActiveState(i, mergeLeaf->isValueOn(i));
                }
            }
        }
    }

    return false;
}

template <typename TreeT, bool Union>
const typename CsgUnionOrIntersectionOp<TreeT, Union>::ValueT&
CsgUnionOrIntersectionOp<TreeT, Union>::background() const
{
    // this operator is only intended to be used with foreachTopDown()
    OPENVDB_ASSERT(mBackground);
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

    auto getTileFlag = [&](auto& valueIter) -> uint8_t
    {
        uint8_t flag(0);
        const ValueT& value = valueIter.getValue();
        if (value < zeroVal<ValueT>())          flag |= INSIDE_TILE;
        if (valueIter.isValueOn())              flag |= ACTIVE_TILE;
        return flag;
    };

    // delete any background tiles
    root.eraseBackgroundTiles();

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

    // finish by removing any background tiles
    root.eraseBackgroundTiles();

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

    if (mPruneCancelledTiles) {
        bool allequal = true;
        for (Index i = 0 ; i < LeafT::SIZE; i++) {
            const ValueT& aValue = leaf.getValue(i);
            ValueT bValue = mergeLeaf->getValue(i);
            allequal &= aValue == bValue;
            bValue = math::negative(bValue);
            if (aValue < bValue) { // a = max(a, -b)
                leaf.setValueOnly(i, bValue);
                leaf.setActiveState(i, mergeLeaf->isValueOn(i));
            }
        }
        if (allequal) {
            // If two diffed tiles have the same values, we know they
            // have both the same distances and gradients.  Thus they will
            // cancel out.
            leaf.fill(background(), false);
        }
    } else {
        for (Index i = 0 ; i < LeafT::SIZE; i++) {
            const ValueT& aValue = leaf.getValue(i);
            ValueT bValue = mergeLeaf->getValue(i);
            bValue = math::negative(bValue);
            if (aValue < bValue) { // a = max(a, -b)
                leaf.setValueOnly(i, bValue);
                leaf.setActiveState(i, mergeLeaf->isValueOn(i));
            }
        }
    }

    return false;
}

template <typename TreeT>
const typename CsgDifferenceOp<TreeT>::ValueT&
CsgDifferenceOp<TreeT>::background() const
{
    // this operator is only intended to be used with foreachTopDown()
    OPENVDB_ASSERT(mBackground);
    return *mBackground;
}

template <typename TreeT>
const typename CsgDifferenceOp<TreeT>::ValueT&
CsgDifferenceOp<TreeT>::otherBackground() const
{
    // this operator is only intended to be used with foreachTopDown()
    OPENVDB_ASSERT(mOtherBackground);
    return *mOtherBackground;
}


////////////////////////////////////////


template <typename TreeT>
bool SumMergeOp<TreeT>::operator()(RootT& root, size_t) const
{
    using ValueT = typename RootT::ValueType;
    using ChildT = typename RootT::ChildNodeType;
    using NonConstChildT = typename std::remove_const<ChildT>::type;

    if (this->empty())     return false;

    // store the background value
    if (!mBackground)   mBackground = &root.background();

    // does the key exist in the root node?
    auto keyExistsInRoot = [](const auto& rootToTest, const Coord& key) -> bool
    {
        return rootToTest.getValueDepth(key) > -1;
    };

    constexpr uint8_t TILE = 0x1;
    constexpr uint8_t CHILD = 0x2;
    constexpr uint8_t TARGET_CHILD = 0x4; // child already exists in the target tree

    std::unordered_map<Coord, /*flags*/uint8_t> children;

    // find all tiles and child nodes in our root

    if (root.getTableSize() > 0) {
        for (auto valueIter = root.cbeginValueAll(); valueIter; ++valueIter) {
            const Coord& key = valueIter.getCoord();
            children.insert({key, TILE});
        }

        for (auto childIter = root.cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();
            children.insert({key, CHILD | TARGET_CHILD});
        }
    }

    // find all tiles and child nodes in other roots

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const auto* mergeRoot = mergeTree.rootPtr();
        if (!mergeRoot)     continue;

        for (auto valueIter = mergeRoot->cbeginValueAll(); valueIter; ++valueIter) {
            const Coord& key = valueIter.getCoord();
            auto it = children.find(key);
            if (it == children.end()) {
                // if no element with this key, insert it
                children.insert({key, TILE});
            } else {
                // mark as tile
                it->second |= TILE;
            }
        }

        for (auto childIter = mergeRoot->cbeginChildOn(); childIter; ++childIter) {
            const Coord& key = childIter.getCoord();
            auto it = children.find(key);
            if (it == children.end()) {
                // if no element with this key, insert it
                children.insert({key, CHILD});
            } else {
                // mark as child
                it->second |= CHILD;
            }
        }
    }

    // if any coords do not already exist in the root, insert an inactive background tile

    for (const auto& it : children) {
        if (!keyExistsInRoot(root, it.first)) {
            root.addTile(it.first, root.background(), false);
        }
    }

    // for each coord, merge each tile into the root until a child is found, then steal it

    for (const auto& it : children) {

        const Coord& key = it.first;

        // do nothing if the target root already contains a child node,
        // merge recursion will need to continue to resolve conflict
        if (it.second & TARGET_CHILD)    continue;

        ValueT value;
        const bool active = root.probeValue(key, value);

        for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
            const auto* mergeRoot = mergeTree.rootPtr();
            if (!mergeRoot)                            continue;

            // steal or deep-copy the first child node that is encountered,
            // then cease processing of this root node coord as merge recursion
            // will need to continue to resolve conflict

            const auto* mergeNode = mergeRoot->template probeConstNode<ChildT>(key);
            if (mergeNode) {
                auto childPtr = mergeTree.template stealOrDeepCopyNode<ChildT>(key);
                if (childPtr) {
                    // apply tile value and active state to the sub-tree
                    merge_internal::ApplyTileSumToNodeOp<TreeT> applyOp(value, active);
                    applyOp.run(*childPtr);
                    root.addChild(childPtr.release());
                }
                break;
            }

            ValueT mergeValue;
            const bool mergeActive = mergeRoot->probeValue(key, mergeValue);

            if (active || mergeActive) {
                value += mergeValue;
                root.addTile(key, value, true);
            } else {
                value += mergeValue;
                root.addTile(key, value, false);
            }

            // reset tile value to zero to prevent it being merged twice
            mergeTree.template addTile<NonConstChildT>(key, zeroVal<ValueT>(), false);
        }
    }

    // set root background to be the sum of all other root backgrounds

    ValueT background = root.background();

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const auto* mergeRoot = mergeTree.rootPtr();
        if (!mergeRoot)     continue;
        background += mergeRoot->background();
    }

    root.setBackground(background, /*updateChildNodes=*/false);

    return true;
}

template<typename TreeT>
template<typename NodeT>
bool SumMergeOp<TreeT>::operator()(NodeT& node, size_t) const
{
    using ChildT = typename NodeT::ChildNodeType;
    using NonConstNodeT = typename std::remove_const<NodeT>::type;

    if (this->empty())     return false;

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const auto* mergeRoot = mergeTree.rootPtr();
        if (!mergeRoot)     continue;

        const auto* mergeNode = mergeTree.template probeConstNode<NonConstNodeT>(node.origin());
        if (mergeNode) {
            // merge node

            for (auto iter = node.beginValueAll(); iter; ++iter) {
                if (mergeNode->isChildMaskOn(iter.pos())) {
                    // steal child node
                    auto childPtr = mergeTree.template stealOrDeepCopyNode<ChildT>(iter.getCoord());
                    if (childPtr) {
                        // apply tile value and active state to the sub-tree
                        merge_internal::ApplyTileSumToNodeOp<TreeT> applyOp(*iter, iter.isValueOn());
                        applyOp.run(*childPtr);
                        node.addChild(childPtr.release());
                    }
                } else {
                    ValueT mergeValue;
                    const bool mergeActive = mergeNode->probeValue(iter.getCoord(), mergeValue);
                    iter.setValue(*iter + mergeValue);
                    if (mergeActive && !iter.isValueOn())   iter.setValueOn();
                }
            }

        } else {
            // merge tile or background value

            if (mergeTree.hasMask()) {
                // if not stealing, test the original tree and if the node exists there, it means it
                // has been stolen so don't merge the tile value with this node
                const ChildT* originalMergeNode = mergeRoot->template probeConstNode<ChildT>(node.origin());
                if (originalMergeNode)  continue;
            }

            ValueT mergeValue;
            const bool mergeActive = mergeRoot->probeValue(node.origin(), mergeValue);
            for (auto iter = node.beginValueAll(); iter; ++iter) {
                iter.setValue(*iter + mergeValue);
                if (mergeActive && !iter.isValueOn())   iter.setValueOn();
            }
        }
    }

    return true;
}

template <typename TreeT>
bool SumMergeOp<TreeT>::operator()(LeafT& leaf, size_t) const
{
    using RootT = typename TreeT::RootNodeType;
    using RootChildT = typename RootT::ChildNodeType;
    using NonConstRootChildT = typename std::remove_const<RootChildT>::type;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename LeafT::ValueType;
    using BufferT = typename LeafT::Buffer;
    using NonConstLeafT = typename std::remove_const<LeafT>::type;

    if (this->empty())      return false;

    const Coord& ijk = leaf.origin();

    // if buffer is not out-of-core and empty, leaf node must have only been
    // partially constructed, so allocate and fill with background value

    merge_internal::UnallocatedBuffer<BufferT, ValueT>::allocateAndFill(
        leaf.buffer(), this->background());

    auto* data = leaf.buffer().data();

    for (TreeToMerge<TreeT>& mergeTree : mTreesToMerge) {
        const RootT* mergeRoot = mergeTree.rootPtr();
        if (!mergeRoot)     continue;

        const LeafT* mergeLeaf = mergeTree.template probeConstNode<NonConstLeafT>(ijk);
        if (mergeLeaf) {
            // merge leaf

            // if buffer is not out-of-core yet empty, leaf node must have only been
            // partially constructed, so skip merge

            if (merge_internal::UnallocatedBuffer<BufferT, ValueT>::isPartiallyConstructed(
                mergeLeaf->buffer())) {
                return false;
            }

            for (Index i = 0; i < LeafT::SIZE; ++i) {
                data[i] += mergeLeaf->getValue(i);
            }

            leaf.getValueMask() |= mergeLeaf->getValueMask();
        } else {
            // merge root tile or background value (as the merge leaf does not exist)

            if (mergeTree.hasMask()) {
                // if not stealing, test the original tree and if the leaf exists there, it means it
                // has been stolen so don't merge the tile value with this leaf
                const LeafT* originalMergeLeaf = mergeRoot->template probeConstNode<NonConstLeafT>(ijk);
                if (originalMergeLeaf)  continue;
            }

            const RootChildT* mergeRootChild = mergeRoot->template probeConstNode<NonConstRootChildT>(ijk);

            ValueT mergeValue;
            bool mergeActive = mergeRootChild ?
                mergeRootChild->probeValue(ijk, mergeValue) : mergeRoot->probeValue(ijk, mergeValue);

            if (mergeValue != zeroVal<ValueT>()) {
                for (Index i = 0; i < LeafT::SIZE; ++i) {
                    data[i] += mergeValue;
                }
            }

            if (mergeActive)    leaf.setValuesOn();
        }
    }

    return false;
}

template <typename TreeT>
const typename SumMergeOp<TreeT>::ValueT&
SumMergeOp<TreeT>::background() const
{
    // this operator is only intended to be used with foreachTopDown()
    OPENVDB_ASSERT(mBackground);
    return *mBackground;
}


////////////////////////////////////////

// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_MERGE
#include <openvdb/util/ExplicitInstantiation.h>
#endif

OPENVDB_INSTANTIATE_STRUCT TreeToMerge<MaskTree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<BoolTree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<FloatTree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<DoubleTree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<Int32Tree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<Int64Tree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<Vec3STree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<Vec3DTree>;
OPENVDB_INSTANTIATE_STRUCT TreeToMerge<Vec3ITree>;

OPENVDB_INSTANTIATE_STRUCT SumMergeOp<MaskTree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<BoolTree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<FloatTree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<DoubleTree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<Int32Tree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<Int64Tree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<Vec3STree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<Vec3DTree>;
OPENVDB_INSTANTIATE_STRUCT SumMergeOp<Vec3ITree>;

OPENVDB_INSTANTIATE_STRUCT CsgUnionOrIntersectionOp<FloatTree, true>;
OPENVDB_INSTANTIATE_STRUCT CsgUnionOrIntersectionOp<DoubleTree, true>;

OPENVDB_INSTANTIATE_STRUCT CsgUnionOrIntersectionOp<FloatTree, false>;
OPENVDB_INSTANTIATE_STRUCT CsgUnionOrIntersectionOp<DoubleTree, false>;

OPENVDB_INSTANTIATE_STRUCT CsgDifferenceOp<FloatTree>;
OPENVDB_INSTANTIATE_STRUCT CsgDifferenceOp<DoubleTree>;

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED
