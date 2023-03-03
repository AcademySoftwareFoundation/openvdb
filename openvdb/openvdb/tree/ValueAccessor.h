// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file tree/ValueAccessor.h
///
/// @brief  ValueAccessors are designed to help accelerate accesses into the
///   OpenVDB Tree structures by storing caches to Tree branches. When
///   traversing a grid in a spatially coherent pattern (e.g., iterating over
///   neighboring voxels), the same branches and nodes of the underlying tree
///   can be hit. If you do this using the Tree/RootNode methods directly,
///   traversal will occur at O(log(n)) (or O(n) depending on the hash map
///   implementation) for every access. However, using a ValueAccessor allows
///   for the Accessor to cache previously visited Nodes, providing possible
///   subsequent access speeds of O(1) if the next access is close to a
///   previously cached Node. Accessors are lightweight and can be configured
///   to cache any number of arbitrary Tree levels.
///
///   The ValueAccessor interfaces matches that of compatible OpenVDB Tree
///   nodes. You can request an Accessor from a Grid (with Grid::getAccessor())
///   or construct one directly from a Tree. You can use, for example, the
///   accessor's @c getValue() and @c setValue() methods in place of those on
///   OpenVDB Nodes/Trees.
///
/// @par Example:
/// @code
///   FloatGrid grid;
///   FloatGrid::Accessor acc = grid.getAccessor();
///   // First access is slow:
///   acc.setValue(Coord(0, 0, 0), 100);
///
///   // Subsequent nearby accesses are fast, since the accessor now holds pointers
///   // to nodes that contain (0, 0, 0) along the path from the root of the grid's
///   // tree to the leaf:
///   acc.setValue(Coord(0, 0, 1), 100);
///   acc.getValue(Coord(0, 2, 0), 100);
///
///   // Slow, because the accessor must be repopulated:
///   acc.getValue(Coord(-1, -1, -1));
///
///   // Fast:
///   acc.getValue(Coord(-1, -1, -2));
///   acc.setValue(Coord(-1, -2, 0), -100);
/// @endcode

#ifndef OPENVDB_TREE_VALUEACCESSOR_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_VALUEACCESSOR_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>

#include <tbb/spin_mutex.h>

#include <cassert>
#include <limits>
#include <type_traits>
#include <mutex>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

// Forward declaration of the generic ValueAccessor API.
template<typename TreeType,
    bool IsSafe = true,
    typename MutexT = void,
    typename IndexSequence = openvdb::make_index_sequence<std::max(size_t(1),TreeType::DEPTH)-1>>
class ValueAccessorImpl;

/// @brief  Default alias for a ValueAccessor. This is simply a helper alias
///   for the generic definition but takes a single Index specifying the number
///   of nodes to cache. This is expanded into an index sequence (required for
///   backward compatibility).
/// @tparam TreeType  The tree type
/// @tparam IsSafe    Whether this accessor registers itself to the tree. See
///   the base class definition for more information on this parameter.
/// @tparam CacheLevels  The number of node levels to cache _excluding_ the
///   Root node. The Root node is implicitly always included, even if this
///   value is zero.
/// @tparam MutexType  An optional std compatible mutex to use which ensures
///   every call to the ValueAccessor API is thread safe. If void (the default)
///   no locking takes place. In general it's not advised to mutex lock
///   ValueAccessor methods (instead consider creating a accessor per thread).
template<typename TreeType, bool IsSafe = true,
    size_t CacheLevels = std::max(Index(1),TreeType::DEPTH)-1, typename MutexType = void>
using ValueAccessor =
    ValueAccessorImpl<TreeType, IsSafe, MutexType,
        openvdb::make_index_sequence<CacheLevels>>;

/// @brief  Helper alias for a ValueAccessor which doesn't cache any Internal
///   or Leaf nodes.
template <typename TreeType, bool IsSafe>
using ValueAccessor0 =
    ValueAccessorImpl<TreeType, IsSafe, void, openvdb::index_sequence<>>;
/// @brief  Helper alias for a ValueAccessor which caches a single node level.
///   By default, the node level is 0, which corresponds to the lowest node
///   level, typically LeafNodes.
template <typename TreeType, bool IsSafe, size_t L0 = 0>
using ValueAccessor1 =
    ValueAccessorImpl<TreeType, IsSafe, void, openvdb::index_sequence<L0>>;
/// @brief  Helper alias for a ValueAccessor which caches two node levels. By
///   default the two lowest node levels are selected (0, 1) which typically
///   correspond to an InternalNode and its child LeafNodes. This instantiation
///   will only be valid for TreeTypes which have at least two levels of nodes
///   (excluding the Root node).
template <typename TreeType, bool IsSafe, size_t L0 = 0, size_t L1 = 1>
using ValueAccessor2 =
    ValueAccessorImpl<TreeType, IsSafe, void, openvdb::index_sequence<L0, L1>>;
/// @brief  Helper alias for a ValueAccessor which caches three node levels. By
///   default the three lowest node levels are selected (0, 1, 2) which
///   typically correspond to two InternalNodes followed by the bottom
///   LeafNodes. This instantiation will only be valid for TreeTypes which have
///   at least three levels of nodes (excluding the Root node).
template <typename TreeType, bool IsSafe, size_t L0 = 0, size_t L1 = 1, size_t L2 = 2>
using ValueAccessor3 =
    ValueAccessorImpl<TreeType, IsSafe, void, openvdb::index_sequence<L0, L1, L2>>;

/// @brief Helper alias for a ValueAccesor which spin locks every API call.
template<typename TreeType, bool IsSafe = true,
    size_t CacheLevels = std::max(Index(1),TreeType::DEPTH)-1>
using ValueAccessorRW =
    ValueAccessorImpl<TreeType, IsSafe, tbb::spin_mutex,
        openvdb::make_index_sequence<CacheLevels>>;


/// @brief This base class for ValueAccessors manages registration of an
///   accessor with a tree so that the tree can automatically clear the
///   accessor whenever one of its nodes is deleted.
///
/// @internal A base class is needed because ValueAccessor is templated on both
///   a Tree type and a mutex type. The various instantiations of the template
///   are distinct, unrelated types, so they can't easily be stored in a
///   container (mainly the Tree's CacheRegistry). This base class, in contrast,
///   is templated only on the Tree type, so for any given Tree, only two
///   distinct instantiations are possible, ValueAccessorBase<Tree> and
///   ValueAccessorBase<const Tree>.
///
/// @warning If IsSafe = false then the ValueAccessor will not register itself
///   with the tree from which it is constructed. While in some rare cases this
///   can lead to better performance (since it avoids the small overhead of
///   insertion on creation and deletion on destruction) it is also unsafe if
///   the tree is modified. So unless you're an expert it is highly recommended
///   to set IsSafe = true, which is the default in all derived ValueAccessors
///   defined below. However if you know that the tree is no being modifed for
///   the lifespan of the ValueAccessor AND the work performed per
///   ValueAccessor is small relative to overhead of registering it you should
///   consider setting IsSafe = false. If this turns out to improve performance
///   you should really rewrite your code so as to better amortize the
///   construction of the ValueAccessor, i.e. reuse it as much as possible!
template<typename TreeType, bool IsSafe>
class ValueAccessorBase
{
public:
    /// @brief  Returns true if this accessor is operating on a const tree type.
    static constexpr bool IsConstTree = std::is_const<TreeType>::value;

    /// @brief Return true if this accessor is safe, i.e. registered by the
    ///   tree from which it is constructed. Un-registered accessors can in
    ///   rare cases be faster because it avoids the (small) overhead of
    ///   registration, but they are unsafe if the tree is modified. So unless
    ///   you're an expert it is highly recommended to set IsSafe = true (which
    ///   is the default).
    static constexpr bool isSafe() { return IsSafe; }

    /// @brief  Construct from a tree. Should rarely be invoked directly, the
    ///   drived implementation class calls this. Remains public for backwards
    ///   compatibility.
    ValueAccessorBase(TreeType& tree)
        : mTree(&tree)
    {
        if (IsSafe) tree.attachAccessor(*this);
    }

    virtual ~ValueAccessorBase() { if (IsSafe && mTree) mTree->releaseAccessor(*this); }

    /// @brief  Copy constructor - if IsSafe, then the copy also registers
    ///   itself against the tree it is accessing.
    ValueAccessorBase(const ValueAccessorBase& other)
        : mTree(other.mTree)
    {
        if (IsSafe && mTree) mTree->attachAccessor(*this);
    }

    ValueAccessorBase& operator=(const ValueAccessorBase& other)
    {
        if (&other != this) {
            if (IsSafe && mTree) mTree->releaseAccessor(*this);
            mTree = other.mTree;
            if (IsSafe && mTree) mTree->attachAccessor(*this);
        }
        return *this;
    }

    /// @brief Return a pointer to the tree associated with this accessor.
    /// @details The pointer will be null only if the tree from which this
    ///   accessor was constructed was subsequently deleted (which generally
    ///   leaves the accessor in an unsafe state).
    TreeType* getTree() const { return mTree; }

    /// @brief Return a reference to the tree associated with this accessor.
    TreeType& tree() const { assert(mTree); return *mTree; }

    /// @brief  Pure virtual method, clears the derived accessor
    virtual void clear() = 0;

protected:
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;
    virtual void release() { mTree = nullptr; }
    TreeType* mTree;
}; // class ValueAccessorBase

///////////////////////////////////////////////////////////////////////////////

/// @cond OPENVDB_DOCS_INTERNAL

namespace value_accessor_internal
{

template<typename ListT, size_t... Ts> struct NodeListBuilderImpl;

template <typename NodeChainT>
struct NodeListBuilderImpl<NodeChainT>
{
    using ListT = TypeList<>;
};

template <typename NodeChainT, size_t Idx>
struct NodeListBuilderImpl<NodeChainT, Idx>
{
    using NodeT = typename NodeChainT::template Get<Idx>;
    using ListT = TypeList<NodeT>;
};

template <typename NodeChainT, size_t ThisIdx, size_t NextIdx, size_t... Idxs>
struct NodeListBuilderImpl<NodeChainT, ThisIdx, NextIdx, Idxs...>
{
    static_assert(ThisIdx < NextIdx,
        "Invalid cache level - Cache levels must be in increasing ascending order");
    static_assert(ThisIdx < NodeChainT::Size,
        "Invalid cache level - Cache level is larger than the number of tree nodes");
    static_assert(ThisIdx < NodeChainT::Back::LEVEL,
        "Invalid cache level - Cache level is larger than the number of tree nodes");

    using NodeT = typename NodeChainT::template Get<ThisIdx>;
    using ListT = typename TypeList<NodeT>::template Append<
        typename NodeListBuilderImpl<NodeChainT, NextIdx, Idxs...>::ListT>;
};

template<typename NodeChainT, size_t RootLevel, typename IntegerSequence>
struct NodeListBuilder;

template<typename NodeChainT, size_t RootLevel, size_t... Is>
struct NodeListBuilder<NodeChainT, RootLevel, std::integer_sequence<size_t, Is...>>
{
    using ListT = typename NodeListBuilderImpl<NodeChainT, Is..., RootLevel>::ListT;
};

template<typename NodeChainT, size_t RootLevel, size_t... Is>
struct NodeListBuilder<NodeChainT, RootLevel, openvdb::index_sequence<Is...>>
{
    using ListT = typename NodeListBuilderImpl<NodeChainT, Is..., RootLevel>::ListT;
};


template<typename TreeTypeT, typename NodeT>
struct EnableLeafBuffer
{
    using LeafNodeT = typename TreeTypeT::LeafNodeType;
    static constexpr bool value =
        std::is_same<NodeT, LeafNodeT>::value &&
        std::is_same<typename LeafNodeT::Buffer::StorageType,
            typename LeafNodeT::ValueType>::value;
};

template<typename TreeTypeT, size_t... Is>
struct EnableLeafBuffer<TreeTypeT, openvdb::index_sequence<Is...>>
{
    // Empty integer seq, no nodes being caches
    static constexpr bool value = false;
};

template<typename TreeTypeT, size_t First, size_t... Is>
struct EnableLeafBuffer<TreeTypeT, openvdb::index_sequence<First, Is...>>
{
private:
    using NodeChainT = typename TreeTypeT::RootNodeType::NodeChainType;
    using FirstNodeT = typename NodeChainT::template Get<First>;
public:
    static constexpr bool value = EnableLeafBuffer<TreeTypeT, FirstNodeT>::value;
};

} // namespace value_accessor_internal

/// @endcond

///////////////////////////////////////////////////////////////////////////////

/// The following classes exist to perform empty base class optimizations
/// with the final ValueAccessor implementation. Depending on the template
/// types provided to the derived implementation, some member variables may not
/// be necessary (mutex, leaf buffer cache, etc). These classes allow for these
/// variables to be compiled out. Note that from C++20 we can switch to
/// [[no_unique_address]] member annotations instead.

/// @brief  A small class that contains a Mutex which is derived from by the
///   internal Value Accessor Implementation. This allows for the empty base
///   class optimization to be performed in the case where a Mutex/Lock is not
///   in use. From C++20 we can instead switch to [[no_unique_address]].
template <typename MutexT>
struct ValueAccessorLock
{
    inline auto lock() const { return std::scoped_lock(m); }
private:
    mutable MutexT m;
};

/// @brief  Specialization for the case where no Mutex is in use. See above.
template <>
struct ValueAccessorLock<void>
{
    inline constexpr auto lock() const { return 0; }
};

/// @brief  A small class that contains a cached pointer to a LeafNode data
///   buffer which is derived from by the internal Value Accessor
///   Implementation. This allows for the empty base class optimization to be
///   performed in the case where a LeafNode does not store a contiguous
///   index-able buffer. From C++20 we can instead switch to
///   [[no_unique_address]].
template<typename TreeTypeT, typename IntegerSequence, typename Enable = void>
struct ValueAccessorLeafBuffer
{
    template <typename NodeT>
    static constexpr bool BypassLeafAPI =
        std::is_same<NodeT, typename TreeTypeT::LeafNodeType>::value;
    inline const typename TreeTypeT::ValueType* buffer() { assert(mBuffer); return mBuffer; }
    inline const typename TreeTypeT::ValueType* buffer() const { assert(mBuffer); return mBuffer; }
    inline void setBuffer(const typename TreeTypeT::ValueType* b) const { mBuffer = b; }
private:
    mutable const typename TreeTypeT::ValueType* mBuffer;
};

/// @brief  Specialization for the case where a Leaf Buffer cannot be cached.
//    These methods should never be invoked. See above.
template<typename TreeTypeT, typename IntegerSequence>
struct ValueAccessorLeafBuffer<TreeTypeT, IntegerSequence,
    typename std::enable_if<
        !value_accessor_internal::EnableLeafBuffer<TreeTypeT, IntegerSequence>::value
    >::type>
{
    template <typename> static constexpr bool BypassLeafAPI = false;
    inline constexpr typename TreeTypeT::ValueType* buffer() { assert(false); return nullptr; }
    inline constexpr typename TreeTypeT::ValueType* buffer() const { assert(false); return nullptr; }
    inline constexpr void setBuffer(const typename TreeTypeT::ValueType*) const { assert(false); }
};

///////////////////////////////////////////////////////////////////////////////

/// @brief  The Value Accessor Implementation and API methods. The majoirty of
///   the API matches the API of a compatible OpenVDB Tree Node.
template<typename _TreeType, bool IsSafe, typename MutexT, typename IntegerSequence>
class ValueAccessorImpl final :
    public ValueAccessorBase<_TreeType, IsSafe>,
    public ValueAccessorLeafBuffer<_TreeType, IntegerSequence>,
    public ValueAccessorLock<MutexT>
{
public:
    /// @note  Not strictly the only Base Type but provided for backwards
    ///   compatibility.
    using BaseT = ValueAccessorBase<_TreeType, IsSafe>;
    using LockT = ValueAccessorLock<MutexT>;
    using LeafCacheT = ValueAccessorLeafBuffer<_TreeType, IntegerSequence>;

    using TreeType = _TreeType;
    using ValueType = typename TreeType::ValueType;
    using RootNodeT = typename TreeType::RootNodeType;
    using LeafNodeT = typename TreeType::LeafNodeType;
    using NodeChainT = typename RootNodeT::NodeChainType;

    /// @brief  A resolved, flattened TypeList of node types which this
    ///   accessor is caching. The nodes index in this list does not
    ///   necessarily correspond to the nodes level in the tree.
    using NodeLevelList =
        typename value_accessor_internal::NodeListBuilder
            <NodeChainT, RootNodeT::LEVEL, IntegerSequence>::ListT;
    using NodePtrList = typename NodeLevelList::template Transform<std::add_pointer_t>;

    /// @brief  Return a node type at a particular cache level in the Value
    ///   accessor. The node type at a given cache level does not necessarily
    ///   equal the same node type in the TreeType as this depends entirely on
    ///   which tree levels this Accessor is caching. For example:
    /// @par Example:
    /// @code
    ///      // Cache tree levels 0 and 2
    ///      using Impl = ValueAccessorImpl<FloatTree, true, void, 0, 2>
    ///      using CacheLevel1 = Impl::template NodeTypeAtLevel<1>
    ///      using TreeLevel2 = TreeType::RootNodeType::NodeChainType::Get<2>;
    ///      static_assert(std::is_same<CacheLevel1, TreeLevel2>::value);
    /// @endcode
    template <size_t Level>
    using NodeTypeAtLevel = typename NodeLevelList::template Get<Level>;

    /// @brief  Given a node type, return whether this Accessor can perform
    ///   optimized value buffer accesses. This is only possible for LeafNodes
    ///   so will always return false for any non LeafNode type. It also
    ///   depends on the value type - if the value buffer is a contiguous
    ///   index-able array of values then this returns true.
    template <typename NodeT>
    static constexpr bool IsLeafAndBypassLeafAPI =
        LeafCacheT::template BypassLeafAPI<NodeT>;

    /// @brief Helper alias which is true if the lowest cached node level is
    ///   a LeafNode type and has a compatible value type for optimized access.
    static constexpr bool BypassLeafAPI =
        IsLeafAndBypassLeafAPI<NodeTypeAtLevel<0>>;

    /// @brief The number of node levels that this accessor can cache,
    ///   excluding the RootNode.
    static constexpr size_t NumCacheLevels = NodeLevelList::Size-1;
    static_assert(TreeType::DEPTH >= NodeLevelList::Size-1, "cache size exceeds tree depth");
    static_assert(NodeLevelList::Size > 0, "unexpected cache size");

    /// @brief Constructor from a tree
    ValueAccessorImpl(TreeType& tree)
        : BaseT(tree)
        , LeafCacheT()
        , LockT()
        , mKeys()
        , mNodes() {
            this->clear();
        }

    ~ValueAccessorImpl() override final = default;
    ValueAccessorImpl(const ValueAccessorImpl&) = default;
    ValueAccessorImpl& operator=(const ValueAccessorImpl&) = default;

    /// @brief Return @c true if any of the nodes along the path to the given
    ///   coordinate have been cached.
    /// @param xyz  The index space coordinate to query
    bool isCached(const Coord& xyz) const
    {
        return this->evalFirstIndex([&](const auto Idx) -> bool
            {
                using NodeType = typename NodeLevelList::template Get<Idx>;
                // @warning  Putting this exp in the if statement crashes GCC9
                constexpr bool IsRoot = std::is_same<RootNodeT, NodeType>::value;
                if constexpr(IsRoot) return false;
                else return (this->isHashed<NodeType>(xyz));
            });
    }

    /// @brief Return the value of the voxel at the given coordinates.
    /// @param xyz  The index space coordinate to query
    const ValueType& getValue(const Coord& xyz) const
    {
        // Don't use evalFirstCached as we don't access the node when
        // IsLeafAndBypassLeafAPI<NodeType> is true.
        return *this->evalFirstIndex([&](const auto Idx) -> const ValueType*
            {
                using NodeType = typename NodeLevelList::template Get<Idx>;
                // If not cached return a nullptr. Note that this operator is
                // guaranteed to return a value as the last node in the chain
                // is a RootNode and isHashed always returns true for this case
                if (!this->isHashed<NodeType>(xyz)) return nullptr;

                if constexpr(IsLeafAndBypassLeafAPI<NodeType>) {
                    return &(LeafCacheT::buffer()[LeafNodeT::coordToOffset(xyz)]);
                }
                else {
                    auto node = mNodes.template get<Idx>();
                    assert(node);
                    return &(node->getValueAndCache(xyz, *this));
                }
            });
    }

    /// @brief Return the active state of the voxel at the given coordinates.
    /// @param xyz  The index space coordinate to query
    bool isValueOn(const Coord& xyz) const
    {
        return this->evalFirstCached(xyz, [&](const auto node) -> bool {
                assert(node);
                return node->isValueOnAndCache(xyz, *this);
            });
    }

    /// @brief Return the active state of the value at a given coordinate as
    ///   well as its value.
    /// @param xyz    The index space coordinate to query
    /// @param value  The value to get
    bool probeValue(const Coord& xyz, ValueType& value) const
    {
        return this->evalFirstCached(xyz, [&](const auto node) -> bool
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);

                if constexpr(IsLeafAndBypassLeafAPI<NodeType>) {
                    const auto offset = LeafNodeT::coordToOffset(xyz);
                    value = LeafCacheT::buffer()[offset];
                    return node->isValueOn(offset);
                }
                else {
                    return node->probeValueAndCache(xyz, value, *this);
                }
            });
    }

    /// @brief Return the tree depth (0 = root) at which the value of voxel
    ///   (x, y, z) resides, or -1 if (x, y, z) isn't explicitly represented in
    ///   the tree (i.e., if it is implicitly a background voxel).
    /// @note  This is the inverse of the node LEVEL (where the RootNode level
    ///   is the highest in the tree).
    /// @param xyz  The index space coordinate to query
    int getValueDepth(const Coord& xyz) const
    {
        return this->evalFirstCached(xyz, [&](const auto node) -> int
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);

                if constexpr(std::is_same<RootNodeT, NodeType>::value) {
                    return node->getValueDepthAndCache(xyz, *this);
                }
                else {
                    return int(RootNodeT::LEVEL - node->getValueLevelAndCache(xyz, *this));
                }
            });
    }

    /// @brief Return @c true if the value of voxel (x, y, z) resides at the
    ///   leaf level of the tree, i.e., if it is not a tile value.
    /// @param xyz  The index space coordinate to query
    bool isVoxel(const Coord& xyz) const
    {
        assert(BaseT::mTree);
        return this->getValueDepth(xyz) ==
            static_cast<int>(RootNodeT::LEVEL);
    }

    //@{
    /// @brief Set a particular value at the given coordinate and mark the
    ///   coordinate as active
    /// @note  This method will densify branches of the tree if the coordinate
    ///   points to a tile and if the provided value or active state is
    ///   different to the tiles
    /// @param xyz    The index space coordinate to set
    /// @param value  The value to set
    void setValue(const Coord& xyz, const ValueType& value)
    {
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        this->evalFirstCached(xyz, [&](const auto node) -> void
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);

                if constexpr(IsLeafAndBypassLeafAPI<NodeType>) {
                    const auto offset = LeafNodeT::coordToOffset(xyz);
                    const_cast<ValueType&>(LeafCacheT::buffer()[offset]) = value;
                    const_cast<NodeType*>(node)->setValueOn(offset);
                }
                else {
                    const_cast<NodeType*>(node)->setValueAndCache(xyz, value, *this);
                }
            });
    }

    void setValueOn(const Coord& xyz, const ValueType& value) { this->setValue(xyz, value); }
    //@}

    /// @brief Set a particular value at the given coordinate but preserve its
    ///   active state
    /// @note  This method will densify branches of the tree if the coordinate
    ///   points to a tile and if the provided value is different to the tiles
    /// @param xyz    The index space coordinate to set
    /// @param value  The value to set
    void setValueOnly(const Coord& xyz, const ValueType& value)
    {
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        // Don't use evalFirstCached as we don't access the node when
        // IsLeafAndBypassLeafAPI<NodeType> is true.
        this->evalFirstIndex([&](const auto Idx) -> bool
            {
                using NodeType = typename NodeLevelList::template Get<Idx>;
                if (!this->isHashed<NodeType>(xyz)) return false;

                if constexpr(IsLeafAndBypassLeafAPI<NodeType>) {
                    const_cast<ValueType&>(LeafCacheT::buffer()[LeafNodeT::coordToOffset(xyz)]) = value;
                }
                else {
                    auto node = mNodes.template get<Idx>();
                    assert(node);
                    const_cast<NodeType*>(node)->setValueOnlyAndCache(xyz, value, *this);
                }
                return true;
            });
    }

    /// @brief Set a particular value at the given coordinate and mark the
    ///   coordinate as inactive.
    /// @note  This method will densify branches of the tree if the coordinate
    ///   points to a tile and if the provided value or active state is
    ///   different to the tiles
    /// @param xyz    The index space coordinate to set
    /// @param value  The value to set
    void setValueOff(const Coord& xyz, const ValueType& value)
    {
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        this->evalFirstCached(xyz, [&](const auto node) -> void
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);

                if constexpr(IsLeafAndBypassLeafAPI<NodeType>) {
                    const auto offset = LeafNodeT::coordToOffset(xyz);
                    const_cast<ValueType&>(LeafCacheT::buffer()[offset]) = value;
                    const_cast<NodeType*>(node)->setValueOff(offset);
                }
                else {
                    const_cast<NodeType*>(node)->setValueOffAndCache(xyz, value, *this);
                }
            });
    }

    /// @brief Apply a functor to the value at the given coordinate and mark
    ///     mark the coordinate as active
    /// @details See Tree::modifyValue() for details.
    /// @param xyz  The index space coordinate to modify
    /// @param op   The modify operation
    template<typename ModifyOp>
    void modifyValue(const Coord& xyz, const ModifyOp& op)
    {
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        this->evalFirstCached(xyz, [&](const auto node) -> void
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);

                if constexpr(IsLeafAndBypassLeafAPI<NodeType>) {
                    const auto offset = LeafNodeT::coordToOffset(xyz);
                    op(const_cast<ValueType&>(LeafCacheT::buffer()[offset]));
                    const_cast<NodeType*>(node)->setActiveState(offset, true);
                }
                else {
                    const_cast<NodeType*>(node)->modifyValueAndCache(xyz, op, *this);
                }
            });
    }

    /// @brief Apply a functor to the voxel at the given coordinates.
    /// @details See Tree::modifyValueAndActiveState() for details.
    /// @param xyz  The index space coordinate to modify
    /// @param op   The modify operation
    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord& xyz, const ModifyOp& op)
    {
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        this->evalFirstCached(xyz, [&](const auto node) -> void
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);

                if constexpr(IsLeafAndBypassLeafAPI<NodeType>) {
                    const auto offset = LeafNodeT::coordToOffset(xyz);
                    bool state = node->isValueOn(offset);
                    op(const_cast<ValueType&>(LeafCacheT::buffer()[offset]), state);
                    const_cast<NodeType*>(node)->setActiveState(offset, state);
                }
                else {
                    const_cast<NodeType*>(node)->modifyValueAndActiveStateAndCache(xyz, op, *this);
                }
            });
    }

    /// @brief Set the active state of the voxel at the given coordinates
    ///   without changing its value.
    /// @note  This method will densify branches of the tree if the coordinate
    ///   points to a tile and if the provided activate state flag is different
    ///   to the tiles
    /// @param xyz  The index space coordinate to modify
    /// @param on   Whether to set the active state to on (true) or off (false)
    void setActiveState(const Coord& xyz, bool on = true)
    {
        static_assert(!BaseT::IsConstTree, "can't modify a const tree's values");
        this->evalFirstCached(xyz, [&](const auto node) -> void
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);
                const_cast<NodeType*>(node)->setActiveStateAndCache(xyz, on, *this);
            });
    }
    /// @brief Mark the voxel at the given coordinates as active without
    ///   changing its value.
    /// @note  This method will densify branches of the tree if the coordinate
    ///   points to a tile and if the tiles active state is off.
    /// @param xyz  The index space coordinate to modify
    void setValueOn(const Coord& xyz) { this->setActiveState(xyz, true); }
    /// @brief Mark the voxel at the given coordinates as inactive without
    ///   changing its value.
    /// @note  This method will densify branches of the tree if the coordinate
    ///   points to a tile and if the tiles active state is on.
    /// @param xyz  The index space coordinate to modify
    void setValueOff(const Coord& xyz) { this->setActiveState(xyz, false); }

    /// @brief Returns the leaf node that contains voxel (x, y, z) and if it
    ///   doesn't exist, create it, but preserve the values and active states
    ///   of the pre-existing branch.
    /// @note You can use this method to preallocate a static tree topology
    ///   over which to safely perform multithreaded processing.
    /// @param xyz  The index space coordinate at which to create a LeafNode.
    ///   Note that if this coordinate is not a LeafNode origin then the
    ///   LeafNode that would otherwise contain this coordinate is created and
    ///   returned.
    LeafNodeT* touchLeaf(const Coord& xyz)
    {
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        return this->evalFirstCached(xyz, [&](const auto node) -> LeafNodeT*
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);
                return const_cast<NodeType*>(node)->touchLeafAndCache(xyz, *this);
            });
    }

    /// @brief Add the specified leaf to this tree, possibly creating a child
    ///   branch in the process.  If the leaf node already exists, replace it.
    /// @param leaf  The LeafNode to insert into the tree. Must not be a nullptr.
    void addLeaf(LeafNodeT* leaf)
    {
        constexpr int64_t Start = NodeLevelList::template Index<LeafNodeT> + 1;
        static_assert(!BaseT::IsConstTree, "can't add a node to a const tree");
        static_assert(Start >= 0);
        assert(leaf);
        this->evalFirstCached<Start>(leaf->origin(), [&](const auto node) -> void
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);
                const_cast<NodeType*>(node)->addLeafAndCache(leaf, *this);
            });
    }

    /// @brief Add a tile at the specified tree level that contains the
    ///   coordinate xyz, possibly deleting existing nodes or creating new
    ///   nodes in the process.
    /// @note Calling this with a level of 0 will modify voxel values. This
    ///   function will always densify a tree branch up to the requested level
    ///   (regardless if the value and active state match).
    /// @param level  The level of the tree to add a tile. Level 0 refers to
    ///   voxels (and is similar to ::setValue, except will always density).
    /// @param xyz    The index space coordinate to add a tile
    /// @param value  The value of the tile
    /// @param state  The active state to set on the new tile
    void addTile(Index level, const Coord& xyz, const ValueType& value, bool state)
    {
        constexpr int64_t Start = NodeLevelList::template Index<LeafNodeT> + 1;
        static_assert(!BaseT::IsConstTree, "can't add a tile to a const tree");
        static_assert(Start >= 0);
        this->evalFirstCached<Start>(xyz, [&](const auto node) -> void
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);
                const_cast<NodeType*>(node)->addTileAndCache(level, xyz, value, state, *this);
            });
    }

    ///@{
    /// @brief Return a pointer to the node of the specified type that contains
    ///   the value located at xyz. If no node of the given NodeT exists which
    ///   contains the value, a nullptr is returned.
    /// @brief This function may return a nullptr even if the coordinate xyz is
    ///   represented in tree, as it depends on the type NodeT provided. For
    ///   example, the value may exist as a tile in an InternalNode but note as
    ///   a LeafNode.
    /// @param xyz  The index space coordinate to query
    template<typename NodeT>
    NodeT* probeNode(const Coord& xyz)
    {
        static_assert(!BaseT::IsConstTree, "can't get a non-const node from a const tree");
        return this->evalFirstPred([&](const auto Idx) -> bool
            {
                using NodeType = typename NodeLevelList::template Get<Idx>;
                // @warning  Putting this exp in the if statement crashes GCC9
                constexpr bool NodeMayBeCached =
                    std::is_same<NodeT, NodeType>::value || NodeT::LEVEL < NodeType::LEVEL;

                if constexpr(NodeMayBeCached) return this->isHashed<NodeType>(xyz);
                else                          return false;
            },
            [&](const auto node) -> NodeT*
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);
                if constexpr(std::is_same<NodeT, NodeType>::value) {
                    return const_cast<NodeT*>(node);
                }
                else {
                    assert(NodeT::LEVEL < NodeType::LEVEL);
                    return const_cast<NodeType*>(node)->template probeNodeAndCache<NodeT>(xyz, *this);
                }
            });
    }

    template<typename NodeT>
    const NodeT* probeConstNode(const Coord& xyz) const
    {
        return this->evalFirstPred([&](const auto Idx) -> bool
            {
                using NodeType = typename NodeLevelList::template Get<Idx>;
                // @warning  Putting this exp in the if statement crashes GCC9
                constexpr bool NodeMayBeCached =
                    std::is_same<NodeT, NodeType>::value || NodeT::LEVEL < NodeType::LEVEL;

                if constexpr(NodeMayBeCached) return this->isHashed<NodeType>(xyz);
                else                          return false;
            },
            [&](const auto node) -> const NodeT*
            {
                using NodeType = std::remove_pointer_t<decltype(node)>;
                assert(node);
                if constexpr(std::is_same<NodeT, NodeType>::value) {
                    return node;
                }
                else {
                    assert(NodeT::LEVEL < NodeType::LEVEL);
                    return const_cast<NodeType*>(node)->template probeConstNodeAndCache<NodeT>(xyz, *this);
                }
            });
    }
    /// @}

    ///@{
    /// @brief Return a pointer to the leaf node that contains the voxel
    ///   coordinate xyz. If no LeafNode exists, returns a nullptr.
    /// @param xyz  The index space coordinate to query
    LeafNodeT* probeLeaf(const Coord& xyz) { return this->template probeNode<LeafNodeT>(xyz); }
    const LeafNodeT* probeLeaf(const Coord& xyz) const { return this->probeConstLeaf(xyz); }
    const LeafNodeT* probeConstLeaf(const Coord& xyz) const
    {
        return this->template probeConstNode<LeafNodeT>(xyz);
    }
    /// @}

    /// @brief Return the node of type @a NodeT that has been cached on this
    ///   accessor. If this accessor does not cache this NodeT, or if no
    ///   node of this type has been cached, returns a nullptr.
    template<typename NodeT>
    NodeT* getNode()
    {
        using NodeType = typename std::decay<NodeT>::type;
        static constexpr int64_t Idx = NodeLevelList::template Index<NodeType>;
        if constexpr (Idx >= 0) return mNodes.template get<Idx>();
        else return nullptr;
    }

    /// @brief  Explicitly insert a node of the type @a NodeT into this Value
    ///   Accessors cache.
    /// @todo deprecate?
    template<typename NodeT>
    void insertNode(const Coord& xyz, NodeT& node)
    {
        this->insert(xyz, &node);
    }

    /// @brief  Explicitly remove this Value Accessors cached node of the given
    ///   NodeT. If this Value Accessor does not support the caching of the
    ///   provided NodeT, this method does nothing.
    template<typename NodeT>
    void eraseNode()
    {
        static constexpr int64_t Idx = NodeLevelList::template Index<NodeT>;
        if constexpr (Idx >= 0) {
            mKeys[Idx] = Coord::max();
            mNodes.template get<Idx>() = nullptr;
        }
    }

    /// @brief  Remove all the cached nodes and invalidate the corresponding
    ///   hash-keys.
    void clear() override final
    {
        mKeys.fill(Coord::max());
        mNodes.foreach([](auto& node) { node = nullptr; });
        if constexpr (BypassLeafAPI) {
            LeafCacheT::setBuffer(nullptr);
        }
        if (BaseT::mTree) {
            static constexpr int64_t Idx = NodeLevelList::template Index<RootNodeT>;
            mNodes.template get<Idx>() = const_cast<RootNodeT*>(&(BaseT::mTree->root()));
        }
    }

public:
    // Backwards compatible support. Use NodeTypeAtLevel<> instead
    using NodeT0 OPENVDB_DEPRECATED_MESSAGE("Use NodeTypeAtLevel<0>") =
        typename std::conditional<(NumCacheLevels > 0), NodeTypeAtLevel<0>, void>::type;
    using NodeT1 OPENVDB_DEPRECATED_MESSAGE("Use NodeTypeAtLevel<1>") =
        typename std::conditional<(NumCacheLevels > 1), NodeTypeAtLevel<1>, void>::type;
    using NodeT2 OPENVDB_DEPRECATED_MESSAGE("Use NodeTypeAtLevel<2>") =
        typename std::conditional<(NumCacheLevels > 2), NodeTypeAtLevel<2>, void>::type;
    /// @brief Return the number of cache levels employed by this ValueAccessor
    OPENVDB_DEPRECATED_MESSAGE("Use the static NumCacheLevels constant")
    static constexpr Index numCacheLevels() { return NumCacheLevels; }

protected:
    // Allow nodes to insert themselves into the cache.
    template<typename> friend class RootNode;
    template<typename, Index> friend class InternalNode;
    template<typename, Index> friend class LeafNode;
    // Allow trees to deregister themselves.
    template<typename> friend class Tree;

    /// @brief  Release this accessor from the tree, set the tree to null and
    ///   clear the accessor cache. After calling this method the accessor
    ///   will be completely invalid.
    void release() override final
    {
        this->BaseT::release();
        this->clear();
    }

    /// ******************************* WARNING *******************************
    ///  Methods here must be force inline otherwise compilers do not optimize
    ///  out the function call due to recursive templates and performance
    ///  degradation is significant.
    /// ***********************************************************************

    /// @brief  Insert a node into this ValueAccessor's cache
    template<typename NodeT>
    OPENVDB_FORCE_INLINE void insert(
        [[maybe_unused]] const Coord& xyz,
        [[maybe_unused]] const NodeT* node) const
    {
        // Early exit if NodeT isn't part of this ValueAccessors cache
        if constexpr(!NodeLevelList::template Contains<NodeT>) return;
        else {
            constexpr uint64_t Idx = uint64_t(NodeLevelList::template Index<NodeT>);
            static_assert(NodeLevelList::template Contains<NodeT>);
            static_assert(Idx < NumCacheLevels);
            mKeys[Idx] = xyz & ~(NodeT::DIM-1);
            mNodes.template get<Idx>() = const_cast<NodeT*>(node);
            if constexpr(IsLeafAndBypassLeafAPI<NodeT>) {
                LeafCacheT::setBuffer(node->buffer().data());
            }
        }
    }

    template<typename NodeT>
    OPENVDB_FORCE_INLINE bool isHashed([[maybe_unused]] const Coord& xyz) const
    {
        if constexpr(!NodeLevelList::template Contains<NodeT>) return false;
        if constexpr(std::is_same<NodeT, RootNodeT>::value) {
            return true;
        }
        else {
            constexpr uint64_t Idx = uint64_t(NodeLevelList::template Index<NodeT>);
            static_assert(NodeLevelList::template Contains<NodeT>);
            static_assert(Idx < NumCacheLevels + 1);
            return (xyz[0] & ~Coord::ValueType(NodeT::DIM-1)) == mKeys[Idx][0]
                && (xyz[1] & ~Coord::ValueType(NodeT::DIM-1)) == mKeys[Idx][1]
                && (xyz[2] & ~Coord::ValueType(NodeT::DIM-1)) == mKeys[Idx][2];
        }
    }

private:
    /// @brief  Evaluate a function on each node until its returns value is not
    ///   null or false.
    /// @param op  The function to run
    template <typename OpT>
    OPENVDB_FORCE_INLINE auto evalFirstIndex(OpT&& op) const
    {
        assert(BaseT::mTree);
        // Mutex lock the accessor. Does nothing if no mutex if in place
        [[maybe_unused]] const auto lock = this->lock();
        // Get the return type of the provided operation OpT
        using IndexT = std::integral_constant<std::size_t, 0>;
        using RetT = typename std::invoke_result<OpT, IndexT>::type;
        return openvdb::evalFirstIndex<0, NumCacheLevels+1>(op, RetT(NULL));
    }

    /// @brief  Evaluate a predicate on each index I from [0,Size] until it
    ///   returns true, then executes the provided op function on the resolved
    ///   node type. Helps in cases where std::get may be unecessarily invoked.
    /// @param pred The predicate to run on the node index
    /// @param op   The function to run on the node where the pred returns true
    template <typename PredT, typename OpT>
    OPENVDB_FORCE_INLINE auto evalFirstPred(PredT&& pred, OpT&& op) const
    {
        assert(BaseT::mTree);
        // Mutex lock the accessor. Does nothing if no mutex if in place
        [[maybe_unused]] const auto lock = this->lock();
        using RetT = typename std::invoke_result<OpT, RootNodeT*>::type;
        if constexpr(!std::is_same<RetT, void>::value) {
            return mNodes.evalFirstPred(pred, op, RetT(false));
        }
        else {
            return mNodes.evalFirstPred(pred, op);
        }
    }

    /// @brief  Helper alias to call this->evalFirstPred(), but with a default
    ///   predicate set to return true when the node at the given index is
    ///   cached
    /// @param xyz The coord to hash
    /// @param op  The function to run on the node where the pred returns true
    template <size_t Start = 0, typename OpT = void>
    OPENVDB_FORCE_INLINE auto evalFirstCached([[maybe_unused]] const Coord& xyz, OpT&& op) const
    {
        return this->evalFirstPred([&](const auto Idx) -> bool
            {
                if constexpr(Idx < Start)             return false;
                if constexpr(Idx > NumCacheLevels+1)  return false;
                using NodeType = typename NodeLevelList::template Get<Idx>;
                return this->isHashed<NodeType>(xyz);
            }, op);
    }

private:
    mutable std::array<Coord, NumCacheLevels> mKeys;
    mutable typename NodePtrList::AsTupleList mNodes;
}; // ValueAccessorImpl

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_VALUEACCESSOR_HAS_BEEN_INCLUDED
