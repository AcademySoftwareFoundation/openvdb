// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file tree/NodeManager.h
///
/// @authors Ken Museth, Dan Bailey
///
/// @brief NodeManager produces linear arrays of all tree nodes
/// allowing for efficient threading and bottom-up processing.
///
/// @note A NodeManager can be constructed from a Tree or LeafManager.

#ifndef OPENVDB_TREE_NODEMANAGER_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_NODEMANAGER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <deque>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

// Produce linear arrays of all tree nodes, to facilitate efficient threading
// and bottom-up processing.
template<typename TreeOrLeafManagerT, Index LEVELS = TreeOrLeafManagerT::RootNodeType::LEVEL>
class NodeManager;


// Produce linear arrays of all tree nodes lazily, to facilitate efficient threading
// of topology-changing top-down workflows.
template<typename TreeOrLeafManagerT, Index _LEVELS = TreeOrLeafManagerT::RootNodeType::LEVEL>
class DynamicNodeManager;


////////////////////////////////////////


// This is a dummy node filtering class used by the NodeManager class to match
// the internal filtering interface used by the DynamicNodeManager.
struct NodeFilter
{
    static bool valid(size_t) { return true; }
}; // struct NodeFilter


/// @brief This class caches tree nodes of a specific type in a linear array.
///
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT>
class NodeList
{
public:
    NodeList() = default;

    NodeT& operator()(size_t n) const { assert(n<mNodeCount); return *(mNodes[n]); }

    NodeT*& operator[](size_t n) { assert(n<mNodeCount); return mNodes[n]; }

    Index64 nodeCount() const { return mNodeCount; }

    void clear()
    {
        mNodePtrs.reset();
        mNodes = nullptr;
        mNodeCount = 0;
    }

    // initialize this node list from the provided root node
    template <typename RootT>
    bool initRootChildren(RootT& root)
    {
        // Allocate (or deallocate) the node pointer array

        size_t nodeCount = root.childCount();

        if (nodeCount != mNodeCount) {
            if (nodeCount > 0) {
                mNodePtrs.reset(new NodeT*[nodeCount]);
                mNodes = mNodePtrs.get();
            } else {
                mNodePtrs.reset();
                mNodes = nullptr;
            }
            mNodeCount = nodeCount;
        }

        if (mNodeCount == 0)    return false;

        // Populate the node pointers

        NodeT** nodePtr = mNodes;
        for (auto iter = root.beginChildOn(); iter; ++iter) {
            *nodePtr++ = &iter.getValue();
        }

        return true;
    }

    // initialize this node list from another node list containing the parent nodes
    template <typename ParentsT, typename NodeFilterT>
    bool initNodeChildren(ParentsT& parents, const NodeFilterT& nodeFilter = NodeFilterT(), bool serial = false)
    {
        // Compute the node counts for each node

        std::vector<Index32> nodeCounts;
        if (serial) {
            nodeCounts.reserve(parents.nodeCount());
            for (size_t i = 0; i < parents.nodeCount(); i++) {
                if (!nodeFilter.valid(i))   nodeCounts.push_back(0);
                else                        nodeCounts.push_back(parents(i).childCount());
            }
        } else {
            nodeCounts.resize(parents.nodeCount());
            tbb::parallel_for(
                // with typical node sizes and SSE enabled, there are only a handful
                // of instructions executed per-operation with a default grainsize
                // of 1, so increase to 64 to reduce parallel scheduling overhead
                tbb::blocked_range<Index64>(0, parents.nodeCount(), /*grainsize=*/64),
                [&](tbb::blocked_range<Index64>& range)
                {
                    for (Index64 i = range.begin(); i < range.end(); i++) {
                        if (!nodeFilter.valid(i))   nodeCounts[i] = 0;
                        else                        nodeCounts[i] = parents(i).childCount();
                    }
                }
            );
        }

        // Turn node counts into a cumulative histogram and obtain total node count

        for (size_t i = 1; i < nodeCounts.size(); i++) {
            nodeCounts[i] += nodeCounts[i-1];
        }

        const size_t nodeCount = nodeCounts.empty() ? 0 : nodeCounts.back();

        // Allocate (or deallocate) the node pointer array

        if (nodeCount != mNodeCount) {
            if (nodeCount > 0) {
                mNodePtrs.reset(new NodeT*[nodeCount]);
                mNodes = mNodePtrs.get();
            } else {
                mNodePtrs.reset();
                mNodes = nullptr;
            }
            mNodeCount = nodeCount;
        }

        if (mNodeCount == 0)    return false;

        // Populate the node pointers

        if (serial) {
            NodeT** nodePtr = mNodes;
            for (size_t i = 0; i < parents.nodeCount(); i++) {
                if (!nodeFilter.valid(i))   continue;
                for (auto iter = parents(i).beginChildOn(); iter; ++iter) {
                    *nodePtr++ = &iter.getValue();
                }
            }
        } else {
            tbb::parallel_for(
                tbb::blocked_range<Index64>(0, parents.nodeCount()),
                [&](tbb::blocked_range<Index64>& range)
                {
                    Index64 i = range.begin();
                    NodeT** nodePtr = mNodes;
                    if (i > 0)  nodePtr += nodeCounts[i-1];
                    for ( ; i < range.end(); i++) {
                        if (!nodeFilter.valid(i))   continue;
                        for (auto iter = parents(i).beginChildOn(); iter; ++iter) {
                            *nodePtr++ = &iter.getValue();
                        }
                    }
                }
            );
        }

        return true;
    }

    class NodeRange
    {
    public:

        NodeRange(size_t begin, size_t end, const NodeList& nodeList, size_t grainSize=1):
            mEnd(end), mBegin(begin), mGrainSize(grainSize), mNodeList(nodeList) {}

        NodeRange(NodeRange& r, tbb::split):
            mEnd(r.mEnd), mBegin(doSplit(r)), mGrainSize(r.mGrainSize),
            mNodeList(r.mNodeList) {}

        size_t size() const { return mEnd - mBegin; }

        size_t grainsize() const { return mGrainSize; }

        const NodeList& nodeList() const { return mNodeList; }

        bool empty() const {return !(mBegin < mEnd);}

        bool is_divisible() const {return mGrainSize < this->size();}

        class Iterator
        {
        public:
            Iterator(const NodeRange& range, size_t pos): mRange(range), mPos(pos)
            {
                assert(this->isValid());
            }
            Iterator(const Iterator&) = default;
            Iterator& operator=(const Iterator&) = default;
            /// Advance to the next node.
            Iterator& operator++() { ++mPos; return *this; }
            /// Return a reference to the node to which this iterator is pointing.
            NodeT& operator*() const { return mRange.mNodeList(mPos); }
            /// Return a pointer to the node to which this iterator is pointing.
            NodeT* operator->() const { return &(this->operator*()); }
            /// Return the index into the list of the current node.
            size_t pos() const { return mPos; }
            bool isValid() const { return mPos>=mRange.mBegin && mPos<=mRange.mEnd; }
            /// Return @c true if this iterator is not yet exhausted.
            bool test() const { return mPos < mRange.mEnd; }
            /// Return @c true if this iterator is not yet exhausted.
            operator bool() const { return this->test(); }
            /// Return @c true if this iterator is exhausted.
            bool empty() const { return !this->test(); }
            bool operator!=(const Iterator& other) const
            {
                return (mPos != other.mPos) || (&mRange != &other.mRange);
            }
            bool operator==(const Iterator& other) const { return !(*this != other); }
            const NodeRange& nodeRange() const { return mRange; }

        private:
            const NodeRange& mRange;
            size_t mPos;
        };// NodeList::NodeRange::Iterator

        Iterator begin() const {return Iterator(*this, mBegin);}

        Iterator end() const {return Iterator(*this, mEnd);}

    private:
        size_t mEnd, mBegin, mGrainSize;
        const NodeList& mNodeList;

        static size_t doSplit(NodeRange& r)
        {
            assert(r.is_divisible());
            size_t middle = r.mBegin + (r.mEnd - r.mBegin) / 2u;
            r.mEnd = middle;
            return middle;
        }
    };// NodeList::NodeRange

    /// Return a TBB-compatible NodeRange.
    NodeRange nodeRange(size_t grainsize = 1) const
    {
        return NodeRange(0, this->nodeCount(), *this, grainsize);
    }

    template<typename NodeOp>
    void foreach(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        NodeTransformerCopy<NodeOp> transform(op); // always deep-copies the op
        transform.run(this->nodeRange(grainSize), threaded);
    }

    template<typename NodeOp>
    void reduce(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        NodeReducer<NodeOp> transform(op);
        transform.run(this->nodeRange(grainSize), threaded);
    }

    // identical to foreach except the operator() method has a node index and
    // the operator is referenced instead of copied in NodeTransformer
    template<typename NodeOp>
    void foreachWithIndex(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        NodeTransformer<NodeOp, OpWithIndex> transform(op);
        transform.run(this->nodeRange(grainSize), threaded);
    }

    // identical to reduce except the operator() method has a node index
    template<typename NodeOp>
    void reduceWithIndex(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        NodeReducer<NodeOp, OpWithIndex> transform(op);
        transform.run(this->nodeRange(grainSize), threaded);
    }

private:

    // default execution in the NodeManager ignores the node index
    // given by the iterator position
    struct OpWithoutIndex
    {
        template <typename T>
        static void eval(T& node, typename NodeRange::Iterator& iter) { node(*iter); }
    };

    // execution in the DynamicNodeManager matches that of the LeafManager in
    // passing through the node index given by the iterator position
    struct OpWithIndex
    {
        template <typename T>
        static void eval(T& node, typename NodeRange::Iterator& iter) { node(*iter, iter.pos()); }
    };

    // Private struct of NodeList that performs parallel_for
    template<typename NodeOp, typename OpT = OpWithoutIndex>
    struct NodeTransformerCopy
    {
        NodeTransformerCopy(const NodeOp& nodeOp) : mNodeOp(nodeOp)
        {
        }
        void run(const NodeRange& range, bool threaded = true)
        {
            threaded ? tbb::parallel_for(range, *this) : (*this)(range);
        }
        void operator()(const NodeRange& range) const
        {
            for (typename NodeRange::Iterator it = range.begin(); it; ++it) {
                OpT::template eval(mNodeOp, it);
            }
        }
        const NodeOp mNodeOp;
    };// NodeList::NodeTransformerCopy

    // Private struct of NodeList that performs parallel_for
    template<typename NodeOp, typename OpT = OpWithoutIndex>
    struct NodeTransformer
    {
        NodeTransformer(const NodeOp& nodeOp) : mNodeOp(nodeOp)
        {
        }
        void run(const NodeRange& range, bool threaded = true)
        {
            threaded ? tbb::parallel_for(range, *this) : (*this)(range);
        }
        void operator()(const NodeRange& range) const
        {
            for (typename NodeRange::Iterator it = range.begin(); it; ++it) {
                OpT::template eval(mNodeOp, it);
            }
        }
        const NodeOp& mNodeOp;
    };// NodeList::NodeTransformer

    // Private struct of NodeList that performs parallel_reduce
    template<typename NodeOp, typename OpT = OpWithoutIndex>
    struct NodeReducer
    {
        NodeReducer(NodeOp& nodeOp) : mNodeOp(&nodeOp)
        {
        }
        NodeReducer(const NodeReducer& other, tbb::split)
            : mNodeOpPtr(std::make_unique<NodeOp>(*(other.mNodeOp), tbb::split()))
            , mNodeOp(mNodeOpPtr.get())
        {
        }
        void run(const NodeRange& range, bool threaded = true)
        {
            threaded ? tbb::parallel_reduce(range, *this) : (*this)(range);
        }
        void operator()(const NodeRange& range)
        {
            for (typename NodeRange::Iterator it = range.begin(); it; ++it) {
                OpT::template eval(*mNodeOp, it);
            }
        }
        void join(const NodeReducer& other)
        {
            mNodeOp->join(*(other.mNodeOp));
        }
        std::unique_ptr<NodeOp> mNodeOpPtr;
        NodeOp *mNodeOp = nullptr;
    };// NodeList::NodeReducer


protected:
    size_t mNodeCount = 0;
    std::unique_ptr<NodeT*[]> mNodePtrs;
    NodeT** mNodes = nullptr;
};// NodeList


/////////////////////////////////////////////


/// @brief This class is a link in a chain that each caches tree nodes
/// of a specific type in a linear array.
///
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT, Index LEVEL>
class NodeManagerLink
{
public:
    using NonConstChildNodeType = typename NodeT::ChildNodeType;
    using ChildNodeType = typename CopyConstness<NodeT, NonConstChildNodeType>::Type;

    NodeManagerLink() = default;

    void clear() { mList.clear(); mNext.clear(); }

    template <typename RootT>
    void initRootChildren(RootT& root, bool serial = false)
    {
        mList.initRootChildren(root);
        mNext.initNodeChildren(mList, serial);
    }

    template<typename ParentsT>
    void initNodeChildren(ParentsT& parents, bool serial = false)
    {
        mList.initNodeChildren(parents, NodeFilter(), serial);
        mNext.initNodeChildren(mList, serial);
    }

    Index64 nodeCount() const { return mList.nodeCount() + mNext.nodeCount(); }

    Index64 nodeCount(Index i) const
    {
        return i==NodeT::LEVEL ? mList.nodeCount() : mNext.nodeCount(i);
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mNext.foreachBottomUp(op, threaded, grainSize);
        mList.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.foreach(op, threaded, grainSize);
        mNext.foreachTopDown(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded, size_t grainSize)
    {
        mNext.reduceBottomUp(op, threaded, grainSize);
        mList.reduce(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.reduce(op, threaded, grainSize);
        mNext.reduceTopDown(op, threaded, grainSize);
    }

protected:
    NodeList<NodeT> mList;
    NodeManagerLink<ChildNodeType, LEVEL-1> mNext;
};// NodeManagerLink class


////////////////////////////////////////


/// @private
/// @brief Specialization that terminates the chain of cached tree nodes
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT>
class NodeManagerLink<NodeT, 0>
{
public:
    NodeManagerLink() = default;

    /// @brief Clear all the cached tree nodes
    void clear() { mList.clear(); }

    template <typename RootT>
    void initRootChildren(RootT& root, bool /*serial*/ = false) { mList.initRootChildren(root); }

    template<typename ParentsT>
    void initNodeChildren(ParentsT& parents, bool serial = false) { mList.initNodeChildren(parents, NodeFilter(), serial); }

    Index64 nodeCount() const { return mList.nodeCount(); }

    Index64 nodeCount(Index) const { return mList.nodeCount(); }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.reduce(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded, size_t grainSize)
    {
        mList.reduce(op, threaded, grainSize);
    }

protected:
    NodeList<NodeT> mList;
};// NodeManagerLink class


////////////////////////////////////////


/// @brief To facilitate threading over the nodes of a tree, cache
/// node pointers in linear arrays, one for each level of the tree.
///
/// @details This implementation works with trees of any depth, but
/// optimized specializations are provided for the most typical tree depths.
template<typename TreeOrLeafManagerT, Index _LEVELS>
class NodeManager
{
public:
    static const Index LEVELS = _LEVELS;
    static_assert(LEVELS > 0,
        "expected instantiation of template specialization"); // see specialization below
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    using NonConstChildNodeType = typename RootNodeType::ChildNodeType;
    using ChildNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstChildNodeType>::Type;
    static_assert(RootNodeType::LEVEL >= LEVELS, "number of levels exceeds root node height");

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false)
        : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    NodeManager(const NodeManager&) = delete;

    /// @brief Clear all the cached tree nodes
    void clear() { mChain.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool serial = false) { mChain.initRootChildren(mRoot, serial); }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mChain.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const { return mChain.nodeCount(i); }

    //@{
    /// @brief   Threaded method that applies a user-supplied functor
    ///          to all the nodes in the tree.
    ///
    /// @param op        user-supplied functor, see examples for interface details.
    /// @param threaded  optional toggle to disable threading, on by default.
    /// @param grainSize optional parameter to specify the grainsize
    ///                  for threading, one by default.
    ///
    /// @warning The functor object is deep-copied to create TBB tasks.
    ///
    /// @par Example:
    /// @code
    /// // Functor to offset all the inactive values of a tree. Note
    /// // this implementation also illustrates how different
    /// // computation can be applied to the different node types.
    /// template<typename TreeType>
    /// struct OffsetOp
    /// {
    ///     using ValueT = typename TreeT::ValueType;
    ///     using RootT = typename TreeT::RootNodeType;
    ///     using LeafT = typename TreeT::LeafNodeType;
    ///     OffsetOp(const ValueT& v) : mOffset(v) {}
    ///
    ///     // Processes the root node. Required by the NodeManager
    ///     void operator()(RootT& root) const
    ///     {
    ///         for (typename RootT::ValueOffIter i = root.beginValueOff(); i; ++i) *i += mOffset;
    ///     }
    ///     // Processes the leaf nodes. Required by the NodeManager
    ///     void operator()(LeafT& leaf) const
    ///     {
    ///         for (typename LeafT::ValueOffIter i = leaf.beginValueOff(); i; ++i) *i += mOffset;
    ///     }
    ///     // Processes the internal nodes. Required by the NodeManager
    ///     template<typename NodeT>
    ///     void operator()(NodeT& node) const
    ///     {
    ///         for (typename NodeT::ValueOffIter i = node.beginValueOff(); i; ++i) *i += mOffset;
    ///     }
    /// private:
    ///     const ValueT mOffset;
    /// };
    ///
    /// // usage:
    /// OffsetOp<FloatTree> op(3.0f);
    /// tree::NodeManager<FloatTree> nodes(tree);
    /// nodes.foreachBottomUp(op);
    ///
    /// // or if a LeafManager already exists
    /// using T = tree::LeafManager<FloatTree>;
    /// OffsetOp<T> op(3.0f);
    /// tree::NodeManager<T> nodes(leafManager);
    /// nodes.foreachBottomUp(op);
    ///
    /// @endcode
    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mChain.foreachBottomUp(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mChain.foreachTopDown(op, threaded, grainSize);
    }

    //@}

    //@{
    /// @brief   Threaded method that processes nodes with a user supplied functor
    ///
    /// @param op        user-supplied functor, see examples for interface details.
    /// @param threaded  optional toggle to disable threading, on by default.
    /// @param grainSize optional parameter to specify the grainsize
    ///                  for threading, one by default.
    ///
    /// @warning The functor object is deep-copied to create TBB tasks.
    ///
    /// @par Example:
    /// @code
    ///  // Functor to count nodes in a tree
    ///  template<typename TreeType>
    ///  struct NodeCountOp
    ///  {
    ///      NodeCountOp() : nodeCount(TreeType::DEPTH, 0), totalCount(0)
    ///      {
    ///      }
    ///      NodeCountOp(const NodeCountOp& other, tbb::split) :
    ///          nodeCount(TreeType::DEPTH, 0), totalCount(0)
    ///      {
    ///      }
    ///      void join(const NodeCountOp& other)
    ///      {
    ///          for (size_t i = 0; i < nodeCount.size(); ++i) {
    ///              nodeCount[i] += other.nodeCount[i];
    ///          }
    ///          totalCount += other.totalCount;
    ///      }
    ///      // do nothing for the root node
    ///      void operator()(const typename TreeT::RootNodeType& node)
    ///      {
    ///      }
    ///      // count the internal and leaf nodes
    ///      template<typename NodeT>
    ///      void operator()(const NodeT& node)
    ///      {
    ///          ++(nodeCount[NodeT::LEVEL]);
    ///          ++totalCount;
    ///      }
    ///      std::vector<openvdb::Index64> nodeCount;
    ///      openvdb::Index64 totalCount;
    /// };
    ///
    /// // usage:
    /// NodeCountOp<FloatTree> op;
    /// tree::NodeManager<FloatTree> nodes(tree);
    /// nodes.reduceBottomUp(op);
    ///
    /// // or if a LeafManager already exists
    /// NodeCountOp<FloatTree> op;
    /// using T = tree::LeafManager<FloatTree>;
    /// T leafManager(tree);
    /// tree::NodeManager<T> nodes(leafManager);
    /// nodes.reduceBottomUp(op);
    ///
    /// @endcode
    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mChain.reduceBottomUp(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mChain.reduceTopDown(op, threaded, grainSize);
    }
    //@}

protected:
    RootNodeType& mRoot;
    NodeManagerLink<ChildNodeType, LEVELS-1> mChain;
};// NodeManager class


////////////////////////////////////////////


// Wraps a user-supplied DynamicNodeManager operator and stores the return
// value of the operator() method to the index of the node in a bool array
template <typename OpT>
struct ForeachFilterOp
{
    explicit ForeachFilterOp(const OpT& op, openvdb::Index64 size)
        : mOp(op)
        , mValidPtr(std::make_unique<bool[]>(size))
        , mValid(mValidPtr.get()) { }

    ForeachFilterOp(const ForeachFilterOp& other)
        : mOp(other.mOp)
        , mValid(other.mValid) { }

    template<typename NodeT>
    void operator()(NodeT& node, size_t idx) const
    {
        mValid[idx] = mOp(node, idx);
    }

    bool valid(size_t idx) const { return mValid[idx]; }

    const OpT& op() const { return mOp; }

private:
    const OpT& mOp;
    std::unique_ptr<bool[]> mValidPtr;
    bool* mValid = nullptr;
}; // struct ForeachFilterOp


// Wraps a user-supplied DynamicNodeManager operator and stores the return
// value of the operator() method to the index of the node in a bool array
template <typename OpT>
struct ReduceFilterOp
{
    ReduceFilterOp(OpT& op, openvdb::Index64 size)
        : mOp(&op)
        , mValidPtr(std::make_unique<bool[]>(size))
        , mValid(mValidPtr.get()) { }

    ReduceFilterOp(const ReduceFilterOp& other)
        : mOp(other.mOp)
        , mValid(other.mValid) { }

    ReduceFilterOp(const ReduceFilterOp& other, tbb::split)
        : mOpPtr(std::make_unique<OpT>(*(other.mOp), tbb::split()))
        , mOp(mOpPtr.get())
        , mValid(other.mValid) { }

    template<typename NodeT>
    void operator()(NodeT& node, size_t idx) const
    {
        mValid[idx] = (*mOp)(node, idx);
    }

    void join(const ReduceFilterOp& other)
    {
        mOp->join(*(other.mOp));
    }

    bool valid(size_t idx) const
    {
        return mValid[idx];
    }

    OpT& op() { return *mOp; }

private:
    std::unique_ptr<OpT> mOpPtr;
    OpT* mOp = nullptr;
    std::unique_ptr<bool[]> mValidPtr;
    bool* mValid = nullptr;
}; // struct ReduceFilterOp


/// @brief This class is a link in a chain that each caches tree nodes
/// of a specific type in a linear array.
///
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT, Index LEVEL>
class DynamicNodeManagerLink
{
public:
    using NonConstChildNodeType = typename NodeT::ChildNodeType;
    using ChildNodeType = typename CopyConstness<NodeT, NonConstChildNodeType>::Type;

    DynamicNodeManagerLink() = default;

    template<typename NodeOpT, typename RootT>
    void foreachTopDown(const NodeOpT& op, RootT& root, bool threaded,
        size_t leafGrainSize, size_t nonLeafGrainSize)
    {
        if (!op(root, /*index=*/0))         return;
        if (!mList.initRootChildren(root))  return;
        ForeachFilterOp<NodeOpT> filterOp(op, mList.nodeCount());
        mList.foreachWithIndex(filterOp, threaded, LEVEL == 0 ? leafGrainSize : nonLeafGrainSize);
        mNext.foreachTopDownRecurse(filterOp, mList, threaded, leafGrainSize, nonLeafGrainSize);
    }

    template<typename FilterOpT, typename ParentT>
    void foreachTopDownRecurse(const FilterOpT& filterOp, ParentT& parent, bool threaded,
        size_t leafGrainSize, size_t nonLeafGrainSize)
    {
        if (!mList.initNodeChildren(parent, filterOp, !threaded))   return;
        FilterOpT childFilterOp(filterOp.op(), mList.nodeCount());
        mList.foreachWithIndex(childFilterOp, threaded, LEVEL == 0 ? leafGrainSize : nonLeafGrainSize);
        mNext.foreachTopDownRecurse(childFilterOp, mList, threaded, leafGrainSize, nonLeafGrainSize);
    }

    template<typename NodeOpT, typename RootT>
    void reduceTopDown(NodeOpT& op, RootT& root, bool threaded,
        size_t leafGrainSize, size_t nonLeafGrainSize)
    {
        if (!op(root, /*index=*/0))         return;
        if (!mList.initRootChildren(root))  return;
        ReduceFilterOp<NodeOpT> filterOp(op, mList.nodeCount());
        mList.reduceWithIndex(filterOp, threaded, LEVEL == 0 ? leafGrainSize : nonLeafGrainSize);
        mNext.reduceTopDownRecurse(filterOp, mList, threaded, leafGrainSize, nonLeafGrainSize);
    }

    template<typename FilterOpT, typename ParentT>
    void reduceTopDownRecurse(FilterOpT& filterOp, ParentT& parent, bool threaded,
        size_t leafGrainSize, size_t nonLeafGrainSize)
    {
        if (!mList.initNodeChildren(parent, filterOp, !threaded))   return;
        FilterOpT childFilterOp(filterOp.op(), mList.nodeCount());
        mList.reduceWithIndex(childFilterOp, threaded, LEVEL == 0 ? leafGrainSize : nonLeafGrainSize);
        mNext.reduceTopDownRecurse(childFilterOp, mList, threaded, leafGrainSize, nonLeafGrainSize);
    }

protected:
    NodeList<NodeT> mList;
    DynamicNodeManagerLink<ChildNodeType, LEVEL-1> mNext;
};// DynamicNodeManagerLink class


/// @private
/// @brief Specialization that terminates the chain of cached tree nodes
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT>
class DynamicNodeManagerLink<NodeT, 0>
{
public:
    DynamicNodeManagerLink() = default;

    template<typename NodeFilterOp, typename ParentT>
    void foreachTopDownRecurse(const NodeFilterOp& nodeFilterOp, ParentT& parent, bool threaded,
        size_t leafGrainSize, size_t /*nonLeafGrainSize*/)
    {
        if (!mList.initNodeChildren(parent, nodeFilterOp, !threaded))   return;
        mList.foreachWithIndex(nodeFilterOp.op(), threaded, leafGrainSize);
    }

    template<typename NodeFilterOp, typename ParentT>
    void reduceTopDownRecurse(NodeFilterOp& nodeFilterOp, ParentT& parent, bool threaded,
        size_t leafGrainSize, size_t /*nonLeafGrainSize*/)
    {
        if (!mList.initNodeChildren(parent, nodeFilterOp, !threaded))   return;
        mList.reduceWithIndex(nodeFilterOp.op(), threaded, leafGrainSize);
    }

protected:
    NodeList<NodeT> mList;
};// DynamicNodeManagerLink class


template<typename TreeOrLeafManagerT, Index _LEVELS>
class DynamicNodeManager
{
public:
    static const Index LEVELS = _LEVELS;
    static_assert(LEVELS > 0,
        "expected instantiation of template specialization"); // see specialization below
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    using NonConstChildNodeType = typename RootNodeType::ChildNodeType;
    using ChildNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstChildNodeType>::Type;
    static_assert(RootNodeType::LEVEL >= LEVELS, "number of levels exceeds root node height");

    explicit DynamicNodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) { }

    DynamicNodeManager(const DynamicNodeManager&) = delete;

    /// @brief Return a reference to the root node.
    const NonConstRootNodeType& root() const { return mRoot; }

    /// @brief   Threaded method that applies a user-supplied functor
    ///          to all the nodes in the tree.
    ///
    /// @param op               user-supplied functor, see examples for interface details.
    /// @param threaded         optional toggle to disable threading, on by default.
    /// @param leafGrainSize    optional parameter to specify the grainsize
    ///                         for threading over leaf nodes, one by default.
    /// @param nonLeafGrainSize optional parameter to specify the grainsize
    ///                         for threading over non-leaf nodes, one by default.
    ///
    /// @note There are two key differences to the interface of the
    /// user-supplied functor to the NodeManager class - (1) the operator()
    /// method aligns with the LeafManager class in expecting the index of the
    /// node in a linear array of identical node types, (2) the operator()
    /// method returns a boolean termination value with true indicating that
    /// children of this node should be processed, false indicating the
    /// early-exit termination should occur.
    ///
    /// @note Unlike the NodeManager, the foreach() method of the
    /// DynamicNodeManager uses copy-by-reference for the user-supplied functor.
    /// This can be an issue when using a shared Accessor or shared Sampler in
    /// the operator as they are not inherently thread-safe. For these use
    /// cases, it is recommended to create the Accessor or Sampler in the
    /// operator execution itself.
    ///
    /// @par Example:
    /// @code
    /// // Functor to densify the first child node in a linear array. Note
    /// // this implementation also illustrates how different
    /// // computation can be applied to the different node types.
    ///
    /// template<typename TreeT>
    /// struct DensifyOp
    /// {
    ///     using RootT = typename TreeT::RootNodeType;
    ///     using LeafT = typename TreeT::LeafNodeType;
    ///
    ///     DensifyOp() = default;
    ///
    ///     // Processes the root node. Required by the DynamicNodeManager
    ///     bool operator()(RootT&, size_t) const { return true; }
    ///
    ///     // Processes the internal nodes. Required by the DynamicNodeManager
    ///     template<typename NodeT>
    ///     bool operator()(NodeT& node, size_t idx) const
    ///     {
    ///         // densify child
    ///         for (auto iter = node.cbeginValueAll(); iter; ++iter) {
    ///             const openvdb::Coord ijk = iter.getCoord();
    ///             node.addChild(new typename NodeT::ChildNodeType(iter.getCoord(), NodeT::LEVEL, true));
    ///         }
    ///         // early-exit termination for all non-zero index children
    ///         return idx == 0;
    ///     }
    ///     // Processes the leaf nodes. Required by the DynamicNodeManager
    ///     bool operator()(LeafT&, size_t) const
    ///     {
    ///         return true;
    ///     }
    /// };// DensifyOp
    ///
    /// // usage:
    /// DensifyOp<FloatTree> op;
    /// tree::DynamicNodeManager<FloatTree> nodes(tree);
    /// nodes.foreachTopDown(op);
    ///
    /// @endcode
    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        mChain.foreachTopDown(op, mRoot, threaded, leafGrainSize, nonLeafGrainSize);
    }

    /// @brief   Threaded method that processes nodes with a user supplied functor
    ///
    /// @param op        user-supplied functor, see examples for interface details.
    /// @param threaded  optional toggle to disable threading, on by default.
    /// @param leafGrainSize    optional parameter to specify the grainsize
    ///                         for threading over leaf nodes, one by default.
    /// @param nonLeafGrainSize optional parameter to specify the grainsize
    ///                         for threading over non-leaf nodes, one by default.
    ///
    /// @note There are two key differences to the interface of the
    /// user-supplied functor to the NodeManager class - (1) the operator()
    /// method aligns with the LeafManager class in expecting the index of the
    /// node in a linear array of identical node types, (2) the operator()
    /// method returns a boolean termination value with true indicating that
    /// children of this node should be processed, false indicating the
    /// early-exit termination should occur.
    ///
    /// @par Example:
    /// @code
    ///  // Functor to count nodes in a tree
    ///  template<typename TreeType>
    ///  struct NodeCountOp
    ///  {
    ///      NodeCountOp() : nodeCount(TreeType::DEPTH, 0), totalCount(0)
    ///      {
    ///      }
    ///      NodeCountOp(const NodeCountOp& other, tbb::split) :
    ///          nodeCount(TreeType::DEPTH, 0), totalCount(0)
    ///      {
    ///      }
    ///      void join(const NodeCountOp& other)
    ///      {
    ///          for (size_t i = 0; i < nodeCount.size(); ++i) {
    ///              nodeCount[i] += other.nodeCount[i];
    ///          }
    ///          totalCount += other.totalCount;
    ///      }
    ///      // do nothing for the root node
    ///      bool operator()(const typename TreeT::RootNodeType& node, size_t)
    ///      {
    ///          return true;
    ///      }
    ///      // count the internal and leaf nodes
    ///      template<typename NodeT>
    ///      bool operator()(const NodeT& node, size_t)
    ///      {
    ///          ++(nodeCount[NodeT::LEVEL]);
    ///          ++totalCount;
    ///          return true;
    ///      }
    ///      std::vector<openvdb::Index64> nodeCount;
    ///      openvdb::Index64 totalCount;
    /// };
    ///
    /// // usage:
    /// NodeCountOp<FloatTree> op;
    /// tree::DynamicNodeManager<FloatTree> nodes(tree);
    /// nodes.reduceTopDown(op);
    ///
    /// @endcode
    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        mChain.reduceTopDown(op, mRoot, threaded, leafGrainSize, nonLeafGrainSize);
    }

protected:
    RootNodeType& mRoot;
    DynamicNodeManagerLink<ChildNodeType, LEVELS-1> mChain;
};// DynamicNodeManager class



////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with no caching of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 0>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static const Index LEVELS = 0;

    NodeManager(TreeOrLeafManagerT& tree, bool /*serial*/ = false) : mRoot(tree.root()) { }

    NodeManager(const NodeManager&) = delete;

    /// @brief Clear all the cached tree nodes
    void clear() {}

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool /*serial*/ = false) { }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return 0; }

    Index64 nodeCount(Index) const { return 0; }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool, size_t) { op(mRoot); }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool, size_t) { op(mRoot); }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool, size_t) { op(mRoot); }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool, size_t) { op(mRoot); }

protected:
    RootNodeType& mRoot;
}; // NodeManager<0>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with one level of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 1>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 0, "expected instantiation of template specialization");
    static const Index LEVELS = 1;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false)
        : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    NodeManager(const NodeManager&) = delete;

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool /*serial*/ = false) { mList0.initRootChildren(mRoot); }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mList0.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const { return i==0 ? mList0.nodeCount() : 0; }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT1 = RootNodeType;
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;
    using ListT0 = NodeList<NodeT0>;

    NodeT1& mRoot;
    ListT0 mList0;
}; // NodeManager<1>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with two levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 2>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 1, "expected instantiation of template specialization");
    static const Index LEVELS = 2;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false) : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    NodeManager(const NodeManager&) = delete;

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool serial = false)
    {
        mList1.initRootChildren(mRoot);
        mList0.initNodeChildren(mList1, NodeFilter(), serial);
    }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mList0.nodeCount() + mList1.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const
    {
        return i==0 ? mList0.nodeCount() : i==1 ? mList1.nodeCount() : 0;
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList1.foreach(op, threaded, grainSize);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList1.reduce(op, threaded, grainSize);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT2 = RootNodeType;
    using NonConstNodeT1 = typename NodeT2::ChildNodeType;
    using NodeT1 = typename CopyConstness<RootNodeType, NonConstNodeT1>::Type;  // upper level
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;  // lower level

    using ListT1 = NodeList<NodeT1>; // upper level
    using ListT0 = NodeList<NodeT0>; // lower level

    NodeT2& mRoot;
    ListT1 mList1;
    ListT0 mList0;
}; // NodeManager<2>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with three levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 3>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 2, "expected instantiation of template specialization");
    static const Index LEVELS = 3;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false) : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    NodeManager(const NodeManager&) = delete;

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); mList2.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool serial = false)
    {
        mList2.initRootChildren(mRoot);
        mList1.initNodeChildren(mList2, NodeFilter(), serial);
        mList0.initNodeChildren(mList1, NodeFilter(), serial);
    }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const { return mList0.nodeCount()+mList1.nodeCount()+mList2.nodeCount(); }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const
    {
        return i==0 ? mList0.nodeCount() : i==1 ? mList1.nodeCount()
             : i==2 ? mList2.nodeCount() : 0;
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList2.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList2.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList2.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList2.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT3 = RootNodeType;
    using NonConstNodeT2 = typename NodeT3::ChildNodeType;
    using NodeT2 = typename CopyConstness<RootNodeType, NonConstNodeT2>::Type;  // upper level
    using NonConstNodeT1 = typename NodeT2::ChildNodeType;
    using NodeT1 = typename CopyConstness<RootNodeType, NonConstNodeT1>::Type;  // mid level
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;  // lower level

    using ListT2 = NodeList<NodeT2>; // upper level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT3& mRoot;
    ListT2 mList2;
    ListT1 mList1;
    ListT0 mList0;
}; // NodeManager<3>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with four levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 4>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 3, "expected instantiation of template specialization");
    static const Index LEVELS = 4;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false) : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    NodeManager(const NodeManager&) = delete; // disallow copy-construction

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); mList2.clear(); mList3.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool serial = false)
    {
        mList3.initRootChildren(mRoot);
        mList2.initNodeChildren(mList3, NodeFilter(), serial);
        mList1.initNodeChildren(mList2, NodeFilter(), serial);
        mList0.initNodeChildren(mList1, NodeFilter(), serial);
    }

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    /// @brief Return the total number of cached nodes (excluding the root node)
    Index64 nodeCount() const
    {
        return mList0.nodeCount() + mList1.nodeCount()
             + mList2.nodeCount() + mList3.nodeCount();
    }

    /// @brief Return the number of cached nodes at level @a i, where
    /// 0 corresponds to the lowest level.
    Index64 nodeCount(Index i) const
    {
        return i==0 ? mList0.nodeCount() : i==1 ? mList1.nodeCount() :
               i==2 ? mList2.nodeCount() : i==3 ? mList3.nodeCount() : 0;
    }

    template<typename NodeOp>
    void foreachBottomUp(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList2.foreach(op, threaded, grainSize);
        mList3.foreach(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList3.foreach(op, threaded, grainSize);
        mList2.foreach(op, threaded, grainSize);
        mList1.foreach(op, threaded, grainSize);
        mList0.foreach(op, threaded, grainSize);
    }

    template<typename NodeOp>
    void reduceBottomUp(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        mList0.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList2.reduce(op, threaded, grainSize);
        mList3.reduce(op, threaded, grainSize);
        op(mRoot);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        op(mRoot);
        mList3.reduce(op, threaded, grainSize);
        mList2.reduce(op, threaded, grainSize);
        mList1.reduce(op, threaded, grainSize);
        mList0.reduce(op, threaded, grainSize);
    }

protected:
    using NodeT4 = RootNodeType;
    using NonConstNodeT3 = typename NodeT4::ChildNodeType;
    using NodeT3 = typename CopyConstness<RootNodeType, NonConstNodeT3>::Type;  // upper level
    using NonConstNodeT2 = typename NodeT3::ChildNodeType;
    using NodeT2 = typename CopyConstness<RootNodeType, NonConstNodeT2>::Type;  // upper mid level
    using NonConstNodeT1 = typename NodeT2::ChildNodeType;
    using NodeT1 = typename CopyConstness<RootNodeType, NonConstNodeT1>::Type;  // lower mid level
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;  // lower level

    using ListT3 = NodeList<NodeT3>; // upper level of internal nodes
    using ListT2 = NodeList<NodeT2>; // upper mid level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower mid level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT4& mRoot;
    ListT3  mList3;
    ListT2  mList2;
    ListT1  mList1;
    ListT0  mList0;
}; // NodeManager<4>


////////////////////////////////////////////


/// @private
/// Template specialization of the DynamicNodeManager with no caching of nodes
template<typename TreeOrLeafManagerT>
class DynamicNodeManager<TreeOrLeafManagerT, 0>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 0, "expected instantiation of template specialization");
    static const Index LEVELS = 0;

    explicit DynamicNodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) { }

    DynamicNodeManager(const DynamicNodeManager&) = delete;

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool /*threaded*/=true, size_t /*grainSize*/=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool /*threaded*/=true, size_t /*grainSize*/=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
    }

protected:
    using NodeT1 = RootNodeType;

    NodeT1& mRoot;
};// DynamicNodeManager<0> class


////////////////////////////////////////////


/// @private
/// Template specialization of the DynamicNodeManager with one level of nodes
template<typename TreeOrLeafManagerT>
class DynamicNodeManager<TreeOrLeafManagerT, 1>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 0, "expected instantiation of template specialization");
    static const Index LEVELS = 1;

    explicit DynamicNodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) { }

    DynamicNodeManager(const DynamicNodeManager&) = delete;

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t /*nonLeafGrainSize*/ =1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list0
        if (!mList0.initRootChildren(mRoot))                        return;
        ForeachFilterOp<NodeOp> nodeOp(op, mList0.nodeCount());
        mList0.foreachWithIndex(nodeOp, threaded, leafGrainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t /*nonLeafGrainSize*/ =1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list0
        if (!mList0.initRootChildren(mRoot))                        return;
        ReduceFilterOp<NodeOp> nodeOp(op, mList0.nodeCount());
        mList0.reduceWithIndex(nodeOp, threaded, leafGrainSize);
    }

protected:
    using NodeT1 = RootNodeType;
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;
    using ListT0 = NodeList<NodeT0>;

    NodeT1& mRoot;
    ListT0 mList0;
};// DynamicNodeManager<1> class


////////////////////////////////////////////


/// @private
/// Template specialization of the DynamicNodeManager with two levels of nodes
template<typename TreeOrLeafManagerT>
class DynamicNodeManager<TreeOrLeafManagerT, 2>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 1, "expected instantiation of template specialization");
    static const Index LEVELS = 2;

    explicit DynamicNodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) { }

    DynamicNodeManager(const DynamicNodeManager&) = delete;

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list1
        if (!mList1.initRootChildren(mRoot))                        return;
        ForeachFilterOp<NodeOp> nodeOp(op, mList1.nodeCount());
        mList1.foreachWithIndex(nodeOp, threaded, nonLeafGrainSize);
        // list0
        if (!mList0.initNodeChildren(mList1, nodeOp, !threaded))   return;
        mList0.foreachWithIndex(op, threaded, leafGrainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list1
        if (!mList1.initRootChildren(mRoot))                        return;
        ReduceFilterOp<NodeOp> nodeOp(op, mList1.nodeCount());
        mList1.reduceWithIndex(nodeOp, threaded, nonLeafGrainSize);
        // list0
        if (!mList0.initNodeChildren(mList1, nodeOp, !threaded))   return;
        mList0.reduceWithIndex(op, threaded, leafGrainSize);
    }

protected:
    using NodeT2 = RootNodeType;
    using NonConstNodeT1 = typename NodeT2::ChildNodeType;
    using NodeT1 = typename CopyConstness<RootNodeType, NonConstNodeT1>::Type;  // upper level
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;  // lower level

    using ListT1 = NodeList<NodeT1>; // upper level
    using ListT0 = NodeList<NodeT0>; // lower level

    NodeT2& mRoot;
    ListT1 mList1;
    ListT0 mList0;
};// DynamicNodeManager<2> class


////////////////////////////////////////////


/// @private
/// Template specialization of the DynamicNodeManager with three levels of nodes
template<typename TreeOrLeafManagerT>
class DynamicNodeManager<TreeOrLeafManagerT, 3>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 2, "expected instantiation of template specialization");
    static const Index LEVELS = 3;

    explicit DynamicNodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) { }

    DynamicNodeManager(const DynamicNodeManager&) = delete;

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list2
        if (!mList2.initRootChildren(mRoot))                        return;
        ForeachFilterOp<NodeOp> nodeOp2(op, mList2.nodeCount());
        mList2.foreachWithIndex(nodeOp2, threaded, nonLeafGrainSize);
        // list1
        if (!mList1.initNodeChildren(mList2, nodeOp2, !threaded))   return;
        ForeachFilterOp<NodeOp> nodeOp1(op, mList1.nodeCount());
        mList1.foreachWithIndex(nodeOp1, threaded, nonLeafGrainSize);
        // list0
        if (!mList0.initNodeChildren(mList1, nodeOp1, !threaded))   return;
        mList0.foreachWithIndex(op, threaded, leafGrainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list2
        if (!mList2.initRootChildren(mRoot))                        return;
        ReduceFilterOp<NodeOp> nodeOp2(op, mList2.nodeCount());
        mList2.reduceWithIndex(nodeOp2, threaded, nonLeafGrainSize);
        // list1
        if (!mList1.initNodeChildren(mList2, nodeOp2, !threaded))   return;
        ReduceFilterOp<NodeOp> nodeOp1(op, mList1.nodeCount());
        mList1.reduceWithIndex(nodeOp1, threaded, nonLeafGrainSize);
        // list0
        if (!mList0.initNodeChildren(mList1, nodeOp1, !threaded))   return;
        mList0.reduceWithIndex(op, threaded, leafGrainSize);
    }

protected:
    using NodeT3 = RootNodeType;
    using NonConstNodeT2 = typename NodeT3::ChildNodeType;
    using NodeT2 = typename CopyConstness<RootNodeType, NonConstNodeT2>::Type;  // upper level
    using NonConstNodeT1 = typename NodeT2::ChildNodeType;
    using NodeT1 = typename CopyConstness<RootNodeType, NonConstNodeT1>::Type;  // mid level
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;  // lower level

    using ListT2 = NodeList<NodeT2>; // upper level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT3& mRoot;
    ListT2 mList2;
    ListT1 mList1;
    ListT0 mList0;
};// DynamicNodeManager<3> class


////////////////////////////////////////////


/// @private
/// Template specialization of the DynamicNodeManager with four levels of nodes
template<typename TreeOrLeafManagerT>
class DynamicNodeManager<TreeOrLeafManagerT, 4>
{
public:
    using NonConstRootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    using RootNodeType = typename CopyConstness<TreeOrLeafManagerT, NonConstRootNodeType>::Type;
    static_assert(RootNodeType::LEVEL > 3, "expected instantiation of template specialization");
    static const Index LEVELS = 4;

    explicit DynamicNodeManager(TreeOrLeafManagerT& tree) : mRoot(tree.root()) { }

    DynamicNodeManager(const DynamicNodeManager&) = delete;

    /// @brief Return a reference to the root node.
    const RootNodeType& root() const { return mRoot; }

    template<typename NodeOp>
    void foreachTopDown(const NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list3
        if (!mList3.initRootChildren(mRoot))                        return;
        ForeachFilterOp<NodeOp> nodeOp3(op, mList3.nodeCount());
        mList3.foreachWithIndex(nodeOp3, threaded, nonLeafGrainSize);
        // list2
        if (!mList2.initNodeChildren(mList3, nodeOp3, !threaded))   return;
        ForeachFilterOp<NodeOp> nodeOp2(op, mList2.nodeCount());
        mList2.foreachWithIndex(nodeOp2, threaded, nonLeafGrainSize);
        // list1
        if (!mList1.initNodeChildren(mList2, nodeOp2, !threaded))   return;
        ForeachFilterOp<NodeOp> nodeOp1(op, mList1.nodeCount());
        mList1.foreachWithIndex(nodeOp1, threaded, nonLeafGrainSize);
        // list0
        if (!mList0.initNodeChildren(mList1, nodeOp1, !threaded))   return;
        mList0.foreachWithIndex(op, threaded, leafGrainSize);
    }

    template<typename NodeOp>
    void reduceTopDown(NodeOp& op, bool threaded = true,
        size_t leafGrainSize=1, size_t nonLeafGrainSize=1)
    {
        // root
        if (!op(mRoot, /*index=*/0))                                return;
        // list3
        if (!mList3.initRootChildren(mRoot))                        return;
        ReduceFilterOp<NodeOp> nodeOp3(op, mList3.nodeCount());
        mList3.reduceWithIndex(nodeOp3, threaded, nonLeafGrainSize);
        // list2
        if (!mList2.initNodeChildren(mList3, nodeOp3, !threaded))   return;
        ReduceFilterOp<NodeOp> nodeOp2(op, mList2.nodeCount());
        mList2.reduceWithIndex(nodeOp2, threaded, nonLeafGrainSize);
        // list1
        if (!mList1.initNodeChildren(mList2, nodeOp2, !threaded))   return;
        ReduceFilterOp<NodeOp> nodeOp1(op, mList1.nodeCount());
        mList1.reduceWithIndex(nodeOp1, threaded, nonLeafGrainSize);
        // list0
        if (!mList0.initNodeChildren(mList1, nodeOp1, !threaded))   return;
        mList0.reduceWithIndex(op, threaded, leafGrainSize);
    }

protected:
    using NodeT4 = RootNodeType;
    using NonConstNodeT3 = typename NodeT4::ChildNodeType;
    using NodeT3 = typename CopyConstness<RootNodeType, NonConstNodeT3>::Type;  // upper level
    using NonConstNodeT2 = typename NodeT3::ChildNodeType;
    using NodeT2 = typename CopyConstness<RootNodeType, NonConstNodeT2>::Type;  // upper mid level
    using NonConstNodeT1 = typename NodeT2::ChildNodeType;
    using NodeT1 = typename CopyConstness<RootNodeType, NonConstNodeT1>::Type;  // lower mid level
    using NonConstNodeT0 = typename NodeT1::ChildNodeType;
    using NodeT0 = typename CopyConstness<RootNodeType, NonConstNodeT0>::Type;  // lower level

    using ListT3 = NodeList<NodeT3>; // upper level of internal nodes
    using ListT2 = NodeList<NodeT2>; // upper mid level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower mid level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT4& mRoot;
    ListT3 mList3;
    ListT2 mList2;
    ListT1 mList1;
    ListT0 mList0;
};// DynamicNodeManager<4> class


} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NODEMANAGER_HAS_BEEN_INCLUDED
