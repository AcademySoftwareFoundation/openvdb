// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file tree/NodeManager.h
///
/// @author Ken Museth
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


////////////////////////////////////////


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
    void initRootChildren(RootT& root)
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

        if (mNodeCount == 0)    return;

        // Populate the node pointers

        NodeT** nodePtr = mNodes;
        for (auto iter = root.beginChildOn(); iter; ++iter) {
            *nodePtr++ = &iter.getValue();
        }
    }

    // initialize this node list from another node list containing the parent nodes
    template <typename ParentsT>
    void initNodeChildren(ParentsT& parents, bool serial = false)
    {
        // Compute the node counts for each node

        std::vector<Index32> nodeCounts;
        if (serial) {
            nodeCounts.reserve(parents.nodeCount());
            for (size_t i = 0; i < parents.nodeCount(); i++) {
                nodeCounts.push_back(parents(i).childCount());
            }
        } else {
            nodeCounts.resize(parents.nodeCount());
            tbb::parallel_for(
                tbb::blocked_range<Index64>(0, parents.nodeCount(), /*grainsize=*/64),
                [&](tbb::blocked_range<Index64>& range)
                {
                    for (Index64 i = range.begin(); i < range.end(); i++) {
                        nodeCounts[i] = parents(i).childCount();
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

        if (mNodeCount == 0)    return;

        // Populate the node pointers

        if (serial) {
            NodeT** nodePtr = mNodes;
            for (size_t i = 0; i < parents.nodeCount(); i++) {
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
                        for (auto iter = parents(i).beginChildOn(); iter; ++iter) {
                            *nodePtr++ = &iter.getValue();
                        }
                    }
                }
            );
        }
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
        NodeTransformer<NodeOp> transform(op);
        transform.run(this->nodeRange(grainSize), threaded);
    }

    template<typename NodeOp>
    void reduce(NodeOp& op, bool threaded = true, size_t grainSize=1)
    {
        NodeReducer<NodeOp> transform(op);
        transform.run(this->nodeRange(grainSize), threaded);
    }

private:

    // Private struct of NodeList that performs parallel_for
    template<typename NodeOp>
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
            for (typename NodeRange::Iterator it = range.begin(); it; ++it) mNodeOp(*it);
        }
        const NodeOp mNodeOp;
    };// NodeList::NodeTransformer

    // Private struct of NodeList that performs parallel_reduce
    template<typename NodeOp>
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
            NodeOp &op = *mNodeOp;
            for (typename NodeRange::Iterator it = range.begin(); it; ++it) op(*it);
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
    NodeManagerLink() {}

    virtual ~NodeManagerLink() {}

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
        mList.initNodeChildren(parents);
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
    NodeManagerLink<typename NodeT::ChildNodeType, LEVEL-1> mNext;
};// NodeManagerLink class


////////////////////////////////////////


/// @private
/// @brief Specialization that terminates the chain of cached tree nodes
/// @note It is for internal use and should rarely be used directly.
template<typename NodeT>
class NodeManagerLink<NodeT, 0>
{
public:
    NodeManagerLink() {}

    virtual ~NodeManagerLink() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList.clear(); }

    template <typename RootT>
    void initRootChildren(RootT& root, bool /*serial*/ = false) { mList.initRootChildren(root); }

    template<typename ParentsT>
    void initNodeChildren(ParentsT& parents, bool serial = false) { mList.initNodeChildren(parents, serial); }

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
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL >= LEVELS, "number of levels exceeds root node height");

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false)
        : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    virtual ~NodeManager() {}

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
    NodeManagerLink<typename RootNodeType::ChildNodeType, LEVELS-1> mChain;

private:
    NodeManager(const NodeManager&) {}//disallow copy-construction
};// NodeManager class


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with no caching of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 0>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static const Index LEVELS = 0;

    NodeManager(TreeOrLeafManagerT& tree, bool /*serial*/ = false) : mRoot(tree.root()) { }

    virtual ~NodeManager() {}

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

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<0>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with one level of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 1>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 0, "expected instantiation of template specialization");
    static const Index LEVELS = 1;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false)
        : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    virtual ~NodeManager() {}

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
    using NodeT0 = typename NodeT1::ChildNodeType;
    using ListT0 = NodeList<NodeT0>;

    NodeT1& mRoot;
    ListT0 mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<1>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with two levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 2>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 1, "expected instantiation of template specialization");
    static const Index LEVELS = 2;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false) : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool serial = false)
    {
        mList1.initRootChildren(mRoot);
        mList0.initNodeChildren(mList1, serial);
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
    using NodeT1 = typename NodeT2::ChildNodeType; // upper level
    using NodeT0 = typename NodeT1::ChildNodeType; // lower level

    using ListT1 = NodeList<NodeT1>; // upper level
    using ListT0 = NodeList<NodeT0>; // lower level

    NodeT2& mRoot;
    ListT1 mList1;
    ListT0 mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<2>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with three levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 3>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 2, "expected instantiation of template specialization");
    static const Index LEVELS = 3;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false) : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); mList2.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool serial = false)
    {
        mList2.initRootChildren(mRoot);
        mList1.initNodeChildren(mList2, serial);
        mList0.initNodeChildren(mList1, serial);
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
    using NodeT2 = typename NodeT3::ChildNodeType; // upper level
    using NodeT1 = typename NodeT2::ChildNodeType; // mid level
    using NodeT0 = typename NodeT1::ChildNodeType; // lower level

    using ListT2 = NodeList<NodeT2>; // upper level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT3& mRoot;
    ListT2 mList2;
    ListT1 mList1;
    ListT0 mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<3>


////////////////////////////////////////////


/// @private
/// Template specialization of the NodeManager with four levels of nodes
template<typename TreeOrLeafManagerT>
class NodeManager<TreeOrLeafManagerT, 4>
{
public:
    using RootNodeType = typename TreeOrLeafManagerT::RootNodeType;
    static_assert(RootNodeType::LEVEL > 3, "expected instantiation of template specialization");
    static const Index LEVELS = 4;

    NodeManager(TreeOrLeafManagerT& tree, bool serial = false) : mRoot(tree.root())
    {
        this->rebuild(serial);
    }

    virtual ~NodeManager() {}

    /// @brief Clear all the cached tree nodes
    void clear() { mList0.clear(); mList1.clear(); mList2.clear(); mList3.clear(); }

    /// @brief Clear and recache all the tree nodes from the
    /// tree. This is required if tree nodes have been added or removed.
    void rebuild(bool serial = false)
    {
        mList3.initRootChildren(mRoot);
        mList2.initNodeChildren(mList3, serial);
        mList1.initNodeChildren(mList2, serial);
        mList0.initNodeChildren(mList1, serial);
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
    using NodeT3 = typename NodeT4::ChildNodeType; // upper level
    using NodeT2 = typename NodeT3::ChildNodeType; // upper mid level
    using NodeT1 = typename NodeT2::ChildNodeType; // lower mid level
    using NodeT0 = typename NodeT1::ChildNodeType; // lower level

    using ListT3 = NodeList<NodeT3>; // upper level of internal nodes
    using ListT2 = NodeList<NodeT2>; // upper mid level of internal nodes
    using ListT1 = NodeList<NodeT1>; // lower mid level of internal nodes
    using ListT0 = NodeList<NodeT0>; // lower level of internal nodes or leafs

    NodeT4& mRoot;
    ListT3  mList3;
    ListT2  mList2;
    ListT1  mList1;
    ListT0  mList0;

private:
    NodeManager(const NodeManager&) {} // disallow copy-construction
}; // NodeManager<4>

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NODEMANAGER_HAS_BEEN_INCLUDED
