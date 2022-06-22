// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file NodeManager.h

    \author Ken Museth

    \date February 12, 2021

    \brief This class allows for sequential access to nodes in a NanoVDB tree.

    \brief Currently it is limited to the host (CPU) but it can easily be ported
           to the device (GPU) if sequential node access is required.
*/

#include "../NanoVDB.h"
#include "Invoke.h"

#ifndef NANOVDB_NODEMANAGER_H_HAS_BEEN_INCLUDED
#define NANOVDB_NODEMANAGER_H_HAS_BEEN_INCLUDED

namespace nanovdb {

/// @brief NodeNanager maintains separate linear arrays of the three nodes types
template<typename GridT>
class NodeManager;

/// @brief LeafNanager maintains a linear array of leaf nodes
template<typename GridT>
class LeafManager;

/// @brief creates a NodeManager from a grid. Move semantics is used.
template<typename GridT>
NodeManager<GridT> createNodeMgr(GridT &grid);

/// @brief creates a LeafManager from a grid. Move semantics is used.
template<typename GridT>
LeafManager<GridT> createLeafMgr(GridT &grid);

/// @brief This host class allows for sequential access to nodes in a NanoVDB tree
///
/// @details Nodes are stored breadth first to allow for sequential access of nodes
///          at a particular level.
template<typename GridT>
class NodeManager
{
    using TreeT = typename GridTree<GridT>::type;
    template<int LEVEL>
    using NodeT = typename NodeTrait<TreeT, LEVEL>::type;
    using RootT = NodeT<3>;// root node
    using Node2 = NodeT<2>;// upper internal node
    using Node1 = NodeT<1>;// lower internal node
    using Node0 = NodeT<0>;// leaf node

    template <typename T>
    void sequential(T **nodes);

public:
    /// @brief Empty constructor
    NodeManager();

    /// @brief Construction from a grid
    NodeManager(GridT &grid);

    /// @brief Disallow copy construction
    NodeManager(const NodeManager&) = delete;

    /// @brief Move constructor
    NodeManager(NodeManager&&);

    /// @brief Destructor
    ~NodeManager() { this->clear(); }

    /// @brief Disallow copy assignment operator
    NodeManager& operator=(const NodeManager&) = delete;

    /// @brief Move assignment operator
    NodeManager& operator=(NodeManager&&);

    /// @brief Return true of this instance is uninitialized
    bool empty() const { return mGrid == nullptr; }

    /// @brief Return the memory footprint in bytes of this instance
    inline size_t memUsage() const;

    /// @brief Return a pointer to the grid, or NULL if it is uninitialized
    GridT* grid() { return mGrid; }

    /// @brief Return a pointer to the tree, or NULL if it is uninitialized
    TreeT* tree() { return mTree; }

    /// @brief Return a pointer to the root, or NULL if it is uninitialized
    RootT* root() { return mRoot; }

    /// @brief Return the number of tree nodes at the specified level
    /// @details 0 is leaf, 1 is lower internal, 2 is upper internal and 3 is root
    uint64_t nodeCount(int level) const { return mNodeCount[level]; }

    /// @brief Return the i'th leaf node
    ///
    /// @warning Never call this method is the NodeManager is un-initialized
    Node0* leaf( uint32_t i) const { return mLeafs[i]; }

    /// @brief Return the i'th lower internal node
    ///
    /// @warning Never call this method is the NodeManager is un-initialized
    Node1* lower(uint32_t i) const { return mLower[i]; }

    /// @brief Return the i'th upper internal node
    ///
    /// @warning Never call this method is the NodeManager is un-initialized
    Node2* upper(uint32_t i) const { return mUpper[i]; }

private:

    void clear();

    GridT*   mGrid;
    TreeT*   mTree;
    RootT*   mRoot;
    uint64_t mNodeCount[3];
    Node2**  mUpper;
    Node1**  mLower;
    Node0**  mLeafs;
}; // NodeManager<GridT> class

template<typename GridT>
NodeManager<GridT>::NodeManager()
    : mGrid(nullptr)
    , mTree(nullptr)
    , mRoot(nullptr)
    , mNodeCount{0,0,0}
    , mUpper(nullptr)
    , mLower(nullptr)
    , mLeafs(nullptr)
{
}

template<typename GridT>
NodeManager<GridT>::NodeManager(GridT &grid)
    : mGrid(&grid)
    , mTree(&grid.tree())
    , mRoot(&grid.tree().root())
    , mNodeCount{mTree->nodeCount(0), mTree->nodeCount(1), mTree->nodeCount(2)}
    , mUpper(new Node2*[mNodeCount[2]])
    , mLower(new Node1*[mNodeCount[1]])
    , mLeafs(new Node0*[mNodeCount[0]])

{
    if (Node0::FIXED_SIZE && Node1::FIXED_SIZE && Node2::FIXED_SIZE &&// resolved at compile-time
        grid.isBreadthFirst()) {
#if 1
            invoke([&](){this->sequential(mLeafs);},
                   [&](){this->sequential(mLower);},
                   [&](){this->sequential(mUpper);});
#else
            this->sequential(mLeafs);
            this->sequential(mLower);
            this->sequential(mUpper);
#endif
    } else {
        auto **ptr2 = mUpper;
        auto **ptr1 = mLower;
        auto **ptr0 = mLeafs;
        auto *data3 = mRoot->data();
        // Performs depth first traversal but breadth first insertion
        for (uint32_t i=0, size=data3->mTableSize; i<size; ++i) {
            auto *tile = data3->tile(i);
            if (!tile->isChild()) continue;
            Node2 *node2 = data3->getChild(tile);
            *ptr2++ = node2;
            auto *data2 = node2->data();
            for (auto it2 = data2->mChildMask.beginOn(); it2; ++it2) {
                Node1 *node1 = data2->getChild(*it2);
                *ptr1++ = node1;
                auto *data1 = node1->data();
                for (auto it1 = data1->mChildMask.beginOn(); it1; ++it1) {
                    Node0 *node0 = data1->getChild(*it1);
                    *ptr0++ = node0;
                }// loop over child nodes of the lower internal node
            }// loop over child nodes of the upper internal node
        }// loop over child nodes of the root node
    }
}

template<typename GridT>
NodeManager<GridT>::NodeManager(NodeManager &&other)
    : mGrid(other.mGrid)
    , mTree(other.mTree)
    , mRoot(other.mRoot)
    , mNodeCount{other.mNodeCount[0],other.mNodeCount[1],other.mNodeCount[2]}
    , mUpper(other.mUpper)
    , mLower(other.mLower)
    , mLeafs(other.mLeafs)
{
    other.mGrid  = nullptr;
    other.mTree  = nullptr;
    other.mRoot  = nullptr;
    other.mNodeCount[0] = 0;
    other.mNodeCount[1] = 0;
    other.mNodeCount[2] = 0;
    other.mUpper = nullptr;
    other.mLower = nullptr;
    other.mLeafs = nullptr;
}

template<typename GridT>
NodeManager<GridT>& NodeManager<GridT>::operator=(NodeManager &&other)
{
    this->clear();
    mGrid  = other.mGrid;
    mTree  = other.mTree;
    mRoot  = other.mRoot;
    mNodeCount[0] = other.mNodeCount[0];
    mNodeCount[1] = other.mNodeCount[1];
    mNodeCount[2] = other.mNodeCount[2];
    mUpper = other.mUpper;
    mLower = other.mLower;
    mLeafs = other.mLeafs;

    other.mGrid  = nullptr;
    other.mTree  = nullptr;
    other.mRoot  = nullptr;
    other.mNodeCount[0] = 0;
    other.mNodeCount[1] = 0;
    other.mNodeCount[2] = 0;
    other.mUpper = nullptr;
    other.mLower = nullptr;
    other.mLeafs = nullptr;

    return *this;
}

template<typename GridT>
void NodeManager<GridT>::clear()
{
    delete [] mUpper;
    delete [] mLower;
    delete [] mLeafs;
}

template<typename GridT>
size_t NodeManager<GridT>::memUsage() const
{
    return sizeof(*this) +
           mNodeCount[0]*sizeof(Node0*) +
           mNodeCount[1]*sizeof(Node1*) +
           mNodeCount[2]*sizeof(Node2*);
}

template<typename GridT>
NodeManager<GridT> createNodeMgr(GridT &grid)
{
    NodeManager<GridT> mgr(grid);
    return mgr;// // is converted to r-value so return value is move constructed!
}


template<typename GridT>
template <typename T>
void NodeManager<GridT>::sequential(T **nodes)
{
    NANOVDB_ASSERT(mGrid->template isSequential<T>());
    auto *ptr = mTree->template getFirstNode<T>();
    uint64_t n = mNodeCount[T::LEVEL] / 4;
    while (n--) {// loop unrolling
        *nodes++ = ptr++;
        *nodes++ = ptr++;
        *nodes++ = ptr++;
        *nodes++ = ptr++;
    }
    n = mNodeCount[T::LEVEL] % 4;
    while (n--) {// loop reminder
        *nodes++ = ptr++;
    }
}

//////////////////////////////////////////////////////////////////////////////

template<typename GridT>
class LeafManager
{
    using LeafT = typename NodeTrait<typename GridTree<GridT>::type, 0>::type;

public:
    /// @brief Empty constructor
    LeafManager() : mGrid(nullptr), mSize(0), mLeafs(nullptr) {}

    /// @brief Construction from a grid
    LeafManager(GridT &grid);

    /// @brief Disallow copy construction
    LeafManager(const LeafManager&) = delete;

    /// @brief Move constructor
    LeafManager(LeafManager&&);

    /// @brief Destructor
    ~LeafManager() { delete [] mLeafs; }

    /// @brief Disallow copy assignment operator
    LeafManager& operator=(const LeafManager&) = delete;

    /// @brief Move assignment operator
    LeafManager& operator=(LeafManager&&);

    /// @brief Return true of this instance is un-initialized
    bool empty() const { return mGrid == nullptr; }

    /// @brief Return the memory footprint in bytes of this instance
    size_t memUsage() const { return sizeof(*this) + mSize*sizeof(LeafT*); }

    /// @brief Return a pointer to the grid, or NULL if it is uninitialized
    GridT* grid() { return mGrid; }

    /// @brief Return the number of leaf nodes
    uint32_t size() const { return mSize; };

    /// @brief Return the i'th leaf node
    ///
    /// @warning Never call this method is the LeafManager is uninitialized
    LeafT* operator[](uint32_t i) const { return mLeafs[i]; };

private:

    GridT   *mGrid;
    uint32_t mSize;
    LeafT  **mLeafs;

}; // LeafManager<GridT> class

template<typename GridT>
LeafManager<GridT>::LeafManager(GridT &grid)
    : mGrid(&grid), mSize(grid.tree().nodeCount(0)), mLeafs(nullptr)
{
    if (mSize>0) {
        mLeafs = new LeafT*[mSize];
        auto **leafs = mLeafs;
        if (grid.template isSequential<LeafT>()) {
            auto *ptr = grid.tree().template getFirstNode<LeafT>();
            uint64_t n = mSize / 4;
            while (n--) {// loop unrolling
                *leafs++ = ptr++;
                *leafs++ = ptr++;
                *leafs++ = ptr++;
                *leafs++ = ptr++;
            }
            n = mSize % 4;
            while (n--) {// loop reminder
                *leafs++ = ptr++;
            }
        } else {
            auto *data3 = grid.tree().root().data();
            for (uint32_t i=0, size=data3->mTableSize; i<size; ++i) {
                auto *tile = data3->tile(i);
                if (!tile->isChild()) continue;
                auto *data2 = data3->getChild(tile)->data();
                for (auto it2 = data2->mChildMask.beginOn(); it2; ++it2) {
                    auto *data1 = data2->getChild(*it2)->data();
                    for (auto it1 = data1->mChildMask.beginOn(); it1; ++it1) {
                        *leafs++ = data1->getChild(*it1);
                    }// loop over child nodes of the lower internal node
                }// loop over child nodes of the upper internal node
            }// loop over child nodes of the root node
        }
    }
}

template<typename GridT>
LeafManager<GridT>::LeafManager(LeafManager &&other)
{
    mGrid = other.mGrid;
    mSize = other.mSize;
    mLeafs = other.mLeafs;
    other.mGrid = nullptr;
    other.mSize = 0;
    other.mLeafs = nullptr;
}

template<typename GridT>
LeafManager<GridT>& LeafManager<GridT>::operator=(LeafManager &&other)
{
    mGrid = other.mGrid;
    mSize = other.mSize;
    delete [] mLeafs;
    mLeafs = other.mLeafs;
    other.mGrid = nullptr;
    other.mSize = 0;
    other.mLeafs = nullptr;
    return *this;
}

template<typename GridT>
LeafManager<GridT> createLeafMgr(GridT &grid)
{
    LeafManager<GridT> mgr(grid);
    return mgr;// // is converted to r-value so return value is move constructed!
}

} // namespace nanovdb

#endif // NANOVDB_NODEMANAGER_H_HAS_BEEN_INCLUDED
