///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file LeafManager.h
///
/// @brief A LeafManager manages a linear array of pointers to a given tree's
/// leaf nodes, as well as optional auxiliary buffers (one or more per leaf)
/// that can be swapped with the leaf nodes' voxel data buffers.
/// @details The leaf array is useful for multithreaded computations over
/// leaf voxels in a tree with static topology but varying voxel values.
/// The auxiliary buffers are convenient for temporal integration.
/// Efficient methods are provided for multithreaded swapping and synching
/// (i.e., copying the contents) of these buffers.

#ifndef OPENVDB_TREE_LEAFMANAGER_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_LEAFMANAGER_HAS_BEEN_INCLUDED

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <openvdb/Types.h>
#include "TreeIterator.h" // for CopyConstness

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

namespace leafmgr {

//@{
/// Useful traits for Tree types
template<typename TreeT> struct TreeTraits {
    static const bool IsConstTree = false;
    typedef typename TreeT::LeafIter LeafIterType;
};
template<typename TreeT> struct TreeTraits<const TreeT> {
    static const bool IsConstTree = true;
    typedef typename TreeT::LeafCIter LeafIterType;
};
//@}

} // namespace leafmgr


/// This helper class implements LeafManager methods that need to be
/// specialized for const vs. non-const trees.
template<typename ManagerT>
struct LeafManagerImpl
{
    typedef typename ManagerT::RangeType  RangeT;
    typedef typename ManagerT::LeafType   LeafT;
    typedef typename ManagerT::BufferType BufT;

    static inline void doSwapLeafBuffer(const RangeT& r, size_t auxBufferIdx,
        LeafT** leafs, BufT* bufs, size_t bufsPerLeaf)
    {
        for (size_t n = r.begin(), m = r.end(), N = bufsPerLeaf; n != m; ++n) {
            leafs[n]->swap(bufs[n * N + auxBufferIdx]);
        }
    }
};


////////////////////////////////////////


/// @brief This class manages a linear array of pointers to a given tree's
/// leaf nodes, as well as optional auxiliary buffers (one or more per leaf)
/// that can be swapped with the leaf nodes' voxel data buffers.
/// @details The leaf array is useful for multithreaded computations over
/// leaf voxels in a tree with static topology but varying voxel values.
/// The auxiliary buffers are convenient for temporal integration.
/// Efficient methods are provided for multithreaded swapping and sync'ing
/// (i.e., copying the contents) of these buffers.
///
/// @note Buffer index 0 denotes a leaf node's internal voxel data buffer.
/// Any auxiliary buffers are indexed starting from one.
template<typename TreeT>
class LeafManager
{
public:
    typedef TreeT                                                      TreeType;
    typedef typename TreeT::ValueType                                  ValueType;
    typedef typename TreeT::RootNodeType                               RootNodeType;
    typedef typename TreeType::LeafNodeType                            NonConstLeafType;
    typedef typename CopyConstness<TreeType, NonConstLeafType>::Type   LeafType;
    typedef LeafType                                                   LeafNodeType;
    typedef typename leafmgr::TreeTraits<TreeT>::LeafIterType          LeafIterType;
    typedef typename LeafType::Buffer                                  NonConstBufferType;
    typedef typename CopyConstness<TreeType, NonConstBufferType>::Type BufferType;
    typedef tbb::blocked_range<size_t>                                 RangeType;//leaf index range
    static const Index DEPTH = 2;//root + leafs

    static const bool IsConstTree = leafmgr::TreeTraits<TreeT>::IsConstTree;

    class LeafRange
    {
    public:
        class Iterator
        {
        public:
            Iterator(const LeafRange& range, size_t pos): mRange(range), mPos(pos)
            {
                assert(this->isValid());
            }
            Iterator& operator=(const Iterator& other)
            {
                mRange = other.mRange; mPos = other.mPos; return *this;
            }
            /// Advance to the next leaf node.
            Iterator& operator++() { ++mPos; return *this; }
            /// Return a reference to the leaf node to which this iterator is pointing.
            LeafType& operator*() const { return mRange.mLeafManager.leaf(mPos); }
            /// Return a pointer to the leaf node to which this iterator is pointing.
            LeafType* operator->() const { return &(this->operator*()); }
            /// @brief Return the nth buffer for the leaf node to which this iterator is pointing,
            /// where n = @a bufferIdx and n = 0 corresponds to the leaf node's own buffer.
            BufferType& buffer(size_t bufferIdx)
            {
                return mRange.mLeafManager.getBuffer(mPos, bufferIdx);
            }
            /// Return the index into the leaf array of the current leaf node.
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
            const LeafRange& leafRange() const { return mRange; }

        private:
            const LeafRange& mRange;
            size_t mPos;
        };// end Iterator

        LeafRange(size_t begin, size_t end, const LeafManager& leafManager, size_t grainSize=1):
            mEnd(end), mBegin(begin), mGrainSize(grainSize), mLeafManager(leafManager) {}

        Iterator begin() const {return Iterator(*this, mBegin);}

        Iterator end() const {return Iterator(*this, mEnd);}

        size_t size() const { return mEnd - mBegin; }

        size_t grainsize() const { return mGrainSize; }

        const LeafManager& leafManager() const { return mLeafManager; }

        bool empty() const {return !(mBegin < mEnd);}

        bool is_divisible() const {return mGrainSize < this->size();}

        LeafRange(LeafRange& r, tbb::split):
            mEnd(r.mEnd), mBegin(doSplit(r)), mGrainSize(r.mGrainSize),
              mLeafManager(r.mLeafManager) {}

    private:
        size_t mEnd, mBegin, mGrainSize;
        const LeafManager& mLeafManager;

        static size_t doSplit(LeafRange& r)
        {
            assert(r.is_divisible());
            size_t middle = r.mBegin + (r.mEnd - r.mBegin) / 2u;
            r.mEnd = middle;
            return middle;
        }
    };// end of LeafRange

    /// @brief Constructor from a tree reference and an auxiliary buffer count
    /// (default is no auxiliary buffers)
    LeafManager(TreeType& tree, size_t auxBuffersPerLeaf=0, bool serial=false):
        mTree(&tree),
        mLeafCount(0),
        mAuxBufferCount(0),
        mAuxBuffersPerLeaf(auxBuffersPerLeaf),
        mLeafs(NULL),
        mAuxBuffers(NULL),
        mTask(0),
        mIsMaster(true)
    {
        this->rebuild(serial);
    }

    /// Shallow copy constructor called by tbb::parallel_for() threads
    ///
    /// @note This should never get called directly
    LeafManager(const LeafManager& other):
        mTree(other.mTree),
        mLeafCount(other.mLeafCount),
        mAuxBufferCount(other.mAuxBufferCount),
        mAuxBuffersPerLeaf(other.mAuxBuffersPerLeaf),
        mLeafs(other.mLeafs),
        mAuxBuffers(other.mAuxBuffers),
        mTask(other.mTask),
        mIsMaster(false)
    {
    }

    virtual ~LeafManager()
    {
        if (mIsMaster) {
            delete [] mLeafs;
            delete [] mAuxBuffers;
        }
    }

    /// @brief (Re)initialize by resizing (if necessary) and repopulating the leaf array
    /// and by deleting existing auxiliary buffers and allocating new ones.
    /// @details Call this method if the tree's topology, and therefore the number
    /// of leaf nodes, changes.  New auxiliary buffers are initialized with copies
    /// of corresponding leaf node buffers.
    void rebuild(bool serial=false)
    {
        this->initLeafArray();
        this->initAuxBuffers(serial);
    }
    //@{
    /// Repopulate the leaf array and delete and reallocate auxiliary buffers.
    void rebuild(size_t auxBuffersPerLeaf, bool serial=false)
    {
        mAuxBuffersPerLeaf = auxBuffersPerLeaf;
        this->rebuild(serial);
    }
    void rebuild(TreeType& tree, bool serial=false)
    {
        mTree = &tree;
        this->rebuild(serial);
    }
    void rebuild(TreeType& tree, size_t auxBuffersPerLeaf, bool serial=false)
    {
        mTree = &tree;
        mAuxBuffersPerLeaf = auxBuffersPerLeaf;
        this->rebuild(serial);
    }
    //@}
    /// @brief Change the number of auxiliary buffers.
    /// @details If auxBuffersPerLeaf is 0, all existing auxiliary buffers are deleted.
    /// New auxiliary buffers are initialized with copies of corresponding leaf node buffers.
    /// This method does not rebuild the leaf array.
    void rebuildAuxBuffers(size_t auxBuffersPerLeaf, bool serial=false)
    {
        mAuxBuffersPerLeaf = auxBuffersPerLeaf;
        this->initAuxBuffers(serial);
    }
    /// @brief Remove the auxiliary buffers, but don't rebuild the leaf array.
    void removeAuxBuffers() { this->rebuildAuxBuffers(0); }

    /// @brief Remove the auxiliary buffers and rebuild the leaf array.
    void rebuildLeafArray()
    {
        this->removeAuxBuffers();
        this->initLeafArray();
    }

    /// Return the total number of allocated auxiliary buffers.
    size_t auxBufferCount() const { return mAuxBufferCount; }
    /// Return the number of auxiliary buffers per leaf node.
    size_t auxBuffersPerLeaf() const { return mAuxBuffersPerLeaf; }

    /// Return the number of leaf nodes.
    size_t leafCount() const { return mLeafCount; }

    /// Return a const reference to tree associated with this manager.
    const TreeType& tree() const { return *mTree; }

    /// Return a reference to the tree associated with this manager.
    TreeType& tree() { return *mTree; }

    /// Return a const reference to root node associated with this manager.
    const RootNodeType& root() const { return mTree->root(); }

    /// Return a reference to the root node associated with this manager.
    RootNodeType& root() { return mTree->root(); }

    /// Return @c true if the tree associated with this manager is immutable.
    bool isConstTree() const { return this->IsConstTree; }

    /// @brief Return a pointer to the leaf node at index @a leafIdx in the array.
    /// @note For performance reasons no range check is performed (other than an assertion)!
    LeafType& leaf(size_t leafIdx) const { assert(leafIdx<mLeafCount); return *mLeafs[leafIdx]; }

    /// @brief Return the leaf or auxiliary buffer for the leaf node at index @a leafIdx.
    /// If @a bufferIdx is zero, return the leaf buffer, otherwise return the nth
    /// auxiliary buffer, where n = @a bufferIdx - 1.
    ///
    /// @note For performance reasons no range checks are performed on the inputs
    /// (other than assertions)! Since auxiliary buffers, unlike leaf buffers,
    /// might not exist, be especially careful when specifying the @a bufferIdx.
    /// @note For const trees, this method always returns a reference to a const buffer.
    /// It is safe to @c const_cast and modify any auxiliary buffer (@a bufferIdx > 0),
    /// but it is not safe to modify the leaf buffer (@a bufferIdx = 0).
    BufferType& getBuffer(size_t leafIdx, size_t bufferIdx) const
    {
        assert(leafIdx < mLeafCount);
        assert(bufferIdx == 0 || bufferIdx - 1 < mAuxBuffersPerLeaf);
        return bufferIdx == 0 ? mLeafs[leafIdx]->buffer()
            : mAuxBuffers[leafIdx * mAuxBuffersPerLeaf + bufferIdx - 1];
    }

    /// @brief Return a @c tbb::blocked_range of leaf array indices.
    ///
    /// @note Consider using leafRange() instead, which provides access methods
    /// to leaf nodes and buffers.
    RangeType getRange(size_t grainsize = 1) const { return RangeType(0, mLeafCount, grainsize); }

    /// Return a TBB-compatible LeafRange.
    LeafRange leafRange(size_t grainsize = 1) const
    {
        return LeafRange(0, mLeafCount, *this, grainsize);
    }

    /// @brief Swap each leaf node's buffer with the nth corresponding auxiliary buffer,
    /// where n = @a bufferIdx.
    /// @return @c true if the swap was successful
    /// @param bufferIdx  index of the buffer that will be swapped with
    ///                   the corresponding leaf node buffer
    /// @param serial     if false, swap buffers in parallel using multiple threads.
    /// @note Recall that the indexing of auxiliary buffers is 1-based, since
    /// buffer index 0 denotes the leaf node buffer.  So buffer index 1 denotes
    /// the first auxiliary buffer.
    bool swapLeafBuffer(size_t bufferIdx, bool serial = false)
    {
        if (bufferIdx == 0 || bufferIdx > mAuxBuffersPerLeaf || this->isConstTree()) return false;
        mTask = boost::bind(&LeafManager::doSwapLeafBuffer, _1, _2, bufferIdx - 1);
        this->cook(serial ? 0 : 512);
        return true;//success
    }
    /// @brief Swap any two buffers for each leaf node.
    /// @note Recall that the indexing of auxiliary buffers is 1-based, since
    /// buffer index 0 denotes the leaf node buffer.  So buffer index 1 denotes
    /// the first auxiliary buffer.
    bool swapBuffer(size_t bufferIdx1, size_t bufferIdx2, bool serial = false)
    {
        const size_t b1 = std::min(bufferIdx1, bufferIdx2);
        const size_t b2 = std::max(bufferIdx1, bufferIdx2);
        if (b1 == b2 || b2 > mAuxBuffersPerLeaf) return false;
        if (b1 == 0) {
            if (this->isConstTree()) return false;
            mTask = boost::bind(&LeafManager::doSwapLeafBuffer, _1, _2, b2-1);
        } else {
            mTask = boost::bind(&LeafManager::doSwapAuxBuffer, _1, _2, b1-1, b2-1);
        }
        this->cook(serial ? 0 : 512);
        return true;//success
    }

    /// @brief Sync up the specified auxiliary buffer with the corresponding leaf node buffer.
    /// @return @c true if the sync was successful
    /// @param bufferIdx index of the buffer that will contain a
    ///                  copy of the corresponding leaf node buffer
    /// @param serial    if false, sync buffers in parallel using multiple threads.
    /// @note Recall that the indexing of auxiliary buffers is 1-based, since
    /// buffer index 0 denotes the leaf node buffer.  So buffer index 1 denotes
    /// the first auxiliary buffer.
    bool syncAuxBuffer(size_t bufferIdx, bool serial = false)
    {
        if (bufferIdx == 0 || bufferIdx > mAuxBuffersPerLeaf) return false;
        mTask = boost::bind(&LeafManager::doSyncAuxBuffer, _1, _2, bufferIdx - 1);
        this->cook(serial ? 0 : 64);
        return true;//success
    }

    /// @brief Sync up all auxiliary buffers with their corresponding leaf node buffers.
    /// @return true if the sync was successful
    /// @param serial  if false, sync buffers in parallel using multiple threads.
    bool syncAllBuffers(bool serial = false)
    {
        switch (mAuxBuffersPerLeaf) {
            case 0: return false;//nothing to do
            case 1: mTask = boost::bind(&LeafManager::doSyncAllBuffers1, _1, _2); break;
            case 2: mTask = boost::bind(&LeafManager::doSyncAllBuffers2, _1, _2); break;
            default: mTask = boost::bind(&LeafManager::doSyncAllBuffersN, _1, _2); break;
        }
        this->cook(serial ? 0 : 64);
        return true;//success
    }

    /// @brief   Threaded method that applies a user-supplied functor
    ///          to each leaf node in the LeafManager
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
    /// // Functor to offset a tree's voxel values with values from another tree.
    /// template<typename TreeType>
    /// struct OffsetOp
    /// {
    ///     typedef tree::ValueAccessor<const TreeType> Accessor;
    ///
    ///     OffsetOp(const TreeType& tree): mRhsTreeAcc(tree) {}
    ///
    ///     template <typename LeafNodeType>
    ///     void operator()(LeafNodeType &lhsLeaf, size_t) const
    ///     {
    ///         const LeafNodeType * rhsLeaf = mRhsTreeAcc.probeConstLeaf(lhsLeaf.origin());
    ///         if (rhsLeaf) {
    ///             typename LeafNodeType::ValueOnIter iter = lhsLeaf.beginValueOn();
    ///             for (; iter; ++iter) {
    ///                 iter.setValue(iter.getValue() + rhsLeaf->getValue(iter.pos()));
    ///             }
    ///         }
    ///     }
    /// private:
    ///     Accessor mRhsTreeAcc;
    /// };
    ///
    /// // usage:
    /// tree::LeafManager<FloatTree> leafNodes(lhsTree);
    /// leafNodes.foreach(OffsetOp<FloatTree>(rhsTree));
    ///
    /// // A functor that performs a min operation between different auxiliary buffers.
    /// template<typename LeafManagerType>
    /// struct MinOp
    /// {
    ///     typedef typename LeafManagerType::BufferType BufferType;
    ///
    ///     MinOp(LeafManagerType& leafNodes): mLeafs(leafNodes) {}
    ///
    ///     template <typename LeafNodeType>
    ///     void operator()(LeafNodeType &leaf, size_t leafIndex) const
    ///     {
    ///         // get the first buffer
    ///         BufferType& buffer = mLeafs.getBuffer(leafIndex, 1);
    ///
    ///         // min ...
    ///     }
    /// private:
    ///     LeafManagerType& mLeafs;
    /// };
    /// @endcode
    template<typename LeafOp>
    void foreach(const LeafOp& op, bool threaded = true, size_t grainSize=1)
    {
        LeafTransformer<LeafOp> transform(op);
        transform.run(this->leafRange(grainSize), threaded);
    }


    template<typename ArrayT>
    void getNodes(ArrayT& array)
    {
        typedef typename ArrayT::value_type T;
        BOOST_STATIC_ASSERT(boost::is_pointer<T>::value);
        typedef typename boost::mpl::if_<boost::is_const<typename boost::remove_pointer<T>::type>,
            const LeafType, LeafType>::type LeafT;

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (boost::is_same<T, LeafT*>::value) {
            array.resize(mLeafCount);
            for (size_t i=0; i<mLeafCount; ++i) array[i] = reinterpret_cast<T>(mLeafs[i]);
        } else {
            mTree->getNodes(array);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
            }

    template<typename ArrayT>
    void getNodes(ArrayT& array) const
    {
        typedef typename ArrayT::value_type T;
        BOOST_STATIC_ASSERT(boost::is_pointer<T>::value);
        BOOST_STATIC_ASSERT(boost::is_const<typename boost::remove_pointer<T>::type>::value);

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (boost::is_same<T, const LeafType*>::value) {
            array.resize(mLeafCount);
            for (size_t i=0; i<mLeafCount; ++i) array[i] = reinterpret_cast<T>(mLeafs[i]);
        } else {
            mTree->getNodes(array);
        }
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // All methods below are for internal use only and should never be called directly

    /// Used internally by tbb::parallel_for() - never call it directly!
    void operator()(const RangeType& r) const
    {
        if (mTask) mTask(const_cast<LeafManager*>(this), r);
        else OPENVDB_THROW(ValueError, "task is undefined");
    }



  private:

    // This a simple wrapper for a c-style array so it mimics the api
    // of a std container, e.g. std::vector or std::deque, and can be
    // passed to Tree::getNodes().
    struct MyArray {
        typedef LeafType* value_type;//required by Tree::getNodes
        value_type* ptr;
        MyArray(value_type* array) : ptr(array) {}
        void push_back(value_type leaf) { *ptr++ = leaf; }//required by Tree::getNodes
    };

    void initLeafArray()
    {
        const size_t leafCount = mTree->leafCount();
        if (leafCount != mLeafCount) {
            delete [] mLeafs;
            mLeafs = (leafCount == 0) ? NULL : new LeafType*[leafCount];
            mLeafCount = leafCount;
        }
        MyArray a(mLeafs);
        mTree->getNodes(a);
    }

    void initAuxBuffers(bool serial)
    {
        const size_t auxBufferCount = mLeafCount * mAuxBuffersPerLeaf;
        if (auxBufferCount != mAuxBufferCount) {
            delete [] mAuxBuffers;
            mAuxBuffers = (auxBufferCount == 0) ? NULL : new NonConstBufferType[auxBufferCount];
            mAuxBufferCount = auxBufferCount;
        }
        this->syncAllBuffers(serial);
    }

    void cook(size_t grainsize)
    {
        if (grainsize>0) {
            tbb::parallel_for(this->getRange(grainsize), *this);
        } else {
            (*this)(this->getRange());
        }
    }

    void doSwapLeafBuffer(const RangeType& r, size_t auxBufferIdx)
    {
        LeafManagerImpl<LeafManager>::doSwapLeafBuffer(
            r, auxBufferIdx, mLeafs, mAuxBuffers, mAuxBuffersPerLeaf);
    }

    void doSwapAuxBuffer(const RangeType& r, size_t auxBufferIdx1, size_t auxBufferIdx2)
    {
        for (size_t N = mAuxBuffersPerLeaf, n = N*r.begin(), m = N*r.end(); n != m; n+=N) {
            mAuxBuffers[n + auxBufferIdx1].swap(mAuxBuffers[n + auxBufferIdx2]);
        }
    }

    void doSyncAuxBuffer(const RangeType& r, size_t auxBufferIdx)
    {
        for (size_t n = r.begin(), m = r.end(), N = mAuxBuffersPerLeaf; n != m; ++n) {
            mAuxBuffers[n*N + auxBufferIdx] = mLeafs[n]->buffer();
        }
    }

    void doSyncAllBuffers1(const RangeType& r)
    {
        for (size_t n = r.begin(), m = r.end(); n != m; ++n) {
            mAuxBuffers[n] = mLeafs[n]->buffer();
        }
    }

    void doSyncAllBuffers2(const RangeType& r)
    {
        for (size_t n = r.begin(), m = r.end(); n != m; ++n) {
            const BufferType& leafBuffer = mLeafs[n]->buffer();
            mAuxBuffers[2*n  ] = leafBuffer;
            mAuxBuffers[2*n+1] = leafBuffer;
        }
    }

    void doSyncAllBuffersN(const RangeType& r)
    {
        for (size_t n = r.begin(), m = r.end(), N = mAuxBuffersPerLeaf; n != m; ++n) {
            const BufferType& leafBuffer = mLeafs[n]->buffer();
            for (size_t i=n*N, j=i+N; i!=j; ++i) mAuxBuffers[i] = leafBuffer;
        }
    }

    /// @brief Private member class that applies a user-defined
    /// functor to all the leaf nodes.
    template<typename LeafOp>
    struct LeafTransformer
    {
        LeafTransformer(const LeafOp& leafOp) : mLeafOp(leafOp) {}
        void run(const LeafRange& range, bool threaded = true)
        {
            threaded ? tbb::parallel_for(range, *this) : (*this)(range);
        }
        void operator()(const LeafRange& range) const
        {
            for (typename LeafRange::Iterator it = range.begin(); it; ++it) mLeafOp(*it, it.pos());
        }
        const LeafOp mLeafOp;
    };

    typedef typename boost::function<void (LeafManager*, const RangeType&)> FuncType;

    TreeType*           mTree;
    size_t              mLeafCount, mAuxBufferCount, mAuxBuffersPerLeaf;
    LeafType**          mLeafs;//array of LeafNode pointers
    NonConstBufferType* mAuxBuffers;//array of auxiliary buffers
    FuncType            mTask;
    const bool          mIsMaster;
};//end of LeafManager class


// Partial specializations of LeafManager methods for const trees
template<typename TreeT>
struct LeafManagerImpl<LeafManager<const TreeT> >
{
    typedef LeafManager<const TreeT> ManagerT;
    typedef typename ManagerT::RangeType      RangeT;
    typedef typename ManagerT::LeafType       LeafT;
    typedef typename ManagerT::BufferType     BufT;

    static inline void doSwapLeafBuffer(const RangeT&, size_t /*auxBufferIdx*/,
        LeafT**, BufT*, size_t /*bufsPerLeaf*/)
    {
        // Buffers can't be swapped into const trees.
    }
};

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_LEAFMANAGER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
