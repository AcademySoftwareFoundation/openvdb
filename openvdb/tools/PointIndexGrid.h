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
/// @file   PointIndexGrid.h
///
/// @brief  Space-partitioning acceleration structure for points. Partitions
///         the points into voxels to accelerate range and nearest neighbor
///         searches.
///
/// @note   Leaf nodes store a single point-index array and the voxels are only
///         integer offsets into that array. The actual points are never stored
///         in the acceleration structure, only offsets into an external array.
///
/// @author Mihai Alden

#ifndef OPENVDB_TOOLS_POINT_INDEX_GRID_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_INDEX_GRID_HAS_BEEN_INCLUDED


#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tree/LeafManager.h>
#include "PointPartitioner.h"

#include <boost/scoped_array.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/atomic.h>
#include <iostream>
#include <deque>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

// forward declaration
namespace tree {
    template<Index, typename> struct SameLeafConfig;
}

namespace tools {

template<typename T, Index Log2Dim> struct PointIndexLeafNode; // forward declaration

/// @brief Point index tree configured to match the default VDB configurations.
typedef tree::Tree<tree::RootNode<tree::InternalNode<tree::InternalNode
    <PointIndexLeafNode<PointIndex32, 3>, 4>, 5> > > PointIndexTree;


/// @brief Point index grid.
typedef Grid<PointIndexTree> PointIndexGrid;


////////////////////////////////////////


// Expected interface for the PointArray container:
//
// template<typename VectorType>
// struct PointList {
//   typedef VectorType value_type;
//   size_t size() const;
//   void getPos(size_t n, VectorType& xyz) const;
// };


////////////////////////////////////////


/// @brief  Partitions points into a point index grid to accelerate range and
///         nearest neighbor searches.
///
/// @param points   list of world space points.
/// @param xform    world to index space transform.
template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
createPointIndexGrid(const PointArrayT& points, const math::Transform& xform);


/// @brief  Checks if the partitioning is still valid, returns false if the given
///         grid does not represent a valid partitioning of the given point list.
///
/// @param points   list of world space points.
/// @param grid     point index grid to validate.
template<typename PointArrayT, typename GridT>
inline bool
isValidPartition(const PointArrayT& points, const GridT& grid);


/// @brief  Re-partitions the @a points if needed, otherwise the input @a grid is returned.
template<typename GridT, typename PointArrayT>
inline typename GridT::ConstPtr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::ConstPtr& grid);


/// @brief  Re-partitions the @a points if needed, otherwise the input @a grid is returned.
template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::Ptr& grid);


////////////////////////////////////////


/// @brief  Accelerated range and nearest neighbor searches for point index grids.
template<typename TreeType = PointIndexTree>
struct PointIndexIterator
{
    typedef tree::ValueAccessor<const TreeType> ConstAccessor;
    typedef typename TreeType::LeafNodeType     LeafNodeType;
    typedef typename TreeType::ValueType        ValueType;


    PointIndexIterator();
    PointIndexIterator(const PointIndexIterator& rhs);
    PointIndexIterator& operator=(const PointIndexIterator& rhs);


    /// @brief Construct an iterator over the indices of the points contained in voxel (i, j, k).
    /// @param ijk  the voxel containing the points over which to iterate
    /// @param acc  an accessor for the grid or tree that holds the point indices
    PointIndexIterator(const Coord& ijk, ConstAccessor& acc);


    /// @brief Construct an iterator over the indices of the points contained in the given bounding box.
    /// @param bbox  the bounding box of the voxels containing the points over which to iterate
    /// @param acc   an accessor for the grid or tree that holds the point indices
    /// @note        The range of the @a bbox is inclusive. Thus, a bounding box with
    ///              min = max is not empty but rather encloses a single voxel.
    PointIndexIterator(const CoordBBox& bbox, ConstAccessor& acc);


    /// @todo Floating-point range query.
    //template<typename PointArray>
    //static PointIndexIterator getPointRange(const BBoxd& bbox, ConstAccessor&,
    //    const PointArray& points, const math::Transform&);

    /// @todo Radial query.
    //template<typename PointArray>
    //static PointIndexIterator getPointRange(const Vec3d& xyz, double radius, ConstAccessor&,
    //    const PointArray& points, const math::Transform&);


    /// @brief  Reset the iterator to point to the first item.
    void reset();

    /// @brief  Returns a const reference to the item to which
    ///         this iterator is pointing.
    const ValueType& operator*() const { return *mRange.first; }

    /// @{
    /// @brief  Returns @c true if this iterator is not yet exhausted.
    bool test() const { return mRange.first < mRange.second && mIter != mRangeList.end(); }
    operator bool() const { return this->test(); }
    /// @}

    /// @brief  Advance iterator to next item.
    void increment();

    /// @brief  Advance iterator to next item.
    void operator++() { this->increment(); }


    /// @brief  Advance iterator to next item.
    /// @return @c true if this iterator is not yet exhausted.
    bool next();

    /// @brief  Number of point indices in the iterator range.
    size_t size() const;

    /// @brief Equality operator, returns true if both iterators point to the same element.
    bool operator==(const PointIndexIterator& other) const {return mRange.first == other.mRange.first;}

    /// @brief Inequality operator.
    bool operator!=(const PointIndexIterator& other) const { return !this->operator==(other); }


private:

    void get(const LeafNodeType&, const Coord& min, const Coord& max);

    typedef std::pair<const ValueType*, const ValueType*> Range;
    typedef std::deque<Range> RangeList;

    Range mRange;
    RangeList mRangeList;
    typename RangeList::const_iterator mIter;
}; // struct PointIndexIterator


template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator()
    : mRange(static_cast<ValueType*>(NULL), static_cast<ValueType*>(NULL))
    , mRangeList()
    , mIter(mRangeList.begin())
{
}


template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator(const PointIndexIterator& rhs)
    : mRange(rhs.mRange)
    , mRangeList(rhs.mRangeList)
    , mIter(mRangeList.begin())
{
}


template<typename TreeType>
inline PointIndexIterator<TreeType>&
PointIndexIterator<TreeType>::operator=(const PointIndexIterator& rhs)
{
    if (&rhs != this) {
        mRange = rhs.mRange;
        mRangeList = rhs.mRangeList;
        mIter = mRangeList.begin();
    }
}


template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator(const Coord& ijk, ConstAccessor& acc)
    : mRange(static_cast<ValueType*>(NULL), static_cast<ValueType*>(NULL))
    , mRangeList()
    , mIter(mRangeList.begin())
{
    const LeafNodeType* leaf = acc.probeConstLeaf(ijk);
    if (leaf && leaf->getIndices(ijk, mRange.first, mRange.second)) {
        mRangeList.push_back(mRange);
        mIter = mRangeList.begin();
    }
}


template<typename TreeType>
inline
PointIndexIterator<TreeType>::PointIndexIterator(const CoordBBox& bbox, ConstAccessor& acc)
    : mRange(static_cast<ValueType*>(NULL), static_cast<ValueType*>(NULL))
    , mRangeList()
    , mIter(mRangeList.begin())
{
    Coord ijk, ijkMax, ijkA, ijkB;
    const Coord leafMin = bbox.min() & ~(LeafNodeType::DIM - 1);
    const Coord leafMax = bbox.max() & ~(LeafNodeType::DIM - 1);

    for (ijk[0] = leafMin[0]; ijk[0] <= leafMax[0]; ijk[0] += LeafNodeType::DIM) {
        for (ijk[1] = leafMin[1]; ijk[1] <= leafMax[1]; ijk[1] += LeafNodeType::DIM) {
            for (ijk[2] = leafMin[2]; ijk[2] <= leafMax[2]; ijk[2] += LeafNodeType::DIM) {

                if (const LeafNodeType* leaf = acc.probeConstLeaf(ijk)) {
                    ijkMax = ijk;
                    ijkMax.offset(LeafNodeType::DIM - 1);

                    // intersect leaf bbox with search region.
                    ijkA = Coord::maxComponent(bbox.min(), ijk);
                    ijkB = Coord::minComponent(bbox.max(), ijkMax);

                    if (ijkA != ijk || ijkB != ijkMax) {
                        get(*leaf, ijkA, ijkB);
                    } else {
                        // leaf bbox is inside the search region, add all indices.
                        const ValueType* begin = &leaf->indices().front();
                        mRangeList.push_back(Range(begin, (begin + leaf->indices().size())));
                    }
                }
            }
        }
    }

    if (!mRangeList.empty()) {
        mIter = mRangeList.begin();
        mRange = mRangeList.front();
    }
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::get(const LeafNodeType& leaf, const Coord& min, const Coord& max)
{
    typedef typename LeafNodeType::ValueType PointIndexT;
    Index xPos, pos, zStride = Index(max[2] - min[2]);
    Coord ijk;

    const ValueType* dataPtr = &leaf.indices().front();
    PointIndexT beginOffset, endOffset,
        previousOffset = PointIndexT(leaf.indices().size() + size_t(1));
    Range range;

    for (ijk[0] = min[0]; ijk[0] <= max[0]; ++ijk[0]) {
        xPos = (ijk[0] & (LeafNodeType::DIM - 1u)) << (2 * LeafNodeType::LOG2DIM);

        for (ijk[1] = min[1]; ijk[1] <= max[1]; ++ijk[1]) {
            pos = xPos + ((ijk[1] & (LeafNodeType::DIM - 1u)) << LeafNodeType::LOG2DIM);
            pos += (min[2] & (LeafNodeType::DIM - 1u));

            beginOffset = (pos == 0 ? PointIndexT(0) : leaf.getValue(pos - 1));
            endOffset = leaf.getValue(pos+zStride);

            if (endOffset > beginOffset) {

                if (beginOffset == previousOffset) {
                    mRangeList.back().second = dataPtr + endOffset;
                } else {
                    mRangeList.push_back(Range(dataPtr + beginOffset, dataPtr + endOffset));
                }

                previousOffset = endOffset;
            }
        }
    }
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::reset()
{
    mIter = mRangeList.begin();
    if (!mRangeList.empty()) {
        mRange = mRangeList.front();
    } else {
        mRange = Range(static_cast<ValueType*>(NULL), static_cast<ValueType*>(NULL));
    }
}


template<typename TreeType>
inline void
PointIndexIterator<TreeType>::increment()
{
    ++mRange.first;
    if (mRange.first >= mRange.second && mIter != mRangeList.end()) {
        ++mIter;
        if (mIter != mRangeList.end()) {
            mRange = *mIter;
        }
    }
}


template<typename TreeType>
inline bool
PointIndexIterator<TreeType>::next()
{
    if (!this->test()) return false;
    this->increment();
    return this->test();
}


template<typename TreeType>
inline size_t
PointIndexIterator<TreeType>::size() const
{
    size_t count = 0;
    typename RangeList::const_iterator it = mRangeList.begin();

    for ( ; it != mRangeList.end(); ++it) {
        count += it->second - it->first;
    }

    return count;
}


////////////////////////////////////////

// Internal operators

namespace point_index_grid_internal {

template<typename PointArrayT>
struct ValidPartitioningOp
{
    ValidPartitioningOp(tbb::atomic<bool>& hasChanged,
        const PointArrayT& points, const math::Transform& xform)
        : mPoints(&points)
        , mTransform(&xform)
        , mHasChanged(&hasChanged)
    {
    }

    template <typename LeafT>
    void operator()(LeafT &leaf, size_t /*leafIndex*/) const
    {
        if ((*mHasChanged)) {
            tbb::task::self().cancel_group_execution();
            return;
        }

        typedef typename LeafT::IndexArray          IndexArrayT;
        typedef typename IndexArrayT::value_type    IndexT;
        typedef typename PointArrayT::value_type    PointT;

        typename LeafT::ValueOnCIter iter;
        Coord voxelCoord;
        PointT point;

        const IndexT *begin = NULL, *end = NULL;

        for (iter = leaf.cbeginValueOn(); iter; ++iter) {

            if ((*mHasChanged)) break;

            voxelCoord = iter.getCoord();
            leaf.getIndices(iter.pos(), begin, end);

            while (begin < end) {

                mPoints->getPos(*begin, point);
                if (voxelCoord != mTransform->worldToIndexCellCentered(point)) {
                    mHasChanged->fetch_and_store(true);
                    break;
                }

                ++begin;
            }
        }
    }

private:
    PointArrayT         const * const mPoints;
    math::Transform     const * const mTransform;
    tbb::atomic<bool>         * const mHasChanged;
};


template<typename LeafNodeT>
struct PopulateLeafNodesOp
{
    typedef uint32_t IndexT;
    typedef PointPartitioner<IndexT, LeafNodeT::LOG2DIM> Partitioner;

    PopulateLeafNodesOp(boost::scoped_array<LeafNodeT*>& leafNodes,
        const Partitioner& partitioner)
        : mLeafNodes(leafNodes.get())
        , mPartitioner(&partitioner)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        typedef typename Partitioner::VoxelOffsetType VoxelOffsetT;

        size_t maxPointCount = 0;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            maxPointCount = std::max(maxPointCount, mPartitioner->indices(n).size());
        }

        const IndexT voxelCount = LeafNodeT::SIZE;

        // allocate histogram buffers
        boost::scoped_array<VoxelOffsetT> offsets(new VoxelOffsetT[maxPointCount]);
        boost::scoped_array<IndexT> histogram(new IndexT[voxelCount]);

        VoxelOffsetT const * const voxelOffsets = mPartitioner->voxelOffsets().get();

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            LeafNodeT* node = new LeafNodeT();
            node->setOrigin(mPartitioner->origin(n));

            typename Partitioner::IndexIterator it = mPartitioner->indices(n);

            const size_t pointCount = it.size();
            IndexT const * const indices = &*it;

            // local copy of voxel offsets.
            for (IndexT i = 0; i < pointCount; ++i) {
                offsets[i] = voxelOffsets[ indices[i] ];
            }

            // compute voxel-offset histogram
            memset(&histogram[0], 0, voxelCount * sizeof(IndexT));
            for (IndexT i = 0; i < pointCount; ++i) {
                ++histogram[ offsets[i] ];
            }

            typename LeafNodeT::NodeMaskType& mask = node->getValueMask();
            typename LeafNodeT::Buffer& buffer = node->buffer();

            // scan histogram (all-prefix-sums)
            IndexT count = 0, startOffset;
            for (int i = 0; i < int(voxelCount); ++i) {
                if (histogram[i] > 0) {
                    startOffset = count;
                    count += histogram[i];
                    histogram[i] = startOffset;
                    mask.setOn(i);
                }
                buffer.setValue(i, count);
            }

            // allocate point-index array
            node->indices().resize(pointCount);
            typename LeafNodeT::ValueType * const orderedIndices = node->indices().data();

            // rank and permute
            for (IndexT i = 0; i < pointCount; ++i) {
                orderedIndices[ histogram[ offsets[i] ]++ ] = indices[i];
            }

            mLeafNodes[n] = node;
        }
    }

    //////////

    LeafNodeT*        * const mLeafNodes;
    Partitioner const * const mPartitioner;
};


/// @brief  Constructs a @c PointIndexTree
template<typename TreeType, typename PointArray>
inline void
constructPointTree(TreeType& tree, const math::Transform& xform, const PointArray& points)
{
    typedef typename TreeType::LeafNodeType LeafType;

    boost::scoped_array<LeafType*> leafNodes;
    size_t leafNodeCount = 0;

    {
        PointPartitioner<uint32_t, LeafType::LOG2DIM> partitioner;
        partitioner.construct(points, xform, /*voxelOrder=*/false, /*recordVoxelOffsets=*/true);

        leafNodeCount = partitioner.size();
        leafNodes.reset(new LeafType*[leafNodeCount]);

        const tbb::blocked_range<size_t> range(0, leafNodeCount);
        tbb::parallel_for(range, PopulateLeafNodesOp<LeafType>(leafNodes, partitioner));
    }

    tree::ValueAccessor<TreeType> acc(tree);
    for (size_t n = 0; n < leafNodeCount; ++n) {
        acc.addLeaf(leafNodes[n]);
    }
}


} // namespace point_index_grid_internal


////////////////////////////////////////

template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
createPointIndexGrid(const PointArrayT& points, const math::Transform& xform)
{
    typename GridT::Ptr grid = GridT::create(typename GridT::ValueType(0));
    grid->setTransform(xform.copy());

    if (points.size() > 0) {
        point_index_grid_internal::constructPointTree(
            grid->tree(), grid->transform(), points);
    }

    return grid;
}


template<typename PointArrayT, typename GridT>
inline bool
isValidPartition(const PointArrayT& points, const GridT& grid)
{
    tree::LeafManager<const typename GridT::TreeType> leafs(grid.tree());

    size_t pointCount = 0;
    for (size_t n = 0, N = leafs.leafCount(); n < N; ++n) {
        pointCount += leafs.leaf(n).indices().size();
    }

    if (points.size() != pointCount) {
        return false;
    }

    tbb::atomic<bool> changed;
    changed = false;

    point_index_grid_internal::ValidPartitioningOp<PointArrayT>
        op(changed, points, grid.transform());

    leafs.foreach(op);

    return !bool(changed);
}


template<typename GridT, typename PointArrayT>
inline typename GridT::ConstPtr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::ConstPtr& grid)
{
    if (isValidPartition(points, *grid)) {
        return grid;
    }

    return createPointIndexGrid<GridT>(points, grid->transform());
}


template<typename GridT, typename PointArrayT>
inline typename GridT::Ptr
getValidPointIndexGrid(const PointArrayT& points, const typename GridT::Ptr& grid)
{
    if (isValidPartition(points, *grid)) {
        return grid;
    }

    return createPointIndexGrid<GridT>(points, grid->transform());
}


////////////////////////////////////////


template<typename T, Index Log2Dim>
struct PointIndexLeafNode : public tree::LeafNode<T, Log2Dim>
{
    typedef PointIndexLeafNode<T, Log2Dim>          LeafNodeType;
    typedef boost::shared_ptr<PointIndexLeafNode>   Ptr;

    typedef T                       ValueType;
    typedef std::vector<ValueType>  IndexArray;


    IndexArray& indices() { return mIndices; }
    const IndexArray& indices() const { return mIndices; }

    bool getIndices(const Coord& ijk, const ValueType*& begin, const ValueType*& end) const;
    bool getIndices(Index offset, const ValueType*& begin, const ValueType*& end) const;

    void setOffsetOn(Index offset, const ValueType& val);
    void setOffsetOnly(Index offset, const ValueType& val);

    bool isEmpty(const CoordBBox& bbox) const;

private:
    IndexArray mIndices;

    ////////////////////////////////////////

    // The following methods had to be copied from the LeafNode class
    // to make the derived PointIndexLeafNode class compatible with the tree structure.

public:
    typedef tree::LeafNode<T, Log2Dim>  BaseLeaf;
    typedef util::NodeMask<Log2Dim>     NodeMaskType;

    using BaseLeaf::LOG2DIM;
    using BaseLeaf::TOTAL;
    using BaseLeaf::DIM;
    using BaseLeaf::NUM_VALUES;
    using BaseLeaf::NUM_VOXELS;
    using BaseLeaf::SIZE;
    using BaseLeaf::LEVEL;

    /// Default constructor
    PointIndexLeafNode() : BaseLeaf(), mIndices() {}

    explicit
    PointIndexLeafNode(const Coord& coords, const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(coords, value, active)
        , mIndices()
    {
    }

#ifndef OPENVDB_2_ABI_COMPATIBLE
    PointIndexLeafNode(PartialCreate, const Coord& coords,
        const T& value = zeroVal<T>(), bool active = false)
        : BaseLeaf(PartialCreate(), coords, value, active)
        , mIndices()
    {
    }
#endif

    /// Deep copy constructor
    PointIndexLeafNode(const PointIndexLeafNode& rhs) : BaseLeaf(rhs), mIndices(rhs.mIndices) {}

    /// @brief Return @c true if the given node (which may have a different @c ValueType
    /// than this node) has the same active value topology as this node.
    template<typename OtherType, Index OtherLog2Dim>
    bool hasSameTopology(const PointIndexLeafNode<OtherType, OtherLog2Dim>* other) const {
        return BaseLeaf::hasSameTopology(other);
    }

    /// Check for buffer, state and origin equivalence.
    bool operator==(const PointIndexLeafNode& other) const { return BaseLeaf::operator==(other); }

    bool operator!=(const PointIndexLeafNode& other) const { return !(other == *this); }

    template<MergePolicy Policy> void merge(const PointIndexLeafNode& rhs) {
        BaseLeaf::merge<Policy>(rhs);
    }
    template<MergePolicy Policy> void merge(const ValueType& tileValue, bool tileActive) {
         BaseLeaf::template merge<Policy>(tileValue, tileActive);
    }

    template<MergePolicy Policy>
    void merge(const PointIndexLeafNode& other,
        const ValueType& /*bg*/, const ValueType& /*otherBG*/)
    {
         BaseLeaf::template merge<Policy>(other);
    }

    void addLeaf(PointIndexLeafNode*) {}
    template<typename AccessorT>
    void addLeafAndCache(PointIndexLeafNode*, AccessorT&) {}

    //@{
    /// @brief Return a pointer to this node.
    PointIndexLeafNode* touchLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointIndexLeafNode* touchLeafAndCache(const Coord&, AccessorT&) { return this; }

    template<typename NodeT, typename AccessorT>
    NodeT* probeNodeAndCache(const Coord&, AccessorT&)
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(boost::is_same<NodeT,PointIndexLeafNode>::value)) return NULL;
        return reinterpret_cast<NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    PointIndexLeafNode* probeLeaf(const Coord&) { return this; }
    template<typename AccessorT>
    PointIndexLeafNode* probeLeafAndCache(const Coord&, AccessorT&) { return this; }
    //@}

    //@{
    /// @brief Return a @const pointer to this node.
    const PointIndexLeafNode* probeConstLeaf(const Coord&) const { return this; }
    template<typename AccessorT>
    const PointIndexLeafNode* probeConstLeafAndCache(const Coord&, AccessorT&) const {return this;}
    template<typename AccessorT>
    const PointIndexLeafNode* probeLeafAndCache(const Coord&, AccessorT&) const { return this; }
    const PointIndexLeafNode* probeLeaf(const Coord&) const { return this; }
    template<typename NodeT, typename AccessorT>
    const NodeT* probeConstNodeAndCache(const Coord&, AccessorT&) const
    {
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
        if (!(boost::is_same<NodeT,PointIndexLeafNode>::value)) return NULL;
        return reinterpret_cast<const NodeT*>(this);
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
    }
    //@}


    // I/O methods

    void readBuffers(std::istream& is, bool fromHalf = false);
    void readBuffers(std::istream& is, const CoordBBox&, bool fromHalf = false);
    void writeBuffers(std::ostream& os, bool toHalf = false) const;


    Index64 memUsage() const;


    ////////////////////////////////////////

    // Disable all write methods to avoid unintentional changes
    // to the point-array offsets.

    void assertNonmodifiable() {
        assert(false && "Cannot modify voxel values in a PointIndexTree.");
    }

    void setActiveState(const Coord&, bool) { assertNonmodifiable(); }
    void setActiveState(Index, bool) { assertNonmodifiable(); }

    void setValueOnly(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOnly(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOff(const Coord&) { assertNonmodifiable(); }
    void setValueOff(Index) { assertNonmodifiable(); }

    void setValueOff(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOff(Index, const ValueType&) { assertNonmodifiable(); }

    void setValueOn(const Coord&) { assertNonmodifiable(); }
    void setValueOn(Index offset) { assertNonmodifiable(); }

    void setValueOn(const Coord&, const ValueType&) { assertNonmodifiable(); }
    void setValueOn(Index, const ValueType&) { assertNonmodifiable(); }

    void setValue(const Coord&, const ValueType&) { assertNonmodifiable(); }

    void setValuesOn() { assertNonmodifiable(); }
    void setValuesOff() { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValue(Index, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValue(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    template<typename ModifyOp>
    void modifyValueAndActiveState(const Coord&, const ModifyOp&) { assertNonmodifiable(); }

    void clip(const CoordBBox&, const ValueType&) { assertNonmodifiable(); }

    void fill(const CoordBBox&, const ValueType&, bool) { assertNonmodifiable(); }
    void fill(const ValueType&) {}
    void fill(const ValueType&, bool) { assertNonmodifiable(); }

    template<typename AccessorT>
    void setValueOnlyAndCache(const Coord&, const ValueType&, AccessorT&) {assertNonmodifiable();}

    template<typename ModifyOp, typename AccessorT>
    void modifyValueAndActiveStateAndCache(const Coord&, const ModifyOp&, AccessorT&) {
        assertNonmodifiable();
    }

    template<typename AccessorT>
    void setValueOffAndCache(const Coord&, const ValueType&, AccessorT&) { assertNonmodifiable(); }

    template<typename AccessorT>
    void setActiveStateAndCache(const Coord&, bool, AccessorT&) { assertNonmodifiable(); }

    void resetBackground(const ValueType&, const ValueType&) { assertNonmodifiable(); }

    void signedFloodFill(const ValueType&) { assertNonmodifiable(); }
    void signedFloodFill(const ValueType&, const ValueType&) { assertNonmodifiable(); }

    void negate() { assertNonmodifiable(); }

protected:
    typedef typename BaseLeaf::ValueOn ValueOn;
    typedef typename BaseLeaf::ValueOff ValueOff;
    typedef typename BaseLeaf::ValueAll ValueAll;
    typedef typename BaseLeaf::ChildOn ChildOn;
    typedef typename BaseLeaf::ChildOff ChildOff;
    typedef typename BaseLeaf::ChildAll ChildAll;

    typedef typename NodeMaskType::OnIterator    MaskOnIterator;
    typedef typename NodeMaskType::OffIterator   MaskOffIterator;
    typedef typename NodeMaskType::DenseIterator MaskDenseIterator;

    // During topology-only construction, access is needed
    // to protected/private members of other template instances.
    template<typename, Index> friend struct PointIndexLeafNode;

    friend class tree::IteratorBase<MaskOnIterator, PointIndexLeafNode>;
    friend class tree::IteratorBase<MaskOffIterator, PointIndexLeafNode>;
    friend class tree::IteratorBase<MaskDenseIterator, PointIndexLeafNode>;

public:


    typedef typename BaseLeaf::template ValueIter<
        MaskOnIterator, PointIndexLeafNode, const ValueType, ValueOn> ValueOnIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskOnIterator, const PointIndexLeafNode, const ValueType, ValueOn> ValueOnCIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskOffIterator, PointIndexLeafNode, const ValueType, ValueOff> ValueOffIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskOffIterator,const PointIndexLeafNode,const ValueType,ValueOff> ValueOffCIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskDenseIterator, PointIndexLeafNode, const ValueType, ValueAll> ValueAllIter;
    typedef typename BaseLeaf::template ValueIter<
        MaskDenseIterator,const PointIndexLeafNode,const ValueType,ValueAll> ValueAllCIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOnIterator, PointIndexLeafNode, ChildOn> ChildOnIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOnIterator, const PointIndexLeafNode, ChildOn> ChildOnCIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOffIterator, PointIndexLeafNode, ChildOff> ChildOffIter;
    typedef typename BaseLeaf::template ChildIter<
        MaskOffIterator, const PointIndexLeafNode, ChildOff> ChildOffCIter;
    typedef typename BaseLeaf::template DenseIter<
        PointIndexLeafNode, ValueType, ChildAll> ChildAllIter;
    typedef typename BaseLeaf::template DenseIter<
        const PointIndexLeafNode, const ValueType, ChildAll> ChildAllCIter;

#define VMASK_ this->getValueMask()
    ValueOnCIter  cbeginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnCIter   beginValueOn() const  { return ValueOnCIter(VMASK_.beginOn(), this); }
    ValueOnIter    beginValueOn()        { return ValueOnIter(VMASK_.beginOn(), this); }
    ValueOffCIter cbeginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffCIter  beginValueOff() const { return ValueOffCIter(VMASK_.beginOff(), this); }
    ValueOffIter   beginValueOff()       { return ValueOffIter(VMASK_.beginOff(), this); }
    ValueAllCIter cbeginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllCIter  beginValueAll() const { return ValueAllCIter(VMASK_.beginDense(), this); }
    ValueAllIter   beginValueAll()       { return ValueAllIter(VMASK_.beginDense(), this); }

    ValueOnCIter  cendValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnCIter   endValueOn() const    { return ValueOnCIter(VMASK_.endOn(), this); }
    ValueOnIter    endValueOn()          { return ValueOnIter(VMASK_.endOn(), this); }
    ValueOffCIter cendValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffCIter  endValueOff() const   { return ValueOffCIter(VMASK_.endOff(), this); }
    ValueOffIter   endValueOff()         { return ValueOffIter(VMASK_.endOff(), this); }
    ValueAllCIter cendValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllCIter  endValueAll() const   { return ValueAllCIter(VMASK_.endDense(), this); }
    ValueAllIter   endValueAll()         { return ValueAllIter(VMASK_.endDense(), this); }

    ChildOnCIter  cbeginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   beginChildOn() const  { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    beginChildOn()        { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cbeginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  beginChildOff() const { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   beginChildOff()       { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cbeginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllCIter  beginChildAll() const { return ChildAllCIter(VMASK_.beginDense(), this); }
    ChildAllIter   beginChildAll()       { return ChildAllIter(VMASK_.beginDense(), this); }

    ChildOnCIter  cendChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnCIter   endChildOn() const    { return ChildOnCIter(VMASK_.endOn(), this); }
    ChildOnIter    endChildOn()          { return ChildOnIter(VMASK_.endOn(), this); }
    ChildOffCIter cendChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffCIter  endChildOff() const   { return ChildOffCIter(VMASK_.endOff(), this); }
    ChildOffIter   endChildOff()         { return ChildOffIter(VMASK_.endOff(), this); }
    ChildAllCIter cendChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllCIter  endChildAll() const   { return ChildAllCIter(VMASK_.endDense(), this); }
    ChildAllIter   endChildAll()         { return ChildAllIter(VMASK_.endDense(), this); }
#undef VMASK_
}; // struct PointIndexLeafNode


template<typename T, Index Log2Dim>
inline bool
PointIndexLeafNode<T, Log2Dim>::getIndices(const Coord& ijk, const ValueType*& begin, const ValueType*& end) const
{
    return getIndices(LeafNodeType::coordToOffset(ijk), begin, end);
}


template<typename T, Index Log2Dim>
inline bool
PointIndexLeafNode<T, Log2Dim>::getIndices(Index offset, const ValueType*& begin, const ValueType*& end) const
{
    if (this->isValueMaskOn(offset)) {
        const ValueType* dataPtr = &mIndices.front();
        begin = dataPtr + (offset == 0 ? ValueType(0) : this->buffer()[offset - 1]);
        end = dataPtr + this->buffer()[offset];
        return true;
    }
    return false;
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::setOffsetOn(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
    this->setValueMaskOn(offset);
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::setOffsetOnly(Index offset, const ValueType& val)
{
    this->buffer().setValue(offset, val);
}


template<typename T, Index Log2Dim>
inline bool
PointIndexLeafNode<T, Log2Dim>::isEmpty(const CoordBBox& bbox) const
{
    Index xPos, pos, zStride = Index(bbox.max()[2] - bbox.min()[2]);
    Coord ijk;

    for (ijk[0] = bbox.min()[0]; ijk[0] <= bbox.max()[0]; ++ijk[0]) {
        xPos = (ijk[0] & (DIM - 1u)) << (2 * LOG2DIM);

        for (ijk[1] = bbox.min()[1]; ijk[1] <= bbox.max()[1]; ++ijk[1]) {
            pos = xPos + ((ijk[1] & (DIM - 1u)) << LOG2DIM);
            pos += (bbox.min()[2] & (DIM - 1u));

            if (this->buffer()[pos+zStride] > (pos == 0 ? T(0) : this->buffer()[pos - 1])) {
                return false;
            }
        }
    }

    return true;
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    BaseLeaf::readBuffers(is, fromHalf);

    Index64 numIndices = Index64(0);
    is.read(reinterpret_cast<char*>(&numIndices), sizeof(Index64));

    mIndices.resize(size_t(numIndices));
    is.read(reinterpret_cast<char*>(mIndices.data()), numIndices * sizeof(T));
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& bbox, bool fromHalf)
{
    // Read and clip voxel values.
    BaseLeaf::readBuffers(is, bbox, fromHalf);

    Index64 numIndices = Index64(0);
    is.read(reinterpret_cast<char*>(&numIndices), sizeof(Index64));

    const Index64 numBytes = numIndices * sizeof(T);

    if (bbox.hasOverlap(this->getNodeBoundingBox())) {
        mIndices.resize(size_t(numIndices));
        is.read(reinterpret_cast<char*>(mIndices.data()), numBytes);

        /// @todo If any voxels were deactivated as a result of clipping in the call to
        /// BaseLeaf::readBuffers(), the point index list will need to be regenerated.
    } else {
        // Read and discard voxel values.
        boost::scoped_array<char> buf(new char[numBytes]);
        is.read(buf.get(), numBytes);
    }
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    BaseLeaf::writeBuffers(os, toHalf);

    Index64 numIndices = Index64(mIndices.size());
    os.write(reinterpret_cast<const char*>(&numIndices), sizeof(Index64));
    os.write(reinterpret_cast<const char*>(mIndices.data()), numIndices * sizeof(T));
}


template<typename T, Index Log2Dim>
inline Index64
PointIndexLeafNode<T, Log2Dim>::memUsage() const
{
    return BaseLeaf::memUsage() + Index64((sizeof(T)*mIndices.capacity()) + sizeof(mIndices));
}


////////////////////////////////////////


} // namespace tools


namespace tree {

/// Helper metafunction used to implement LeafNode::SameConfiguration
/// (which, as an inner class, can't be independently specialized)
template<Index Dim1, typename T2>
struct SameLeafConfig<Dim1, openvdb::tools::PointIndexLeafNode<T2, Dim1> >
{
    static const bool value = true;
};

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_INDEX_GRID_HAS_BEEN_INCLUDED


// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
