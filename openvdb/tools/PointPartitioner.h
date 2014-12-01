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
/// @file   PointPartitioner.h
///
/// @brief  Multi-threaded space-partitioning scheme for points.
///
/// @note   This tool is deterministic; partitioning the same point
///         sequence will produce the same result each time.
///         The actual points are never stored in the tool, only
///         offsets into an external array.
///
/// @author Mihai Alden

#ifndef OPENVDB_TOOLS_POINT_PARTITIONER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_PARTITIONER_HAS_BEEN_INCLUDED


#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>

#include <vector>
#include <deque>
#include <algorithm> // std::swap

#include <boost/scoped_array.hpp>
#include <boost/integer.hpp> // boost::int_t<N>::least

#include <tbb/atomic.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


// Expected interface for the PointArray container:
//
// template<typename VectorType>
// struct PointList {
//   typedef VectorType position_type;
//   size_t size() const;
//   void getPos(size_t n, VectorType& xyz) const;
// };


/// @brief  Partitions points into @c Log2Dim aligned buckets.
///
/// @note   This tool is deterministic; partitioning the same point
///         sequence will produce the same result each time.
///         @c Log2Dim defines the bucket coordinate dimensions,
///         i.e. Log2Dim = 3 corresponds to a bucket that spans
///         a (2^3)^3 = 8^3 voxel region.
template<typename PointIndexT = uint32_t, Index Log2Dim = 3>
class PointPartitioner
{
public:
    enum { LOG2DIM = Log2Dim };

    typedef boost::shared_ptr<PointPartitioner>                 Ptr;
    typedef boost::shared_ptr<const PointPartitioner>           ConstPtr;

    typedef PointIndexT                                         IndexType;
    typedef typename boost::int_t<1 + (3 * Log2Dim)>::least     VoxelOffsetType;
    typedef boost::scoped_array<VoxelOffsetType>                VoxelOffsetArray;

    class IndexIterator;

    //////////

    PointPartitioner();

    /// @brief  Partitions point indices into @c Log2Dim aligned buckets.
    ///
    /// @param points               list of world space points.
    /// @param xform                world to index space transform.
    /// @param voxelOrder           sort point indices by local voxel offsets.
    /// @param recordVoxelOffsets   construct local voxel offsets
    template<typename PointArray>
    void construct(const PointArray& points, const math::Transform& xform,
        bool voxelOrder = false, bool recordVoxelOffsets = false);


    /// @brief  Partitions point indices into @c Log2Dim aligned buckets.
    ///
    /// @param points               list of world space points.
    /// @param xform                world to index space transform.
    /// @param voxelOrder           sort point indices by local voxel offsets.
    /// @param recordVoxelOffsets   construct local voxel offsets
    template<typename PointArray>
    static Ptr create(const PointArray& points, const math::Transform& xform,
        bool voxelOrder = false, bool recordVoxelOffsets = false);


    /// @brief Returns the number of buckets.
    size_t size() const { return mPageCount; }

    /// @brief true if the container size is 0, false otherwise.
    bool empty() const { return mPageCount == 0; }

    /// @brief Removes all data and frees up memory.
    void clear();

    /// @brief Exchanges the content of the container by another.
    void swap(PointPartitioner&);

    /// @brief Returns the point indices for bucket @a n
    IndexIterator indices(size_t n) const;

    /// @brief Returns the coordinate-aligned bounding box for bucket @a n
    CoordBBox getBBox(size_t n) const {
        return CoordBBox::createCube(mPageCoordinates[n], (1u << Log2Dim));
    }

    /// @brief Returns the origin coordinate for bucket @a n
    const Coord& origin(size_t n) const  { return mPageCoordinates[n]; }

    /// @brief  Returns a list of @c LeafNode voxel offsets for the points.
    /// @note   The list is optionally constructed.
    const VoxelOffsetArray&  voxelOffsets() const { return mVoxelOffsets; }

private:
    // Disallow copying
    PointPartitioner(const PointPartitioner&);
    PointPartitioner& operator=(const PointPartitioner&);

    boost::scoped_array<IndexType>  mPointIndices;
    VoxelOffsetArray                mVoxelOffsets;

    boost::scoped_array<IndexType>  mPageOffsets;
    boost::scoped_array<Coord>      mPageCoordinates;
    IndexType mPageCount;
}; // class PointPartitioner


typedef PointPartitioner<uint32_t, 3> UInt32PointPartitioner;


template<typename PointIndexT, Index Log2Dim>
class PointPartitioner<PointIndexT, Log2Dim>::IndexIterator
{
public:
    typedef PointIndexT     IndexType;

    IndexIterator(IndexType* begin = NULL, IndexType* end = NULL)
        : mBegin(begin), mEnd(end), mItem(begin) {}

    /// @brief Rewind to first item.
    void reset() { mItem = mBegin; }

    /// @brief  Number of point indices in the iterator range.
    size_t size() const { return mEnd - mBegin; }

    /// @brief  Returns the item to which this iterator is currently pointing.
    IndexType& operator*() { assert(mItem != NULL); return *mItem; }
    const IndexType& operator*() const { assert(mItem != NULL); return *mItem; }

    /// @brief  Return @c true if this iterator is not yet exhausted.
    operator bool() const { return mItem < mEnd; }
    bool test() const { return mItem < mEnd; }

    /// @brief  Advance to the next item.
    IndexIterator& operator++() { assert(this->test()); ++mItem; return *this; }

    /// @brief  Advance to the next item.
    bool next() { this->operator++(); return this->test(); }
    bool increment() { this->next(); return this->test(); }

    /// @brief Equality operators
    bool operator==(const IndexIterator& other) const { return mItem == other.mItem; }
    bool operator!=(const IndexIterator& other) const { return !this->operator==(other); }

private:
    IndexType * const mBegin, * const mEnd;
    IndexType * mItem;
}; // class PointPartitioner::IndexIterator


////////////////////////////////////////

// Implementation details


namespace point_partitioner_internal {

enum { LEAF_NODE_LIMIT = 1000000000 };

////////////////////////////////////////


template<typename PointArray>
struct ComputeBBoxOp {
    typedef typename PointArray::value_type PointType;
    typedef typename PointType::value_type  ElementType;

    ComputeBBoxOp(const PointArray& points)
        : mPoints(&points)
        , mMin(std::numeric_limits<ElementType>::max())
        , mMax(-std::numeric_limits<ElementType>::max())
    {
    }

    ComputeBBoxOp(const ComputeBBoxOp& other, tbb::split)
        : mPoints(other.mPoints)
        , mMin(std::numeric_limits<ElementType>::max())
        , mMax(-std::numeric_limits<ElementType>::max())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {

        PointType point, tmpMin(mMin), tmpMax(mMax);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mPoints->getPos(n, point);
            tmpMin[0] = std::min(tmpMin[0], point[0]);
            tmpMin[1] = std::min(tmpMin[1], point[1]);
            tmpMin[2] = std::min(tmpMin[2], point[2]);
            tmpMax[0] = std::max(tmpMax[0], point[0]);
            tmpMax[1] = std::max(tmpMax[1], point[1]);
            tmpMax[2] = std::max(tmpMax[2], point[2]);
        }

        mMin[0] = std::min(tmpMin[0], mMin[0]);
        mMin[1] = std::min(tmpMin[1], mMin[1]);
        mMin[2] = std::min(tmpMin[2], mMin[2]);
        mMax[0] = std::max(tmpMax[0], mMax[0]);
        mMax[1] = std::max(tmpMax[1], mMax[1]);
        mMax[2] = std::max(tmpMax[2], mMax[2]);
    }

    void join(ComputeBBoxOp& other) {
        mMin[0] = std::min(mMin[0], other.mMin[0]);
        mMin[1] = std::min(mMin[1], other.mMin[1]);
        mMin[2] = std::min(mMin[2], other.mMin[2]);
        mMax[0] = std::max(mMax[0], other.mMax[0]);
        mMax[1] = std::max(mMax[1], other.mMax[1]);
        mMax[2] = std::max(mMax[2], other.mMax[2]);
    }

    //////////

    PointArray const * const mPoints;
    PointType mMin, mMax;
};


////////////////////////////////////////


template<typename IndexT>
struct IndexPair {
    IndexT first, second;

    bool operator<(const IndexPair& rhs) const {
        return first < rhs.first;
    }
};


template<typename PointArray, typename IndexT, typename VoxelOffsetT>
struct BucketAndVoxelOffsetOp
{
    typedef typename PointArray::value_type   PointType;
    typedef boost::scoped_array<IndexT>       IndexArray;
    typedef boost::scoped_array<VoxelOffsetT> VoxelOffsetArray;

    BucketAndVoxelOffsetOp(
        VoxelOffsetArray& voxelOffsets, IndexArray& bucketOffsets,
        const PointArray& points, const math::Transform& m,
        const CoordBBox& bbox, int log2dim)
        : mVoxelOffsets(voxelOffsets.get())
        , mBucketOffsets(bucketOffsets.get())
        , mPoints(&points)
        , mXForm(m)
        , mBBox(bbox)
        , mDim(bbox.dim())
        , mBlockLog2Dim(log2dim)
    {
    }


    void operator()(const tbb::blocked_range<size_t>& range) const {
        PointType pos;
        Coord ijk(0, 0, 0), loc(0, 0, 0);

        const int xMin = mBBox.min()[0], yMin = mBBox.min()[1], zMin = mBBox.min()[2];
        const int yzDim = mDim[1] * mDim[2], zDim = mDim[2];

        const int log2dim = mBlockLog2Dim, log2dim2 = 2 * mBlockLog2Dim,
            mask = unsigned(1u << mBlockLog2Dim) - 1u;

        IndexT bucketOffset = 0;
        VoxelOffsetT voxelOffset = 0;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            mPoints->getPos(n, pos);
            ijk = mXForm.worldToIndexCellCentered(pos);

            // coord to offset
            if (mVoxelOffsets) {
                loc[0] = ijk[0] & mask;
                loc[1] = ijk[1] & mask;
                loc[2] = ijk[2] & mask;

                voxelOffset = VoxelOffsetT((loc[0] << log2dim2) + (loc[1] << log2dim) + loc[2]);
            }

            ijk[0] >>= log2dim;
            ijk[1] >>= log2dim;
            ijk[2] >>= log2dim;

            ijk[0] -= xMin;
            ijk[1] -= yMin;
            ijk[2] -= zMin;

            bucketOffset = IndexT(ijk[0] * yzDim + ijk[1] * zDim + ijk[2]);
            mBucketOffsets[n] = bucketOffset;

            if (mVoxelOffsets) {
                mVoxelOffsets[n] = voxelOffset;
            }
        }
    }

    //////////

    VoxelOffsetT       * const mVoxelOffsets;
    IndexT             * const mBucketOffsets;
    PointArray   const * const mPoints;

    const math::Transform mXForm;
    const CoordBBox mBBox;
    const Coord mDim;
    const int mBlockLog2Dim;
};


template<typename PointArray, typename IndexT, typename VoxelOffsetT>
struct ComputeOffsetOp
{
    typedef typename PointArray::value_type     PointType;
    typedef IndexPair<IndexT>                   IndexPairT;
    typedef boost::scoped_array<IndexPairT>     IndexPairArray;
    typedef boost::scoped_array<VoxelOffsetT>   VoxelOffsetArray;

    ComputeOffsetOp(
        VoxelOffsetArray& voxelOffsets, IndexPairArray& bucketOffsets,
        const PointArray& points, const math::Transform& m,
        const CoordBBox& bbox, int log2dim)
        : mVoxelOffsets(voxelOffsets.get())
        , mBucketOffsets(bucketOffsets.get())
        , mPoints(&points)
        , mXForm(m)
        , mBBox(bbox)
        , mDim(bbox.dim())
        , mBlockLog2Dim(log2dim)
    {
    }


    void operator()(const tbb::blocked_range<size_t>& range) const {
        PointType pos;
        Coord ijk(0, 0, 0), loc(0, 0, 0);

        const int xMin = mBBox.min()[0], yMin = mBBox.min()[1], zMin = mBBox.min()[2];
        const int yzDim = mDim[1] * mDim[2], zDim = mDim[2];

        const int log2dim = mBlockLog2Dim, log2dim2 = 2 * mBlockLog2Dim,
            mask = unsigned(1u << mBlockLog2Dim) - 1u;

        IndexT bucketOffset = 0;
        VoxelOffsetT voxelOffset = 0;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            mPoints->getPos(n, pos);
            ijk = mXForm.worldToIndexCellCentered(pos);

            // coord to offset
            if (mVoxelOffsets) {
                loc[0] = ijk[0] & mask;
                loc[1] = ijk[1] & mask;
                loc[2] = ijk[2] & mask;

                voxelOffset = VoxelOffsetT((loc[0] << log2dim2) + (loc[1] << log2dim) + loc[2]);
            }

            ijk[0] >>= log2dim;
            ijk[1] >>= log2dim;
            ijk[2] >>= log2dim;

            ijk[0] -= xMin;
            ijk[1] -= yMin;
            ijk[2] -= zMin;

            bucketOffset = IndexT(ijk[0] * yzDim + ijk[1] * zDim + ijk[2]);

            IndexPairT& item = mBucketOffsets[n];

            item.first  = bucketOffset;
            item.second = IndexT(n);

            if (mVoxelOffsets) {
                mVoxelOffsets[n] = voxelOffset;
            }
        }
    }

    //////////

    VoxelOffsetT       * const mVoxelOffsets;
    IndexPairT         * const mBucketOffsets;
    PointArray   const * const mPoints;

    const math::Transform mXForm;
    const CoordBBox mBBox;
    const Coord mDim;
    const int mBlockLog2Dim;
};



template<typename IndexT>
struct BucketMapOp
{
    typedef tbb::atomic<IndexT>                 AtomicIndex;
    typedef boost::scoped_array<AtomicIndex>    AtomicIndexArray;
    typedef boost::scoped_array<IndexT>         IndexArray;

    BucketMapOp(IndexArray& bucketIndices,
        AtomicIndexArray& bucketMap, const IndexArray& bucketOffsets)
        : mBucketIndices(bucketIndices.get())
        , mBucketMap(bucketMap.get())
        , mBucketOffsets(bucketOffsets.get())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mBucketIndices[n] = mBucketMap[mBucketOffsets[n]].fetch_and_increment();
        }
    }

    IndexT             * const mBucketIndices;
    AtomicIndex        * const mBucketMap;
    IndexT       const * const mBucketOffsets;
};


template<typename IndexT>
struct MergeOffsetsOp
{
    typedef tbb::atomic<IndexT>                 AtomicIndex;
    typedef boost::scoped_array<AtomicIndex>    AtomicIndexArray;
    typedef boost::scoped_array<IndexT>         IndexArray;

    MergeOffsetsOp(IndexArray& pointIndices,
        const AtomicIndexArray& bucketMap, const IndexArray& bucketOffsets)
        : mPointIndices(pointIndices.get())
        , mBucketMap(bucketMap.get())
        , mBucketOffsets(bucketOffsets.get())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mPointIndices[n] += mBucketMap[mBucketOffsets[n]];
        }
    }

    IndexT            * const mPointIndices;
    AtomicIndex const * const mBucketMap;
    IndexT      const * const mBucketOffsets;
};


template<typename IndexT>
struct BucketOrderOp
{
    typedef boost::scoped_array<IndexT> IndexArray;

    BucketOrderOp(IndexArray& pointIndices, IndexArray& bucketOffsets)
        : mPointIndices(pointIndices.get())
        , mBucketOffsets(bucketOffsets.get())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (IndexT n = static_cast<IndexT>(range.begin()), N = static_cast<IndexT>(range.end());
            n != N; ++n)
        {
            mBucketOffsets[mPointIndices[n]] = n;
        }
    }

    IndexT const * const mPointIndices;
    IndexT       * const mBucketOffsets;
};

template<typename IndexT>
struct PageOrderOp
{
    typedef boost::scoped_array<IndexT> IndexArray;
    typedef boost::scoped_array<IndexPair<IndexT> > IndexPairArray;

    PageOrderOp(IndexArray& pointIndices, const IndexPairArray& pairs)
        : mPointIndices(pointIndices.get())
        , mPairs(pairs.get())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n(range.begin()), N(range.end()); n != N; ++n) {
            mPointIndices[n] = mPairs[n].second;
        }
    }

    IndexT                  * const mPointIndices;
    IndexPair<IndexT> const * const mPairs;
};

template<typename IndexT>
struct PageBreakOp
{
    typedef std::pair<IndexT, IndexT> Range;
    typedef boost::scoped_array<IndexT> IndexArray;
    typedef boost::scoped_array<IndexPair<IndexT> > IndexPairArray;

    PageBreakOp(const Range& range, const IndexPairArray& pairs,
        IndexArray& segment, IndexT& size)
        : mRange(range)
        , mPairs(pairs.get())
        , mSegment(&segment)
        , mSize(&size)
    {
    }

    void operator()() const {

        const IndexT start = mRange.first;
        const IndexT end = mRange.second;

        std::deque<IndexT> pageBreaks;

        IndexT last = mPairs[start].first;
        for (IndexT n = start + 1; n != end; ++n) {
            const IndexPair<IndexT>& pair = mPairs[n];
            if (last != pair.first) {
                last = pair.first;
                pageBreaks.push_back(n);
            }
        }

        if (!pageBreaks.empty()) {

            IndexArray segment(new IndexT[pageBreaks.size()]);
            IndexT* item = segment.get();

            typename std::deque<IndexT>::iterator it = pageBreaks.begin();
            while (it != pageBreaks.end()) {
                *item++ = *it++;
            }

            mSegment->swap(segment);
            *mSize = static_cast<IndexT>(pageBreaks.size());
        }
    }

    Range mRange;
    IndexPair<IndexT> const * const mPairs;
    IndexArray              * const mSegment;
    IndexT                  * const mSize;
};



template<typename IndexT, Index Log2Dim>
struct VoxelOrderOp
{
    typedef typename boost::int_t<1 + (3 * Log2Dim)>::least VoxelOffsetT;
    typedef boost::scoped_array<VoxelOffsetT>   VoxelOffsetArray;
    typedef boost::scoped_array<IndexT>         IndexArray;

    VoxelOrderOp(IndexArray& indices, const IndexArray& pages,const VoxelOffsetArray& offsets)
        : mIndices(indices.get())
        , mPages(pages.get())
        , mVoxelOffsets(offsets.get())
    {
    }


    void operator()(const tbb::blocked_range<size_t>& range) const {

        IndexT pointCount = 0;
        for (size_t n(range.begin()), N(range.end()); n != N; ++n) {
            pointCount = std::max(pointCount, (mPages[n + 1] - mPages[n]));
        }

        const IndexT voxelCount = 1 << (3 * Log2Dim);

        // allocate histogram buffers
        boost::scoped_array<VoxelOffsetT> offsets(new VoxelOffsetT[pointCount]);
        boost::scoped_array<IndexT> sortedIndices(new IndexT[pointCount]);
        boost::scoped_array<IndexT> histogram(new IndexT[voxelCount]);

        for (size_t n(range.begin()), N(range.end()); n != N; ++n) {

            IndexT * const indices = mIndices + mPages[n];
            pointCount = mPages[n + 1] - mPages[n];

            // local copy of voxel offsets.
            for (IndexT i = 0; i < pointCount; ++i) {
                offsets[i] = mVoxelOffsets[ indices[i] ];
            }

            // reset histogram
            memset(&histogram[0], 0, voxelCount * sizeof(IndexT));

            // compute histogram
            for (IndexT i = 0; i < pointCount; ++i) {
                ++histogram[ offsets[i] ];
            }

            IndexT count = 0, startOffset;
            for (int i = 0; i < int(voxelCount); ++i) {
                if (histogram[i] > 0) {
                    startOffset = count;
                    count += histogram[i];
                    histogram[i] = startOffset;
                }
            }

            // sort indices based on voxel offset
            for (IndexT i = 0; i < pointCount; ++i) {
                sortedIndices[ histogram[ offsets[i] ]++ ] = indices[i];
            }

            memcpy(&indices[0], &sortedIndices[0], sizeof(IndexT) * pointCount);
        }
    }


    IndexT                * const mIndices;
    IndexT          const * const mPages;
    VoxelOffsetT    const * const mVoxelOffsets;
};


template<typename IndexT>
struct IndexOrderOp
{
    typedef boost::scoped_array<IndexT> IndexArray;

    IndexOrderOp(IndexArray& indices, const IndexArray& pages)
        : mIndices(indices.get()) , mPages(pages.get()) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n(range.begin()), N(range.end()); n != N; ++n)
            std::sort(mIndices + mPages[n], mIndices + mPages[n+1]);
    }

    IndexT       * const mIndices;
    IndexT const * const mPages;
};


template<typename PointArray, typename IndexT>
struct LeafNodeOriginOp
{
    typedef boost::scoped_array<IndexT>   IndexArray;
    typedef boost::scoped_array<Coord>    CoordArray;

    LeafNodeOriginOp(CoordArray& coordinates,
        const IndexArray& indices, const IndexArray& pages,
        const PointArray& points, const math::Transform& m, int log2dim)
        : mCoordinates(coordinates.get())
        , mIndices(indices.get())
        , mPages(pages.get())
        , mPoints(&points)
        , mXForm(m)
        , mLog2Dim(log2dim)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const int mask = ~((1 << mLog2Dim) - 1);
        Coord ijk;
        typename PointArray::value_type pos;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            mPoints->getPos(mIndices[mPages[n]], pos);
            ijk = mXForm.worldToIndexCellCentered(pos);

            ijk[0] &= mask;
            ijk[1] &= mask;
            ijk[2] &= mask;

            mCoordinates[n] = ijk;
        }
    }

    //////////

    Coord             * const mCoordinates;
    IndexT      const * const mIndices;
    IndexT      const * const mPages;
    PointArray  const * const mPoints;

    const math::Transform mXForm;
    const int mLog2Dim;
};


////////////////////////////////////////


// Tests whether the given bbox volume can be computed without
// overflowing the given IntType
template<typename IntType>
inline bool
isVolumeCalculationOverflowSafe(const CoordBBox& bbox)
{
    const uint64_t xdim = uint64_t(bbox.max()[0] - bbox.min()[0]);
    const uint64_t ydim = uint64_t(bbox.max()[1] - bbox.min()[1]);
    const uint64_t zdim = uint64_t(bbox.max()[2] - bbox.min()[2]);

    uint64_t product = xdim * ydim;
    if (product > std::numeric_limits<IntType>::max()) return false;

    product *= zdim;
    if (product > std::numeric_limits<IntType>::max()) return false;
    return true;
}


////////////////////////////////////////

template<typename PointArray>
inline CoordBBox
computeLeafBounds(const PointArray& points, const math::Transform& m, unsigned log2dim)
{
    ComputeBBoxOp<PointArray> bboxOp(points);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, points.size()), bboxOp);

    CoordBBox box;

    if (m.isLinear()) {
        box.min() = m.worldToIndexCellCentered(bboxOp.mMin);
        box.max() = m.worldToIndexCellCentered(bboxOp.mMax);
    } else {
        Vec3d minIS, maxIS;
        math::calculateBounds(m, bboxOp.mMin, bboxOp.mMax, minIS, maxIS);
        box.min() = math::Coord::round(minIS);
        box.max() = math::Coord::round(maxIS);
    }

    box.min() >>= log2dim;
    box.max() >>= log2dim;
    return box;
}


template<Index Log2Dim, typename IndexT, typename VoxelOffsetT, typename PointArray>
inline void partition(
    const PointArray& points,
    const math::Transform& xform,
    const CoordBBox& bbox,
    boost::scoped_array<IndexT>& pointIndices,
    boost::scoped_array<IndexT>& pageOffsets,
    IndexT& pageCount,
    boost::scoped_array<VoxelOffsetT>& voxelOffsets,
    bool recordVoxelOffsets)
{
    typedef tbb::atomic<IndexT>                 AtomicIndexT;
    typedef boost::scoped_array<AtomicIndexT>   AtomicIndexArray;
    typedef boost::scoped_array<IndexT>         IndexArray;

    //////////

    // Compute voxel and bucket offsets

    const size_t pointCount = points.size();
    const tbb::blocked_range<size_t> pointRange(0, pointCount);

    if (recordVoxelOffsets) {
        voxelOffsets.reset(new VoxelOffsetT[pointCount]);
    } else {
        voxelOffsets.reset();
    }

    IndexArray bucketOffsets(new IndexT[pointCount]);

    tbb::parallel_for(pointRange, BucketAndVoxelOffsetOp<
        PointArray, IndexT, VoxelOffsetT>(
            voxelOffsets, bucketOffsets, points, xform, bbox, int(Log2Dim)));

    // Compute bucket indices and bucket point counts

    const Index64 volume = bbox.volume();

    pointIndices.reset(new IndexT[pointCount]);
    AtomicIndexArray bucketMap(new AtomicIndexT[volume]);
    memset(&bucketMap[0], 0, sizeof(AtomicIndexT) * volume);

    tbb::parallel_for(pointRange,
        BucketMapOp<IndexT>(pointIndices, bucketMap, bucketOffsets));

    // Initialize page offsets and update bucket map with global start index.
    {
        pageCount = 0;
        for (size_t n(0), N(volume); n < N; ++n) {
            pageCount += static_cast<IndexT>(bucketMap[n] != 0);
        }

        pageOffsets.reset(new IndexT[pageCount + 1]);
        IndexT count = 0;
        for (size_t n = 0, idx = 0; n < volume; ++n) {
            if (bucketMap[n] != 0) {
                pageOffsets[idx] = count;
                count += bucketMap[n];
                bucketMap[n] = pageOffsets[idx];
                ++idx;
            }
        }

        pageOffsets[pageCount] = count;
    }


    // Merge bucket offsets with bucket indices
    tbb::parallel_for(pointRange,
        MergeOffsetsOp<IndexT>(pointIndices, bucketMap, bucketOffsets));

    bucketMap.reset();

    // Bucket order indices
    tbb::parallel_for(pointRange, BucketOrderOp<IndexT>(pointIndices, bucketOffsets));

    pointIndices.swap(bucketOffsets);
    bucketOffsets.reset();
}


template<Index Log2Dim, typename IndexT, typename VoxelOffsetT, typename PointArray>
inline void sortPartition(
    const PointArray& points,
    const math::Transform& xform,
    const CoordBBox& bbox,
    boost::scoped_array<IndexT>& pointIndices,
    boost::scoped_array<IndexT>& pageOffsets,
    IndexT& pageCount,
    boost::scoped_array<VoxelOffsetT>& voxelOffsets,
    bool recordVoxelOffsets)
{
    typedef boost::scoped_array<IndexT>         IndexArray;
    typedef IndexPair<IndexT>                   IndexPairT;
    typedef boost::scoped_array<IndexPairT>     IndexPairArray;

    //////////

    const size_t pointCount = points.size();
    const tbb::blocked_range<size_t> pointRange(0, pointCount);

    if (recordVoxelOffsets) {
        voxelOffsets.reset(new VoxelOffsetT[pointCount]);
    } else {
        voxelOffsets.reset();
    }

    IndexPairArray bucketOffsets(new IndexPairT[pointCount]);
    tbb::parallel_for(pointRange, ComputeOffsetOp<
        PointArray, IndexT, VoxelOffsetT>(
            voxelOffsets, bucketOffsets, points, xform, bbox, int(Log2Dim)));

    tbb::parallel_sort(bucketOffsets.get(), bucketOffsets.get() + pointCount);

    { // Compute page offsets
        const size_t nthreads = tbb::task_scheduler_init::default_num_threads();
        const size_t ntasks = nthreads > 1 ? 2 * nthreads : 1;

        if (ntasks > 1) {

            IndexArray segmentSizes(new IndexT[ntasks]);
            memset(segmentSizes.get(), 0, ntasks * sizeof(IndexT));
            boost::scoped_array<IndexArray> segments(new IndexArray[ntasks]);

            const IndexT grainSize = static_cast<IndexT>(pointCount / ntasks);
            const IndexT end = static_cast<IndexT>(grainSize * (ntasks - 1));

            tbb::task_group tasks;

            IndexT idx = 0;
            std::pair<IndexT, IndexT> range;

            // create tasks
            range.first  = 0;
            range.second = grainSize;
            tasks.run(PageBreakOp<IndexT>(range, bucketOffsets, segments[idx], segmentSizes[idx]));
            ++idx;

            for (IndexT n = grainSize; n < end; n += grainSize) {
                range.first  = n-1;
                range.second = n+grainSize;
                tasks.run(PageBreakOp<IndexT>(range, bucketOffsets, segments[idx], segmentSizes[idx]));
                ++idx;
            }

            range.first  = end-1;
            range.second = static_cast<IndexT>(pointCount);
            tasks.run(PageBreakOp<IndexT>(range, bucketOffsets, segments[idx], segmentSizes[idx]));

            tasks.wait();

            // collect data
            size_t pcount = 1;
            for (size_t n = 0; n < ntasks; ++n) {
                pcount += segmentSizes[n];
            }

            pageCount = static_cast<IndexT>(pcount);
            pageOffsets.reset(new IndexT[pageCount + 1]);

            pcount = 1;
            for (size_t n = 0; n < ntasks; ++n) {
                const IndexT size = segmentSizes[n];
                if(size != 0) {
                    memcpy(pageOffsets.get() + pcount, segments[n].get(), size * sizeof(IndexT));
                    pcount += size;
                }
            }

            pageOffsets[0] = 0;
            pageOffsets[pageCount] = static_cast<IndexT>(pointCount);

        } else {

            std::deque<IndexT> pageBreaks;
            IndexT last = bucketOffsets[0].first;

            for (IndexT n = 1; n != pointCount; ++n) {
                if (last != bucketOffsets[n].first) {
                    last = bucketOffsets[n].first;
                    pageBreaks.push_back(n);
                }
            }

            pageCount = static_cast<IndexT>(pageBreaks.size() + 1);
            pageOffsets.reset(new IndexT[pageCount + 1]);

            if (!pageBreaks.empty()) {

                IndexT* item = pageOffsets.get() + 1;

                typename std::deque<IndexT>::iterator it = pageBreaks.begin();
                while (it != pageBreaks.end()) {
                    *item++ = *it++;
                }
            }

            pageOffsets[0] = 0;
            pageOffsets[pageCount] = static_cast<IndexT>(pointCount);
        }
    }

    pointIndices.reset(new IndexT[pointCount]);
    tbb::parallel_for(pointRange, PageOrderOp<IndexT>(pointIndices, bucketOffsets));
}

} // namespace point_partitioner_internal


////////////////////////////////////////


template<typename PointIndexT, Index Log2Dim>
inline PointPartitioner<PointIndexT, Log2Dim>::PointPartitioner()
    : mPointIndices(NULL)
    , mVoxelOffsets(NULL)
    , mPageOffsets(NULL)
    , mPageCoordinates(NULL)
    , mPageCount(0)
{
}


template<typename PointIndexT, Index Log2Dim>
inline void
PointPartitioner<PointIndexT, Log2Dim>::clear()
{
    mPageCount = 0;
    mPointIndices.reset();
    mVoxelOffsets.reset();
    mPageOffsets.reset();
    mPageCoordinates.reset();
}


template<typename PointIndexT, Index Log2Dim>
inline void
PointPartitioner<PointIndexT, Log2Dim>::swap(PointPartitioner& rhs)
{
    std::swap(mPageCount, rhs.mPageCount);
    mPointIndices.swap(rhs.mPointIndices);
    mVoxelOffsets.swap(rhs.mVoxelOffsets);
    mPageOffsets.swap(rhs.mPageOffsets);
    mPageCoordinates.swap(rhs.mPageCoordinates);
}


template<typename PointIndexT, Index Log2Dim>
inline typename PointPartitioner<PointIndexT, Log2Dim>::IndexIterator
PointPartitioner<PointIndexT, Log2Dim>::indices(size_t n) const
{
    assert(bool(mPointIndices) && bool(mPageCount));
    return IndexIterator(
        mPointIndices.get() + mPageOffsets[n],
        mPointIndices.get() + mPageOffsets[n + 1]);
}


template<typename PointIndexT, Index Log2Dim>
template<typename PointArray>
inline void
PointPartitioner<PointIndexT, Log2Dim>::construct(const PointArray& points,
    const math::Transform& xform, bool voxelOrder, bool recordVoxelOffsets)
{
    const CoordBBox bbox =
        point_partitioner_internal::computeLeafBounds(points, xform, int(Log2Dim));

    if(!point_partitioner_internal::isVolumeCalculationOverflowSafe<PointIndexT>(bbox)) {
        // the bbox is computed in leafnode space (the lattice composing of only
        // leafnode origins) and should rarely overflow the volume calc. in practice.
        OPENVDB_THROW(ArithmeticError, "Detected overflow in bbox volume computation, "
            "use uint64 for the PointIndexT type in the PointPartitioner.");

        /// @todo This can be avoided using boost::int_t<N>::least to determine the
        ///       bucket offset type instead of PointIndexT in the @c sortPartition
        ///       and @c partition algorithms.
    }

    if (bbox.volume() > Index64(point_partitioner_internal::LEAF_NODE_LIMIT)) {
        point_partitioner_internal::sortPartition<Log2Dim>(points, xform, bbox,
            mPointIndices, mPageOffsets, mPageCount, mVoxelOffsets, (voxelOrder || recordVoxelOffsets));
    } else {
        point_partitioner_internal::partition<Log2Dim>(points, xform, bbox,
            mPointIndices, mPageOffsets, mPageCount, mVoxelOffsets, (voxelOrder || recordVoxelOffsets));
    }

    const tbb::blocked_range<size_t> pageRange(0, mPageCount);

    tbb::parallel_for(pageRange,
        point_partitioner_internal::IndexOrderOp<IndexType>(mPointIndices, mPageOffsets));

    mPageCoordinates.reset(new Coord[mPageCount]);

    tbb::parallel_for(pageRange,
        point_partitioner_internal::LeafNodeOriginOp<PointArray, IndexType>
            (mPageCoordinates, mPointIndices, mPageOffsets, points, xform, Log2Dim));

    if (mVoxelOffsets && voxelOrder) {
        tbb::parallel_for(pageRange, point_partitioner_internal::VoxelOrderOp<
            IndexType, Log2Dim>(mPointIndices, mPageOffsets, mVoxelOffsets));
    }

    if (mVoxelOffsets && !recordVoxelOffsets) {
        mVoxelOffsets.reset();
    }
}


template<typename PointIndexT, Index Log2Dim>
template<typename PointArray>
inline typename PointPartitioner<PointIndexT, Log2Dim>::Ptr
PointPartitioner<PointIndexT, Log2Dim>::create(const PointArray& points, const math::Transform& xform,
     bool voxelOrder, bool recordVoxelOffsets)
{
    Ptr ret(new PointPartitioner());
    ret->construct(points, xform, voxelOrder, recordVoxelOffsets);
    return ret;
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_PARTITIONER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
