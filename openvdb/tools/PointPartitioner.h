///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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
/// @file    PointPartitioner.h
///
/// @brief   Spatially partitions points using a parallel radix-based
///          sorting algorithm.
///
/// @details Performs a stable deterministic sort; partitioning the same
///          point sequence will produce the same result each time.
/// @details The algorithm is unbounded meaning that points may be
///          distributed anywhere in index space.
/// @details The actual points are never stored in the tool, only
///          offsets into an external array.
///
/// @author  Mihai Alden

#ifndef OPENVDB_TOOLS_POINT_PARTITIONER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_PARTITIONER_HAS_BEEN_INCLUDED


#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>

#include <deque>
#include <map>
#include <set>
#include <vector>

#include <boost/integer.hpp> // boost::int_t<N>::least
#include <boost/scoped_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/math/special_functions/fpclassify.hpp>//for boost::math::isfinite

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
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
////////////////////////////////////////

// Implementation details


namespace point_partitioner_internal {


template<typename IndexT>
struct PointBin
{
    typedef boost::shared_ptr<PointBin> Ptr;

    PointBin(size_t size, const Coord& min, const Coord& max)
        : mSize(size)
        , mIndices(new IndexT[size])
        , mBBox(min, max)
    {
    }

    const CoordBBox& bbox() const { return mBBox; }

    size_t size() const { return mSize; }

    IndexT& operator[](size_t n) { return mIndices[n]; }
    const IndexT& operator[](size_t n) const { return mIndices[n]; }

    IndexT* data() { return mIndices.get(); }
    const IndexT* data() const { return mIndices.get(); }

private:
    const size_t mSize;
    boost::scoped_array<IndexT> mIndices;
    const CoordBBox mBBox;
}; // struct PointBin


////////////////////////////////////////


template<typename PointArray, typename IndexT, typename VoxelOffsetT>
struct BucketAndVoxelOffsetOp
{
    typedef typename PointArray::value_type   PointType;

    BucketAndVoxelOffsetOp(
        VoxelOffsetT* voxelOffsets, IndexT* bucketOffsets,
        const PointArray& points, const math::Transform& m,
        const CoordBBox& bbox, int log2dim,
        const IndexT* pointIndexMap = NULL)
        : mVoxelOffsets(voxelOffsets)
        , mBucketOffsets(bucketOffsets)
        , mPoints(&points)
        , mXForm(m)
        , mBBox(bbox)
        , mDim(bbox.dim())
        , mBlockLog2Dim(log2dim)
        , mPointIndexMap(pointIndexMap)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        PointType pos;
        Coord ijk(0, 0, 0), loc(0, 0, 0);

        const int xMin = mBBox.min()[0], yMin = mBBox.min()[1], zMin = mBBox.min()[2];
        const int yzDim = mDim[1] * mDim[2], zDim = mDim[2];

        const int log2dim = mBlockLog2Dim, log2dim2 = 2 * mBlockLog2Dim,
            mask = unsigned(1u << mBlockLog2Dim) - 1u;

        const bool remapPointIndex = mPointIndexMap != NULL;

        IndexT bucketOffset = 0;
        VoxelOffsetT voxelOffset = 0;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const size_t pointIndex = remapPointIndex ? size_t(mPointIndexMap[n]) : n;

            mPoints->getPos(pointIndex, pos);

            if (boost::math::isfinite(pos[0]) &&
                boost::math::isfinite(pos[1]) &&
                boost::math::isfinite(pos[2])) {

                ijk = mXForm.worldToIndexCellCentered(pos);

                // coord to voxel offset
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

                if (mVoxelOffsets) mVoxelOffsets[pointIndex] = voxelOffset;
            }
        }
    }

    VoxelOffsetT          * const mVoxelOffsets;
    IndexT                * const mBucketOffsets;
    PointArray      const * const mPoints;
    math::Transform         const mXForm;
    CoordBBox               const mBBox;
    Coord                   const mDim;
    int                     const mBlockLog2Dim;
    IndexT          const * const mPointIndexMap;
}; // struct BucketAndVoxelOffsetOp


template<typename IndexT>
struct ComputePointOrderOp
{
    ComputePointOrderOp(IndexT* pointIndices, const IndexT* bucketCounters,
        const IndexT* bucketOffsets)
        : mPointIndices(pointIndices)
        , mBucketCounters(bucketCounters)
        , mBucketOffsets(bucketOffsets)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mPointIndices[n] += mBucketCounters[mBucketOffsets[n]];
        }
    }

    IndexT       * const mPointIndices;
    IndexT const * const mBucketCounters;
    IndexT const * const mBucketOffsets;
}; // struct ComputePointOrderOp


template<typename IndexT>
struct CreateOrderedPointIndexArrayOp
{
    typedef boost::scoped_array<IndexT> IndexArray;

    CreateOrderedPointIndexArrayOp(IndexArray& pointIndices,
        IndexArray& bucketOffsets, IndexT* indexMap)
        : mPointIndices(pointIndices.get())
        , mBucketOffsets(bucketOffsets.get())
        , mIndexMap(indexMap)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (IndexT i = IndexT(range.begin()), I = IndexT(range.end()); i != I; ++i) {
            mBucketOffsets[mPointIndices[i]] = mIndexMap[i];
        }
    }

    IndexT const * const mPointIndices;
    IndexT       * const mBucketOffsets;
    IndexT const * const mIndexMap;
}; // struct CreateOrderedPointIndexArrayOp


template<typename IndexT>
struct CopyIndexArrayOp
{
    CopyIndexArrayOp(IndexT* lhs, const IndexT* rhs) : mLhs(lhs), mRhs(rhs) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mLhs[n] = mRhs[n];
        }
    }

    IndexT       * const mLhs;
    IndexT const * const mRhs;
}; // struct CopyIndexArrayOp


template<typename IndexT>
struct CopyBinArrayOp
{
    typedef typename PointBin<IndexT>::Ptr PointBinPtr;

    CopyBinArrayOp(std::vector<IndexT*>& lhs, std::vector<PointBinPtr>& rhs)
        : mIndexLists(&lhs[0])
        , mBins(&rhs[0])
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n(range.begin()), N(range.end()); n != N; ++n) {

            IndexT* indices = mIndexLists[n];
            PointBinPtr& bin = mBins[n];

            tbb::parallel_for(tbb::blocked_range<size_t>(0, bin->size()),
                CopyIndexArrayOp<IndexT>(indices, bin->data()));
            bin.reset();
        }
    }

    IndexT      * const * const mIndexLists;
    PointBinPtr         * const mBins;
}; // struct CopyBinArrayOp


template<typename IndexT, Index Log2Dim>
struct VoxelOrderOp
{
    typedef typename boost::int_t<1 + (3 * Log2Dim)>::least VoxelOffsetT;
    typedef boost::scoped_array<VoxelOffsetT>               VoxelOffsetArray;
    typedef boost::scoped_array<IndexT>                     IndexArray;

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
}; // struct VoxelOrderOp


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

            if (boost::math::isfinite(pos[0]) &&
                boost::math::isfinite(pos[1]) &&
                boost::math::isfinite(pos[2])) {

                ijk = mXForm.worldToIndexCellCentered(pos);

                ijk[0] &= mask;
                ijk[1] &= mask;
                ijk[2] &= mask;

                mCoordinates[n] = ijk;
            }
        }
    }

    Coord                 * const mCoordinates;
    IndexT          const * const mIndices;
    IndexT          const * const mPages;
    PointArray      const * const mPoints;
    math::Transform         const mXForm;
    int                     const mLog2Dim;
}; // struct LeafNodeOriginOp


////////////////////////////////////////


template<typename IndexT>
struct MergePointBins
{
    typedef typename PointBin<IndexT>::Ptr          PointBinPtr;
    typedef std::deque<IndexT>                      PointIndexList;
    typedef boost::shared_ptr<PointIndexList>       PointIndexListPtr;
    typedef std::map<Coord, PointIndexListPtr>      PointIndexListMap;
    typedef boost::shared_ptr<PointIndexListMap>    PointIndexListMapPtr;

    MergePointBins(PointIndexListMapPtr* data, std::vector<PointBinPtr>& bins,
        const std::vector<Coord>& coords, Index binLog2Dim, size_t numSegments)
        : mData(data), mBins(&bins[0]), mCoords(&coords[0])
        , mBinLog2Dim(binLog2Dim), mNumSegments(numSegments)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const int binDim = int(1u << mBinLog2Dim) - 1;

        std::vector<PointIndexListPtr*> lists;
        std::vector<IndexT*> dataPointers;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const Coord& ijk = mCoords[n];
            size_t numIndices = 0;

            lists.clear();

            for (size_t i = 0; i < mNumSegments; ++i) {

                PointIndexListMap& idxMap = *mData[i];
                typename PointIndexListMap::iterator iter = idxMap.find(ijk);

                if (iter != idxMap.end() && iter->second) {
                    PointIndexListPtr& idxListPtr = iter->second;

                    lists.push_back(&idxListPtr);
                    numIndices += idxListPtr->size();
                }
            }

            if (lists.empty() || numIndices == 0) continue;

            PointBinPtr& binPtr = mBins[n];
            if (!binPtr) binPtr.reset(new PointBin<IndexT>(numIndices, ijk, ijk.offsetBy(binDim)));

            dataPointers.clear();
            IndexT* data = binPtr->data();

            for (size_t i = 0, count = 0, I = lists.size(); i < I; ++i) {
                dataPointers.push_back(data);
                count += (*lists[i])->size();
                data += (*lists[i])->size();
            }

            tbb::parallel_for(tbb::blocked_range<size_t>(0, dataPointers.size()),
                CopyData(&lists[0], &dataPointers[0]));
        }
    }

private:

    struct CopyData
    {
        CopyData(PointIndexListPtr** indexLists, IndexT** dataPointers)
            : mIndexLists(indexLists), mDataPointers(dataPointers)
        {
        }

        void operator()(const tbb::blocked_range<size_t>& range) const {
            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

                PointIndexListPtr& listPtr = *mIndexLists[n];
                IndexT* dataPtr = mDataPointers[n];

                typename PointIndexList::iterator it = listPtr->begin(), end = listPtr->end();

                for ( ; it != end; ++it) {
                    *dataPtr = *it;
                    ++dataPtr;
                }

                listPtr.reset();
            }
        }

        PointIndexListPtr * const * const mIndexLists;
        IndexT            * const * const mDataPointers;
    }; // struct CopyData

    PointIndexListMapPtr      * const mData;
    PointBinPtr               * const mBins;
    Coord               const * const mCoords;
    Index                       const mBinLog2Dim;
    size_t                      const mNumSegments;
}; // struct MergePointBins


template<typename PointArray, typename IndexT>
struct BinPointIndices
{
    typedef typename PointArray::value_type         PointType;
    typedef std::deque<IndexT>                      PointIndexList;
    typedef boost::shared_ptr<PointIndexList>       PointIndexListPtr;
    typedef std::map<Coord, PointIndexListPtr>      PointIndexListMap;
    typedef boost::shared_ptr<PointIndexListMap>    PointIndexListMapPtr;

    BinPointIndices(PointIndexListMapPtr* data, const PointArray& points,
        const math::Transform& m, Index binLog2Dim, size_t numSegments)
        : mData(data), mPoints(&points), mXForm(m)
        , mBinLog2Dim(binLog2Dim), mNumSegments(numSegments)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const int mask = ~((1u << mBinLog2Dim) - 1u);

        PointIndexList * idxList = NULL;
        Coord ijk(0, 0, 0), lastCoord(1, 2, 3);
        PointType pos;

        const size_t numPoints = mPoints->size();
        const size_t segmentSize = numPoints / mNumSegments;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            PointIndexListMapPtr& dataPtr = mData[n];
            if (!dataPtr) dataPtr.reset(new PointIndexListMap());
            PointIndexListMap& idxMap = *dataPtr;

            const bool isLastSegment = (n + 1) >= mNumSegments;

            const size_t start = n * segmentSize;
            const size_t end = isLastSegment ? numPoints : (start + segmentSize);

            for (size_t i = start; i != end; ++i) {

                mPoints->getPos(i, pos);

                if (boost::math::isfinite(pos[0]) &&
                    boost::math::isfinite(pos[1]) &&
                    boost::math::isfinite(pos[2])) {

                    ijk = mXForm.worldToIndexCellCentered(pos);

                    ijk[0] &= mask;
                    ijk[1] &= mask;
                    ijk[2] &= mask;

                    if (ijk != lastCoord) {
                        lastCoord = ijk;
                        PointIndexListPtr& idxListPtr = idxMap[ijk];
                        if (!idxListPtr) idxListPtr.reset(new PointIndexList());
                        idxList = idxListPtr.get();
                    }

                    idxList->push_back(IndexT(i));
                }
            }
        }
    }

    PointIndexListMapPtr       * const mData;
    PointArray           const * const mPoints;
    math::Transform              const mXForm;
    Index                        const mBinLog2Dim;
    size_t                       const mNumSegments;
}; // struct BinPointIndices


template<typename IndexT, typename PointArray>
inline void
binPoints(std::vector<typename PointBin<IndexT>::Ptr>& bins,
    const PointArray& points, const math::Transform& xform, Index binLog2Dim)
{
    typedef std::deque<IndexT>                      PointIndexList;
    typedef boost::shared_ptr<PointIndexList>       PointIndexListPtr;
    typedef std::map<Coord, PointIndexListPtr>      PointIndexListMap;
    typedef boost::shared_ptr<PointIndexListMap>    PointIndexListMapPtr;

    const size_t numThreads = size_t(tbb::task_scheduler_init::default_num_threads());

    size_t numTasks = 1;
    if (points.size() > (numThreads * 2)) numTasks = numThreads * 2;
    else if (points.size() > numThreads) numTasks = numThreads;

    boost::scoped_array<PointIndexListMapPtr> binData(new PointIndexListMapPtr[numTasks]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numTasks),
        BinPointIndices<PointArray, IndexT>(binData.get(), points, xform, binLog2Dim, numTasks));

    std::set<Coord> uniqueCoords;

    for (size_t i = 0; i < numTasks; ++i) {
        PointIndexListMap& idxMap = *binData[i];
        for (typename PointIndexListMap::iterator it = idxMap.begin(); it != idxMap.end(); ++it) {
            uniqueCoords.insert(it->first);
        }
    }

    std::vector<Coord> coords(uniqueCoords.begin(), uniqueCoords.end());

    const size_t binCount = coords.size();
    bins.resize(binCount);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, binCount),
        MergePointBins<IndexT>(binData.get(), bins, coords, binLog2Dim, numTasks));
}


template<typename IndexT, typename VoxelOffsetT, typename PointArray>
struct OrderBinIndices
{
    typedef boost::scoped_array<IndexT>     IndexArray;
    typedef typename PointBin<IndexT>::Ptr  PointBinPtr;

    OrderBinIndices(std::vector<PointBinPtr>& pointBins, const PointArray& points,
        IndexArray* pageOffsetArrays, VoxelOffsetT* voxelOffsets,
        const math::Transform& xform, Index log2Dim)
        : mBins(&pointBins[0])
        , mPoints(&points)
        , mPageOffsetArrays(pageOffsetArrays)
        , mVoxelOffsets(voxelOffsets)
        , mLog2Dim(log2Dim)
        , mXform(xform)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        size_t maxBinSize = 0;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            maxBinSize = std::max(maxBinSize, mBins[n]->size());
        }

        IndexArray bucketOffsets(new IndexT[maxBinSize]), bucketIndices(new IndexT[maxBinSize]);
        IndexArray bucketCounters;

        size_t bucketCountersSize = 0;
        int log2Dim = int(mLog2Dim);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const size_t pointCount = mBins[n]->size();
            const tbb::blocked_range<size_t> pointRange(0, pointCount);

            IndexT* binPointIndices = mBins[n]->data();

            CoordBBox bbox = mBins[n]->bbox();
            bbox.min() >>= log2Dim;
            bbox.max() >>= log2Dim;

            // Compute bucket and (optionally) voxel offsets for all points.
            tbb::parallel_for(pointRange,
                BucketAndVoxelOffsetOp<PointArray, IndexT, VoxelOffsetT>(
                    mVoxelOffsets, bucketOffsets.get(), *mPoints, mXform, bbox, log2Dim, binPointIndices));


            const size_t volume = size_t(bbox.volume());
            if (!bucketCounters || volume != bucketCountersSize) {
                bucketCountersSize = volume;
                bucketCounters.reset(new Index[bucketCountersSize]);
            }

            memset(bucketCounters.get(), 0, sizeof(IndexT) * bucketCountersSize);

            // Count the number of points per bucket and assign a local bucket index
            // to each point.
            for (size_t i = 0; i < pointCount; ++i) {
                bucketIndices[i] = bucketCounters[bucketOffsets[i]]++;
            }


            IndexT nonemptyBucketCount = 0;
            for (size_t i = 0; i < bucketCountersSize; ++i) {
                nonemptyBucketCount += static_cast<IndexT>(bucketCounters[i] != 0);
            }

            IndexArray& pageOffsets = mPageOffsetArrays[n];
            pageOffsets.reset(new IndexT[nonemptyBucketCount + 1]);
            pageOffsets[0] = nonemptyBucketCount + 1; // stores array size in first element

            // Compute prefix sum of bucket counters.
            IndexT count = 0, idx = 1;
            for (size_t i = 0; i < bucketCountersSize; ++i) {
                if (bucketCounters[i] != 0) {
                    pageOffsets[idx] = bucketCounters[i];
                    bucketCounters[i] = count;
                    count += pageOffsets[idx];
                    ++idx;
                }
            }

            // Compute final point order by incrementing the local bucket point index
            // with the prefix sum offset.
            tbb::parallel_for(pointRange, ComputePointOrderOp<IndexT>(
                bucketIndices.get(), bucketCounters.get(), bucketOffsets.get()));

            tbb::parallel_for(pointRange, CreateOrderedPointIndexArrayOp<IndexT>(
                bucketIndices, bucketOffsets, binPointIndices));

            // Update bin index array
            tbb::parallel_for(pointRange, CopyIndexArrayOp<IndexT>(binPointIndices, bucketOffsets.get()));
        }
    }

    PointBinPtr           * const mBins;
    PointArray      const * const mPoints;
    IndexArray            * const mPageOffsetArrays;
    VoxelOffsetT          * const mVoxelOffsets;
    Index                   const mLog2Dim;
    math::Transform         const mXform;
}; // struct OrderBinIndices


////////////////////////////////////////


template<Index Log2Dim, typename IndexT, typename VoxelOffsetT, typename PointArray>
inline void partition(
    const PointArray& points,
    const math::Transform& xform,
    boost::scoped_array<IndexT>& pointIndices,
    boost::scoped_array<IndexT>& pageOffsets,
    IndexT& pageCount,
    boost::scoped_array<VoxelOffsetT>& voxelOffsets,
    bool recordVoxelOffsets)
{
    typedef boost::scoped_array<IndexT>         IndexArray;

    Index binLog2Dim = 5u + Log2Dim;

    std::vector<typename PointBin<IndexT>::Ptr> pointBins;
    binPoints<IndexT>(pointBins, points, xform, binLog2Dim);

    const size_t pointCount = points.size();

    if (recordVoxelOffsets) {
        voxelOffsets.reset(new VoxelOffsetT[pointCount]);
    } else {
        voxelOffsets.reset();
    }


    const size_t numBins = pointBins.size();
    const tbb::blocked_range<size_t> binRange(0, numBins);

    boost::scoped_array<IndexArray> pageOffsetArrays(new IndexArray[numBins]);

    tbb::parallel_for(binRange, OrderBinIndices<IndexT, VoxelOffsetT, PointArray>
        (pointBins, points, pageOffsetArrays.get(), voxelOffsets.get(), xform, Log2Dim));

    pageCount = 0;
    for (size_t n = 0; n < numBins; ++n) {
        pageCount += pageOffsetArrays[n][0] - 1;
    }

    pageOffsets.reset(new IndexT[pageCount + 1]);

    IndexT count = 0, idx = 0;
    for (size_t n = 0; n < numBins; ++n) {

        IndexT* offsets = pageOffsetArrays[n].get();
        size_t size = size_t(offsets[0]);

        for (size_t i = 1; i < size; ++i) {
            pageOffsets[idx++] = count;
            count += offsets[i];
        }
    }

    pageOffsets[pageCount] = count;

    pointIndices.reset(new IndexT[pointCount]);

    std::vector<IndexT*> indexArray;

    IndexT* index = pointIndices.get();
    for (size_t n = 0; n < numBins; ++n) {
        indexArray.push_back(index);
        index += pointBins[n]->size();
    }

    tbb::parallel_for(binRange, CopyBinArrayOp<IndexT>(indexArray, pointBins));
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
    const IndexType tmpLhsPageCount = mPageCount;
    mPageCount = rhs.mPageCount;
    rhs.mPageCount = tmpLhsPageCount;

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
    point_partitioner_internal::partition<Log2Dim>(points, xform,
        mPointIndices, mPageOffsets, mPageCount, mVoxelOffsets, (voxelOrder || recordVoxelOffsets));

    const tbb::blocked_range<size_t> pageRange(0, mPageCount);
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

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
