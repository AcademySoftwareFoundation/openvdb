// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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
#include <openvdb/util/Assert.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath> // for std::isfinite()
#include <deque>
#include <map>
#include <set>
#include <utility> // std::pair
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


/// @brief   Partitions points into @c BucketLog2Dim aligned buckets
///          using a parallel radix-based sorting algorithm.
///
/// @interface PointArray
/// Expected interface for the PointArray container:
/// @code
/// template<typename VectorType>
/// struct PointArray
/// {
///     // The type used to represent world-space point positions
///     using PosType = VectorType;
///
///     // Return the number of points in the array
///     size_t size() const;
///
///     // Return the world-space position of the nth point in the array.
///     void getPos(size_t n, PosType& xyz) const;
/// };
/// @endcode
///
/// @details Performs a stable deterministic sort; partitioning the same
///          point sequence will produce the same result each time.
/// @details The algorithm is unbounded meaning that points may be
///          distributed anywhere in index space.
/// @details The actual points are never stored in the tool, only
///          offsets into an external array.
/// @details @c BucketLog2Dim defines the bucket coordinate dimensions,
///          i.e. BucketLog2Dim = 3 corresponds to a bucket that spans
///          a (2^3)^3 = 8^3 voxel region.
template<typename PointIndexType = uint32_t, Index BucketLog2Dim = 3>
class PointPartitioner
{
public:
    enum { LOG2DIM = BucketLog2Dim };

    using Ptr = SharedPtr<PointPartitioner>;
    using ConstPtr = SharedPtr<const PointPartitioner>;

    using IndexType = PointIndexType;

    static constexpr Index bits = 1 + (3 * BucketLog2Dim);
    // signed, so if bits is exactly 16, int32 is required
    using VoxelOffsetType = typename std::conditional<(bits < 16),
        int16_t, typename std::conditional<(bits < 32), int32_t, int64_t>::type>::type;

    using VoxelOffsetArray = std::unique_ptr<VoxelOffsetType[]>;

    class IndexIterator;

    //////////

    PointPartitioner();

    /// @brief  Partitions point indices into @c BucketLog2Dim aligned buckets.
    ///
    /// @param points                 list of world space points.
    /// @param xform                  world to index space transform.
    /// @param voxelOrder             sort point indices by local voxel offsets.
    /// @param recordVoxelOffsets     construct local voxel offsets
    /// @param cellCenteredTransform  toggle the cell-centered interpretation that imagines world
    ///                               space as divided into discrete cells (e.g., cubes) centered
    ///                               on the image of the index-space lattice points.
    template<typename PointArray>
    void construct(const PointArray& points, const math::Transform& xform,
        bool voxelOrder = false, bool recordVoxelOffsets = false,
        bool cellCenteredTransform = true);


    /// @brief  Partitions point indices into @c BucketLog2Dim aligned buckets.
    ///
    /// @param points                 list of world space points.
    /// @param xform                  world to index space transform.
    /// @param voxelOrder             sort point indices by local voxel offsets.
    /// @param recordVoxelOffsets     construct local voxel offsets
    /// @param cellCenteredTransform  toggle the cell-centered interpretation that imagines world
    ///                               space as divided into discrete cells (e.g., cubes) centered
    ///                               on the image of the index-space lattice points.
    template<typename PointArray>
    static Ptr create(const PointArray& points, const math::Transform& xform,
        bool voxelOrder = false, bool recordVoxelOffsets = false,
        bool cellCenteredTransform = true);


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
        return CoordBBox::createCube(mPageCoordinates[n], (1u << BucketLog2Dim));
    }

    /// @brief Returns the origin coordinate for bucket @a n
    const Coord& origin(size_t n) const  { return mPageCoordinates[n]; }

    /// @brief  Returns a list of @c LeafNode voxel offsets for the points.
    /// @note   The list is optionally constructed.
    const VoxelOffsetArray&  voxelOffsets() const { return mVoxelOffsets; }

    /// @brief  Returns @c true if this point partitioning was constructed
    ///         using a cell-centered transform.
    /// @note   Cell-centered interpretation is the default behavior.
    bool usingCellCenteredTransform() const { return mUsingCellCenteredTransform; }

private:
    // Disallow copying
    PointPartitioner(const PointPartitioner&);
    PointPartitioner& operator=(const PointPartitioner&);

    std::unique_ptr<IndexType[]>    mPointIndices;
    VoxelOffsetArray                mVoxelOffsets;

    std::unique_ptr<IndexType[]>    mPageOffsets;
    std::unique_ptr<Coord[]>        mPageCoordinates;
    IndexType mPageCount;
    bool      mUsingCellCenteredTransform;
}; // class PointPartitioner


using UInt32PointPartitioner = PointPartitioner<uint32_t, 3>;


template<typename PointIndexType, Index BucketLog2Dim>
class PointPartitioner<PointIndexType, BucketLog2Dim>::IndexIterator
{
public:
    using IndexType = PointIndexType;

    IndexIterator(IndexType* begin = nullptr, IndexType* end = nullptr)
        : mBegin(begin), mEnd(end), mItem(begin) {}

    /// @brief Rewind to first item.
    void reset() { mItem = mBegin; }

    /// @brief  Number of point indices in the iterator range.
    size_t size() const { return mEnd - mBegin; }

    /// @brief  Returns the item to which this iterator is currently pointing.
    IndexType& operator*() { OPENVDB_ASSERT(mItem != nullptr); return *mItem; }
    const IndexType& operator*() const { OPENVDB_ASSERT(mItem != nullptr); return *mItem; }

    /// @brief  Return @c true if this iterator is not yet exhausted.
    operator bool() const { return mItem < mEnd; }
    bool test() const { return mItem < mEnd; }

    /// @brief  Advance to the next item.
    IndexIterator& operator++() { OPENVDB_ASSERT(this->test()); ++mItem; return *this; }

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

/// @cond OPENVDB_DOCS_INTERNAL

namespace point_partitioner_internal {


template<typename PointIndexType>
struct ComputePointOrderOp
{
    ComputePointOrderOp(PointIndexType* pointOrder,
        const PointIndexType* bucketCounters, const PointIndexType* bucketOffsets)
        : mPointOrder(pointOrder)
        , mBucketCounters(bucketCounters)
        , mBucketOffsets(bucketOffsets)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mPointOrder[n] += mBucketCounters[mBucketOffsets[n]];
        }
    }

    PointIndexType       * const mPointOrder;
    PointIndexType const * const mBucketCounters;
    PointIndexType const * const mBucketOffsets;
}; // struct ComputePointOrderOp


template<typename PointIndexType>
struct CreateOrderedPointIndexArrayOp
{
    CreateOrderedPointIndexArrayOp(PointIndexType* orderedIndexArray,
        const PointIndexType* pointOrder, const PointIndexType* indices)
        : mOrderedIndexArray(orderedIndexArray)
        , mPointOrder(pointOrder)
        , mIndices(indices)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            mOrderedIndexArray[mPointOrder[n]] = mIndices[n];
        }
    }

    PointIndexType       * const mOrderedIndexArray;
    PointIndexType const * const mPointOrder;
    PointIndexType const * const mIndices;
}; // struct CreateOrderedPointIndexArrayOp


template<typename PointIndexType, Index BucketLog2Dim>
struct VoxelOrderOp
{
    static constexpr Index bits = 1 + (3 * BucketLog2Dim);
    // signed, so if bits is exactly 16, int32 is required
    using VoxelOffsetType = typename std::conditional<(bits < 16),
        int16_t, typename std::conditional<(bits < 32), int32_t, int64_t>::type>::type;

    using VoxelOffsetArray = std::unique_ptr<VoxelOffsetType[]>;
    using IndexArray = std::unique_ptr<PointIndexType[]>;

    VoxelOrderOp(IndexArray& indices, const IndexArray& pages,const VoxelOffsetArray& offsets)
        : mIndices(indices.get())
        , mPages(pages.get())
        , mVoxelOffsets(offsets.get())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        PointIndexType pointCount = 0;
        for (size_t n(range.begin()), N(range.end()); n != N; ++n) {
            pointCount = std::max(pointCount, (mPages[n + 1] - mPages[n]));
        }

        const PointIndexType voxelCount = 1 << (3 * BucketLog2Dim);

        // allocate histogram buffers
        std::unique_ptr<VoxelOffsetType[]> offsets(new VoxelOffsetType[pointCount]);
        std::unique_ptr<PointIndexType[]> sortedIndices(new PointIndexType[pointCount]);
        std::unique_ptr<PointIndexType[]> histogram(new PointIndexType[voxelCount]);

        for (size_t n(range.begin()), N(range.end()); n != N; ++n) {

            PointIndexType * const indices = mIndices + mPages[n];
            pointCount = mPages[n + 1] - mPages[n];

            // local copy of voxel offsets.
            for (PointIndexType i = 0; i < pointCount; ++i) {
                offsets[i] = mVoxelOffsets[ indices[i] ];
            }

            // reset histogram
            memset(&histogram[0], 0, voxelCount * sizeof(PointIndexType));

            // compute histogram
            for (PointIndexType i = 0; i < pointCount; ++i) {
                ++histogram[ offsets[i] ];
            }

            PointIndexType count = 0, startOffset;
            for (int i = 0; i < int(voxelCount); ++i) {
                if (histogram[i] > 0) {
                    startOffset = count;
                    count += histogram[i];
                    histogram[i] = startOffset;
                }
            }

            // sort indices based on voxel offset
            for (PointIndexType i = 0; i < pointCount; ++i) {
                sortedIndices[ histogram[ offsets[i] ]++ ] = indices[i];
            }

            memcpy(&indices[0], &sortedIndices[0], sizeof(PointIndexType) * pointCount);
        }
    }

    PointIndexType        * const mIndices;
    PointIndexType  const * const mPages;
    VoxelOffsetType const * const mVoxelOffsets;
}; // struct VoxelOrderOp


////////////////////////////////////////


template<typename T>
struct Array
{
    using Ptr = std::unique_ptr<Array>;

    Array(size_t size) : mSize(size), mData(new T[size]) { }

    size_t size() const { return mSize; }

    T* data() { return mData.get(); }
    const T* data() const { return mData.get(); }

    void clear() { mSize = 0; mData.reset(); }

private:
    size_t                  mSize;
    std::unique_ptr<T[]>    mData;
}; // struct Array


template<typename PointIndexType>
struct MoveSegmentDataOp
{
    using SegmentPtr = typename Array<PointIndexType>::Ptr;

    MoveSegmentDataOp(std::vector<PointIndexType*>& indexLists, SegmentPtr* segments)
        : mIndexLists(&indexLists[0]), mSegments(segments)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        for (size_t n(range.begin()), N(range.end()); n != N; ++n) {
            PointIndexType* indices = mIndexLists[n];
            SegmentPtr& segment = mSegments[n];

            tbb::parallel_for(tbb::blocked_range<size_t>(0, segment->size()),
                CopyData(indices, segment->data()));

            segment.reset(); // clear data
        }
    }

private:

    struct CopyData
    {
        CopyData(PointIndexType* lhs, const PointIndexType* rhs) : mLhs(lhs), mRhs(rhs) { }

        void operator()(const tbb::blocked_range<size_t>& range) const {
            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                mLhs[n] = mRhs[n];
            }
        }

        PointIndexType       * const mLhs;
        PointIndexType const * const mRhs;
    };

    PointIndexType * const * const mIndexLists;
    SegmentPtr             * const mSegments;
}; // struct MoveSegmentDataOp


template<typename PointIndexType>
struct MergeBinsOp
{
    using Segment = Array<PointIndexType>;
    using SegmentPtr = typename Segment::Ptr;

    using IndexPair = std::pair<PointIndexType, PointIndexType>;
    using IndexPairList = std::deque<IndexPair>;
    using IndexPairListPtr = std::shared_ptr<IndexPairList>;
    using IndexPairListMap = std::map<Coord, IndexPairListPtr>;
    using IndexPairListMapPtr = std::shared_ptr<IndexPairListMap>;

    MergeBinsOp(IndexPairListMapPtr* bins,
        SegmentPtr* indexSegments,
        SegmentPtr* offsetSegments,
        Coord* coords,
        size_t numSegments)
        : mBins(bins)
        , mIndexSegments(indexSegments)
        , mOffsetSegments(offsetSegments)
        , mCoords(coords)
        , mNumSegments(numSegments)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        std::vector<IndexPairListPtr*> data;
        std::vector<PointIndexType> arrayOffsets;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const Coord& ijk = mCoords[n];
            size_t numIndices = 0;

            data.clear();

            for (size_t i = 0, I = mNumSegments; i < I; ++i) {

                IndexPairListMap& idxMap = *mBins[i];
                typename IndexPairListMap::iterator iter = idxMap.find(ijk);

                if (iter != idxMap.end() && iter->second) {
                    IndexPairListPtr& idxListPtr = iter->second;

                    data.push_back(&idxListPtr);
                    numIndices += idxListPtr->size();
                }
            }

            if (data.empty() || numIndices == 0) continue;

            SegmentPtr& indexSegment = mIndexSegments[n];
            SegmentPtr& offsetSegment = mOffsetSegments[n];

            indexSegment.reset(new Segment(numIndices));
            offsetSegment.reset(new Segment(numIndices));

            arrayOffsets.clear();
            arrayOffsets.reserve(data.size());

            for (size_t i = 0, count = 0, I = data.size(); i < I; ++i) {
                arrayOffsets.push_back(PointIndexType(count));
                count += (*data[i])->size();
            }

            tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()),
                CopyData(&data[0], &arrayOffsets[0], indexSegment->data(), offsetSegment->data()));
        }
    }

private:

    struct CopyData
    {
        CopyData(IndexPairListPtr** indexLists,
            const PointIndexType* arrayOffsets,
            PointIndexType* indices,
            PointIndexType* offsets)
            : mIndexLists(indexLists)
            , mArrayOffsets(arrayOffsets)
            , mIndices(indices)
            , mOffsets(offsets)
        {
        }

        void operator()(const tbb::blocked_range<size_t>& range) const {

            using CIter = typename IndexPairList::const_iterator;

            for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

                const PointIndexType arrayOffset = mArrayOffsets[n];
                PointIndexType* indexPtr = &mIndices[arrayOffset];
                PointIndexType* offsetPtr = &mOffsets[arrayOffset];

                IndexPairListPtr& list = *mIndexLists[n];

                for (CIter it = list->begin(), end = list->end(); it != end; ++it) {
                    const IndexPair& data = *it;
                    *indexPtr++ = data.first;
                    *offsetPtr++ = data.second;
                }

                list.reset(); // clear data
            }
        }

        IndexPairListPtr * const * const mIndexLists;
        PointIndexType     const * const mArrayOffsets;
        PointIndexType           * const mIndices;
        PointIndexType           * const mOffsets;
    }; // struct CopyData

    IndexPairListMapPtr       * const mBins;
    SegmentPtr                * const mIndexSegments;
    SegmentPtr                * const mOffsetSegments;
    Coord               const * const mCoords;
    size_t                      const mNumSegments;
}; // struct MergeBinsOp


template<typename PointArray, typename PointIndexType, typename VoxelOffsetType>
struct BinPointIndicesOp
{
    using PosType = typename PointArray::PosType;
    using IndexPair = std::pair<PointIndexType, PointIndexType>;
    using IndexPairList = std::deque<IndexPair>;
    using IndexPairListPtr = std::shared_ptr<IndexPairList>;
    using IndexPairListMap = std::map<Coord, IndexPairListPtr>;
    using IndexPairListMapPtr = std::shared_ptr<IndexPairListMap>;

    BinPointIndicesOp(IndexPairListMapPtr* data,
        const PointArray& points,
        VoxelOffsetType* voxelOffsets,
        const math::Transform& m,
        Index binLog2Dim,
        Index bucketLog2Dim,
        size_t numSegments,
        bool cellCenteredTransform)
        : mData(data)
        , mPoints(&points)
        , mVoxelOffsets(voxelOffsets)
        , mXForm(m)
        , mBinLog2Dim(binLog2Dim)
        , mBucketLog2Dim(bucketLog2Dim)
        , mNumSegments(numSegments)
        , mCellCenteredTransform(cellCenteredTransform)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const Index log2dim = mBucketLog2Dim;
        const Index log2dim2 = 2 * log2dim;
        const Index bucketMask = (1u << log2dim) - 1u;

        const Index binLog2dim = mBinLog2Dim;
        const Index binLog2dim2 = 2 * binLog2dim;

        const Index binMask = (1u << (log2dim + binLog2dim)) - 1u;
        const Index invBinMask = ~binMask;

        IndexPairList * idxList = nullptr;
        Coord ijk(0, 0, 0), loc(0, 0, 0), binCoord(0, 0, 0), lastBinCoord(1, 2, 3);
        PosType pos;

        PointIndexType bucketOffset = 0;
        VoxelOffsetType voxelOffset = 0;

        const bool cellCentered = mCellCenteredTransform;

        const size_t numPoints = mPoints->size();
        const size_t segmentSize = numPoints / mNumSegments;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            IndexPairListMapPtr& dataPtr = mData[n];
            if (!dataPtr) dataPtr.reset(new IndexPairListMap());
            IndexPairListMap& idxMap = *dataPtr;

            const bool isLastSegment = (n + 1) >= mNumSegments;

            const size_t start = n * segmentSize;
            const size_t end = isLastSegment ? numPoints : (start + segmentSize);

            for (size_t i = start; i != end; ++i) {

                mPoints->getPos(i, pos);

                if (std::isfinite(pos[0]) && std::isfinite(pos[1]) && std::isfinite(pos[2])) {
                    ijk = cellCentered ? mXForm.worldToIndexCellCentered(pos) :
                        mXForm.worldToIndexNodeCentered(pos);

                    if (mVoxelOffsets) {
                        loc[0] = ijk[0] & bucketMask;
                        loc[1] = ijk[1] & bucketMask;
                        loc[2] = ijk[2] & bucketMask;
                        voxelOffset = VoxelOffsetType(
                            (loc[0] << log2dim2) + (loc[1] << log2dim) + loc[2]);
                    }

                    binCoord[0] = ijk[0] & invBinMask;
                    binCoord[1] = ijk[1] & invBinMask;
                    binCoord[2] = ijk[2] & invBinMask;

                    ijk[0] &= binMask;
                    ijk[1] &= binMask;
                    ijk[2] &= binMask;

                    ijk[0] >>= log2dim;
                    ijk[1] >>= log2dim;
                    ijk[2] >>= log2dim;

                    bucketOffset = PointIndexType(
                        (ijk[0] << binLog2dim2) + (ijk[1] << binLog2dim) + ijk[2]);

                    if (lastBinCoord != binCoord) {
                        lastBinCoord = binCoord;
                        IndexPairListPtr& idxListPtr = idxMap[lastBinCoord];
                        if (!idxListPtr) idxListPtr.reset(new IndexPairList());
                        idxList = idxListPtr.get();
                    }

                    idxList->push_back(IndexPair(PointIndexType(i), bucketOffset));
                    if (mVoxelOffsets) mVoxelOffsets[i] = voxelOffset;
                }
            }
        }
    }

    IndexPairListMapPtr        * const mData;
    PointArray           const * const mPoints;
    VoxelOffsetType            * const mVoxelOffsets;
    math::Transform              const mXForm;
    Index                        const mBinLog2Dim;
    Index                        const mBucketLog2Dim;
    size_t                       const mNumSegments;
    bool                         const mCellCenteredTransform;
}; // struct BinPointIndicesOp


template<typename PointIndexType>
struct OrderSegmentsOp
{
    using IndexArray = std::unique_ptr<PointIndexType[]>;
    using SegmentPtr = typename Array<PointIndexType>::Ptr;

    OrderSegmentsOp(SegmentPtr* indexSegments, SegmentPtr* offsetSegments,
        IndexArray* pageOffsetArrays, IndexArray* pageIndexArrays, Index binVolume)
        : mIndexSegments(indexSegments)
        , mOffsetSegments(offsetSegments)
        , mPageOffsetArrays(pageOffsetArrays)
        , mPageIndexArrays(pageIndexArrays)
        , mBinVolume(binVolume)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

        const size_t bucketCountersSize = size_t(mBinVolume);
        IndexArray bucketCounters(new PointIndexType[bucketCountersSize]);

        size_t maxSegmentSize = 0;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            maxSegmentSize = std::max(maxSegmentSize, mIndexSegments[n]->size());
        }

        IndexArray bucketIndices(new PointIndexType[maxSegmentSize]);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            memset(bucketCounters.get(), 0, sizeof(PointIndexType) * bucketCountersSize);

            const size_t segmentSize = mOffsetSegments[n]->size();
            PointIndexType* offsets = mOffsetSegments[n]->data();

            // Count the number of points per bucket and assign a local bucket index
            // to each point.
            for (size_t i = 0; i < segmentSize; ++i) {
                bucketIndices[i] = bucketCounters[offsets[i]]++;
            }

            PointIndexType nonemptyBucketCount = 0;
            for (size_t i = 0; i < bucketCountersSize; ++i) {
                nonemptyBucketCount += static_cast<PointIndexType>(bucketCounters[i] != 0);
            }


            IndexArray& pageOffsets = mPageOffsetArrays[n];
            pageOffsets.reset(new PointIndexType[nonemptyBucketCount + 1]);
            pageOffsets[0] = nonemptyBucketCount + 1; // stores array size in first element

            IndexArray& pageIndices = mPageIndexArrays[n];
            pageIndices.reset(new PointIndexType[nonemptyBucketCount]);

            // Compute bucket counter prefix sum
            PointIndexType count = 0, idx = 0;
            for (size_t i = 0; i < bucketCountersSize; ++i) {
                if (bucketCounters[i] != 0) {
                    pageIndices[idx] = static_cast<PointIndexType>(i);
                    pageOffsets[idx+1] = bucketCounters[i];
                    bucketCounters[i] = count;
                    count += pageOffsets[idx+1];
                    ++idx;
                }
            }

            PointIndexType* indices = mIndexSegments[n]->data();
            const tbb::blocked_range<size_t> segmentRange(0, segmentSize);

            // Compute final point order by incrementing the local bucket point index
            // with the prefix sum offset.
            tbb::parallel_for(segmentRange, ComputePointOrderOp<PointIndexType>(
                bucketIndices.get(), bucketCounters.get(), offsets));

            tbb::parallel_for(segmentRange, CreateOrderedPointIndexArrayOp<PointIndexType>(
                offsets, bucketIndices.get(), indices));

            mIndexSegments[n]->clear(); // clear data
        }
    }

    SegmentPtr * const mIndexSegments;
    SegmentPtr * const mOffsetSegments;
    IndexArray * const mPageOffsetArrays;
    IndexArray * const mPageIndexArrays;
    Index        const mBinVolume;
}; // struct OrderSegmentsOp


////////////////////////////////////////


/// @brief Segment points using one level of least significant digit radix bins.
template<typename PointIndexType, typename VoxelOffsetType, typename PointArray>
inline void binAndSegment(
    const PointArray& points,
    const math::Transform& xform,
    std::unique_ptr<typename Array<PointIndexType>::Ptr[]>& indexSegments,
    std::unique_ptr<typename Array<PointIndexType>::Ptr[]>& offsetSegments,
    std::vector<Coord>& coords,
    const Index binLog2Dim,
    const Index bucketLog2Dim,
    VoxelOffsetType* voxelOffsets = nullptr,
    bool cellCenteredTransform = true)
{
    using IndexPair = std::pair<PointIndexType, PointIndexType>;
    using IndexPairList = std::deque<IndexPair>;
    using IndexPairListPtr = std::shared_ptr<IndexPairList>;
    using IndexPairListMap = std::map<Coord, IndexPairListPtr>;
    using IndexPairListMapPtr = std::shared_ptr<IndexPairListMap>;

    size_t numTasks = 1, numThreads = size_t(tbb::this_task_arena::max_concurrency());
    if (points.size() > (numThreads * 2)) numTasks = numThreads * 2;
    else if (points.size() > numThreads) numTasks = numThreads;

    std::unique_ptr<IndexPairListMapPtr[]> bins(new IndexPairListMapPtr[numTasks]);

    using BinOp = BinPointIndicesOp<PointArray, PointIndexType, VoxelOffsetType>;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numTasks),
        BinOp(bins.get(), points, voxelOffsets, xform, binLog2Dim, bucketLog2Dim,
            numTasks, cellCenteredTransform));

    std::set<Coord> uniqueCoords;

    for (size_t i = 0; i < numTasks; ++i) {
        IndexPairListMap& idxMap = *bins[i];
        for (typename IndexPairListMap::iterator it = idxMap.begin(); it != idxMap.end(); ++it) {
            uniqueCoords.insert(it->first);
        }
    }

    coords.assign(uniqueCoords.begin(), uniqueCoords.end());
    uniqueCoords.clear();

    size_t segmentCount = coords.size();

    using SegmentPtr = typename Array<PointIndexType>::Ptr;

    indexSegments.reset(new SegmentPtr[segmentCount]);
    offsetSegments.reset(new SegmentPtr[segmentCount]);

    using MergeOp = MergeBinsOp<PointIndexType>;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, segmentCount),
        MergeOp(bins.get(), indexSegments.get(), offsetSegments.get(), &coords[0], numTasks));
}


template<typename PointIndexType, typename VoxelOffsetType, typename PointArray>
inline void partition(
    const PointArray& points,
    const math::Transform& xform,
    const Index bucketLog2Dim,
    std::unique_ptr<PointIndexType[]>& pointIndices,
    std::unique_ptr<PointIndexType[]>& pageOffsets,
    std::unique_ptr<Coord[]>& pageCoordinates,
    PointIndexType& pageCount,
    std::unique_ptr<VoxelOffsetType[]>& voxelOffsets,
    bool recordVoxelOffsets,
    bool cellCenteredTransform)
{
    using SegmentPtr = typename Array<PointIndexType>::Ptr;

    if (recordVoxelOffsets) voxelOffsets.reset(new VoxelOffsetType[points.size()]);
    else  voxelOffsets.reset();

    const Index binLog2Dim = 5u;
    // note: Bins span a (2^(binLog2Dim + bucketLog2Dim))^3 voxel region,
    //       i.e. bucketLog2Dim = 3 and binLog2Dim = 5 corresponds to a
    //       (2^8)^3 = 256^3 voxel region.


    std::vector<Coord> segmentCoords;

    std::unique_ptr<SegmentPtr[]> indexSegments;
    std::unique_ptr<SegmentPtr[]> offsetSegments;

    binAndSegment<PointIndexType, VoxelOffsetType, PointArray>(points, xform,
        indexSegments, offsetSegments, segmentCoords, binLog2Dim, bucketLog2Dim,
            voxelOffsets.get(), cellCenteredTransform);

    size_t numSegments = segmentCoords.size();

    const tbb::blocked_range<size_t> segmentRange(0, numSegments);

    using IndexArray = std::unique_ptr<PointIndexType[]>;
    std::unique_ptr<IndexArray[]> pageOffsetArrays(new IndexArray[numSegments]);
    std::unique_ptr<IndexArray[]> pageIndexArrays(new IndexArray[numSegments]);

    const Index binVolume = 1u << (3u * binLog2Dim);

    tbb::parallel_for(segmentRange, OrderSegmentsOp<PointIndexType>
        (indexSegments.get(), offsetSegments.get(),
            pageOffsetArrays.get(), pageIndexArrays.get(), binVolume));

    indexSegments.reset();

    std::vector<Index> segmentOffsets;
    segmentOffsets.reserve(numSegments);

    pageCount = 0;
    for (size_t n = 0; n < numSegments; ++n) {
        segmentOffsets.push_back(pageCount);
        pageCount += pageOffsetArrays[n][0] - 1;
    }

    pageOffsets.reset(new PointIndexType[pageCount + 1]);

    PointIndexType count = 0;
    for (size_t n = 0, idx = 0; n < numSegments; ++n) {

        PointIndexType* offsets = pageOffsetArrays[n].get();
        size_t size = size_t(offsets[0]);

        for (size_t i = 1; i < size; ++i) {
            pageOffsets[idx++] = count;
            count += offsets[i];
        }
    }

    pageOffsets[pageCount] = count;

    pointIndices.reset(new PointIndexType[points.size()]);

    std::vector<PointIndexType*> indexArray;
    indexArray.reserve(numSegments);

    PointIndexType* index = pointIndices.get();
    for (size_t n = 0; n < numSegments; ++n) {
        indexArray.push_back(index);
        index += offsetSegments[n]->size();
    }

    // compute leaf node origin for each page

    pageCoordinates.reset(new Coord[pageCount]);

    tbb::parallel_for(segmentRange,
        [&](tbb::blocked_range<size_t>& range)
        {
            for (size_t n = range.begin(); n < range.end(); n++)
            {
                Index segmentOffset = segmentOffsets[n];
                PointIndexType* indices = pageIndexArrays[n].get();

                const Coord& segmentCoord = segmentCoords[n];

                // segment size stored in the first value of the offset array
                const size_t segmentSize = pageOffsetArrays[n][0] - 1;
                tbb::blocked_range<size_t> copyRange(0, segmentSize);
                tbb::parallel_for(copyRange,
                    [&](tbb::blocked_range<size_t>& r)
                    {
                        for (size_t i = r.begin(); i < r.end(); i++)
                        {
                            Index pageIndex = indices[i];
                            Coord& ijk = pageCoordinates[segmentOffset+i];

                            ijk[0] = pageIndex >> (2 * binLog2Dim);
                            Index pageIndexModulo = pageIndex - (ijk[0] << (2 * binLog2Dim));
                            ijk[1] = pageIndexModulo >> binLog2Dim;
                            ijk[2] = pageIndexModulo - (ijk[1] << binLog2Dim);

                            ijk = (ijk << bucketLog2Dim) + segmentCoord;
                        }
                    }
                );
            }
        }
    );

    // move segment data

    tbb::parallel_for(segmentRange,
        MoveSegmentDataOp<PointIndexType>(indexArray, offsetSegments.get()));
}


} // namespace point_partitioner_internal

/// @endcond

////////////////////////////////////////


template<typename PointIndexType, Index BucketLog2Dim>
inline PointPartitioner<PointIndexType, BucketLog2Dim>::PointPartitioner()
    : mPointIndices(nullptr)
    , mVoxelOffsets(nullptr)
    , mPageOffsets(nullptr)
    , mPageCoordinates(nullptr)
    , mPageCount(0)
    , mUsingCellCenteredTransform(true)
{
}


template<typename PointIndexType, Index BucketLog2Dim>
inline void
PointPartitioner<PointIndexType, BucketLog2Dim>::clear()
{
    mPageCount = 0;
    mUsingCellCenteredTransform = true;
    mPointIndices.reset();
    mVoxelOffsets.reset();
    mPageOffsets.reset();
    mPageCoordinates.reset();
}


template<typename PointIndexType, Index BucketLog2Dim>
inline void
PointPartitioner<PointIndexType, BucketLog2Dim>::swap(PointPartitioner& rhs)
{
    const IndexType tmpLhsPageCount = mPageCount;
    mPageCount = rhs.mPageCount;
    rhs.mPageCount = tmpLhsPageCount;

    mPointIndices.swap(rhs.mPointIndices);
    mVoxelOffsets.swap(rhs.mVoxelOffsets);
    mPageOffsets.swap(rhs.mPageOffsets);
    mPageCoordinates.swap(rhs.mPageCoordinates);

    bool lhsCellCenteredTransform = mUsingCellCenteredTransform;
    mUsingCellCenteredTransform = rhs.mUsingCellCenteredTransform;
    rhs.mUsingCellCenteredTransform = lhsCellCenteredTransform;
}


template<typename PointIndexType, Index BucketLog2Dim>
inline typename PointPartitioner<PointIndexType, BucketLog2Dim>::IndexIterator
PointPartitioner<PointIndexType, BucketLog2Dim>::indices(size_t n) const
{
    OPENVDB_ASSERT(bool(mPointIndices) && bool(mPageCount));
    return IndexIterator(
        mPointIndices.get() + mPageOffsets[n],
        mPointIndices.get() + mPageOffsets[n + 1]);
}


template<typename PointIndexType, Index BucketLog2Dim>
template<typename PointArray>
inline void
PointPartitioner<PointIndexType, BucketLog2Dim>::construct(
    const PointArray& points,
    const math::Transform& xform,
    bool voxelOrder,
    bool recordVoxelOffsets,
    bool cellCenteredTransform)
{
    mUsingCellCenteredTransform = cellCenteredTransform;

    point_partitioner_internal::partition(points, xform, BucketLog2Dim,
        mPointIndices, mPageOffsets, mPageCoordinates, mPageCount, mVoxelOffsets,
            (voxelOrder || recordVoxelOffsets), cellCenteredTransform);

    const tbb::blocked_range<size_t> pageRange(0, mPageCount);

    if (mVoxelOffsets && voxelOrder) {
        tbb::parallel_for(pageRange, point_partitioner_internal::VoxelOrderOp<
            IndexType, BucketLog2Dim>(mPointIndices, mPageOffsets, mVoxelOffsets));
    }

    if (mVoxelOffsets && !recordVoxelOffsets) {
        mVoxelOffsets.reset();
    }
}


template<typename PointIndexType, Index BucketLog2Dim>
template<typename PointArray>
inline typename PointPartitioner<PointIndexType, BucketLog2Dim>::Ptr
PointPartitioner<PointIndexType, BucketLog2Dim>::create(
    const PointArray& points,
    const math::Transform& xform,
    bool voxelOrder,
    bool recordVoxelOffsets,
    bool cellCenteredTransform)
{
    Ptr ret(new PointPartitioner());
    ret->construct(points, xform, voxelOrder, recordVoxelOffsets, cellCenteredTransform);
    return ret;
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_PARTITIONER_HAS_BEEN_INCLUDED
