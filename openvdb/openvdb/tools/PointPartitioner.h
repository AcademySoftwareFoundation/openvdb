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

#include <boost/integer.hpp> // boost::int_t<N>::least

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include <algorithm>
#include <cmath> // for std::isfinite()
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
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
template<typename PointIndexType = uint32_t, Index BucketLog2Dim = 3u,
         typename PointOffsetType = PointIndexType>
class PointPartitioner
{
public:
    enum { LOG2DIM = BucketLog2Dim };

    using Ptr = SharedPtr<PointPartitioner>;
    using ConstPtr = SharedPtr<const PointPartitioner>;

    using IndexType = PointIndexType;
    using OffsetType = PointOffsetType;
    using VoxelOffsetType = typename boost::int_t<1 + (3 * BucketLog2Dim)>::least;
    using VoxelOffsetArray = std::unique_ptr<VoxelOffsetType[]>;

    class IndexIterator;
    class ThreadLocalBin;

    using ThreadLocalBins = std::vector<ThreadLocalBin>;

    //////////

    PointPartitioner() = default;
    explicit PointPartitioner(bool threaded) : mThreaded(threaded) { }


    /// @brief  Partitions point indices into @c BucketLog2Dim aligned buckets.
    ///
    /// @param points                 list of world space points.
    /// @param xform                  world to index space transform.
    /// @param voxelOrder             sort point indices by local voxel offsets.
    /// @param recordVoxelOffsets     construct local voxel offsets
    /// @param cellCenteredTransform  toggle the cell-centered interpretation that imagines world
    ///                               space as divided into discrete cells (e.g., cubes) centered
    ///                               on the image of the index-space lattice points.
    template<typename PointArrayType>
    void construct(const PointArrayType& points, const math::Transform& xform,
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
    template<typename PointArrayType>
    static Ptr create(const PointArrayType& points, const math::Transform& xform,
        bool voxelOrder = false, bool recordVoxelOffsets = false,
        bool cellCenteredTransform = true, bool threaded = true);

    template<Index BinLog2Dim = 5u>
    void construct(ThreadLocalBins& bins, size_t pointCount,
        bool cellCenteredTransform = true);

    template<Index BinLog2Dim = 5u>
    static Ptr create(ThreadLocalBins& bins, size_t pointCount,
        bool cellCenteredTransform = true, bool threaded = true);

    template<typename IndexMappingType>
    void sort(std::unique_ptr<Index[]>& histogram,
        const IndexMappingType& indexMapping);

    /// @brief Returns the number of buckets.
    size_t size() const { return mPageCount; }

    /// @brief true if the container size is 0, false otherwise.
    bool empty() const { return mPageCount == 0; }

    /// @brief Removes all data and frees up memory.
    void clear();

    /// @brief Clears the page coordinates.
    void clearPageCoordinates() { mPageCoordinates.reset(); }

    /// @brief Clears the page offsets.
    void clearPageOffsets() { mPageOffsets.reset(); }

    /// @brief Clears the point indices.
    void clearPointIndices() { mPointIndices.reset(); }

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

    std::unique_ptr<OffsetType[]>   mPageOffsets;
    std::unique_ptr<Coord[]>        mPageCoordinates;
    Index                           mPageCount = 0;
    bool                            mUsingCellCenteredTransform = true;
    bool                            mThreaded = true;
}; // class PointPartitioner


using UInt32PointPartitioner = PointPartitioner<uint32_t, 3>;


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
class PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::IndexIterator
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
    IndexType& operator*() { assert(mItem != nullptr); return *mItem; }
    const IndexType& operator*() const { assert(mItem != nullptr); return *mItem; }

    /// @brief  Returns a pointer to the item to which this iterator is pointing.
    IndexType* operator->() { return &(this->operator*()); }
    const IndexType* operator->() const { return &(this->operator*()); }

    /// @brief  Return @c true if this iterator is not yet exhausted.
    operator bool() const { return mItem < mEnd; }
    bool test() const { return mItem < mEnd; }

    /// @brief  Advance to the next item.
    IndexIterator& operator++() { assert(this->test()); ++mItem; return *this; }

    /// @brief  Advance iterator forwards a number of steps.
    IndexIterator& operator+=(size_t pos) { assert(this->test()); mItem += pos; return *this; }

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


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
class PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::ThreadLocalBin
{
public:
    using IndexType = PointIndexType;
    using OffsetType = PointOffsetType;
    using VoxelOffsetType = PointPartitioner::VoxelOffsetType;
    using IndexPair = std::pair<PointIndexType, PointOffsetType>;
    using IndexPairList = std::deque<IndexPair>;
    using IndexPairListPtr = std::unique_ptr<IndexPairList>;
    using IndexPairListMap = std::unordered_map<Coord, IndexPairListPtr>;
    using IndexPairListMapPtr = std::unique_ptr<IndexPairListMap>;

    ThreadLocalBin()
        : mData(new IndexPairListMap) { }

    explicit ThreadLocalBin(IndexPairListMap* data)
        : mData(data) { }

    IndexPairListMap& map() { return *mData; }
    const IndexPairListMap& constMap() const { return *mData; }

// private:
    IndexPairListMapPtr mData;
}; // class PointPartitioner::ThreadLocalBin


////////////////////////////////////////
////////////////////////////////////////

// Implementation details


namespace point_partitioner_internal {


template <typename VoxelOffsetT>
struct LinearIndexMapping
{
    LinearIndexMapping(const std::unique_ptr<VoxelOffsetT[]>& voxelOffsets)
        : mVoxelOffsets(voxelOffsets) { }

    template <typename VoxelOffsetType, typename IndexT>
    VoxelOffsetType voxelOffset(const IndexT& index) const
    {
        return static_cast<VoxelOffsetType>(mVoxelOffsets[index]);
    }

    template <typename IndexT>
    IndexT index(const IndexT& index) const
    {
        return index;
    }

private:
    const std::unique_ptr<VoxelOffsetT[]>& mVoxelOffsets;
}; // struct LinearIndexMapping


template<typename IndexType, typename OffsetType>
struct Bin
{
    struct IndexOffsetArray
    {
        size_t size = 0;
        std::unique_ptr<IndexType[]> indices;
        std::unique_ptr<OffsetType[]> offsets;
    };

    struct OffsetOffsetArray
    {
        size_t size = 0;
        std::unique_ptr<OffsetType[]> indices;
        std::unique_ptr<OffsetType[]> offsets;
    };

    explicit Bin(const Coord& ijk)
        : ijk(ijk) { }

    Coord                   ijk;
    IndexOffsetArray        bin;
    OffsetOffsetArray        bucket;
}; // struct Bin


////////////////////////////////////////


template<Index BucketLog2Dim, Index BinLog2Dim,
    typename PointIndexType, typename PointOffsetType, typename VoxelOffsetType, typename PointArray>
inline void computeBins(
    typename PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::ThreadLocalBins& bins,
    const PointArray& points,
    const math::Transform& xform,
    bool cellCenteredTransform,
    std::unique_ptr<VoxelOffsetType[]>& voxelOffsets,
    bool recordVoxelOffsets,
    bool threaded)
{
    using PosType = typename PointArray::PosType;
    using ThreadLocalBin = typename PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::ThreadLocalBin;
    using IndexPairList = typename ThreadLocalBin::IndexPairList;
    using IndexPairListMap = typename ThreadLocalBin::IndexPairListMap;

    static constexpr Index BucketLog2Dim2 = 2 * BucketLog2Dim;
    static constexpr Index BucketMask = (1u << BucketLog2Dim) - 1u;
    static constexpr Index BinLog2dim2 = 2 * BinLog2Dim;
    static constexpr Index BinMask = (1u << (BucketLog2Dim + BinLog2Dim)) - 1u;
    static constexpr Index InvBinMask = ~BinMask;

    if (recordVoxelOffsets) voxelOffsets.reset(new VoxelOffsetType[points.size()]);
    else  voxelOffsets.reset();

    const size_t numPoints = points.size();

    size_t numTasks = 1;
    if (threaded) {
        size_t numThreads = size_t(tbb::task_scheduler_init::default_num_threads());
        if (numPoints > (numThreads * 2))   numTasks = numThreads * 2;
        else if (numPoints > numThreads)    numTasks = numThreads;
    }

    bins.resize(numTasks);

    // build task queue

    auto computeBinsOp = [&](tbb::blocked_range<size_t>& r) {
        Coord ijk(0, 0, 0), loc(0, 0, 0), binCoord(0, 0, 0), lastBinCoord(1, 2, 3);
        PosType pos;

        PointIndexType bucketOffset = 0;
        VoxelOffsetType voxelOffset = 0;

        IndexPairList* idxList = nullptr;

        const size_t taskSize = numPoints / numTasks;

        for (size_t n = r.begin(); n < r.end(); n++) {
            IndexPairListMap& idxMap = bins[n].map();

            const bool isLastTask = (n + 1) >= numTasks;

            const size_t start = n * taskSize;
            const size_t end = isLastTask ? numPoints : (start + taskSize);

            for (size_t i = start; i != end; ++i) {

                points.getPos(i, pos);

                if (std::isfinite(pos[0]) && std::isfinite(pos[1]) && std::isfinite(pos[2])) {
                    ijk = cellCenteredTransform ? xform.worldToIndexCellCentered(pos) :
                        xform.worldToIndexNodeCentered(pos);

                    if (recordVoxelOffsets) {
                        loc[0] = ijk[0] & BucketMask;
                        loc[1] = ijk[1] & BucketMask;
                        loc[2] = ijk[2] & BucketMask;
                        voxelOffset = VoxelOffsetType(
                            (loc[0] << BucketLog2Dim2) +
                            (loc[1] << BucketLog2Dim) +
                            loc[2]);
                    }

                    binCoord[0] = ijk[0] & InvBinMask;
                    binCoord[1] = ijk[1] & InvBinMask;
                    binCoord[2] = ijk[2] & InvBinMask;

                    ijk[0] &= BinMask;
                    ijk[1] &= BinMask;
                    ijk[2] &= BinMask;

                    ijk[0] >>= BucketLog2Dim;
                    ijk[1] >>= BucketLog2Dim;
                    ijk[2] >>= BucketLog2Dim;

                    bucketOffset = PointIndexType(
                        (ijk[0] << BinLog2dim2) +
                        (ijk[1] << BinLog2Dim) +
                        ijk[2]);

                    if (lastBinCoord != binCoord) {
                        lastBinCoord = binCoord;
                        auto& idxPtr = idxMap[binCoord];
                        if (!idxPtr)    idxPtr.reset(new IndexPairList);
                        idxList = idxPtr.get();
                    }

                    idxList->emplace_back(PointIndexType(i), bucketOffset);
                    if (recordVoxelOffsets) voxelOffsets[i] = voxelOffset;
                }
            }
        }
    };

    tbb::blocked_range<size_t> range(0, numTasks);
    if (threaded) {
        tbb::parallel_for(range, computeBinsOp);
    } else {
        computeBinsOp(range);
    }
}


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
inline void mergeBins(
    std::vector<Bin<PointIndexType, PointOffsetType>>& partitions,
    typename PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::ThreadLocalBins& bins,
    bool threaded)
{
    using ThreadLocalBin = typename PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::ThreadLocalBin;
    using IndexPair = typename ThreadLocalBin::IndexPair;
    using IndexPairList = typename ThreadLocalBin::IndexPairList;
    using IndexPairListPtr = typename ThreadLocalBin::IndexPairListPtr;

    { // allocate one partition per bin coordinate

        // extract unique bin coordinates

        std::set<Coord> coords;
        for (const ThreadLocalBin& bin : bins) {
            for (const auto& element : bin.constMap()) {
                coords.insert(element.first);
            }
        }

        // allocate partitions

        partitions.reserve(coords.size());
        for (const Coord& coord : coords) {
            partitions.emplace_back(coord);
        }
    }

    // merge bins into partitions

    auto mergeOp = [&](tbb::blocked_range<size_t>& range) {
        std::vector<IndexPairList*> data;
        std::vector<PointOffsetType> arrayOffsets;
        arrayOffsets.reserve(partitions.size());

        for (size_t n = range.begin(); n < range.end(); n++) {

            Bin<PointIndexType, PointOffsetType>& partition = partitions[n];

            size_t numIndices = 0;

            data.clear();

            for (ThreadLocalBin& bin : bins) {
                auto iter = bin.map().find(partition.ijk);

                if (iter != bin.map().end()) {
                    IndexPairListPtr& list = iter->second;
                    assert(list);
                    data.push_back(list.get());
                    numIndices += list->size();
                }
            }

            if (data.empty() || numIndices == 0) continue;

            partition.bin.indices.reset(new PointIndexType[numIndices]);
            partition.bin.offsets.reset(new PointOffsetType[numIndices]);
            partition.bin.size = numIndices;

            arrayOffsets.clear();

            size_t count = 0;
            for (const IndexPairList* idxList : data) {
                arrayOffsets.emplace_back(count);
                count += idxList->size();
            }

            // copy data

            auto copyDataOp = [&](tbb::blocked_range<size_t>& r) {
                PointIndexType* indices = partition.bin.indices.get();
                PointOffsetType* offsets = partition.bin.offsets.get();

                for (size_t i = r.begin(); i < r.end(); i++) {
                    const PointOffsetType arrayOffset = arrayOffsets[i];

                    PointIndexType* indexPtr = &indices[arrayOffset];
                    PointOffsetType* offsetPtr = &offsets[arrayOffset];

                    IndexPairList& list = *data[i];

                    for (auto it = list.begin(), end = list.end(); it != end; ++it) {
                        const IndexPair& element = *it;

                        *indexPtr++ = element.first;
                        *offsetPtr++ = element.second;
                    }

                    list.clear();
                }
            };

            tbb::blocked_range<size_t> copyRange(0, data.size());
            if (threaded) {
                tbb::parallel_for(copyRange, copyDataOp);
            } else {
                copyDataOp(copyRange);
            }
        }
    };

    tbb::blocked_range<size_t> range(0, partitions.size());
    if (threaded) {
        tbb::parallel_for(range, mergeOp);
    } else {
        mergeOp(range);
    }

    // clear all bins

    bins.clear();
}


template<Index BinLog2Dim, typename PointIndexType, typename PointOffsetType>
inline void orderPartitions(
    std::vector<Bin<PointIndexType, PointOffsetType>>& partitions, bool threaded)
{
    static constexpr Index BinVolume = 1u << (3u * BinLog2Dim);

    using IndexArray = std::unique_ptr<PointIndexType[]>;
    using OffsetArray = std::unique_ptr<PointOffsetType[]>;

    auto orderOp = [&](tbb::blocked_range<size_t>& range) {
        std::array<PointOffsetType, BinVolume> localBucketCounters;

        size_t maxBinSize = 0;
        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
            maxBinSize = std::max(maxBinSize, partitions[n].bin.size);
        }

        OffsetArray localBucketOffsets(new PointOffsetType[maxBinSize]);
        IndexArray localBinIndices(new PointIndexType[maxBinSize]);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            localBucketCounters.fill(0);

            Bin<PointIndexType, PointOffsetType>& partition = partitions[n];

            const size_t binSize = partition.bin.size;
            PointOffsetType* binOffsets = partition.bin.offsets.get();

            // Count the number of points per bucket and assign a local bucket index
            // to each point.
            for (size_t i = 0; i < binSize; ++i) {
                localBucketOffsets[i] = localBucketCounters[binOffsets[i]]++;
            }

            PointOffsetType nonemptyBucketCount = 0;
            for (size_t i = 0; i < BinVolume; ++i) {
                nonemptyBucketCount += static_cast<PointOffsetType>(localBucketCounters[i] != 0);
            }

            partition.bucket.offsets.reset(new PointOffsetType[nonemptyBucketCount + 1]);
            partition.bucket.indices.reset(new PointOffsetType[nonemptyBucketCount]);
            partition.bucket.offsets[0] = nonemptyBucketCount + 1;
            partition.bucket.size = nonemptyBucketCount;

            // Compute bucket counter prefix sum
            PointOffsetType count = 0, idx = 0;
            for (size_t i = 0; i < BinVolume; ++i) {
                if (localBucketCounters[i] != 0) {
                    partition.bucket.indices[idx] = Index(i);
                    partition.bucket.offsets[idx+1] = localBucketCounters[i];
                    localBucketCounters[i] = count;
                    count += partition.bucket.offsets[idx+1];
                    ++idx;
                }
            }

            // PointIndexType* indices = indexSegments[n]->data();
            PointIndexType* binIndices = partition.bin.indices.get();
            tbb::blocked_range<size_t> binSizeRange(0, binSize);

            // Compute final point order by incrementing the local bucket point index
            // with the prefix sum offset.
            auto incrementOp = [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                    localBucketOffsets[n] += localBucketCounters[binOffsets[n]];
                }
            };
            if (threaded) {
                tbb::parallel_for(binSizeRange, incrementOp);
            } else {
                incrementOp(binSizeRange);
            }

            partition.bin.offsets.reset();

            for (size_t i = 0; i < binSize; i++) {
                localBinIndices[i] = binIndices[i];
            }

            auto copyIndicesOp = [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t n = range.begin(), N = range.end(); n != N; ++n) {
                    binIndices[localBucketOffsets[n]] = localBinIndices[n];
                }
            };
            if (threaded) {
                tbb::parallel_for(binSizeRange, copyIndicesOp);
            } else {
                copyIndicesOp(binSizeRange);
            }
        }
    };

    tbb::blocked_range<size_t> range(0, partitions.size());
    if (threaded) {
        tbb::parallel_for(range, orderOp);
    } else {
        orderOp(range);
    }
}


template<Index BucketLog2Dim, Index BinLog2Dim, typename PointIndexType, typename PointOffsetType>
inline void computePageCoordinates(
    std::unique_ptr<Coord[]>& bucketCoordinates,
    Index& bucketCount,
    std::vector<Bin<PointIndexType, PointOffsetType>>& bins,
    bool threaded)
{
    std::vector<Index> bucketOffsets;
    bucketOffsets.reserve(bins.size());

    size_t numBins = bins.size();

    bucketCount = 0;
    for (size_t n = 0; n < numBins; ++n) {
        bucketOffsets.push_back(bucketCount);
        bucketCount += static_cast<Index>(bins[n].bucket.size);
    }

    // compute leaf node origin for each bucket

    bucketCoordinates.reset(new Coord[bucketCount]);

    auto computeLeafOriginOp = [&](tbb::blocked_range<size_t>& range)
    {
        for (size_t n = range.begin(); n < range.end(); n++)
        {
            Bin<PointIndexType, PointOffsetType>& bin = bins[n];

            Index bucketOffset = bucketOffsets[n];
            PointOffsetType* indices = bin.bucket.indices.get();

            const Coord& binCoord = bin.ijk;

            auto computeOriginOp = [&](tbb::blocked_range<size_t>& subrange)
            {
                for (size_t i = subrange.begin(); i < subrange.end(); i++)
                {
                    Index bucketIndex = indices[i];
                    Coord& ijk = bucketCoordinates[bucketOffset+i];

                    ijk[0] = bucketIndex >> (2 * BinLog2Dim);
                    Index bucketIndexModulo = bucketIndex - (ijk[0] << (2 * BinLog2Dim));
                    ijk[1] = bucketIndexModulo >> BinLog2Dim;
                    ijk[2] = bucketIndexModulo - (ijk[1] << BinLog2Dim);

                    ijk = (ijk << BucketLog2Dim) + binCoord;
                }
            };
            tbb::blocked_range<size_t> bucketRange(0, bin.bucket.size);
            if (threaded) {
                tbb::parallel_for(bucketRange, computeOriginOp);
            } else {
                computeOriginOp(bucketRange);
            }
        }
    };
    tbb::blocked_range<size_t> binRange(0, bins.size());
    if (threaded) {
        tbb::parallel_for(binRange, computeLeafOriginOp);
    } else {
        computeLeafOriginOp(binRange);
    }
}


template<Index BucketLog2Dim, Index BinLog2Dim, typename PointIndexType, typename PointOffsetType>
inline void linearizePages(
    std::unique_ptr<PointIndexType[]>& pointIndices,
    std::unique_ptr<PointOffsetType[]>& bucketOffsets,
    std::vector<Bin<PointIndexType, PointOffsetType>>& partitions,
    size_t pointCount,
    size_t bucketCount,
    bool threaded)
{
    tbb::blocked_range<size_t> binRange(0, partitions.size());

    // move bin data

    bucketOffsets.reset(new PointOffsetType[bucketCount + 1]);

    PointOffsetType count = 0;
    for (size_t n = 0, idx = 0; n < partitions.size(); ++n) {

        PointOffsetType* offsets = partitions[n].bucket.offsets.get();
        size_t size = size_t(offsets[0]);

        for (size_t i = 1; i < size; ++i) {
            bucketOffsets[idx++] = count;
            count += offsets[i];
        }
    }

    auto resetOp = [&](tbb::blocked_range<size_t>& range)
    {
        for (size_t n = range.begin(); n < range.end(); n++) {
            partitions[n].bucket.offsets.reset();
        }
    };
    if (threaded) {
        tbb::parallel_for(binRange, resetOp);
    } else {
        resetOp(binRange);
    }

    bucketOffsets[bucketCount] = count;

    pointIndices.reset(new PointIndexType[pointCount]);

    std::vector<PointIndexType*> indexArray;
    indexArray.reserve(partitions.size());

    PointIndexType* index = pointIndices.get();
    for (size_t n = 0; n < partitions.size(); ++n) {
        indexArray.push_back(index);
        index += partitions[n].bin.size;
    }

    auto linearizeOp = [&](tbb::blocked_range<size_t>& range)
    {
        for (size_t n = range.begin(); n < range.end(); n++) {
            PointIndexType* lhs = indexArray[n];
            const PointIndexType* rhs = partitions[n].bin.indices.get();

            tbb::blocked_range<size_t> copyRange(0, partitions[n].bin.size);
            tbb::parallel_for(copyRange,
                [&](tbb::blocked_range<size_t>& r)
                {
                    for (size_t i = r.begin(); i < r.end(); i++) {
                        lhs[i] = rhs[i];
                    }
                }
            );

            partitions[n].bin.indices.reset();
        }
    };
    if (threaded) {
        tbb::parallel_for(binRange, linearizeOp);
    } else {
        linearizeOp(binRange);
    }
}


template<Index BucketLog2Dim, typename PointIndexType, typename PointOffsetType, typename VoxelOffsetType, typename IndexMappingType>
inline void sortIndices(
    std::unique_ptr<PointIndexType[]>& pointIndices,
    std::unique_ptr<Index[]>& histograms,
    const std::unique_ptr<PointOffsetType[]>& bucketOffsets,
    const Index& bucketCount,
    const IndexMappingType& indexMapping,
    bool threaded)
{
    static constexpr PointOffsetType VoxelCount = 1 << (3 * BucketLog2Dim);

    tbb::blocked_range<size_t> bucketRange(0, bucketCount);
    auto sortOp = [&](tbb::blocked_range<size_t>& range)
    {
        PointOffsetType pointCount = 0;
        for (size_t n = range.begin(); n < range.end(); n++) {
            pointCount = std::max(pointCount, (bucketOffsets[n + 1] - bucketOffsets[n]));
        }

        // allocate histogram buffers
        std::unique_ptr<VoxelOffsetType[]> offsets(new VoxelOffsetType[pointCount]);
        std::unique_ptr<PointIndexType[]> sortedIndices(new PointIndexType[pointCount]);
        std::unique_ptr<Index[]> histogram;

        Index* histogramPtr = nullptr;
        if (!histograms) {
            histogram.reset(new Index[VoxelCount]);
            histogramPtr = histogram.get();
        }

        for (size_t n = range.begin(); n < range.end(); n++) {

            PointIndexType * const indices = pointIndices.get() + bucketOffsets[n];
            pointCount = bucketOffsets[n + 1] - bucketOffsets[n];

            // local copy of voxel offsets.
            for (PointOffsetType i = 0; i < pointCount; ++i) {
                offsets[i] = indexMapping.template voxelOffset<VoxelOffsetType>(indices[i]);
            }

            if (histograms) {
                histogramPtr = &histograms[n*VoxelCount];
            }

            // reset histogram
            memset(histogramPtr, 0, sizeof(Index) * VoxelCount);

            // compute histogram
            for (PointOffsetType i = 0; i < pointCount; ++i) {
                ++histogramPtr[ offsets[i] ];
            }

            // turn histogram into cumulative histogram

            PointOffsetType count = 0, startOffset;
            for (int i = 0; i < int(VoxelCount); ++i) {
                if (histogramPtr[i] > 0) {
                    startOffset = count;
                    count += histogramPtr[i];
                    histogramPtr[i] = startOffset;
                } else {
                    histogramPtr[i] = count;
                }
            }

            // sort indices based on voxel offset
            for (PointOffsetType i = 0; i < pointCount; ++i) {
                sortedIndices[ histogramPtr[ offsets[i] ]++ ] = indexMapping.index(indices[i]);
            }

            memcpy(&indices[0], &sortedIndices[0], sizeof(PointIndexType) * pointCount);
        }
    };
    if (threaded) {
        tbb::parallel_for(bucketRange, sortOp);
    } else {
        sortOp(bucketRange);
    }
}


////////////////////////////////////////


} // namespace point_partitioner_internal


////////////////////////////////////////


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
inline void
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::clear()
{
    mPageCount = 0;
    mUsingCellCenteredTransform = true;
    mPointIndices.reset();
    mVoxelOffsets.reset();
    mPageOffsets.reset();
    mPageCoordinates.reset();
}


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
inline void
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::swap(PointPartitioner& rhs)
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


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
inline typename PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::IndexIterator
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::indices(size_t n) const
{
    assert(bool(mPointIndices) && bool(mPageCount));
    return IndexIterator(
        mPointIndices.get() + mPageOffsets[n],
        mPointIndices.get() + mPageOffsets[n + 1]);
}


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
template<typename PointArrayType>
inline void
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::construct(
    const PointArrayType& points,
    const math::Transform& xform,
    bool voxelOrder,
    bool recordVoxelOffsets,
    bool cellCenteredTransform)
{
    using namespace point_partitioner_internal;

    const Index BinLog2Dim = 5u;
    // note: Bins span a (2^(BinLog2Dim + BucketLog2Dim))^3 voxel region,
    //       i.e. BucketLog2Dim = 3 and BinLog2Dim = 5 corresponds to a
    //       (2^8)^3 = 256^3 voxel region.

    ThreadLocalBins bins;
    computeBins<BucketLog2Dim, BinLog2Dim, PointIndexType, PointOffsetType>(
        bins, points, xform, cellCenteredTransform, mVoxelOffsets, recordVoxelOffsets, mThreaded);

    this->construct(bins, points.size(), cellCenteredTransform);

    if (mVoxelOffsets) {
        if (voxelOrder) {
            std::unique_ptr<Index[]> histogram;
            LinearIndexMapping<VoxelOffsetType> indexMapping(mVoxelOffsets);
            this->sort(histogram, indexMapping);
        }
        if (!recordVoxelOffsets) {
            mVoxelOffsets.reset();
        }
    }
}


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
template<typename PointArrayType>
inline typename PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::Ptr
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::create(
    const PointArrayType& points,
    const math::Transform& xform,
    bool voxelOrder,
    bool recordVoxelOffsets,
    bool cellCenteredTransform,
    bool threaded)
{
    Ptr ret(new PointPartitioner(threaded));
    ret->construct(points, xform, voxelOrder, recordVoxelOffsets, cellCenteredTransform);
    return ret;
}


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
template<Index BinLog2Dim>
inline void
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::construct(
    ThreadLocalBins& bins,
    size_t pointCount,
    bool cellCenteredTransform)
{
    using namespace point_partitioner_internal;

    using PartitionType = Bin<PointIndexType, PointOffsetType>;
    using PartitionsType = std::vector<PartitionType>;

    mUsingCellCenteredTransform = cellCenteredTransform;

    PartitionsType partitions;
    mergeBins<PointIndexType, BucketLog2Dim, PointOffsetType>(partitions, bins, mThreaded);

    orderPartitions<BinLog2Dim>(partitions, mThreaded);

    // compute the local coordinate for each partition

    computePageCoordinates<BucketLog2Dim, BinLog2Dim, PointIndexType, PointOffsetType>(
        mPageCoordinates, mPageCount, partitions, mThreaded);

    // move the segment data

    linearizePages<BucketLog2Dim, BinLog2Dim, PointIndexType, PointOffsetType>(
        mPointIndices, mPageOffsets, partitions, pointCount, mPageCount, mThreaded);

    partitions.clear();
}


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
template<typename IndexMappingType>
void
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::sort(
    std::unique_ptr<Index[]>& histogram,
    const IndexMappingType& indexMapping)
{
    using namespace point_partitioner_internal;

    sortIndices<BucketLog2Dim, PointIndexType, PointOffsetType, VoxelOffsetType, IndexMappingType>(
        mPointIndices, histogram, mPageOffsets, mPageCount, indexMapping, mThreaded);
}


template<typename PointIndexType, Index BucketLog2Dim, typename PointOffsetType>
template<Index BinLog2Dim>
inline typename PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::Ptr
PointPartitioner<PointIndexType, BucketLog2Dim, PointOffsetType>::create(
    ThreadLocalBins& bins,
    size_t pointCount,
    bool cellCenteredTransform,
    bool threaded)
{
    Ptr ret(new PointPartitioner(threaded));
    ret->construct(bins, pointCount, cellCenteredTransform);
    return ret;
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_PARTITIONER_HAS_BEEN_INCLUDED
