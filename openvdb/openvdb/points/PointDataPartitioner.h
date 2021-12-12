// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file PointDataPartitioner.h
///
/// @author Dan Bailey
///
/// @brief   Spatially partitions point data grids using a parallel
///          radix-based sorting algorithm.
///
/// @details Performs a stable deterministic sort; partitioning the same
///          point sequence will produce the same result each time.
/// @details The algorithm is unbounded meaning that points may be
///          distributed anywhere in index space.
/// @details The actual points are never stored in the tool, only
///          offsets into an external array.
///
/// Deformers used when moving points are in world space by default and must adhere
/// to the interface described in the example below:
/// @code
/// struct MyDeformer
/// {
///     // A reset is performed on each leaf in turn before the points in that leaf are
///     // deformed. A leaf and leaf index (standard leaf traversal order) are supplied as
///     // the arguments, which matches the functor interface for LeafManager::foreach().
///     template <typename LeafNoteType>
///     void reset(LeafNoteType& leaf, size_t idx);
///
///     // Evaluate the deformer and modify the given position to generate the deformed
///     // position. An index iterator is supplied as the argument to allow querying the
///     // point offset or containing voxel coordinate.
///     template <typename IndexIterT>
///     void apply(Vec3d& position, const IndexIterT& iter) const;
/// };
/// @endcode
///
/// @note The DeformerTraits struct (defined in PointMask.h) can be used to configure
/// a deformer to evaluate in index space.

#ifndef OPENVDB_POINTS_POINT_DATA_PARTITIONER_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_DATA_PARTITIONER_HAS_BEEN_INCLUDED

#include <openvdb/tools/PointPartitioner.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointCount.h>

#include <openvdb/util/CpuTimer.h>

#include <tbb/task_group.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {


template<typename PointDataGridT>
struct PointDataLeafArray;


template<typename PointDataGridT>
class PointDataPartitioner
{
public:
    using Ptr = SharedPtr<PointDataPartitioner>;
    using ConstPtr = SharedPtr<const PointDataPartitioner>;
    using PointDataTreeT = typename PointDataGridT::TreeType;
    using PointDataLeafT = typename PointDataTreeT::LeafNodeType;
    using PointDataLeafBufferT = typename PointDataLeafT::Buffer;
    using ValueT = typename PointDataTreeT::ValueType;
    using PointDataGridPtrArrayT = std::vector<typename PointDataGridT::Ptr>;

    static constexpr Index NUM_VOXELS = PointDataLeafT::NUM_VOXELS;
    static constexpr Index LOG2DIM = PointDataLeafT::LOG2DIM;
    static constexpr Index INVALID_INDEX = std::numeric_limits<Index>::max();

    using PointPartitionerT = tools::PointPartitioner<
        uint64_t, PointDataLeafT::LOG2DIM, uint32_t>;

    class LocalMove;

    using LocalMovePtr = std::unique_ptr<LocalMove>;
    using LocalMovePtrArray = std::vector<LocalMovePtr>;
    using LocalBin = std::vector<LocalMovePtrArray>;

    class IndexIterator;

    //////////

    PointDataPartitioner() = default;

    template <typename DeformerT = NullDeformer, typename FilterT = NullFilter, typename OtherFilterT = NullFilter>
    void construct(
        PointDataLeafArray<PointDataGridT>& sourceLeafArray,
        const DeformerT& deformer = NullDeformer(),
        const FilterT& filter = NullFilter(),
        bool updatePosition = true,
        const OtherFilterT& otherFilter = NullFilter());

    void linearize();

    void clear();

    size_t size() const { return mPageCount; }

    size_t memUsage() const;

    size_t size(size_t n) const;

    /// @brief Returns the origin coordinate for bucket @a n
    const Coord& origin(size_t n) const  { return mPageCoordinates[n]; }

    Index stealableLeafIndex(size_t n) const;

    IndexIterator indices(size_t n) const;

    std::unique_ptr<Index[]>& sourceGridIndices() { return mSourceGridIndices; }

    template<typename VoxelBuffersT>
    void assignVoxelValues(VoxelBuffersT& voxelBuffers, bool cleanup = true);

private:
    // Disallow copying
    PointDataPartitioner(const PointDataPartitioner&);
    PointDataPartitioner& operator=(const PointDataPartitioner&);

    std::vector<std::pair<size_t, size_t>>  mBucketIndices;
    std::unique_ptr<Coord[]>                mPageCoordinates;
    Index                                   mPageCount = 0;
    std::unique_ptr<Index[]>                mSourceGridIndices;
    typename PointPartitionerT::Ptr         mPartitioner;
    LocalBin                                mLocalBin;
    std::unique_ptr<Index[]>                mHistogram;

    std::unique_ptr<Index64[]>              mPageOffsets;
    std::unique_ptr<Index64[]>              mPageIndices;
}; // class PointDataPartitioner


template<typename PointDataGridT>
class PointDataPartitioner<PointDataGridT>::LocalMove
{
public:
    using VoxelArray = std::array<Index, PointDataPartitioner::NUM_VOXELS>;
    using PointArray = std::vector<Index>;

    static constexpr Index NUM_VOXELS =
        PointDataPartitioner::NUM_VOXELS;

    inline size_t memUsage() const
    {
        return sizeof(*this) + sizeof(Index) * points.capacity();
    }

    bool hasSamePointOrder = false;
    Index index;
    VoxelArray voxels;
    PointArray points;
}; // class PointDataPartitioner::LocalMove


template<typename PointDataGridT>
class PointDataPartitioner<PointDataGridT>::IndexIterator
{
public:
    IndexIterator(Index64* begin = nullptr, Index64* end = nullptr)
        : mBegin(begin), mEnd(end), mItem(begin) {}

    /// @brief Rewind to first item.
    void reset() { mItem = mBegin; }

    /// @brief  Number of point indices in the iterator range.
    size_t size() const { return mEnd - mBegin; }

    Index sourceBufferIndex() const { return static_cast<Index>(*mItem >> 32); }
    Index sourceIndex() const { return static_cast<Index>(*mItem); }

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
    Index64 * const mBegin, * const mEnd;
    Index64 * mItem;
}; // class PointDataPartitioner::IndexIterator



namespace point_data_partitioner_internal {


template <Index BucketLog2DimT>
struct PointDataMapping
{
    PointDataMapping() = default;

    template <typename VoxelOffsetType, typename IndexT>
    VoxelOffsetType voxelOffset(const IndexT& index) const
    {
        static constexpr VoxelOffsetType VoxelOffsetMask = (1u << (3 * BucketLog2DimT)) - 1u;

        return static_cast<VoxelOffsetType>(index) & VoxelOffsetMask;
    }

    template <typename IndexT>
    IndexT index(const IndexT& index) const
    {
        return (IndexT(static_cast<Index>(index >> 32)) << 32) |
                IndexT(static_cast<Index>(index) >> (1u + (3 * BucketLog2DimT)));
    }
}; // struct PointDataMapping


// A simple task scheduler that assigns tasks to the thread that owns them until the
// thread runs out of work, then resorts to thread-stealing to load balance.
// This is designed to address the tradeoff between optimizing for tail latency and
// maintaining determinism in the specific case where this is prohibitively expensive.
class TaskScheduler
{
    using AtomicBool = std::atomic<bool>;
    using AtomicIndex = std::atomic<Int64>;

public:
    TaskScheduler() = default;

    template <typename TaskIndexArrayT>
    void initialize(const TaskIndexArrayT& indexArray)
    {
        if (indexArray.empty()) {
            this->reset();
            return;
        }

        mThreadCount = indexArray.size();

        // allocate arrays

        mStolenTasks.reset(new bool[mThreadCount]);
        mTaskArraySizes.reset(new Int64[mThreadCount]);
        mTaskArrayOffsets.reset(new AtomicIndex[mThreadCount]);
        mPreviousIndices.reset(new Int64[mThreadCount]);

        // populate initial values

        for (size_t n = 0; n < indexArray.size(); n++) {
            mStolenTasks[n] = true;
            mTaskArrayOffsets[n] = mTaskArraySizes[n] = indexArray[n].size();
        }

        mHasTasks = true;
    }

    inline size_t size() const { return mThreadCount; }

    inline void reset()
    {
        mThreadCount = 0;
        mHasTasks = false;
        mTaskArrayOffsets.reset();
        mTaskArraySizes.reset();
        mStolenTasks.reset();
        mPreviousIndices.reset();
    }

    inline bool hasStolenTasks(size_t threadIndex) const { return mStolenTasks[threadIndex]; }

    inline bool getTask(size_t threadIndex, Int64& taskIndex)
    {
        return this->getLoadBalancedTask<true>(threadIndex, taskIndex);
    }

    inline bool stealTask(size_t threadIndex, size_t& otherThreadIndex, Int64& taskIndex)
    {
        if (!mHasTasks)  return false;

        // Use circular indexing to more evenly balance tasks between the threads
        // that are stealing.
        // Load-balancing state is tracked only for the current thread.

        for (size_t i = 1; i < mThreadCount; i++) {
            otherThreadIndex = (threadIndex + i) % mThreadCount;
            if (this->getLoadBalancedTask<false>(otherThreadIndex, taskIndex)) {
                return true;
            }
        }
        mHasTasks = false;
        return false;
    }

private:
    template <bool LoadBalance>
    bool getLoadBalancedTask(size_t threadIndex, Int64& taskIndex)
    {
        assert(threadIndex < mThreadCount);

        // The task array offset for this thread is initialized to the number of
        // tasks available for this thread and is decremented towards zero.
        // These values are stored in an atomic because multiple threads can access
        // them concurrently, however each task is only assigned to one thread.
        // When the returned value goes below zero, all tasks have been assigned.

        Int64 taskOffset = mTaskArrayOffsets[threadIndex]--;
        if (taskOffset <= 0) {
            if (LoadBalance) {
                // When tracking the load-balancing state, if the last task executed
                // by this thread was not the last task in the array, it is safe to
                // assume that this was stolen by another thread.
                if (mPreviousIndices[threadIndex] != mTaskArraySizes[threadIndex] - 1) {
                    mStolenTasks[threadIndex] = true;
                }
            }
            return false;
        }

        // task offsets descend to zero, task indices ascend from zero

        taskIndex = mTaskArraySizes[threadIndex] - taskOffset;
        assert(taskIndex >= 0 && taskIndex < mTaskArraySizes[threadIndex]);

        if (LoadBalance) {
            // When tracking the load-balancing state, mark this thread as having
            // not been stolen from if the first task is being assigned, otherwise
            // mark the thread as having been stolen from if the last task
            // executed by this thread was not immediately preceeding this one.
            if (taskIndex == 0) {
                mStolenTasks[threadIndex] = false;
            } else if (taskIndex != mPreviousIndices[threadIndex] + 1) {
                mStolenTasks[threadIndex] = true;
            }
            mPreviousIndices[threadIndex] = taskIndex;
        }

        return true;
    }

    size_t mThreadCount = 0;
    AtomicBool mHasTasks{false};
    std::unique_ptr<bool[]> mStolenTasks;
    std::unique_ptr<AtomicIndex[]> mTaskArrayOffsets;
    std::unique_ptr<Int64[]> mTaskArraySizes;
    std::unique_ptr<Int64[]> mPreviousIndices;
}; // struct TaskScheduler


template <bool UseLocalBins, typename PointDataGridT, typename DeformerT,
    typename FilterT, typename OtherFilterT, typename ThreadLocalBinT, typename LocalBinMapT>
inline void computeBins(PointDataLeafArray<PointDataGridT>& sourceLeafArray,
    const DeformerT& deformer,
    const FilterT& filter,
    bool updatePosition,
    const OtherFilterT& otherFilter,
    TaskScheduler& taskScheduler,
    std::vector<std::deque<size_t>>& taskIndices,
    std::vector<LocalBinMapT>& localBins,
    std::vector<std::vector<std::pair<size_t, ThreadLocalBinT>>>& globalLoadBalancedBins,
    std::vector<ThreadLocalBinT>& globalBins)
{
    (void) updatePosition;
    (void) otherFilter;

    using LocalMovePtrArrayT = typename LocalBinMapT::mapped_type;
    using LocalMovePtrT = typename LocalMovePtrArrayT::value_type;
    using LocalMoveT = typename LocalMovePtrT::element_type;

    using PointDataTreeT = typename PointDataGridT::TreeType;
    using PointDataLeafT = typename PointDataTreeT::LeafNodeType;

    using IndexType = typename ThreadLocalBinT::IndexType;
    using OffsetType = typename ThreadLocalBinT::OffsetType;
    using VoxelOffsetType = typename ThreadLocalBinT::VoxelOffsetType;
    using IndexPairList = typename ThreadLocalBinT::IndexPairList;
    using IndexPairListMap = typename ThreadLocalBinT::IndexPairListMap;
    using IndexPairListPtr = typename ThreadLocalBinT::IndexPairListPtr;

    static constexpr Index BucketLog2Dim = PointDataLeafT::LOG2DIM;
    static constexpr Index BinLog2Dim = 5u;

    static constexpr Index BucketLog2Dim2 = 2 * BucketLog2Dim;
    static constexpr Index BucketMask = (1u << BucketLog2Dim) - 1u;
    static constexpr Int32 InvBucketMask = static_cast<Int32>(~BucketMask);
    static constexpr Index BinLog2dim2 = 2 * BinLog2Dim;
    static constexpr Index BinMask = (1u << (BucketLog2Dim + BinLog2Dim)) - 1u;
    static constexpr Index InvBinMask = ~BinMask;

    const math::Transform& transform = sourceLeafArray.transform();

    size_t numPoints = sourceLeafArray.pointCount();
    size_t leafCount = sourceLeafArray.leafCount();

    size_t numTasks = 1, numThreads = size_t(tbb::task_scheduler_init::default_num_threads());
    if (numPoints > numThreads)         numTasks = numThreads;

    taskIndices.resize(numTasks);
    globalLoadBalancedBins.resize(numTasks);

    localBins.resize(numTasks);
    globalBins.resize(numTasks);

    // TODO: neither of these methods take into account a filter yet

    // compute the maximum source leaf point count

    Index64 maxLeafPointCount = sourceLeafArray.maxLeafPointCount();

    // partition point leafs into tasks, the leaf order must be
    // preserved to ensure determinism

    size_t pointsPerTaskThreshold = numPoints / numTasks;
    size_t taskIndex = 0;
    size_t leafIndex = 0;
    size_t pointsPerTask = 0;
    while (leafIndex < leafCount) {
        while (leafIndex < leafCount) {
            assert(taskIndex < taskIndices.size());
            taskIndices[taskIndex].push_back(leafIndex);
            const PointDataLeafT* sourceLeaf = sourceLeafArray.leaf(leafIndex);
            if (sourceLeaf)     pointsPerTask += sourceLeaf->pointCount();
            leafIndex++;

            if (taskIndex < taskIndices.size() - 1 &&
                    pointsPerTask > pointsPerTaskThreshold) {
                pointsPerTask -= pointsPerTaskThreshold;
                taskIndex++;
                break;
            }
        }
    }

    taskScheduler.initialize(taskIndices);

    auto localPositionOp = [&](tbb::blocked_range<size_t>& r)
    {
        for (size_t n = r.begin(); n < r.end(); n++) {
            if (sourceLeafArray.isLeafStealable(n))     continue;
            const PointDataLeafT* leaf = sourceLeafArray.leaf(n);
            if (leaf && leaf->beginIndexOn(filter)) {
                // for const leafs, make a local copy of the position array
                sourceLeafArray.mInfo[n].positionArray = leaf->constAttributeArray(0).copy();
            }
        }
    };

    tbb::blocked_range<size_t> range(0, leafCount);
    localPositionOp(range);

    auto op = [&](tbb::blocked_range<size_t>& r)
    {
        DeformerT newDeformer(deformer);

        std::vector<Index> localVoxelOffsets;
        localVoxelOffsets.reserve(maxLeafPointCount);

        for (size_t n = r.begin(); n < r.end(); n++) {

            Coord lastBinCoord(1, 2, 3);
            IndexPairList* idxList = nullptr;
            IndexPairListMap* binMap = nullptr;

            auto& localBinMap = localBins[n];

            while (true) {
                bool loadBalance;
                size_t leafIndex;
                {
                    size_t stealThreadIndex;
                    Int64 taskIndex;
                    if (taskScheduler.getTask(n, taskIndex)) {
                        leafIndex = taskIndices[n][taskIndex];
                        loadBalance = taskScheduler.hasStolenTasks(n);
                    } else if (taskScheduler.stealTask(n, stealThreadIndex, taskIndex)) {
                        leafIndex = taskIndices[stealThreadIndex][taskIndex];
                        loadBalance = true;
                    } else {
                        // no more tasks
                        break;
                    }
                }

                if (loadBalance) {
                    lastBinCoord.reset(1, 2, 3);
                    globalLoadBalancedBins[n].emplace_back(leafIndex, ThreadLocalBinT());
                    binMap = &globalLoadBalancedBins[n].back().second.map();
                } else {
                    binMap = &globalBins[n].map();
                }

                IndexType packedLeafIndex = leafIndex << 32;

                const PointDataLeafT* leaf = sourceLeafArray.leaf(leafIndex);
                if (!leaf)  continue;
                const math::Transform& sourceTransform =
                    sourceLeafArray.transform(static_cast<Index>(leafIndex));

                localVoxelOffsets.clear();

                const Coord leafOrigin = leaf->origin();

                newDeformer.reset(*leaf, leafIndex);

                bool localOnly = true;

                auto pointIterOp = [&](auto& handle)
                {
                    Coord ijk;

                    Vec3d half(0.5);
                    double integer;

                    using HandleT = typename std::remove_reference<decltype(handle)>::type;
                    using CodecT = typename HandleT::CodecT;

                    AttributeWriteHandle<Vec3f, CodecT> writeHandle(
                        sourceLeafArray.isLeafStealable(leafIndex) ?
                        (const_cast<PointDataLeafT&>(*leaf).attributeArray(0)) :
                        *sourceLeafArray.mInfo[leafIndex].positionArray);

                    for (auto iter = leaf->beginIndexOn(filter); iter; ++iter) {
                        Vec3d pos = handle.get(*iter) + iter.getCoord().asVec3d();

                        pos = sourceTransform.indexToWorld(pos);

                        newDeformer.apply(pos, iter);

                        pos = transform.worldToIndex(pos);

                        pos.x() = modf(pos.x() + 0.5, &integer) - 0.5;
                        ijk.setX(static_cast<Int32>(integer));
                        pos.y() = modf(pos.y() + 0.5, &integer) - 0.5;
                        ijk.setY(static_cast<Int32>(integer));
                        pos.z() = modf(pos.z() + 0.5, &integer) - 0.5;
                        ijk.setZ(static_cast<Int32>(integer));

                        writeHandle.set(*iter, pos);

                        VoxelOffsetType voxelOffset(
                            static_cast<VoxelOffsetType>(
                                ((ijk[0] & BucketMask) << BucketLog2Dim2) +
                                ((ijk[1] & BucketMask) << BucketLog2Dim) +
                                (ijk[2] & BucketMask)));

                        if (UseLocalBins &&
                            (ijk.x() & InvBucketMask) == leafOrigin.x() &&
                            (ijk.y() & InvBucketMask) == leafOrigin.y() &&
                            (ijk.z() & InvBucketMask) == leafOrigin.z()) {

                            uint32_t packedIndex = (*iter << 16) | voxelOffset;

                            localVoxelOffsets.emplace_back(packedIndex);
                        } else {
                            Coord binCoord(ijk[0] & InvBinMask,
                                           ijk[1] & InvBinMask,
                                           ijk[2] & InvBinMask);

                            ijk[0] &= BinMask;
                            ijk[1] &= BinMask;
                            ijk[2] &= BinMask;

                            ijk[0] >>= BucketLog2Dim;
                            ijk[1] >>= BucketLog2Dim;
                            ijk[2] >>= BucketLog2Dim;

                            OffsetType bucketOffset(
                                (ijk[0] << BinLog2dim2) +
                                (ijk[1] << BinLog2Dim) +
                                ijk[2]);

                            if (lastBinCoord != binCoord) {
                                lastBinCoord = binCoord;
                                assert(binMap);
                                IndexPairListPtr& idxPtr = (*binMap)[binCoord];
                                if (!idxPtr)    idxPtr.reset(new IndexPairList);
                                idxList = idxPtr.get();
                            }
                            assert(idxList);

                            IndexType packedIndex = packedLeafIndex | (*iter << (1u + (3 * BucketLog2Dim))) | voxelOffset;

                            idxList->emplace_back(packedIndex, bucketOffset);

                            localOnly = false;
                        }
                    }
                };

                const AttributeArray& positionArray = sourceLeafArray.constAttributeArray(leafIndex, 0);

                if (positionArray.codecType() == FixedPointCodec<false, PositionRange>::name()) {
                    points::AttributeHandle<Vec3f, FixedPointCodec<false, PositionRange>> handle(positionArray);
                    pointIterOp(handle);
                } else if (positionArray.codecType() == FixedPointCodec<true, PositionRange>::name()) {
                    points::AttributeHandle<Vec3f, FixedPointCodec<true, PositionRange>> handle(positionArray);
                    pointIterOp(handle);
                } else if (positionArray.codecType() == NullCodec::name()) {
                    points::AttributeHandle<Vec3f, NullCodec> handle(positionArray);
                    pointIterOp(handle);
                } else {
                    points::AttributeHandle<Vec3f> handle(positionArray);
                    pointIterOp(handle);
                }

                if (localOnly) {
                    sourceLeafArray.markLocalOnly(leafIndex);
                }

                if (!localVoxelOffsets.empty()) {
                    LocalMovePtrT localMove(new LocalMoveT);
                    localMove->index = static_cast<Index>(leafIndex);
                    localMove->points.assign(localVoxelOffsets.begin(), localVoxelOffsets.end());
                    localMove->hasSamePointOrder = localVoxelOffsets.size() == leaf->pointCount();
                    localBinMap[leafOrigin].emplace_back(localMove.release());
                }
            }
        }
    };

    tbb::blocked_range<size_t> taskRange(0, numTasks);

    tbb::parallel_for(taskRange, op, tbb::simple_partitioner());
    // op(taskRange);
}


template <typename ThreadLocalBinT>
inline void mergeGlobalBins(std::vector<ThreadLocalBinT>& globalBins,
    std::vector<std::vector<std::pair<size_t, ThreadLocalBinT>>>& globalLoadBalancedBins,
    TaskScheduler& taskScheduler,
    std::vector<std::deque<size_t>>& taskIndices)
{
    // merge load-balanced bins into a single map, keyed by the leafIndex

    std::unordered_map<size_t, ThreadLocalBinT> mergedLoadBalancedBins;

    for (auto& loadBalancedBin : globalLoadBalancedBins) {
        for (auto& bin : loadBalancedBin) {
            mergedLoadBalancedBins[bin.first].mData.reset(bin.second.mData.release());
        }
    }

    // perform in-order interleaving of load-balanced and non load-balanced
    // global bins to ensure determinism

    std::vector<ThreadLocalBinT> tempGlobalBins;

    for (size_t n = 0; n < taskScheduler.size(); n++) {
        tempGlobalBins.emplace_back(globalBins[n].mData.release());

        // early-exit if no tasks were stolen from this thread
        if (!taskScheduler.hasStolenTasks(n))  continue;

        for (const size_t& leafIndex : taskIndices[n]) {
            auto it = mergedLoadBalancedBins.find(leafIndex);
            if (it != mergedLoadBalancedBins.end()) {
                tempGlobalBins.emplace_back(it->second.mData.release());
            }
        }
    }

    globalBins.swap(tempGlobalBins);
}


template <typename LocalBinT, typename MultiBinsT>
inline void mergeLocalBins(LocalBinT& localBin, MultiBinsT& localBins, std::vector<Coord>& localCoords)
{
    using LocalMovePtrArrayT = typename LocalBinT::value_type;
    using LocalMovePtrT = typename LocalMovePtrArrayT::value_type;
    using SingleBin = typename MultiBinsT::value_type;

    SingleBin mergeBin;

    for (SingleBin& bin : localBins) {
        for (auto& localPair : bin) {
            const Coord& ijk = localPair.first;
            for (LocalMovePtrT& move : localPair.second) {
                mergeBin[ijk].emplace_back(move.release());
            }
        }
    }

    for (auto& bin : mergeBin) {
        const Coord& ijk = bin.first;
        localCoords.push_back(ijk);
        localBin.emplace_back();
        LocalMovePtrArrayT& localArray = localBin.back();
        for (LocalMovePtrT& move : bin.second) {
            localArray.emplace_back(move.release());
        }
    }
}


template <typename LocalBinT>
inline void orderLocalPartitions(LocalBinT& localBin)
{
    using LocalMovePtrArrayT = typename LocalBinT::value_type;
    using LocalMovePtrT = typename LocalMovePtrArrayT::value_type;
    using LocalMoveT = typename LocalMovePtrT::element_type;
    using VoxelArrayT = typename LocalMoveT::VoxelArray;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, localBin.size()),
        [&](tbb::blocked_range<size_t>& range)
        {
            VoxelArrayT localHistogram;

            for (size_t n = range.begin(); n < range.end(); n++) {
                LocalMovePtrArrayT& bin = localBin[n];

                for (LocalMovePtrT& localMove : bin) {

                    std::vector<Index> offsets(localMove->points.size());
                    std::vector<Index> sortedIndices(localMove->points.size());

                    localMove->voxels.fill(Index(0));

                    for (size_t i = 0; i < localMove->points.size(); ++i) {
                        offsets[i] = static_cast<uint16_t>(localMove->points[i]);
                    }

                    for (size_t i = 0; i < localMove->points.size(); ++i) {
                        ++localMove->voxels[ offsets[i] ];
                    }

                    for (size_t i = 0; i < localMove->points.size(); i++) {
                        localMove->points[i] = localMove->points[i] >> 16;
                    }

                    for (size_t i = 0; i < localHistogram.size(); i++) {
                        localHistogram[i] = localMove->voxels[i];
                    }

                    // turn histogram into cumulative histogram

                    Index count = 0, startOffset;
                    for (int i = 0; i < int(localHistogram.size()); ++i) {
                        if (localHistogram[i] > 0) {
                            startOffset = count;
                            count += localHistogram[i];
                            localHistogram[i] = startOffset;
                        } else {
                            localHistogram[i] = count;
                        }
                    }

                    // sort indices based on voxel offset
                    for (size_t i = 0; i < localMove->points.size(); ++i) {
                        sortedIndices[ localHistogram[ offsets[i] ]++ ] = localMove->points[i];
                    }

                    for (size_t i = 0; i < localMove->points.size(); ++i) {
                        Index pointIndex = sortedIndices[i];
                        localMove->points[i] = pointIndex;
                        if (localMove->hasSamePointOrder && pointIndex != i) {
                            localMove->hasSamePointOrder = false;
                        }
                    }
                }
            }
        }
    );
}


template <typename LocalBinT, typename OriginToIndexMapT, typename PointPartitionerT>
inline void computePageCoordinates(std::unique_ptr<Coord[]>& pageCoordinates,
    Index& pageCount,
    LocalBinT& localBin,
    std::vector<Coord>& localCoords,
    OriginToIndexMapT& originToIndexMap,
    PointPartitionerT& partitioner)
{
    // build a mask tree, a leaf manager and a coord->index map to determine the leaf order remapping

    MaskTree maskTree;

    for (size_t n = 0; n < partitioner.size(); n++) {
        const Coord& ijk = partitioner.origin(n);
        originToIndexMap[ijk] = {n+1, 0};
        maskTree.touchLeaf(ijk);
    }

    for (size_t n = 0; n < localBin.size(); n++) {
        const Coord& ijk = localCoords[n];
        originToIndexMap[ijk].second = n+1;
        maskTree.touchLeaf(ijk);
    }

    pageCount = maskTree.leafCount();
    pageCoordinates.reset(new Coord[pageCount]);

    size_t pageIndex = 0;
    for (auto iter = maskTree.cbeginLeaf(); iter; ++iter) {
        pageCoordinates[pageIndex++].reset(iter->origin().x(), iter->origin().y(), iter->origin().z());
    }
}


template <typename OriginToIndexMapT>
inline void computeBucketIndices(std::vector<std::pair<size_t, size_t>>& bucketIndices,
    std::unique_ptr<Coord[]>& pageCoordinates,
    Index pageCount,
    OriginToIndexMapT& originToIndexMap)
{
    bucketIndices.reserve(originToIndexMap.size());

    for (size_t i = 0; i < pageCount; i++) {
        bucketIndices.emplace_back(originToIndexMap[pageCoordinates[i]]);
    }
}


template <typename LocalBinT, typename PointPartitionerT>
inline void allocateLinearIndices(const std::vector<std::pair<size_t, size_t>>& bucketIndices,
    const PointPartitionerT& partitioner,
    const LocalBinT& localBin,
    const Index pageCount,
    std::unique_ptr<Index64[]>& pageOffsets,
    std::unique_ptr<Index64[]>& pageIndices)
{
    pageOffsets.reset(new Index64[pageCount]);

    Index64 cumulativeSize(0);

    for (size_t n = 0; n < pageCount; n++) {

        const std::pair<size_t, size_t>& indices = bucketIndices[n];

        size_t pointCount = 0;

        if (indices.first > 0) {
            pointCount += partitioner.indices(indices.first-1).size();
        }

        if (indices.second > 0) {
            const auto& localMoves = localBin[indices.second-1];
            for (const auto& localMove : localMoves) {
                pointCount += localMove->points.size();
            }
        }

        cumulativeSize += pointCount;
        pageOffsets[n] = cumulativeSize;
    }

    pageIndices.reset(new Index64[cumulativeSize]);
}


template <Index NUM_VOXELS, Index LOG2DIM, typename PointPartitionerT>
inline void sortGlobalIndices(std::unique_ptr<Index[]>& histogram,
    PointPartitionerT& partitioner)
{
    // allocate histogram

    histogram.reset(new Index[partitioner.size()*NUM_VOXELS]);

    // sort global indices

    PointDataMapping<LOG2DIM> indexToOffset;

    // sort the point indices and populate the histogram

    partitioner.sort(histogram, indexToOffset);
}


template <Index NUM_VOXELS, typename LocalBinT, typename PointPartitionerT>
inline void computeLinearIndices(const std::vector<std::pair<size_t, size_t>>& bucketIndices,
    const PointPartitionerT& partitioner,
    const LocalBinT& localBin,
    const std::unique_ptr<Index[]>& histogram,
    const Index pageCount,
    std::unique_ptr<Index64[]>& pageOffsets,
    std::unique_ptr<Index64[]>& pageIndices)
{
    using LocalMovePtrArrayT = typename LocalBinT::value_type;

    auto op = [&](tbb::blocked_range<size_t>& range)
    {
        std::vector<Index*> pointPtrs;
        std::vector<Index*> voxelPtrs;
        std::vector<Index64> packedIndices;

        for (size_t n = range.begin(); n < range.end(); n++) {
            size_t startOffset = n > 0 ? pageOffsets[n-1] : 0;
            Index64* pageIndicesPtr = pageIndices.get() + startOffset;

            const std::pair<size_t, size_t>& indices = bucketIndices[n];

            // global move setup

            std::unique_ptr<typename PointPartitionerT::IndexIterator> partitionerIter;
            Index* leafHistogram = nullptr;

            if (indices.first > 0) {
                partitionerIter.reset(new typename PointPartitionerT::IndexIterator(partitioner.indices(indices.first-1)));
                leafHistogram = histogram.get() + (indices.first - 1) * NUM_VOXELS;
            }

            // local move setup

            pointPtrs.clear();
            voxelPtrs.clear();
            packedIndices.clear();

            if (indices.second > 0) {
                const LocalMovePtrArrayT& locals = localBin[indices.second-1];
                for (const auto& local : locals) {
                    pointPtrs.emplace_back(local->points.data());
                    voxelPtrs.emplace_back(local->voxels.data());
                    packedIndices.emplace_back(Index64(local->index) << 32);
                }
            }

            // perform interleave

            if (partitionerIter && voxelPtrs.empty()) {
                // optimized copy for only global indices
                size_t count = pageOffsets[n] - startOffset;
                assert(count == partitionerIter->size());
                std::memcpy(pageIndicesPtr, std::addressof(**partitionerIter), count*sizeof(Index64));
            } else if (!partitionerIter && voxelPtrs.size() == 1) {
                // optimized copy for only one set of local indices
                size_t count = pageOffsets[n] - startOffset;
                assert(count == localBin[indices.second-1][0]->points.size());
                for (size_t i = 0; i < count; i++) {
                    *pageIndicesPtr++ = packedIndices[0] | *pointPtrs[0]++;
                }
            } else {
                size_t voxelStart = 0;
                for (size_t voxelOffset = 0; voxelOffset < NUM_VOXELS; voxelOffset++) {
                    for (size_t i = 0; i < voxelPtrs.size(); i++) {
                        size_t count = *voxelPtrs[i]++;
                        // this loop also expands the point index to 64-bit and
                        // packs the leaf index into the first 32-bits - using
                        // 64-bit indices for all points makes iteration faster
                        for (size_t j = 0; j < count; j++) {
                            *pageIndicesPtr++ = packedIndices[i] | *pointPtrs[i]++;
                        }
                    }
                    if (partitionerIter) {
                        size_t voxelEnd = *leafHistogram++;
                        if (voxelEnd > voxelStart) {
                            size_t count = voxelEnd - voxelStart;
                            std::memcpy(pageIndicesPtr, std::addressof(**partitionerIter), count*sizeof(Index64));
                            pageIndicesPtr += count;
                            *partitionerIter += count;
                        }
                        voxelStart = voxelEnd;
                    }
                }
            }

            // cleanup local point indices

            if (!voxelPtrs.empty()) {
                const LocalMovePtrArrayT& locals = localBin[indices.second-1];
                for (const auto& local : locals) {
                    local->points.clear();
                }
             }
        }
    };

    tbb::blocked_range<size_t> range(0, pageCount);
    tbb::parallel_for(range, op);
}


} // namespace point_data_partitioner_internal


template<typename PointDataGridT>
struct PointDataLeafArray
{
    struct EmptyDeleter
    {
        template <typename T>
        void operator() (T const &) const noexcept { }
    }; // struct EmptyDeleter

    using LeafT = typename PointDataGridT::TreeType::LeafNodeType;
    using LeafUniquePtr = std::unique_ptr<LeafT>;
    using ConstLeafRawPtr = std::unique_ptr<const LeafT, EmptyDeleter>;

    template <typename T>
    struct LeafArray
    {
        using ArrayT = std::vector<T>;

        struct Inserter
        {
            using value_type = typename T::element_type*;

            Inserter(ArrayT& _data, Index& _idx) : data(_data), idx(_idx) { }

            void push_back(value_type leaf)
            {
                assert(idx < data.size());
                data[idx++].reset(leaf);
            }

        private:
            ArrayT& data;
            Index& idx;
        }; // struct Inserter

        LeafArray() = default;
        size_t size() const { return mData.size(); }
        void resize(size_t size) { mData.resize(size); }
        void insert(size_t idx, LeafT* newLeaf) { mData[idx].reset(newLeaf); }
        const LeafT* operator[](size_t idx) const
        {
            assert(idx < mData.size());
            if (!bool(mData[idx]))  return nullptr;
            return mData[idx].get();
        }
        LeafT* steal(size_t idx) { assert(idx < mData.size()); return mData[idx].release(); }
        Inserter inserter(Index& idx) { return Inserter(mData, idx); }

    private:
        ArrayT mData;
    }; // struct LeafArray

    struct Info
    {
        Info() = default;

        void reset(bool _writeable, bool _deform)
        {
            writeable = _writeable;
            deform = _deform;
        }

        AttributeArray::Ptr positionArray;
        bool localOnly = false;
        bool writeable = false;
        bool deform = false;
    }; // struct Info

    static inline bool stealable(PointDataGridT& points)
    {
        TreeBase::Ptr localTreePtr = points.baseTreePtr();
        return localTreePtr.use_count() == 2; // points + localTreePtr = 2
    }

    PointDataLeafArray( PointDataGridT& points,
                        const math::Transform& transform,
                        const std::vector<typename PointDataGridT::Ptr>& otherPoints = {},
                        const std::vector<typename PointDataGridT::ConstPtr>& otherConstPoints = {},
                        bool deformOthers = false)
        : mTransform(transform)
    {
        (void) deformOthers;

        // pre-allocate info and leaf arrays

        mGridCount = 1;
        size_t leafCount = points.constTree().leafCount();
        for (const auto& otherPointsGrid : otherPoints) {
            if (!otherPointsGrid) continue;
            mGridCount++;
            leafCount += otherPointsGrid->constTree().leafCount();
        }
        for (const auto& otherPointsGrid : otherConstPoints) {
            if (!otherPointsGrid) continue;
            mGridCount++;
            leafCount += otherPointsGrid->constTree().leafCount();
        }

        mLeafArray.resize(leafCount);
        mLeafReferenceArray.resize(leafCount);

        mInfo.resize(leafCount);
        mGridIndices.reset(new Index[leafCount]);

        // populate attribute sets

        mAttributeSets.clear();
        mAttributeSets.reserve(otherPoints.size()+1);
        if (points.constTree().leafCount() > 0) {
            mAttributeSets.push_back(&points.constTree().cbeginLeaf()->attributeSet());
        } else {
            mAttributeSets.push_back(nullptr);
        }
        for (const auto& otherPointsGrid : otherPoints) {
            if (otherPointsGrid) {
                mAttributeSets.push_back(&otherPointsGrid->constTree().cbeginLeaf()->attributeSet());
            } else {
                mAttributeSets.push_back(nullptr);
            }
        }
        for (const auto& otherPointsGrid : otherConstPoints) {
            if (otherPointsGrid) {
                mAttributeSets.push_back(&otherPointsGrid->constTree().cbeginLeaf()->attributeSet());
            } else {
                mAttributeSets.push_back(nullptr);
            }
        }

        // populate leaf array and source grid indices

        Index index = 0;

        auto leafInserter = mLeafArray.inserter(index);
        auto leafReferenceInserter = mLeafReferenceArray.inserter(index);

        if (stealable(points)) {
            points.tree().stealNodes(leafInserter);
        }
        else {
            points.constTree().getNodes(leafReferenceInserter);
        }

        for (Index i = 0; i < index; i++) {
            mGridIndices[i] = 0;
        }

        Index sourceGridIndex = 1;
        Index sourceOffset = index;

        for (auto& otherGridPtr : otherPoints) {
            if (stealable(*otherGridPtr)) {
                otherGridPtr->tree().stealNodes(leafInserter);
            }
            else {
                otherGridPtr->constTree().getNodes(leafReferenceInserter);
            }

            Index newSourceOffset = index;

            for (Index i = sourceOffset; i < newSourceOffset; i++) {
                mInfo[i].reset(false, false);
                mGridIndices[i] = sourceGridIndex;
            }
            sourceGridIndex++;
            sourceOffset = newSourceOffset;
        }
        for (auto& otherGridPtr : otherConstPoints) {
            otherGridPtr->constTree().getNodes(leafReferenceInserter);

            Index newSourceOffset = index;

            for (Index i = sourceOffset; i < newSourceOffset; i++) {
                mInfo[i].reset(false, false);
                mGridIndices[i] = sourceGridIndex;
            }
            sourceGridIndex++;
            sourceOffset = newSourceOffset;
        }

        // compute array of leaf counts

        std::vector<size_t> leafCounts(leafCount);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, leafCount),
            [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t n = range.begin(); n < range.end(); n++) {
                    if (mLeafArray[n]) {
                        leafCounts[n] = mLeafArray[n]->pointCount();
                    } else {
                        leafCounts[n] = mLeafReferenceArray[n]->pointCount();
                    }
                }
            }
        );

        // compute total and max leaf points

        for (const size_t count : leafCounts) {
            mPointCount += count;
            if (count > mMaxLeafPointCount)     mMaxLeafPointCount = count;
        }
        leafCounts.clear();

        // populate transform array

        bool sourceChangingTransform = false;
        bool othersChangingTransform = false;

        mTransforms.clear();
        mTransforms.reserve(otherPoints.size()+1);
        if (points.constTransform() != transform) {
            mTransforms.push_back(points.constTransformPtr());
            sourceChangingTransform = true;
        } else {
            mTransforms.push_back(math::Transform::ConstPtr());
        }
        for (const auto& otherGridPtr : otherPoints) {
            if (otherGridPtr->constTransform() != transform) {
                mTransforms.push_back(otherGridPtr->constTransformPtr());
                othersChangingTransform = true;
            } else {
                mTransforms.push_back(math::Transform::ConstPtr());
            }
        }
        if (!othersChangingTransform) {
            if (sourceChangingTransform) {
                mTransforms.resize(1);
            } else {
                mTransforms.clear();
            }
        }
    }

    const AttributeSet* attributeSet() const
    {
        for (const AttributeSet* attrSet : mAttributeSets) {
            if (attrSet)    return attrSet;
        }
        return nullptr;
    }
    const AttributeSet* attributeSet(Index index) const
    {
        return mAttributeSets[index];
    }

    const math::Transform& transform() const { return mTransform; }
    const math::Transform& transform(Index index) const
    {
        if (mTransforms.size() == 1) {
            if (mTransforms[0]) {
                return *mTransforms[0];
            }
        } else if (mTransforms.size() > 1) {
            assert(mGridIndices);
            Index gridIndex = mGridIndices[index];
            if (mTransforms[gridIndex]) {
                return *mTransforms[gridIndex];
            }
        }
        return this->transform();
    }

    LeafT* stealLeaf(size_t index)
    {
        if (isLeafStealable(index)) {
            return mLeafArray.steal(index);
        }
        return nullptr;
    }

    Index gridIndex(Index idx) { return mGridIndices[idx]; }

    Index* stealGridIndices() { return mGridIndices.release(); }

    void collapseArray(Index idx, const Name& name) {
        if (this->isLeafStealable(idx)) {
            const LeafT* leaf = mLeafArray[idx];
            const_cast<LeafT*>(leaf)->attributeArray(name).collapse();
        }
    }

    size_t gridCount() const { return mGridCount; }
    size_t leafCount() const { return mLeafArray.size(); }

    inline bool isLeafStealable(size_t index) const { return bool(mLeafArray[index]); }
    inline bool isLeafConst(size_t index) const { return !this->isLeafStealable(index); }
    inline bool isLeafValid(size_t index) const { return bool(mLeafArray[index]) || bool(mLeafReferenceArray[index]); }

    const LeafT* leaf(size_t index) const { return isLeafStealable(index) ? mLeafArray[index] : mLeafReferenceArray[index]; }
    const Info& info(size_t index) const { assert(index < mInfo.size()); return mInfo[index]; }

    void markLocalOnly(size_t index) { mInfo[index].localOnly = true; }

    const AttributeArray& constAttributeArray(size_t index, size_t pos) const
    {
        const LeafT* leaf = this->leaf(index);
        assert(leaf);
        return leaf->constAttributeArray(pos);
    }

    template <typename CodecT = UnknownCodec>
    AttributeHandle<Vec3f, CodecT> positionHandle(size_t index) const
    {
        const LeafT* leaf = this->leaf(index);
        assert(leaf);
        return AttributeHandle<Vec3f, CodecT>(leaf->constAttributeArray(0));
    }

    template <typename CodecT = UnknownCodec>
    void setPositionWriteHandle(typename AttributeWriteHandle<Vec3f, CodecT>::ScopedPtr& writeHandle,
        size_t leafIndex)
    {
        AttributeArray* array = nullptr;
        const LeafT* leaf = this->leaf(leafIndex);
        assert(leaf);
        if (isLeafStealable(leafIndex)) {
            // for non-const leafs, it is safe to const cast the leaf and
            // access the array directly
            LeafT& nonConstLeaf = const_cast<LeafT&>(*leaf);
            array = &nonConstLeaf.attributeArray(0);
        } else {
            // for const leafs, make a local copy of the position array
            mInfo[leafIndex].positionArray = leaf->constAttributeArray(0).copy();
            array = mInfo[leafIndex].positionArray.get();
        }

        assert(array);
        writeHandle.reset(new AttributeWriteHandle<Vec3f, CodecT>(*array));
    }

    size_t maxLeafPointCount() const { return mMaxLeafPointCount; }
    size_t pointCount() const { return mPointCount; }

private:
    const math::Transform mTransform;
    LeafArray<LeafUniquePtr> mLeafArray;
    LeafArray<ConstLeafRawPtr> mLeafReferenceArray;
public:
    std::vector<Info> mInfo;
private:
    std::unique_ptr<Index[]> mGridIndices;
    std::vector<math::Transform::ConstPtr> mTransforms;
    std::vector<const AttributeSet*> mAttributeSets;
    size_t mGridCount = 0;
    size_t mMaxLeafPointCount = 0;
    size_t mPointCount = 0;
}; // struct PointDataLeafArray


template<typename PointDataGridT>
template <typename DeformerT, typename FilterT, typename OtherFilterT>
void PointDataPartitioner<PointDataGridT>::construct(
    PointDataLeafArray<PointDataGridT>& sourceLeafArray,
    const DeformerT& deformer,
    const FilterT& filter,
    bool updatePosition,
    const OtherFilterT& otherFilter)
{
    using namespace point_data_partitioner_internal;

    using ThreadLocalBin = typename PointPartitionerT::ThreadLocalBin;
    using ThreadLocalBins = typename PointPartitionerT::ThreadLocalBins;

    using MultiBins = std::vector<std::unordered_map<Coord, LocalMovePtrArray>>;

    MultiBins localBins;
    ThreadLocalBins globalBins;

    std::vector<std::vector<std::pair<size_t, ThreadLocalBin>>> globalLoadBalancedBins;
    TaskScheduler taskScheduler;
    std::vector<std::deque<size_t>> taskIndices;

    // compute bins

    static constexpr bool UseLocalBins = true;
    computeBins<UseLocalBins>(sourceLeafArray, deformer, filter, updatePosition, otherFilter,
        taskScheduler, taskIndices, localBins, globalLoadBalancedBins, globalBins);

    std::vector<Coord> localCoords;

    tbb::task_group tasks;
    tasks.run([&]
        {
            // merge global bins

            mergeGlobalBins(globalBins, globalLoadBalancedBins, taskScheduler, taskIndices);

            taskIndices.clear();
            taskScheduler.reset();

            // build global partitioner

            size_t numPoints = sourceLeafArray.pointCount();
            mPartitioner = PointPartitionerT::create(globalBins, numPoints,
                /*cellCenteredTransform=*/true);
            mHistogram.reset(new Index[mPartitioner->size()*NUM_VOXELS]);
            PointDataMapping<LOG2DIM> indexToOffset;
            mPartitioner->sort(mHistogram, indexToOffset);
        }
    );

    tasks.run([&]
        {
            // merge local bins

            mergeLocalBins(mLocalBin, localBins, localCoords);

            // order local partitions

            orderLocalPartitions(mLocalBin);
        }
    );
    tasks.wait();

    // compute page coordinates

    std::unordered_map<Coord, std::pair<size_t, size_t>> originToIndexMap;

    computePageCoordinates(mPageCoordinates, mPageCount, mLocalBin, localCoords, originToIndexMap, *mPartitioner);

    mPartitioner->clearPageCoordinates();

    // compute bucket indices

    computeBucketIndices(mBucketIndices, mPageCoordinates, mPageCount, originToIndexMap);
    originToIndexMap.clear();

    // allocate linear indices

    this->linearize();

    // compute source offsets

    if (sourceLeafArray.gridCount() > 1) {
        mSourceGridIndices.reset(sourceLeafArray.stealGridIndices());
    } else {
        mSourceGridIndices.reset();
    }
}


template<typename PointDataGridT>
inline void PointDataPartitioner<PointDataGridT>::linearize()
{
    using namespace point_data_partitioner_internal;

    allocateLinearIndices(mBucketIndices, *mPartitioner, mLocalBin, mPageCount, mPageOffsets, mPageIndices);
    computeLinearIndices<NUM_VOXELS>(mBucketIndices, *mPartitioner, mLocalBin, mHistogram, mPageCount, mPageOffsets, mPageIndices);
}


template<typename PointDataGridT>
template<typename VoxelBuffersT>
inline void
PointDataPartitioner<PointDataGridT>::assignVoxelValues(VoxelBuffersT& voxelBuffers, bool cleanup)
{
    using namespace point_data_partitioner_internal;

    using AccessorT = typename VoxelBuffersT::Accessor;

    // interleave indices

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, mPageCount),
        [&](tbb::blocked_range<size_t>& r) {
            std::array<Index, NUM_VOXELS> voxelIndices;

            for (size_t idx = r.begin(); idx < r.end(); idx++) {
                const std::pair<size_t, size_t>& indices = mBucketIndices[idx];

                bool global = indices.first > 0;
                bool local = indices.second > 0;

                if (global) {
                    const size_t n = indices.first-1;
                    const size_t leafOffset = n*NUM_VOXELS;

                    if (local) {
                        // convert global cumulative histogram back into local histogram for interleaving

                        for (Index i = NUM_VOXELS-1; i >= 1; i--) {
                            voxelIndices[i] = mHistogram[leafOffset+i] - mHistogram[leafOffset+i-1];
                        }
                        voxelIndices[0] = mHistogram[leafOffset];
                    } else {
                        // assign voxel values from global histogram directly

                        AccessorT acc = voxelBuffers.accessor(idx);

                        if (!acc.isStolen()) {
                            for (Index i = 0; i < NUM_VOXELS; i++) {
                                acc.set(i, mHistogram[leafOffset+i]);
                            }
                        }

                        continue;
                    }
                } else {
                    voxelIndices.fill(Index(0));
                }

                // sum histograms

                const size_t n = indices.second-1;

                const LocalMovePtrArray& localMoves = mLocalBin[n];
                for (const LocalMovePtr& localMove : localMoves) {
                    for (Index i = 0; i < NUM_VOXELS; ++i) {
                        voxelIndices[i] += localMove->voxels[i];
                    }
                }

                // rebuild offsets from histogram

                Index count = 0;
                for (int i = 0; i < int(NUM_VOXELS); ++i) {
                    if (voxelIndices[i] > 0) {
                        count += voxelIndices[i];
                        voxelIndices[i] = count;
                    } else {
                        voxelIndices[i] = count;
                    }
                }

                // rebuild local offsets

                for (Index i = 0; i < NUM_VOXELS; ++i) {
                    Index previousIndex = i == 0 ? 0 : voxelIndices[i-1];
                    for (const LocalMovePtr& localMove : localMoves) {
                        Index beforeIndex = localMove->voxels[i];
                        localMove->voxels[i] += previousIndex;
                        previousIndex += beforeIndex;
                    }
                }

                // assign voxel values

                AccessorT acc = voxelBuffers.accessor(idx);

                if (!acc.isStolen()) {
                    for (Index i = 0; i < NUM_VOXELS; i++) {
                        acc.set(i, voxelIndices[i]);
                    }
                }
            }
        }
    );

    if (cleanup)    mHistogram.reset();
}


template<typename PointDataGridT>
inline void
PointDataPartitioner<PointDataGridT>::clear()
{
    mPageCount = 0;
    mBucketIndices.clear();
    mPageCoordinates.reset();
    mSourceGridIndices.reset();
    mPartitioner.reset();
    mHistogram.reset();

    // deallocate the local bin in parallel

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, mLocalBin.size()),
        [&](tbb::blocked_range<size_t>& range)
        {
            for (size_t n = range.begin(); n < range.end(); n++) {
                mLocalBin[n].clear();
            }
        }
    );
    mLocalBin.clear();
}


template<typename PointDataGridT>
inline size_t
PointDataPartitioner<PointDataGridT>::memUsage() const
{
    size_t bytes = sizeof(*this) +
        // mPartitioner->memUsage() +
        /*mBucketIndices*/sizeof(std::pair<size_t, size_t>) * mBucketIndices.capacity() +
        /*mPageCoordinates*/sizeof(Coord) * mPageCount +
        /*histogram*/sizeof(Index) * mPageCount;

    bytes += sizeof(typename LocalBin::value_type) * mLocalBin.capacity();
    for (const LocalMovePtrArray& localMoveArray : mLocalBin) {
        bytes += sizeof(typename LocalMovePtrArray::value_type) * localMoveArray.capacity();
        for (const LocalMovePtr& localMovePtr : localMoveArray) {
            bytes += localMovePtr->memUsage();
        }
    }

    return bytes;
}


template<typename PointDataGridT>
size_t PointDataPartitioner<PointDataGridT>::size(size_t n) const
{
    assert(mPageOffsets);
    assert(n < mPageCount);
    size_t end = mPageOffsets[n];
    size_t start(0);
    if (n > 0)  start = mPageOffsets[n-1];
    return end - start;
}


template<typename PointDataGridT>
Index PointDataPartitioner<PointDataGridT>::stealableLeafIndex(size_t n) const
{
    assert(n < mPageCount);

    const std::pair<size_t, size_t>& indices = mBucketIndices[n];

    // leaf is not stealable if there are any global indices (or no local indices)

    if (indices.first > 0 || indices.second == 0)   return INVALID_INDEX;

    const LocalMovePtrArray& localMoves = mLocalBin[indices.second-1];

    // leaf is not stealable if there are more than one local indices
    // (this corresponds to multiple source leafs with the same origin)

    if (localMoves.size() == 1) {
        const LocalMovePtr& localMove = localMoves.front();

        if (localMove->hasSamePointOrder) {
            return localMove->index;
        }
    }

    return INVALID_INDEX;
}


template<typename PointDataGridT>
inline typename PointDataPartitioner<PointDataGridT>::IndexIterator
PointDataPartitioner<PointDataGridT>::indices(size_t n) const
{
    Index64* start = const_cast<Index64*>(mPageIndices.get());
    if (n > 0)  start += mPageOffsets[n-1];
    Index64* end = const_cast<Index64*>(mPageIndices.get()) + mPageOffsets[n];

    return IndexIterator(start, end);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_DATA_PARTITIONER_HAS_BEEN_INCLUDED
