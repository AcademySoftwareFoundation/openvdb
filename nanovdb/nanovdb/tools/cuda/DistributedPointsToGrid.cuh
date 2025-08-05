// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/DistributedPointsToGrid.cuh

    \brief Generates NanoVDB grids from a list of voxels or points
           in parallel using multiple GPUs

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_DISTRIBUTEDPOINTSTOGRID_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_DISTRIBUTEDPOINTSTOGRID_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceMesh.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/util/cuda/Util.h>

namespace nanovdb {

namespace tools::cuda {

/// @brief Strided iterator for per-device leaf counts which are interleaved with upper and lower counts
struct LeafCountIterator
{
    LeafCountIterator(uint32_t* nodeCounts) : mNodeCounts(nodeCounts) {}

    template <typename Distance>
    uint32_t operator[](Distance n) const { return mNodeCounts[3 * n]; }

private:
    uint32_t* mNodeCounts;
};

/// @brief Indicator functor that returns 1 if the input value matches the member value, 0 otherwise
template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
struct EqualityIndicator
{
    EqualityIndicator(const T* value) : mValue(value) {}

    __hostdev__
    T operator()(const T& x) const
    {
        return x == (*mValue);
    }
private:
    const T* mValue;
};

/// @brief Implements the merge path binary search algorithm in order to find the median across two sorted input key arrays
template<typename KeyIteratorIn>
__device__
void mergePath(KeyIteratorIn keys1, size_t keys1Count, KeyIteratorIn keys2, size_t keys2Count, size_t* key1Intervals, size_t* key2Intervals, int intervalIndex)
{
    using key_type = typename ::cuda::std::iterator_traits<KeyIteratorIn>::value_type;

    const int combinedIndex = intervalIndex * (keys1Count + keys2Count) / 2;
    int leftTop = combinedIndex > keys1Count ? keys1Count : combinedIndex;
    int rightTop = combinedIndex > keys1Count ? combinedIndex - keys1Count : 0;
    int leftBottom = rightTop;

    key_type leftKey;
    key_type rightKey;
    while(true)
    {
        int offset = (leftTop - leftBottom) / 2;
        int leftMid = leftTop - offset;
        int rightMid = rightTop + offset;

        if (leftMid > keys1Count - 1 || rightMid < 1) {
            leftKey = 1;
            rightKey = 0;
        }
        else {
            leftKey = *(keys1 + leftMid);
            rightKey = *(keys2 + rightMid - 1);
        }

        if (leftKey > rightKey) {
            if (rightMid > keys2Count - 1 || leftMid < 1) {
                leftKey = 0;
                rightKey = 1;
            }
            else {
                leftKey = *(keys1 + leftMid - 1);
                rightKey = *(keys2 + rightMid);
            }

            if (leftKey <= rightKey) {
                *key1Intervals = leftMid;
                *key2Intervals = rightMid;
                break;
            }
            else {
                leftTop = leftMid - 1;
                rightTop = rightMid + 1;
            }
        }
        else {
            leftBottom = leftMid + 1;
        }
    }
}

namespace kernels {

/// @brief Kernel wrapper for the merge path algorithm
template<typename KeyIteratorIn>
__global__
void mergePathKernel(KeyIteratorIn keys1, size_t keys1Count, KeyIteratorIn keys2, size_t keys2Count, size_t* key1Intervals, size_t* key2Intervals, size_t intervalOffset)
{
    const unsigned int intervalIndex = threadIdx.x + blockIdx.x * blockDim.x + intervalOffset;
    mergePath(keys1, keys1Count, keys2, keys2Count, key1Intervals, key2Intervals, intervalIndex);
}

/// @brief Extends or shortens the left end of an array interval
template<typename DistanceIteratorIn, typename CountIteratorOut, typename OffsetIteratorOut>
__global__
void leftRebalanceKernel(DistanceIteratorIn leftDistance, DistanceIteratorIn rightDistance, CountIteratorOut leftCount, OffsetIteratorOut leftOffset)
{
    if (*leftDistance < *rightDistance) {
        *leftCount -= *leftDistance;
    }
    else {
        *leftCount += *rightDistance;
    }
}

/// @brief Extends or shortens the right end of an array interval
template<typename DistanceIteratorIn, typename CountIteratorOut, typename OffsetIteratorOut>
__global__
void rightRebalanceKernel(DistanceIteratorIn leftDistance, DistanceIteratorIn rightDistance, CountIteratorOut rightCount, OffsetIteratorOut rightOffset)
{
    if (*leftDistance < *rightDistance) {
        *rightCount += *leftDistance;
        *rightOffset -= *leftDistance;
    }
    else {
        *rightCount -= *rightDistance;
        *rightOffset += *rightDistance;
    }
}

} // namespace kernels

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CUB_LAUNCH
#ifdef _WIN32
#define CUB_LAUNCH(func, pool, stream, ...) \
    cudaCheck(cub::func(nullptr, pool.requestedSize(), __VA_ARGS__, stream)); \
    pool.reallocate(stream); \
    cudaCheck(cub::func(pool.data(), pool.size(), __VA_ARGS__, stream));
#else// fdef _WIN32
#define CUB_LAUNCH(func, pool, stream, args...) \
    cudaCheck(cub::func(nullptr, pool.requestedSize(), args, stream)); \
    pool.reallocate(stream); \
    cudaCheck(cub::func(pool.data(), pool.size(), args, stream));
#endif// ifdef _WIN32
#endif// ifndef CUB_LAUNCH

/// @brief Launches an async exclusive sum operation across multiple devices. The operator waits on the per-device preEvents[deviceId] before summing over that device's contributions and records postEvents[deviceId] when the device's contribution is summed.
template<typename InputIteratorT, typename OutputIteratorT, typename CountIteratorT>
void exclusiveSumAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, InputIteratorT in, OutputIteratorT out, CountIteratorT counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    InputIteratorT deviceIn = in;
    OutputIteratorT deviceOut = out;
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        // Required for the host to pass the correct value of counts[deviceId]
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));
        uint32_t deviceNumItems = counts[deviceId];
        if (deviceId < (deviceMesh.deviceCount() - 1))
            ++deviceNumItems;

        if (deviceId == 0) {
            CUB_LAUNCH(DeviceScan::ExclusiveScan, pools[deviceId], stream, deviceIn, deviceOut, ::cuda::std::plus(), 0, deviceNumItems);
        }
        else {
            cudaCheck(cudaStreamWaitEvent(stream, postEvents[deviceId - 1]));
            deviceIn += counts[deviceId - 1];
            deviceOut += counts[deviceId - 1];
            cub::FutureValue<uint32_t> futureValue(deviceOut);
            CUB_LAUNCH(DeviceScan::ExclusiveScan, pools[deviceId], stream, deviceIn, deviceOut, ::cuda::std::plus(), futureValue, deviceNumItems);
        }
        cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
    }
}

/// @brief Launches an async inclusive sum operation across multiple devices. The operator waits on the per-device preEvents[deviceId] before summing over that device's contributions and records postEvents[deviceId] when the device's contribution is summed.
template<typename InputIteratorT, typename OutputIteratorT, typename CountIteratorT>
void inclusiveSumAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, InputIteratorT in, OutputIteratorT out, CountIteratorT counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    InputIteratorT deviceIn = in;
    OutputIteratorT deviceOut = out;
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        // Required for the host to pass the correct value of counts[deviceId]
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));
        uint32_t deviceNumItems = counts[deviceId];

        if (deviceId == 0) {
            CUB_LAUNCH(DeviceScan::InclusiveScanInit, pools[deviceId], stream, deviceIn, deviceOut, ::cuda::std::plus(), 0, deviceNumItems);
        }
        else {
            cudaCheck(cudaStreamWaitEvent(stream, postEvents[deviceId - 1]));
            deviceIn += counts[deviceId - 1];
            deviceOut += counts[deviceId - 1];
            cub::FutureValue<uint64_t> futureValue(deviceOut - 1);
            CUB_LAUNCH(DeviceScan::InclusiveScanInit, pools[deviceId], stream, deviceIn, deviceOut, ::cuda::std::plus(), futureValue, deviceNumItems);
        }
        cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
    }
}

/// @brief This class implements a multiGPU approach for building NanoVDB grids from input arrays of points
template <typename BuildT>
class DistributedPointsToGrid
{
public:
    /// @brief Constructor that specifies the devices on which to execute and the map for the output grid
    /// @param deviceMesh DeviceMesh on which to run/distribute the operation
    /// @param map Map to be used for the output grid
    DistributedPointsToGrid(const nanovdb::cuda::DeviceMesh& deviceMesh, const Map &map);
    /// @brief Constructor that specifies the devices on which to execute and the scale and translation used to create the map for the output grid
    /// @param deviceMesh DeviceMesh on which to run/distribute the operation
    /// @param scale optional scale factor
    /// @param trans optional translation
    DistributedPointsToGrid(const nanovdb::cuda::DeviceMesh& deviceMesh, const double scale = 1.0, const Vec3d &trans = Vec3d(0.0));

    /// @brief Destructor
    ~DistributedPointsToGrid();

    /// @brief Creates a handle to a grid with the specified build type from a list of points in index or world space
    /// @tparam BuildT Build type of the output grid, i.e NanoGrid<BuildT>
    /// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world or index space.
    /// @tparam BufferT Template type of buffer used for memory allocation on the device. Must support Unified Memory.
    /// @param points device pointer to an array of points or voxels
    /// @param pointCount number of input points or voxels
    /// @param buffer Optional buffer to guide the allocation
    /// @return returns a handle with a grid of type NanoGrid<BuildT> in unified memory
    template <typename PtrT, typename BufferT = nanovdb::cuda::UnifiedBuffer>
    GridHandle<BufferT> getHandle(const PtrT points,
                                  size_t pointCount,
                                  const BufferT &buffer = BufferT());

    template <typename PtrT>
    void countNodes(const PtrT coords, size_t coordCount);

    template <typename PtrT, typename BufferT = nanovdb::cuda::UnifiedBuffer>
    BufferT getBuffer(const PtrT, size_t pointCount, const BufferT &buffer);

    template <typename PtrT>
    void processGridTreeRoot(const PtrT points, size_t pointCount);

    void processNodes();

    template <typename PtrT>
    void processPoints(const PtrT points, size_t pointCount);

    void processBBox();

private:
    static constexpr unsigned int mNumThreads = 128;
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    uint32_t* deviceNodeCount(int deviceId) const { return mNodeCounts + 3 * deviceId; }

    uint32_t* deviceNodeOffset(int deviceId) const { return mNodeOffsets + 3 * deviceId; }

    const nanovdb::cuda::DeviceMesh& mDeviceMesh;
    nanovdb::cuda::TempDevicePool* mTempDevicePools;

    PointType mPointType;
    std::string mGridName;
    PointsToGridData<BuildT> *mData;
    CheckMode mChecksum{CheckMode::Disable};

    size_t* mStripeCounts;
    ptrdiff_t* mStripeOffsets;
    uint32_t* mNodeCounts;
    uint32_t* mNodeOffsets;
    uint32_t* mVoxelCounts;
    uint32_t* mVoxelOffsets;
    size_t* mLeftIntervals;
    size_t* mRightIntervals;

    uint64_t* mKeys;
    uint32_t* mIndices;
    uint32_t* mPointsPerTile;
    uint64_t* mValueIndex;
    uint64_t* mValueIndexPrefix;
};

template <typename BuildT>
DistributedPointsToGrid<BuildT>::DistributedPointsToGrid(const nanovdb::cuda::DeviceMesh& deviceMesh, const Map &map)
    : mDeviceMesh(deviceMesh), mPointType(PointType::Disable)
{
    mTempDevicePools = new nanovdb::cuda::TempDevicePool[mDeviceMesh.deviceCount()];

    cudaCheck(cudaMallocManaged(&mData, sizeof(PointsToGridData<BuildT>)));
    mData->flags.initMask({GridFlags::HasBBox, GridFlags::IsBreadthFirst});
    mData->map = map;

    mStripeCounts = nullptr;
    cudaCheck(cudaMallocManaged(&mStripeCounts, mDeviceMesh.deviceCount() * sizeof(size_t)));
    mStripeOffsets = nullptr;
    cudaCheck(cudaMallocManaged(&mStripeOffsets, mDeviceMesh.deviceCount() * sizeof(ptrdiff_t)));
    mNodeCounts = nullptr;
    cudaCheck(cudaMallocManaged(&mNodeCounts, 3 * mDeviceMesh.deviceCount() * sizeof(uint32_t)));
    mNodeOffsets = nullptr;
    cudaCheck(cudaMallocManaged(&mNodeOffsets, 3 * mDeviceMesh.deviceCount() * sizeof(uint32_t)));
    mVoxelCounts = nullptr;
    cudaCheck(cudaMallocManaged(&mVoxelCounts, mDeviceMesh.deviceCount() * sizeof(uint32_t)));
    mVoxelOffsets = nullptr;
    cudaCheck(cudaMallocManaged(&mVoxelOffsets, mDeviceMesh.deviceCount() * sizeof(uint32_t)));
    mLeftIntervals = nullptr;
    cudaCheck(cudaMallocManaged(&mLeftIntervals, mDeviceMesh.deviceCount() * sizeof(size_t)));
    mRightIntervals = nullptr;
    cudaCheck(cudaMallocManaged(&mRightIntervals, mDeviceMesh.deviceCount() * sizeof(size_t)));
}

template <typename BuildT>
DistributedPointsToGrid<BuildT>::DistributedPointsToGrid(const nanovdb::cuda::DeviceMesh& deviceMesh, const double scale, const Vec3d &trans)
    : DistributedPointsToGrid(deviceMesh, Map(scale, trans))
{
}

template <typename BuildT>
DistributedPointsToGrid<BuildT>::~DistributedPointsToGrid()
{
    cudaCheck(cudaFree(mRightIntervals));
    cudaCheck(cudaFree(mLeftIntervals));
    cudaCheck(cudaFree(mVoxelOffsets));
    cudaCheck(cudaFree(mVoxelCounts));
    cudaCheck(cudaFree(mNodeOffsets));
    cudaCheck(cudaFree(mNodeCounts));
    cudaCheck(cudaFree(mStripeOffsets));
    cudaCheck(cudaFree(mStripeCounts));

    cudaCheck(cudaFree(mData));

    delete[] mTempDevicePools;
}

template<typename BuildT>
template<typename PtrT, typename BufferT>
inline GridHandle<BufferT>
DistributedPointsToGrid<BuildT>::getHandle(const PtrT points, size_t pointCount, const BufferT &pool)
{
    this->countNodes(points, pointCount);

    auto buffer = this->getBuffer<PtrT, BufferT>(points, pointCount, pool);

    this->processGridTreeRoot(points, pointCount);

    this->processNodes();

    this->processPoints(points, pointCount);

    this->processBBox();

    {
        int deviceId = 0;
        auto stream = mDeviceMesh[deviceId].stream;
        cudaCheck(cudaSetDevice(deviceId));
        tools::cuda::updateChecksum((GridData*)buffer.deviceData(), mChecksum, stream);
        cudaCheck(cudaStreamSynchronize(stream));
    }

    return GridHandle<BufferT>(std::move(buffer));
}// DistributedPointsToGrid<BuildT>::getHandle

template <typename BuildT>
template <typename PtrT>
void DistributedPointsToGrid<BuildT>::countNodes(const PtrT coords, size_t coordCount)
{
    // Use cudaMallocManaged calls for now in order to share the PointsToGrid::Data structure
    cudaCheck(cudaMallocManaged(&mData->d_keys, coordCount * sizeof(uint64_t)));
    cudaCheck(cudaMallocManaged(&mData->d_tile_keys, coordCount * sizeof(uint64_t))); // oversubscribe to avoid sync point later
    cudaCheck(cudaMallocManaged(&mData->d_lower_keys, coordCount * sizeof(uint64_t))); // oversubscribe to avoid sync point later
    cudaCheck(cudaMallocManaged(&mData->d_leaf_keys, coordCount * sizeof(uint64_t))); // oversubscribe to avoid sync point later
    cudaCheck(cudaMallocManaged(&mData->d_indx, coordCount * sizeof(uint32_t)));

    cudaCheck(cudaMallocManaged(&mData->pointsPerLeaf, coordCount * sizeof(uint32_t)));
    cudaCheck(cudaMallocManaged(&mData->pointsPerLeafPrefix, coordCount * sizeof(uint32_t)));

    cudaCheck(cudaMallocManaged(&mData->pointsPerVoxel, coordCount * sizeof(uint32_t)));
    cudaCheck(cudaMallocManaged(&mData->pointsPerVoxelPrefix, coordCount * sizeof(uint32_t)));

    cudaCheck(cudaMallocManaged(&mKeys, coordCount * sizeof(uint64_t)));
    cudaCheck(cudaMallocManaged(&mIndices, coordCount * sizeof(uint32_t)));

    cudaCheck(cudaMallocManaged(&mPointsPerTile, coordCount * sizeof(uint32_t)));

    if constexpr(BuildTraits<BuildT>::is_onindex) {
        cudaCheck(cudaMallocManaged(&mValueIndex, coordCount * sizeof(uint64_t))); // oversubscribe to avoid sync point later
        cudaCheck(cudaMallocManaged(&mValueIndexPrefix, coordCount * sizeof(uint64_t))); // oversubscribe to avoid sync point later
    }

    // Create events required for host-device and cross-device synchronization. Disable timing if not needed in order
    // to reduce overhead.
    std::vector<cudaEvent_t> sortEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> runLengthEncodeEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> transformReduceEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> rebalanceEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> tilePrefixSumEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> voxelCountEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> leafCountEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> lowerCountEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> voxelPrefixSumEvents(mDeviceMesh.deviceCount());
    std::vector<cudaEvent_t> leafPrefixSumEvents(mDeviceMesh.deviceCount());
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaEventCreateWithFlags(&sortEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&runLengthEncodeEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&transformReduceEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&rebalanceEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&tilePrefixSumEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&voxelCountEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&leafCountEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&lowerCountEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&voxelPrefixSumEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&leafPrefixSumEvents[deviceId], cudaEventDisableTiming);
    }

    // Advise per-coord quantities to be split even across devices
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        size_t deviceStripeCount = (coordCount + mDeviceMesh.deviceCount() - 1) / mDeviceMesh.deviceCount();
        const ptrdiff_t deviceStripeOffset = deviceStripeCount * deviceId;
        deviceStripeCount = std::min(deviceStripeCount, coordCount - deviceStripeOffset);

        mStripeCounts[deviceId] = deviceStripeCount;
        mStripeOffsets[deviceId] = deviceStripeOffset;

        nanovdb::Coord* deviceCoords = coords + deviceStripeOffset;
        uint64_t* deviceInputKeys = mKeys + deviceStripeOffset;
        uint32_t* deviceInputIndices = mIndices + deviceStripeOffset;
        uint64_t* deviceOutputKeys = mData->d_keys + deviceStripeOffset;
        uint32_t* deviceOutputIndices = mData->d_indx + deviceStripeOffset;

        util::cuda::memAdvise(deviceCoords, deviceStripeCount * sizeof(nanovdb::Coord), cudaMemAdviseSetPreferredLocation, deviceId);
        util::cuda::memAdvise(deviceCoords, deviceStripeCount * sizeof(nanovdb::Coord), cudaMemAdviseSetReadMostly, deviceId);

        util::cuda::memAdvise(deviceInputKeys, deviceStripeCount * sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, deviceId);
        util::cuda::memAdvise(deviceInputIndices, deviceStripeCount * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);
        util::cuda::memAdvise(deviceOutputKeys, deviceStripeCount * sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, deviceId);
        util::cuda::memAdvise(deviceOutputIndices, deviceStripeCount * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);

        uint32_t* devicePointsPerTile = mPointsPerTile + deviceStripeOffset;
        util::cuda::memAdvise(devicePointsPerTile, deviceStripeCount * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);
        util::cuda::memAdvise(deviceNodeCount(deviceId), 3 * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);
    }

    // Radix sort the subset of keys assigned to each device in parallel
    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];

        uint64_t* deviceInputKeys = mKeys + deviceStripeOffset;
        uint32_t* deviceInputIndices = mIndices + deviceStripeOffset;
        uint64_t* deviceOutputKeys = mData->d_keys + deviceStripeOffset;
        uint32_t* deviceOutputIndices = mData->d_indx + deviceStripeOffset;

        util::cuda::memPrefetchAsync(coords, coordCount * sizeof(nanovdb::Coord), deviceId, stream);

        nanovdb::util::cuda::offsetLambdaKernel<<<numBlocks(deviceStripeCount), mNumThreads, 0, stream>>>(deviceStripeCount, deviceStripeOffset, TileKeyFunctor<BuildT, PtrT>(), mData, coords, mKeys, mIndices);

        CUB_LAUNCH(DeviceRadixSort::SortPairs, mTempDevicePools[deviceId], stream, deviceInputKeys, deviceOutputKeys, deviceInputIndices, deviceOutputIndices, deviceStripeCount, 0, 63);
        cudaEventRecord(sortEvents[deviceId], stream);
    });

    // TODO: Generalize to numbers of GPUs that aren't powers of two

    // For each pair of devices, merge the local sorts by first computing the median across the two devices followed by merging
    // the elements less than and greater than/equal to the median onto the first and second device of the pair respectively.
    // This avoids the allocating memory for and gathering the values from both devices onto a single device.

    std::vector<std::thread> threads;
    const size_t log2DeviceCount = log2(mDeviceMesh.deviceCount());
    for (size_t deviceExponent = 0; deviceExponent < log2DeviceCount; ++deviceExponent) {
        const size_t deviceGroupCount = mDeviceMesh.deviceCount() >> deviceExponent;
        for (size_t deviceGroupId = 0; deviceGroupId < deviceGroupCount; deviceGroupId += 2) {
            threads.emplace_back([&, deviceGroupId]() {
                const int leftDeviceGroupId = deviceGroupId;
                const int rightDeviceGroupId = deviceGroupId + 1;
                const int leftDeviceId = leftDeviceGroupId << deviceExponent;
                const int rightDeviceId = rightDeviceGroupId << deviceExponent;

                size_t deviceStripeCount = (coordCount + deviceGroupCount - 1) / deviceGroupCount;
                const ptrdiff_t deviceStripeOffset = deviceStripeCount * leftDeviceGroupId;
                deviceStripeCount = std::min(deviceStripeCount, coordCount - deviceStripeOffset);

                const auto leftDeviceStripeOffset = deviceStripeCount * leftDeviceGroupId;
                const auto leftDeviceStripeCount = std::min(deviceStripeCount, coordCount - leftDeviceStripeOffset);
                const auto rightDeviceStripeOffset = deviceStripeCount * rightDeviceGroupId;
                const auto rightDeviceStripeCount = std::min(deviceStripeCount, coordCount - rightDeviceStripeOffset);

                const uint64_t* inputKeys = (deviceExponent % 2) ? mKeys : mData->d_keys;
                const uint32_t* inputIndices = (deviceExponent % 2) ? mIndices : mData->d_indx;
                uint64_t* outputKeys = (deviceExponent % 2) ? mData->d_keys : mKeys;
                uint32_t* outputIndices = (deviceExponent % 2) ? mData->d_indx : mIndices;

                const uint64_t* leftDeviceInputKeys = inputKeys + leftDeviceStripeOffset;
                const uint32_t* leftDeviceInputIndices = inputIndices + leftDeviceStripeOffset;
                const uint64_t* rightDeviceInputKeys = inputKeys + rightDeviceStripeOffset;
                const uint32_t* rightDeviceInputIndices = inputIndices + rightDeviceStripeOffset;

                // Wait on the prior sort to finish on both devices before computing the median across both devices
                auto mergePathFunc = [&](int deviceId, int otherDeviceId, int intervalIndex) {
                    cudaError_t cudaStatus = cudaSetDevice(deviceId);
                    assert(cudaStatus == cudaSuccess);

                    cudaStreamWaitEvent(mDeviceMesh[deviceId].stream, sortEvents[otherDeviceId]);
                    kernels::mergePathKernel<<<1, 1, 0, mDeviceMesh[deviceId].stream>>>(leftDeviceInputKeys, leftDeviceStripeCount, rightDeviceInputKeys, rightDeviceStripeCount, &mLeftIntervals[deviceId], &mRightIntervals[deviceId], intervalIndex);
                    cudaStreamSynchronize(mDeviceMesh[deviceId].stream);
                };

                std::thread leftMergePathThread(mergePathFunc, leftDeviceId, rightDeviceId, 0);
                std::thread rightMergePathThread(mergePathFunc, rightDeviceId, leftDeviceId, 1);
                leftMergePathThread.join();
                rightMergePathThread.join();

                // Wait on the median computation prior to merging the sorts
                auto leftMergeFunc = [&](int leftDeviceId, int rightDeviceId) {
                    cudaError_t cudaStatus = cudaSetDevice(leftDeviceId);
                    assert(cudaStatus == cudaSuccess);

                    const uint64_t* leftKeysIn = leftDeviceInputKeys + mLeftIntervals[leftDeviceId];
                    const uint32_t* leftIndicesIn = leftDeviceInputIndices + mLeftIntervals[leftDeviceId];
                    size_t leftCount = mLeftIntervals[rightDeviceId] - mLeftIntervals[leftDeviceId];

                    const uint64_t* rightKeysIn = rightDeviceInputKeys + mRightIntervals[leftDeviceId];
                    const uint32_t* rightIndicesIn = rightDeviceInputIndices + mRightIntervals[leftDeviceId];
                    size_t rightCount = mRightIntervals[rightDeviceId] - mRightIntervals[leftDeviceId];

                    size_t outputOffset = leftDeviceStripeOffset + mLeftIntervals[leftDeviceId] + mRightIntervals[leftDeviceId];
                    uint64_t* keysOut = outputKeys + outputOffset;
                    uint32_t* indicesOut = outputIndices + outputOffset;

                    CUB_LAUNCH(DeviceMerge::MergePairs, mTempDevicePools[leftDeviceId], mDeviceMesh[leftDeviceId].stream, leftKeysIn, leftIndicesIn, leftCount, rightKeysIn, rightIndicesIn, rightCount, keysOut, indicesOut, {});
                    cudaEventRecord(sortEvents[leftDeviceId], mDeviceMesh[leftDeviceId].stream);
                };

                auto rightMergeFunc = [&](int leftDeviceId, int rightDeviceId) {
                    cudaError_t cudaStatus = cudaSetDevice(rightDeviceId);
                    assert(cudaStatus == cudaSuccess);

                    const uint64_t* leftKeysIn = leftDeviceInputKeys + mLeftIntervals[rightDeviceId];
                    const uint32_t* leftIndicesIn = leftDeviceInputIndices + mLeftIntervals[rightDeviceId];
                    size_t leftCount = leftDeviceStripeCount - mLeftIntervals[rightDeviceId];

                    const uint64_t* rightKeysIn = rightDeviceInputKeys + mRightIntervals[rightDeviceId];
                    const uint32_t* rightIndicesIn = rightDeviceInputIndices + mRightIntervals[rightDeviceId];
                    size_t rightCount = rightDeviceStripeCount - mRightIntervals[rightDeviceId];

                    size_t outputOffset = leftDeviceStripeOffset + mLeftIntervals[rightDeviceId] + mRightIntervals[rightDeviceId];
                    uint64_t* keysOut = outputKeys + outputOffset;
                    uint32_t* indicesOut = outputIndices + outputOffset;

                    CUB_LAUNCH(DeviceMerge::MergePairs, mTempDevicePools[rightDeviceId], mDeviceMesh[rightDeviceId].stream, leftKeysIn, leftIndicesIn, leftCount, rightKeysIn, rightIndicesIn, rightCount, keysOut, indicesOut, {});
                    cudaEventRecord(sortEvents[rightDeviceId], mDeviceMesh[rightDeviceId].stream);
                };

                // Merge the pairs less than the median to the left device
                std::thread leftMergeThread(leftMergeFunc, leftDeviceId, rightDeviceId);
                // Merge the pairs greater than/equal to the median to the right device
                std::thread rightMergeThread(rightMergeFunc, leftDeviceId, rightDeviceId);
                leftMergeThread.join();
                rightMergeThread.join();

                cudaCheck(cudaEventSynchronize(sortEvents[leftDeviceId]));
                cudaCheck(cudaEventSynchronize(sortEvents[rightDeviceId]));
            });
        }
        std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
        threads.clear();
    }

    // There is no merging required for a single device so we simply copy the sorted result to the destination array (where the sort would have been merged to).
    if (!(log2DeviceCount % 2)) {
        parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
            cudaCheck(cudaSetDevice(deviceId));

            auto deviceStripeCount = mStripeCounts[deviceId];
            auto deviceStripeOffset = mStripeOffsets[deviceId];

            cudaMemcpyAsync(mKeys + deviceStripeOffset, mData->d_keys + deviceStripeOffset, deviceStripeCount * sizeof(uint64_t), cudaMemcpyDefault, stream);
            cudaMemcpyAsync(mIndices + deviceStripeOffset, mData->d_indx + deviceStripeOffset, deviceStripeCount * sizeof(uint32_t), cudaMemcpyDefault, stream);

            cudaEventRecord(sortEvents[deviceId], stream);
        });
    }

    // For each segment of sorted keys on each device, we count how many of the leftmost key occur past the left boundary of the segment. The same is done for the rightmost key with the right boundary of the segment.
    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];
        uint64_t* deviceInputKeys = mKeys + deviceStripeOffset;

        if (deviceId > 0) {
            cudaStreamWaitEvent(stream, sortEvents[deviceId - 1]);
            EqualityIndicator<uint64_t> indicator(deviceInputKeys - 1);
            CUB_LAUNCH(DeviceReduce::TransformReduce, mTempDevicePools[deviceId], stream, deviceInputKeys, mRightIntervals + deviceId, deviceStripeCount, ::cuda::std::plus(), indicator, 0);
        }
        else {
            mRightIntervals[deviceId] = 0;
        }

        if (deviceId < static_cast<int>(mDeviceMesh.deviceCount() - 1)) {
            cudaStreamWaitEvent(stream, sortEvents[deviceId + 1]);
            EqualityIndicator<uint64_t> indicator(deviceInputKeys + deviceStripeCount);
            CUB_LAUNCH(DeviceReduce::TransformReduce, mTempDevicePools[deviceId], stream, deviceInputKeys, mLeftIntervals + deviceId, deviceStripeCount, ::cuda::std::plus(), indicator, 0);
        }
        else {
            mLeftIntervals[deviceId] = 0;
        }
        cudaEventRecord(transformReduceEvents[deviceId], stream);
    });

    // Rebalance the segments so that a device segment boundary also corresponds to a change in key value. Effectively, this aligns upper node boundaries with device ownership boundaries.
    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));

        if (deviceId > 0)
        {
            cudaStreamWaitEvent(stream, transformReduceEvents[deviceId - 1]);
            kernels::rightRebalanceKernel<<<1, 1, 0, stream>>>(mLeftIntervals + deviceId - 1, mRightIntervals + deviceId, mStripeCounts + deviceId, mStripeOffsets + deviceId);
        }

        if (deviceId < static_cast<int>(mDeviceMesh.deviceCount() - 1))
        {
            cudaStreamWaitEvent(stream, transformReduceEvents[deviceId + 1]);
            kernels::leftRebalanceKernel<<<1, 1, 0, stream>>>(mLeftIntervals + deviceId, mRightIntervals + deviceId + 1, mStripeCounts + deviceId, mStripeOffsets + deviceId);
        }
        cudaEventRecord(rebalanceEvents[deviceId], stream);
    });

    // Parallel RLE in order to obtain tiles
    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));

        cudaCheck(cudaEventSynchronize(rebalanceEvents[deviceId]));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];

        uint64_t* deviceInputKeys = mKeys + deviceStripeOffset;
        uint64_t* deviceOutputKeys = mData->d_keys + deviceStripeOffset;
        uint32_t* devicePointsPerTile = mPointsPerTile + deviceStripeOffset;

        // util::cuda::memPrefetchAsync(deviceInputKeys, deviceStripeCount * sizeof(uint64_t), deviceId, stream);

        CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, deviceInputKeys, deviceOutputKeys, devicePointsPerTile, deviceNodeCount(deviceId) + 2, deviceStripeCount);
        cudaCheck(cudaEventRecord(runLengthEncodeEvents[deviceId], stream));
    });

    uint32_t upperOffset = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaEventSynchronize(runLengthEncodeEvents[deviceId]));
        auto deviceStripeOffset = mStripeOffsets[deviceId];
        uint64_t* deviceKeys = mData->d_keys + deviceStripeOffset;
        cudaCheck(cudaMemcpyAsync(mData->d_tile_keys + upperOffset, deviceKeys, sizeof(uint64_t) * deviceNodeCount(deviceId)[2], cudaMemcpyDefault, stream));
        deviceNodeOffset(deviceId)[2] = upperOffset;
        upperOffset += deviceNodeCount(deviceId)[2];
    }

    // For each tile in parallel, we construct another set of keys for the lower nodes, leaf nodes, and voxels within that tile followed by a radix sort of these keys.
    for (int deviceId = 0, id = 0; deviceId < static_cast<int>(mDeviceMesh.deviceCount()); ++deviceId) {
        auto stream = mDeviceMesh[deviceId].stream;
        cudaCheck(cudaSetDevice(deviceId));

        uint32_t* devicePointsPerTile = mPointsPerTile + mStripeOffsets[deviceId];
        for (uint32_t i = 0, tileOffset = 0; i < deviceNodeCount(deviceId)[2]; ++i) {
            if (!devicePointsPerTile[i]) continue;

            nanovdb::util::cuda::offsetLambdaKernel<<<numBlocks(devicePointsPerTile[i]), mNumThreads, 0, stream>>>(devicePointsPerTile[i], tileOffset + mStripeOffsets[deviceId], VoxelKeyFunctor<BuildT, PtrT>(), mData, coords, id, mKeys, mIndices);

            uint64_t* tileInputKeys = mKeys + tileOffset + mStripeOffsets[deviceId];
            uint32_t* tileInputIndices = mIndices + tileOffset + mStripeOffsets[deviceId];
            uint64_t* tileOutputKeys = mData->d_keys + tileOffset + mStripeOffsets[deviceId];
            uint32_t* tileOutputIndices = mData->d_indx + tileOffset + mStripeOffsets[deviceId];

            CUB_LAUNCH(DeviceRadixSort::SortPairs, mTempDevicePools[deviceId], stream, tileInputKeys, tileOutputKeys, tileInputIndices, tileOutputIndices, devicePointsPerTile[i], 0, 36);// 9+12+15=36
            ++id;
            tileOffset += devicePointsPerTile[i];
        }
    }

    // For each of the following operations, the input on the current device depends on the output of the prior device. Thus, for maximum throughput, we pipeline these operations.
    // 1) RLE for pointsPerLeaf
    // 2) RLE for pointsPerVoxel
    // 3) Prefix sum over pointsPerLeaf
    // 4) Prefix sum over pointsPerVoxel
    // Without this pipelining, each operation would have to wait until ALL devices to finish their prior operation instead of just the previous device which significantly degrades scaling.
    // Based on profiling, we launch the per-device kernels for steps 1 through 3 in a single loop, followed by launching the per-device kernels for step 4 in a separate loop.
    // This can be improved when/if CUB implements cub::FutureValue support for size parameters.
    {
        LeafCountIterator leafCountIterator(mNodeCounts);
        uint32_t* devicePointsPerVoxel = mData->pointsPerVoxel;
        uint32_t* devicePointsPerLeaf = mData->pointsPerLeaf;
        uint32_t* devicePointsPerVoxelPrefix = mData->pointsPerVoxelPrefix;
        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaCheck(cudaSetDevice(deviceId));

            uint64_t* deviceInputKeys = mKeys + mStripeOffsets[deviceId];
            uint64_t* deviceOutputKeys = mData->d_keys + mStripeOffsets[deviceId];

            if (deviceId == 0) {
                CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, deviceOutputKeys, deviceInputKeys, devicePointsPerVoxel, mVoxelCounts + deviceId, mStripeCounts[deviceId]);
                cudaCheck(cudaEventRecord(voxelCountEvents[deviceId], stream));

                CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, thrust::make_transform_iterator(deviceOutputKeys, ShiftRight<9>()), deviceInputKeys, devicePointsPerLeaf, deviceNodeCount(deviceId), mStripeCounts[deviceId]);
                cudaCheck(cudaEventRecord(leafCountEvents[deviceId], stream));
            }
            else
            {
                cudaCheck(cudaEventSynchronize(voxelCountEvents[deviceId - 1]));
                devicePointsPerVoxel += mVoxelCounts[deviceId - 1];
                CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, deviceOutputKeys, deviceInputKeys, devicePointsPerVoxel, mVoxelCounts + deviceId, mStripeCounts[deviceId]);
                cudaCheck(cudaEventRecord(voxelCountEvents[deviceId], stream));

                cudaCheck(cudaEventSynchronize(leafCountEvents[deviceId - 1]));
                devicePointsPerLeaf += deviceNodeCount(deviceId - 1)[0];
                CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, thrust::make_transform_iterator(deviceOutputKeys, ShiftRight<9>()), deviceInputKeys, devicePointsPerLeaf, deviceNodeCount(deviceId), mStripeCounts[deviceId]);
                cudaCheck(cudaEventRecord(leafCountEvents[deviceId], stream));
            }

            cudaCheck(cudaEventSynchronize(voxelCountEvents[deviceId]));
            uint32_t deviceNumItems = mVoxelCounts[deviceId];
            if (deviceId < static_cast<int>(mDeviceMesh.deviceCount() - 1))
                ++deviceNumItems;

            if (deviceId == 0) {
                CUB_LAUNCH(DeviceScan::ExclusiveScan, mTempDevicePools[deviceId], stream, devicePointsPerVoxel, devicePointsPerVoxelPrefix, ::cuda::std::plus(), 0, deviceNumItems);
            }
            else {
                cudaCheck(cudaStreamWaitEvent(stream, voxelPrefixSumEvents[deviceId - 1]));
                devicePointsPerVoxelPrefix += mVoxelCounts[deviceId - 1];
                cub::FutureValue<uint32_t> futureValue(devicePointsPerVoxelPrefix);
                CUB_LAUNCH(DeviceScan::ExclusiveScan, mTempDevicePools[deviceId], stream, devicePointsPerVoxel, devicePointsPerVoxelPrefix, ::cuda::std::plus(), futureValue, deviceNumItems);
            }

            cudaCheck(cudaEventRecord(voxelPrefixSumEvents[deviceId], stream));

        }
    }

    {
        LeafCountIterator leafCountIterator(mNodeCounts);
        uint32_t* devicePointsPerLeaf = mData->pointsPerLeaf;
        uint32_t* devicePointsPerLeafPrefix = mData->pointsPerLeafPrefix;
        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaCheck(cudaSetDevice(deviceId));

            // Required for the host to pass the correct value of counts[deviceId]
            cudaCheck(cudaEventSynchronize(leafCountEvents[deviceId]));
            uint32_t deviceNumItems = leafCountIterator[deviceId];
            if (deviceId < static_cast<int>(mDeviceMesh.deviceCount() - 1))
                ++deviceNumItems;

            if (deviceId == 0) {
                CUB_LAUNCH(DeviceScan::ExclusiveScan, mTempDevicePools[deviceId], stream, devicePointsPerLeaf, devicePointsPerLeafPrefix, ::cuda::std::plus(), 0, deviceNumItems);
            }
            else {
                cudaCheck(cudaStreamWaitEvent(stream, leafPrefixSumEvents[deviceId - 1]));
                devicePointsPerLeaf += leafCountIterator[deviceId - 1];
                devicePointsPerLeafPrefix += leafCountIterator[deviceId - 1];
                cub::FutureValue<uint32_t> futureValue(devicePointsPerLeafPrefix);
                CUB_LAUNCH(DeviceScan::ExclusiveScan, mTempDevicePools[deviceId], stream, devicePointsPerLeaf, devicePointsPerLeafPrefix, ::cuda::std::plus(), futureValue, deviceNumItems);
            }
            cudaCheck(cudaEventRecord(leafPrefixSumEvents[deviceId], stream));
        }
    }

    uint32_t leafOffset = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        uint64_t* deviceKeys = mKeys + mStripeOffsets[deviceId];
        cudaCheck(cudaMemcpyAsync(mData->d_leaf_keys + leafOffset, deviceKeys, sizeof(uint64_t) * deviceNodeCount(deviceId)[0], cudaMemcpyDefault, stream));
        deviceNodeOffset(deviceId)[0] = leafOffset;
        leafOffset += deviceNodeCount(deviceId)[0];
    }

    // Parallel RLE with (shifted) keys in order to count leaves and points per leaf
    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));

        uint64_t* deviceInputKeys = mKeys + mStripeOffsets[deviceId];
        uint64_t* deviceOutputKeys = mData->d_keys + mStripeOffsets[deviceId];

        CUB_LAUNCH(DeviceSelect::Unique, mTempDevicePools[deviceId], stream, thrust::make_transform_iterator(deviceOutputKeys, ShiftRight<21>()), deviceInputKeys, deviceNodeCount(deviceId) + 1, mStripeCounts[deviceId]);
        cudaEventRecord(lowerCountEvents[deviceId], stream);
    });

    uint32_t lowerOffset = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaEventSynchronize(lowerCountEvents[deviceId]));
        uint64_t* deviceKeys = mKeys + mStripeOffsets[deviceId];
        cudaCheck(cudaMemcpyAsync(mData->d_lower_keys + lowerOffset, deviceKeys, sizeof(uint64_t) * deviceNodeCount(deviceId)[1], cudaMemcpyDefault, stream));
        deviceNodeOffset(deviceId)[1] = lowerOffset;
        lowerOffset += deviceNodeCount(deviceId)[1];
    }

    uint32_t voxelOffset = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        mVoxelOffsets[deviceId] = voxelOffset;
        voxelOffset += mVoxelCounts[deviceId];
    }

    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaStreamSynchronize(stream));
    });

    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaEventDestroy(sortEvents[deviceId]);
        cudaEventDestroy(runLengthEncodeEvents[deviceId]);
        cudaEventDestroy(transformReduceEvents[deviceId]);
        cudaEventDestroy(rebalanceEvents[deviceId]);
        cudaEventDestroy(tilePrefixSumEvents[deviceId]);
        cudaEventDestroy(voxelCountEvents[deviceId]);
        cudaEventDestroy(leafCountEvents[deviceId]);
        cudaEventDestroy(lowerCountEvents[deviceId]);
        cudaEventDestroy(voxelPrefixSumEvents[deviceId]);
        cudaEventDestroy(leafPrefixSumEvents[deviceId]);
    }
} // DistributedPointsToGrid<BuildT>::countNodes

template <typename BuildT>
template <typename PtrT, typename BufferT>
inline BufferT DistributedPointsToGrid<BuildT>::getBuffer(const PtrT, size_t pointCount, const BufferT &pool)
{
    auto sizeofPoint = [&]()->size_t{
        switch (mPointType){
        case PointType::PointID: return sizeof(uint32_t);
        case PointType::World64: return sizeof(Vec3d);
        case PointType::World32: return sizeof(Vec3f);
        case PointType::Grid64:  return sizeof(Vec3d);
        case PointType::Grid32:  return sizeof(Vec3f);
        case PointType::Voxel32: return sizeof(Vec3f);
        case PointType::Voxel16: return sizeof(Vec3u16);
        case PointType::Voxel8:  return sizeof(Vec3u8);
        case PointType::Default: return pointer_traits<PtrT>::element_size;
        default: return size_t(0);// PointType::Disable
        }
    };

    mData->grid  = 0;// grid is always stored at the start of the buffer!
    mData->tree  = NanoGrid<BuildT>::memUsage(); // grid ends and tree begins
    mData->root  = mData->tree  + NanoTree<BuildT>::memUsage(); // tree ends and root node begins

    mData->nodeCount[0] = 0;
    mData->nodeCount[1] = 0;
    mData->nodeCount[2] = 0;
    mData->voxelCount = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        mData->nodeCount[0] += deviceNodeCount(deviceId)[0];
        mData->nodeCount[1] += deviceNodeCount(deviceId)[1];
        mData->nodeCount[2] += deviceNodeCount(deviceId)[2];
        mData->voxelCount += mVoxelCounts[deviceId];
    }

    mData->upper = mData->root  + NanoRoot<BuildT>::memUsage(mData->nodeCount[2]); // root node ends and upper internal nodes begin
    mData->lower = mData->upper + NanoUpper<BuildT>::memUsage()*(mData->nodeCount[2]); // upper internal nodes ends and lower internal nodes begin
    mData->leaf  = mData->lower + NanoLower<BuildT>::memUsage()*(mData->nodeCount[1]); // lower internal nodes ends and leaf nodes begin
    mData->meta  = mData->leaf  + NanoLeaf<BuildT>::DataType::memUsage()*(mData->nodeCount[0]);// leaf nodes end and blind meta data begins
    mData->blind = mData->meta  + sizeof(GridBlindMetaData)*int( mPointType!=PointType::Disable ); // meta data ends and blind data begins
    mData->size  = mData->blind + pointCount*sizeofPoint();// end of buffer

    auto buffer = BufferT::create(mData->size, &pool);
    mData->d_bufferPtr = buffer.deviceData();
    if (!mData->d_bufferPtr)
        throw std::runtime_error("Failed to allocate grid buffer in Unified Memory");
    return buffer;
}// DistributedPointsToGrid<BuildT>::getBuffer

template <typename BuildT>
template <typename PtrT>
inline void DistributedPointsToGrid<BuildT>::processGridTreeRoot(const PtrT points, size_t pointCount)
{
    // Process root node on device 0. Other devices will wait until root node processing is complete.
    int deviceId = 0;
    auto stream = mDeviceMesh[deviceId].stream;
    cudaCheck(cudaSetDevice(deviceId));
    cudaEvent_t processGridTreeRootEvent;
    cudaEventCreateWithFlags(&processGridTreeRootEvent, cudaEventDisableTiming);
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, BuildGridTreeRootFunctor<BuildT, PtrT>(), mData, mPointType, pointCount);// lambdaKernel
    cudaCheckError();

    char *dst = mData->getGrid().mGridName;
    if (const char *src = mGridName.data()) {
        cudaCheck(cudaMemcpyAsync(dst, src, GridData::MaxNameSize, cudaMemcpyHostToDevice, stream));
    } else {
        cudaCheck(cudaMemsetAsync(dst, 0, GridData::MaxNameSize, stream));
    }
    cudaEventRecord(processGridTreeRootEvent);

    parallelForEach(mDeviceMesh, [&](int otherDeviceId, cudaStream_t otherStream) {
        cudaSetDevice(otherDeviceId);
        cudaStreamWaitEvent(otherStream, processGridTreeRootEvent);
    });

    cudaCheck(cudaSetDevice(deviceId));
    cudaEventDestroy(processGridTreeRootEvent);
}// DistributedPointsToGrid<BuildT>::processGridTreeRoot

template <typename BuildT>
inline void DistributedPointsToGrid<BuildT>::processNodes()
{
    // Parallel construction of upper, lower, and leaf nodes
    const uint8_t flags = static_cast<uint8_t>(mData->flags.data());// mIncludeStats ? 16u : 0u;// 4th bit indicates stats

    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));

        if (deviceNodeCount(deviceId)[2]) {
            util::cuda::offsetLambdaKernel<<<numBlocks(deviceNodeCount(deviceId)[2]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[2], deviceNodeOffset(deviceId)[2], BuildUpperNodesFunctor<BuildT>(), mData);
            cudaCheckError();

            const uint64_t valueCount = deviceNodeCount(deviceId)[2] << 15;
            const uint64_t valueOffset = deviceNodeOffset(deviceId)[2] << 15;
            util::cuda::offsetLambdaKernel<<<numBlocks(valueCount), mNumThreads, 0, stream>>>(valueCount, valueOffset, SetUpperBackgroundValuesFunctor<BuildT>(), mData);
            cudaCheckError();
        }

        if (deviceNodeCount(deviceId)[1]) {
            util::cuda::offsetLambdaKernel<<<numBlocks(deviceNodeCount(deviceId)[1]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[1], deviceNodeOffset(deviceId)[1], BuildLowerNodesFunctor<BuildT>(), mData);
            cudaCheckError();

            const uint64_t valueCount = deviceNodeCount(deviceId)[1] << 12;
            const uint64_t valueOffset = deviceNodeOffset(deviceId)[1] << 12;
            util::cuda::offsetLambdaKernel<<<numBlocks(valueCount), mNumThreads, 0, stream>>>(valueCount, valueOffset, SetLowerBackgroundValuesFunctor<BuildT>(), mData);
            cudaCheckError();
        }


        if (deviceNodeCount(deviceId)[0]) {
            // loop over leaf nodes and add it to its parent node
            util::cuda::offsetLambdaKernel<<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], ProcessLeafMetaDataFunctor<BuildT>(), mData, flags);
            cudaCheckError();

            // loop over all active voxels and set LeafNode::mValueMask and LeafNode::mValues
            util::cuda::offsetLambdaKernel<<<numBlocks(mVoxelCounts[deviceId]), mNumThreads, 0, stream>>>(mVoxelCounts[deviceId], mVoxelOffsets[deviceId], SetLeafActiveVoxelStateAndValuesFunctor<BuildT>(), mData);
            cudaCheckError();

            const uint64_t denseVoxelCount = deviceNodeCount(deviceId)[0] << 9;
            const uint64_t denseVoxelOffset = deviceNodeOffset(deviceId)[0] << 9;
            util::cuda::offsetLambdaKernel<<<numBlocks(denseVoxelCount), mNumThreads, 0, stream>>>(denseVoxelCount, denseVoxelOffset, SetLeafInactiveVoxelValuesFunctor<BuildT>(), mData);
            cudaCheckError();
        }
    });

    if constexpr(BuildTraits<BuildT>::is_onindex) {
        std::vector<cudaEvent_t> leafCountEvents(mDeviceMesh.deviceCount());
        std::vector<cudaEvent_t> valueIndexPrefixSumEvents(mDeviceMesh.deviceCount());

        parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
            cudaSetDevice(deviceId);
            cudaEventCreateWithFlags(&leafCountEvents[deviceId], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&valueIndexPrefixSumEvents[deviceId], cudaEventDisableTiming);

            if (deviceNodeCount(deviceId)[0]) {
                kernels::fillValueIndexKernel<BuildT><<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], mValueIndex, mData);
                cudaCheckError();
            }
        });

        LeafCountIterator leafCountIterator(mNodeCounts);
        inclusiveSumAsync(mDeviceMesh, mTempDevicePools, mValueIndex, mValueIndexPrefix, leafCountIterator, leafCountEvents.data(), valueIndexPrefixSumEvents.data());

        parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
            cudaSetDevice(deviceId);
            cudaStreamWaitEvent(stream, valueIndexPrefixSumEvents.back());
            if (deviceNodeCount(deviceId)[0]) {
                kernels::leafPrefixSumKernel<BuildT><<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], mValueIndexPrefix, mData);
                cudaCheckError();
            }
        });

        parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
            cudaSetDevice(deviceId);
            cudaEventDestroy(valueIndexPrefixSumEvents[deviceId]);
            cudaEventDestroy(leafCountEvents[deviceId]);
        });
    }

    if constexpr(BuildTraits<BuildT>::is_indexmask) {
        parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
            cudaCheck(cudaSetDevice(deviceId));
            if (deviceNodeCount(deviceId)[0]) {
                kernels::setMaskEqValMaskKernel<BuildT><<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], mData);
                cudaCheckError();
            }
        });
    }
}// DistributedPointsToGrid<BuildT>::processNodes

template <typename BuildT>
template <typename PtrT>
inline void DistributedPointsToGrid<BuildT>::processPoints(const PtrT, size_t)
{
}

template <typename BuildT>
inline void DistributedPointsToGrid<BuildT>::processBBox()
{
    if (mData->flags.isMaskOn(GridFlags::HasBBox)) {
        // Compute and propagate bounding boxes for the upper nodes and their descendents belonging to each device in parallel.
        std::vector<cudaEvent_t> propagateLowerBBoxEvents(mDeviceMesh.deviceCount());
        parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
            cudaCheck(cudaSetDevice(deviceId));
            // reset bbox in lower nodes
            if (deviceNodeCount(deviceId)[1]) {
                util::cuda::offsetLambdaKernel<<<numBlocks(deviceNodeCount(deviceId)[1]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[1], deviceNodeOffset(deviceId)[1], ResetLowerNodeBBoxFunctor<BuildT>(), mData);
                cudaCheckError();
            }

            // update and propagate bbox from leaf -> lower/parent nodes
            if (deviceNodeCount(deviceId)[0]) {
                util::cuda::offsetLambdaKernel<<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], UpdateAndPropagateLeafBBoxFunctor<BuildT>(), mData);
                cudaCheckError();
            }

            // reset bbox in upper nodes
            if (deviceNodeCount(deviceId)[2]) {
                util::cuda::offsetLambdaKernel<<<numBlocks(deviceNodeCount(deviceId)[2]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[2], deviceNodeOffset(deviceId)[2], ResetUpperNodeBBoxFunctor<BuildT>(), mData);
                cudaCheckError();
            }

            // propagate bbox from lower -> upper/parent node
            if (deviceNodeCount(deviceId)[1]) {
                util::cuda::offsetLambdaKernel<<<numBlocks(deviceNodeCount(deviceId)[1]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[1], deviceNodeOffset(deviceId)[1], PropagateLowerBBoxFunctor<BuildT>(), mData);
                cudaCheckError();
            }

            cudaEventCreate(&propagateLowerBBoxEvents[deviceId]);
            cudaEventRecord(propagateLowerBBoxEvents[deviceId], stream);
        });

        // Wait until bounding boxes are computed for each upper node and then compute the root bounding box on the zeroth device
        {
            int deviceId = 0;
            auto stream = mDeviceMesh[deviceId].stream;
            cudaCheck(cudaSetDevice(deviceId));
            for (const auto& propagateLowerBBoxEvent : propagateLowerBBoxEvents)
            {
                cudaStreamWaitEvent(stream, propagateLowerBBoxEvent);
            }
            // propagate bbox from upper -> root/parent node
            util::cuda::lambdaKernel<<<numBlocks(mData->nodeCount[2]), mNumThreads, 0, stream>>>(mData->nodeCount[2], PropagateUpperBBoxFunctor<BuildT>(), mData);
            cudaCheckError();

            // update the world-bbox in the root node
            util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, UpdateRootWorldBBoxFunctor<BuildT>(), mData);
            cudaCheckError();

            cudaCheck(cudaEventDestroy(propagateLowerBBoxEvents[deviceId]));
        }
    }

    // Explicitly synchronize so that move constructor in getHandle doesn't fail
    parallelForEach(mDeviceMesh, [&](int deviceId, cudaStream_t stream) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaStreamSynchronize(stream);
    });

    if constexpr(BuildTraits<BuildT>::is_onindex) {
        cudaCheck(cudaFree(mValueIndexPrefix));
        cudaCheck(cudaFree(mValueIndex));
    }

    cudaCheck(cudaFree(mPointsPerTile));
    cudaCheck(cudaFree(mIndices));
    cudaCheck(cudaFree(mKeys));

    cudaCheck(cudaFree(mData->pointsPerLeafPrefix));
    cudaCheck(cudaFree(mData->pointsPerLeaf));

    cudaCheck(cudaFree(mData->pointsPerVoxelPrefix));
    cudaCheck(cudaFree(mData->pointsPerVoxel));

    cudaCheck(cudaFree(mData->d_indx));
    cudaCheck(cudaFree(mData->d_leaf_keys));
    cudaCheck(cudaFree(mData->d_lower_keys));
    cudaCheck(cudaFree(mData->d_tile_keys));
    cudaCheck(cudaFree(mData->d_keys));
}// DistributedPointsToGrid<BuildT>::processBBox

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_DISTRIBUTEDPOINTSTOGRID_CUH_HAS_BEEN_INCLUDED
