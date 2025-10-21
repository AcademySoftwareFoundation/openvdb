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
void mergePath(KeyIteratorIn keys1, size_t keys1Count, KeyIteratorIn keys2, size_t keys2Count, ptrdiff_t* key1Intervals, ptrdiff_t* key2Intervals, int intervalIndex)
{
    using key_type = typename ::cuda::std::iterator_traits<KeyIteratorIn>::value_type;

    const size_t combinedIndex = intervalIndex * (keys1Count + keys2Count) / 2;
    size_t leftTop = combinedIndex > keys1Count ? keys1Count : combinedIndex;
    size_t rightTop = combinedIndex > keys1Count ? combinedIndex - keys1Count : 0;
    size_t leftBottom = rightTop;

    key_type leftKey;
    key_type rightKey;
    while(true)
    {
        ptrdiff_t offset = (leftTop - leftBottom) / 2;
        ptrdiff_t leftMid = leftTop - offset;
        ptrdiff_t rightMid = rightTop + offset;

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
void mergePathKernel(KeyIteratorIn keys1, size_t keys1Count, KeyIteratorIn keys2, size_t keys2Count, ptrdiff_t* key1Intervals, ptrdiff_t* key2Intervals, size_t intervalOffset)
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

template<typename KeyT, typename ValueT, typename NumItemsT, typename OffsetT, typename CountT>
void radixSortAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, KeyT* keysIn, KeyT* keysOut, ValueT* valuesIn, ValueT* valuesOut, NumItemsT numItems, OffsetT* mergeIntervals, const OffsetT* offsets, const CountT* counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    // Radix sort the subset of keys assigned to each device in parallel
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));

        const KeyT* deviceKeysIn = keysIn + offsets[deviceId];
        const ValueT* deviceValuesIn = valuesIn + offsets[deviceId];
        KeyT* deviceKeysOut = keysOut + offsets[deviceId];
        ValueT* deviceValuesOut = valuesOut + offsets[deviceId];

        cudaCheck(util::cuda::memPrefetchAsync(deviceKeysIn, counts[deviceId] * sizeof(KeyT), deviceId, stream));

        // TODO: Add begin and end bit support
        CUB_LAUNCH(DeviceRadixSort::SortPairs, pools[deviceId], stream, deviceKeysIn, deviceKeysOut, deviceValuesIn, deviceValuesOut, counts[deviceId], 0, sizeof(KeyT) * 8);
        cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
    }

    // TODO: Generalize to numbers of GPUs that aren't powers of two
    // For each pair of devices, merge the local sorts by first computing the median across the two devices followed by merging
    // the elements less than and greater than/equal to the median onto the first and second device of the pair respectively.
    // This avoids the allocating memory for and gathering the values from both devices onto a single device.
    const int log2DeviceCount = log2(deviceMesh.deviceCount());
    OffsetT* leftIntervals = mergeIntervals;
    OffsetT* rightIntervals = mergeIntervals + deviceMesh.deviceCount();
    for (int deviceExponent = 0; deviceExponent < log2DeviceCount; ++deviceExponent) {
        std::swap(keysIn, keysOut);
        std::swap(valuesIn, valuesOut);
        const int deviceIncrement = 1 << deviceExponent;

        std::vector<std::thread> threads;
        for (int leftDeviceId = 0; leftDeviceId < static_cast<int>(deviceMesh.deviceCount()); leftDeviceId += 2 * deviceIncrement) {
            threads.emplace_back([&, leftDeviceId]() {
                const int rightDeviceId = leftDeviceId + deviceIncrement;

                CountT leftDeviceItemCount = 0;
                for (int deviceId = leftDeviceId; deviceId < rightDeviceId; ++deviceId)
                    leftDeviceItemCount += counts[deviceId];

                CountT rightDeviceItemCount = 0;
                for (int deviceId = rightDeviceId; deviceId < rightDeviceId + deviceIncrement; ++deviceId)
                    rightDeviceItemCount += counts[deviceId];

                const KeyT* leftDeviceKeysIn = keysIn + offsets[leftDeviceId];
                const ValueT* leftDeviceValuesIn = valuesIn + offsets[leftDeviceId];
                const KeyT* rightDeviceKeysIn = keysIn + offsets[leftDeviceId] + leftDeviceItemCount;
                const ValueT* rightDeviceValuesIn = valuesIn + offsets[leftDeviceId] + leftDeviceItemCount;

                // Wait on the prior sort to finish on both devices before computing the median across both devices
                auto mergePathSubfunc = [&](int deviceId, int otherDeviceId, int intervalIndex) {
                    cudaCheck(cudaSetDevice(deviceId));

                    cudaCheck(cudaStreamWaitEvent(deviceMesh[deviceId].stream, postEvents[otherDeviceId]));
                    kernels::mergePathKernel<<<1, 1, 0, deviceMesh[deviceId].stream>>>(leftDeviceKeysIn, leftDeviceItemCount, rightDeviceKeysIn, rightDeviceItemCount, leftIntervals + deviceId, rightIntervals + deviceId, intervalIndex);
                    cudaCheck(cudaEventRecord(postEvents[deviceId], deviceMesh[deviceId].stream));
                };
                mergePathSubfunc(leftDeviceId, rightDeviceId, 0);
                mergePathSubfunc(rightDeviceId, leftDeviceId, 1);

                cudaCheck(cudaEventSynchronize(postEvents[leftDeviceId]));
                cudaCheck(cudaEventSynchronize(postEvents[rightDeviceId]));

                // Merge the pairs less than the median to the left device
                {
                    cudaCheck(cudaSetDevice(leftDeviceId));

                    const KeyT* leftKeysIn = leftDeviceKeysIn + leftIntervals[leftDeviceId];
                    const ValueT* leftValuesIn = leftDeviceValuesIn + leftIntervals[leftDeviceId];
                    CountT leftCount = leftIntervals[rightDeviceId] - leftIntervals[leftDeviceId];

                    const KeyT* rightKeysIn = rightDeviceKeysIn + rightIntervals[leftDeviceId];
                    const ValueT* rightValuesIn = rightDeviceValuesIn + rightIntervals[leftDeviceId];
                    CountT rightCount = rightIntervals[rightDeviceId] - rightIntervals[leftDeviceId];

                    OffsetT outputOffset = offsets[leftDeviceId] + leftIntervals[leftDeviceId] + rightIntervals[leftDeviceId];

                    CUB_LAUNCH(DeviceMerge::MergePairs, pools[leftDeviceId], deviceMesh[leftDeviceId].stream, leftKeysIn, leftValuesIn, leftCount, rightKeysIn, rightValuesIn, rightCount, keysOut + outputOffset, valuesOut + outputOffset, {});
                    cudaCheck(cudaEventRecord(postEvents[leftDeviceId], deviceMesh[leftDeviceId].stream));
                };

                // Merge the pairs greater than/equal to the median to the right device
                {
                    cudaCheck(cudaSetDevice(rightDeviceId));

                    const KeyT* leftKeysIn = leftDeviceKeysIn + leftIntervals[rightDeviceId];
                    const ValueT* leftValuesIn = leftDeviceValuesIn + leftIntervals[rightDeviceId];
                    CountT leftCount = leftDeviceItemCount - leftIntervals[rightDeviceId];

                    const KeyT* rightKeysIn = rightDeviceKeysIn + rightIntervals[rightDeviceId];
                    const ValueT* rightValuesIn = rightDeviceValuesIn + rightIntervals[rightDeviceId];
                    CountT rightCount = rightDeviceItemCount - rightIntervals[rightDeviceId];

                    OffsetT outputOffset = offsets[leftDeviceId] + leftIntervals[rightDeviceId] + rightIntervals[rightDeviceId];

                    CUB_LAUNCH(DeviceMerge::MergePairs, pools[rightDeviceId], deviceMesh[rightDeviceId].stream, leftKeysIn, leftValuesIn, leftCount, rightKeysIn, rightValuesIn, rightCount, keysOut + outputOffset, valuesOut + outputOffset, {});
                    cudaCheck(cudaEventRecord(postEvents[rightDeviceId], deviceMesh[rightDeviceId].stream));
                };

                cudaCheck(cudaEventSynchronize(postEvents[leftDeviceId]));
                cudaCheck(cudaEventSynchronize(postEvents[rightDeviceId]));
            });
        }
        std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
    }

    // There is no merging required for a single device so we simply copy the sorted result to the destination array (where the sort would have been merged to).
    if (log2DeviceCount % 2) {
        std::swap(keysIn, keysOut);
        std::swap(valuesIn, valuesOut);
        for (const auto& [deviceId, stream] : deviceMesh) {
            cudaCheck(cudaSetDevice(deviceId));

            cudaMemcpyAsync(keysOut + offsets[deviceId], keysIn + offsets[deviceId], counts[deviceId] * sizeof(KeyT), cudaMemcpyDefault, stream);
            cudaMemcpyAsync(valuesOut + offsets[deviceId], valuesIn + offsets[deviceId], counts[deviceId] * sizeof(ValueT), cudaMemcpyDefault, stream);

            cudaEventRecord(postEvents[deviceId], stream);
        }
    }
}

template<typename KeyT, typename ValueT, typename NumItemsT, typename OffsetT, typename CountT>
void radixSortAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, KeyT* keysIn, KeyT* keysOut, ValueT* valuesIn, ValueT* valuesOut, NumItemsT numItems, const OffsetT* offsets, const CountT* counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    ptrdiff_t* mergeIntervals = nullptr;
    cudaCheck(cudaMallocManaged(&mergeIntervals, 2 * deviceMesh.deviceCount() * sizeof(ptrdiff_t)));
    radixSortAsync(deviceMesh, pools, keysIn, keysOut, valuesIn, valuesOut, numItems, mergeIntervals, offsets, counts, preEvents, postEvents);
    cudaCheck(cudaFree(mergeIntervals));
}

/// @brief Launches an async exclusive sum operation across multiple devices. The operator waits on the per-device preEvents[deviceId] before summing over that device's contributions and records postEvents[deviceId] when the device's contribution is summed.
template<typename InputIteratorT, typename OutputIteratorT, typename CountIteratorT, int NumThreads = 128>
void exclusiveSumAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, InputIteratorT in, OutputIteratorT out, CountIteratorT counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    InputIteratorT deviceIn = in;
    OutputIteratorT deviceOut = out;
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        // Required for the host to pass the correct value of counts[deviceId]
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));
        uint32_t deviceNumItems = counts[deviceId];
        CUB_LAUNCH(DeviceScan::ExclusiveSum, pools[deviceId], stream, deviceIn, deviceOut, deviceNumItems);
        cudaCheck(cudaEventRecord(preEvents[deviceId], stream));
        deviceIn += counts[deviceId];
        deviceOut += counts[deviceId];
    }

    deviceIn = in;
    deviceOut = out;
    auto partialExclusiveSum = 0;
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        // Required for the host to read-back the per-segment inclusive sum
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));
        if (counts[deviceId]) {
            auto segmentExclusiveSum = deviceOut[counts[deviceId] - 1] + deviceIn[counts[deviceId] - 1];

            unsigned int numBlocks = (counts[deviceId] + NumThreads - 1) / NumThreads;
            util::cuda::lambdaKernel<<<numBlocks, NumThreads, 0, stream>>>(counts[deviceId], [=] __device__ (size_t tid) { deviceOut[tid] += partialExclusiveSum; });
            cudaCheckError();

            partialExclusiveSum += segmentExclusiveSum;
        }
        cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
        deviceIn += counts[deviceId];
        deviceOut += counts[deviceId];
    }
}

/// @brief Launches an async inclusive sum operation across multiple devices. The operator waits on the per-device preEvents[deviceId] before summing over that device's contributions and records postEvents[deviceId] when the device's contribution is summed.
template<typename InputIteratorT, typename OutputIteratorT, typename CountIteratorT, int NumThreads = 128>
void inclusiveSumAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, InputIteratorT in, OutputIteratorT out, CountIteratorT counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    InputIteratorT deviceIn = in;
    OutputIteratorT deviceOut = out;
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        // Required for the host to pass the correct value of counts[deviceId]
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));
        uint32_t deviceNumItems = counts[deviceId];
        CUB_LAUNCH(DeviceScan::InclusiveSum, pools[deviceId], stream, deviceIn, deviceOut, deviceNumItems);
        cudaCheck(cudaEventRecord(preEvents[deviceId], stream));
        deviceIn += counts[deviceId];
        deviceOut += counts[deviceId];
    }

    deviceIn = in;
    deviceOut = out;
    auto partialInclusiveSum = 0;
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        // Required for the host to read-back the per-segment inclusive sum
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));
        if (counts[deviceId]) {
            auto segmentInclusiveSum = deviceOut[counts[deviceId] - 1];

            unsigned int numBlocks = (counts[deviceId] + NumThreads - 1) / NumThreads;
            util::cuda::lambdaKernel<<<numBlocks, NumThreads, 0, stream>>>(counts[deviceId], [=] __device__ (size_t tid) { deviceOut[tid] += partialInclusiveSum; });
            cudaCheckError();

            partialInclusiveSum += segmentInclusiveSum;
        }
        cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
        deviceIn += counts[deviceId];
        deviceOut += counts[deviceId];
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
    ptrdiff_t* mIntervals;

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
    mIntervals = nullptr;
    cudaCheck(cudaMallocManaged(&mIntervals, 2 * mDeviceMesh.deviceCount() * sizeof(ptrdiff_t)));
}

template <typename BuildT>
DistributedPointsToGrid<BuildT>::DistributedPointsToGrid(const nanovdb::cuda::DeviceMesh& deviceMesh, const double scale, const Vec3d &trans)
    : DistributedPointsToGrid(deviceMesh, Map(scale, trans))
{
}

template <typename BuildT>
DistributedPointsToGrid<BuildT>::~DistributedPointsToGrid()
{
    cudaCheck(cudaFree(mIntervals));
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
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];

        util::cuda::memPrefetchAsync(coords, coordCount * sizeof(nanovdb::Coord), deviceId, stream);

        util::cuda::offsetLambdaKernel<<<numBlocks(deviceStripeCount), mNumThreads, 0, stream>>>(deviceStripeCount, deviceStripeOffset, TileKeyFunctor<BuildT, PtrT>(), mData, coords, mData->d_keys, mData->d_indx);
    }

    radixSortAsync(mDeviceMesh, mTempDevicePools, mData->d_keys, mKeys, mData->d_indx, mIndices, coordCount, mIntervals, mStripeOffsets, mStripeCounts, sortEvents.data(), sortEvents.data());

    // For each segment of sorted keys on each device, we count how many of the leftmost key occur past the left boundary of the segment. The same is done for the rightmost key with the right boundary of the segment.
    auto leftIntervals = mIntervals;
    auto rightIntervals = mIntervals + mDeviceMesh.deviceCount();
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];
        uint64_t* deviceInputKeys = mKeys + deviceStripeOffset;

        if (deviceId > 0) {
            cudaStreamWaitEvent(stream, sortEvents[deviceId - 1]);
            EqualityIndicator<uint64_t> indicator(deviceInputKeys - 1);
            CUB_LAUNCH(DeviceReduce::TransformReduce, mTempDevicePools[deviceId], stream, deviceInputKeys, rightIntervals + deviceId, deviceStripeCount, ::cuda::std::plus(), indicator, 0);
        }
        else {
            rightIntervals[deviceId] = 0;
        }

        if (deviceId < static_cast<int>(mDeviceMesh.deviceCount() - 1)) {
            cudaStreamWaitEvent(stream, sortEvents[deviceId + 1]);
            EqualityIndicator<uint64_t> indicator(deviceInputKeys + deviceStripeCount);
            CUB_LAUNCH(DeviceReduce::TransformReduce, mTempDevicePools[deviceId], stream, deviceInputKeys, leftIntervals + deviceId, deviceStripeCount, ::cuda::std::plus(), indicator, 0);
        }
        else {
            leftIntervals[deviceId] = 0;
        }
        cudaEventRecord(transformReduceEvents[deviceId], stream);
    }

    // Rebalance the segments so that a device segment boundary also corresponds to a change in key value. Effectively, this aligns upper node boundaries with device ownership boundaries.
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        if (deviceId > 0)
        {
            cudaStreamWaitEvent(stream, transformReduceEvents[deviceId - 1]);
            kernels::rightRebalanceKernel<<<1, 1, 0, stream>>>(leftIntervals + deviceId - 1, rightIntervals + deviceId, mStripeCounts + deviceId, mStripeOffsets + deviceId);
        }

        if (deviceId < static_cast<int>(mDeviceMesh.deviceCount() - 1))
        {
            cudaStreamWaitEvent(stream, transformReduceEvents[deviceId + 1]);
            kernels::leftRebalanceKernel<<<1, 1, 0, stream>>>(leftIntervals + deviceId, rightIntervals + deviceId + 1, mStripeCounts + deviceId, mStripeOffsets + deviceId);
        }
        cudaEventRecord(rebalanceEvents[deviceId], stream);
    }

    // Parallel RLE in order to obtain tiles
    for (const auto& [deviceId, stream] : mDeviceMesh) {
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
    }

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

            util::cuda::offsetLambdaKernel<<<numBlocks(devicePointsPerTile[i]), mNumThreads, 0, stream>>>(devicePointsPerTile[i], tileOffset + mStripeOffsets[deviceId], VoxelKeyFunctor<BuildT, PtrT>(), mData, coords, id, mKeys, mIndices);

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
    // Without this pipelining, each operation would have to wait until ALL devices to finish their prior operation instead of just the previous device which significantly degrades scaling.
    // Based on profiling, we launch the per-device kernels for steps 1 and 2 in a single loop.
    {
        uint32_t* devicePointsPerVoxel = mData->pointsPerVoxel;
        uint32_t* devicePointsPerLeaf = mData->pointsPerLeaf;
        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaCheck(cudaSetDevice(deviceId));

            uint64_t* deviceInputKeys = mKeys + mStripeOffsets[deviceId];
            const uint64_t* deviceOutputKeys = mData->d_keys + mStripeOffsets[deviceId];

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
        }
    }

    exclusiveSumAsync(mDeviceMesh, mTempDevicePools, mData->pointsPerVoxel, mData->pointsPerVoxelPrefix, mVoxelCounts, voxelCountEvents.data(), voxelPrefixSumEvents.data());
    LeafCountIterator leafCountIterator(mNodeCounts);
    exclusiveSumAsync(mDeviceMesh, mTempDevicePools, mData->pointsPerLeaf, mData->pointsPerLeafPrefix, leafCountIterator, leafCountEvents.data(), leafPrefixSumEvents.data());

    uint32_t leafOffset = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        uint64_t* deviceKeys = mKeys + mStripeOffsets[deviceId];
        cudaCheck(cudaMemcpyAsync(mData->d_leaf_keys + leafOffset, deviceKeys, sizeof(uint64_t) * deviceNodeCount(deviceId)[0], cudaMemcpyDefault, stream));
        deviceNodeOffset(deviceId)[0] = leafOffset;
        leafOffset += deviceNodeCount(deviceId)[0];
    }

    // Parallel RLE with (shifted) keys in order to count leaves and points per leaf
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        uint64_t* deviceInputKeys = mKeys + mStripeOffsets[deviceId];
        uint64_t* deviceOutputKeys = mData->d_keys + mStripeOffsets[deviceId];

        CUB_LAUNCH(DeviceSelect::Unique, mTempDevicePools[deviceId], stream, thrust::make_transform_iterator(deviceOutputKeys, ShiftRight<21>()), deviceInputKeys, deviceNodeCount(deviceId) + 1, mStripeCounts[deviceId]);
        cudaEventRecord(lowerCountEvents[deviceId], stream);
    }

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

    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaStreamSynchronize(stream));
    }

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

    for (const auto& [otherDeviceId, otherStream] : mDeviceMesh) {
        cudaSetDevice(otherDeviceId);
        cudaStreamWaitEvent(otherStream, processGridTreeRootEvent);
    }

    cudaCheck(cudaSetDevice(deviceId));
    cudaEventDestroy(processGridTreeRootEvent);
}// DistributedPointsToGrid<BuildT>::processGridTreeRoot

template <typename BuildT>
inline void DistributedPointsToGrid<BuildT>::processNodes()
{
    // Parallel construction of upper, lower, and leaf nodes
    const uint8_t flags = (uint8_t) GridFlags::HasBBox;

    for (const auto& [deviceId, stream] : mDeviceMesh) {
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
    }

    if constexpr(BuildTraits<BuildT>::is_onindex) {
        std::vector<cudaEvent_t> leafCountEvents(mDeviceMesh.deviceCount());
        std::vector<cudaEvent_t> valueIndexPrefixSumEvents(mDeviceMesh.deviceCount());

        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaSetDevice(deviceId);
            cudaEventCreateWithFlags(&leafCountEvents[deviceId], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&valueIndexPrefixSumEvents[deviceId], cudaEventDisableTiming);

            if (deviceNodeCount(deviceId)[0]) {
                kernels::fillValueIndexKernel<BuildT><<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], mValueIndex, mData);
                cudaCheckError();
            }
        }

        LeafCountIterator leafCountIterator(mNodeCounts);
        inclusiveSumAsync(mDeviceMesh, mTempDevicePools, mValueIndex, mValueIndexPrefix, leafCountIterator, leafCountEvents.data(), valueIndexPrefixSumEvents.data());

        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaSetDevice(deviceId);
            cudaStreamWaitEvent(stream, valueIndexPrefixSumEvents.back());
            if (deviceNodeCount(deviceId)[0]) {
                kernels::leafPrefixSumKernel<BuildT><<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], mValueIndexPrefix, mData);
                cudaCheckError();
            }
        }

        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaSetDevice(deviceId);
            cudaEventDestroy(valueIndexPrefixSumEvents[deviceId]);
            cudaEventDestroy(leafCountEvents[deviceId]);
        }
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
    // Compute and propagate bounding boxes for the upper nodes and their descendents belonging to each device in parallel.
    std::vector<cudaEvent_t> propagateLowerBBoxEvents(mDeviceMesh.deviceCount());
    for (const auto& [deviceId, stream] : mDeviceMesh) {
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
    }

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

    // Explicitly synchronize so that move constructor in getHandle doesn't fail
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaStreamSynchronize(stream);
    }

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
