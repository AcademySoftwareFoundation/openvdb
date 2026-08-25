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
#include <algorithm>

#include <cuda/cmath>

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

/// @brief Find a partition at an arbitrary diagonal in the conceptual merge of two sorted input arrays.
template<typename KeyIteratorIn>
__device__
void mergePath(KeyIteratorIn keys1, size_t keys1Count, KeyIteratorIn keys2, size_t keys2Count, ptrdiff_t* key1Intervals, ptrdiff_t* key2Intervals, size_t combinedIndex)
{
    size_t begin = combinedIndex > keys2Count ? combinedIndex - keys2Count : 0;
    size_t end = combinedIndex < keys1Count ? combinedIndex : keys1Count;
    while (begin < end) {
        const size_t key1Index = (begin + end) / 2;
        const size_t key2Index = combinedIndex - 1 - key1Index;
        if (!(keys2[key2Index] < keys1[key1Index])) {
            begin = key1Index + 1;
        } else {
            end = key1Index;
        }
    }

    *key1Intervals = begin;
    *key2Intervals = combinedIndex - begin;
}

namespace kernels {

/// @brief Kernel wrapper for the merge path algorithm.
template<typename KeyIteratorIn>
__global__
void mergePathKernel(KeyIteratorIn keys1, size_t keys1Count, KeyIteratorIn keys2, size_t keys2Count, ptrdiff_t* key1Intervals, ptrdiff_t* key2Intervals, size_t combinedIndex)
{
    mergePath(keys1, keys1Count, keys2, keys2Count, key1Intervals, key2Intervals, combinedIndex);
}

/// @brief Snap each device boundary in stripeOffsets to the nearest edge of the run of equal
/// keys containing it, so that no run (i.e. upper-node tile) straddles two devices. The keys
/// must be globally sorted. The boundaries are adjusted monotonically from left to right, so a
/// run spanning several stripes is consolidated onto one device and the fully-interior devices
/// are left with empty stripes. Runs sequentially on a single thread: the boundary chain is a
/// sequential dependence over deviceCount entries and each run extent is found by binary search.
template<typename KeyT>
__global__
void snapBoundariesToRunsKernel(const KeyT* keys, ptrdiff_t keyCount, int deviceCount, ptrdiff_t* stripeOffsets, size_t* stripeCounts)
{
    ptrdiff_t previousBoundary = stripeOffsets[0]; // device 0 always starts at 0
    for (int deviceId = 1; deviceId < deviceCount; ++deviceId) {
        ptrdiff_t boundary = stripeOffsets[deviceId];
        if (boundary >= keyCount) {
            boundary = keyCount;
        } else if (boundary > previousBoundary && keys[boundary] == keys[boundary - 1]) {
            // The even-split boundary falls inside a run; binary search for the run's extent
            // and snap to whichever end keeps the boundary closest to the even split without
            // crossing the previous boundary.
            const KeyT key = keys[boundary];
            ptrdiff_t lo = previousBoundary, hi = boundary;
            while (lo < hi) {
                const ptrdiff_t mid = lo + (hi - lo) / 2;
                if (keys[mid] < key) lo = mid + 1; else hi = mid;
            }
            const ptrdiff_t runStart = lo;
            lo = boundary; hi = keyCount;
            while (lo < hi) {
                const ptrdiff_t mid = lo + (hi - lo) / 2;
                if (keys[mid] <= key) lo = mid + 1; else hi = mid;
            }
            const ptrdiff_t runEnd = lo;
            if (runStart <= previousBoundary) {
                boundary = runEnd; // the run reaches the previous device, give the whole tile away
            } else {
                boundary = (boundary - runStart <= runEnd - boundary) ? runStart : runEnd;
            }
        }
        if (boundary < previousBoundary) boundary = previousBoundary;
        stripeOffsets[deviceId] = boundary;
        previousBoundary = boundary;
    }

    // Recompute the per-device counts from the adjusted, monotonic offsets.
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        const ptrdiff_t nextOffset = (deviceId + 1 < deviceCount) ? stripeOffsets[deviceId + 1] : keyCount;
        stripeCounts[deviceId] = static_cast<size_t>(nextOffset - stripeOffsets[deviceId]);
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

/// @brief Make every device stream wait for work submitted to all device streams.
inline void mergeStreams(const nanovdb::cuda::DeviceMesh& deviceMesh, cudaEvent_t* events)
{
    static constexpr int mergeDeviceId = 0;

    // Record an event for each device in its respective stream.
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaEventRecord(events[deviceId], stream));
    }

    // Fan in the per-device events on the merge stream.
    cudaCheck(cudaSetDevice(mergeDeviceId));
    const auto mergeStream = deviceMesh[mergeDeviceId].stream;
    for (const auto& deviceNode : deviceMesh) {
        cudaCheck(cudaStreamWaitEvent(mergeStream, events[deviceNode.id]));
    }
    cudaCheck(cudaEventRecord(events[mergeDeviceId], mergeStream));

    // Fan the merged dependency back out to every device stream.
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaStreamWaitEvent(stream, events[mergeDeviceId]));
    }
}

template<typename KeyT, typename ValueT, typename NumItemsT, typename OffsetT, typename CountT>
void radixSortAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, KeyT* keysIn, KeyT* keysOut, ValueT* valuesIn, ValueT* valuesOut, NumItemsT numItems, OffsetT* mergeIntervals, const OffsetT* offsets, const CountT* counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    if (!numItems) return;

    const int deviceCount = static_cast<int>(deviceMesh.deviceCount());
    OffsetT* leftIntervals = mergeIntervals;
    OffsetT* rightIntervals = mergeIntervals + deviceCount + 1;
    const auto offset = [&](int deviceIndex) {
        return deviceIndex == deviceCount ? static_cast<OffsetT>(numItems) : offsets[deviceIndex];
    };

    // Radix sort the subset of key/value pairs assigned to each GPU.
    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaEventSynchronize(preEvents[deviceId]));

        if (!counts[deviceId]) {
            cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
            continue;
        }

        const KeyT* deviceKeysIn = keysIn + offsets[deviceId];
        const ValueT* deviceValuesIn = valuesIn + offsets[deviceId];
        KeyT* deviceKeysOut = keysOut + offsets[deviceId];
        ValueT* deviceValuesOut = valuesOut + offsets[deviceId];

        cudaCheck(util::cuda::memPrefetchAsync(deviceKeysIn, counts[deviceId] * sizeof(KeyT), deviceId, stream));
        cudaCheck(util::cuda::memPrefetchAsync(deviceValuesIn, counts[deviceId] * sizeof(ValueT), deviceId, stream));
        cudaCheck(util::cuda::memPrefetchAsync(deviceKeysOut, counts[deviceId] * sizeof(KeyT), deviceId, stream));
        cudaCheck(util::cuda::memPrefetchAsync(deviceValuesOut, counts[deviceId] * sizeof(ValueT), deviceId, stream));

        // TODO: Add begin and end bit support
        CUB_LAUNCH(DeviceRadixSort::SortPairs, pools[deviceId], stream, deviceKeysIn, deviceKeysOut, deviceValuesIn, deviceValuesOut, counts[deviceId], 0, sizeof(KeyT) * 8);
        cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
    }
    mergeStreams(deviceMesh, postEvents);

    KeyT* currentKeys = keysOut;
    KeyT* nextKeys = keysIn;
    ValueT* currentValues = valuesOut;
    ValueT* nextValues = valuesIn;

    // At each level, split every pair of sorted runs at all of the group's output-shard boundaries.
    // Every GPU in the group merges one balanced output shard, including at the upper merge-tree levels.
    for (int runDeviceCount = 1; runDeviceCount < deviceCount; runDeviceCount *= 2) {
        for (int groupStart = 0; groupStart < deviceCount; groupStart += 2 * runDeviceCount) {
            const int groupMiddle = std::min(groupStart + runDeviceCount, deviceCount);
            const int groupEnd = std::min(groupStart + 2 * runDeviceCount, deviceCount);
            if (groupMiddle == groupEnd) continue;

            const CountT leftCount = offset(groupMiddle) - offset(groupStart);
            const CountT rightCount = offset(groupEnd) - offset(groupMiddle);
            const KeyT* leftKeys = currentKeys + offset(groupStart);
            const KeyT* rightKeys = currentKeys + offset(groupMiddle);

            for (int boundaryDevice = groupStart + 1; boundaryDevice < groupEnd; ++boundaryDevice) {
                cudaCheck(cudaSetDevice(boundaryDevice));
                const auto stream = deviceMesh[boundaryDevice].stream;
                const CountT diagonal = offset(boundaryDevice) - offset(groupStart);
                kernels::mergePathKernel<<<1, 1, 0, stream>>>(leftKeys, leftCount, rightKeys, rightCount, leftIntervals + boundaryDevice, rightIntervals + boundaryDevice, diagonal);
                cudaCheckError();
                cudaCheck(cudaEventRecord(postEvents[boundaryDevice], stream));
            }
        }

        // CUB needs the partition sizes on the host. Synchronize only the tiny merge-path kernels.
        for (int groupStart = 0; groupStart < deviceCount; groupStart += 2 * runDeviceCount) {
            const int groupMiddle = std::min(groupStart + runDeviceCount, deviceCount);
            const int groupEnd = std::min(groupStart + 2 * runDeviceCount, deviceCount);
            if (groupMiddle == groupEnd) continue;

            for (int boundaryDevice = groupStart + 1; boundaryDevice < groupEnd; ++boundaryDevice) {
                cudaCheck(cudaEventSynchronize(postEvents[boundaryDevice]));
            }
        }

        for (int groupStart = 0; groupStart < deviceCount; groupStart += 2 * runDeviceCount) {
            const int groupMiddle = std::min(groupStart + runDeviceCount, deviceCount);
            const int groupEnd = std::min(groupStart + 2 * runDeviceCount, deviceCount);

            // A trailing run without a merge partner must still be copied because the
            // current and next buffers are swapped after every merge-tree level.
            if (groupMiddle == groupEnd) {
                for (int outputDevice = groupStart; outputDevice < groupEnd; ++outputDevice) {
                    cudaCheck(cudaSetDevice(outputDevice));
                    const auto stream = deviceMesh[outputDevice].stream;
                    if (counts[outputDevice]) {
                        cudaCheck(cudaMemcpyAsync(nextKeys + offset(outputDevice), currentKeys + offset(outputDevice), counts[outputDevice] * sizeof(KeyT), cudaMemcpyDefault, stream));
                        cudaCheck(cudaMemcpyAsync(nextValues + offset(outputDevice), currentValues + offset(outputDevice), counts[outputDevice] * sizeof(ValueT), cudaMemcpyDefault, stream));
                    }
                    cudaCheck(cudaEventRecord(postEvents[outputDevice], stream));
                }
                continue;
            }

            const CountT leftCount = offset(groupMiddle) - offset(groupStart);
            const CountT rightCount = offset(groupEnd) - offset(groupMiddle);
            const KeyT* leftKeys = currentKeys + offset(groupStart);
            const KeyT* rightKeys = currentKeys + offset(groupMiddle);
            const ValueT* leftValues = currentValues + offset(groupStart);
            const ValueT* rightValues = currentValues + offset(groupMiddle);

            for (int outputDevice = groupStart; outputDevice < groupEnd; ++outputDevice) {
                cudaCheck(cudaSetDevice(outputDevice));
                const auto stream = deviceMesh[outputDevice].stream;
                const CountT leftBegin = outputDevice == groupStart ? 0 : leftIntervals[outputDevice];
                const CountT leftEnd = outputDevice + 1 == groupEnd ? leftCount : leftIntervals[outputDevice + 1];
                const CountT rightBegin = outputDevice == groupStart ? 0 : rightIntervals[outputDevice];
                const CountT rightEnd = outputDevice + 1 == groupEnd ? rightCount : rightIntervals[outputDevice + 1];
                const CountT leftSegmentCount = leftEnd - leftBegin;
                const CountT rightSegmentCount = rightEnd - rightBegin;
                const CountT outputCount = leftSegmentCount + rightSegmentCount;
                NANOVDB_ASSERT(outputCount == counts[outputDevice]);

                if (outputCount) {
                    CUB_LAUNCH(DeviceMerge::MergePairs, pools[outputDevice], stream,
                        leftKeys + leftBegin, leftValues + leftBegin, leftSegmentCount,
                        rightKeys + rightBegin, rightValues + rightBegin, rightSegmentCount,
                        nextKeys + offset(outputDevice), nextValues + offset(outputDevice), {});
                }
                cudaCheck(cudaEventRecord(postEvents[outputDevice], stream));
            }
        }

        mergeStreams(deviceMesh, postEvents);
        std::swap(currentKeys, nextKeys);
        std::swap(currentValues, nextValues);
    }

    // Keep the existing contract: callers always receive the final data in the output buffers.
    if (currentKeys != keysOut) {
        for (const auto& [deviceId, stream] : deviceMesh) {
            cudaCheck(cudaSetDevice(deviceId));
            if (counts[deviceId]) {
                cudaCheck(cudaMemcpyAsync(keysOut + offsets[deviceId], currentKeys + offsets[deviceId], counts[deviceId] * sizeof(KeyT), cudaMemcpyDefault, stream));
                cudaCheck(cudaMemcpyAsync(valuesOut + offsets[deviceId], currentValues + offsets[deviceId], counts[deviceId] * sizeof(ValueT), cudaMemcpyDefault, stream));
            }
            cudaCheck(cudaEventRecord(postEvents[deviceId], stream));
        }
    }
    mergeStreams(deviceMesh, postEvents);
}

template<typename KeyT, typename ValueT, typename NumItemsT, typename OffsetT, typename CountT>
void radixSortAsync(const nanovdb::cuda::DeviceMesh& deviceMesh, nanovdb::cuda::TempDevicePool* pools, KeyT* keysIn, KeyT* keysOut, ValueT* valuesIn, ValueT* valuesOut, NumItemsT numItems, const OffsetT* offsets, const CountT* counts, cudaEvent_t* preEvents, cudaEvent_t* postEvents)
{
    ptrdiff_t* mergeIntervals = nullptr;
    cudaCheck(cudaMallocHost(&mergeIntervals, 2 * (deviceMesh.deviceCount() + 1) * sizeof(ptrdiff_t)));
    radixSortAsync(deviceMesh, pools, keysIn, keysOut, valuesIn, valuesOut, numItems, mergeIntervals, offsets, counts, preEvents, postEvents);
    cudaCheck(cudaFreeHost(mergeIntervals));
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
        if (deviceNumItems) {
            CUB_LAUNCH(DeviceScan::ExclusiveSum, pools[deviceId], stream, deviceIn, deviceOut, deviceNumItems);
        }
        cudaCheck(cudaEventRecord(preEvents[deviceId], stream));
        deviceIn += deviceNumItems;
        deviceOut += deviceNumItems;
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

            unsigned int numBlocks = ::cuda::ceil_div<int>(counts[deviceId], NumThreads);
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
        if (deviceNumItems) {
            CUB_LAUNCH(DeviceScan::InclusiveSum, pools[deviceId], stream, deviceIn, deviceOut, deviceNumItems);
        }
        cudaCheck(cudaEventRecord(preEvents[deviceId], stream));
        deviceIn += deviceNumItems;
        deviceOut += deviceNumItems;
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

            unsigned int numBlocks = ::cuda::ceil_div<int>(counts[deviceId], NumThreads);
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
    static unsigned int numBlocks(unsigned int n) {return ::cuda::ceil_div(n, mNumThreads);}

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
    cudaCheck(cudaMallocHost(&mIntervals, 2 * (mDeviceMesh.deviceCount() + 1) * sizeof(ptrdiff_t)));
}

template <typename BuildT>
DistributedPointsToGrid<BuildT>::DistributedPointsToGrid(const nanovdb::cuda::DeviceMesh& deviceMesh, const double scale, const Vec3d &trans)
    : DistributedPointsToGrid(deviceMesh, Map(scale, trans))
{
}

template <typename BuildT>
DistributedPointsToGrid<BuildT>::~DistributedPointsToGrid()
{
    cudaCheck(cudaFreeHost(mIntervals));
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
        cudaEventCreateWithFlags(&tilePrefixSumEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&voxelCountEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&leafCountEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&lowerCountEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&voxelPrefixSumEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&leafPrefixSumEvents[deviceId], cudaEventDisableTiming);
    }

    // Advise per-coord quantities to be split evenly across devices. Clamp each stripe to
    // the input range so that inputs smaller than the device count produce valid trailing empty stripes.
    const size_t deviceStripeSize = ::cuda::ceil_div(coordCount, mDeviceMesh.deviceCount());
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        const ptrdiff_t deviceStripeOffset = std::min(deviceStripeSize * deviceId, coordCount);
        const size_t deviceStripeCount = std::min(deviceStripeSize, coordCount - deviceStripeOffset);

        mStripeCounts[deviceId] = deviceStripeCount;
        mStripeOffsets[deviceId] = deviceStripeOffset;

        if (deviceStripeCount) {
            uint64_t* deviceInputKeys = mKeys + deviceStripeOffset;
            uint32_t* deviceInputIndices = mIndices + deviceStripeOffset;
            uint64_t* deviceOutputKeys = mData->d_keys + deviceStripeOffset;
            uint32_t* deviceOutputIndices = mData->d_indx + deviceStripeOffset;

            util::cuda::memAdvise(deviceInputKeys, deviceStripeCount * sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, deviceId);
            util::cuda::memAdvise(deviceInputIndices, deviceStripeCount * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);
            util::cuda::memAdvise(deviceOutputKeys, deviceStripeCount * sizeof(uint64_t), cudaMemAdviseSetPreferredLocation, deviceId);
            util::cuda::memAdvise(deviceOutputIndices, deviceStripeCount * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);

            uint32_t* devicePointsPerTile = mPointsPerTile + deviceStripeOffset;
            util::cuda::memAdvise(devicePointsPerTile, deviceStripeCount * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);
        }
        util::cuda::memAdvise(deviceNodeCount(deviceId), 3 * sizeof(uint32_t), cudaMemAdviseSetPreferredLocation, deviceId);
    }

    // Radix sort the subset of keys assigned to each device in parallel
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];

        if (deviceStripeCount) {
            util::cuda::memPrefetchAsync(coords, coordCount * sizeof(nanovdb::Coord), deviceId, stream);
            nanovdb::util::cuda::offsetLambdaKernel<<<numBlocks(deviceStripeCount), mNumThreads, 0, stream>>>(deviceStripeCount, deviceStripeOffset, TileKeyFunctor<BuildT, PtrT>(), mData, coords, mData->d_keys, mData->d_indx);
            cudaCheckError();
        }
    }

    radixSortAsync(mDeviceMesh, mTempDevicePools, mData->d_keys, mKeys, mData->d_indx, mIndices, coordCount, mIntervals, mStripeOffsets, mStripeCounts, sortEvents.data(), sortEvents.data());

    // Rebalance the device segments so that a device boundary always coincides
    // with a change in key value. Because TileKeyFunctor assigns identical keys
    // to every point that falls in the same upper-node "tile", this aligns the
    // device ownership boundaries with tile boundaries. Downstream construction
    // assumes each tile (and therefore each lower node, leaf node, and voxel) is
    // owned by exactly one device; if a tile straddled a boundary, multiple
    // devices would concurrently build the same leaf and race on its value mask.
    //
    // A single tile can span three or more devices (e.g. one dense leaf whose
    // points are split evenly across the mesh). Adjusting only adjacent pairs of
    // boundaries cannot consolidate such a tile because a fully-interior device
    // lies entirely within it, so we compute the boundaries globally and
    // monotonically. mKeys is globally sorted at this point, so a tile boundary
    // is simply a position where mKeys changes. Snapping runs in a single-thread
    // kernel on one device (every device stream is already ordered after the
    // whole sort by radixSortAsync's final stream merge) with binary searches for
    // the run extents; the host only waits on its completion event before reading
    // back the small offset/count arrays, mirroring how the sort's merge-path
    // partitions are synchronized. Fully-interior devices are left with empty
    // stripes, which the rest of the pipeline already handles.
    {
        static constexpr int snapDeviceId = 0;
        cudaCheck(cudaSetDevice(snapDeviceId));
        const auto snapStream = mDeviceMesh[snapDeviceId].stream;
        kernels::snapBoundariesToRunsKernel<<<1, 1, 0, snapStream>>>(mKeys, static_cast<ptrdiff_t>(coordCount), static_cast<int>(mDeviceMesh.deviceCount()), mStripeOffsets, mStripeCounts);
        cudaCheckError();
        cudaCheck(cudaEventRecord(sortEvents[snapDeviceId], snapStream));
        // CUB needs the rebalanced partition sizes on the host. Synchronize only the tiny snap kernel.
        cudaCheck(cudaEventSynchronize(sortEvents[snapDeviceId]));
    }

    // Parallel RLE in order to obtain tiles. The device boundaries were finalized
    // above before the host read them back, so no per-device rebalance event is needed.
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];

        uint64_t* deviceInputKeys = mKeys + deviceStripeOffset;
        uint64_t* deviceOutputKeys = mData->d_keys + deviceStripeOffset;
        uint32_t* devicePointsPerTile = mPointsPerTile + deviceStripeOffset;

        // util::cuda::memPrefetchAsync(deviceInputKeys, deviceStripeCount * sizeof(uint64_t), deviceId, stream);

        if (deviceStripeCount) {
            CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, deviceInputKeys, deviceOutputKeys, devicePointsPerTile, deviceNodeCount(deviceId) + 2, deviceStripeCount);
        } else {
            cudaCheck(cudaMemsetAsync(deviceNodeCount(deviceId) + 2, 0, sizeof(uint32_t), stream));
        }
        cudaCheck(cudaEventRecord(runLengthEncodeEvents[deviceId], stream));
    }

    uint32_t upperOffset = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaEventSynchronize(runLengthEncodeEvents[deviceId]));
        auto deviceStripeOffset = mStripeOffsets[deviceId];
        uint64_t* deviceKeys = mData->d_keys + deviceStripeOffset;
        if (deviceNodeCount(deviceId)[2]) {
            cudaCheck(cudaMemcpyAsync(mData->d_tile_keys + upperOffset, deviceKeys, sizeof(uint64_t) * deviceNodeCount(deviceId)[2], cudaMemcpyDefault, stream));
        }
        deviceNodeOffset(deviceId)[2] = upperOffset;
        upperOffset += deviceNodeCount(deviceId)[2];
    }

    // For each tile in parallel, we construct another set of keys for the lower nodes, leaf nodes, and voxels within that tile followed by a radix sort of these keys.
    static constexpr uint32_t SEGMENTED_SORT_TILE_THRESHOLD = 32;
    for (int deviceId = 0; deviceId < static_cast<int>(mDeviceMesh.deviceCount()); ++deviceId) {
        auto stream = mDeviceMesh[deviceId].stream;
        cudaCheck(cudaSetDevice(deviceId));

        auto deviceStripeCount = mStripeCounts[deviceId];
        auto deviceStripeOffset = mStripeOffsets[deviceId];
        uint32_t* devicePointsPerTile = mPointsPerTile + deviceStripeOffset;
        uint32_t numDeviceTiles = deviceNodeCount(deviceId)[2];
        uint32_t tileIdOffset = deviceNodeOffset(deviceId)[2];

        if (numDeviceTiles >= SEGMENTED_SORT_TILE_THRESHOLD) {
            // Bulk segmented sort: one kernel launch + one segmented radix sort (faster for many tiles)
            uint32_t* d_tile_offsets = nullptr;
            cudaCheck(cudaMallocManaged(&d_tile_offsets, (numDeviceTiles + 1) * sizeof(uint32_t)));
            cudaCheck(cudaMemsetAsync(d_tile_offsets, 0, sizeof(uint32_t), stream));
            CUB_LAUNCH(DeviceScan::InclusiveSum, mTempDevicePools[deviceId], stream, devicePointsPerTile, d_tile_offsets + 1, numDeviceTiles);

            uint64_t* deviceKeys = mKeys + deviceStripeOffset;
            uint32_t* deviceIndices = mIndices + deviceStripeOffset;
            uint64_t* deviceOutputKeys = mData->d_keys + deviceStripeOffset;
            uint32_t* deviceOutputIndices = mData->d_indx + deviceStripeOffset;

            util::cuda::lambdaKernel<<<numBlocks(deviceStripeCount), mNumThreads, 0, stream>>>(deviceStripeCount, BulkVoxelKeyFunctor<BuildT, PtrT>(), mData, coords, d_tile_offsets, numDeviceTiles, deviceKeys, deviceIndices, tileIdOffset);
            cudaCheckError();
            CUB_LAUNCH(DeviceSegmentedRadixSort::SortPairs, mTempDevicePools[deviceId], stream, deviceKeys, deviceOutputKeys, deviceIndices, deviceOutputIndices, (int)deviceStripeCount, (int)numDeviceTiles, d_tile_offsets, d_tile_offsets + 1, 0, 36);

            cudaCheck(cudaFree(d_tile_offsets));
        } else {
            // Serial per-tile sort: individual kernel + sort per tile (lower overhead for few tiles)
            for (uint32_t i = 0, tileOffset = 0, id = tileIdOffset; i < numDeviceTiles; ++i) {
                if (!devicePointsPerTile[i]) continue;

                util::cuda::offsetLambdaKernel<<<numBlocks(devicePointsPerTile[i]), mNumThreads, 0, stream>>>(devicePointsPerTile[i], tileOffset + deviceStripeOffset, VoxelKeyFunctor<BuildT, PtrT>(), mData, coords, id, mKeys, mIndices);

                uint64_t* tileInputKeys = mKeys + tileOffset + deviceStripeOffset;
                uint32_t* tileInputIndices = mIndices + tileOffset + deviceStripeOffset;
                uint64_t* tileOutputKeys = mData->d_keys + tileOffset + deviceStripeOffset;
                uint32_t* tileOutputIndices = mData->d_indx + tileOffset + deviceStripeOffset;

                CUB_LAUNCH(DeviceRadixSort::SortPairs, mTempDevicePools[deviceId], stream, tileInputKeys, tileOutputKeys, tileInputIndices, tileOutputIndices, devicePointsPerTile[i], 0, 36);
                ++id;
                tileOffset += devicePointsPerTile[i];
            }
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

            if (deviceId > 0) {
                cudaCheck(cudaEventSynchronize(voxelCountEvents[deviceId - 1]));
                devicePointsPerVoxel += mVoxelCounts[deviceId - 1];

                cudaCheck(cudaEventSynchronize(leafCountEvents[deviceId - 1]));
                devicePointsPerLeaf += deviceNodeCount(deviceId - 1)[0];
            }

            if (mStripeCounts[deviceId]) {
                CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, deviceOutputKeys, deviceInputKeys, devicePointsPerVoxel, mVoxelCounts + deviceId, mStripeCounts[deviceId]);
                cudaCheck(cudaEventRecord(voxelCountEvents[deviceId], stream));

                CUB_LAUNCH(DeviceRunLengthEncode::Encode, mTempDevicePools[deviceId], stream, thrust::make_transform_iterator(deviceOutputKeys, ShiftRight<9>()), deviceInputKeys, devicePointsPerLeaf, deviceNodeCount(deviceId), mStripeCounts[deviceId]);
                cudaCheck(cudaEventRecord(leafCountEvents[deviceId], stream));
            } else {
                cudaCheck(cudaMemsetAsync(mVoxelCounts + deviceId, 0, sizeof(uint32_t), stream));
                cudaCheck(cudaEventRecord(voxelCountEvents[deviceId], stream));

                cudaCheck(cudaMemsetAsync(deviceNodeCount(deviceId), 0, sizeof(uint32_t), stream));
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
        if (deviceNodeCount(deviceId)[0]) {
            cudaCheck(cudaMemcpyAsync(mData->d_leaf_keys + leafOffset, deviceKeys, sizeof(uint64_t) * deviceNodeCount(deviceId)[0], cudaMemcpyDefault, stream));
        }
        deviceNodeOffset(deviceId)[0] = leafOffset;
        leafOffset += deviceNodeCount(deviceId)[0];
    }

    // Parallel RLE with (shifted) keys in order to count leaves and points per leaf
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));

        uint64_t* deviceInputKeys = mKeys + mStripeOffsets[deviceId];
        uint64_t* deviceOutputKeys = mData->d_keys + mStripeOffsets[deviceId];

        if (mStripeCounts[deviceId]) {
            CUB_LAUNCH(DeviceSelect::Unique, mTempDevicePools[deviceId], stream, thrust::make_transform_iterator(deviceOutputKeys, ShiftRight<21>()), deviceInputKeys, deviceNodeCount(deviceId) + 1, mStripeCounts[deviceId]);
        } else {
            cudaCheck(cudaMemsetAsync(deviceNodeCount(deviceId) + 1, 0, sizeof(uint32_t), stream));
        }
        cudaCheck(cudaEventRecord(lowerCountEvents[deviceId], stream));
    }

    uint32_t lowerOffset = 0;
    for (const auto& [deviceId, stream] : mDeviceMesh) {
        cudaCheck(cudaSetDevice(deviceId));
        cudaCheck(cudaEventSynchronize(lowerCountEvents[deviceId]));
        uint64_t* deviceKeys = mKeys + mStripeOffsets[deviceId];
        if (deviceNodeCount(deviceId)[1]) {
            cudaCheck(cudaMemcpyAsync(mData->d_lower_keys + lowerOffset, deviceKeys, sizeof(uint64_t) * deviceNodeCount(deviceId)[1], cudaMemcpyDefault, stream));
        }
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

    // Zero the name field, then copy only the actual string (if any).
    char *dst = mData->getGrid().mGridName;
    cudaCheck(cudaMemsetAsync(dst, 0, GridData::MaxNameSize, stream));
    if (!mGridName.empty()) {
        // Copy at most MaxNameSize-1 bytes so the memset's trailing '\0' always
        // survives; a name >= MaxNameSize is truncated, never left unterminated.
        const size_t nameSize = std::min<size_t>(mGridName.size(), GridData::MaxNameSize - 1);
        cudaCheck(cudaMemcpyAsync(dst, mGridName.c_str(), nameSize, cudaMemcpyHostToDevice, stream));
    }
    cudaCheck(cudaEventRecord(processGridTreeRootEvent, stream));

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

        auto lastDeviceId = cudaInvalidDeviceId;
        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaSetDevice(deviceId);
            cudaEventCreateWithFlags(&leafCountEvents[deviceId], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&valueIndexPrefixSumEvents[deviceId], cudaEventDisableTiming);

            if (deviceNodeCount(deviceId)[0]) {
                lastDeviceId = deviceId;
                kernels::fillValueIndexKernel<BuildT><<<numBlocks(deviceNodeCount(deviceId)[0]), mNumThreads, 0, stream>>>(deviceNodeCount(deviceId)[0], deviceNodeOffset(deviceId)[0], mValueIndex, mData);
                cudaCheckError();
            }
        }

        LeafCountIterator leafCountIterator(mNodeCounts);
        inclusiveSumAsync(mDeviceMesh, mTempDevicePools, mValueIndex, mValueIndexPrefix, leafCountIterator, leafCountEvents.data(), valueIndexPrefixSumEvents.data());

        // The first leaf on each device reads the last prefix value produced by the
        // previous device, while the first leaf globally reads the final prefix value.
        // Wait for the corresponding producer before launching each leaf-processing kernel.
        auto previousDeviceId = cudaInvalidDeviceId;
        for (const auto& [deviceId, stream] : mDeviceMesh) {
            cudaSetDevice(deviceId);
            if (deviceNodeCount(deviceId)[0]) {
                cudaStreamWaitEvent(stream, valueIndexPrefixSumEvents[lastDeviceId]);
                if (previousDeviceId != cudaInvalidDeviceId) {
                    cudaStreamWaitEvent(stream, valueIndexPrefixSumEvents[previousDeviceId]);
                }
                previousDeviceId = deviceId;
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
