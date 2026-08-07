// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/DistributedPointsToGrid.cuh>
#include <nanovdb/util/Timer.h>
#include <nanovdb/cuda/DeviceMesh.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <thread> // for std::thread
#include <algorithm> // for std::sort, std::unique
#include <cstdint>
#include <vector>

#include <thrust/fill.h>
#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>

/// @brief Tests the correctness of multi-GPU radix sort
TEST(TestNanoVDBMultiGPU, RadixSort)
{
    nanovdb::cuda::DeviceMesh deviceMesh;

    std::vector<cudaEvent_t> preEvents(deviceMesh.deviceCount());
    std::vector<cudaEvent_t> postEvents(deviceMesh.deviceCount());
    std::vector<nanovdb::cuda::TempDevicePool> tempDevicePools(deviceMesh.deviceCount());
    std::vector<ptrdiff_t> deviceOffsets(deviceMesh.deviceCount());
    std::vector<size_t> deviceSizes(deviceMesh.deviceCount());

    using KeyT = int;
    using ValueT = double;

    std::srand(444);
    const size_t numItems = 1571;
    thrust::universal_vector<KeyT> keysIn(numItems);
    thrust::universal_vector<ValueT> valuesIn(numItems);
    for (size_t i = 0; i < numItems; ++i)
    {
        keysIn[i] = rand();
        valuesIn[i] = -static_cast<ValueT>(keysIn[i]);
    }

    thrust::universal_vector<KeyT> keysOut(numItems);
    thrust::universal_vector<ValueT> valuesOut(numItems);

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaEventCreateWithFlags(&preEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&postEvents[deviceId], cudaEventDisableTiming);

        auto deviceSize = (numItems + deviceMesh.deviceCount() - 1) / deviceMesh.deviceCount();
        deviceOffsets[deviceId] = deviceSize * deviceId;
        deviceSizes[deviceId] = std::min(deviceSize, numItems - deviceOffsets[deviceId]);
    }

    nanovdb::tools::cuda::radixSortAsync(deviceMesh, tempDevicePools.data(), keysIn.data().get(), keysOut.data().get(), valuesIn.data().get(), valuesOut.data().get(), numItems, deviceOffsets.data(), deviceSizes.data(), preEvents.data(), postEvents.data());

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaEventSynchronize(postEvents[deviceId]);
    }

    for (size_t i = 0; i < numItems -1; ++i) {
        EXPECT_LE(keysOut[i], keysOut[i + 1]);
        EXPECT_GE(valuesOut[i], valuesOut[i + 1]);
    }

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaEventDestroy(postEvents[deviceId]);
        cudaEventDestroy(preEvents[deviceId]);
    }
}

/// @brief Tests the correctness of multi-GPU exclusive sums against an equivalent CPU implementation
TEST(TestNanoVDBMultiGPU, ExclusiveSum)
{
    nanovdb::cuda::DeviceMesh deviceMesh;

    std::vector<cudaEvent_t> preEvents(deviceMesh.deviceCount());
    std::vector<cudaEvent_t> postEvents(deviceMesh.deviceCount());
    std::vector<nanovdb::cuda::TempDevicePool> tempDevicePools(deviceMesh.deviceCount());
    std::vector<size_t> deviceSizes(deviceMesh.deviceCount());

    thrust::universal_vector<int> input(937);
    thrust::universal_vector<int> output(input.size());

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaEventCreateWithFlags(&preEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&postEvents[deviceId], cudaEventDisableTiming);

        auto deviceSize = (input.size() + deviceMesh.deviceCount() - 1) / deviceMesh.deviceCount();
        const ptrdiff_t deviceOffset = deviceSize * deviceId;
        deviceSizes[deviceId] = std::min(deviceSize, input.size() - deviceOffset);

        thrust::fill(thrust::cuda::par.on(stream), input.begin() + deviceOffset, input.begin() + deviceOffset + deviceSizes[deviceId], 0);
        thrust::fill(thrust::cuda::par.on(stream), output.begin() + deviceOffset, output.begin() + deviceOffset + deviceSizes[deviceId], 0);
    }

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaStreamSynchronize(stream);
    }

    {
        // Set the input indices corresponding to the Fibbonacci sequence to be 1, rest 0
        input[0] = 1;
        size_t i = 0;
        size_t j = 1;
        size_t k = i + j;
        while(k < input.size()) {
            input[k] = 1;
            i = j;
            j = k;
            k = (i + j);
        }
    }

    nanovdb::tools::cuda::exclusiveSumAsync(deviceMesh, tempDevicePools.data(), input.data(), output.data(), deviceSizes.begin(), preEvents.data(), postEvents.data());

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaStreamSynchronize(stream);
    }

    int accumulator = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], accumulator);
        accumulator += input[i];
    }

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaEventDestroy(postEvents[deviceId]);
        cudaEventDestroy(preEvents[deviceId]);
    }
}

/// @brief Tests the correctness of multi-GPU inclusive sums against an equivalent CPU implementation
TEST(TestNanoVDBMultiGPU, InclusiveSum)
{
    nanovdb::cuda::DeviceMesh deviceMesh;

    std::vector<cudaEvent_t> preEvents(deviceMesh.deviceCount());
    std::vector<cudaEvent_t> postEvents(deviceMesh.deviceCount());
    std::vector<nanovdb::cuda::TempDevicePool> tempDevicePools(deviceMesh.deviceCount());
    std::vector<size_t> deviceSizes(deviceMesh.deviceCount());

    thrust::universal_vector<int> input(937);
    thrust::universal_vector<int> output(input.size());

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaEventCreateWithFlags(&preEvents[deviceId], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&postEvents[deviceId], cudaEventDisableTiming);

        auto deviceSize = (input.size() + deviceMesh.deviceCount() - 1) / deviceMesh.deviceCount();
        const ptrdiff_t deviceOffset = deviceSize * deviceId;
        deviceSizes[deviceId] = std::min(deviceSize, input.size() - deviceOffset);

        thrust::fill(thrust::cuda::par.on(stream), input.begin() + deviceOffset, input.begin() + deviceOffset + deviceSizes[deviceId], 0);
        thrust::fill(thrust::cuda::par.on(stream), output.begin() + deviceOffset, output.begin() + deviceOffset + deviceSizes[deviceId], 0);
    }

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaStreamSynchronize(stream);
    }

    {
        // Set the input indices corresponding to the Fibbonacci sequence to be 1, rest 0
        input[0] = 1;
        size_t i = 0;
        size_t j = 1;
        size_t k = i + j;
        while(k < input.size()) {
            input[k] = 1;
            i = j;
            j = k;
            k = (i + j);
        }
    }

    nanovdb::tools::cuda::inclusiveSumAsync(deviceMesh, tempDevicePools.data(), input.data(), output.data(), deviceSizes.begin(), preEvents.data(), postEvents.data());

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaStreamSynchronize(stream);
    }

    int accumulator = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        accumulator += input[i];
        EXPECT_EQ(output[i], accumulator);
    }

    for (const auto& [deviceId, stream] : deviceMesh) {
        cudaSetDevice(deviceId);
        cudaEventDestroy(postEvents[deviceId]);
        cudaEventDestroy(preEvents[deviceId]);
    }
}

/// @brief Tests multi-GPU creation of grids for a single dense leaf
TEST(TestNanoVDBMultiGPU, DenseLeaf_DistributedCudaPointsToGrid_UnifiedBuffer)
{
    int current = 0;
    cudaCheck(cudaGetDevice(&current));

    using BufferT = nanovdb::cuda::UnifiedBuffer;
    using BuildT = nanovdb::ValueOnIndex;
    // Initialize coordinates corresponding to a single dense leaf. In
    // DistributedPointsToGrid, individual leaf nodes are resident and
    // processed entirely on a single GPU. Thus, the single leaf case results
    // in the edge case where one GPU constructs a leaf while the others idle.
    const size_t voxelCount = 8 * 8 * 8;
    nanovdb::Coord* voxels =  nullptr;
    const size_t voxelSize = voxelCount * sizeof(nanovdb::Coord);
    cudaCheck(cudaMallocManaged(&voxels, voxelSize));
    for (int32_t i = 0; i < 8; ++i)
        for (int32_t j = 0; j < 8; ++j)
            for (int32_t k = 0; k < 8; ++k)
                voxels[i * 8 * 8 + j * 8 + k] = nanovdb::Coord(i, j, k);

    nanovdb::cuda::DeviceMesh deviceMesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);
    auto handle = converter.getHandle(voxels, voxelCount);

    EXPECT_TRUE(handle.deviceData());// grid exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));
    EXPECT_TRUE(handle.data());// grid also exists on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// grid also exists on the CPU
    EXPECT_TRUE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(voxelCount, grid->activeVoxelCount());
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    cudaCheck(cudaFree(voxels));
    cudaSetDevice(current); // restore device so subsequent tests don't fail
}// Large_DistributedCudaPointsToGrid_UnifiedBuffer

/// @brief Tests multi-GPU creation of grids for a large number of randomly sampled voxels
TEST(TestNanoVDBMultiGPU, Large_DistributedCudaPointsToGrid_UnifiedBuffer)
{
    int current = 0;
    cudaCheck(cudaGetDevice(&current));

    using BufferT = nanovdb::cuda::UnifiedBuffer;
    using BuildT = nanovdb::ValueOnIndex;
    nanovdb::util::Timer timer;
    const size_t voxelCount = 1 << 20;// 1048576
    nanovdb::Coord* voxels =  nullptr;
    const size_t voxelSize = voxelCount * sizeof(nanovdb::Coord);
    cudaCheck(cudaMallocManaged(&voxels, voxelSize));
    {//generate random voxels
        std::srand(98765);
        const int max = 512, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        for (size_t i = 0; i < voxelCount; ++i)
            voxels[i] = nanovdb::Coord(op(), op(), op());
    }

    nanovdb::cuda::DeviceMesh deviceMesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);
    auto handle = converter.getHandle(voxels, voxelCount);
    // auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT, nanovdb::Coord*, BufferT>(voxels, voxelCount);

    EXPECT_TRUE(handle.deviceData());// grid exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));
    EXPECT_TRUE(handle.data());// grid also exists on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// grid also exists on the CPU
    EXPECT_TRUE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->valueCount()>0);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(0, voxelCount, 1, [&](const nanovdb::util::Range1D &r){
        auto acc = grid->getAccessor();
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord &ijk = voxels[i];
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_TRUE(acc.getValue(ijk) > 0u);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
        }
    });

    cudaCheck(cudaFree(voxels));
    cudaSetDevice(current); // restore device so subsequent tests don't fail
}// Large_DistributedCudaPointsToGrid_UnifiedBuffer

/// @brief Exercises the serial per-tile sort path (< 32 tiles per device) in DistributedPointsToGrid.
///        Coordinates in [-512, 512] produce at most 2^3 = 8 upper internal node tiles
///        (each upper node covers 4096 voxels per dimension), well below the threshold of 32.
TEST(TestNanoVDBMultiGPU, FewTiles_DistributedCudaPointsToGrid)
{
    int current = 0;
    cudaCheck(cudaGetDevice(&current));

    using BuildT = nanovdb::ValueOnIndex;
    const size_t voxelCount = 1 << 16;// 65536
    nanovdb::Coord* voxels = nullptr;
    cudaCheck(cudaMallocManaged(&voxels, voxelCount * sizeof(nanovdb::Coord)));
    {
        std::srand(12345);
        const int max = 512, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        for (size_t i = 0; i < voxelCount; ++i)
            voxels[i] = nanovdb::Coord(op(), op(), op());
    }

    nanovdb::cuda::DeviceMesh deviceMesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);
    auto handle = converter.getHandle(voxels, voxelCount);

    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    handle.deviceDownload();
    auto *grid = handle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->valueCount() > 0);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    nanovdb::util::forEach(0, voxelCount, 1, [&](const nanovdb::util::Range1D &r){
        auto acc = grid->getAccessor();
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord &ijk = voxels[i];
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_TRUE(acc.getValue(ijk) > 0u);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
        }
    });

    cudaCheck(cudaFree(voxels));
    cudaSetDevice(current);
}// FewTiles_DistributedCudaPointsToGrid

/// @brief Exercises the segmented sort path (>= 32 tiles per device) in DistributedPointsToGrid.
///        Coordinates in [-16000, 16000] produce ~8^3 = 512 upper internal node tiles
///        (each upper node covers 4096 voxels per dimension). Even with 8 GPUs, each device
///        receives ~64 tiles, exceeding the threshold of 32.
TEST(TestNanoVDBMultiGPU, ManyTiles_DistributedCudaPointsToGrid)
{
    int current = 0;
    cudaCheck(cudaGetDevice(&current));

    using BuildT = nanovdb::ValueOnIndex;
    const size_t voxelCount = 1 << 20;// 1048576
    nanovdb::Coord* voxels = nullptr;
    cudaCheck(cudaMallocManaged(&voxels, voxelCount * sizeof(nanovdb::Coord)));
    {
        std::srand(54321);
        const int max = 16000, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        for (size_t i = 0; i < voxelCount; ++i)
            voxels[i] = nanovdb::Coord(op(), op(), op());
    }

    nanovdb::cuda::DeviceMesh deviceMesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);
    auto handle = converter.getHandle(voxels, voxelCount);

    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    handle.deviceDownload();
    auto *grid = handle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->valueCount() > 0);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    nanovdb::util::forEach(0, voxelCount, 1, [&](const nanovdb::util::Range1D &r){
        auto acc = grid->getAccessor();
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord &ijk = voxels[i];
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_TRUE(acc.getValue(ijk) > 0u);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
        }
    });

    cudaCheck(cudaFree(voxels));
    cudaSetDevice(current);
}// ManyTiles_DistributedCudaPointsToGrid

/// @brief Regression test for the multi-GPU shared-tile race. All coordinates
///        fall inside a single upper-node tile (each upper node spans 4096
///        voxels per axis, so ijk in [0, 4095] maps to tile (0,0,0)), yet they
///        populate many lower and leaf nodes. With the default even initial
///        split this one tile is shared across every GPU, so the builder must
///        consolidate it onto a single device. A cross-device race in leaf
///        construction would drop a device's contribution and undercount the
///        active voxels, which the exact-count assertion below catches
///        deterministically.
TEST(TestNanoVDBMultiGPU, SingleUpperNode_DistributedCudaPointsToGrid_UnifiedBuffer)
{
    int current = 0;
    cudaCheck(cudaGetDevice(&current));

    using BufferT = nanovdb::cuda::UnifiedBuffer;
    using BuildT = nanovdb::ValueOnIndex;

    const size_t inputCount = 1 << 18;// 262144
    nanovdb::Coord* voxels = nullptr;
    cudaCheck(cudaMallocManaged(&voxels, inputCount * sizeof(nanovdb::Coord)));
    std::srand(24680);
    auto op = [](){ return rand() % 4096; };// stays within a single upper node
    for (size_t i = 0; i < inputCount; ++i)
        voxels[i] = nanovdb::Coord(op(), op(), op());

    // Deterministic expected active-voxel count (the input may contain duplicates).
    std::vector<uint64_t> packed(inputCount);
    for (size_t i = 0; i < inputCount; ++i)
        packed[i] = (uint64_t(voxels[i][0]) << 24) | (uint64_t(voxels[i][1]) << 12) | uint64_t(voxels[i][2]);
    std::sort(packed.begin(), packed.end());
    const size_t uniqueCount = static_cast<size_t>(std::unique(packed.begin(), packed.end()) - packed.begin());

    nanovdb::cuda::DeviceMesh deviceMesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);
    auto handle = converter.getHandle(voxels, inputCount);

    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    handle.deviceDownload();
    auto *grid = handle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());
    // The input occupies exactly one upper-node tile, so the shared-tile
    // consolidation path is exercised.
    EXPECT_EQ(1u, grid->tree().nodeCount(2));
    // Every unique input voxel must be active exactly once.
    EXPECT_EQ(static_cast<uint64_t>(uniqueCount), grid->activeVoxelCount());

    nanovdb::util::forEach(0, inputCount, 1, [&](const nanovdb::util::Range1D &r){
        auto acc = grid->getAccessor();
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord &ijk = voxels[i];
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_TRUE(acc.getValue(ijk) > 0u);
        }
    });

    cudaCheck(cudaFree(voxels));
    cudaSetDevice(current);
}// SingleUpperNode_DistributedCudaPointsToGrid_UnifiedBuffer

/// @brief Cross-checks the distributed builder against the trusted single-GPU
///        PointsToGrid on an input that forces a single tile to be split across
///        devices. Index assignment order may differ between the two builders,
///        so we compare topology (node counts, active-voxel count) and voxel
///        occupancy rather than the ValueOnIndex indices themselves.
TEST(TestNanoVDBMultiGPU, MatchesSingleGpu_DistributedCudaPointsToGrid)
{
    int current = 0;
    cudaCheck(cudaGetDevice(&current));

    using BufferT = nanovdb::cuda::UnifiedBuffer;
    using BuildT = nanovdb::ValueOnIndex;

    const size_t inputCount = 1 << 17;// 131072, all within a single upper node
    nanovdb::Coord* voxels = nullptr;
    cudaCheck(cudaMallocManaged(&voxels, inputCount * sizeof(nanovdb::Coord)));
    std::srand(1357);
    auto op = [](){ return rand() % 4096; };
    for (size_t i = 0; i < inputCount; ++i)
        voxels[i] = nanovdb::Coord(op(), op(), op());

    nanovdb::cuda::DeviceMesh deviceMesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);
    auto distributedHandle = converter.getHandle(voxels, inputCount);
    distributedHandle.deviceDownload();
    auto *distributedGrid = distributedHandle.grid<BuildT>();
    EXPECT_TRUE(distributedGrid);

    cudaSetDevice(current);
    auto referenceHandle = nanovdb::tools::cuda::voxelsToGrid<BuildT, nanovdb::Coord*, BufferT>(voxels, inputCount);
    referenceHandle.deviceDownload();
    auto *referenceGrid = referenceHandle.grid<BuildT>();
    EXPECT_TRUE(referenceGrid);

    EXPECT_EQ(referenceGrid->activeVoxelCount(), distributedGrid->activeVoxelCount());
    EXPECT_EQ(referenceGrid->tree().nodeCount(0), distributedGrid->tree().nodeCount(0));
    EXPECT_EQ(referenceGrid->tree().nodeCount(1), distributedGrid->tree().nodeCount(1));
    EXPECT_EQ(referenceGrid->tree().nodeCount(2), distributedGrid->tree().nodeCount(2));

    nanovdb::util::forEach(0, inputCount, 1, [&](const nanovdb::util::Range1D &r){
        auto distributedAcc = distributedGrid->getAccessor();
        auto referenceAcc = referenceGrid->getAccessor();
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord &ijk = voxels[i];
            EXPECT_TRUE(referenceAcc.isActive(ijk));
            EXPECT_TRUE(distributedAcc.isActive(ijk));
        }
    });

    cudaCheck(cudaFree(voxels));
    cudaSetDevice(current);
}// MatchesSingleGpu_DistributedCudaPointsToGrid

/// @brief Edge case: fewer input voxels than devices. Interior devices receive
///        no work, so this guards against out-of-range indexing on empty stripes.
TEST(TestNanoVDBMultiGPU, FewerVoxelsThanDevices_DistributedCudaPointsToGrid)
{
    int current = 0;
    cudaCheck(cudaGetDevice(&current));

    using BufferT = nanovdb::cuda::UnifiedBuffer;
    using BuildT = nanovdb::ValueOnIndex;

    // Three well-separated voxels (each in its own upper-node tile).
    const size_t inputCount = 3;
    nanovdb::Coord* voxels = nullptr;
    cudaCheck(cudaMallocManaged(&voxels, inputCount * sizeof(nanovdb::Coord)));
    voxels[0] = nanovdb::Coord(0, 0, 0);
    voxels[1] = nanovdb::Coord(10000, -4000, 7000);
    voxels[2] = nanovdb::Coord(-9000, 12000, -15000);

    nanovdb::cuda::DeviceMesh deviceMesh;
    nanovdb::tools::cuda::DistributedPointsToGrid<BuildT> converter(deviceMesh);
    auto handle = converter.getHandle(voxels, inputCount);
    handle.deviceDownload();
    auto *grid = handle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(static_cast<uint64_t>(inputCount), grid->activeVoxelCount());

    {
        auto acc = grid->getAccessor();
        for (size_t i = 0; i < inputCount; ++i)
            EXPECT_TRUE(acc.isActive(voxels[i]));
    }

    cudaCheck(cudaFree(voxels));
    cudaSetDevice(current);
}// FewerVoxelsThanDevices_DistributedCudaPointsToGrid
