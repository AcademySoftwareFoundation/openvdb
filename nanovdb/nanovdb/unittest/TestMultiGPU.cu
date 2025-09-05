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
    EXPECT_TRUE(grid->activeVoxelCount() == 512);
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

