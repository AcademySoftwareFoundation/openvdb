// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <cstring>
#include <type_traits>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/NodeManager.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/cuda/SignedFloodFill.cuh>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/cuda/IndexToGrid.cuh>
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/AddBlindData.cuh>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/tools/cuda/GridValidator.cuh>
#include <nanovdb/tools/cuda/GridStats.cuh>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/CoarsenGrid.cuh>
#include <nanovdb/tools/cuda/RefineGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/math/Proximity.h>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/Timer.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/PinnedResource.h>
#include <nanovdb/cuda/HandleStorage.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/cuda/ManagedResource.h>
#include <nanovdb/cuda/DeviceStreamMap.h>
#include <nanovdb/cuda/DeviceMesh.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <algorithm>// for std::sort
#include <cstring> // for std::memcpy
#include <unordered_set>
#include <iomanip> // for std::setw, std::setfill
#include <random> // for std::mt19937, std::bernoulli_distribution
#include <thread> // for std::thread
#include <type_traits>
#include <utility> // for std::declval

/// @brief Single-space device storage that the GPU builders write into.
using SingleBufT = nanovdb::cuda::Buffer<std::byte>;
static_assert(nanovdb::BufferIsDeviceOnly<SingleBufT>::value, "device grids have no host accessors");


namespace nanovdb {// this namespace is required by gtest

namespace test {

// used to test cudaStr methods
int signum(int val)
{
    return (0 < val) - (val < 0);
}// signum

void cudaStr()
{
    const size_t size = 50;
    char *str, *d_str;
    cudaCheck(cudaMallocHost((void**)&str, size));
    cudaCheck(cudaMalloc(  (void**)&d_str, size));
    int n, *d_n;
    cudaCheck(cudaMalloc((void**)&d_n, sizeof(int)));

    nanovdb::util::cuda::lambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        nanovdb::util::strcpy(d_str, "this is a test");
    });
    cudaCheck(cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost));
    EXPECT_STREQ(str, "this is a test");
    nanovdb::util::cuda::lambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        nanovdb::util::strcat(d_str, " #2");
    });
    cudaCheck(cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost));
    EXPECT_STREQ(str, "this is a test #2");

    nanovdb::util::cuda::lambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        *d_n = nanovdb::util::strcmp(d_str, "this is a test");
    });
    cudaCheck(cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost));
    //std::cerr << "n = " << n << std::endl;
    EXPECT_EQ(signum(std::strcmp(str, "this is a test")), signum(n));
    nanovdb::util::cuda::lambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        *d_n = nanovdb::util::strcmp(d_str, "this is a test #2");
    });
    cudaCheck(cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(std::strcmp(str, "this is a test #2"), n);
    EXPECT_EQ(0, n);

    cudaCheck(cudaFreeHost(str));
    cudaCheck(cudaFree(d_n));
    cudaCheck(cudaFree(d_str));
}// cudaStr

}// namespace test

}// namespace nanovdb

TEST(TestNanoVDBCUDA, CudaStr)
{
   nanovdb::test::cudaStr();
}

__global__ void testKernel(int device)
{
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    //if (err != cudaSuccess) printf("kernel cuda error: %d\n", (int)err);
    if (dev != device) printf("Error: expected device ID = %i but was called with %i\n", dev, device);
}

TEST(TestNanoVDBCUDA, DeviceStreamMap)
{
    using DevMap = nanovdb::cuda::DeviceStreamMap;
    int count = 0, verbose = 0, current = 0;
    {
        cudaCheck(cudaGetDeviceCount(&count));
        cudaCheck(cudaGetDevice(&current));
        EXPECT_EQ(count,   nanovdb::util::cuda::deviceCount());
        EXPECT_EQ(current, nanovdb::util::cuda::currentDevice());
        float *ptr = new float;
        const int deviceID = nanovdb::util::cuda::ptrToDevice(ptr);
        EXPECT_TRUE(cudaInvalidDeviceId == deviceID || cudaCpuDeviceId == deviceID);
        EXPECT_GT(0, deviceID);
        //EXPECT_EQ(cudaInvalidDeviceId, deviceID);
        //EXPECT_EQ(cudaCpuDeviceId, deviceID);
        delete ptr;
        cudaCheck(cudaMalloc((void**)&ptr, sizeof(float)));
        EXPECT_EQ(current, nanovdb::util::cuda::ptrToDevice(ptr));
        cudaCheck(cudaFree(ptr));
    }
    //std::cout << "Total device count = " << count << std::endl;
    {
        //std::cout << "Any:\n";
        DevMap map(DevMap::Any, {}, verbose);
        EXPECT_EQ(count, map.size());
        for (const auto& [device, stream] : map) {
            cudaSetDevice(device);
            testKernel<<<1, 1, 0, stream>>>(device);
            cudaStreamSynchronize(stream);
        }
    }
    {
        //std::cout << "Any excluding {current}:\n";
        DevMap map(DevMap::Any, {current}, verbose);
        EXPECT_EQ(count-1, map.size());
        for (const auto& [device, stream] : map) {
            cudaSetDevice(device);
            testKernel<<<1, 1, 0, stream>>>(device);
            cudaStreamSynchronize(stream);
        }
    }
    {
        //std::cout << "PeerToPeer:\n";
        DevMap map(DevMap::PeerToPeer, {},  verbose);
        EXPECT_GE(count, map.size());
        for (const auto& [device, stream] : map) {
            cudaSetDevice(device);
            testKernel<<<1, 1, 0, stream>>>(device);
            cudaStreamSynchronize(stream);
        }
    }
    {
        //std::cout << "Unified:\n";
        DevMap map(DevMap::Unified, {}, verbose);
        EXPECT_GE(count, map.size());
        for (const auto& [device, stream] : map) {
            cudaSetDevice(device);
            testKernel<<<1, 1, 0, stream>>>(device);
            cudaStreamSynchronize(stream);
        }
    }
    cudaSetDevice(current); // restore device so subsequent tests don't fail
}

TEST(TestNanoVDBCUDA, DeviceMesh)
{
    int count = 0, current = 0;
    {
        cudaCheck(cudaGetDeviceCount(&count));
        cudaCheck(cudaGetDevice(&current));
        EXPECT_EQ(count,   nanovdb::util::cuda::deviceCount());
        EXPECT_EQ(current, nanovdb::util::cuda::currentDevice());
    }
    //std::cout << "Total device count = " << count << std::endl;
    nanovdb::cuda::DeviceMesh serialMesh;
    EXPECT_EQ(count, serialMesh.deviceCount());
    EXPECT_GT(nanovdb::cuda::minDevicePageSize(serialMesh), 0);
    // Construction of deviceMesh (and destruction) should not modify current device
    EXPECT_EQ(current, nanovdb::util::cuda::currentDevice());
    // Serial kernel dispatch with iterator
    for (const auto& [device, stream] : serialMesh) {
        cudaSetDevice(device);
        testKernel<<<1, 1, 0, stream>>>(device);
    }
    // Wait on each kernel to finish using indexing operation
    for (size_t i = 0; i < serialMesh.deviceCount(); ++i) {
        cudaSetDevice(serialMesh[i].id);
        cudaStreamSynchronize(serialMesh[i].stream);
    }

    // Test move constructor
    nanovdb::cuda::DeviceMesh parallelMesh(std::move(serialMesh));
    EXPECT_EQ(count, parallelMesh.deviceCount());
    // After move constructor, serialMesh should have no devices
    EXPECT_EQ(0, serialMesh.deviceCount());
    // Parallel kernel dispatch with iterator
    std::vector<std::thread> threads;
    for (const auto& [device, stream] : parallelMesh) {
        threads.emplace_back([](int device, cudaStream_t stream) {
            cudaSetDevice(device);
            testKernel<<<1, 1, 0, stream>>>(device);
            cudaStreamSynchronize(stream);
        }, device, stream);
    }
    std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
    threads.clear();

    cudaSetDevice(current); // Restore device so subsequent tests don't fail
}

TEST(TestNanoVDBCUDA, Basic_CudaPointsToGrid_float)
{
    using BuildT = float;
    using GridT = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 1;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points, 1.0, SingleBufT());
    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.bufferSize(), size);

    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));// copy to the CPU
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    GridT *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);

    auto acc = grid->getAccessor();
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(0,2,3)));
    EXPECT_TRUE( acc.isActive(nanovdb::Coord(1,2,3)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(1,2,4)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(2,2,3)));
    auto *leaf = acc.probeLeaf(nanovdb::Coord(1,2,3));
    EXPECT_TRUE(leaf);
    EXPECT_EQ(nanovdb::Coord(0), leaf->origin());
    EXPECT_EQ(1u, leaf->valueMask().countOn());
    EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->bbox()[1]);
}// Basic_CudaPointsToGrid_float

namespace nanovdb {
namespace test {

/// @brief Implements Tree::probeValue(Coord)
/// @tparam BuildT Build type of the grid being called
template <typename BuildT>
struct ProbeValueNew {
    using ValueT = typename BuildToValueMap<BuildT>::Type;
    struct Probe {
        bool state;
        ValueT value;
        operator bool() const { return state; }
    };
    static constexpr int LEVEL = 0;// minimum level for the descent during top-down traversal
    using Type = Probe;
    __hostdev__ static Probe get(const NanoRoot<BuildT>  &root) {
        return Probe{false, root.mBackground};
    }
    __hostdev__ static Probe get(const typename NanoRoot<BuildT>::Tile &tile) {
        return Probe{tile.state>0, tile.value};
    }
    __hostdev__ static Probe get(const NanoUpper<BuildT> &node, uint32_t n) {
        return Probe{node.mValueMask.isOn(n), node.mTable[n].value};
    }
    __hostdev__ static Probe get(const NanoLower<BuildT> &node, uint32_t n) {
        return Probe{node.mValueMask.isOn(n), node.mTable[n].value};
    }
    __hostdev__ static Probe get(const NanoLeaf<BuildT>  &leaf, uint32_t n) {
        return Probe{leaf.isActive(n), leaf.getValue(n)};
    }
};// ProbeValueNew<BuildT>

}// end of test namespace
}// end of nanovdb namespace

TEST(TestNanoVDBCUDA, Basic_CudaPointsToGrid_ValueIndex)
{
    using BuildT = nanovdb::ValueIndex;
    using GridT  = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 3;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3),
                                         nanovdb::Coord(1, 2, 4),
                                         nanovdb::Coord(8, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU
#if 0
    nanovdb::tools::cuda::PointsToGrid converter;
    auto handle = converter.getHandle<BuildT>(d_coords, num_points);
#else
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points, 1.0, SingleBufT());
#endif
    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          2*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.bufferSize(), size);

    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));// copy to the CPU
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    GridT *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(1u + 2*512u, grid->valueCount());

    auto acc = grid->getAccessor();
    EXPECT_FALSE( acc.isActive(nanovdb::Coord(0,2,3)));
    EXPECT_TRUE(  acc.isActive(nanovdb::Coord(1,2,3)));
    EXPECT_TRUE(  acc.isActive(nanovdb::Coord(1,2,4)));
    EXPECT_TRUE(  acc.isActive(nanovdb::Coord(8,2,3)));
    EXPECT_EQ(1u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(0,2,3)), acc.getValue(nanovdb::Coord(0,2,3)));
    EXPECT_EQ(1u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(1,2,3)), acc.getValue(nanovdb::Coord(1,2,3)));
    EXPECT_EQ(1u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(2,2,3)), acc.getValue(nanovdb::Coord(2,2,3)));
    EXPECT_EQ(1u + 512u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(8,2,3)), acc.getValue(nanovdb::Coord(8,2,3)));

    using OpT = nanovdb::GetValue<BuildT>;
    EXPECT_EQ(1u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(0,2,3)), acc.get<OpT>(nanovdb::Coord(0,2,3)));
    EXPECT_EQ(1u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(1,2,3)), acc.get<OpT>(nanovdb::Coord(1,2,3)));
    EXPECT_EQ(1u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(2,2,3)), acc.get<OpT>(nanovdb::Coord(2,2,3)));
    EXPECT_EQ(1u + 512u + nanovdb::NanoLeaf<BuildT>::CoordToOffset(nanovdb::Coord(8,2,3)), acc.get<OpT>(nanovdb::Coord(8,2,3)));

    for (size_t i=0; i<num_points; ++i)  {
        const nanovdb::Coord ijk = coords[i];
        const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
        EXPECT_TRUE(leaf);
        EXPECT_EQ(0u, leaf->LEVEL);
        const auto offset = leaf->CoordToOffset(ijk);
        EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
        const auto *lower = acc.get<nanovdb::GetLower<BuildT>>(ijk);
        EXPECT_TRUE(lower);
        EXPECT_EQ(1u, lower->LEVEL);
        const auto *upper = acc.get<nanovdb::GetUpper<BuildT>>(ijk);
        EXPECT_TRUE(upper);
        EXPECT_EQ(2u, upper->LEVEL);
    }
}// Basic_CudaPointsToGrid_ValueIndex

TEST(TestNanoVDBCUDA, Basic_CudaPointsToGrid_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;
    using GridT = nanovdb::NanoGrid<BuildT>;
    EXPECT_TRUE(nanovdb::BuildTraits<BuildT>::is_index);
    EXPECT_TRUE(nanovdb::BuildTraits<BuildT>::is_onindex);
    EXPECT_FALSE(nanovdb::BuildTraits<BuildT>::is_offindex);
    const size_t num_points = 3;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3),
                                         nanovdb::Coord(1, 2, 4),
                                         nanovdb::Coord(8, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

#if 0
    nanovdb::tools::cuda::PointsToGrid converter;
    auto handle = converter.getHandle<BuildT>(d_coords, num_points);
#else
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points, 1.0, SingleBufT());
#endif

    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          2*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.bufferSize(), size);

    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));// copy to the CPU
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    GridT *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(1u + num_points, grid->valueCount());

    auto acc = grid->getAccessor();
    EXPECT_FALSE( acc.isActive(nanovdb::Coord(0,2,3)));
    EXPECT_TRUE(  acc.isActive(nanovdb::Coord(1,2,3)));
    EXPECT_TRUE(  acc.isActive(nanovdb::Coord(1,2,4)));
    EXPECT_TRUE(  acc.isActive(nanovdb::Coord(8,2,3)));
    EXPECT_EQ(0u, acc.getValue(nanovdb::Coord(0,2,3)));
    EXPECT_EQ(1u, acc.getValue(nanovdb::Coord(1,2,3)));
    EXPECT_EQ(2u, acc.getValue(nanovdb::Coord(1,2,4)));
    EXPECT_EQ(3u, acc.getValue(nanovdb::Coord(8,2,3)));

    using GetT = nanovdb::GetValue<BuildT>;
    EXPECT_EQ(0u, acc.get<GetT>(nanovdb::Coord(0,2,3)));
    EXPECT_EQ(1u, acc.get<GetT>(nanovdb::Coord(1,2,3)));
    EXPECT_EQ(2u, acc.get<GetT>(nanovdb::Coord(1,2,4)));
    EXPECT_EQ(3u, acc.get<GetT>(nanovdb::Coord(8,2,3)));

    {
        using T = nanovdb::test::ProbeValueNew<BuildT>;
        auto tmp = acc.get<T>(nanovdb::Coord(0,2,3));
        EXPECT_EQ(false, tmp.state);
        EXPECT_EQ(0u,    tmp.value);
        tmp = acc.get<T>(nanovdb::Coord(1,2,3));
        EXPECT_EQ(true, tmp.state);
        EXPECT_EQ(1u,   tmp.value);
        tmp = acc.get<T>(nanovdb::Coord(1,2,4));
        EXPECT_EQ(true, tmp.state);
        EXPECT_EQ(2u,   tmp.value);
        tmp = acc.get<T>(nanovdb::Coord(8,2,3));
        EXPECT_EQ(true, tmp.state);
        EXPECT_EQ(3u,   tmp.value);
    }
    {
        using T = nanovdb::ProbeValue<BuildT>;
        uint64_t value = 0;
        EXPECT_EQ(false, acc.get<T>(nanovdb::Coord(0,2,3), value) );
        EXPECT_EQ(0u,    value);
        EXPECT_EQ(true, acc.get<T>(nanovdb::Coord(1,2,3), value) );
        EXPECT_EQ(1u,    value);
        EXPECT_EQ(true, acc.get<T>(nanovdb::Coord(1,2,4), value) );
        EXPECT_EQ(2u,    value);
        EXPECT_EQ(true, acc.get<T>(nanovdb::Coord(8,2,3), value) );
        EXPECT_EQ(3u,    value);
    }

    for (size_t i=0; i<num_points; ++i)  {
        const nanovdb::Coord ijk = coords[i];
        const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
        EXPECT_TRUE(leaf);
        const auto offset = leaf->CoordToOffset(ijk);
        EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
    }
}// Basic_CudaPointsToGrid_ValueOnIndex

TEST(TestNanoVDBCUDA, Large_CudaPointsToGrid)
{
    using BuildT = nanovdb::ValueOnIndex;
    nanovdb::util::Timer timer;
    const size_t voxelCount = 1 << 20;// 1048576
    std::vector<nanovdb::Coord> voxels;
    {//generate random voxels
        voxels.reserve(voxelCount);
        std::srand(98765);
        const int max = 512, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        //timer.start("Creating "+std::to_string(voxelCount)+" random voxels on the CPU");
        while (voxels.size() < voxelCount) voxels.push_back(nanovdb::Coord(op(), op(), op()));
        //timer.stop();
        EXPECT_EQ(voxelCount, voxels.size());
    }
#if 0
    {// Build grid on CPU
        nanovdb::tools::build::Grid<float> buildGrid(0.0f);
        //timer.start("Building grid on CPU from "+std::to_string(voxels.size())+" points");
        nanovdb::util::forEach0, voxelCount, voxelCount >> 6, [&](const nanovdb::util::Range1D &r){
            auto acc = buildGrid.getWriteAccessor();
            for (size_t i=r.begin(); i!=r.end(); ++i) acc.setValueOn(voxels[i]);
        });
        //timer.restart("Converting CPU build::Grid to nanovdb");
        auto handle = nanovdb::tools::createNanoGrid(buildGrid);
        //timer.stop();
    }
#endif
    nanovdb::Coord* d_coords;
    const size_t voxelSize = voxels.size() * sizeof(nanovdb::Coord);
    //timer.start("Allocating "+std::to_string(voxelSize >> 20)+" MB on the GPU");
    cudaCheck(cudaMalloc(&d_coords, voxelSize));
    //timer.restart("Copying voxels from CPU to GPU");
    cudaCheck(cudaMemcpy(d_coords, voxels.data(), voxelSize, cudaMemcpyHostToDevice));
    //timer.stop();

    timer.start("voxelsToGrid("+std::to_string(voxels.size())+" points)");
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, voxelCount, 1.0);
    timer.stop();
    static_assert(std::is_same<decltype(handle), nanovdb::GridHandle<SingleBufT>>::value,
                  "the GPU builders default to a device-only cuda::Buffer");

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->valueCount()>0);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(voxels,[&](const nanovdb::util::Range1D &r){
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

    //timer.stop();
    cudaCheck(cudaFree(d_coords));
}// Large_CudaPointsToGrid

TEST(TestNanoVDBCUDA, Large_CudaPointsToGrid_ManagedBuffer)
{
    using BuildT = nanovdb::ValueOnIndex;
    nanovdb::util::Timer timer;
    const size_t voxelCount = 1 << 20;// 1048576
    std::vector<nanovdb::Coord> voxels;
    {//generate random voxels
        voxels.reserve(voxelCount);
        std::srand(98765);
        const int max = 512, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        //timer.start("Creating "+std::to_string(voxelCount)+" random voxels on the CPU");
        while (voxels.size() < voxelCount) voxels.push_back(nanovdb::Coord(op(), op(), op()));
        //timer.stop();
        EXPECT_EQ(voxelCount, voxels.size());
    }
#if 0
    {// Build grid on CPU
        nanovdb::tools::build::Grid<float> buildGrid(0.0f);
        //timer.start("Building grid on CPU from "+std::to_string(voxels.size())+" points");
        nanovdb::util::forEach0, voxelCount, voxelCount >> 6, [&](const nanovdb::util::Range1D &r){
            auto acc = buildGrid.getWriteAccessor();
            for (size_t i=r.begin(); i!=r.end(); ++i) acc.setValueOn(voxels[i]);
        });
        //timer.restart("Converting CPU build::Grid to nanovdb");
        auto handle = nanovdb::tools::createNanoGrid(buildGrid);
        //timer.stop();
    }
#endif
    nanovdb::Coord* d_coords;
    const size_t voxelSize = voxels.size() * sizeof(nanovdb::Coord);
    //timer.start("Allocating "+std::to_string(voxelSize >> 20)+" MB on the GPU");
    cudaCheck(cudaMalloc(&d_coords, voxelSize));
    //timer.restart("Copying voxels from CPU to GPU");
    cudaCheck(cudaMemcpy(d_coords, voxels.data(), voxelSize, cudaMemcpyHostToDevice));
    //timer.stop();

    using ManagedBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ManagedResource>;
    timer.start("voxelsToGrid<ManagedBufT>("+std::to_string(voxels.size())+" points)");
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT, nanovdb::Coord*, ManagedBufT>(d_coords, voxelCount, 1.0, ManagedBufT());
    cudaCheck(cudaDeviceSynchronize());// the grid is read on the CPU below
    timer.stop();

    EXPECT_TRUE(handle.deviceData());// grid exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));
    EXPECT_TRUE(handle.data());// the same grid is accessible on the CPU
    EXPECT_EQ(handle.data(), handle.deviceData());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    auto *grid = handle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->valueCount()>0);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(voxels,[&](const nanovdb::util::Range1D &r){
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

    //timer.stop();
    cudaCheck(cudaFree(d_coords));
}// Large_CudaPointsToGrid_ManagedBuffer

/// @brief Exercises the serial per-tile sort path (< 32 tiles) in PointsToGrid.
///        Coordinates in [-512, 512] produce at most 2^3 = 8 upper internal node tiles
///        (each upper node covers 4096 voxels per dimension), well below the threshold of 32.
TEST(TestNanoVDBCUDA, FewTiles_CudaPointsToGrid)
{
    using BuildT = nanovdb::ValueOnIndex;
    const size_t voxelCount = 1 << 16;// 65536
    std::vector<nanovdb::Coord> voxels;
    {
        voxels.reserve(voxelCount);
        std::srand(12345);
        const int max = 512, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        while (voxels.size() < voxelCount) voxels.push_back(nanovdb::Coord(op(), op(), op()));
    }

    nanovdb::Coord* d_coords;
    cudaCheck(cudaMalloc(&d_coords, voxels.size() * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, voxels.data(), voxels.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, voxelCount, 1.0, SingleBufT());

    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->valueCount() > 0);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    nanovdb::util::forEach(voxels, [&](const nanovdb::util::Range1D &r){
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

    cudaCheck(cudaFree(d_coords));
}// FewTiles_CudaPointsToGrid

/// @brief Exercises the segmented sort path (>= 32 tiles) in PointsToGrid.
///        Coordinates in [-8000, 8000] produce 4^3 = 64 upper internal node tiles
///        (each upper node covers 4096 voxels per dimension), exceeding the threshold of 32.
TEST(TestNanoVDBCUDA, ManyTiles_CudaPointsToGrid)
{
    using BuildT = nanovdb::ValueOnIndex;
    const size_t voxelCount = 1 << 20;// 1048576
    std::vector<nanovdb::Coord> voxels;
    {
        voxels.reserve(voxelCount);
        std::srand(54321);
        const int max = 8000, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        while (voxels.size() < voxelCount) voxels.push_back(nanovdb::Coord(op(), op(), op()));
    }

    nanovdb::Coord* d_coords;
    cudaCheck(cudaMalloc(&d_coords, voxels.size() * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, voxels.data(), voxels.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, voxelCount, 1.0, SingleBufT());

    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_TRUE(grid->valueCount() > 0);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid->voxelSize());

    nanovdb::util::forEach(voxels, [&](const nanovdb::util::Range1D &r){
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

    cudaCheck(cudaFree(d_coords));
}// ManyTiles_CudaPointsToGrid

TEST(TestNanoVDBCUDA, mergeSplitGrids)
{
    size_t size1 = 0, size2 = 0;
    std::vector<nanovdb::GridHandle<>> handles1, handles2;
    std::vector<std::string> gridNames;
    //nanovdb::util::Timer timer("create 5 host grids");
    for (int radius = 100; radius<150; radius += 10) {
        gridNames.emplace_back("sphere_" + std::to_string(radius));
        handles1.emplace_back(nanovdb::tools::createLevelSetSphere(radius,nanovdb::Vec3d(0),1,3,
                                                            nanovdb::Vec3d(0), gridNames.back()));
        EXPECT_FALSE(handles1.back().isPadded());
        size1 += handles1.back().bufferSize();
    }
    EXPECT_EQ(5u, gridNames.size());
    EXPECT_EQ(5u, handles1.size());
    //timer.restart("create 5 host grids");
    for (int radius = 150; radius<200; radius += 10) {
        gridNames.emplace_back("sphere_" + std::to_string(radius));
        handles2.emplace_back(nanovdb::tools::createLevelSetSphere(radius,nanovdb::Vec3d(0),1,3,
                                                            nanovdb::Vec3d(0), gridNames.back()));
        size2 += handles2.back().bufferSize();
    }
    EXPECT_EQ(10u, gridNames.size());
    EXPECT_EQ( 5u, handles2.size());
    //timer.restart("merging 5 host grids");
    auto mergedHandle = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(handles2);// merge last 5 grid handles
    EXPECT_EQ(size2, mergedHandle.bufferSize());
    EXPECT_FALSE(mergedHandle.isPadded());
    EXPECT_TRUE(mergedHandle.data());
    auto *gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(5u, gridData->mGridCount);
    EXPECT_EQ(0u, gridData->mGridIndex);
    EXPECT_EQ(handles2[0].bufferSize(), gridData->mGridSize);
    //timer.restart("unit-test host grids");
    for (int i=0; i<5; ++i){
        gridData = mergedHandle.gridData(i);
        EXPECT_TRUE(gridData);
        EXPECT_EQ(i, gridData->mGridIndex);
        EXPECT_EQ(handles2[i].bufferSize(), gridData->mGridSize);
        EXPECT_EQ(strcmp(gridNames[i+5].c_str(), gridData->mGridName),0);
    }

    EXPECT_FALSE(mergedHandle.empty());
    handles1.push_back(std::move(mergedHandle));// append one handle with 5 merged grids
    EXPECT_TRUE(mergedHandle.empty());
    EXPECT_EQ(6u, handles1.size());
    //timer.restart("merging 10 host grids");
    mergedHandle = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(handles1);
    EXPECT_EQ(size1 + size2, mergedHandle.bufferSize());
    EXPECT_TRUE(mergedHandle.data());
    gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(10u, gridData->mGridCount);
    EXPECT_EQ( 0u, gridData->mGridIndex);
    EXPECT_EQ(handles1[0].bufferSize(), gridData->mGridSize);

    //timer.restart("splitting host grids");
    auto splitHandles = nanovdb::splitGrids(mergedHandle);
    //timer.restart("unit-test split grids");
    EXPECT_EQ(10u, splitHandles.size());
    for (int i=0; i<5; ++i){
        EXPECT_EQ(handles1[i].bufferSize(), splitHandles[i].bufferSize());
        gridData = splitHandles[i].gridData();
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    for (int i=5; i<10; ++i){
        EXPECT_EQ(handles2[i-5].bufferSize(), splitHandles[i].bufferSize());
        gridData = splitHandles[i].gridData();
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    //timer.stop();
}//  mergeSplitGrids

TEST(TestNanoVDBCUDA, mergeSplitDeviceGrids)
{
    using HandleT = nanovdb::GridHandle<SingleBufT>;
    const cudaStream_t stream = 0;
    size_t size = 0;
    std::vector<HandleT> handles;
    std::vector<std::string> gridNames;
    //nanovdb::util::Timer timer("create 10 host grids and copy them to the device");
    for (int radius = 100; radius<200; radius += 10) {
        gridNames.emplace_back("sphere_" + std::to_string(radius));
        auto hostHandle = nanovdb::tools::createLevelSetSphere(radius,nanovdb::Vec3d(0),1,3,
                                                               nanovdb::Vec3d(0), gridNames.back());
        EXPECT_FALSE(hostHandle.isPadded());
        size += hostHandle.bufferSize();
        handles.push_back(nanovdb::cuda::copyTo<SingleBufT>(hostHandle, stream));
    }
    EXPECT_EQ(10u, handles.size());
    //timer.restart("merging device grids");
    auto mergedHandle = nanovdb::cuda::mergeGridHandles<SingleBufT, std::vector>(handles, stream);
    EXPECT_EQ(size, mergedHandle.bufferSize());
    EXPECT_TRUE(mergedHandle.deviceData());
    EXPECT_FALSE(mergedHandle.isPadded());
    //timer.restart("copy grids to host");
    auto mergedHost = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(mergedHandle, stream);
    EXPECT_TRUE(mergedHost.data());
    EXPECT_FALSE(mergedHost.isPadded());
    auto *gridData = mergedHost.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(10u, gridData->mGridCount);
    EXPECT_EQ(0u, gridData->mGridIndex);
    //timer.restart("unit-test host grids");
    for (uint32_t i=0; i<10; ++i) {
        gridData = mergedHost.gridData(i);
        EXPECT_TRUE(gridData);
        EXPECT_EQ(i, gridData->mGridIndex);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    //timer.restart("splitting device grids");
    auto splitHandles = nanovdb::cuda::splitGridHandles<SingleBufT, std::vector>(mergedHandle, stream);
    //timer.restart("unit-test split grids");
    EXPECT_EQ(10u, splitHandles.size());
    for (uint32_t i=0u; i<10u; ++i) {
        EXPECT_EQ(handles[i].bufferSize(), splitHandles[i].bufferSize());
        EXPECT_FALSE(splitHandles[i].isPadded());
        EXPECT_TRUE(splitHandles[i].deviceData());
        auto splitHost = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(splitHandles[i], stream);
        gridData = splitHost.gridData();
        EXPECT_TRUE(gridData);
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    //timer.stop();
}//  mergeSplitDeviceGrids

TEST(TestNanoVDBCUDA, voxelSetsToDeviceGrids)
{
    nanovdb::Coord coords[3] = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8), nanovdb::Coord(-5,0,7)}, *d_coords = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coords, sizeof(coords)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords, coords, sizeof(coords), cudaMemcpyHostToDevice));
    std::vector<std::tuple<const nanovdb::Coord* const, size_t, double>> voxelSets{{d_coords, 2, 1.0}, {d_coords + 2, 1, 0.5}};
    auto devHandle = nanovdb::tools::cuda::voxelsToGrid<float, const nanovdb::Coord*, SingleBufT>(voxelSets);
    ASSERT_EQ(2u, devHandle.gridCount());
    auto handle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHandle, cudaStream_t(0));
    EXPECT_EQ(2u, handle.grid<float>(0)->activeVoxelCount());
    EXPECT_EQ(1u, handle.grid<float>(1)->activeVoxelCount());
    EXPECT_EQ(1u, handle.gridData(1)->mGridIndex);
    EXPECT_EQ(2u, handle.gridData(1)->mGridCount);
    ASSERT_EQ(cudaSuccess, cudaFree(d_coords));
}// voxelSetsToDeviceGrids
TEST(TestNanoVDBCUDA, mergePointGridsAligned)
{
    // Voxel8 points are 3 bytes each, so an unpadded blind-data payload would
    // leave the first grid's size misaligned and the second grid in the merged
    // buffer at a misaligned address.
    using nanovdb::Vec3f;
    using nanovdb::Point;
    using nanovdb::PointType;
    const std::vector<Vec3f> pointsA = {Vec3f(0.5f, 1.5f, 2.5f), Vec3f(3.0f, 3.0f, 3.0f)}, pointsB = {Vec3f(-1.0f, -2.0f, -3.0f)};
    Vec3f *d_ptsA = nullptr, *d_ptsB = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_ptsA, pointsA.size() * sizeof(Vec3f)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_ptsB, pointsB.size() * sizeof(Vec3f)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_ptsA, pointsA.data(), pointsA.size() * sizeof(Vec3f), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_ptsB, pointsB.data(), pointsB.size() * sizeof(Vec3f), cudaMemcpyHostToDevice));

    std::vector<nanovdb::GridHandle<SingleBufT>> handles;
    for (auto [d_pts, count] : {std::make_pair(d_ptsA, pointsA.size()), std::make_pair(d_ptsB, pointsB.size())}) {
        nanovdb::tools::cuda::PointsToGrid<Point> converter;
        converter.setPointType(PointType::Voxel8);
        handles.push_back(converter.getHandle<Vec3f*, SingleBufT>(d_pts, count, SingleBufT()));
        EXPECT_EQ(0u, handles.back().gridSize(0) % NANOVDB_DATA_ALIGNMENT);
    }
    auto devMerged = nanovdb::cuda::mergeGridHandles<SingleBufT, std::vector>(handles, cudaStream_t(0));
    ASSERT_EQ(2u, devMerged.gridCount());
    EXPECT_EQ(0u, devMerged.gridSize(0) % NANOVDB_DATA_ALIGNMENT);
    auto merged = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devMerged, cudaStream_t(0));
    EXPECT_EQ(1u, merged.gridData(1)->mGridIndex);
    EXPECT_EQ(2u, merged.gridData(1)->mGridCount);
    ASSERT_NE(merged.grid<Point>(0), nullptr);
    ASSERT_NE(merged.grid<Point>(1), nullptr);
    EXPECT_EQ(2u, merged.grid<Point>(0)->pointCount());
    EXPECT_EQ(1u, merged.grid<Point>(1)->pointCount());
    ASSERT_EQ(cudaSuccess, cudaFree(d_ptsA));
    ASSERT_EQ(cudaSuccess, cudaFree(d_ptsB));
}//  mergePointGridsAligned

TEST(TestNanoVDBCUDA, mergeSplitPinnedGrids)
{
    // Host-accessible single-space handles merge and split on the host, with
    // or without a pool, and match the HostBuffer result byte for byte.
    using PinBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>;
    std::vector<nanovdb::GridHandle<>> hosts;
    std::vector<nanovdb::GridHandle<PinBufT>> pinned;
    for (int radius = 100; radius < 130; radius += 10) {
        hosts.push_back(nanovdb::tools::createLevelSetSphere(radius, nanovdb::Vec3d(0), 1, 3, nanovdb::Vec3d(0), "sphere_" + std::to_string(radius)));
        pinned.push_back(hosts.back().copy<PinBufT>());// host-to-host deep copy into pinned storage
    }
    auto hostMerged = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(hosts);
    ASSERT_EQ(3u, hostMerged.gridCount());

    auto merged = nanovdb::mergeGrids<PinBufT, std::vector>(pinned);// no pool: a default-constructed PinnedResource
    ASSERT_EQ(3u, merged.gridCount());
    ASSERT_EQ(hostMerged.bufferSize(), merged.bufferSize());
    EXPECT_EQ(0, std::memcmp(merged.data(), hostMerged.data(), hostMerged.bufferSize()));
    EXPECT_NE(merged.grid<float>(2), nullptr);
    EXPECT_TRUE(nanovdb::tools::validateChecksum(merged.gridData(2), nanovdb::CheckMode::Default));

    PinBufT pool(16, nanovdb::cuda::noInit);
    auto merged2 = nanovdb::mergeGrids<PinBufT, std::vector>(pinned, &pool);
    ASSERT_EQ(hostMerged.bufferSize(), merged2.bufferSize());
    EXPECT_EQ(0, std::memcmp(merged2.data(), hostMerged.data(), hostMerged.bufferSize()));

    auto parts = nanovdb::splitGrids<PinBufT, std::vector>(merged);
    ASSERT_EQ(3u, parts.size());
    for (size_t i = 0; i < parts.size(); ++i) {
        ASSERT_EQ(hosts[i].bufferSize(), parts[i].bufferSize());
        EXPECT_EQ(0, std::memcmp(parts[i].data(), hosts[i].data(), hosts[i].bufferSize()));
        EXPECT_NE(parts[i].grid<float>(), nullptr);
    }

    // a buffer over a ResourceRef is not default-constructible, so merge and split must allocate through the pool
    using RefT = nanovdb::cuda::ResourceRef<nanovdb::cuda::PinnedResource>;
    using RefBufT = nanovdb::cuda::Buffer<std::byte, RefT>;
    nanovdb::cuda::PinnedResource resource;
    RefBufT refPool(RefT(resource), 16, nanovdb::cuda::noInit);
    std::vector<nanovdb::GridHandle<RefBufT>> refs;
    for (auto& h : hosts) {
        RefBufT buffer(RefT(resource), h.bufferSize(), nanovdb::cuda::noInit);
        std::memcpy(buffer.data(), h.data(), h.bufferSize());
        refs.emplace_back(std::move(buffer));
    }
    auto refMerged = nanovdb::mergeGrids<RefBufT, std::vector>(refs, &refPool);
    ASSERT_EQ(hostMerged.bufferSize(), refMerged.bufferSize());
    EXPECT_EQ(0, std::memcmp(refMerged.data(), hostMerged.data(), hostMerged.bufferSize()));
    auto refParts = nanovdb::splitGrids<RefBufT, std::vector>(refMerged, &refPool);
    ASSERT_EQ(3u, refParts.size());
    for (size_t i = 0; i < refParts.size(); ++i) {
        ASSERT_EQ(hosts[i].bufferSize(), refParts[i].bufferSize());
        EXPECT_EQ(0, std::memcmp(refParts[i].data(), hosts[i].data(), hosts[i].bufferSize()));
    }
}//  mergeSplitPinnedGrids

// make -j 4 testNanoVDB && ./unittest/testNanoVDB --gtest_filter="*Cuda*" --gtest_break_on_failure
TEST(TestNanoVDBCUDA, CudaIndexGridToGrid_basic)
{
    using BufferT = nanovdb::HostBuffer;
    const float value = 1.23456f, backgroud = 1.0f;
    const nanovdb::Coord ijk(1,2,3);
    nanovdb::GridHandle<BufferT> floatHdl;
    nanovdb::FloatGrid *floatGrid = nullptr;
    //nanovdb::util::Timer timer;
    {// create float grid with one active voxel
        nanovdb::tools::build::Grid<float> grid(backgroud);
        auto srcAcc = grid.getAccessor();
        srcAcc.setValue(ijk, value);
        auto nodeCount = grid.nodeCount();
        EXPECT_EQ(1u, nodeCount[0]);
        EXPECT_EQ(1u, nodeCount[1]);
        EXPECT_EQ(1u, nodeCount[2]);
        EXPECT_EQ(value, srcAcc.getValue(ijk));
        EXPECT_EQ(value, srcAcc.getValue(1,2,3));
        //timer.start("Create FloatGrid on CPU");
        floatHdl = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>, float, BufferT>(grid);
        EXPECT_TRUE(floatHdl);
        floatGrid = floatHdl.grid<float>();
        EXPECT_TRUE(floatGrid);
        EXPECT_EQ(ijk, floatGrid->indexBBox()[0]);
        EXPECT_EQ(ijk, floatGrid->indexBBox()[1]);
        auto acc = floatGrid->getAccessor();
        EXPECT_EQ(backgroud, acc.getValue(nanovdb::Coord(-1)));
        EXPECT_FALSE(acc.isActive(nanovdb::Coord(-1)));
        EXPECT_EQ(backgroud, acc.getValue(nanovdb::Coord(8)));
        EXPECT_FALSE(acc.isActive(nanovdb::Coord(8)));
        EXPECT_EQ(backgroud, acc.getValue(nanovdb::Coord(0)));
        EXPECT_FALSE(acc.isActive(nanovdb::Coord(0)));
        EXPECT_EQ(value, acc.getValue(ijk));
        EXPECT_TRUE(acc.isActive(ijk));
    }
    //timer.restart("Create IndexGrid on CPU");
    auto idxHdl = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, nanovdb::ValueIndex, BufferT>(*floatGrid, 0u, false, false, 1);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    auto devIdxHdl = nanovdb::cuda::copyTo<SingleBufT>(idxHdl, cudaStream_t(0));
    EXPECT_TRUE(devIdxHdl.deviceGrid<nanovdb::ValueIndex>());
    auto *idxGrid = idxHdl.grid<nanovdb::ValueIndex>();
    EXPECT_TRUE(idxGrid);
    //timer.restart("Create value list on CPU");
    EXPECT_EQ(1u + 512u, idxGrid->valueCount());// background + 512 values in one leaf node
    float *values = new float[idxGrid->valueCount()], *d_values = nullptr;
    values[0] = backgroud;
    const float *q = floatGrid->tree().getFirstLeaf()->mValues;
    for (float *p=values+1, *e=p+512;p!=e; ++p) *p = *q++;
    //timer.restart("Allocate and copy values from CPU to GPU");
    cudaCheck(cudaMalloc((void**)&d_values, idxGrid->valueCount()*sizeof(float)));
    EXPECT_TRUE(d_values);
    cudaCheck(cudaMemcpy(d_values, values, idxGrid->valueCount()*sizeof(float), cudaMemcpyHostToDevice));
    EXPECT_FALSE(devIdxHdl.deviceGrid<float>());
    auto *d_idxGrid = devIdxHdl.deviceGrid<nanovdb::ValueIndex>();
    EXPECT_TRUE(d_idxGrid);
    //timer.restart("Call CudaIndexToGrid");
    auto hdl = nanovdb::tools::cuda::indexToGrid<float>(d_idxGrid, d_values, SingleBufT());
    //timer.restart("unit-test");
    EXPECT_TRUE(hdl.deviceGrid<float>());
    auto hostHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(hdl, cudaStream_t(0));
    auto *floatGrid2 = hostHdl.grid<float>();
    EXPECT_TRUE(floatGrid2);
    auto *leaf2 = floatGrid2->tree().getFirstLeaf();
    EXPECT_TRUE(leaf2);
    auto acc  = floatGrid->getAccessor();
    auto acc2 = floatGrid2->getAccessor();
    EXPECT_EQ(floatGrid->indexBBox(), floatGrid2->indexBBox());
    EXPECT_EQ(floatGrid->worldBBox(), floatGrid2->worldBBox());
    // probe background in root node
    EXPECT_EQ(backgroud, acc.getValue(nanovdb::Coord(-1)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(-1)));
    EXPECT_EQ(backgroud, acc2.getValue(nanovdb::Coord(-1)));
    EXPECT_FALSE(acc2.isActive(nanovdb::Coord(-1)));
    // probe background in upper node
    EXPECT_EQ(backgroud, acc.getValue(nanovdb::Coord(128)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(128)));
    EXPECT_EQ(backgroud, floatGrid2->tree().getValue(nanovdb::Coord(128)));
    EXPECT_EQ(backgroud, acc2.getValue(nanovdb::Coord(128)));
    EXPECT_FALSE(acc2.isActive(nanovdb::Coord(128)));
    // probe background in leaf node
    EXPECT_EQ(backgroud, acc.getValue(nanovdb::Coord(0)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(0)));
    EXPECT_EQ(backgroud, leaf2->getValue(nanovdb::Coord(0)));
    EXPECT_FALSE(leaf2->isActive(nanovdb::Coord(0)));
    EXPECT_EQ(backgroud, floatGrid2->tree().getValue(nanovdb::Coord(0)));
    EXPECT_EQ(backgroud, acc2.getValue(nanovdb::Coord(0)));
    EXPECT_FALSE(acc2.isActive(nanovdb::Coord(0)));

    EXPECT_EQ(value, acc2.getValue(ijk));
    EXPECT_TRUE(acc2.isActive(ijk));
    //timer.stop();
    cudaFree(d_values);
}//  CudaIndexGridToGrid_basic

TEST(TestNanoVDBCUDA, CudaIndexGridToGrid_ValueIndex)
{
    using BuildT = nanovdb::ValueIndex;
    using BufferT = nanovdb::HostBuffer;
    //nanovdb::util::Timer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::tools::createLevelSetSphere<float, BufferT>(100,nanovdb::Vec3d(0),1,3, nanovdb::Vec3d(0), "test");
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    //timer.restart("Create IndexGrid on CPU");
    auto idxHdl = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, BuildT, BufferT>(*floatGrid);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    auto devIdxHdl = nanovdb::cuda::copyTo<SingleBufT>(idxHdl, cudaStream_t(0));
    auto *idxGrid = idxHdl.grid<BuildT>();
    EXPECT_TRUE(idxGrid);
    //timer.restart("Create value list on CPU");
    float *values = new float[idxGrid->valueCount()], *d_values = nullptr;
    values[0] = floatGrid->tree().root().background();
    for (auto it = floatGrid->indexBBox().begin(); it; ++it) {
        EXPECT_EQ(acc.isActive(*it), idxGrid->tree().isActive(*it));
        const uint64_t idx = idxGrid->tree().getValue(*it);
        EXPECT_TRUE(idx < idxGrid->valueCount());
        values[idx] = acc.getValue(*it);
    }
    //timer.restart("Allocate and copy values from CPU to GPU");
    cudaCheck(cudaMalloc((void**)&d_values, idxGrid->valueCount()*sizeof(float)));
    cudaCheck(cudaMemcpy(d_values, values, idxGrid->valueCount()*sizeof(float), cudaMemcpyHostToDevice));
    EXPECT_FALSE(devIdxHdl.deviceGrid<float>());
    auto *d_idxGrid = devIdxHdl.deviceGrid<BuildT>();
    EXPECT_TRUE(d_idxGrid);
    //timer.restart("Call CudaIndexToGrid");
    auto hdl = nanovdb::tools::cuda::indexToGrid<float>(d_idxGrid, d_values, SingleBufT());
    //timer.restart("unit-test");
    EXPECT_TRUE(hdl.deviceGrid<float>());
    auto hostHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(hdl, cudaStream_t(0));
    auto *floatGrid2 = hostHdl.grid<float>();
    EXPECT_TRUE(floatGrid2);
    auto acc2 = floatGrid2->getAccessor();
    EXPECT_EQ(floatGrid->indexBBox(), floatGrid2->indexBBox());
    EXPECT_EQ(floatGrid->worldBBox(), floatGrid2->worldBBox());
    EXPECT_EQ(floatGrid->tree().root().background(), floatGrid2->tree().root().background());
    for (auto it = floatGrid->indexBBox().begin(); it; ++it) {
        EXPECT_EQ(acc.isActive(*it), acc2.isActive(*it));
        EXPECT_EQ(acc.getValue(*it), acc2.getValue(*it));
    }
    //timer.stop();
    cudaFree(d_values);
}//  CudaPointToGrid_ValueIndex

TEST(TestNanoVDBCUDA, CudaIndexGridToGrid_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;
    using BufferT = nanovdb::HostBuffer;
    //nanovdb::util::Timer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::tools::createLevelSetSphere<float, BufferT>(100,nanovdb::Vec3d(0),1,3, nanovdb::Vec3d(0), "test");
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    //timer.restart("Create IndexGrid on CPU");
    auto idxHdl = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, BuildT, BufferT>(*floatGrid);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    auto devIdxHdl = nanovdb::cuda::copyTo<SingleBufT>(idxHdl, cudaStream_t(0));
    auto *idxGrid = idxHdl.grid<BuildT>();
    EXPECT_TRUE(idxGrid);

    auto checkConvertedValueGrid = [&](const auto& fromOriginalData, const auto& toOriginalData) {
        using FloatType = decltype(fromOriginalData(0.0f));

        //timer.restart("Create value list on CPU");
        FloatType *values = new FloatType[idxGrid->valueCount()], *d_values = nullptr;
        values[0] = fromOriginalData(floatGrid->tree().root().background());
        for (auto it = floatGrid->indexBBox().begin(); it; ++it) {
            EXPECT_EQ(acc.isActive(*it), idxGrid->tree().isActive(*it));
            if (acc.isActive(*it)) {
                const uint64_t idx = idxGrid->tree().getValue(*it);
                EXPECT_TRUE(idx < idxGrid->valueCount());
                values[idx] = fromOriginalData(acc.getValue(*it));
            }
        }
        //timer.restart("Allocate and copy values from CPU to GPU");
        cudaCheck(cudaMalloc((void**)&d_values, idxGrid->valueCount()*sizeof(FloatType)));
        cudaCheck(cudaMemcpy(d_values, values, idxGrid->valueCount()*sizeof(FloatType), cudaMemcpyHostToDevice));
        EXPECT_FALSE(devIdxHdl.deviceGrid<FloatType>());
        auto *d_idxGrid = devIdxHdl.deviceGrid<BuildT>();
        EXPECT_TRUE(d_idxGrid);
        //timer.restart("Call CudaIndexToGrid");
        auto hdl = nanovdb::tools::cuda::indexToGrid<FloatType>(d_idxGrid, d_values, SingleBufT());
        //timer.restart("unit-test");
        EXPECT_TRUE(hdl.template deviceGrid<FloatType>());
        auto hostHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(hdl, cudaStream_t(0));
        auto *floatGrid2 = hostHdl.template grid<FloatType>();
        EXPECT_TRUE(floatGrid2);
        auto acc2 = floatGrid2->getAccessor();
        EXPECT_EQ(floatGrid->indexBBox(), floatGrid2->indexBBox());
        EXPECT_EQ(floatGrid->worldBBox(), floatGrid2->worldBBox());
        EXPECT_EQ(floatGrid->tree().root().background(), toOriginalData(floatGrid2->tree().root().background()));
        for (auto it = floatGrid->indexBBox().begin(); it; ++it) {
            EXPECT_EQ(acc.isActive(*it), acc2.isActive(*it));
            if (acc.isActive(*it)) EXPECT_EQ(acc.getValue(*it), toOriginalData(acc2.getValue(*it)));
        }
        //timer.stop();
        cudaFree(d_values);
    };// checkConvertedValueGrid lambda
    checkConvertedValueGrid([](float x) { return x; }, [](float x) { return x; });
    checkConvertedValueGrid([](double x) { return float(x); }, [](float x) {return double(x); });

    // Convert index grid to grid of Vec3fs, whereas the original float data is just stored in all components of a Vec3f.
    // This test covers code in indexToGrid that is only relevant if the size of grid data changes and does not align.
    checkConvertedValueGrid([](float x) { return nanovdb::Vec3f(x); }, [](const nanovdb::Vec3f& x) { return x[0]; });
}//  CudaPointToGrid_ValueOnIndex

TEST(TestNanoVDBCUDA, CudaSignedFloodFill)
{
    using BufferT = nanovdb::HostBuffer;
    //nanovdb::util::Timer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::tools::createLevelSetSphere<float, BufferT>(1000);
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(1003,0,0)));
    EXPECT_TRUE( acc.isActive(nanovdb::Coord(1000,0,0)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord( 997,0,0)));
    EXPECT_EQ( 3.0f, acc(1003,0,0));
    EXPECT_EQ( 0.0f, acc(1000,0,0));
    EXPECT_EQ(-3.0f, acc( 997,0,0));
    using OpT = nanovdb::SetVoxel<float>;// only set the voxel value
    acc.set<OpT>(nanovdb::Coord(1003,0,0),-1.0f);// flip sign and value of inactive voxel
    acc.set<OpT>(nanovdb::Coord( 997,0,0), 1.0f);// flip sign and value of inactive voxel
    EXPECT_EQ(-1.0f, acc(1003,0,0));
    EXPECT_EQ( 0.0f, acc(1000,0,0));
    EXPECT_EQ( 1.0f, acc( 997,0,0));
    //timer.restart("Copy FloatGrid from CPU to GPU");
    auto devHdl = nanovdb::cuda::copyTo<SingleBufT>(floatHdl, cudaStream_t(0));// CPU -> GPU
    auto *d_floatGrid = devHdl.deviceGrid<float>();
    EXPECT_TRUE(d_floatGrid);
    //timer.restart("Signed flood-fill on the GPU");
    //nanovdb::cuda::signedFloodFill(d_floatGrid, true);
    nanovdb::tools::cuda::signedFloodFill(d_floatGrid);
    //timer.restart("Copy FloatGrid from GPU to CPU");
    auto hostHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHdl, cudaStream_t(0));// GPU -> CPU
    //timer.stop();
    floatGrid = hostHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    acc = floatGrid->getAccessor();
    EXPECT_EQ( 3.0f, acc(1003,0,0));
    EXPECT_EQ( 0.0f, acc(1000,0,0));
    EXPECT_EQ(-3.0f, acc( 997,0,0));
    //EXPECT_FALSE(floatGrid->isLexicographic());
    EXPECT_TRUE(floatGrid->isBreadthFirst());
}//  CudaSignedFloodFill

// Regression test for https://github.com/AcademySoftwareFoundation/openvdb/issues/2300: builds
// two non-adjacent root children on the same z-scanline, with a background tile sandwiched
// between them, to exercise processRoot's scanline-repair pass (untested until now). Node b's
// near corner is left "outside" so the correct outcome is that the tile stays untouched - run
// under compute-sanitizer to also catch the out-of-bounds read the bug caused.
TEST(TestNanoVDBCUDA, CudaSignedFloodFillNonAdjacentRootChildren)
{
    using BufferT = nanovdb::HostBuffer;
    using RootT   = nanovdb::NanoRoot<float>;
    static const int dim = int(RootT::ChildNodeType::DIM);// = 4096, size of a root child (upper) node
    const float background = 1.0f;

    nanovdb::tools::build::Grid<float> grid(background);
    auto srcAcc = grid.getAccessor();
    // last voxel of the root child at origin (0,0,0): drives RootChild::val[1] (getLastValue); inside
    srcAcc.setValue(nanovdb::Coord(dim - 1, dim - 1, dim - 1), -1.0f);
    // far corner of the root child at origin (0,0,2*dim), explicitly active but still at the
    // background value, just to force that child node to exist without flipping its sign - a
    // single inside voxel would flood the whole (otherwise empty) node to inside during the
    // leaf/internal passes that run before processRoot, which would also flip the near corner.
    // The near corner (0,0,2*dim), which drives RootChild::val[0] (getFirstValue), thus stays at
    // the background (outside) value - the scanline must NOT be filled
    srcAcc.setValue(nanovdb::Coord(dim - 1, dim - 1, 3 * dim - 1), background);
    // explicit inactive background tile at the sandwiched root tile (0,0,dim), so the scanline
    // pass has a tile entry available to (not) touch instead of throwing "missing internal tile"
    grid.tree().root().addTile<3>(nanovdb::Coord(0, 0, dim), background, false);
    ASSERT_EQ(3u, grid.tree().root().getTableSize());// the two children plus the sandwiched tile

    auto floatHdl = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>, float, BufferT>(grid);
    ASSERT_TRUE(floatHdl);
    auto devHdl = nanovdb::cuda::copyTo<SingleBufT>(floatHdl, cudaStream_t(0));
    auto *d_floatGrid = devHdl.deviceGrid<float>();
    ASSERT_TRUE(d_floatGrid);

    nanovdb::tools::cuda::signedFloodFill(d_floatGrid);

    auto hostHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHdl, cudaStream_t(0));
    auto *floatGrid = hostHdl.grid<float>();
    ASSERT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    // node b is outside at its near corner, so the sandwiched tile must be left at background
    EXPECT_EQ(background, acc.getValue(nanovdb::Coord(0, 0, dim)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(0, 0, dim)));
    // sanity: the explicit voxels that drove the decision are unaffected
    EXPECT_EQ(-1.0f, acc.getValue(nanovdb::Coord(dim - 1, dim - 1, dim - 1)));
    EXPECT_EQ(background, acc.getValue(nanovdb::Coord(0, 0, 2 * dim)));
}// CudaSignedFloodFillNonAdjacentRootChildren

// Companion to CudaSignedFloodFillNonAdjacentRootChildren: same two non-adjacent root children
// on the same z-scanline, but node b's near corner is also "inside" this time, so the sandwiched
// tile must be filled with the inside value. This deterministically verifies the repaired
// scanline pass does the right thing, not just that it doesn't fault on garbage data.
TEST(TestNanoVDBCUDA, CudaSignedFloodFillNonAdjacentRootChildrenFilled)
{
    using BufferT = nanovdb::HostBuffer;
    using RootT   = nanovdb::NanoRoot<float>;
    static const int dim = int(RootT::ChildNodeType::DIM);// = 4096, size of a root child (upper) node
    const float background = 1.0f;

    nanovdb::tools::build::Grid<float> grid(background);
    auto srcAcc = grid.getAccessor();
    // last voxel of the root child at origin (0,0,0): drives RootChild::val[1] (getLastValue); inside
    srcAcc.setValue(nanovdb::Coord(dim - 1, dim - 1, dim - 1), -1.0f);
    // near corner of the root child at origin (0,0,2*dim): drives RootChild::val[0]
    // (getFirstValue); also inside, so both ends of the scanline are inside
    srcAcc.setValue(nanovdb::Coord(0, 0, 2 * dim), -1.0f);
    // explicit inactive background tile at the sandwiched root tile (0,0,dim), so the scanline
    // pass has a tile entry available to fill instead of throwing "missing internal tile"
    grid.tree().root().addTile<3>(nanovdb::Coord(0, 0, dim), background, false);
    ASSERT_EQ(3u, grid.tree().root().getTableSize());// the two children plus the sandwiched tile

    auto floatHdl = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>, float, BufferT>(grid);
    ASSERT_TRUE(floatHdl);
    auto devHdl = nanovdb::cuda::copyTo<SingleBufT>(floatHdl, cudaStream_t(0));
    auto *d_floatGrid = devHdl.deviceGrid<float>();
    ASSERT_TRUE(d_floatGrid);

    nanovdb::tools::cuda::signedFloodFill(d_floatGrid);

    auto hostHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHdl, cudaStream_t(0));
    auto *floatGrid = hostHdl.grid<float>();
    ASSERT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    // both ends of the scanline are inside, so the sandwiched tile must be filled to -background
    EXPECT_EQ(-background, acc.getValue(nanovdb::Coord(0, 0, dim)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(0, 0, dim)));
    // sanity: the explicit voxels that drove the decision are unaffected
    EXPECT_EQ(-1.0f, acc.getValue(nanovdb::Coord(dim - 1, dim - 1, dim - 1)));
    EXPECT_EQ(-1.0f, acc.getValue(nanovdb::Coord(0, 0, 2 * dim)));
}// CudaSignedFloodFillNonAdjacentRootChildrenFilled

// Regression test for https://github.com/AcademySoftwareFoundation/openvdb/issues/2357: flips the sign of
// every inactive leaf value and every internal-node tile of a sphere that was flood filled on the host
// (build::sdfToLevelSet), and checks that the device flood fill gives each of them the host value again
TEST(TestNanoVDBCUDA, CudaSignedFloodFillMatchesHost)
{
    using DevBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    auto refHdl = nanovdb::tools::createLevelSetSphere<float>(100.0, nanovdb::Vec3d(64.0));
    auto hdl = refHdl.copy<nanovdb::HostBuffer>();
    const auto &refTree = refHdl.grid<float>()->tree();
    auto &tree = hdl.grid<float>()->tree();

    for (auto *leaf = tree.getFirstLeaf(), *end = leaf + tree.nodeCount(0); leaf != end; ++leaf) {
        for (uint32_t n = 0; n < leaf->SIZE; ++n) {
            if (!leaf->valueMask().isOn(n)) leaf->setValueOnly(n, -leaf->getValue(n));
        }
    }
    auto flipTiles = [](auto *node, uint32_t count) {
        for (auto *end = node + count; node != end; ++node) {
            for (uint32_t n = 0; n < node->SIZE; ++n) {
                if (!node->childMask().isOn(n)) node->data()->setValue(n, -node->data()->getValue(n));
            }
        }
    };
    flipTiles(tree.getFirstLower(), tree.nodeCount(1));
    flipTiles(tree.getFirstUpper(), tree.nodeCount(2));

    auto devHdl = nanovdb::cuda::copyTo<DevBufT>(hdl, cudaStream_t(0));
    nanovdb::tools::cuda::signedFloodFill(devHdl.deviceGrid<float>());
    auto outHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHdl, cudaStream_t(0)); // synchronizes before returning
    const auto &outTree = outHdl.grid<float>()->tree();

    uint64_t wrongVoxels = 0;
    for (uint32_t i = 0; i < outTree.nodeCount(0); ++i) {
        const auto &leaf = outTree.getFirstLeaf()[i], &ref = refTree.getFirstLeaf()[i];
        for (uint32_t n = 0; n < leaf.SIZE; ++n) wrongVoxels += leaf.getValue(n) != ref.getValue(n);
    }
    auto wrongTiles = [](const auto *node, const auto *ref, uint32_t count) {
        uint64_t wrong = 0;
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t n = 0; n < node[i].SIZE; ++n) {
                wrong += !node[i].childMask().isOn(n) && node[i].data()->getValue(n) != ref[i].data()->getValue(n);
            }
        }
        return wrong;
    };
    EXPECT_EQ(0u, wrongVoxels);
    EXPECT_EQ(0u, wrongTiles(outTree.getFirstLower(), refTree.getFirstLower(), outTree.nodeCount(1)));
    EXPECT_EQ(0u, wrongTiles(outTree.getFirstUpper(), refTree.getFirstUpper(), outTree.nodeCount(2)));
}// CudaSignedFloodFillMatchesHost

TEST(TestNanoVDBCUDA, OneVoxelToGrid)
{
    using BuildT = float;
    using GridT = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 1;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

    //nanovdb::util::cuda::Timer timer("Create FloatGrid on GPU");
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    auto handle = converter.getHandle(d_coords, num_points, SingleBufT());
    cudaCheck(cudaFree(d_coords));
    //timer.stop();

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.bufferSize(), size);

    //timer.start("Copy data from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    //timer.stop();
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    GridT *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);

    //timer.start("Unit-testing grid on the CPU");
    auto acc = grid->getAccessor();
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(0,2,3)));
    EXPECT_TRUE( acc.isActive(nanovdb::Coord(1,2,3)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(1,2,4)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(2,2,3)));
    auto *leaf = acc.probeLeaf(nanovdb::Coord(1,2,3));
    EXPECT_TRUE(leaf);
    EXPECT_EQ(nanovdb::Coord(0), leaf->origin());
    EXPECT_EQ(1u, leaf->valueMask().countOn());
    EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->bbox()[1]);
    auto *lower = acc.getNode<1>();
    EXPECT_TRUE(lower);
    EXPECT_EQ(nanovdb::Coord(1,2,3), lower->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), lower->bbox()[1]);
    auto *upper = acc.getNode<2>();
    EXPECT_TRUE(upper);
    EXPECT_EQ(nanovdb::Coord(1,2,3), upper->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), upper->bbox()[1]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), acc.root().bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), acc.root().bbox()[1]);
    //timer.stop();
}// OneVoxelToGrid

TEST(TestNanoVDBCUDA, ThreePointsToGrid)
{
    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3f;
    using GridT  = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 3;
    Vec3T points[num_points] = {Vec3T(1, 0, 0),Vec3T(1, 2, 3),Vec3T(1, 2, 3)}, *d_points = nullptr;
    cudaCheck(cudaMalloc(&d_points, num_points * sizeof(Vec3T)));
    cudaCheck(cudaMemcpy(d_points, points, num_points * sizeof(Vec3T), cudaMemcpyHostToDevice));// CPU -> GPU

    //nanovdb::util::cuda::Timer timer("Create FloatGrid on GPU");
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    auto handle = converter.getHandle(d_points, num_points, SingleBufT());
    cudaCheck(cudaFree(d_points));
    //timer.stop();

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType) +
                          sizeof(nanovdb::GridBlindMetaData) +
                          nanovdb::math::AlignUp<NANOVDB_DATA_ALIGNMENT>(num_points*sizeof(Vec3T));// blind data is padded
    EXPECT_EQ(handle.bufferSize(), size);

    //timer.start("Copy data from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    //timer.stop();
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    GridT *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    EXPECT_EQ(1u, grid->blindDataCount());
    const Vec3T *blindData = grid->getBlindData<Vec3T>(0);
    EXPECT_TRUE(blindData);
    for (const Vec3T *p = blindData, *q=p+num_points, *ptr=points; p!=q; ++p) {
        EXPECT_EQ(*ptr++, *p);
    }
    //timer.start("Unit-testing grid on the CPU");
    nanovdb::PointAccessor<Vec3T, BuildT> acc(*grid);
    EXPECT_TRUE(acc);
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(0,2,3)));
    EXPECT_TRUE( acc.isActive(nanovdb::Coord(1,0,0)));
    EXPECT_TRUE( acc.isActive(nanovdb::Coord(1,2,3)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(1,2,4)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(2,2,3)));
    auto *leaf = acc.probeLeaf(nanovdb::Coord(1,2,3));
    EXPECT_TRUE(leaf);
    EXPECT_EQ(nanovdb::Coord(0), leaf->origin());
    EXPECT_EQ(2u, leaf->valueMask().countOn());
    EXPECT_EQ(nanovdb::Coord(1,0,0), leaf->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), leaf->bbox()[1]);
    nanovdb::CoordBBox bbox(nanovdb::Coord(0), nanovdb::Coord(7));
    for (auto it = bbox.begin(); it; ++it) {
        //std::cerr << *it << " offset = " << leaf->CoordToOffset(*it) << " value = " << leaf->getValue(*it) << std::endl;
        if (*it < nanovdb::Coord(1,0,0)) {
            EXPECT_EQ(0u, leaf->getValue(*it));
        } else if (*it < nanovdb::Coord(1,2,3)) {
            EXPECT_EQ(1u, leaf->getValue(*it));
        } else {
            EXPECT_EQ(3u, leaf->getValue(*it));
        }
    }
    const Vec3T *start=nullptr, *stop=nullptr;

    EXPECT_EQ(0u, acc.voxelPoints(nanovdb::Coord(0,0,0), start, stop));
    EXPECT_FALSE(start);
    EXPECT_FALSE(stop);

    EXPECT_EQ(1u, acc.voxelPoints(nanovdb::Coord(1,0,0), start, stop));
    EXPECT_TRUE(start);
    EXPECT_TRUE(stop);
    EXPECT_LT(start, stop);
    EXPECT_EQ(Vec3T(1, 0, 0), start[0]);

    EXPECT_EQ(2u, acc.voxelPoints(nanovdb::Coord(1,2,3), start, stop));
    EXPECT_TRUE(start);
    EXPECT_TRUE(stop);
    EXPECT_LT(start, stop);
    EXPECT_EQ(Vec3T(1, 2, 3), start[0]);
    EXPECT_EQ(Vec3T(1, 2, 3), start[1]);

    auto *lower = acc.getNode<1>();
    EXPECT_TRUE(lower);
    EXPECT_EQ(nanovdb::Coord(1,0,0), lower->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), lower->bbox()[1]);
    auto *upper = acc.getNode<2>();
    EXPECT_TRUE(upper);
    EXPECT_EQ(nanovdb::Coord(1,0,0), upper->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), upper->bbox()[1]);
    EXPECT_EQ(nanovdb::Coord(1,0,0), acc.root().bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,2,3), acc.root().bbox()[1]);
    //timer.stop();
}// ThreePointsToGrid

TEST(TestNanoVDBCUDA, EightVoxelsToFloatGrid)
{
    using BuildT = float;
    using GridT = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 8;
    //std::cerr << nanovdb::NanoLeaf<uint32_t>::CoordToOffset(nanovdb::Coord( 1, 1, 1)) << std::endl;
    //std::cerr << nanovdb::NanoLeaf<uint32_t>::CoordToOffset(nanovdb::Coord(-7, 1, 1)) << std::endl;
    //std::cerr << nanovdb::NanoLeaf<uint32_t>::CoordToOffset(nanovdb::Coord( 1,-7, 1)) << std::endl;
    //std::cerr << nanovdb::NanoLeaf<uint32_t>::CoordToOffset(nanovdb::Coord( 1,-7, 1)) << std::endl;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord( 1, 1, 1),
                                         nanovdb::Coord(-7, 1, 1),
                                         nanovdb::Coord( 1,-7, 1),
                                         nanovdb::Coord( 1, 1,-7),
                                         nanovdb::Coord(-7,-7, 1),
                                         nanovdb::Coord(-7, 1,-7),
                                         nanovdb::Coord( 1,-7,-7),
                                         nanovdb::Coord(-7,-7,-7)}, *d_coords = nullptr;
    for (int i=0; i<8; ++i) EXPECT_EQ(73u, nanovdb::NanoLeaf<uint32_t>::CoordToOffset(coords[i]));
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

    //nanovdb::util::cuda::Timer timer("Create FloatGrid on GPU");
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    auto handle = converter.getHandle(d_coords, num_points, SingleBufT());
    //timer.stop();
    cudaCheck(cudaFree(d_coords));

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(8) +
                          8*sizeof(GridT::UpperNodeType) +
                          8*sizeof(GridT::LowerNodeType) +
                          8*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.bufferSize(), size);

    //timer.start("Copy data from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    //timer.stop();
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    GridT *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());

    //timer.start("Unit-testing grid on the CPU");
    auto acc = grid->getAccessor();
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(0,2,3)));
    EXPECT_TRUE( acc.isActive(nanovdb::Coord(1,1,1)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(1,2,4)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(2,2,3)));
    auto *leaf = acc.probeLeaf(nanovdb::Coord(1,0,0));
    EXPECT_TRUE(leaf);
    EXPECT_EQ(nanovdb::Coord(0), leaf->origin());
    EXPECT_EQ(1u, leaf->valueMask().countOn());
    EXPECT_EQ(nanovdb::Coord( 1, 1, 1), leaf->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord( 1, 1, 1), leaf->bbox()[1]);
    auto *lower = acc.getNode<1>();
    EXPECT_TRUE(lower);
    EXPECT_EQ(nanovdb::Coord(1,1,1), lower->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,1,1), lower->bbox()[1]);
    auto *upper = acc.getNode<2>();
    EXPECT_TRUE(upper);
    EXPECT_EQ(nanovdb::Coord(1,1,1), upper->bbox()[0]);
    EXPECT_EQ(nanovdb::Coord(1,1,1), upper->bbox()[1]);
    EXPECT_EQ(nanovdb::Coord(-7,-7,-7), acc.root().bbox()[0]);
    EXPECT_EQ(nanovdb::Coord( 1, 1, 1), acc.root().bbox()[1]);
    //timer.stop();
}// EightVoxelsToFloatGrid

TEST(TestNanoVDBCUDA, Random_CudaPointsToGrid_World64)
{
    using BuildT = nanovdb::Point;//uint32_t;
    using Vec3T = nanovdb::Vec3d;
    //nanovdb::util::Timer timer;
    const size_t pointCount = 1 << 20;// 1048576
    std::vector<Vec3T> points;
    //generate random points
    points.reserve(pointCount);
    std::srand(98765);
    const int max = 512, min = -max;
    auto op = [&](){return rand() % (max - min) + min;};
    //timer.start("Creating "+std::to_string(pointCount)+" random points on the CPU");
    while (points.size() < pointCount) points.emplace_back(op(), op(), op());
    //timer.stop();
    EXPECT_EQ(pointCount, points.size());
    Vec3T* d_points;
    const size_t pointSize = points.size() * sizeof(Vec3T);
    //std::cerr << "Point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, points.data(), pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    const double voxelSize = 8.0;
    //timer.start("Building grid on GPU from "+std::to_string(points.size())+" points");
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(voxelSize);// unit map
    //converter.setVerbose();
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_GT(maxPointsPerVoxel, 0u);
    EXPECT_LT(maxPointsPerLeaf, 1024u);
    EXPECT_LE(maxPointsPerVoxel, maxPointsPerLeaf);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    EXPECT_EQ(nanovdb::Vec3d(voxelSize), grid->voxelSize());
    EXPECT_TRUE(nanovdb::CoordBBox::createCube(min, max-1).isInside(grid->indexBBox()));
    //std::cerr << grid->indexBBox() << std::endl;
    EXPECT_STREQ("World64: Vec3<double> point coordinates in world space", grid->blindMetaData(0).mName);
    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(points,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<Vec3T, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const Vec3T *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(points[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel);
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) test = (points[i] - start[j]).length() < 1e-9;
            EXPECT_TRUE(test);
        }
    });

    //timer.stop();
}// Random_CudaPointsToGrid_World64


TEST(TestNanoVDBCUDA, Large_CudaPointsToGrid_World64)
{
    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3d;
    //nanovdb::util::Timer timer;
    const size_t pointCount = 1 << 20;// 1048576
    std::vector<Vec3T> points;
    //generate random points
    points.reserve(pointCount);
    std::srand(98765);
    const int max = 512, min = -max;
    auto op = [&](){return rand() % (max - min) + min;};
    //timer.start("Creating "+std::to_string(pointCount)+" random points on the CPU");
    while (points.size() < pointCount) points.emplace_back(op(), op(), op());
    //timer.stop();
    EXPECT_EQ(pointCount, points.size());
    Vec3T* d_points;
    const size_t pointSize = points.size() * sizeof(Vec3T);
    //std::cerr << "Point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, points.data(), pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    const double voxelSize = 8.0;
    //timer.start("Building grid on GPU from "+std::to_string(points.size())+" points");
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(voxelSize);// fixed voxel size
    //converter.setVerbose();
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_GT(maxPointsPerVoxel, 0u);
    EXPECT_LT(maxPointsPerLeaf, 1024u);
    EXPECT_LE(maxPointsPerVoxel, maxPointsPerLeaf);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    EXPECT_EQ(nanovdb::Vec3d(voxelSize), grid->voxelSize());
    EXPECT_EQ(pointCount, grid->pointCount());
    EXPECT_TRUE(nanovdb::CoordBBox::createCube(min, max-1).isInside(grid->indexBBox()));
    //std::cerr << grid->indexBBox() << std::endl;

    EXPECT_STREQ("World64: Vec3<double> point coordinates in world space", grid->blindMetaData(0).mName);
    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(points,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<Vec3T, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const Vec3T *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(points[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel);
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) {
                const nanovdb::Vec3d &xyz = start[j];
                test = nanovdb::math::isApproxZero<double>( (points[i] - xyz).lengthSqr() );
            }
            EXPECT_TRUE(test);
        }
    });

    //timer.stop();
}// Large_CudaPointsToGrid_World64

TEST(TestNanoVDBCUDA, Large_CudaPointsToGrid_World64_density)
{// unlike the previous unit-test this one selects the dx to match a specefic point density
    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3d;
    //nanovdb::util::Timer timer;
    const size_t pointCount = 1 << 20;// 1048576
    std::vector<Vec3T> points;
    //generate random points
    points.reserve(pointCount);
    std::srand(98765);
    const int max = 512, min = -max;
    auto op = [&](){return rand() % (max - min) + min;};
    //timer.start("Creating "+std::to_string(pointCount)+" random points on the CPU");
    while (points.size() < pointCount) points.emplace_back(op(), op(), op());
    //timer.stop();
    EXPECT_EQ(pointCount, points.size());
    Vec3T* d_points;
    const size_t pointSize = points.size() * sizeof(Vec3T);
    //std::cerr << "Point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, points.data(), pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    const int targetPointsPerVoxel = 60, tolerance = 1;
    //timer.start("Building grid on GPU from "+std::to_string(points.size())+" points");
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(targetPointsPerVoxel, tolerance);// fixed density
    //converter.setVerbose(2);
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_NEAR(maxPointsPerVoxel, targetPointsPerVoxel, tolerance);
    EXPECT_LE(maxPointsPerLeaf, targetPointsPerVoxel*512);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    //EXPECT_EQ(nanovdb::Vec3d(voxelSize), grid->voxelSize());
    EXPECT_EQ(pointCount, grid->pointCount());
    EXPECT_TRUE(nanovdb::CoordBBox::createCube(min, max-1).isInside(grid->indexBBox()));
    //std::cerr << grid->indexBBox() << std::endl;

    EXPECT_STREQ("World64: Vec3<double> point coordinates in world space", grid->blindMetaData(0).mName);
    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel + tolerance);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(points,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<Vec3T, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const Vec3T *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(points[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel + tolerance);
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) {
                const nanovdb::Vec3d &xyz = start[j];
                test = nanovdb::math::isApproxZero<double>( (points[i] - xyz).lengthSqr() );
            }
            EXPECT_TRUE(test);
        }
    });

    //timer.stop();
}// Large_CudaPointsToGrid_World64_density

TEST(TestNanoVDBCUDA, Sphere_CudaPointsToGrid_World32)
{
    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3f;

    //nanovdb::util::Timer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::tools::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    //timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    nanovdb::PointAccessor<Vec3T> acc2(*pointGrid);
    EXPECT_TRUE(acc2);
    const Vec3T *begin, *end;
    const size_t pointCount = acc2.gridPoints(begin, end);
    EXPECT_TRUE(begin);
    EXPECT_TRUE(end);
    EXPECT_LT(begin, end);

    const size_t pointSize = pointCount * sizeof(Vec3T);
    //std::cerr << "Point count = " << pointCount << ", point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //std::cerr << "Upper count: " << pointGrid->tree().nodeCount(2) << ", lower count: " << pointGrid->tree().nodeCount(1)
    //          << ", leaf count: " << pointGrid->tree().nodeCount(0) << ", voxelSize = " << pointGrid->voxelSize()[0] << std::endl;

    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    Vec3T* d_points;
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, begin, pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    //timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_GT(maxPointsPerVoxel, 0u);
    EXPECT_LT(maxPointsPerLeaf, 1024u);
    EXPECT_LE(maxPointsPerVoxel, maxPointsPerLeaf);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(pointGrid->voxelSize(), grid->voxelSize());
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    //std::cerr << grid->indexBBox() << std::endl;

    EXPECT_STREQ("World32: Vec3<float> point coordinates in world space", grid->blindMetaData(0).mName);

    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(0u, pointCount, 1u,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<Vec3T, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const Vec3T *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(begin[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel);
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) {
                const nanovdb::Vec3f &xyz = start[j];
                test = (begin[i] - xyz).length() < 1e-9;
            }
            EXPECT_TRUE(test);
        }
    });

    //timer.stop();
}// Sphere_CudaPointsToGrid_World32

TEST(TestNanoVDBCUDA, Sphere_CudaPointsToGrid_Voxel32)
{
    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3f;

    //nanovdb::util::Timer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::tools::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    //timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    nanovdb::PointAccessor<Vec3T, uint32_t> acc2(*pointGrid);
    EXPECT_TRUE(acc2);
    const Vec3T *begin, *end;
    const size_t pointCount = acc2.gridPoints(begin, end);
    EXPECT_TRUE(begin);
    EXPECT_TRUE(end);
    EXPECT_LT(begin, end);

    const size_t pointSize = pointCount * sizeof(Vec3T);
    //std::cerr << "Point count = " << pointCount << ", point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //std::cerr << "Upper count: " << pointGrid->tree().nodeCount(2) << ", lower count: " << pointGrid->tree().nodeCount(1)
    //          << ", leaf count: " << pointGrid->tree().nodeCount(0) << ", voxelSize = " << pointGrid->voxelSize()[0] << std::endl;

    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    Vec3T* d_points;
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, begin, pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    //timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    converter.setPointType(nanovdb::PointType::Voxel32);
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_GT(maxPointsPerVoxel, 0u);
    EXPECT_LT(maxPointsPerLeaf, 1024u);
    EXPECT_LE(maxPointsPerVoxel, maxPointsPerLeaf);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(pointGrid->voxelSize(), grid->voxelSize());
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    //std::cerr << grid->indexBBox() << std::endl;

    EXPECT_STREQ("Voxel32: Vec3<float> point coordinates in voxel space", grid->blindMetaData(0).mName);

    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(0u, pointCount, 1u,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<Vec3T, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const Vec3T *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(begin[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel);
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) {
                const nanovdb::Vec3f voxel = start[j];// local coordinates relative to voxel
                EXPECT_GE(voxel[0], -0.5f);
                EXPECT_GE(voxel[1], -0.5f);
                EXPECT_GE(voxel[2], -0.5f);
                EXPECT_LE(voxel[0],  0.5f);
                EXPECT_LE(voxel[1],  0.5f);
                EXPECT_LE(voxel[2],  0.5f);
                test = (begin[i] - Vec3T(nanovdb::voxelToWorld(voxel, ijk, grid->map()))).length() < 1e-9;
            }
            EXPECT_TRUE(test);
        }
    });

    //timer.stop();
}// Sphere_CudaPointsToGrid_Voxel32

TEST(TestNanoVDBCUDA, Sphere_CudaPointsToGrid_Voxel16)
{
    EXPECT_EQ(6u, sizeof(nanovdb::Vec3u16));
    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3f;

    //nanovdb::util::Timer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::tools::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    //timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    nanovdb::PointAccessor<Vec3T, uint32_t> acc2(*pointGrid);
    EXPECT_TRUE(acc2);
    const Vec3T *begin, *end;
    const size_t pointCount = acc2.gridPoints(begin, end);
    EXPECT_TRUE(begin);
    EXPECT_TRUE(end);
    EXPECT_LT(begin, end);

    const size_t pointSize = pointCount * sizeof(Vec3T);
    //std::cerr << "Point count = " << pointCount << ", point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //std::cerr << "Upper count: " << pointGrid->tree().nodeCount(2) << ", lower count: " << pointGrid->tree().nodeCount(1)
    //          << ", leaf count: " << pointGrid->tree().nodeCount(0) << ", voxelSize = " << pointGrid->voxelSize()[0] << std::endl;

    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    Vec3T* d_points;
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, begin, pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    //timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    converter.setPointType(nanovdb::PointType::Voxel16);
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_GT(maxPointsPerVoxel, 0u);
    EXPECT_LT(maxPointsPerLeaf, 1024u);
    EXPECT_LE(maxPointsPerVoxel, maxPointsPerLeaf);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(pointGrid->voxelSize(), grid->voxelSize());
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    //std::cerr << grid->indexBBox() << std::endl;

    EXPECT_STREQ("Voxel16: Vec3<uint16_t> point coordinates in voxel space", grid->blindMetaData(0).mName);

    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(0u, pointCount, 1u,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<nanovdb::Vec3u16, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const nanovdb::Vec3u16 *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(begin[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel);
            // 16-bit offsets truncate, so each axis can be off by up to one step of
            // voxelSize/65535 = 7.6e-6, i.e. a distance of up to sqrt(3) * 7.6e-6 = 1.32e-5.
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) {
                test = (begin[i] - Vec3T(nanovdb::voxelToWorld(start[j], ijk, grid->map()))).length() < 2e-5;
            }
            EXPECT_TRUE(test);
        }
    });

    //timer.stop();
}// Sphere_CudaPointsToGrid_Voxel16

TEST(TestNanoVDBCUDA, Sphere_CudaPointsToGrid_Voxel8)
{
    EXPECT_EQ(3u, sizeof(nanovdb::Vec3u8));

    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3f;

    //nanovdb::util::Timer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::tools::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    //timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    //std::cerr << "nanovdb::bbox = " << pointGrid->indexBBox() << " voxel count = " << pointGrid->activeVoxelCount() << std::endl;
    nanovdb::PointAccessor<Vec3T, uint32_t> acc2(*pointGrid);
    EXPECT_TRUE(acc2);
    const Vec3T *begin, *end;
    const size_t pointCount = acc2.gridPoints(begin, end);
    EXPECT_TRUE(begin);
    EXPECT_TRUE(end);
    EXPECT_LT(begin, end);

    const size_t pointSize = pointCount * sizeof(Vec3T);
    //std::cerr << "Point count = " << pointCount << ", point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //std::cerr << "Upper count: " << pointGrid->tree().nodeCount(2) << ", lower count: " << pointGrid->tree().nodeCount(1)
    //          << ", leaf count: " << pointGrid->tree().nodeCount(0) << ", voxelSize = " << pointGrid->voxelSize()[0] << std::endl;

    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    Vec3T* d_points;
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, begin, pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    //timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    //auto handle = nanovdb::tools::cuda::pointsToGrid(d_points, pointCount, 1.0, nanovdb::PointType::Voxel8);
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    converter.setPointType(nanovdb::PointType::Voxel8);
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_GT(maxPointsPerVoxel, 0u);
    EXPECT_LT(maxPointsPerLeaf, 1024u);
    EXPECT_LE(maxPointsPerVoxel, maxPointsPerLeaf);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(pointGrid->voxelSize(), grid->voxelSize());
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    //std::cerr << grid->indexBBox() << std::endl;

    EXPECT_STREQ("Voxel8: Vec3<uint8_t> point coordinates in voxel space", grid->blindMetaData(0).mName);

    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(0u, pointCount, 1u,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<nanovdb::Vec3u8, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const nanovdb::Vec3u8 *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(begin[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel);
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) {
                test = (begin[i] - Vec3T(nanovdb::voxelToWorld(start[j], ijk, grid->map()))).length() < 1e-2;
            }
            EXPECT_TRUE(test);
        }
    });
    //timer.stop();
}// Sphere_CudaPointsToGrid_Voxel8

TEST(TestNanoVDBCUDA, Sphere_CudaPointsToGrid_PointID)
{
    EXPECT_EQ(3u, sizeof(nanovdb::Vec3u8));

    using BuildT = nanovdb::Point;
    using Vec3T  = nanovdb::Vec3f;

    //nanovdb::util::Timer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::tools::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    //timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    //std::cerr << "nanovdb::bbox = " << pointGrid->indexBBox() << " voxel count = " << pointGrid->activeVoxelCount() << std::endl;
    nanovdb::PointAccessor<Vec3T, uint32_t> acc2(*pointGrid);
    EXPECT_TRUE(acc2);
    const Vec3T *begin, *end;
    const size_t pointCount = acc2.gridPoints(begin, end);
    EXPECT_TRUE(begin);
    EXPECT_TRUE(end);
    EXPECT_LT(begin, end);

    const size_t pointSize = pointCount * sizeof(Vec3T);
    //std::cerr << "Point count = " << pointCount << ", point footprint: " << (pointSize >> 20) << " MB" << std::endl;
    //std::cerr << "Upper count: " << pointGrid->tree().nodeCount(2) << ", lower count: " << pointGrid->tree().nodeCount(1)
    //          << ", leaf count: " << pointGrid->tree().nodeCount(0) << ", voxelSize = " << pointGrid->voxelSize()[0] << std::endl;

    //timer.start("Allocating "+std::to_string(pointSize >> 20)+" MB on the GPU");
    Vec3T* d_points;
    cudaCheck(cudaMalloc(&d_points, pointSize));
    //timer.restart("Copying points from CPU to GPU");
    cudaCheck(cudaMemcpy(d_points, begin, pointSize, cudaMemcpyHostToDevice));
    //timer.stop();

    //timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    //auto handle = nanovdb::tools::cuda::pointsToGrid(d_points, pointCount, 1.0, nanovdb::PointType::PointID);
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose(2);
    converter.setPointType(nanovdb::PointType::PointID);
    auto handle = converter.getHandle(d_points, pointCount, SingleBufT());
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.bufferSize() >> 20) << " MB" << std::endl;

    const uint32_t maxPointsPerVoxel = converter.maxPointsPerVoxel();
    const uint32_t maxPointsPerLeaf  = converter.maxPointsPerLeaf();
    EXPECT_GT(maxPointsPerVoxel, 0u);
    EXPECT_LT(maxPointsPerLeaf, 1024u);
    EXPECT_LE(maxPointsPerVoxel, maxPointsPerLeaf);
    //std::cerr << "maxPointsPerLeaf = " << maxPointsPerLeaf << " maxPointsPerVoxel = " << maxPointsPerVoxel << std::endl;

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    auto *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
    EXPECT_EQ(pointGrid->voxelSize(), grid->voxelSize());
    //EXPECT_TRUE(grid->isLexicographic());
    EXPECT_TRUE(grid->isBreadthFirst());
    //std::cerr << grid->indexBBox() << std::endl;

    EXPECT_STREQ("PointID: uint32_t indices to points", grid->blindMetaData(0).mName);

    {
        auto mgrHdl = nanovdb::createNodeManager(*grid);
        auto *mgr = mgrHdl.mgr<BuildT>();
        EXPECT_TRUE(mgr);
        for (uint32_t i=0; i<mgr->leafCount(); ++i) {
            const auto &leaf = mgr->leaf(i);
            for (int j=0; j<512; ++j) {
                EXPECT_LE(leaf.getValue(j), maxPointsPerLeaf);
                if (leaf.isActive(j)) {
                    if (j>0) {
                        EXPECT_LE(leaf.getValue(j) - leaf.getValue(j-1), maxPointsPerVoxel);
                    } else {
                        EXPECT_LE(leaf.getValue(0), maxPointsPerVoxel);
                    }
                } else if (j>0) {
                    EXPECT_EQ(leaf.getValue(j), leaf.getValue(j-1));
                } else {
                    EXPECT_EQ(leaf.getValue(0), 0u);
                }
            }// loop over voxels
        }// loop over leaf nodes
    }

    //timer.restart("Parallel unit-testing on CPU");
    nanovdb::util::forEach(0u, pointCount, 1u,[&](const nanovdb::util::Range1D &r){
        nanovdb::PointAccessor<uint32_t, BuildT> acc(*grid);
        EXPECT_TRUE(acc);
        const uint32_t *start = nullptr, *stop = nullptr;
        for (size_t i=r.begin(); i!=r.end(); ++i) {
            const nanovdb::Coord ijk = grid->worldToIndex(begin[i]).round();
            EXPECT_TRUE(acc.probeLeaf(ijk)!=nullptr);
            EXPECT_TRUE(acc.isActive(ijk));
            EXPECT_LE(acc.getValue(ijk), pointCount);
            const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
            EXPECT_TRUE(leaf);
            const auto offset = leaf->CoordToOffset(ijk);
            EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
            const uint64_t count = acc.voxelPoints(ijk, start, stop);
            EXPECT_TRUE(start);
            EXPECT_TRUE(stop);
            EXPECT_LT(start, stop);
            EXPECT_LE(count, maxPointsPerVoxel);
        }
    });

    //timer.stop();
}// Sphere_CudaPointsToGrid_PointID

TEST(TestNanoVDBCUDA, NanoGrid_Rgba8)
{
    using BuildT = nanovdb::math::Rgba8;
    using GridT  = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 1;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    auto handle = converter.getHandle(d_coords, num_points, SingleBufT());
    cudaCheck(cudaFree(d_coords));

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.bufferSize(), size);

    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));// copy to the CPU
    EXPECT_TRUE(hostHandle.data());
    auto *data = hostHandle.gridData();
    EXPECT_TRUE(data);
    GridT *grid = hostHandle.grid<BuildT>();
    EXPECT_TRUE(grid);
}// NanoGrid_Rgba8

TEST(TestNanoVDBCUDA, cudaAddBlindData)
{
    using BuildT = float;
    using GridT  = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 2;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3), nanovdb::Coord(10,20,8)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points, 1.0, SingleBufT());
    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          2*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.bufferSize(), size);
    GridT *d_grid = handle.deviceGrid<BuildT>();
    EXPECT_TRUE(d_grid);
    float *d_blind = nullptr, blind[num_points] = {1.2f, 3.0f};
    cudaCheck(cudaMalloc(&d_blind, num_points * sizeof(float)));
    cudaCheck(cudaMemcpy(d_blind, blind, num_points * sizeof(float), cudaMemcpyHostToDevice));// CPU -> GPU

    //nanovdb::util::cuda::Timer timer("cudaAddBlindData");
    auto handle2 = nanovdb::tools::cuda::addBlindData(d_grid, d_blind, num_points,
                                                     nanovdb::GridBlindDataClass::Unknown,
                                                     nanovdb::GridBlindDataSemantic::Unknown,
                                                     "", SingleBufT());
    cudaCheck(cudaFree(d_blind));
    //timer.stop();
    EXPECT_TRUE(handle2.deviceData());// grid only exists on the GPU
    EXPECT_EQ(handle2.bufferSize(), handle.bufferSize() + sizeof(nanovdb::GridBlindMetaData) + nanovdb::math::AlignUp<NANOVDB_DATA_ALIGNMENT>(num_points*sizeof(float)));

    auto hostHandle2 = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle2, cudaStream_t(0));
    EXPECT_TRUE(hostHandle2.data());
    auto *data = hostHandle2.gridData();
    EXPECT_TRUE(data);
    auto *grid2 = hostHandle2.grid<BuildT>();
    EXPECT_TRUE(grid2);
    EXPECT_EQ(nanovdb::Vec3d(1.0), grid2->voxelSize());
    EXPECT_EQ(1u, grid2->blindDataCount());
    const auto &bd2 = grid2->blindMetaData(0);
    EXPECT_EQ(num_points, bd2.mValueCount);
    EXPECT_EQ(nanovdb::GridBlindDataSemantic::Unknown, bd2.mSemantic);
    EXPECT_EQ(nanovdb::GridBlindDataClass::Unknown, bd2.mDataClass);
    EXPECT_EQ(nanovdb::GridType::Float, bd2.mDataType);
    EXPECT_STREQ("", bd2.mName);
    const float *dataPtr = bd2.getBlindData<float>();
    EXPECT_TRUE(dataPtr);
    for (size_t i=0; i<num_points; ++i) EXPECT_EQ(blind[i], dataPtr[i]);

    GridT *d_grid2 = handle2.deviceGrid<BuildT>();
    EXPECT_TRUE(d_grid2);

    nanovdb::Vec3f *d_blind2 = nullptr, blind2[num_points] = {nanovdb::Vec3f(1.2f), nanovdb::Vec3f(3.0f)};
    cudaCheck(cudaMalloc(&d_blind2, num_points * sizeof(nanovdb::Vec3f)));
    cudaCheck(cudaMemcpy(d_blind2, blind2, num_points * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));// CPU -> GPU

    auto handle3 = nanovdb::tools::cuda::addBlindData(d_grid2, d_blind2, num_points,
                                             nanovdb::GridBlindDataClass::AttributeArray,
                                             nanovdb::GridBlindDataSemantic::PointPosition,
                                             "this is a test", SingleBufT());
    cudaCheck(cudaFree(d_blind2));
    auto hostHandle3 = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle3, cudaStream_t(0));
    GridT *grid3 = hostHandle3.grid<BuildT>();
    EXPECT_TRUE(grid3);
    EXPECT_EQ(2, grid3->blindDataCount());

    const auto &bd3 = grid3->blindMetaData(0);
    EXPECT_EQ(num_points, bd3.mValueCount);
    EXPECT_EQ(nanovdb::GridBlindDataSemantic::Unknown, bd3.mSemantic);
    EXPECT_EQ(nanovdb::GridBlindDataClass::Unknown, bd3.mDataClass);
    EXPECT_EQ(nanovdb::GridType::Float, bd3.mDataType);
    EXPECT_STREQ("", bd3.mName);
    dataPtr = grid3->getBlindData<float>(0);
    EXPECT_TRUE(dataPtr);
    for (size_t i=0; i<num_points; ++i) EXPECT_EQ(blind[i], dataPtr[i]);

    const auto &bd4 = grid3->blindMetaData(1);
    EXPECT_EQ(num_points, bd4.mValueCount);
    EXPECT_EQ(nanovdb::GridBlindDataSemantic::PointPosition, bd4.mSemantic);
    EXPECT_EQ(nanovdb::GridBlindDataClass::AttributeArray, bd4.mDataClass);
    EXPECT_EQ(nanovdb::GridType::Vec3f, bd4.mDataType);
    EXPECT_STREQ("this is a test", bd4.mName);
    auto *dataPtr2 = grid3->getBlindData<nanovdb::Vec3f>(1);
    EXPECT_TRUE(dataPtr2);
    for (size_t i=0; i<num_points; ++i) EXPECT_EQ(blind2[i], dataPtr2[i]);
}// cudaAddBlindData

TEST(TestNanoVDBCUDA, cudaAddBlindDataFromMultiGridSource)
{
    using BufferT = nanovdb::HostBuffer;
    // two grids merged on the host and copied into one device buffer give source grids whose headers
    // carry a non-trivial grid index/count, which the output must not inherit
    std::vector<nanovdb::GridHandle<BufferT>> handles;
    handles.emplace_back(nanovdb::tools::createLevelSetSphere<float, BufferT>(20, nanovdb::Vec3d(0), 1, 3, nanovdb::Vec3d(0), "a"));
    handles.emplace_back(nanovdb::tools::createLevelSetSphere<float, BufferT>(30, nanovdb::Vec3d(0), 1, 3, nanovdb::Vec3d(0), "b"));
    auto merged = nanovdb::cuda::copyTo<SingleBufT>(nanovdb::mergeGrids<BufferT, std::vector>(handles), cudaStream_t(0));
    EXPECT_EQ(2u, merged.gridCount());

    const size_t num_values = 2;
    float *d_blind = nullptr, blind[num_values] = {1.2f, 3.0f};
    cudaCheck(cudaMalloc(&d_blind, num_values * sizeof(float)));
    cudaCheck(cudaMemcpy(d_blind, blind, num_values * sizeof(float), cudaMemcpyHostToDevice));
    for (uint32_t n = 0; n < merged.gridCount(); ++n) {// source header: mGridIndex=n, mGridCount=2
        auto *d_grid = merged.deviceGrid<float>(n);
        EXPECT_TRUE(d_grid);
        auto out = nanovdb::tools::cuda::addBlindData(d_grid, d_blind, num_values,
                                                      nanovdb::GridBlindDataClass::Unknown,
                                                      nanovdb::GridBlindDataSemantic::Unknown,
                                                      "", SingleBufT());
        EXPECT_EQ(1u, out.gridCount());
        auto hostOut = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(out, cudaStream_t(0));
        auto *gridData = hostOut.gridData();
        EXPECT_TRUE(gridData);
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        auto *grid = hostOut.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(1u, grid->blindDataCount());
    }
    cudaCheck(cudaFree(d_blind));
}// cudaAddBlindDataFromMultiGridSource

TEST(TestNanoVDBCUDA, cudaIndexToGridFromMultiGridSource)
{
    using BufferT = nanovdb::HostBuffer;
    auto floatHdl = nanovdb::tools::createLevelSetSphere<float, nanovdb::HostBuffer>(20, nanovdb::Vec3d(0), 1, 3, nanovdb::Vec3d(0), "sphere");
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    std::vector<nanovdb::GridHandle<BufferT>> idxHandles;
    idxHandles.emplace_back(nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, nanovdb::ValueIndex, BufferT>(*floatGrid, 0u, false, false));
    idxHandles.emplace_back(nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, nanovdb::ValueIndex, BufferT>(*floatGrid, 0u, false, false));
    auto *idxGrid = idxHandles[0].grid<nanovdb::ValueIndex>();
    EXPECT_TRUE(idxGrid);

    // host-side value list for the (identical) index grids
    std::vector<float> values(idxGrid->valueCount(), 0.0f);
    values[0] = floatGrid->tree().root().background();
    auto acc = floatGrid->getAccessor();
    for (auto it = floatGrid->indexBBox().begin(); it; ++it) {
        const uint64_t idx = idxGrid->tree().getValue(*it);
        if (idx > 0) values[idx] = acc.getValue(*it);
    }
    float *d_values = nullptr;
    cudaCheck(cudaMalloc(&d_values, values.size() * sizeof(float)));
    cudaCheck(cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto merged = nanovdb::cuda::copyTo<SingleBufT>(nanovdb::mergeGrids<BufferT, std::vector>(idxHandles), cudaStream_t(0));
    EXPECT_EQ(2u, merged.gridCount());
    for (uint32_t n = 0; n < merged.gridCount(); ++n) {// source header: mGridIndex=n, mGridCount=2
        auto *d_idxGrid = merged.deviceGrid<nanovdb::ValueIndex>(n);
        EXPECT_TRUE(d_idxGrid);
        auto out = nanovdb::tools::cuda::indexToGrid<float>(d_idxGrid, d_values, SingleBufT());
        EXPECT_EQ(1u, out.gridCount());
        auto hostOut = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(out, cudaStream_t(0));
        auto *gridData = hostOut.gridData();
        EXPECT_TRUE(gridData);
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        auto *outGrid = hostOut.grid<float>();
        EXPECT_TRUE(outGrid);
        EXPECT_EQ(floatGrid->indexBBox(), outGrid->indexBBox());
    }
    cudaCheck(cudaFree(d_values));
}// cudaIndexToGridFromMultiGridSource

TEST(TestNanoVDBCUDA, testGridHandleCopy)
{
    using PinBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>;
    auto cudaHandle = nanovdb::tools::createLevelSetSphere(100).copy<PinBufT>();
    {
        auto *floatGrid = cudaHandle.grid<float>();
        EXPECT_TRUE(floatGrid);
        auto acc = floatGrid->getAccessor();
        EXPECT_EQ( 3.0f, acc(103,0,0));
        EXPECT_EQ( 0.0f, acc(100,0,0));
        EXPECT_EQ(-3.0f, acc( 97,0,0));
    }
    auto hostHandle = cudaHandle.copy<nanovdb::HostBuffer>();
    EXPECT_TRUE(cudaHandle.grid<float>());// should be unchanged
    {
        auto *floatGrid = hostHandle.grid<float>();
        EXPECT_TRUE(floatGrid);
        auto acc = floatGrid->getAccessor();
        EXPECT_EQ( 3.0f, acc(103,0,0));
        EXPECT_EQ( 0.0f, acc(100,0,0));
        EXPECT_EQ(-3.0f, acc( 97,0,0));
    }
}// testGridHandleCopy

// make -j testNanoVDB && ./unittest/testNanoVDB --gtest_break_on_failure --gtest_filter="*compareNodeOrdering"
TEST(TestNanoVDBCUDA, compareNodeOrdering)
{
#if 0
    const int voxelCount = 2;
    Coord coords[voxelCount]={Coord(-1,0,0), Coord(0,0,0)};
#else
    const int voxelCount = 5;
    nanovdb::Coord coords[voxelCount]={
        nanovdb::Coord(0,0,0),
        nanovdb::Coord(256,0,0),
        nanovdb::Coord(0,0,8),
        nanovdb::Coord(0,-256,0),
        nanovdb::Coord(0,2,4)
    };
#endif

    {// check coordToKey and keyToCoord used in CudaPointsToGrid
        auto coordToKey = [](const nanovdb::Coord &ijk)->uint64_t{
            static constexpr int32_t offset = 1 << 30;
            return (uint64_t(uint32_t(ijk[2] + offset) >> 12)) | //       z is the lower 21 bits
                   (uint64_t(uint32_t(ijk[1] + offset) >> 12) << 21) | // y is the middle 21 bits
                   (uint64_t(uint32_t(ijk[0] + offset) >> 12) << 42); //  x is the upper 21 bits
        };
        auto keyToCoord = [](uint64_t key)->nanovdb::Coord{
            static constexpr int32_t offset = 1 << 30;
            static constexpr uint64_t MASK = (1u << 21) - 1; // used to mask out 21 lower bits
            return nanovdb::Coord((((key >> 42) & MASK) << 12) - offset, // x are the upper 21 bits
                                  (((key >> 21) & MASK) << 12) - offset, // y are the middle 21 bits
                                   ((key & MASK) << 12) - offset); // z are the lower 21 bits
        };
        using KeyT = std::pair<nanovdb::Coord, uint64_t>;
        KeyT keys[voxelCount];
        for (int i=0; i<voxelCount; ++i) {
            keys[i].second = coordToKey(coords[i]);
            keys[i].first  = keyToCoord(keys[i].second);//coords[i];
            //std::cerr << "i=" << i << " ijk=" << coords[i] << " key=" << keys[i].second << " coord=" << keyToCoord(keys[i].second) << std::endl;
        }
        std::sort(keys, keys + voxelCount, [](KeyT a, KeyT b){return a.second < b.second;});
        for (int i=1; i<voxelCount; ++i) {
            EXPECT_LE(keys[i-1].second, keys[i].second);// due to sort
            EXPECT_LE(keys[i-1].first,  keys[i].first);// if keys are sorted then by design so are the coordinates
        }
    }

    nanovdb::GridHandle<nanovdb::HostBuffer> handle1, handle2;

    {
        nanovdb::tools::build::FloatGrid grid(0.0f);
        auto acc = grid.getAccessor();
        for (int i=0; i<voxelCount; ++i) acc.setValue(coords[i], 1.0f);
        handle1 = nanovdb::tools::createNanoGrid(grid);
    }
    auto grid1 = handle1.grid<float>();
    EXPECT_TRUE(grid1);
    EXPECT_TRUE(grid1->isBreadthFirst());
    //EXPECT_FALSE(grid1->isLexicographic());

    {// check that nodes are arranged breath-first in memory
        int n2=0, n1=0, n0=0;
        for (auto it2 = grid1->tree().root().beginChild(); it2; ++it2) {
            EXPECT_EQ(grid1->tree().getFirstUpper() + n2++, &(*it2));
            for (auto it1 = it2->beginChild(); it1; ++it1) {
                EXPECT_EQ(grid1->tree().getFirstLower() + n1++, &(*it1));
                for (auto it0 = it1->beginChild(); it0; ++it0) {
                    EXPECT_EQ(grid1->tree().getFirstLeaf() + n0++, &(*it0));
                }// loop over child nodes of the lower internal node
            }// loop over child nodes of the upper internal node
        }// loop over child nodes of the root node
        EXPECT_EQ(n2, grid1->tree().nodeCount(2));
        EXPECT_EQ(n1, grid1->tree().nodeCount(1));
        EXPECT_EQ(n0, grid1->tree().nodeCount(0));
    }

    {
        nanovdb::Coord *d_coords = nullptr;
        cudaCheck(cudaMalloc(&d_coords, voxelCount * sizeof(nanovdb::Coord)));
        cudaCheck(cudaMemcpy(d_coords, coords, voxelCount * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU
#if 0
        auto cudaHandle = nanovdb::tools::cuda::voxelsToGrid<float>(d_coords, voxelCount);
#else
        auto cudaHandle = nanovdb::tools::cuda::voxelsToGrid<float>(nanovdb::make_fancy(d_coords), voxelCount, 1.0, SingleBufT());
#endif
        cudaCheck(cudaFree(d_coords));
        handle2 = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(cudaHandle, cudaStream_t(0));
    }
    auto grid2 = handle2.grid<float>();
    EXPECT_TRUE(grid2);
    EXPECT_TRUE(grid2->isBreadthFirst());
    //EXPECT_TRUE(grid2->isLexicographic());

    // Check that both grids have the expected voxel values
    for (int i=0; i<voxelCount; ++i) {
        EXPECT_EQ(1.0f, grid2->tree().getValue(coords[i]));
        EXPECT_EQ(1.0f, grid1->tree().getValue(coords[i]));
    }

    // Check that both grid have the same node counts
    for (int i=0; i<3; ++i) EXPECT_EQ(grid1->tree().nodeCount(i), grid2->tree().nodeCount(i));

     {// Check that the order of the upper nodes are identical
        auto *upper1 = grid1->tree().getFirstUpper(), *upper2 = grid2->tree().getFirstUpper();
        EXPECT_TRUE(upper1);
        EXPECT_TRUE(upper2);
        for (uint32_t i=0; i<grid1->tree().nodeCount(2); ++i) {
            //std::cerr << "#" << i << " origin(CPU)=" << upper1[i].origin() << " origin(GPU)=" << upper2[i].origin() << std::endl;
            EXPECT_EQ(upper1[i].origin(),    upper2[i].origin());
            EXPECT_EQ(upper1[i].valueMask(), upper2[i].valueMask());
            EXPECT_EQ(upper1[i].childMask(), upper2[i].childMask());
        }
    }

    {// Check that the order of the lower nodes are identical
        auto *lower1 = grid1->tree().getFirstLower(), *lower2 = grid2->tree().getFirstLower();
        EXPECT_TRUE(lower1);
        EXPECT_TRUE(lower2);
        for (uint32_t i=0; i<grid1->tree().nodeCount(1); ++i) {
            EXPECT_EQ(lower1[i].origin(),    lower2[i].origin());
            EXPECT_EQ(lower1[i].valueMask(), lower2[i].valueMask());
            EXPECT_EQ(lower1[i].childMask(), lower2[i].childMask());
        }
    }

    {// Check that the order of the leaf nodes are identical
        auto *leaf1 = grid1->tree().getFirstLeaf(), *leaf2 = grid2->tree().getFirstLeaf();
        EXPECT_TRUE(leaf1);
        EXPECT_TRUE(leaf2);
        for (uint32_t i=0; i<grid1->tree().nodeCount(0); ++i) {
            EXPECT_EQ(leaf1[i].origin(),    leaf2[i].origin());
            EXPECT_EQ(leaf1[i].valueMask(), leaf2[i].valueMask());
        }
    }

}// compareNodeOrdering

namespace {
template <typename PtrT>
void test_ptr(const PtrT ptr)
{
    using T = typename nanovdb::pointer_traits<PtrT>::element_type;
    static const bool test = nanovdb::util::is_same<float, typename nanovdb::util::remove_const<T>::type>::value;
    EXPECT_TRUE(test);
    EXPECT_EQ(sizeof(float), nanovdb::pointer_traits<PtrT>::element_size);
    EXPECT_EQ(3.14f, *ptr);
    EXPECT_EQ(3.14f, ptr[0]);
}
}// anonymous namespace

TEST(TestNanoVDBCUDA, fancy_ptr)
{
    EXPECT_EQ(sizeof(uint8_t), nanovdb::pointer_traits<uint8_t*>::element_size);
    EXPECT_EQ(sizeof(float),   nanovdb::pointer_traits<nanovdb::fancy_ptr<float>>::element_size);
    EXPECT_EQ(sizeof(uint8_t), nanovdb::pointer_traits<nanovdb::fancy_ptr<uint8_t>>::element_size);

    {// test raw pointer
        bool test = nanovdb::util::is_same<nanovdb::pointer_traits<float*>::element_type, float>::value;
        EXPECT_TRUE(test);
        test = nanovdb::util::is_same<nanovdb::pointer_traits<const float*>::element_type, const float>::value;
        EXPECT_TRUE(test);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<float*>::element_size);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<const float*>::element_size);
    }
    {// test std::shared_ptr<float>
        bool test = nanovdb::util::is_same<nanovdb::pointer_traits<std::shared_ptr<float>>::element_type, float>::value;
        EXPECT_TRUE(test);
        test = nanovdb::util::is_same<nanovdb::pointer_traits<std::shared_ptr<const float>>::element_type, const float>::value;
        EXPECT_TRUE(test);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<std::shared_ptr<float>>::element_size);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<std::shared_ptr<const float>>::element_size);
    }
    {// test std::unique_ptr<float>
        bool test = nanovdb::util::is_same<nanovdb::pointer_traits<std::unique_ptr<float>>::element_type, float>::value;
        EXPECT_TRUE(test);
        test = nanovdb::util::is_same<nanovdb::pointer_traits<std::unique_ptr<const float>>::element_type, const float>::value;
        EXPECT_TRUE(test);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<std::unique_ptr<float>>::element_size);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<std::unique_ptr<const float>>::element_size);
    }
    {// test fancy_ptr<float>
        bool test = nanovdb::util::is_same<nanovdb::pointer_traits<nanovdb::fancy_ptr<float>>::element_type, const float>::value;
        EXPECT_TRUE(test);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<nanovdb::fancy_ptr<float>>::element_size);
        test = nanovdb::util::is_same<nanovdb::pointer_traits<nanovdb::fancy_ptr<const float>>::element_type, const float>::value;
        EXPECT_TRUE(test);
        EXPECT_EQ(sizeof(float),  nanovdb::pointer_traits<nanovdb::fancy_ptr<const float>>::element_size);
    }

    float *p = new float;
    *p = 3.14f;
    test_ptr(p);
    auto q = nanovdb::make_fancy(p);
    test_ptr(q);
    delete p;
}// fancy_ptr

TEST(TestNanoVDBCUDA, CudaGridChecksum)
{
    uint32_t checksum;
    const std::string s{"The quick brown fox jumps over the lazy dog"};
    { // test CPU implementation of crc32 without a lookup table
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << nanovdb::util::crc32(s.c_str(), s.size());
        EXPECT_EQ("414fa339", ss.str());// 414FA339 from https://rosettagit.org/drafts/crc-32/#c-1
    }
    { // test CPU implementation of crc32 with a lookup table
        auto lut = nanovdb::util::createCrc32Lut();
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << nanovdb::util::crc32(s.c_str(), s.size(), lut.get());
        EXPECT_EQ("414fa339", ss.str());// 414FA339 from https://rosettagit.org/drafts/crc-32/#c-1
    }
    {// test GPU implementation
        char* d_str;
        uint32_t *d_checksum;
        cudaCheck(cudaMalloc((void**)&d_checksum, 4));
        cudaCheck(cudaMalloc((void**)&d_str, s.size()));
        cudaCheck(cudaMemcpy(d_str, s.data(), s.size(), cudaMemcpyHostToDevice));
        nanovdb::util::cuda::crc32Kernel<<<1, 1>>>((const uint8_t*)d_str, d_checksum, 1, s.size());
        cudaCheck(cudaMemcpy(&checksum, d_checksum, 4, cudaMemcpyDeviceToHost));
        cudaCheck(cudaFree(d_str));
        cudaCheck(cudaFree(d_checksum));
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << checksum;
        EXPECT_EQ("414fa339", ss.str());// 414FA339 from https://rosettagit.org/drafts/crc-32/#c-1
    }
    auto handle = nanovdb::tools::createLevelSetSphere<float>(100);
    EXPECT_TRUE(handle.data());
    auto *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    auto devHandle = nanovdb::cuda::copyTo<SingleBufT>(handle, cudaStream_t(0));
    EXPECT_TRUE(devHandle.deviceData());
#if 0// entire grid or just GridData+TreeData+RootData
    const size_t size = handle.bufferSize();
#else
    //const uint64_t size = grid->memUsage() + grid->tree().memUsage() + grid->tree().root().memUsage() - 16;
    const uint64_t size = grid->memUsage() + grid->tree().memUsage() - 16;
#endif
    //std::cerr << "Grid + tree + root data is " << size << " bytes\n";
    nanovdb::util::Timer       cpuTimer;
    nanovdb::util::cuda::Timer gpuTimer;
    auto  lut = nanovdb::util::createCrc32Lut();
    void *ptr = nanovdb::util::PtrAdd(handle.data(), 16);
    {//benchmark CPU version that uses a table
        //cpuTimer.start("CPU Tabled CRC of level set sphere");
        checksum = nanovdb::util::crc32(ptr, size, lut.get());
        //cpuTimer.stop();
        //std::cerr << checksum << std::endl;
    }
    {//benchmark CPU version that uses no table
        //cpuTimer.start("CPU Untabled CRC of level set sphere");
        auto checksum2 = nanovdb::util::crc32(ptr, size);
        //cpuTimer.stop();
        //std::cerr << checksum2 << std::endl;
        EXPECT_EQ(checksum, checksum2);
    }
    {//benchmark CPU version that uses table
        //cpuTimer.start("CPU tabled util::CRC of level set sphere");
        auto checksum2 = nanovdb::util::crc32(ptr, size, lut.get());
        //cpuTimer.stop();
        //std::cerr << checksum2 << std::endl;
        EXPECT_EQ(checksum, checksum2);
    }
    uint32_t checksum2, *d_checksum;
    cudaCheck(cudaMalloc((void**)&d_checksum, 4));
    void *d_ptr = nanovdb::util::PtrAdd(devHandle.deviceData(), 16);
    {//benchmark GPU version that uses no table
        //gpuTimer.start("GPU Untabled CRC of level set sphere");
        nanovdb::util::cuda::crc32Kernel<<<1, 1>>>(d_ptr, d_checksum, 1, size);
        //gpuTimer.stop();
        cudaCheck(cudaMemcpy(&checksum2, d_checksum, 4, cudaMemcpyDeviceToHost));
        //std::cerr << checksum2 << std::endl;
        EXPECT_EQ(checksum, checksum2);
    }
    {//benchmark GPU version that uses no table
        //gpuTimer.start("GPU tabled CRC of level set sphere");
        auto lut = nanovdb::util::cuda::createCrc32Lut();
        uint32_t *d_lut = lut.get();
        nanovdb::util::cuda::crc32Kernel<<<1, 1>>>(d_ptr, d_checksum, 1, size, d_lut);
        //gpuTimer.stop();
        cudaCheck(cudaMemcpy(&checksum2, d_checksum, 4, cudaMemcpyDeviceToHost));
        //std::cerr << checksum2 << std::endl;
        EXPECT_EQ(checksum, checksum2);
    }
    {
        //cpuTimer.start("CPU GridChecksum of level set sphere");
        nanovdb::Checksum cs = nanovdb::tools::evalChecksum(grid, nanovdb::CheckMode::Partial);
        //cs(*grid);
        //checksum2 = cs.head();// only check the checksum of grid, tree and root data
        //cpuTimer.stop();
        //std::cerr << checksum2 << std::endl;
        EXPECT_EQ(checksum, cs.head());
    }
    nanovdb::Checksum fullChecksum;
    {
        //cpuTimer.start("CPU FULL cudaGridChecksum tabled CRC of level set sphere");
        nanovdb::tools::updateChecksum(handle.grid<float>(), nanovdb::CheckMode::Full);
        //cpuTimer.stop();
        fullChecksum = handle.grid<float>()->checksum();
        EXPECT_EQ(checksum, fullChecksum.head());
    }
    {
        //gpuTimer.start("GPU FULL cudaGridChecksum tabled CRC of level set sphere");
        nanovdb::tools::cuda::updateChecksum(devHandle.deviceGrid<float>(), nanovdb::CheckMode::Full);
        //gpuTimer.stop();
        nanovdb::Checksum fullChecksum2;
        cudaCheck(cudaMemcpy(&fullChecksum2, (const uint8_t*)devHandle.deviceGrid<float>() + 8, 8, cudaMemcpyDeviceToHost));
        EXPECT_EQ(checksum, fullChecksum2.head());
        EXPECT_EQ(fullChecksum, fullChecksum2);
    }
    cudaCheck(cudaFree(d_checksum));
}// CudaGridChecksum

// An empty grid's tail is shorter than one 4 KB checksum block
TEST(TestNanoVDBCUDA, CudaGridChecksum_EmptyGrid)
{
    nanovdb::tools::build::Grid<float> buildGrid(0.0f);
    auto hostHandle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>, nanovdb::ValueOnIndex>(buildGrid);
    ASSERT_LT(hostHandle.gridSize() - sizeof(nanovdb::GridData) - sizeof(nanovdb::TreeData), size_t(1) << NANOVDB_CRC32_LOG2_BLOCK_SIZE);
    auto devHandle = nanovdb::cuda::copyTo<SingleBufT>(hostHandle, cudaStream_t(0));
    nanovdb::tools::cuda::updateChecksum(devHandle.deviceGrid<nanovdb::ValueOnIndex>(), nanovdb::CheckMode::Full);
    auto handle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHandle, cudaStream_t(0));
    auto *grid = handle.grid<nanovdb::ValueOnIndex>();
    ASSERT_TRUE(grid);
    EXPECT_TRUE(grid->checksum().isFull());
    EXPECT_EQ(grid->checksum(), nanovdb::tools::evalChecksum(grid, nanovdb::CheckMode::Full));
}// CudaGridChecksum_EmptyGrid

template <typename BuildT>
size_t countActiveVoxels(const nanovdb::NodeManager<BuildT> *d_mgr)
{
    size_t count[2], *d_count;
    cudaCheck(cudaMalloc((void**)&d_count, 2*sizeof(size_t)));
    nanovdb::util::cuda::lambdaKernel<<<1,1>>>(1, [=] __device__ (size_t){
        d_count[0] = 0;
        for (int i=0; i<d_mgr->leafCount();  ++i)  d_count[0] += d_mgr->leaf(i).valueMask().countOn();
        for (int i=0; i<d_mgr->lowerCount(); ++i)  d_count[0] += d_mgr->lower(i).valueMask().countOn();
        for (int i=0; i<d_mgr->upperCount(); ++i)  d_count[0] += d_mgr->upper(i).valueMask().countOn();
        d_count[1] = d_mgr->tree().activeVoxelCount();
        //printf("active count = %lu %lu\n", d_count[0], d_count[1]);
    });
    cudaCheck(cudaMemcpy(count, d_count, 2*sizeof(size_t), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(d_count));
    EXPECT_EQ(count[0], count[1]);
    return count[0];
}// countActiveVoxels

TEST(TestNanoVDBCUDA, NodeManager)
{
    auto handle = nanovdb::tools::createLevelSetSphere<float>(100);
    EXPECT_TRUE(handle.data());
    auto *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    auto devHandle = nanovdb::cuda::copyTo<SingleBufT>(handle, cudaStream_t(0));
    auto *d_grid = devHandle.deviceGrid<float>();
    EXPECT_TRUE(d_grid);
    size_t count = 0;
    nanovdb::util::Timer cpuTimer;
    {
        //cpuTimer.start("CPU NodeManager");
        auto handle2 = nanovdb::createNodeManager<>(*grid);
        //cpuTimer.stop();
        auto *mgr = handle2.mgr<float>();
        EXPECT_TRUE(mgr);
        count = mgr->grid().tree().activeVoxelCount();
    }

    nanovdb::util::cuda::Timer gpuTimer;
    {
        //gpuTimer.start("GPU NodeManager");
        nanovdb::cuda::DeviceResource resource;
        auto handle2 = nanovdb::cuda::createNodeManager(d_grid, resource);
        //gpuTimer.stop();
        auto *d_mgr = handle2.deviceMgr<float>();
        EXPECT_TRUE(d_mgr);
        EXPECT_EQ(count, countActiveVoxels(d_mgr));
    }
}// NodeManager

TEST(TestNanoVDBCUDA, GridStats)
{
    using GridT = nanovdb::NanoGrid<float>;
    auto handle = nanovdb::tools::createLevelSetSphere<float>(100,
                                                                                  nanovdb::Vec3d(0),
                                                                                  1.0,
                                                                                  3.0,
                                                                                  nanovdb::Vec3d(0),
                                                                                  "test",
                                                                                  nanovdb::tools::StatsMode::Disable);
    EXPECT_TRUE(handle.data());
    GridT *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    auto devHandle = nanovdb::cuda::copyTo<SingleBufT>(handle, cudaStream_t(0));
    GridT *d_grid = devHandle.deviceGrid<float>();
    EXPECT_TRUE(d_grid);

    {// check min/max using const iterators
        float min = std::numeric_limits<float>::max(), max = -min;
        int n2=0, n1=0, n0=0;// check that nodes are arranged breath-first in memory
        for (auto it2 = grid->tree().root().cbeginChild(); it2; ++it2) {
            EXPECT_EQ(grid->tree().getFirstUpper() + n2++, &(*it2));
            for (auto it1 = it2->cbeginChild(); it1; ++it1) {
                EXPECT_EQ(grid->tree().getFirstLower() + n1++, &(*it1));
                for (auto it0 = it1->cbeginChild(); it0; ++it0) {
                    EXPECT_EQ(grid->tree().getFirstLeaf() + n0++, &(*it0));
                    for (auto it = it0->cbeginValueOn(); it; ++it) {
                        if (*it < min) min = *it;
                        if (*it > max) max = *it;
                    }
                }// loop over child nodes of the lower internal node
            }// loop over child nodes of the upper internal node
        }// loop over child nodes of the root node
        EXPECT_NE(min, grid->tree().root().minimum());
        EXPECT_NE(max, grid->tree().root().maximum());
        EXPECT_EQ(n2, grid->tree().nodeCount(2));
        EXPECT_EQ(n1, grid->tree().nodeCount(1));
        EXPECT_EQ(n0, grid->tree().nodeCount(0));
    }
    {
        //nanovdb::util::Timer cpuTimer("CPU gridStats: Default = Full");
        nanovdb::tools::updateGridStats(grid);
        //cpuTimer.stop();
    }
    {// check min/max using const iterators
        float min = std::numeric_limits<float>::max(), max = -min;
        int n2=0, n1=0, n0=0;// check that nodes are arranged breath-first in memory
        for (auto it2 = grid->tree().root().cbeginChild(); it2; ++it2) {
            EXPECT_EQ(grid->tree().getFirstUpper() + n2++, &(*it2));
            for (auto it1 = it2->cbeginChild(); it1; ++it1) {
                EXPECT_EQ(grid->tree().getFirstLower() + n1++, &(*it1));
                for (auto it0 = it1->cbeginChild(); it0; ++it0) {
                    EXPECT_EQ(grid->tree().getFirstLeaf() + n0++, &(*it0));
                    for (auto it = it0->cbeginValueOn(); it; ++it) {
                        if (*it < min) min = *it;
                        if (*it > max) max = *it;
                    }
                }// loop over child nodes of the lower internal node
            }// loop over child nodes of the upper internal node
        }// loop over child nodes of the root node
        EXPECT_EQ(min, grid->tree().root().minimum());
        EXPECT_EQ(max, grid->tree().root().maximum());
        EXPECT_EQ(n2, grid->tree().nodeCount(2));
        EXPECT_EQ(n1, grid->tree().nodeCount(1));
        EXPECT_EQ(n0, grid->tree().nodeCount(0));
    }
    {// check min/max using non-const iterators
        float min = std::numeric_limits<float>::max(), max = -min;
        int n2=0, n1=0, n0=0;// check that nodes are arranged breath-first in memory
        for (auto it2 = grid->tree().root().beginChild(); it2; ++it2) {
            EXPECT_EQ(grid->tree().getFirstUpper() + n2++, &(*it2));
            for (auto it1 = it2->beginChild(); it1; ++it1) {
                EXPECT_EQ(grid->tree().getFirstLower() + n1++, &(*it1));
                for (auto it0 = it1->beginChild(); it0; ++it0) {
                    EXPECT_EQ(grid->tree().getFirstLeaf() + n0++, &(*it0));
                    for (auto it = it0->beginValueOn(); it; ++it) {
                        if (*it < min) min = *it;
                        if (*it > max) max = *it;
                    }
                }// loop over child nodes of the lower internal node
            }// loop over child nodes of the upper internal node
        }// loop over child nodes of the root node
        EXPECT_EQ(min, grid->tree().root().minimum());
        EXPECT_EQ(max, grid->tree().root().maximum());
        EXPECT_EQ(n2, grid->tree().nodeCount(2));
        EXPECT_EQ(n1, grid->tree().nodeCount(1));
        EXPECT_EQ(n0, grid->tree().nodeCount(0));
    }

    {
        //nanovdb::util::cuda::Timer gpuTimer("GPU gridStats: Default = Full");
        nanovdb::tools::cuda::updateGridStats(d_grid);
        //gpuTimer.stop();
    }
    {// check bbox and stats of device grid
        using DataT = nanovdb::NanoRoot<float>::DataType;
        std::unique_ptr<char[]> buffer(new char[sizeof(DataT)]);
        cudaMemcpy(buffer.get(), (char*)d_grid + sizeof(nanovdb::GridData) + sizeof(nanovdb::TreeData), sizeof(DataT), cudaMemcpyDeviceToHost);
        auto *data = (const DataT*)buffer.get();
        EXPECT_EQ(grid->indexBBox(), data->mBBox);
        EXPECT_EQ(grid->tree().root().background(),   data->mBackground);
        EXPECT_EQ(grid->tree().root().minimum(),      data->mMinimum);
        EXPECT_EQ(grid->tree().root().maximum(),      data->mMaximum);
        EXPECT_EQ(grid->tree().root().average(),      data->mAverage);
        EXPECT_EQ(grid->tree().root().stdDeviation(), data->mStdDevi);
    }
}// GridStats

// A stream argument, in any spelling, selects the default-buffer
// createNodeManager overload, never the resource-taking one.
using DefaultNodeManagerHandleT = nanovdb::NodeManagerHandle<nanovdb::cuda::Buffer<std::byte>>;
static_assert(std::is_same<decltype(nanovdb::cuda::createNodeManager(std::declval<const nanovdb::NanoGrid<float>*>())),
                           DefaultNodeManagerHandleT>::value, "no stream: default buffer");
static_assert(std::is_same<decltype(nanovdb::cuda::createNodeManager(std::declval<const nanovdb::NanoGrid<float>*>(), 0)),
                           DefaultNodeManagerHandleT>::value, "literal 0 is a stream");
static_assert(std::is_same<decltype(nanovdb::cuda::createNodeManager(std::declval<const nanovdb::NanoGrid<float>*>(), nullptr)),
                           DefaultNodeManagerHandleT>::value, "nullptr is a stream");
static_assert(std::is_same<decltype(nanovdb::cuda::createNodeManager(std::declval<const nanovdb::NanoGrid<float>*>(), std::declval<cudaStream_t&>())),
                           DefaultNodeManagerHandleT>::value, "a cudaStream_t lvalue is a stream");
static_assert(std::is_same<decltype(nanovdb::cuda::createNodeManager(std::declval<const nanovdb::NanoGrid<float>*>(), std::declval<nanovdb::cuda::DeviceResource&>())),
                           nanovdb::NodeManagerHandle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<nanovdb::cuda::DeviceResource>>>>::value,
              "a resource lvalue is borrowed");

// Stats on a managed grid: the host updates one copy through grid(), the
// device updates another through deviceGrid(), and each side reads back what
// the other wrote.
TEST(TestNanoVDBCUDA, GridStats_ManagedBuffer)
{
    using ManagedBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ManagedResource>;
    using DataT = nanovdb::NanoRoot<float>::DataType;
    auto handle = nanovdb::tools::createLevelSetSphere<float>(100, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0),
                                                              "test", nanovdb::tools::StatsMode::Disable);
    auto refHandle = handle.copy();
    auto *ref = refHandle.grid<float>();
    ASSERT_TRUE(ref);
    nanovdb::tools::updateGridStats(ref);

    auto hostSide = nanovdb::cuda::copyTo<ManagedBufT>(handle, cudaStream_t(0));
    auto devSide  = nanovdb::cuda::copyTo<ManagedBufT>(handle, cudaStream_t(0));
    cudaCheck(cudaDeviceSynchronize());// the host reads and writes the managed grids below
    auto *hostGrid = hostSide.grid<float>();
    ASSERT_TRUE(hostGrid);
    ASSERT_TRUE(devSide.deviceGrid<float>());
    EXPECT_NE(ref->tree().root().minimum(), hostGrid->tree().root().minimum());// stats start disabled

    nanovdb::tools::updateGridStats(hostGrid);
    nanovdb::tools::cuda::updateGridStats(devSide.deviceGrid<float>());
    cudaCheck(cudaDeviceSynchronize());

    auto expectStats = [&](const DataT *data) {
        EXPECT_EQ(ref->indexBBox(),                 data->mBBox);
        EXPECT_EQ(ref->tree().root().background(),   data->mBackground);
        EXPECT_EQ(ref->tree().root().minimum(),      data->mMinimum);
        EXPECT_EQ(ref->tree().root().maximum(),      data->mMaximum);
        EXPECT_EQ(ref->tree().root().average(),      data->mAverage);
        EXPECT_EQ(ref->tree().root().stdDeviation(), data->mStdDevi);
    };
    {// the host-side update, read on the device
        std::unique_ptr<char[]> buffer(new char[sizeof(DataT)]);
        cudaCheck(cudaMemcpy(buffer.get(), (const char*)hostSide.deviceGrid<float>() + sizeof(nanovdb::GridData) + sizeof(nanovdb::TreeData),
                              sizeof(DataT), cudaMemcpyDeviceToHost));
        expectStats(reinterpret_cast<const DataT*>(buffer.get()));
    }
    // the device-side update, read on the host
    expectStats(devSide.grid<float>()->tree().root().data());
}// GridStats_ManagedBuffer

// make -j && ./unittest/testNanoVDB --gtest_filter="*cudaIsValid*" --gtest_repeat=10
TEST(TestNanoVDBCUDA, cudaIsValid)
{
    const auto mode = nanovdb::CheckMode::Full;
    using GridT = nanovdb::NanoGrid<float>;
    nanovdb::util::Timer timer("createLevelSetSphere(500)");
    auto handle = nanovdb::tools::createLevelSetSphere<float>(500,
                                                                       nanovdb::Vec3d(0),
                                                                       1.0,
                                                                       3.0,
                                                                       nanovdb::Vec3d(0),
                                                                       "test",
                                                                       nanovdb::tools::StatsMode::Disable,
                                                                       mode);
    timer.stop();
    EXPECT_TRUE(handle.data());
    GridT *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    auto devHandle = nanovdb::cuda::copyTo<SingleBufT>(handle, cudaStream_t(0));
    GridT *d_grid = devHandle.deviceGrid<float>();
    EXPECT_TRUE(d_grid);
    const bool verbose = false;

    EXPECT_TRUE(nanovdb::tools::isValid(grid,        mode, verbose));
    EXPECT_TRUE(nanovdb::tools::cuda::isValid(d_grid,  mode, verbose));

    grid->mGridType = nanovdb::GridType::Vec3f;
    EXPECT_FALSE(nanovdb::tools::isValid(grid,       mode, verbose));
    cudaCheck(cudaMemcpy(d_grid, grid, handle.bufferSize(), cudaMemcpyHostToDevice));// overwrite the device grid
    EXPECT_FALSE(nanovdb::tools::cuda::isValid(d_grid, mode, verbose));
}// cudaIsValid

TEST(TestNanoVDBCUDA, cudaIsValid_HostBuffer)
{
    using BufferT = nanovdb::HostBuffer;
    const auto mode = nanovdb::CheckMode::Full;
    using GridT = nanovdb::NanoGrid<float>;
    nanovdb::util::Timer timer("createLevelSetSphere<HostBuffer>(500)");
    auto handle = nanovdb::tools::createLevelSetSphere<float, BufferT>(500,
                                                                       nanovdb::Vec3d(0),
                                                                       1.0,
                                                                       3.0,
                                                                       nanovdb::Vec3d(0),
                                                                       "test",
                                                                       nanovdb::tools::StatsMode::Disable,
                                                                       mode);
    timer.stop();
    EXPECT_TRUE(handle.data());
    GridT *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
}// cudaIsValid_HostBuffer

TEST(TestNanoVDBCUDA, overSizedDeviceHandle)
{
    // create a grid in a host buffer that exactly fits the grid
    auto handle1 = nanovdb::tools::createLevelSetSphere();
    const size_t gridSize = handle1.gridSize();
    EXPECT_EQ(gridSize, handle1.bufferSize());
    EXPECT_EQ(gridSize, handle1.totalGridSize());
    EXPECT_EQ(0       , handle1.freeSize());
    EXPECT_EQ(1       , handle1.gridCount());
    EXPECT_TRUE(handle1.isFull());

    // copy grid to an oversized device buffer
    SingleBufT overSizedBuffer(cudaStream_t(0), 2*gridSize, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(overSizedBuffer.data(), handle1.data(), gridSize, cudaMemcpyHostToDevice));

    // construct handle from over-sized device buffer and test it
    nanovdb::GridHandle<SingleBufT> handle2(std::move(overSizedBuffer));
    EXPECT_EQ(1         , handle2.gridCount());
    EXPECT_EQ(  gridSize, handle2.gridSize());
    EXPECT_EQ(  gridSize, handle2.totalGridSize());
    EXPECT_EQ(2*gridSize, handle2.bufferSize());
    EXPECT_EQ(  gridSize, handle2.freeSize());
    EXPECT_FALSE(handle2.isFull());
}// overSizedDeviceHandle

template<class BuildT>
__global__ void testComputeStencilNeighborsKernel(
    nanovdb::NanoGrid<BuildT> *grid,
    uint32_t *firstLeafIDArray,
    uint64_t *jumpMapArray,
    uint64_t *stencilNeighborsArray)
{
    static constexpr int Log2BlockWidth = 7;
    using VBM = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;
    static constexpr int BlockWidth = VBM::BlockWidth;
    static constexpr int JumpMapLength = VBM::JumpMapLength;
    int bID = blockIdx.x;
    int tID = threadIdx.x;

    uint32_t &firstLeafID = firstLeafIDArray[bID];
    uint64_t *jumpMap = jumpMapArray + JumpMapLength * bID;
    int firstOffset = 1;
    int blockFirstOffset = firstOffset + bID * BlockWidth;
    // Decode through the cooperative wrapper (which exercises the per-slot
    // decodeInverseMap internally), then resolve this thread's own slot.
    __shared__ uint32_t leafIndex[BlockWidth];
    __shared__ uint16_t voxelOffset[BlockWidth];

    nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>::decodeInverseMaps(
        grid, firstLeafID, jumpMap, blockFirstOffset, &leafIndex[0], &voxelOffset[0]);

    uint64_t localNeighbors[27] = {};
    nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>::computeBoxStencil(
        grid, leafIndex[tID], voxelOffset[tID], localNeighbors);

    using StencilNeighborsType = uint64_t (*)[27];
    auto stencilNeighbors = reinterpret_cast<StencilNeighborsType>(stencilNeighborsArray+27*BlockWidth*bID);
    for (int i = 0; i < 27; i++)
        stencilNeighbors[tID][i] = localNeighbors[i];
}

template<class BuildT, class BufferT>
void testVoxelBlockManager()
{
    // Create a test domain (21^3 box)
    std::vector<nanovdb::Coord> voxels;
    nanovdb::CoordBBox bbox = nanovdb::CoordBBox::createCube(125,145); // Coordinates chosen to span more than 1 lower node
    for (auto it = bbox.begin(); it; it++) voxels.push_back(*it);
    constexpr int Log2BlockWidth = 7;
    using VBM = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;
    constexpr auto BlockWidth = VBM::BlockWidth;
    const std::size_t nVoxels = voxels.size();
    const std::size_t nBlocks = (nVoxels + BlockWidth - 1) >> Log2BlockWidth;
    nanovdb::Coord *deviceVoxels;
    cudaCheck(cudaMalloc(&deviceVoxels, nVoxels * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(deviceVoxels, voxels.data(), nVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice)); // CPU -> GPU
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(deviceVoxels, nVoxels);
    cudaCheck(cudaFree(deviceVoxels));
    EXPECT_TRUE((handle.deviceData())); // grid only exists on the GPU
    auto deviceGrid = handle.template deviceGrid<BuildT>();
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    EXPECT_TRUE(hostHandle.data());
    auto grid = hostHandle.template grid<BuildT>();
    EXPECT_TRUE(grid);
    // Construct VBM structure
    auto vbmHandle = nanovdb::tools::cuda::buildVoxelBlockManager<Log2BlockWidth, BufferT>(
        deviceGrid);

    // Compute stencil neighbors
    const size_t neighborStencilCount = nBlocks * BlockWidth * 27;
    nanovdb::cuda::Buffer<uint64_t> neighborStencilBuffer(cudaStream_t(0), neighborStencilCount, nanovdb::cuda::noInit);
    EXPECT_TRUE(neighborStencilBuffer.data());
    testComputeStencilNeighborsKernel<<<nBlocks,BlockWidth>>>(deviceGrid,
        vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap(),
        neighborStencilBuffer.data());
    cudaCheck(cudaGetLastError());

    // Verify stencil neighbors against results of standard accessors
    std::vector<uint64_t> hostNeighborStencil(neighborStencilCount);
    cudaCheck(cudaMemcpy(hostNeighborStencil.data(), neighborStencilBuffer.data(),
                         neighborStencilCount * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    using StencilNeighborsType = uint64_t (*)[27];
    auto stencilNeighbors = reinterpret_cast<StencilNeighborsType>(hostNeighborStencil.data());
    auto acc = grid->getAccessor();
    for (const auto& coord : voxels) {
        const auto index = acc.getValue(coord);
        for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
        for (int dk = -1; dk <= 1; dk++) {
            int spokeID = ( di + 1 ) * 9 + ( dj + 1 ) * 3 + dk + 1;
            const auto neighbor = coord.offsetBy( di, dj, dk );
            const auto neighborIndex = acc.getValue(neighbor);
            EXPECT_EQ( neighborIndex, stencilNeighbors[index-1][spokeID] ); } }
}

TEST(TestNanoVDBCUDA, VoxelBlockManager_ValueOnIndex)
{
    testVoxelBlockManager<nanovdb::ValueOnIndex,SingleBufT>();
}// VoxelBlockManager_ValueOnIndex

TEST(TestNanoVDBCUDA, VoxelBlockManager_ValueOnIndex_ManagedBuffer)
{
    testVoxelBlockManager<nanovdb::ValueOnIndex,nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ManagedResource>>();
}// VoxelBlockManager_ValueOnIndex_ManagedBuffer

TEST(TestNanoVDBCUDA, DilateInjectPrune_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Create an input (original) grid to use as input to dilation op
    std::vector<nanovdb::Coord> inputPoints;
    inputPoints.emplace_back(1,0,0); // Added nodes after dilation: 3 root, 3 upper, 3 lower, 3 leaf
    inputPoints.emplace_back(0,1,1); // Added nodes after dilation: 1 root, 1 upper, 1 lower, 1 leaf
    inputPoints.emplace_back(127,127,127); // Added nodes after dilation: 7 lower, 7 leaf
    nanovdb::cuda::Buffer<nanovdb::Coord> inputBuffer(cudaStream_t(0), inputPoints.size(), nanovdb::cuda::noInit);
    EXPECT_TRUE(inputBuffer.data());
    cudaCheck(cudaMemcpy(inputBuffer.data(), inputPoints.data(), inputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto inputHandle = converter.getHandle(inputBuffer.data(), inputPoints.size(), SingleBufT());
    EXPECT_TRUE(inputHandle.deviceGrid<BuildT>());
    auto inputGrid = inputHandle.deviceGrid<BuildT>();

    // Perform dilation
    nanovdb::tools::cuda::DilateGrid<BuildT> dilator( inputGrid );
    dilator.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
    dilator.setChecksum(nanovdb::CheckMode::Default);
    dilator.setVerbose(0);
    auto dilatedHandle = dilator.getHandle(SingleBufT());
    auto dilatedGrid = dilatedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(dilatedGrid);
    auto dilatedTreeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(dilatedGrid);
    EXPECT_EQ(dilatedTreeData.mNodeCount[0], 13);
    EXPECT_EQ(dilatedTreeData.mNodeCount[1], 12);
    EXPECT_EQ(dilatedTreeData.mNodeCount[2], 5);
    EXPECT_EQ(dilatedTreeData.mVoxelCount, 73);

    // Create a prune mask (set bits correspond to retained voxels) from the occupancy of the original grid
    nanovdb::cuda::Buffer<nanovdb::Mask<3>> maskBuffer(cudaStream_t(0), dilatedTreeData.mNodeCount[0], nanovdb::cuda::noInit);
    EXPECT_TRUE(maskBuffer.data());
    auto leafMasks = maskBuffer.data();
    constexpr unsigned int num_threads = 128;
    unsigned int num_blocks = (static_cast<unsigned int>(dilatedTreeData.mNodeCount[0]) + num_threads - 1) / num_threads;
    nanovdb::util::cuda::lambdaKernel<<<num_blocks, num_threads>>>(dilatedTreeData.mNodeCount[0],
        nanovdb::util::cuda::InjectGridMaskFunctor<BuildT>(),
        inputGrid, dilatedGrid, leafMasks);

    // Prune with computed mask
    nanovdb::tools::cuda::PruneGrid<BuildT> pruner( dilatedGrid, leafMasks );
    pruner.setChecksum(nanovdb::CheckMode::Default);
    pruner.setVerbose(0);
    auto prunedHandle = pruner.getHandle(SingleBufT());
    auto prunedGrid = prunedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(prunedGrid);

    // The pruned grid should be identical to the original input
    auto hostInput  = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(inputHandle, cudaStream_t(0));
    auto hostPruned = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(prunedHandle, cudaStream_t(0));
    EXPECT_TRUE(hostInput.grid<BuildT>());
    EXPECT_TRUE(hostPruned.grid<BuildT>());
    EXPECT_EQ(hostInput.grid<BuildT>()->mChecksum.full(), hostPruned.grid<BuildT>()->mChecksum.full());
}// DilateInjectPrune_ValueOnIndex

// Busy-wait on whatever stream it is launched on; used to occupy the default
// stream so that a kernel mistakenly launched there (instead of the caller's
// stream) is reordered relative to the rest of the operation.
__global__ void streamBusyWaitKernel(unsigned long long cycles)
{
    const unsigned long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

// Regression test: the NN_FACE_EDGE_VERTEX (26-neighbour) leaf dilation kernel
// must run on the caller's stream. It previously launched on the default stream,
// silently corrupting the result when the caller used a non-blocking stream.
// Dilating on the default stream and on a non-blocking stream (while the default
// stream is occupied) must produce identical output.
TEST(TestNanoVDBCUDA, NonBlockingStreamDilate_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;
    std::vector<nanovdb::Coord> pts;
    for (int i = 0; i < 64; ++i) pts.emplace_back(i * 3, (i * 7) % 40, (i * 13) % 50);
    pts.emplace_back(127, 127, 127);
    nanovdb::cuda::Buffer<nanovdb::Coord> inBuf(cudaStream_t(0), pts.size(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(inBuf.data(), pts.data(), pts.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    nanovdb::tools::cuda::PointsToGrid<BuildT> conv;
    conv.setChecksum(nanovdb::CheckMode::Full);
    auto inHandle = conv.getHandle(inBuf.data(), pts.size(), SingleBufT());
    auto* inGrid = inHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(inGrid);
    cudaCheck(cudaDeviceSynchronize());

    auto dilateOn = [&](cudaStream_t stream, bool occupyDefault) {
        if (occupyDefault) {// keep the default stream busy for ~20 ms
            int clockKHz = 0, dev = 0;
            cudaCheck(cudaGetDevice(&dev));
            cudaCheck(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev));
            streamBusyWaitKernel<<<1, 1, 0, 0>>>(static_cast<unsigned long long>(clockKHz) * 20ull);
            cudaCheck(cudaGetLastError());
        }
        nanovdb::tools::cuda::DilateGrid<BuildT> dilator(inGrid, stream);
        dilator.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
        dilator.setChecksum(nanovdb::CheckMode::Full);
        dilator.setVerbose(0);
        auto handle = dilator.getHandle(SingleBufT());
        cudaCheck(cudaStreamSynchronize(stream));
        const auto treeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(handle.deviceGrid<BuildT>());
        auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, stream);
        return std::make_pair(treeData.mVoxelCount, hostHandle.grid<BuildT>()->mChecksum.full());
    };

    const auto reference = dilateOn(static_cast<cudaStream_t>(0), false);
    cudaCheck(cudaDeviceSynchronize());
    cudaStream_t nb = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking));
    const auto candidate = dilateOn(nb, /*occupyDefault=*/true);
    cudaCheck(cudaStreamSynchronize(nb));
    cudaCheck(cudaDeviceSynchronize());// drain the default-stream busy-wait so it can't leak into later tests
    cudaCheck(cudaStreamDestroy(nb));

    EXPECT_EQ(reference.first, candidate.first);   // identical active-voxel count
    EXPECT_EQ(reference.second, candidate.second); // identical full checksum
}// NonBlockingStreamDilate_ValueOnIndex

// Cross-stream determinism coverage: signedFloodFill must run entirely on the caller's stream.
// Mirrors the dilation test above: repair the same corrupted level set once on the default
// stream and once on a non-blocking stream while the default stream is occupied, reading each
// result back on the stream that produced it (no default-stream rescue), and require identical
// output. Node passes escaping to the occupied default stream show up as an unrepaired or
// partially repaired grid.
//
// Scope note: this locks the observable contract but cannot discriminate processRoot's internal
// ordering on constructible inputs - its root-tile scanline repair only does work when interior
// root-level tiles exist, which needs a level set thousands of voxels across, and the pre-fix
// code was accidentally host-synchronous for other topologies (a blocking legacy-stream memcpy).
TEST(TestNanoVDBCUDA, NonBlockingStreamSignedFloodFill)
{
    using BufferT = nanovdb::HostBuffer;
    auto runOn = [](cudaStream_t stream, bool occupyDefault) {
        auto hdl = nanovdb::tools::createLevelSetSphere<float, BufferT>(100);
        auto* grid = hdl.grid<float>();
        auto acc = grid->getAccessor();
        using OpT = nanovdb::SetVoxel<float>;
        acc.set<OpT>(nanovdb::Coord(103,0,0), -1.0f);// flip sign and value of an inactive voxel
        acc.set<OpT>(nanovdb::Coord( 97,0,0),  1.0f);// (the corruption CudaSignedFloodFill uses)
        auto devHdl = nanovdb::cuda::copyTo<SingleBufT>(hdl, stream);
        if (occupyDefault) {// keep the default stream busy for ~20 ms
            int clockKHz = 0, dev = 0;
            cudaCheck(cudaGetDevice(&dev));
            cudaCheck(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev));
            streamBusyWaitKernel<<<1, 1, 0, 0>>>(static_cast<unsigned long long>(clockKHz) * 20ull);
            cudaCheck(cudaGetLastError());
        }
        nanovdb::tools::cuda::signedFloodFill(devHdl.deviceGrid<float>(), false, stream);
        auto outHdl = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHdl, stream);
        auto* out = outHdl.grid<float>();
        auto outAcc = out->getAccessor();
        EXPECT_EQ( 3.0f, outAcc(103,0,0));
        EXPECT_EQ( 0.0f, outAcc(100,0,0));
        EXPECT_EQ(-3.0f, outAcc( 97,0,0));
        return out->mChecksum.full();
    };
    const uint64_t reference = runOn(static_cast<cudaStream_t>(0), false);
    cudaCheck(cudaDeviceSynchronize());
    cudaStream_t nb = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking));
    const uint64_t candidate = runOn(nb, /*occupyDefault=*/true);
    cudaCheck(cudaStreamSynchronize(nb));
    cudaCheck(cudaDeviceSynchronize());// drain the default-stream busy-wait so it can't leak into later tests
    cudaCheck(cudaStreamDestroy(nb));
    EXPECT_EQ(reference, candidate);
}// NonBlockingStreamSignedFloodFill

// Regression test: every DeviceGridTraits getter must read on the stream it is given. A plain
// cudaMemcpy is not ordered after work on a non-blocking stream, so a grid still being uploaded on
// that stream would be read before it arrives. Each getter below is called while the grid's upload
// is queued behind a busy-wait on a non-blocking stream, into a device buffer zeroed beforehand;
// a read that is not ordered on that stream sees the zeros. The grid is an OnIndex grid, so that
// every value read (including the value count) differs from what the zeroed buffer holds.
TEST(TestNanoVDBCUDA, NonBlockingStreamDeviceGridTraits)
{
    using BuildT  = nanovdb::ValueOnIndex;
    using GridT   = nanovdb::NanoGrid<BuildT>;
    using TraitsT = nanovdb::util::cuda::DeviceGridTraits<BuildT>;
    auto sphereHandle = nanovdb::tools::createLevelSetSphere<float>(100);
    auto hostHandle = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, BuildT>(*sphereHandle.grid<float>(), 0u, false, false);// no channels, stats or tiles
    const GridT* hostGrid = hostHandle.grid<BuildT>();
    ASSERT_TRUE(hostGrid);
    const auto& hostTreeData = *hostGrid->tree().data();
    const size_t gridBytes = hostHandle.bufferSize();

    // A check can only fail if its expected value differs from a read of the zeroed buffer.
    ASSERT_NE(0u, hostTreeData.mNodeCount[0]);
    ASSERT_NE(0u, hostTreeData.mVoxelCount);
    ASSERT_NE(0u, hostGrid->activeVoxelCount());
    ASSERT_NE(0u, hostGrid->data()->mData1);
    ASSERT_NE(0u, hostGrid->gridSize());
    ASSERT_TRUE(hostGrid->tree().root().bbox()[0] != nanovdb::Coord(0) || hostGrid->tree().root().bbox()[1] != nanovdb::Coord(0));
    ASSERT_NE(0u, hostGrid->tree().root().tileCount());

    void* pinned = nullptr;// pinned, so the upload below is truly asynchronous
    cudaCheck(cudaMallocHost(&pinned, gridBytes));
    std::memcpy(pinned, hostHandle.data(), gridBytes);
    void* d_bytes = nullptr;
    cudaCheck(cudaMalloc(&d_bytes, gridBytes));
    const auto* d_grid = static_cast<const GridT*>(d_bytes);

    cudaStream_t nb = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking));
    int clockKHz = 0, dev = 0;
    cudaCheck(cudaGetDevice(&dev));
    cudaCheck(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev));
    auto queueUpload = [&]() {// zero the device grid, then queue its upload behind ~20 ms of work on nb
        cudaCheck(cudaMemsetAsync(d_bytes, 0, gridBytes, nb));// on nb: ordered after the previous upload
        cudaCheck(cudaStreamSynchronize(nb));
        streamBusyWaitKernel<<<1, 1, 0, nb>>>(static_cast<unsigned long long>(clockKHz) * 20ull);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpyAsync(d_bytes, pinned, gridBytes, cudaMemcpyHostToDevice, nb));
    };

    queueUpload();
    const auto treeData = TraitsT::getTreeData(d_grid, nb);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(hostTreeData.mNodeCount[i], treeData.mNodeCount[i]);
        EXPECT_EQ(hostTreeData.mTileCount[i], treeData.mTileCount[i]);
    }
    EXPECT_EQ(hostTreeData.mVoxelCount, treeData.mVoxelCount);
    queueUpload();
    EXPECT_EQ(hostGrid->activeVoxelCount(), TraitsT::getActiveVoxelCount(d_grid, nb));
    queueUpload();
    EXPECT_EQ(hostGrid->data()->mData1, TraitsT::getValueCount(d_grid, nb));
    queueUpload();
    EXPECT_EQ(hostGrid->gridSize(), TraitsT::getGridSize(d_grid, nb));
    queueUpload();
    EXPECT_EQ(hostGrid->tree().root().bbox(), TraitsT::getIndexBBox(d_grid, hostTreeData, nb));
    queueUpload();
    EXPECT_EQ(hostGrid->tree().root().tileCount(), TraitsT::getRootTableSize(d_grid, hostTreeData, nb));

    cudaCheck(cudaStreamSynchronize(nb));// no upload may still be reading the pinned buffer below
    cudaCheck(cudaStreamDestroy(nb));
    cudaCheck(cudaFree(d_bytes));
    cudaCheck(cudaFreeHost(pinned));
}// NonBlockingStreamDeviceGridTraits

__global__ void bufferFillKernel(unsigned char *p, size_t n, unsigned char v)
{
    for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) p[i] = v;
}
__global__ void bufferCountKernel(const unsigned char *p, size_t n, unsigned char v, unsigned long long *bad)
{
    for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x)
        if (p[i] != v) atomicAdd(bad, 1ull);
}

// A write through data() on a stream other than the buffer's retained stream
// is still in flight when the buffer is destroyed. The caller orders the free
// after it as cuda::Buffer documents: either by making the retained stream wait
// on the write, or by retargeting the free with set_stream. All streams are
// non-blocking, so nothing else orders them. If the free is not ordered after
// the write, the pool recycles the block into the next allocation and the late
// write corrupts it.
static void testBufferCrossStreamFreeOrdering(bool retarget)
{
    const size_t N = size_t(64) << 20;// large enough that a recycled block is the same address
    const unsigned char LATE = 0xAA, VICTIM = 0x55;
    const unsigned long long CYCLES = 400000000ull;// parks 'user' for O(100 ms)

    cudaStream_t user = nullptr, owner = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&user, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&owner, cudaStreamNonBlocking));
    unsigned long long *bad = nullptr;
    cudaCheck(cudaMallocManaged(&bad, sizeof(*bad)));

    {// warm-up: on a cold context the first launches serialize, which would hide the race
        unsigned char *w = nullptr;
        cudaCheck(cudaMallocAsync((void**)&w, N, owner));
        streamBusyWaitKernel<<<1,1,0,user>>>(CYCLES/10);
        bufferFillKernel<<<1024,256,0,owner>>>(w, N, 0);
        bufferCountKernel<<<1024,256,0,owner>>>(w, N, 0, bad);
        cudaCheck(cudaFreeAsync(w, owner));
        cudaCheck(cudaDeviceSynchronize());
    }

    void *devPtr = nullptr;
    cudaEvent_t written;
    cudaCheck(cudaEventCreateWithFlags(&written, cudaEventDisableTiming));
    {
        nanovdb::cuda::Buffer<unsigned char> buf(owner, N, nanovdb::cuda::noInit);
        devPtr = buf.data();
        EXPECT_TRUE(devPtr);
        streamBusyWaitKernel<<<1,1,0,user>>>(CYCLES);// park 'user'
        if (devPtr) bufferFillKernel<<<1024,256,0,user>>>(buf.data(), N, LATE);
        if (retarget) {
            buf.set_stream(user);
        } else {
            cudaCheck(cudaEventRecord(written, user));
            cudaCheck(cudaStreamWaitEvent(buf.stream(), written, 0));
        }
    }// destroyed here; the free is ordered after the write on 'user'

    unsigned char *victim = nullptr;
    cudaCheck(cudaMallocAsync((void**)&victim, N, owner));
    bufferFillKernel<<<1024,256,0,owner>>>(victim, N, VICTIM);
    cudaCheck(cudaStreamSynchronize(owner));

    // Diagnostics only: with a correctly ordered free the pool cannot hand the
    // block out again until 'user' has drained.
    const bool stillPending = (cudaStreamQuery(user) == cudaErrorNotReady);
    cudaGetLastError();// clear the cudaErrorNotReady left by the query above
    const bool recycled = (victim == devPtr);

    cudaCheck(cudaStreamSynchronize(user));// let the late write land
    *bad = 0;
    bufferCountKernel<<<1024,256,0,owner>>>(victim, N, VICTIM, bad);
    cudaCheck(cudaStreamSynchronize(owner));
    const unsigned long long clobbered = *bad;

    cudaCheck(cudaFreeAsync(victim, owner));
    cudaCheck(cudaStreamSynchronize(owner));
    cudaCheck(cudaEventDestroy(written));
    cudaCheck(cudaFree(bad));
    cudaCheck(cudaStreamDestroy(user));
    cudaCheck(cudaStreamDestroy(owner));

    EXPECT_EQ(0u, clobbered) << "device memory was freed while another stream still had work in "
                                "flight (block recycled: " << recycled
                             << ", work still pending when it was reused: " << stillPending << ")";
}// testBufferCrossStreamFreeOrdering

TEST(TestNanoVDBCUDA, BufferFreeOrderedByEventWait)
{
    testBufferCrossStreamFreeOrdering(/*retarget=*/false);
}// BufferFreeOrderedByEventWait

TEST(TestNanoVDBCUDA, BufferFreeOrderedBySetStream)
{
    testBufferCrossStreamFreeOrdering(/*retarget=*/true);
}// BufferFreeOrderedBySetStream

// Regression test: the grid-name copy must not over-read the source string.
// It previously copied a fixed MaxNameSize bytes from the (possibly shorter)
// std::string, over-reading host heap into the grid; now it copies only
// min(name.size()+1, MaxNameSize) bytes. Empty, short, and maximum-length
// names must round-trip.
TEST(TestNanoVDBCUDA, GridName_CudaPointsToGrid)
{
    using BuildT = nanovdb::ValueOnIndex;
    const std::vector<nanovdb::Coord> pts{ {0, 0, 0}, {1, 2, 3} };
    auto build = [&](const std::string& name) {
        nanovdb::cuda::Buffer<nanovdb::Coord> buf(cudaStream_t(0), pts.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(buf.data(), pts.data(), pts.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
        nanovdb::tools::cuda::PointsToGrid<BuildT> conv;
        conv.setGridName(name);
        conv.setChecksum(nanovdb::CheckMode::Full);
        auto h = conv.getHandle(buf.data(), pts.size(), SingleBufT());
        auto hostH = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(h, cudaStream_t(0));
        return std::string(hostH.template grid<BuildT>()->gridName());
    };
    EXPECT_EQ(std::string(""),    build(""));       // empty name
    EXPECT_EQ(std::string("rho"), build("rho"));    // short name
    // Maximum name that fits with a null terminator (MaxNameSize includes it).
    const std::string maxName(nanovdb::GridData::MaxNameSize - 1, 'x');
    EXPECT_EQ(maxName, build(maxName));
    // Over-long name (>= MaxNameSize) must be truncated to MaxNameSize-1 chars
    // and stay null-terminated - never a non-terminated field that gridName()
    // would over-read. The result is the max-length string, and reading it back
    // as a C-string does not exceed MaxNameSize.
    const std::string tooLong(nanovdb::GridData::MaxNameSize + 37, 'y');
    const std::string got = build(tooLong);
    EXPECT_EQ(size_t(nanovdb::GridData::MaxNameSize - 1), got.size());
    EXPECT_EQ(std::string(nanovdb::GridData::MaxNameSize - 1, 'y'), got);
}// GridName_CudaPointsToGrid

// Regression test: PointsToGrid output must be byte-deterministic. Alignment
// padding and stats fields previously carried recycled pool bytes, so the output
// (and its full checksum) was nondeterministic; the buffer-wide zero-init fixes
// this. Two independent builds of the same input must have identical full
// checksums.
TEST(TestNanoVDBCUDA, DeterministicOutput_CudaPointsToGrid)
{
    using BuildT = nanovdb::ValueOnIndex;
    std::vector<nanovdb::Coord> pts;
    for (int i = 0; i < 256; ++i) pts.emplace_back((i * 5) % 97, (i * 11) % 53, (i * 17) % 71);
    auto buildChecksum = [&]() {
        nanovdb::cuda::Buffer<nanovdb::Coord> buf(cudaStream_t(0), pts.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(buf.data(), pts.data(), pts.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
        nanovdb::tools::cuda::PointsToGrid<BuildT> conv;
        conv.setChecksum(nanovdb::CheckMode::Full);
        auto h = conv.getHandle(buf.data(), pts.size(), SingleBufT());
        auto hostH = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(h, cudaStream_t(0));
        return hostH.template grid<BuildT>()->mChecksum.full();
    };
    EXPECT_EQ(buildChecksum(), buildChecksum());
}// DeterministicOutput_CudaPointsToGrid

TEST(TestNanoVDBCUDA, RefineCoarsen_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Create an input (original) grid to use as input to refinement op
    std::vector<nanovdb::Coord> inputPoints;
    inputPoints.emplace_back(0,0,0); // Refinement doesn't introduce extra nodes
    inputPoints.emplace_back(4,0,0); // Refinement will introduce extra leaf
    inputPoints.emplace_back(0,64,0); // Refinement will introduce extra lower node
    inputPoints.emplace_back(0,0,2048); // Refinement will introduce extra upper node (and root tile)
    nanovdb::cuda::Buffer<nanovdb::Coord> inputBuffer(cudaStream_t(0), inputPoints.size(), nanovdb::cuda::noInit);
    EXPECT_TRUE(inputBuffer.data());
    cudaCheck(cudaMemcpy(inputBuffer.data(), inputPoints.data(), inputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto inputHandle = converter.getHandle(inputBuffer.data(), inputPoints.size(), SingleBufT());
    EXPECT_TRUE(inputHandle.deviceGrid<BuildT>());
    auto inputGrid = inputHandle.deviceGrid<BuildT>();

    // Perform refinement
    nanovdb::tools::cuda::RefineGrid<BuildT> refiner( inputGrid );
    refiner.setChecksum(nanovdb::CheckMode::Default);
    refiner.setVerbose(0);
    auto refinedHandle = refiner.getHandle(SingleBufT());
    auto refinedGrid = refinedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(refinedGrid);
    auto refinedTreeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(refinedGrid);
    EXPECT_EQ(refinedTreeData.mNodeCount[0], 4);
    EXPECT_EQ(refinedTreeData.mNodeCount[1], 3);
    EXPECT_EQ(refinedTreeData.mNodeCount[2], 2);
    EXPECT_EQ(refinedTreeData.mVoxelCount, 32);

    // Coarsen back to original
    nanovdb::tools::cuda::CoarsenGrid<BuildT> coarsener( refinedGrid );
    coarsener.setChecksum(nanovdb::CheckMode::Default);
    coarsener.setVerbose(0);
    auto coarsenedHandle = coarsener.getHandle(SingleBufT());
    auto coarsenedGrid = coarsenedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(coarsenedGrid);

    // The coarsened grid should be identical to the original input
    auto hostInput     = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(inputHandle, cudaStream_t(0));
    auto hostCoarsened = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(coarsenedHandle, cudaStream_t(0));
    EXPECT_TRUE(hostInput.grid<BuildT>());
    EXPECT_TRUE(hostCoarsened.grid<BuildT>());
    EXPECT_EQ(hostInput.grid<BuildT>()->mChecksum.full(), hostCoarsened.grid<BuildT>()->mChecksum.full());
}// RefineCoarsen_ValueOnIndex

namespace {

/// Drives RefineInternalNodesFunctor over every source leaf of a batch: the leaf's member is
/// found from the per-grid leaf offsets, and the member's processed root and mask rows are
/// selected by pointer offset.
template <typename BuildT>
struct BatchRefineInternalNodes
{
    const nanovdb::NanoGrid<BuildT>* const* srcGrids;
    const nanovdb::NanoRoot<BuildT>* const* roots;
    const uint32_t* tileBase;
    const uint32_t* leafStart;
    nanovdb::Mask<5>* upperMasks;
    nanovdb::Mask<4> (*lowerMasks)[nanovdb::Mask<5>::SIZE];

    __device__ void operator()(size_t leaf) const {
        uint32_t g = 0;
        while (leafStart[g + 1] <= leaf) ++g;
        nanovdb::util::morphology::cuda::RefineInternalNodesFunctor<BuildT>()(
            leaf - leafStart[g], srcGrids[g], roots[g], upperMasks + tileBase[g], lowerMasks + tileBase[g]);
    }
};

/// Drives RefineLeafMasksFunctor over every source leaf of a batch into that member's destination grid
template <typename BuildT>
struct BatchRefineLeafMasks
{
    const nanovdb::NanoGrid<BuildT>* const* srcGrids;
    nanovdb::NanoGrid<BuildT>* const* dstGrids;
    const uint32_t* leafStart;

    __device__ void operator()(size_t leaf) const {
        uint32_t g = 0;
        while (leafStart[g + 1] <= leaf) ++g;
        nanovdb::util::morphology::cuda::RefineLeafMasksFunctor<BuildT>()(leaf - leafStart[g], srcGrids[g], dstGrids[g]);
    }
};

}// anonymous namespace

template struct nanovdb::util::morphology::cuda::ProcessLowerNodesFunctor<nanovdb::ValueOnIndex>;// compiles the single-grid operator()

// Consumers hold the builder by value, so it must stay movable for them to be
static_assert(std::is_move_constructible_v<nanovdb::tools::cuda::TopologyBuilder<nanovdb::ValueOnIndex>> &&
              std::is_move_assignable_v<nanovdb::tools::cuda::TopologyBuilder<nanovdb::ValueOnIndex>>);
static_assert(std::is_move_constructible_v<nanovdb::tools::cuda::DilateGrid<nanovdb::ValueOnIndex>>);
static_assert(std::is_move_constructible_v<nanovdb::tools::cuda::RefineGrid<nanovdb::ValueOnIndex>>);
static_assert(std::is_move_constructible_v<nanovdb::tools::cuda::CoarsenGrid<nanovdb::ValueOnIndex>>);
static_assert(std::is_move_constructible_v<nanovdb::tools::cuda::PruneGrid<nanovdb::ValueOnIndex>>);
static_assert(std::is_move_constructible_v<nanovdb::tools::cuda::MergeGrids<nanovdb::ValueOnIndex>>);

// Builds five grids in one TopologyBuilder pass (refinement, driven directly through the
// batch API) and checks every member against RefineGrid on that member alone. The batch is
// built twice on one builder without draining the stream in between, on a non-blocking
// stream, so the second build relies on getBuffer's own wait for the node counts and on the
// builder ordering its reuse of the host staging after the first build's copies. The second
// build also computes full checksums.
TEST(TestNanoVDBCUDA, TopologyBuilderBatch_ValueOnIndex)
{
    using BuildT  = nanovdb::ValueOnIndex;
    using GridT   = nanovdb::NanoGrid<BuildT>;
    using RootT   = nanovdb::NanoRoot<BuildT>;
    using DevBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;// single-space device handles throughout
    using PinBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>;// host-readable copies for the byte comparison
    namespace topo = nanovdb::tools::cuda::topology::detail;
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::unique_ptr<CUstream_st, decltype(&cudaStreamDestroy)> streamGuard(stream, &cudaStreamDestroy);// outlives every buffer below

    // Members 0 and 4: empty, so the batch starts and ends with grids that have no tiles.
    // Member 1: the RefineCoarsen fixture. Member 2: empty, but given speculative tiles that
    // receive no nodes. Member 3: voxels across the +-4096 tile boundaries and in negative octants.
    std::vector<std::vector<nanovdb::Coord>> members = {
        {},
        {{0,0,0}, {4,0,0}, {0,64,0}, {0,0,2048}},
        {},
        {{-1,-1,-1}, {4095,0,0}, {4096,0,0}, {-4097,8,8}, {7,-4096,2047}},
        {} };
    const uint32_t gridCount = static_cast<uint32_t>(members.size());
    std::vector<nanovdb::CoordBBox> unusedTiles(gridCount);
    unusedTiles[2] = nanovdb::CoordBBox(nanovdb::Coord(-1), nanovdb::Coord(0));// the 8 tiles around the origin

    std::vector<nanovdb::GridHandle<DevBufT>> inputs;
    for (const auto &points : members) {
        if (points.empty()) {
            nanovdb::tools::build::Grid<float> buildGrid(0.0f);
            auto hostHandle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>, BuildT, nanovdb::HostBuffer>(buildGrid);
            inputs.push_back(nanovdb::cuda::copyTo<DevBufT>(hostHandle, stream));
        } else {
            nanovdb::cuda::Buffer<nanovdb::Coord> pointBuffer(stream, points.size(), nanovdb::cuda::noInit);
            cudaCheck(cudaMemcpyAsync(pointBuffer.data(), points.data(), pointBuffer.size_bytes(), cudaMemcpyHostToDevice, stream));
            nanovdb::tools::cuda::PointsToGrid<BuildT> converter(1.0, nanovdb::Vec3d(0), stream);
            inputs.push_back(converter.getHandle<nanovdb::Coord*, DevBufT>(pointBuffer.data(), points.size()));
            cudaCheck(cudaStreamSynchronize(stream));// the point buffer is released at the end of this iteration
        }
    }
    std::vector<const GridT*> srcGrids(gridCount);
    std::vector<nanovdb::TreeData> srcTrees(gridCount);
    std::vector<uint32_t> leafStart(gridCount + 1, 0);
    for (uint32_t g = 0; g < gridCount; ++g) {
        srcGrids[g] = inputs[g].deviceGrid<BuildT>();
        ASSERT_TRUE(srcGrids[g]);
        srcTrees[g] = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(srcGrids[g]);
        leafStart[g + 1] = leafStart[g] + srcTrees[g].mNodeCount[0];
    }
    const uint32_t srcLeafCount = leafStart[gridCount];

    // Reference: refine each member on its own, copied to pinned memory for the byte comparison
    std::vector<nanovdb::GridHandle<PinBufT>> refs;
    for (uint32_t g = 0; g < gridCount; ++g) {
        nanovdb::tools::cuda::RefineGrid<BuildT> refiner(srcGrids[g], stream);
        auto deviceRef = refiner.getHandle<DevBufT>();
        refs.push_back(nanovdb::cuda::copyTo<PinBufT>(deviceRef, stream));
    }
    cudaCheck(cudaStreamSynchronize(stream));

    // Speculative roots per grid, exactly as RefineGrid::refineRoot derives them, plus the unused tiles
    std::vector<topo::ProcessedTileMap<RootT>> tiles(gridCount);
    std::vector<uint32_t> tileCounts(gridCount);
    for (uint32_t g = 0; g < gridCount; ++g) {
        if (srcTrees[g].mVoxelCount) {
            auto deviceSrcRoot = static_cast<const RootT*>(nanovdb::util::PtrAdd(srcGrids[g], GridT::memUsage() + srcTrees[g].mNodeOffset[3]));
            const uint64_t rootAndUpperSize = srcTrees[g].mNodeOffset[1] - srcTrees[g].mNodeOffset[3];
            auto hostCopy = nanovdb::HostBuffer::create(rootAndUpperSize);
            cudaCheck(cudaMemcpy(hostCopy.data(), deviceSrcRoot, rootAndUpperSize, cudaMemcpyDeviceToHost));
            auto srcRoot = static_cast<RootT*>(hostCopy.data());
            for (uint32_t t = 0; t < srcRoot->tileCount(); ++t) {
                const auto bbox = srcRoot->getChild(srcRoot->tile(t))->bbox();
                topo::insertProcessedTiles<RootT>(tiles[g], nanovdb::CoordBBox(
                    nanovdb::util::morphology::refineCoord(bbox.min()),
                    nanovdb::util::morphology::refineCoord(bbox.max()).offsetBy(1)));
            }
        }
        topo::insertProcessedTiles<RootT>(tiles[g], unusedTiles[g]);
        tileCounts[g] = static_cast<uint32_t>(tiles[g].size());
    }
    EXPECT_EQ(tileCounts[0], 0u);
    EXPECT_EQ(tileCounts[2], 8u);
    EXPECT_GT(tileCounts[3], 1u);
    EXPECT_EQ(tileCounts[4], 0u);

    auto upload = [&](const auto &host) {// std::vector<T> -> device cuda::Buffer<T>; pageable source, so the copy completes before returning
        using T = typename std::decay_t<decltype(host)>::value_type;
        nanovdb::cuda::Buffer<T> buf(stream, host.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpyAsync(buf.data(), host.data(), buf.size_bytes(), cudaMemcpyHostToDevice, stream));
        return buf;
    };
    auto dSrcGrids  = upload(srcGrids);
    auto dLeafStart = upload(leafStart);
    const GridT* const* d_srcGrids  = dSrcGrids.data();
    const uint32_t*     d_leafStart = dLeafStart.data();

    auto expectNodeCounts = [&](const nanovdb::tools::cuda::TopologyBuilder<BuildT> &builder) {
        EXPECT_EQ(builder.data(1)->nodeCount[0], 4u);
        EXPECT_EQ(builder.data(1)->nodeCount[1], 3u);
        EXPECT_EQ(builder.data(1)->nodeCount[2], 2u);
        for (uint32_t g : {0u, 2u, 4u}) {
            EXPECT_EQ(builder.data(g)->nodeCount[0], 0u) << "grid " << g;
            EXPECT_EQ(builder.data(g)->nodeCount[2], 0u) << "grid " << g;
        }
    };

    // One batched refinement; returns the output buffer without draining the stream
    auto buildBatch = [&](nanovdb::tools::cuda::TopologyBuilder<BuildT> &builder, bool waitForNodeCounts, nanovdb::CheckMode checksum) {
        builder.mChecksum = checksum;
        builder.allocateProcessedRoots(tileCounts);
        for (uint32_t g = 0; g < gridCount; ++g) topo::packProcessedRoot(tiles[g], builder.hostProcessedRoot(g));
        builder.uploadProcessedRoot(stream);
        builder.allocateInternalMaskBuffers(stream);

        // Per-grid dispatch tables for the mask kernel: processed root and mask row base of every member
        std::vector<const RootT*> roots(gridCount);
        std::vector<uint32_t> tileBase(gridCount);
        for (uint32_t g = 0; g < gridCount; ++g) {
            roots[g] = builder.deviceProcessedRoot(g);
            tileBase[g] = builder.data(g)->tileBase;
        }
        auto dRoots    = upload(roots);
        auto dTileBase = upload(tileBase);
        if (srcLeafCount) {
            nanovdb::util::cuda::lambdaKernel<<<(srcLeafCount + 127) / 128, 128, 0, stream>>>(srcLeafCount,
                BatchRefineInternalNodes<BuildT>{d_srcGrids, dRoots.data(), dTileBase.data(), d_leafStart,
                                                 builder.deviceUpperMasks(), builder.deviceLowerMasks()});
            cudaCheckError();
        }

        builder.countNodes(stream);
        if (waitForNodeCounts) {
            builder.waitForNodeCounts();
            expectNodeCounts(builder);
        }
        auto buffer = builder.getBuffer(DevBufT(), stream);
        expectNodeCounts(builder);
        for (uint32_t g = 0; g < gridCount; ++g)
            cudaCheck(cudaMemcpyAsync(&builder.data(g)->getGrid(), srcGrids[g]->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, stream));
        builder.processGridTreeRoot(stream);
        builder.processUpperNodes(stream);
        builder.processLowerNodes(stream);

        std::vector<GridT*> dstGrids(gridCount);
        for (uint32_t g = 0; g < gridCount; ++g) dstGrids[g] = &builder.data(g)->getGrid();
        auto dDstGrids = upload(dstGrids);
        if (srcLeafCount) {
            nanovdb::util::cuda::lambdaKernel<<<(srcLeafCount + 127) / 128, 128, 0, stream>>>(srcLeafCount,
                BatchRefineLeafMasks<BuildT>{d_srcGrids, dDstGrids.data(), d_leafStart});
            cudaCheckError();
        }
        builder.processLeafOffsets(stream);
        builder.processBBox(stream);
        builder.postProcessGridTree(stream);
        return buffer;
    };

    nanovdb::tools::cuda::TopologyBuilder<BuildT> builder(stream);
    const std::vector<nanovdb::CheckMode> checksums = {nanovdb::CheckMode::Disable, nanovdb::CheckMode::Full};
    std::vector<DevBufT> buffers;
    buffers.push_back(buildBatch(builder, true, checksums[0]));
    buffers.push_back(buildBatch(builder, false, checksums[1]));
    cudaCheck(cudaStreamSynchronize(stream));

    for (size_t build = 0; build < buffers.size(); ++build) {
        SCOPED_TRACE("build " + std::to_string(build));
        // The device buffer holds a valid chain of gridCount grids (the single-space handle ctor validates it)
        nanovdb::GridHandle<DevBufT> device(std::move(buffers[build]));
        ASSERT_EQ(device.gridCount(), gridCount);
        auto batch = nanovdb::cuda::copyTo<PinBufT>(device, stream);
        cudaCheck(cudaStreamSynchronize(stream));
        ASSERT_EQ(batch.gridCount(), gridCount);

        for (uint32_t g = 0; g < gridCount; ++g) {
            ASSERT_EQ(batch.gridSize(g), refs[g].gridSize());
            std::vector<uint8_t> expected(refs[g].gridSize());
            std::memcpy(expected.data(), refs[g].gridData(), expected.size());
            auto *header = reinterpret_cast<nanovdb::GridData*>(expected.data());
            header->mGridIndex = g;
            header->mGridCount = gridCount;
            if (checksums[build] == nanovdb::CheckMode::Full) {// the head checksum covers mGridIndex and mGridCount, so check it on its own
                EXPECT_TRUE(batch.gridData(g)->mChecksum.isFull()) << "grid " << g;
                EXPECT_TRUE(nanovdb::tools::validateChecksum(batch.gridData(g), nanovdb::CheckMode::Full)) << "grid " << g;
                header->mChecksum = batch.gridData(g)->mChecksum;
            }
            EXPECT_EQ(0, std::memcmp(batch.gridData(g), expected.data(), expected.size())) << "grid " << g << " differs from its single-grid refinement";
            EXPECT_EQ(batch.gridData(g)->mGridIndex, g);
            EXPECT_EQ(batch.gridData(g)->mGridCount, gridCount);
        }
        auto *grid1 = batch.grid<BuildT>(1);
        ASSERT_TRUE(grid1);
        EXPECT_EQ(grid1->tree().nodeCount(0), 4u);
        EXPECT_EQ(grid1->tree().nodeCount(1), 3u);
        EXPECT_EQ(grid1->tree().nodeCount(2), 2u);
        EXPECT_EQ(grid1->tree().activeVoxelCount(), 32u);
        for (uint32_t g : {0u, 2u, 4u}) EXPECT_EQ(batch.grid<BuildT>(g)->tree().activeVoxelCount(), 0u) << "grid " << g;
        EXPECT_EQ(batch.grid<BuildT>(3)->tree().activeVoxelCount(), 8u * members[3].size());
    }
}// TopologyBuilderBatch_ValueOnIndex

// RefineGrid and DilateGrid enumerate their speculative root tiles with insertProcessedTiles.
// This checks it against the per-tile loops and offset-shifted key it replaced, for upper-node
// bboxes with every face on an octant or tile boundary, in tiles on both sides of the origin
// and far from it.
TEST(TestNanoVDBCUDA, TopologyBuilder_ProcessedTileEnumeration)
{
    using RootT = nanovdb::NanoRoot<nanovdb::ValueOnIndex>;
    using TileT = typename RootT::DataType::Tile;
    using nanovdb::Coord;
    using nanovdb::CoordBBox;
    namespace topo = nanovdb::tools::cuda::topology::detail;
    static constexpr int32_t kDim = 4096;// root tile width

    auto coordToKey = [](const Coord &ijk)->uint64_t{// the consumers' former sort key
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) |
               (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) |
               (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);
    };
    auto insertOld = [&](topo::ProcessedTileMap<RootT> &tiles, const Coord &origin) {
        tiles.emplace(coordToKey(origin), TileT{RootT::CoordToKey(origin)});
    };
    auto sameTiles = [](const topo::ProcessedTileMap<RootT> &a, const topo::ProcessedTileMap<RootT> &b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
            [](const auto &x, const auto &y) { return x.first == y.first && x.second.key == y.second.key; });
    };

    const int32_t origins[] = {-(1 << 30), -kDim, 0};
    const int32_t offsets[] = {0, 1, kDim/2 - 1, kDim/2, kDim - 1};// octant and tile faces within a tile
    uint32_t bboxCount = 0;
    for (int32_t ox : origins) for (int32_t oy : origins) for (int32_t oz : origins) {
        const Coord origin(ox, oy, oz);
        for (int n = 0; n < 15625; ++n) {// every (min, max) pair of offsets on each axis
            Coord lo, hi;
            for (int a = 0, m = n; a < 3; ++a, m /= 25) { lo[a] = offsets[m % 5]; hi[a] = offsets[m / 5 % 5]; }
            if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) continue;
            const CoordBBox bbox(origin + lo, origin + hi);// an upper node's bbox
            ++bboxCount;

            // Refinement: one tile per 2048^3 octant of the source tile that the bbox overlaps
            topo::ProcessedTileMap<RootT> oldRefine, newRefine;
            for (int di = 0; di <= kDim/2; di += kDim/2)
            for (int dj = 0; dj <= kDim/2; dj += kDim/2)
            for (int dk = 0; dk <= kDim/2; dk += kDim/2) {
                const auto octant = CoordBBox::createCube(origin.offsetBy(di, dj, dk), kDim/2);
                if (bbox.hasOverlap(octant)) insertOld(oldRefine, octant.min() + octant.min());
            }
            topo::insertProcessedTiles<RootT>(newRefine, CoordBBox(nanovdb::util::morphology::refineCoord(bbox.min()),
                                                                   nanovdb::util::morphology::refineCoord(bbox.max()).offsetBy(1)));
            EXPECT_TRUE(sameTiles(oldRefine, newRefine)) << "refine: origin " << ox << ',' << oy << ',' << oz << ", case " << n;

            // Dilation: every tile in the 27-neighborhood that the bbox, grown by one voxel, overlaps
            topo::ProcessedTileMap<RootT> oldDilate, newDilate;
            const auto dilated = bbox.expandBy(1);
            for (int di = -kDim; di <= kDim; di += kDim)
            for (int dj = -kDim; dj <= kDim; dj += kDim)
            for (int dk = -kDim; dk <= kDim; dk += kDim) {
                const auto neighbor = CoordBBox::createCube(origin.offsetBy(di, dj, dk), kDim);
                if (neighbor.hasOverlap(dilated)) insertOld(oldDilate, neighbor.min());
            }
            topo::insertProcessedTiles<RootT>(newDilate, dilated);
            EXPECT_TRUE(sameTiles(oldDilate, newDilate)) << "dilate: origin " << ox << ',' << oy << ',' << oz << ", case " << n;
        }
    }
    EXPECT_EQ(bboxCount, 27u * 15u * 15u * 15u);

    // An empty bbox (a node whose bbox was never computed) adds no tiles, including after dilation.
    // RefineGrid skips empty bboxes itself because refineCoord overflows on the sentinel min/max.
    const CoordBBox emptyBBox;
    ASSERT_TRUE(emptyBBox.empty());
    topo::ProcessedTileMap<RootT> emptyTiles, emptyDilate;
    topo::insertProcessedTiles<RootT>(emptyTiles, emptyBBox);
    topo::insertProcessedTiles<RootT>(emptyDilate, emptyBBox.expandBy(1));
    EXPECT_TRUE(emptyTiles.empty());
    EXPECT_TRUE(emptyDilate.empty());
}// TopologyBuilder_ProcessedTileEnumeration

TEST(TestNanoVDBCUDA, TopologyBuilder_ProcessedRootAllocation)
{
    using BuildT = nanovdb::ValueOnIndex;
    using RootT  = nanovdb::NanoRoot<BuildT>;
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::unique_ptr<CUstream_st, decltype(&cudaStreamDestroy)> streamGuard(stream, &cudaStreamDestroy);
    nanovdb::tools::cuda::TopologyBuilder<BuildT> builder(stream);

    // The tile-count form sets mTableSize, for zero tiles too
    for (uint32_t tileCount : {5u, 0u}) {
        RootT *root = builder.allocateProcessedRootTiles(tileCount);
        ASSERT_NE(root, nullptr);
        EXPECT_EQ(root, builder.hostProcessedRoot(0));
        EXPECT_EQ(builder.gridCount(), 1u);
        EXPECT_EQ(root->mTableSize, tileCount);
    }

    // The deprecated byte form returns the single root and resets a batch to one grid
    builder.allocateProcessedRoots({2, 3});
    EXPECT_EQ(builder.gridCount(), 2u);
    NANOVDB_SUPPRESS_DEPRECATION_BEGIN
    RootT *root = builder.allocateProcessedRoot(RootT::memUsage(3));
    NANOVDB_SUPPRESS_DEPRECATION_END
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root, builder.hostProcessedRoot(0));
    EXPECT_EQ(builder.gridCount(), 1u);
}// TopologyBuilder_ProcessedRootAllocation

TEST(TestNanoVDBCUDA, MergeGrids_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Create the first input grid for the merge op
    std::vector<nanovdb::Coord> inputPointsA;
    for (int i = 0; i <= 2; i++)
        for (int j = 0; j <= 2; j++)
            for (int k = 0; k <= 2; k++)
                inputPointsA.emplace_back(i-1, j*8-1, k*128); // 4 upper, 12 lower, 18 leaf nodes
    nanovdb::cuda::Buffer<nanovdb::Coord> inputBufferA(cudaStream_t(0), inputPointsA.size(), nanovdb::cuda::noInit);
    EXPECT_TRUE(inputBufferA.data());
    cudaCheck(cudaMemcpy(inputBufferA.data(), inputPointsA.data(), inputPointsA.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto inputHandleA = nanovdb::tools::cuda::voxelsToGrid<BuildT>(inputBufferA.data(), inputPointsA.size());
    EXPECT_TRUE(inputHandleA.deviceGrid<BuildT>());
    auto inputGridA = inputHandleA.deviceGrid<BuildT>();

    // Create the second input grid for the merge op
    std::vector<nanovdb::Coord> inputPointsB;
    for (int i = 0; i <= 2; i++)
        for (int j = 0; j <= 2; j++)
            for (int k = 0; k <= 2; k++)
                inputPointsB.emplace_back(i, j*8-1, (k-1)*128); // 4 upper, 6 lower, 9 leaf nodes
    nanovdb::cuda::Buffer<nanovdb::Coord> inputBufferB(cudaStream_t(0), inputPointsB.size(), nanovdb::cuda::noInit);
    EXPECT_TRUE(inputBufferB.data());
    cudaCheck(cudaMemcpy(inputBufferB.data(), inputPointsB.data(), inputPointsB.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto inputHandleB = nanovdb::tools::cuda::voxelsToGrid<BuildT>(inputBufferB.data(), inputPointsB.size());
    EXPECT_TRUE(inputHandleB.deviceGrid<BuildT>());
    auto inputGridB = inputHandleB.deviceGrid<BuildT>();

    // Perform the merge operation
    nanovdb::tools::cuda::MergeGrids<BuildT> merger( inputGridA, inputGridB );
    merger.setChecksum(nanovdb::CheckMode::Disable);
    merger.setVerbose(0);
    auto mergedHandle = merger.getHandle();
    auto mergedGrid = mergedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(mergedGrid);
    auto mergedTreeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(mergedGrid);
    EXPECT_EQ(mergedTreeData.mNodeCount[0], 21);
    EXPECT_EQ(mergedTreeData.mNodeCount[1], 14);
    EXPECT_EQ(mergedTreeData.mNodeCount[2], 6);
    EXPECT_EQ(mergedTreeData.mVoxelCount, 42); // Each input grid has 27 active voxels, 12 shared between the two
}// DilateInjectPrune_ValueOnIndex

TEST(TestNanoVDBCUDA, MergeGridsNary_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;
    using nanovdb::Coord;
    using GridT = nanovdb::NanoGrid<BuildT>;
    auto treeData = [](const GridT* g) {
        return nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(g);
    };

    // Three inputs: A and B overlap (12 shared voxels, as in the binary test),
    // C is disjoint and lives in a separate upper region.
    std::vector<Coord> ptsA, ptsB, ptsC;
    for (int i = 0; i <= 2; i++)
        for (int j = 0; j <= 2; j++)
            for (int k = 0; k <= 2; k++) {
                ptsA.emplace_back(i - 1,    j * 8 - 1, k * 128);
                ptsB.emplace_back(i,        j * 8 - 1, (k - 1) * 128);
                ptsC.emplace_back(i + 4096, j,         k);
            }

    auto buildGrid = [](const std::vector<Coord>& pts, nanovdb::cuda::Buffer<Coord>& buf) {
        buf = nanovdb::cuda::Buffer<Coord>(cudaStream_t(0), pts.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(buf.data(), pts.data(), pts.size() * sizeof(Coord),
                             cudaMemcpyHostToDevice));
        return nanovdb::tools::cuda::voxelsToGrid<BuildT>(buf.data(), pts.size());
    };
    nanovdb::cuda::Buffer<Coord> bufA, bufB, bufC;
    auto hA = buildGrid(ptsA, bufA); auto gA = hA.deviceGrid<BuildT>();
    auto hB = buildGrid(ptsB, bufB); auto gB = hB.deviceGrid<BuildT>();
    auto hC = buildGrid(ptsC, bufC); auto gC = hC.deviceGrid<BuildT>();

    // N-ary merge of all three in a single call
    auto naryH = nanovdb::tools::cuda::MergeGrids<BuildT>(
        std::vector<const GridT*>{gA, gB, gC}).getHandle();
    auto nary = treeData(naryH.deviceGrid<BuildT>());

    // Chaining the binary form must yield byte-identical topology
    auto abH = nanovdb::tools::cuda::MergeGrids<BuildT>(gA, gB).getHandle();
    auto abcH = nanovdb::tools::cuda::MergeGrids<BuildT>(
        abH.deviceGrid<BuildT>(), gC).getHandle();
    auto chain = treeData(abcH.deviceGrid<BuildT>());
    EXPECT_EQ(nary.mNodeCount[0], chain.mNodeCount[0]);
    EXPECT_EQ(nary.mNodeCount[1], chain.mNodeCount[1]);
    EXPECT_EQ(nary.mNodeCount[2], chain.mNodeCount[2]);
    EXPECT_EQ(nary.mVoxelCount,   chain.mVoxelCount);

    // Ground truth: |A u B| = 42, C disjoint (+27) => 69 unique voxels
    EXPECT_EQ(nary.mVoxelCount, 69u);

    // A single-element list returns a copy of that grid's topology
    auto soloH = nanovdb::tools::cuda::MergeGrids<BuildT>(
        std::vector<const GridT*>{gA}).getHandle();
    EXPECT_EQ(treeData(soloH.deviceGrid<BuildT>()).mVoxelCount, treeData(gA).mVoxelCount);

    // An empty input list is rejected
    EXPECT_THROW(nanovdb::tools::cuda::MergeGrids<BuildT>(
        std::vector<const GridT*>{}).getHandle(), std::runtime_error);
}// MergeGridsNary_ValueOnIndex

TEST(TestNanoVDBCUDA, GridHandle_from_HostBuffer)
{
    using namespace nanovdb;
    auto hostHandle = tools::createLevelSetSphere<float>(100, Vec3d(0),1,3, Vec3d(0), "test");

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

    {// longer version
        SingleBufT devBuffer(stream, hostHandle.bufferSize(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpyAsync(devBuffer.data(), hostHandle.data(), hostHandle.bufferSize(), cudaMemcpyHostToDevice, stream));
        EXPECT_EQ(hostHandle.bufferSize(), devBuffer.size());
        auto devHandle = GridHandle<SingleBufT>(std::move(devBuffer));

        // testing
        EXPECT_EQ(hostHandle.bufferSize(), devHandle.bufferSize());
        EXPECT_EQ(devBuffer.size(), 0);
        auto copyHandle = nanovdb::cuda::copyTo<HostBuffer>(devHandle, stream);
        ASSERT_EQ(hostHandle.bufferSize(), copyHandle.bufferSize());
        EXPECT_EQ(0, std::memcmp(hostHandle.data(), copyHandle.data(), hostHandle.bufferSize()));
    }
    {// compact version
        auto devHandle = nanovdb::cuda::copyTo<SingleBufT>(hostHandle, stream);

        // testing
        EXPECT_EQ(hostHandle.bufferSize(), devHandle.bufferSize());
        auto copyHandle = nanovdb::cuda::copyTo<HostBuffer>(devHandle, stream);
        ASSERT_EQ(hostHandle.bufferSize(), copyHandle.bufferSize());
        EXPECT_EQ(0, std::memcmp(hostHandle.data(), copyHandle.data(), hostHandle.bufferSize()));
    }
    cudaCheck(cudaStreamDestroy(stream));
}

TEST(TestNanoVDBCUDA, MeshToGrid_EmptyMesh)
{
    using BuildT = nanovdb::ValueOnIndex;

    nanovdb::Map map;
    map.set(0.1, nanovdb::Vec3d(0.0));

    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(nullptr, 0u, nullptr, 0u, map);
    converter.setVerbose(0);
    auto handle = converter.getHandle(SingleBufT());
    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    const auto* grid = hostHandle.grid<BuildT>();
    ASSERT_NE(grid, nullptr);
    EXPECT_EQ(grid->tree().activeVoxelCount(), 0);
}// MeshToGrid_EmptyMesh

TEST(TestNanoVDBCUDA, MeshToGrid_UnitTetrahedron)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Unit tetrahedron: four vertices, four triangular faces.
    // Vertex coordinates are in world space.
    const std::vector<nanovdb::Vec3f> hostPoints = {
        {0.f, 0.f, 0.f},  // p0
        {1.f, 0.f, 0.f},  // p1
        {0.f, 1.f, 0.f},  // p2
        {0.f, 0.f, 1.f},  // p3
    };
    // t0: z=0 face, t1: x=0 face, t2: y=0 face, t3: diagonal face (x+y+z=1)
    const std::vector<nanovdb::Vec3i> hostTriangles = {
        {0, 1, 2},
        {0, 2, 3},
        {0, 3, 1},
        {1, 2, 3},
    };

    auto nPoints = hostPoints.size();
    auto nTriangles = hostTriangles.size();

    // Upload points and triangles to device
    nanovdb::cuda::Buffer<nanovdb::Vec3f> pointsBuf(cudaStream_t(0), nPoints, nanovdb::cuda::noInit);
    ASSERT_TRUE(pointsBuf.data());
    cudaCheck(cudaMemcpy(pointsBuf.data(), hostPoints.data(),
                         nPoints * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));

    nanovdb::cuda::Buffer<nanovdb::Vec3i> trisBuf(cudaStream_t(0), nTriangles, nanovdb::cuda::noInit);
    ASSERT_TRUE(trisBuf.data());
    cudaCheck(cudaMemcpy(trisBuf.data(), hostTriangles.data(),
                         nTriangles * sizeof(nanovdb::Vec3i), cudaMemcpyHostToDevice));

    const nanovdb::Vec3f* dPoints = pointsBuf.data();
    const nanovdb::Vec3i* dTriangles = trisBuf.data();

    // Uniform-scale map: world = dx * index
    const double dx = 0.1;
    nanovdb::Map map;
    map.set(dx, nanovdb::Vec3d(0.0));

    const float bandWidth = 3.0f;  // default, in voxels
    const float bandWidthWorld = bandWidth * float(dx);

    // CPU brute-force UDF in index space (matches GPU arithmetic exactly):
    // transform world-space verts to index space, compute pointToTriangleDistSqr
    // with integer voxel centers, scale result to world space.
    std::array<nanovdb::Vec3f, 4> idxVerts;
    for (uint32_t i = 0; i < nPoints; ++i)
        idxVerts[i] = map.applyInverseMap(hostPoints[i]);

    auto cpuUDF = [&](int ix, int iy, int iz) -> float {
        const nanovdb::Vec3f p{float(ix), float(iy), float(iz)};
        float minDistSqr = std::numeric_limits<float>::max();
        for (const auto& tri : hostTriangles) {
            const float d = nanovdb::math::pointToTriangleDistSqr<nanovdb::Vec3f>(
                idxVerts[tri[0]], idxVerts[tri[1]], idxVerts[tri[2]], p);
            minDistSqr = std::min(minDistSqr, d);
        }
        return std::sqrt(minDistSqr) * float(dx);  // world-space distance
    };

    // --- Topology-only path ---
    uint64_t topoChecksum = 0;
    {
        nanovdb::tools::cuda::MeshToGrid<BuildT> conv(dPoints, nPoints, dTriangles, nTriangles, map);
        conv.setVerbose(0);
        conv.setChecksum(nanovdb::CheckMode::Full);
        auto handle = conv.getHandle(SingleBufT());
        auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
        const auto* topoGrid = hostHandle.grid<BuildT>();
        ASSERT_NE(topoGrid, nullptr);
        topoChecksum = topoGrid->mChecksum.full();
        EXPECT_GT(topoGrid->tree().activeVoxelCount(), uint64_t(0));
    }

    // --- UDF path ---
    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(dPoints, nPoints, dTriangles, nTriangles, map);
    converter.setVerbose(0);
    converter.setChecksum(nanovdb::CheckMode::Full);
    auto [handle, sidecarBuf] = converter.getHandleAndUDF(SingleBufT(), SingleBufT());

    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    const auto* grid = hostHandle.grid<BuildT>();
    ASSERT_NE(grid, nullptr);

    const uint64_t sidecarCount = sidecarBuf.size() / sizeof(float);
    std::vector<float> hostSidecar(sidecarCount);
    cudaCheck(cudaMemcpy(hostSidecar.data(), sidecarBuf.data(),
                         sidecarBuf.size(), cudaMemcpyDeviceToHost));

    // Per-voxel UDF correctness: every active voxel must lie inside the narrow band
    // and its sidecar distance must match the CPU brute-force value within 1e-3 voxels.
    const uint32_t nLeaves = grid->tree().nodeCount(0);
    const auto* leaves  = grid->tree().getFirstLeaf();
    uint64_t activeCount = 0;

    for (uint32_t li = 0; li < nLeaves; ++li) {
        const auto& leaf = leaves[li];
        const auto  org  = leaf.origin();
        for (int vi = 0; vi < 512; ++vi) {
            if (!leaf.isActive(vi)) continue;
            ++activeCount;

            const auto local = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(vi);
            const int ix = org[0]+local[0], iy = org[1]+local[1], iz = org[2]+local[2];

            const float exactUDF = cpuUDF(ix, iy, iz);
            ASSERT_LE(exactUDF, bandWidthWorld * (1.f + 1e-5f))
                << "Active voxel at (" << ix << "," << iy << "," << iz
                << ") is outside the narrow band (distance=" << exactUDF/float(dx) << " voxels)";

            const uint64_t sIdx = leaf.getValue(vi);
            ASSERT_LT(sIdx, sidecarCount) << "Sidecar index out of range";

            const float ourUDF    = hostSidecar[sIdx];
            const float errVoxels = std::abs(ourUDF - exactUDF) / float(dx);
            EXPECT_LT(errVoxels, 1e-3f)
                << "UDF error at (" << ix << "," << iy << "," << iz
                << "): ours=" << ourUDF/float(dx) << " exact=" << exactUDF/float(dx) << " voxels";
        }
    }
    EXPECT_GT(activeCount, uint64_t(0));

    // Build a flat set of active coord keys from the leaf iteration
    // Encode each (ix,iy,iz) as a uint64_t with 21 bits per axis, offset by 2^20.
    auto encodeCoord = [](int x, int y, int z) -> uint64_t {
        return (uint64_t(x + (1<<20)))
             | (uint64_t(y + (1<<20)) << 21)
             | (uint64_t(z + (1<<20)) << 42);
    };
    std::unordered_set<uint64_t> activeSet;
    activeSet.reserve(activeCount);
    for (uint32_t li = 0; li < nLeaves; ++li) {
        const auto& leaf = leaves[li];
        const auto  org  = leaf.origin();
        for (int vi = 0; vi < 512; ++vi) {
            if (!leaf.isActive(vi)) continue;
            const auto local = nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(vi);
            activeSet.insert(encodeCoord(org[0]+local[0], org[1]+local[1], org[2]+local[2]));
        }
    }

    // No false negatives: every voxel with CPU UDF strictly inside the band must be active.
    const int ilo = (int)std::floor(-bandWidth) - 1;
    const int ihi = (int)std::ceil(1.0 / dx + bandWidth) + 1;
    uint64_t missedCount = 0;
    for (int ix = ilo; ix <= ihi; ++ix)
        for (int iy = ilo; iy <= ihi; ++iy)
            for (int iz = ilo; iz <= ihi; ++iz)
                if (cpuUDF(ix, iy, iz) < bandWidthWorld)
                    if (activeSet.count(encodeCoord(ix, iy, iz)) == 0)
                        ++missedCount;
    EXPECT_EQ(missedCount, uint64_t(0));

    // getHandle() and getHandleAndUDF() must produce identical grids.
    EXPECT_EQ(grid->mChecksum.full(), topoChecksum);
}// MeshToGrid_UnitTetrahedron

// Checks a ConnectedComponents result against a CPU union-find over the 6-connected active voxels of
// a host copy of the grid. The labels of the active voxels must induce exactly the CPU partition,
// using every id in [0, N), and every other entry of the value-indexed label buffer, including slot 0
// (the background), must hold the all-ones sentinel.
static void checkConnectedComponentLabels(const nanovdb::NanoGrid<nanovdb::ValueOnIndex>& grid,
                                          const std::vector<uint32_t>& labels, uint32_t N)
{
    ASSERT_EQ(grid.valueCount(), labels.size());
    std::vector<uint64_t> parent(labels.size());// CPU union-find over value indices
    for (uint64_t i = 0; i < parent.size(); ++i) parent[i] = i;
    auto find = [&parent](uint64_t x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
    std::vector<bool> isVoxel(labels.size(), false);
    auto acc = grid.getAccessor();
    const auto* leaves = grid.tree().getFirstNode<0>();
    for (uint32_t l = 0; l < grid.tree().nodeCount(0); ++l) {
        for (uint32_t n = 0; n < 512u; ++n) {
            if (!leaves[l].isActive(n)) continue;
            const uint64_t i = leaves[l].getValue(n);
            isVoxel[i] = true;
            const nanovdb::Coord ijk = leaves[l].offsetToGlobalCoord(n);
            for (int axis = 0; axis < 3; ++axis) {// the +x, +y, +z faces cover every 6-adjacency once
                nanovdb::Coord nbr = ijk;
                nbr[axis] += 1;
                if (!acc.isActive(nbr)) continue;
                const uint64_t ri = find(i), rj = find(acc.getValue(nbr));
                if (ri != rj) parent[std::max(ri, rj)] = std::min(ri, rj);
            }
        }
    }
    std::vector<uint32_t> rootToLabel(labels.size(), ~0u);
    std::vector<uint64_t> labelToRoot(N, ~uint64_t(0));
    uint64_t sentinelErrors = 0, rangeErrors = 0, splitErrors = 0, mergeErrors = 0, usedLabels = 0;
    for (uint64_t i = 0; i < labels.size(); ++i) {
        if (!isVoxel[i]) { sentinelErrors += (labels[i] != ~0u); continue; }
        const uint32_t label = labels[i];
        if (label >= N) { ++rangeErrors; continue; }
        const uint64_t root = find(i);
        if (rootToLabel[root] == ~0u) rootToLabel[root] = label;
        else splitErrors += (rootToLabel[root] != label);// one CPU component, several labels
        if (labelToRoot[label] == ~uint64_t(0)) { labelToRoot[label] = root; ++usedLabels; }
        else mergeErrors += (labelToRoot[label] != root);// one label, several CPU components
    }
    EXPECT_EQ(0u, sentinelErrors);
    EXPECT_EQ(0u, rangeErrors);
    EXPECT_EQ(0u, splitErrors);
    EXPECT_EQ(0u, mergeErrors);
    EXPECT_EQ(uint64_t(N), usedLabels);
}// checkConnectedComponentLabels

// Runs the full ConnectedComponents pipeline on a device grid with the given active voxels, checks the
// returned labels with checkConnectedComponentLabels, and returns the component count N.
static uint32_t checkedComponentCount(const std::vector<nanovdb::Coord>& coords)
{
    using BuildT = nanovdb::ValueOnIndex;
    nanovdb::Coord* d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, coords.size() * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords.data(), coords.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, coords.size(), 1.0, SingleBufT());
    cudaCheck(cudaFree(d_coords));

    nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(handle.deviceGrid<BuildT>());
    auto [labels, N] = cc.getVoxelLabelsAndCount();// the contents are complete on return
    std::vector<uint32_t> hostLabels(labels.size());
    cudaCheck(cudaMemcpy(hostLabels.data(), labels.data(), hostLabels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    auto hostHandle = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));// for the CPU reference
    checkConnectedComponentLabels(*hostHandle.grid<BuildT>(), hostLabels, N);
    return N;
}// checkedComponentCount

TEST(TestNanoVDBCUDA, ConnectedComponents_MultiSphere)
{
    // Reproduce the connected-components pipeline's CC input: the rasterized *narrow band* of a
    // closed surface with the surface shell removed. The example prunes voxels whose unsigned
    // distance to the surface is within sqrt(3)/2 voxels (the barrier), leaving, for a solid sphere,
    // two disjoint concentric shells (inner + outer) => 2 components. N well-separated spheres each
    // contribute their own two shells (their narrow bands never touch) => 2N.
    const float band  = 3.0f;        // narrow-band half-width (voxels), matches the default bandWidth
    const float shell = 0.8660254f;  // sqrt(3)/2: the barrier shell the example removes

    auto sphereShells = [&](int N, float R, int spacing) {
        const int margin = int(band) + 2;
        const int x0 = -int(R) - margin, x1 = (N - 1) * spacing + int(R) + margin;
        const int yz = int(R) + margin;
        std::vector<nanovdb::Coord> coords;
        for (int x = x0; x <= x1; ++x)
            for (int y = -yz; y <= yz; ++y)
                for (int z = -yz; z <= yz; ++z)
                    for (int i = 0; i < N; ++i) {  // a separated voxel is in <=1 sphere's band
                        const float dx = float(x - i * spacing), dy = float(y), dz = float(z);
                        const float dd = std::fabs(std::sqrt(dx * dx + dy * dy + dz * dz) - R);
                        if (dd > shell && dd <= band) { coords.emplace_back(x, y, z); break; }
                    }
        return coords;
    };

    EXPECT_EQ(checkedComponentCount(sphereShells(1, 8.f, 30)), 2u);  // inner shell + outer shell
    EXPECT_EQ(checkedComponentCount(sphereShells(2, 8.f, 30)), 4u);  // 2 disjoint spheres -> 2*2
    EXPECT_EQ(checkedComponentCount(sphereShells(3, 8.f, 30)), 6u);  // 3 disjoint spheres -> 2*3

    // Two heavily-overlapping spheres are a single connected solid (a "peanut") with one boundary
    // surface, so the pruned narrow band of its union is one inner shell + one outer shell => 2.
    // The overlap merges what would be 4 (two separated spheres) down to 2. The narrow band follows
    // the *union* surface, i.e. the union SDF min(sdfA,sdfB); a small center separation (D = R) keeps
    // the neck fat so the inner shell does not pinch.
    {
        const float R = 8.f, D = 8.f;  // D < 2R => overlap
        const int   margin = int(band) + 2;
        const int   x0 = -int(R) - margin, x1 = int(D) + int(R) + margin;
        const int   yz = int(R) + margin;
        std::vector<nanovdb::Coord> coords;
        for (int x = x0; x <= x1; ++x)
            for (int y = -yz; y <= yz; ++y)
                for (int z = -yz; z <= yz; ++z) {
                    const float dxA = float(x), dxB = float(x) - D, fy = float(y), fz = float(z);
                    const float sdfA = std::sqrt(dxA*dxA + fy*fy + fz*fz) - R;
                    const float sdfB = std::sqrt(dxB*dxB + fy*fy + fz*fz) - R;
                    const float a    = std::fabs(std::min(sdfA, sdfB));  // |union SDF|
                    if (a > shell && a <= band) coords.emplace_back(x, y, z);
                }
        EXPECT_EQ(checkedComponentCount(coords), 2u);  // overlap merges -> inner shell + outer shell
    }
}// ConnectedComponents_MultiSphere

// Checks the per-voxel labels on inputs that the sphere shells above do not produce: a few hand-placed
// components with a known count, and a random fill whose leaves each hold many components and merge
// with their neighbors across all six faces.
TEST(TestNanoVDBCUDA, ConnectedComponents_VoxelLabels)
{
    {// known components
        std::vector<nanovdb::Coord> coords;
        for (int x = -4; x <= 20; ++x) coords.emplace_back(x, 0, 0);// a line through four leaves
        for (int i = 28; i <= 30; ++i)
            for (int j = 28; j <= 30; ++j)
                for (int k = -10; k <= -8; ++k) coords.emplace_back(i, j, k);// a 3^3 block across a leaf face
        coords.emplace_back(50, 50, 50);// two voxels sharing only an edge:
        coords.emplace_back(51, 51, 50);// two separate 6-connected components
        coords.emplace_back(100, -100, 100);// an isolated voxel
        EXPECT_EQ(5u, checkedComponentCount(coords));// the line, the block, the two edge-touching voxels and the isolated voxel
    }
    {// random fill of 64^3 voxels across 9^3 leaves, near the site-percolation threshold (~0.31), so
     // the clusters range from single voxels to ones spanning many leaves
        std::mt19937 rng(2261);
        std::bernoulli_distribution isActive(0.3);
        std::vector<nanovdb::Coord> coords;
        for (int x = -20; x < 44; ++x)
            for (int y = -20; y < 44; ++y)
                for (int z = -20; z < 44; ++z)
                    if (isActive(rng)) coords.emplace_back(x, y, z);
        checkedComponentCount(coords);// the labels are checked against the CPU partition
    }
}// ConnectedComponents_VoxelLabels

// Regression test: ConnectedComponents must read the source grid in order on its stream, so that a
// grid still being uploaded on that stream is read only after it arrives. As in
// NonBlockingStreamDeviceGridTraits, the upload is queued behind a busy-wait on a non-blocking stream,
// into a device buffer zeroed beforehand, and the operator runs on that stream right away; a header
// read that is not ordered on the stream sees an empty tree.
TEST(TestNanoVDBCUDA, ConnectedComponents_InFlightGrid)
{
    using BuildT = nanovdb::ValueOnIndex;
    using GridT  = nanovdb::NanoGrid<BuildT>;
    auto sphereHandle = nanovdb::tools::createLevelSetSphere<float>(30);
    auto hostHandle = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, BuildT>(*sphereHandle.grid<float>(), 0u, false, false);// no channels, stats or tiles
    const GridT* hostGrid = hostHandle.grid<BuildT>();
    ASSERT_TRUE(hostGrid);
    ASSERT_NE(0u, hostGrid->tree().nodeCount(0));
    const size_t gridBytes = hostHandle.bufferSize();

    void* pinned = nullptr;// pinned, so the upload below is truly asynchronous
    cudaCheck(cudaMallocHost(&pinned, gridBytes));
    std::memcpy(pinned, hostHandle.data(), gridBytes);
    void* d_bytes = nullptr;
    cudaCheck(cudaMalloc(&d_bytes, gridBytes));
    cudaStream_t nb = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking));
    cudaCheck(cudaMemsetAsync(d_bytes, 0, gridBytes, nb));
    cudaCheck(cudaStreamSynchronize(nb));
    int clockKHz = 0, dev = 0;
    cudaCheck(cudaGetDevice(&dev));
    cudaCheck(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev));
    streamBusyWaitKernel<<<1, 1, 0, nb>>>(static_cast<unsigned long long>(clockKHz) * 20ull);// ~20 ms
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpyAsync(d_bytes, pinned, gridBytes, cudaMemcpyHostToDevice, nb));
    {
        nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(static_cast<const GridT*>(d_bytes), nb);
        auto [labels, N] = cc.getVoxelLabelsAndCount();// the contents are complete on return
        std::vector<uint32_t> hostLabels(labels.size());
        cudaCheck(cudaMemcpy(hostLabels.data(), labels.data(), hostLabels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        EXPECT_EQ(1u, N);// the sphere's narrow band is a single shell
        checkConnectedComponentLabels(*hostGrid, hostLabels, N);
    }// the labels are freed on nb here, before the grid's memory below
    cudaCheck(cudaStreamSynchronize(nb));
    cudaCheck(cudaStreamDestroy(nb));
    cudaCheck(cudaFree(d_bytes));
    cudaCheck(cudaFreeHost(pinned));
}// ConnectedComponents_InFlightGrid
