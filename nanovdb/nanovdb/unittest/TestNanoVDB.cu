// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <vector>
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
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/Timer.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/cuda/DeviceStreamMap.h>
#include <nanovdb/cuda/DeviceMesh.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <algorithm>// for std::sort
#include <iomanip> // for std::setw, std::setfill
#include <thread> // for std::thread

namespace nanovdb {// this namespace is required by gtest

namespace test {

// used for testing cuda::DeviceBuffer
void device2host(size_t count)
{
    const size_t size = count * sizeof(float);
    auto buffer = nanovdb::cuda::DeviceBuffer::create(size, nullptr, false);// on device only
    EXPECT_EQ(size, buffer.size());
    EXPECT_FALSE(buffer.data());
    EXPECT_TRUE(buffer.deviceData());
    float *d_array = reinterpret_cast<float*>(buffer.deviceData());
    constexpr unsigned int num_threads = 256;
    unsigned int num_blocks = (static_cast<unsigned int>(count) + num_threads - 1) / num_threads;
    nanovdb::util::cuda::lambdaKernel<<<num_blocks, num_threads>>>(count, [=] __device__ (size_t i) {d_array[i] = float(i);});
    buffer.deviceDownload();// copy device -> host
    EXPECT_EQ(size, buffer.size());
    EXPECT_TRUE(buffer.data());
    EXPECT_TRUE(buffer.deviceData());
    float *array = reinterpret_cast<float*>(buffer.data());
    for (size_t i=0; i<count; ++i) EXPECT_EQ(array[i], float(i));
}// device2host

void host2device(size_t count)
{
    const size_t size = count * sizeof(float);
    auto hostBuffer = nanovdb::HostBuffer(size);
    for (size_t i=0; i<count; ++i) *hostBuffer.data<float>(i) = float(i);

    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) printf("kernel cuda error: %d\n", (int)err);

    auto devBuffer = nanovdb::cuda::DeviceBuffer::create(hostBuffer, dev);// on device only
    EXPECT_EQ(size, devBuffer.size());
    EXPECT_FALSE(devBuffer.data());
    EXPECT_TRUE(devBuffer.deviceData());
    float *d_array = reinterpret_cast<float*>(devBuffer.deviceData());
    constexpr unsigned int num_threads = 256;
    unsigned int num_blocks = (static_cast<unsigned int>(count) + num_threads - 1) / num_threads;

    bool *test, *d_test;
    cudaCheck(cudaMallocHost((void**)&test, sizeof(bool)));
    cudaCheck(cudaMalloc((void**)&d_test, sizeof(bool)));
    *test = true;
    cudaCheck(cudaMemcpyAsync(d_test, test, sizeof(bool), cudaMemcpyHostToDevice));// on host only

    nanovdb::util::cuda::lambdaKernel<<<num_blocks, num_threads>>>(count, [=] __device__ (size_t i) {
        if (d_array[i] != float(i)) *d_test = false;
        d_array[i] = float(i) + 1.0f;
    });
    cudaCheck(cudaMemcpy(test, d_test, sizeof(bool), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(*test);
    cudaCheck(cudaFreeHost(test));
    cudaCheck(cudaFree(d_test));
    devBuffer.deviceDownload();// copy device -> host
    EXPECT_EQ(size, devBuffer.size());
    EXPECT_TRUE(devBuffer.data());
    EXPECT_TRUE(devBuffer.deviceData());
    for (size_t i=0; i<count; ++i) EXPECT_EQ(*hostBuffer.data<float>(i) + 1.0f, *devBuffer.data<float>(i));
}// host2device

// used for testing cuda::DeviceBuffer
void host2device2host(size_t count)
{
    bool *test, *d_test;
    cudaCheck(cudaMallocHost((void**)&test, sizeof(bool)));
    cudaCheck(cudaMalloc((void**)&d_test, sizeof(bool)));
    *test = true;
    cudaCheck(cudaMemcpyAsync(d_test, test, sizeof(bool), cudaMemcpyHostToDevice));// on host only

    const size_t size = count * sizeof(float);
    auto buffer = nanovdb::cuda::DeviceBuffer::create(size);
    EXPECT_EQ(size, buffer.size());
    EXPECT_TRUE(buffer.data());
    EXPECT_FALSE(buffer.deviceData());
    float *array = reinterpret_cast<float*>(buffer.data());
    for (size_t i=0; i<count; ++i) array[i] = i;
    buffer.deviceUpload();// copy host -> device
    EXPECT_EQ(size, buffer.size());
    EXPECT_TRUE(buffer.data());
    EXPECT_TRUE(buffer.deviceData());
    float *d_array = reinterpret_cast<float*>(buffer.deviceData());
    constexpr unsigned int num_threads = 256;
    unsigned int num_blocks = (static_cast<unsigned int>(count) + num_threads - 1) / num_threads;
    nanovdb::util::cuda::lambdaKernel<<<num_blocks, num_threads>>>(count, [=] __device__ (size_t i) {
        if (d_array[i] != float(i)) *d_test = false;
        d_array[i] = float(i) + 1.0f;
    });
    cudaCheck(cudaMemcpy(test, d_test, sizeof(bool), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(*test);
    cudaCheck(cudaFreeHost(test));
    cudaCheck(cudaFree(d_test));
    buffer.deviceDownload();// copy device -> host
    EXPECT_EQ(size, buffer.size());
    EXPECT_TRUE(buffer.data());
    EXPECT_TRUE(buffer.deviceData());
    for (size_t i=0; i<count; ++i) EXPECT_EQ(array[i], float(i) + 1.0f);
}// host2device2host

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

TEST(TestNanoVDBCUDA, CudaDeviceBuffer)
{
    {
        nanovdb::cuda::DeviceBuffer buffer;
        EXPECT_EQ(0, buffer.deviceCount());
        EXPECT_EQ(0, buffer.bufferCount());
        EXPECT_EQ(0, buffer.size());
        EXPECT_TRUE(buffer.empty());
    }
    {
        nanovdb::cuda::DeviceBuffer buffer(1024);
        int count = 0;
        cudaGetDeviceCount(&count);
        EXPECT_EQ(count, buffer.deviceCount());
        EXPECT_EQ(1, buffer.bufferCount());
        EXPECT_EQ(1024, buffer.size());
        EXPECT_FALSE(buffer.empty());
    }
    nanovdb::test::device2host(1000);
    nanovdb::test::host2device(1000);
    nanovdb::test::host2device2host(1000);
}

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

    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points);
    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.size(), size);

    GridT *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy up the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points);
#endif
    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          2*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.size(), size);

    GridT *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy up the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points);
#endif

    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          2*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.size(), size);

    GridT *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy up the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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

TEST(TestNanoVDBCUDA, Large_CudaPointsToGrid_DeviceBuffer)
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

    timer.start("voxelsToGrid<DeviceBuffer>("+std::to_string(voxels.size())+" points)");
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, voxelCount, 1.0);
    timer.stop();

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_TRUE(handle.deviceGrid<BuildT>());
    EXPECT_FALSE(handle.deviceGrid<int>(0));
    EXPECT_TRUE(handle.deviceGrid<BuildT>(0));
    EXPECT_FALSE(handle.deviceGrid<BuildT>(1));
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
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
}// Large_CudaPointsToGrid_DeviceBuffer

TEST(TestNanoVDBCUDA, Large_CudaPointsToGrid_UnifiedBuffer)
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

    timer.start("voxelsToGrid<UnifiedBuffer>("+std::to_string(voxels.size())+" points)");
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT, nanovdb::Coord*, nanovdb::cuda::UnifiedBuffer>(d_coords, voxelCount, 1.0);
    timer.stop();

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
}// Large_CudaPointsToGrid_UnifiedBuffer

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
    EXPECT_EQ(size2, mergedHandle.size());
    EXPECT_FALSE(mergedHandle.isPadded());
    EXPECT_TRUE(mergedHandle.data());
    auto *gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(5u, gridData->mGridCount);
    EXPECT_EQ(0u, gridData->mGridIndex);
    EXPECT_EQ(handles2[0].size(), gridData->mGridSize);
    //timer.restart("unit-test host grids");
    for (int i=0; i<5; ++i){
        gridData = mergedHandle.gridData(i);
        EXPECT_TRUE(gridData);
        EXPECT_EQ(i, gridData->mGridIndex);
        EXPECT_EQ(handles2[i].size(), gridData->mGridSize);
        EXPECT_EQ(strcmp(gridNames[i+5].c_str(), gridData->mGridName),0);
    }

    EXPECT_FALSE(mergedHandle.empty());
    handles1.push_back(std::move(mergedHandle));// append one handle with 5 merged grids
    EXPECT_TRUE(mergedHandle.empty());
    EXPECT_EQ(6u, handles1.size());
    //timer.restart("merging 10 host grids");
    mergedHandle = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(handles1);
    EXPECT_EQ(size1 + size2, mergedHandle.size());
    EXPECT_TRUE(mergedHandle.data());
    gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(10u, gridData->mGridCount);
    EXPECT_EQ( 0u, gridData->mGridIndex);
    EXPECT_EQ(handles1[0].size(), gridData->mGridSize);

    //timer.restart("splitting host grids");
    auto splitHandles = nanovdb::splitGrids(mergedHandle);
    //timer.restart("unit-test split grids");
    EXPECT_EQ(10u, splitHandles.size());
    for (int i=0; i<5; ++i){
        EXPECT_EQ(handles1[i].size(), splitHandles[i].size());
        gridData = splitHandles[i].gridData();
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    for (int i=5; i<10; ++i){
        EXPECT_EQ(handles2[i-5].size(), splitHandles[i].size());
        gridData = splitHandles[i].gridData();
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    //timer.stop();
}//  mergeSplitGrids

TEST(TestNanoVDBCUDA, mergeSplitDeviceGrids)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    using HandleT = nanovdb::GridHandle<BufferT>;
    size_t size = 0;
    std::vector<HandleT> handles;
    std::vector<std::string> gridNames;
    //nanovdb::util::Timer timer("create 10 host grids");
    for (int radius = 100; radius<200; radius += 10) {
        gridNames.emplace_back("sphere_" + std::to_string(radius));
        handles.emplace_back(nanovdb::tools::createLevelSetSphere<float, BufferT>(radius,nanovdb::Vec3d(0),1,3,
                                                           nanovdb::Vec3d(0), gridNames.back()));
        EXPECT_FALSE(handles.back().isPadded());
        size += handles.back().bufferSize();
    }
    //timer.restart("copy grids to device");
    for (auto &h : handles) h.deviceUpload();
    EXPECT_EQ(10u, handles.size());
    //timer.restart("merging device grids");
    auto mergedHandle = nanovdb::cuda::mergeGridHandles<BufferT, std::vector>(handles);
    EXPECT_EQ(size, mergedHandle.size());
    EXPECT_FALSE(mergedHandle.data());
    EXPECT_TRUE(mergedHandle.deviceData());
    EXPECT_FALSE(mergedHandle.isPadded());
    //timer.restart("copy grids to host");
    mergedHandle.deviceDownload();
    EXPECT_TRUE(mergedHandle.data());
    EXPECT_TRUE(mergedHandle.deviceData());
    EXPECT_FALSE(mergedHandle.isPadded());
    auto *gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(10u, gridData->mGridCount);
    EXPECT_EQ(0u, gridData->mGridIndex);
    //timer.restart("unit-test host grids");
    for (uint32_t i=0; i<10; ++i) {
        gridData = mergedHandle.gridData(i);
        EXPECT_TRUE(gridData);
        EXPECT_EQ(i, gridData->mGridIndex);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    //timer.restart("splitting device grids");
    auto splitHandles = nanovdb::cuda::splitGridHandles<BufferT, std::vector>(mergedHandle);
    //timer.restart("unit-test split grids");
    EXPECT_EQ(10u, splitHandles.size());
    for (uint32_t i=0u; i<10u; ++i) {
        EXPECT_EQ(handles[i].size(), splitHandles[i].size());
        EXPECT_FALSE(splitHandles[i].isPadded());
        EXPECT_FALSE(splitHandles[i].gridData());
        splitHandles[i].deviceDownload();
        gridData = splitHandles[i].gridData();
        EXPECT_TRUE(gridData);
        EXPECT_EQ(0u, gridData->mGridIndex);
        EXPECT_EQ(1u, gridData->mGridCount);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    //timer.stop();
}//  mergeSplitDeviceGrids

// make -j 4 testNanoVDB && ./unittest/testNanoVDB --gtest_filter="*Cuda*" --gtest_break_on_failure
TEST(TestNanoVDBCUDA, CudaIndexGridToGrid_basic)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
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
    using BufferT = nanovdb::cuda::DeviceBuffer;
    auto idxHdl = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, nanovdb::ValueIndex, BufferT>(*floatGrid, 0u, false, false, 1);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    EXPECT_EQ(1u, idxHdl.buffer().bufferCount());
    EXPECT_TRUE(idxHdl.buffer().data(0, cudaCpuDeviceId));
    EXPECT_FALSE(idxHdl.buffer().data(0, 0));
    EXPECT_FALSE(idxHdl.deviceGrid<nanovdb::ValueIndex>());
    idxHdl.deviceUpload();
    EXPECT_TRUE(idxHdl.deviceGrid<nanovdb::ValueIndex>());
    EXPECT_EQ(2u, idxHdl.buffer().bufferCount());
    EXPECT_TRUE(idxHdl.buffer().data(0, cudaCpuDeviceId));
    EXPECT_TRUE(idxHdl.buffer().data(0, 0));
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
    EXPECT_FALSE(idxHdl.deviceGrid<float>());
    auto *d_idxGrid = idxHdl.deviceGrid<nanovdb::ValueIndex>();
    EXPECT_TRUE(d_idxGrid);
    //timer.restart("Call CudaIndexToGrid");
    auto hdl = nanovdb::tools::cuda::indexToGrid<float>(d_idxGrid, d_values);
    //timer.restart("unit-test");
    EXPECT_FALSE(hdl.grid<float>());// no host grid
    EXPECT_TRUE(hdl.deviceGrid<float>());
    hdl.deviceDownload();
    auto *floatGrid2 = hdl.grid<float>();
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
    using BufferT = nanovdb::cuda::DeviceBuffer;
    //nanovdb::util::Timer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::tools::createLevelSetSphere<float, BufferT>(100,nanovdb::Vec3d(0),1,3, nanovdb::Vec3d(0), "test");
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    //timer.restart("Create IndexGrid on CPU");
    auto idxHdl = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, BuildT, BufferT>(*floatGrid);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    idxHdl.deviceUpload();
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
    EXPECT_FALSE(idxHdl.deviceGrid<float>());
    auto *d_idxGrid = idxHdl.deviceGrid<BuildT>();
    EXPECT_TRUE(d_idxGrid);
    //timer.restart("Call CudaIndexToGrid");
    auto hdl = nanovdb::tools::cuda::indexToGrid<float>(d_idxGrid, d_values);
    //timer.restart("unit-test");
    EXPECT_FALSE(hdl.grid<float>());// no host grid
    EXPECT_TRUE(hdl.deviceGrid<float>());
    hdl.deviceDownload();
    auto *floatGrid2 = hdl.grid<float>();
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
    using BufferT = nanovdb::cuda::DeviceBuffer;
    //nanovdb::util::Timer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::tools::createLevelSetSphere<float, BufferT>(100,nanovdb::Vec3d(0),1,3, nanovdb::Vec3d(0), "test");
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    //timer.restart("Create IndexGrid on CPU");
    auto idxHdl = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, BuildT, BufferT>(*floatGrid);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    idxHdl.deviceUpload();
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
        EXPECT_FALSE(idxHdl.deviceGrid<FloatType>());
        auto *d_idxGrid = idxHdl.deviceGrid<BuildT>();
        EXPECT_TRUE(d_idxGrid);
        //timer.restart("Call CudaIndexToGrid");
        auto hdl = nanovdb::tools::cuda::indexToGrid<FloatType>(d_idxGrid, d_values);
        //timer.restart("unit-test");
        EXPECT_FALSE(hdl.template grid<FloatType>());// no host grid
        EXPECT_TRUE(hdl.template deviceGrid<FloatType>());
        hdl.deviceDownload();
        auto *floatGrid2 = hdl.template grid<FloatType>();
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
    using BufferT = nanovdb::cuda::DeviceBuffer;
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
    floatHdl.deviceUpload();// CPU -> GPU
    auto *d_floatGrid = floatHdl.deviceGrid<float>();
    EXPECT_TRUE(d_floatGrid);
    //timer.restart("Signed flood-fill on the GPU");
    //nanovdb::cuda::signedFloodFill(d_floatGrid, true);
    nanovdb::tools::cuda::signedFloodFill(d_floatGrid);
    //timer.restart("Copy FloatGrid from GPU to CPU");
    floatHdl.deviceDownload();// GPU -> CPU
    //timer.stop();
    floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    acc = floatGrid->getAccessor();
    EXPECT_EQ( 3.0f, acc(1003,0,0));
    EXPECT_EQ( 0.0f, acc(1000,0,0));
    EXPECT_EQ(-3.0f, acc( 997,0,0));
    //EXPECT_FALSE(floatGrid->isLexicographic());
    EXPECT_TRUE(floatGrid->isBreadthFirst());
}//  CudaSignedFloodFill

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
    auto handle = converter.getHandle(d_coords, num_points);
    cudaCheck(cudaFree(d_coords));
    //timer.stop();

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.size(), size);

    GridT *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    //timer.start("Copy data from GPU to CPU");
    handle.deviceDownload();// creates a copy up the CPU
    //timer.stop();
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_points, num_points);
    cudaCheck(cudaFree(d_points));
    //timer.stop();

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType) +
                          sizeof(nanovdb::GridBlindMetaData) +
                          num_points*sizeof(Vec3T);
    EXPECT_EQ(handle.size(), size);

    GridT *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    //timer.start("Copy data from GPU to CPU");
    handle.deviceDownload();// creates a copy on the CPU
    //timer.stop();
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_coords, num_points);
    //timer.stop();
    cudaCheck(cudaFree(d_coords));

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(8) +
                          8*sizeof(GridT::UpperNodeType) +
                          8*sizeof(GridT::LowerNodeType) +
                          8*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.size(), size);

    GridT *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    //timer.start("Copy data from GPU to CPU");
    handle.deviceDownload();// creates a copy up the CPU
    //timer.stop();
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_points, pointCount);
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_points, pointCount);
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_points, pointCount);
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_points, pointCount);
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
                test = (begin[i] - nanovdb::voxelToWorld(voxel, ijk, grid->map())).length() < 1e-9;
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
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
            bool test = false;
            for (uint64_t j=0; test == false && j<count; ++j) {
                test = (begin[i] - nanovdb::voxelToWorld(start[j], ijk, grid->map())).length() < 1e-6;
            }
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
    //auto handle = nanovdb::cudaPointsToGrid(d_points, pointCount, nanovdb::PointType::Voxel8);
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    converter.setPointType(nanovdb::PointType::Voxel8);
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
                test = (begin[i] - nanovdb::voxelToWorld(start[j], ijk, grid->map())).length() < 1e-2;
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
    //auto handle = nanovdb::cudaPointsToGrid(d_points, pointCount, nanovdb::PointType::Voxel8);
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose(2);
    converter.setPointType(nanovdb::PointType::PointID);
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    //timer.stop();
    cudaCheck(cudaFree(d_points));
    //std::cerr << "Grid size: " << (handle.size() >> 20) << " MB" << std::endl;

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
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    //timer.start("Allocating and copying grid from GPU to CPU");
    auto *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = converter.getHandle(d_coords, num_points);
    cudaCheck(cudaFree(d_coords));

    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU

    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.size(), size);

    GridT *grid = handle.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid);
    handle.deviceDownload();// creates a copy up the CPU
    EXPECT_TRUE(handle.deviceData());
    EXPECT_TRUE(handle.data());
    auto *data = handle.gridData();
    EXPECT_TRUE(data);
    grid = handle.grid<BuildT>();
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
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(d_coords, num_points);
    cudaCheck(cudaFree(d_coords));
    EXPECT_TRUE(handle.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle.data());// no grid was yet allocated on the CPU
    const uint64_t size = sizeof(GridT) +
                          sizeof(GridT::TreeType) +
                          GridT::RootType::memUsage(1) +
                          sizeof(GridT::UpperNodeType) +
                          sizeof(GridT::LowerNodeType) +
                          2*sizeof(GridT::LeafNodeType);
    EXPECT_EQ(handle.size(), size);
    GridT *d_grid = handle.deviceGrid<BuildT>();// no grid on the CPU
    EXPECT_TRUE(d_grid);
    float *d_blind = nullptr, blind[num_points] = {1.2f, 3.0f};
    cudaCheck(cudaMalloc(&d_blind, num_points * sizeof(float)));
    cudaCheck(cudaMemcpy(d_blind, blind, num_points * sizeof(float), cudaMemcpyHostToDevice));// CPU -> GPU

    //nanovdb::util::cuda::Timer timer("cudaAddBlindData");
    auto handle2 = nanovdb::tools::cuda::addBlindData(d_grid, d_blind, num_points);
    cudaCheck(cudaFree(d_blind));
    //timer.stop();
    EXPECT_TRUE(handle2.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle2.data());// no grid was yet allocated on the CPU
    EXPECT_EQ(handle2.size(), handle.size() + sizeof(nanovdb::GridBlindMetaData) + nanovdb::math::AlignUp<NANOVDB_DATA_ALIGNMENT>(num_points*sizeof(float)));

    auto *grid2 = handle2.grid<BuildT>();// no grid on the CPU
    EXPECT_FALSE(grid2);
    handle2.deviceDownload();// creates a copy on the CPU
    EXPECT_TRUE(handle2.deviceData());
    EXPECT_TRUE(handle2.data());
    auto *data = handle2.gridData();
    EXPECT_TRUE(data);
    grid2 = handle2.grid<BuildT>();
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

    GridT *d_grid2 = handle2.deviceGrid<BuildT>();// no grid on the CPU
    EXPECT_TRUE(d_grid2);

    nanovdb::Vec3f *d_blind2 = nullptr, blind2[num_points] = {nanovdb::Vec3f(1.2f), nanovdb::Vec3f(3.0f)};
    cudaCheck(cudaMalloc(&d_blind2, num_points * sizeof(nanovdb::Vec3f)));
    cudaCheck(cudaMemcpy(d_blind2, blind2, num_points * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));// CPU -> GPU

    auto handle3 = nanovdb::tools::cuda::addBlindData(d_grid2, d_blind2, num_points,
                                             nanovdb::GridBlindDataClass::AttributeArray,
                                             nanovdb::GridBlindDataSemantic::PointPosition,
                                             "this is a test");
    cudaCheck(cudaFree(d_blind2));
    handle3.deviceDownload();// creates a copy on the CPU
    GridT *grid3 = handle3.grid<BuildT>();// no grid on the CPU
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

TEST(TestNanoVDBCUDA, testGridHandleCopy)
{
    auto cudaHandle = nanovdb::tools::createLevelSetSphere<float, nanovdb::cuda::DeviceBuffer>(100);
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
        float min = std::numeric_limits<float>::max(), max = -min;
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
        auto cudaHandle = nanovdb::tools::cuda::voxelsToGrid<float>(nanovdb::make_fancy(d_coords), voxelCount);
#endif
        cudaCheck(cudaFree(d_coords));
        cudaHandle.deviceDownload();
        handle2 = cudaHandle.copy<nanovdb::HostBuffer>();
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
    auto handle = nanovdb::tools::createLevelSetSphere<float, nanovdb::cuda::DeviceBuffer>(100);
    EXPECT_TRUE(handle.data());
    auto *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    handle.deviceUpload();
    EXPECT_TRUE(handle.deviceData());
#if 0// entire grid or just GridData+TreeData+RootData
    const size_t size = handle.size();
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
    void *d_ptr = nanovdb::util::PtrAdd(handle.deviceData(), 16);
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
        //checksum2 = cs.checksum(0);// only check the checksum of grid, tree and root data
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
        nanovdb::tools::cuda::updateChecksum(handle.deviceGrid<float>(), nanovdb::CheckMode::Full);
        //gpuTimer.stop();
        nanovdb::Checksum fullChecksum2;
        cudaCheck(cudaMemcpy(&fullChecksum2, (const uint8_t*)handle.deviceGrid<float>() + 8, 8, cudaMemcpyDeviceToHost));
        EXPECT_EQ(checksum, fullChecksum2.head());
        EXPECT_EQ(fullChecksum, fullChecksum2);
    }
    cudaCheck(cudaFree(d_checksum));
}// CudaGridChecksum

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
    auto handle = nanovdb::tools::createLevelSetSphere<float, nanovdb::cuda::DeviceBuffer>(100);
    EXPECT_TRUE(handle.data());
    auto *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    handle.deviceUpload();
    auto *d_grid = handle.deviceGrid<float>();
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
        auto handle2 = nanovdb::cuda::createNodeManager(d_grid);
        //gpuTimer.stop();
        auto *d_mgr = handle2.deviceMgr<float>();
        EXPECT_TRUE(d_mgr);
        EXPECT_EQ(count, countActiveVoxels(d_mgr));
    }
}// NodeManager


TEST(TestNanoVDBCUDA, GridStats_UnifiedBuffer)
{

    using BufferT = nanovdb::cuda::UnifiedBuffer;
    using GridT = nanovdb::NanoGrid<float>;
    auto handle = nanovdb::tools::createLevelSetSphere<float, BufferT>(100,
                                                                                  nanovdb::Vec3d(0),
                                                                                  1.0,
                                                                                  3.0,
                                                                                  nanovdb::Vec3d(0),
                                                                                  "test",
                                                                                  nanovdb::tools::StatsMode::Disable);
    EXPECT_TRUE(handle.data());
    GridT *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    handle.deviceUpload();
    GridT *d_grid = handle.deviceGrid<float>();
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
}// GridStats_UnifiedBuffer

TEST(TestNanoVDBCUDA, GridStats_DeviceBuffer)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    using GridT = nanovdb::NanoGrid<float>;
    auto handle = nanovdb::tools::createLevelSetSphere<float, BufferT>(100,
                                                                                  nanovdb::Vec3d(0),
                                                                                  1.0,
                                                                                  3.0,
                                                                                  nanovdb::Vec3d(0),
                                                                                  "test",
                                                                                  nanovdb::tools::StatsMode::Disable);
    EXPECT_TRUE(handle.data());
    GridT *grid = handle.grid<float>();
    EXPECT_TRUE(grid);
    handle.deviceUpload();
    GridT *d_grid = handle.deviceGrid<float>();
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
}// GridStats_DeviceBuffer

// make -j && ./unittest/testNanoVDB --gtest_filter="*cudaIsValid*" --gtest_repeat=10
TEST(TestNanoVDBCUDA, cudaIsValid_DeviceBuffer)
{
    using BufferT = nanovdb::cuda::DeviceBuffer;
    const auto mode = nanovdb::CheckMode::Full;
    using GridT = nanovdb::NanoGrid<float>;
    nanovdb::util::Timer timer("createLevelSetSphere<DeviceBuffer>(500)");
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
    handle.deviceUpload();
    GridT *d_grid = handle.deviceGrid<float>();
    EXPECT_TRUE(d_grid);
    const bool verbose = false;

    EXPECT_TRUE(nanovdb::isValid(grid,        mode, verbose));
    EXPECT_TRUE(nanovdb::tools::cuda::isValid(d_grid,  mode, verbose));

    grid->mGridType = nanovdb::GridType::Vec3f;
    EXPECT_FALSE(nanovdb::isValid(grid,       mode, verbose));
    handle.deviceUpload();
    EXPECT_FALSE(nanovdb::tools::cuda::isValid(d_grid, mode, verbose));
}// cudaIsValid_DeviceBuffer

TEST(TestNanoVDBCUDA, cudaIsValid_UnifiedBuffer)
{
    using BufferT = nanovdb::cuda::UnifiedBuffer;
    const auto mode = nanovdb::CheckMode::Full;
    using GridT = nanovdb::NanoGrid<float>;
    nanovdb::util::Timer timer("createLevelSetSphere<UnifiedBuffer>(500)");
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
    handle.deviceUpload();
    GridT *d_grid = handle.deviceGrid<float>();
    EXPECT_TRUE(d_grid);
    const bool verbose = false;

    EXPECT_TRUE(nanovdb::isValid(grid,        mode, verbose));
    EXPECT_TRUE(nanovdb::tools::cuda::isValid(d_grid,  mode, verbose));

    grid->mGridType = nanovdb::GridType::Vec3f;
    EXPECT_FALSE(nanovdb::isValid(grid,       mode, verbose));
    handle.deviceUpload();
    EXPECT_FALSE(nanovdb::tools::cuda::isValid(d_grid, mode, verbose));
}// cudaIsValid_UnifiedBuffer

TEST(TestNanoVDBCUDA, cudaIsValid_HostBuffer)
{
    using BufferT = nanovdb::HostBuffer;
    const auto mode = nanovdb::CheckMode::Full;
    using GridT = nanovdb::NanoGrid<float>;
    nanovdb::util::Timer timer("createLevelSetSphere<UnifiedBuffer>(500)");
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

TEST(TestNanoVDBCUDA, overSizedDeviceBuffer)
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
    using BufferT = nanovdb::cuda::DeviceBuffer;
    auto overSizedBuffer = BufferT(2*gridSize, false);// only allocate the device buffer
    cudaMemcpy(overSizedBuffer.deviceData(), handle1.data(), gridSize, cudaMemcpyHostToDevice);

    // construct handle from over-sized device buffer and test it
    nanovdb::GridHandle<BufferT> handle2(std::move(overSizedBuffer));
    EXPECT_EQ(1         , handle2.gridCount());
    EXPECT_EQ(  gridSize, handle2.gridSize());
    EXPECT_EQ(  gridSize, handle2.totalGridSize());
    EXPECT_EQ(2*gridSize, handle2.bufferSize());
    EXPECT_EQ(  gridSize, handle2.freeSize());
    EXPECT_FALSE(handle2.isFull());
}// overSizedDeviceBuffer

__global__ void initKernel(int N, int *x)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] = 2;
}

// make -j && ./unittest/testNanoVDB --gtest_filter="*UnifiedBuffer*" --gtest_repeat=2
TEST(TestNanoVDBCUDA, UnifiedBuffer_basic)
{
    int device = 0;
    cudaSetDevice(device);// set the default device to GPU #0
    size_t free, total;
    cudaMemGetInfo( &free, &total );
    std::cout << "GPU #" << device << " memory: free = " << (free>>30) << " GB, total = " << (total>>30) << " GB\n";

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));
    const size_t N = 1<<20, size = N*sizeof(int), capacity = 20*size;// over-allocation

    nanovdb::cuda::UnifiedBuffer buffer(size, capacity);
    EXPECT_EQ(size, buffer.size());
    EXPECT_EQ(capacity, buffer.capacity());
    {// set N values on the host
        //buffer.deviceDownload(stream);
        buffer.prefetch(0, size, cudaCpuDeviceId, stream);
        nanovdb::util::Timer timer("Setting values on CPU with unified memory");
        int *x = buffer.data<int>();
        for (size_t i = 0; i < N; i++) *x++ = 1;
        timer.stop();
    }
    {// resize unified buffers
        nanovdb::util::Timer timer("resize unified buffer");
        buffer.resize(2*size);
        timer.stop();
        EXPECT_EQ(2*size, buffer.size());
        EXPECT_EQ(capacity, buffer.capacity());
    }
    {
        //buffer.deviceUpload(0, stream);
        buffer.prefetch(size, size, 0, stream);
        nanovdb::util::cuda::Timer timer("Setting values on GPU with unified memory", stream);
        static const int blockSize = 256, numBlocks = (N + blockSize - 1) / blockSize;
        initKernel<<<numBlocks, blockSize, 0, stream>>>(N, buffer.data<int>(N));
        timer.stop();
    }

    EXPECT_EQ(CUDA_SUCCESS, cudaStreamSynchronize(stream));
    int *x = buffer.data<int>();
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(1, *x++);
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(2, *x++);

    nanovdb::cuda::UnifiedBuffer otherBuffer(std::move(buffer));
    int *y = otherBuffer.data<int>();
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(1, *y++);
    for (size_t i = 0; i < N; ++i) EXPECT_EQ(2, *y++);
}// UnifiedBuffer_basic

TEST(TestNanoVDBCUDA, UnifiedBuffer_IO)
{
    //cudaSetDevice(0);// loads runtime context
    using BufferT = nanovdb::cuda::UnifiedBuffer;
    const size_t size = 8*(1ULL << 30);// 8 GB
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));
    {
        auto buffer = BufferT::create(size, nullptr, true, (void*)stream);
        EXPECT_EQ(CUDA_SUCCESS, cudaStreamSynchronize(stream));
        EXPECT_EQ(size, buffer.size());
        EXPECT_EQ(size, buffer.capacity());
    }
    {// size = capacity
        auto handle = nanovdb::io::readGrid<BufferT>("data/3_spheres.nvdb", 0);
        auto nanoGrid = handle.grid<float>();
        EXPECT_EQ(handle.size(), handle.buffer().capacity());
        EXPECT_EQ(handle.buffer().size(), handle.buffer().capacity());
        EXPECT_TRUE(nanoGrid);
        EXPECT_EQ(1u, handle.gridCount());
        auto *grid = handle.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(0u, grid->gridIndex());
        EXPECT_EQ(1u, grid->gridCount());
        EXPECT_TRUE(nanovdb::tools::validateChecksum(grid));
        EXPECT_TRUE(nanovdb::tools::validateChecksum(grid, nanovdb::CheckMode::Full));
    }
    {// size < capacity
        BufferT reference(0, size);
        auto handle = nanovdb::io::readGrid<BufferT>("data/3_spheres.nvdb", 0, 0, reference);
        auto nanoGrid = handle.grid<float>();
        EXPECT_LT(handle.buffer().size(), handle.buffer().capacity());
        EXPECT_EQ(size, handle.buffer().capacity());
        EXPECT_TRUE(nanoGrid);
        EXPECT_EQ(1u, handle.gridCount());
        auto *grid = handle.grid<float>();
        EXPECT_TRUE(grid);
        EXPECT_EQ(0u, grid->gridIndex());
        EXPECT_EQ(1u, grid->gridCount());
        EXPECT_TRUE(nanovdb::tools::validateChecksum(grid));
        EXPECT_TRUE(nanovdb::tools::validateChecksum(grid, nanovdb::CheckMode::Full));
    }
}// UnifiedBuffer_IO

TEST(TestNanoVDBCUDA, UnifiedBuffer_createLevelSetSphere)
{
    using BufferT = nanovdb::cuda::UnifiedBuffer;
    const int radius = 100, center = 50, width = 3, voxelSize = 1;
    const std::string gridName("sphere_" + std::to_string(radius));
    auto handle = nanovdb::tools::createLevelSetSphere<float, BufferT>(radius, nanovdb::Vec3d(center),
                                                                       voxelSize, width, nanovdb::Vec3d(0), gridName);

    EXPECT_TRUE(handle);
    EXPECT_EQ(1u, handle.gridCount());
    EXPECT_EQ(handle.size(), handle.buffer().capacity());
    auto* meta = handle.gridMetaData();
    EXPECT_TRUE(meta);
    EXPECT_EQ(gridName, std::string(meta->shortGridName()));
    EXPECT_EQ(nanovdb::GridType::Float, meta->gridType());
    EXPECT_EQ(nanovdb::GridClass::LevelSet, meta->gridClass());
    auto* dstGrid = handle.grid<float>();
    EXPECT_TRUE(dstGrid);
    EXPECT_EQ(dstGrid->gridSize(), handle.buffer().capacity());
    EXPECT_EQ(gridName, std::string(dstGrid->gridName()));

    EXPECT_TRUE(dstGrid->hasBBox());
    EXPECT_TRUE(dstGrid->hasMinMax());
    EXPECT_TRUE(dstGrid->hasAverage());
    EXPECT_TRUE(dstGrid->hasStdDeviation());

    EXPECT_NEAR( -3.0f, dstGrid->tree().root().minimum(), 0.04f);
    EXPECT_NEAR(  3.0f, dstGrid->tree().root().maximum(), 0.04f);
    EXPECT_NEAR(  0.0f, dstGrid->tree().root().average(), 0.30f);

    EXPECT_EQ(nanovdb::Coord(center - radius - 2), dstGrid->indexBBox()[0]);
    EXPECT_EQ(nanovdb::Coord(center + radius + 2), dstGrid->indexBBox()[1]);
} // UnifiedBuffer_createLevelSetSphere

template<class BuildT>
__global__ void testComputeStencilNeighborsKernel(
    nanovdb::NanoGrid<BuildT> *grid,
    uint32_t *firstLeafIDArray,
    uint64_t *jumpMapArray,
    uint64_t *stencilNeighborsArray)
{
    static constexpr int BlockWidth = 128;
    static constexpr int JumpMapLength = BlockWidth >> 6;
    int bID = blockIdx.x;
    int tID = threadIdx.x;

    uint32_t &firstLeafID = firstLeafIDArray[bID];
    uint64_t *jumpMap = jumpMapArray + JumpMapLength * bID;
    int firstOffset = 1;
    int blockFirstOffset = firstOffset + bID * BlockWidth;
    __shared__ uint32_t leafIndex[BlockWidth];
    __shared__ uint16_t voxelOffset[BlockWidth];

    nanovdb::tools::cuda::VoxelBlockManager<BlockWidth>::decodeInverseMaps(
        grid, firstLeafID, jumpMap, blockFirstOffset, &leafIndex[0], &voxelOffset[0]);

    uint64_t localNeighbors[27] = {};
    nanovdb::tools::cuda::VoxelBlockManager<BlockWidth>::computeBoxStencil(
        grid, &leafIndex[0], &voxelOffset[0], localNeighbors);
    __syncthreads();

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
    constexpr std::size_t BlockWidthLog2 = 7;
    constexpr std::size_t BlockWidth = 1 << BlockWidthLog2; // 128
    const std::size_t nVoxels = voxels.size();
    const std::size_t nBlocks = (nVoxels + BlockWidth - 1) >> BlockWidthLog2;
    nanovdb::Coord *deviceVoxels;
    cudaCheck(cudaMalloc(&deviceVoxels, nVoxels * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(deviceVoxels, voxels.data(), nVoxels * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice)); // CPU -> GPU
    auto handle = nanovdb::tools::cuda::voxelsToGrid<BuildT>(deviceVoxels, nVoxels);
    cudaCheck(cudaFree(deviceVoxels));
    EXPECT_TRUE((handle.deviceData())); // grid only exists on the GPU
    auto deviceGrid = handle.template deviceGrid<BuildT>();
    EXPECT_FALSE(handle.data()); // no grid was yet allocated on the CPU
    handle.deviceDownload();
    EXPECT_TRUE(handle.data()); // no grid was yet allocated on the CPU
    auto grid = handle.template grid<BuildT>();
    EXPECT_TRUE(grid);
    std::size_t nLowerNodes = grid->tree().nodeCount(1);

    // Construct VBM structure
    nanovdb::tools::VoxelBlockManagerHandle<BufferT> vbmHandle;
    vbmHandle.template deviceResize<BlockWidthLog2>(nBlocks);
    vbmHandle.setOffsets(1, nVoxels);
    vbmHandle.setLowerCount(nLowerNodes);

    nanovdb::tools::cuda::buildVoxelBlockManager<BlockWidthLog2>(
        vbmHandle.firstOffset(), vbmHandle.lastOffset(),
        vbmHandle.blockCount(), vbmHandle.lowerCount(),
        deviceGrid, vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap() );

    // Compute stencil neighbors
    const size_t neighborStencilSize = nBlocks * BlockWidth * 27 * sizeof(uint64_t);
    auto neighborStencilBuffer = nanovdb::cuda::DeviceBuffer::create(neighborStencilSize, nullptr, false);
    EXPECT_TRUE(neighborStencilBuffer.deviceData());
    testComputeStencilNeighborsKernel<<<nBlocks,BlockWidth>>>(deviceGrid,
        vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap(),
        reinterpret_cast<uint64_t*>(neighborStencilBuffer.deviceData()));
    cudaCheck(cudaGetLastError());

    // Verify stencil neighbors against results of standard accessors
    neighborStencilBuffer.deviceDownload();
    EXPECT_TRUE(neighborStencilBuffer.data());
    using StencilNeighborsType = uint64_t (*)[27];
    auto stencilNeighbors = reinterpret_cast<StencilNeighborsType>(neighborStencilBuffer.data());
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
    testVoxelBlockManager<nanovdb::ValueOnIndex,nanovdb::cuda::DeviceBuffer>();
}// VoxelBlockManager_ValueOnIndex

TEST(TestNanoVDBCUDA, VoxelBlockManager_ValueOnIndex_UnifiedBuffer)
{
    testVoxelBlockManager<nanovdb::ValueOnIndex,nanovdb::cuda::UnifiedBuffer>();
}// VoxelBlockManager_ValueOnIndex_UnifiedBuffer

TEST(TestNanoVDBCUDA, DilateInjectPrune_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Create an input (original) grid to use as input to dilation op
    std::vector<nanovdb::Coord> inputPoints;
    inputPoints.emplace_back(1,0,0); // Added nodes after dilation: 3 root, 3 upper, 3 lower, 3 leaf
    inputPoints.emplace_back(0,1,1); // Added nodes after dilation: 1 root, 1 upper, 1 lower, 1 leaf
    inputPoints.emplace_back(127,127,127); // Added nodes after dilation: 7 lower, 7 leaf
    auto inputBuffer = nanovdb::cuda::DeviceBuffer::create( inputPoints.size() * sizeof(nanovdb::Coord), nullptr, false);
    EXPECT_FALSE(inputBuffer.data());
    EXPECT_TRUE(inputBuffer.deviceData());
    cudaCheck(cudaMemcpy(inputBuffer.deviceData(), inputPoints.data(), inputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto inputHandle = converter.getHandle(static_cast<nanovdb::Coord*>(inputBuffer.deviceData()), inputPoints.size());
    EXPECT_TRUE(inputHandle.deviceGrid<BuildT>());
    auto inputGrid = inputHandle.deviceGrid<BuildT>();
    EXPECT_FALSE(inputHandle.grid<BuildT>());

    // Perform dilation
    nanovdb::tools::cuda::DilateGrid<BuildT> dilator( inputGrid );
    dilator.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
    dilator.setChecksum(nanovdb::CheckMode::Default);
    dilator.setVerbose(0);
    auto dilatedHandle = dilator.getHandle();
    auto dilatedGrid = dilatedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(dilatedGrid);
    auto dilatedTreeData = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(dilatedGrid);
    EXPECT_EQ(dilatedTreeData.mNodeCount[0], 13);
    EXPECT_EQ(dilatedTreeData.mNodeCount[1], 12);
    EXPECT_EQ(dilatedTreeData.mNodeCount[2], 5);
    EXPECT_EQ(dilatedTreeData.mVoxelCount, 73);

    // Create a prune mask (set bits correspond to retained voxels) from the occupancy of the original grid
    auto maskBuffer = nanovdb::cuda::DeviceBuffer::create( dilatedTreeData.mNodeCount[0] * sizeof(nanovdb::Mask<3>), nullptr, false);
    EXPECT_TRUE(maskBuffer.deviceData());
    auto leafMasks = static_cast<nanovdb::Mask<3>*>(maskBuffer.deviceData());
    constexpr unsigned int num_threads = 128;
    unsigned int num_blocks = (static_cast<unsigned int>(dilatedTreeData.mNodeCount[0]) + num_threads - 1) / num_threads;
    nanovdb::util::cuda::lambdaKernel<<<num_blocks, num_threads>>>(dilatedTreeData.mNodeCount[0],
        nanovdb::util::cuda::InjectGridMaskFunctor<BuildT>(),
        inputGrid, dilatedGrid, leafMasks);

    // Prune with computed mask
    nanovdb::tools::cuda::PruneGrid<BuildT> pruner( dilatedGrid, leafMasks );
    pruner.setChecksum(nanovdb::CheckMode::Default);
    pruner.setVerbose(0);
    auto prunedHandle = pruner.getHandle();
    auto prunedGrid = prunedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(prunedGrid);

    // The pruned grid should be identical to the original input
    EXPECT_FALSE(prunedHandle.grid<BuildT>());
    inputHandle.deviceDownload();
    prunedHandle.deviceDownload();
    EXPECT_TRUE(inputHandle.grid<BuildT>());
    EXPECT_TRUE(prunedHandle.grid<BuildT>());
    EXPECT_EQ(inputHandle.grid<BuildT>()->mChecksum.full(), prunedHandle.grid<BuildT>()->mChecksum.full());
}// DilateInjectPrune_ValueOnIndex

TEST(TestNanoVDBCUDA, RefineCoarsen_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Create an input (original) grid to use as input to refinement op
    std::vector<nanovdb::Coord> inputPoints;
    inputPoints.emplace_back(0,0,0); // Refinement doesn't introduce extra nodes
    inputPoints.emplace_back(4,0,0); // Refinement will introduce extra leaf
    inputPoints.emplace_back(0,64,0); // Refinement will introduce extra lower node
    inputPoints.emplace_back(0,0,2048); // Refinement will introduce extra upper node (and root tile)
    auto inputBuffer = nanovdb::cuda::DeviceBuffer::create( inputPoints.size() * sizeof(nanovdb::Coord), nullptr, false);
    EXPECT_FALSE(inputBuffer.data());
    EXPECT_TRUE(inputBuffer.deviceData());
    cudaCheck(cudaMemcpy(inputBuffer.deviceData(), inputPoints.data(), inputPoints.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    nanovdb::tools::cuda::PointsToGrid<BuildT> converter;
    converter.setChecksum(nanovdb::CheckMode::Default);
    auto inputHandle = converter.getHandle(static_cast<nanovdb::Coord*>(inputBuffer.deviceData()), inputPoints.size());
    EXPECT_TRUE(inputHandle.deviceGrid<BuildT>());
    auto inputGrid = inputHandle.deviceGrid<BuildT>();
    EXPECT_FALSE(inputHandle.grid<BuildT>());

    // Perform refinement
    nanovdb::tools::cuda::RefineGrid<BuildT> refiner( inputGrid );
    refiner.setChecksum(nanovdb::CheckMode::Default);
    refiner.setVerbose(0);
    auto refinedHandle = refiner.getHandle();
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
    auto coarsenedHandle = coarsener.getHandle();
    auto coarsenedGrid = coarsenedHandle.deviceGrid<BuildT>();
    EXPECT_TRUE(coarsenedGrid);

    // The coarsened grid should be identical to the original input
    EXPECT_FALSE(coarsenedHandle.grid<BuildT>());
    inputHandle.deviceDownload();
    coarsenedHandle.deviceDownload();
    EXPECT_TRUE(inputHandle.grid<BuildT>());
    EXPECT_TRUE(coarsenedHandle.grid<BuildT>());
    EXPECT_EQ(inputHandle.grid<BuildT>()->mChecksum.full(), coarsenedHandle.grid<BuildT>()->mChecksum.full());
}// RefineCoarsen_ValueOnIndex

TEST(TestNanoVDBCUDA, MergeGrids_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;

    // Create the first input grid for the merge op
    std::vector<nanovdb::Coord> inputPointsA;
    for (int i = 0; i <= 2; i++)
        for (int j = 0; j <= 2; j++)
            for (int k = 0; k <= 2; k++)
                inputPointsA.emplace_back(i-1, j*8-1, k*128); // 4 upper, 12 lower, 18 leaf nodes
    auto inputBufferA = nanovdb::cuda::DeviceBuffer::create( inputPointsA.size() * sizeof(nanovdb::Coord), nullptr, false);
    EXPECT_FALSE(inputBufferA.data());
    EXPECT_TRUE(inputBufferA.deviceData());
    cudaCheck(cudaMemcpy(inputBufferA.deviceData(), inputPointsA.data(), inputPointsA.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto inputHandleA = nanovdb::tools::cuda::voxelsToGrid<BuildT>(static_cast<nanovdb::Coord*>(inputBufferA.deviceData()), inputPointsA.size());
    EXPECT_TRUE(inputHandleA.deviceGrid<BuildT>());
    auto inputGridA = inputHandleA.deviceGrid<BuildT>();
    EXPECT_FALSE(inputHandleA.grid<BuildT>());

    // Create the second input grid for the merge op
    std::vector<nanovdb::Coord> inputPointsB;
    for (int i = 0; i <= 2; i++)
        for (int j = 0; j <= 2; j++)
            for (int k = 0; k <= 2; k++)
                inputPointsB.emplace_back(i, j*8-1, (k-1)*128); // 4 upper, 6 lower, 9 leaf nodes
    auto inputBufferB = nanovdb::cuda::DeviceBuffer::create( inputPointsB.size() * sizeof(nanovdb::Coord), nullptr, false);
    EXPECT_FALSE(inputBufferB.data());
    EXPECT_TRUE(inputBufferB.deviceData());
    cudaCheck(cudaMemcpy(inputBufferB.deviceData(), inputPointsB.data(), inputPointsB.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto inputHandleB = nanovdb::tools::cuda::voxelsToGrid<BuildT>(static_cast<nanovdb::Coord*>(inputBufferB.deviceData()), inputPointsB.size());
    EXPECT_TRUE(inputHandleB.deviceGrid<BuildT>());
    auto inputGridB = inputHandleB.deviceGrid<BuildT>();
    EXPECT_FALSE(inputHandleB.grid<BuildT>());

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

TEST(TestNanoVDBCUDA, GridHandle_from_HostBuffer)
{
    using namespace nanovdb;
    using BufferT = nanovdb::cuda::DeviceBuffer;
    auto hostHandle = tools::createLevelSetSphere<float>(100, Vec3d(0),1,3, Vec3d(0), "test");

    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    EXPECT_EQ(err, cudaSuccess);
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

    {// longer version
        auto devBuffer = BufferT::create(hostHandle.buffer(), dev, stream);
        EXPECT_EQ(hostHandle.bufferSize(), devBuffer.size());
        auto devHandle = GridHandle<BufferT>(std::move(devBuffer));

        // testing
        EXPECT_EQ(hostHandle.bufferSize(), devHandle.bufferSize());
        EXPECT_EQ(devBuffer.size(), 0);
        devHandle.deviceDownload(stream);
        for (uint64_t i=0; i<hostHandle.bufferSize(); ++i) {
            EXPECT_EQ(*hostHandle.buffer().data<char>(i), *devHandle.buffer().data<char>(i));
        }
    }
    {// compact version
        auto devHandle = GridHandle<BufferT>(BufferT::create(hostHandle.buffer(), dev, stream));

        // testing
        EXPECT_EQ(hostHandle.bufferSize(), devHandle.bufferSize());
        devHandle.deviceDownload(stream);
        for (uint64_t i=0; i<hostHandle.bufferSize(); ++i) {
            EXPECT_EQ(*hostHandle.buffer().data<char>(i), *devHandle.buffer().data<char>(i));
        }
    }
}

