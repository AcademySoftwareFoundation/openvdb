// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <vector>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/CreateNanoGrid.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/cuda/CudaUtils.h>
#include <nanovdb/util/cuda/CudaSignedFloodFill.h>
#include <nanovdb/util/cuda/CudaPointsToGrid.h>
#include <nanovdb/util/cuda/CudaIndexToGrid.h>
#include <nanovdb/util/cuda/CudaAddBlindData.h>
#include <nanovdb/util/CpuTimer.h>
#include <nanovdb/util/IO.h>

#include <gtest/gtest.h>

namespace nanovdb {// this namespace is required by gtest

namespace test {
// used for testing CudaDeviceBuffer
void device2host(size_t count)
{
    const size_t size = count * sizeof(float);
    auto buffer = nanovdb::CudaDeviceBuffer::create(size, nullptr, false);// on device only
    EXPECT_EQ(size, buffer.size());
    EXPECT_FALSE(buffer.data());
    EXPECT_TRUE(buffer.deviceData());
    float *d_array = reinterpret_cast<float*>(buffer.deviceData());
    constexpr unsigned int num_threads = 256;
    unsigned int num_blocks = num_blocks = (static_cast<unsigned int>(count) + num_threads - 1) / num_threads;
    cudaLambdaKernel<<<num_blocks, num_threads>>>(count, [=] __device__ (size_t i) {d_array[i] = float(i);});
    buffer.deviceDownload();// copy device -> host
    EXPECT_EQ(size, buffer.size());
    EXPECT_TRUE(buffer.data());
    EXPECT_TRUE(buffer.deviceData());
    float *array = reinterpret_cast<float*>(buffer.data());
    for (size_t i=0; i<count; ++i) EXPECT_EQ(array[i], float(i));
}// device2host
// used for testing CudaDeviceBuffer
void host2device2host(size_t count)
{
    bool *test, *d_test;
    cudaCheck(cudaMallocHost((void**)&test, sizeof(bool)));
    cudaCheck(cudaMalloc((void**)&d_test, sizeof(bool)));
    *test = true;
    cudaCheck(cudaMemcpyAsync(d_test, test, sizeof(bool), cudaMemcpyHostToDevice));// on host only

    const size_t size = count * sizeof(float);
    auto buffer = nanovdb::CudaDeviceBuffer::create(size);
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
    unsigned int num_blocks = num_blocks = (static_cast<unsigned int>(count) + num_threads - 1) / num_threads;
    cudaLambdaKernel<<<num_blocks, num_threads>>>(count, [=] __device__ (size_t i) {
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
void cudaStr()
{
    const size_t size = 50;
    char *str, *d_str;
    cudaCheck(cudaMallocHost((void**)&str, size));
    cudaCheck(cudaMalloc(  (void**)&d_str, size));
    int n, *d_n;
    cudaCheck(cudaMalloc((void**)&d_n, sizeof(int)));

    cudaLambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        cudaStrcpy(d_str, "this is a test");
    });
    cudaCheck(cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost));
    EXPECT_STREQ(str, "this is a test");
    cudaLambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        cudaStrcat(d_str, " #2");
    });
    cudaCheck(cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost));
    EXPECT_STREQ(str, "this is a test #2");

    cudaLambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        *d_n = cudaStrcmp(d_str, "this is a test");
    });
    cudaCheck(cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost));
    //std::cerr << "n = " << n << std::endl;
    EXPECT_EQ(std::strcmp(str, "this is a test"), n);
    cudaLambdaKernel<<<1, 1>>>(1, [=] __device__ (size_t) {
        *d_n = cudaStrcmp(d_str, "this is a test #2");
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
   nanovdb::test::device2host(1000);
   nanovdb::test::host2device2host(1000);
}

TEST(TestNanoVDBCUDA, CudaStr)
{
   nanovdb::test::cudaStr();
}

TEST(TestNanoVDBCUDA, Basic_CudaPointsToGrid_float)
{
    using BuildT = float;
    using GridT = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 1;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

    nanovdb::CudaPointsToGrid<BuildT> converter;
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

template <typename BuildT>
struct AccessLeafMask;

// template specialization of AccessLeafMask wrt ValueOnIndexMask
template <>
struct AccessLeafMask<ValueOnIndexMask>{
    __hostdev__ static bool get(const NanoRoot<ValueOnIndexMask>&) {return false;}
    __hostdev__ static bool get(const typename NanoRoot<ValueOnIndexMask>::Tile&) {return false;}
    __hostdev__ static bool get(const NanoUpper<ValueOnIndexMask>&, uint32_t) {return false;}
    __hostdev__ static bool get(const NanoLower<ValueOnIndexMask>&, uint32_t) {return false;}
    __hostdev__ static bool get(const NanoLeaf<ValueOnIndexMask> &leaf, uint32_t n) {return leaf.mMask.isOn(n);}
    __hostdev__ static void set(NanoRoot<ValueOnIndexMask>&) {}
    __hostdev__ static void set(typename NanoRoot<ValueOnIndexMask>::Tile&) {}
    __hostdev__ static void set(NanoUpper<ValueOnIndexMask>&, uint32_t) {}
    __hostdev__ static void set(NanoLower<ValueOnIndexMask>&, uint32_t) {}
    __hostdev__ static void set(NanoLeaf<ValueOnIndexMask> &leaf, uint32_t n) {leaf.mMask.setOn(n);}
};// AccessLeafMask<BuildT>

}// end of test namespace
}// end of nanovdb namespace

TEST(TestNanoVDBCUDA, Basic_CudaPointsToGrid_ValueIndex)
{
    using BuildT = nanovdb::ValueIndex;
    using GridT = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 3;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3),
                                         nanovdb::Coord(1, 2, 4),
                                         nanovdb::Coord(8, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU
#if 0
    nanovdb::CudaPointsToGrid converter;
    auto handle = converter.getHandle<BuildT>(d_coords, num_points);
#else
    auto handle = nanovdb::cudaVoxelsToGrid<BuildT>(d_coords, num_points);
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
        const auto offset = leaf->CoordToOffset(ijk);
        EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
    }
}// Basic_CudaPointsToGrid_ValueIndex

TEST(TestNanoVDBCUDA, Basic_CudaPointsToGrid_ValueOnIndex)
{
    using BuildT = nanovdb::ValueOnIndex;
    using GridT = nanovdb::NanoGrid<BuildT>;
    EXPECT_TRUE(nanovdb::BuildTraits<BuildT>::is_index);
    EXPECT_FALSE(nanovdb::BuildTraits<BuildT>::is_indexmask);
    EXPECT_TRUE(nanovdb::BuildTraits<BuildT>::is_onindex);
    EXPECT_FALSE(nanovdb::BuildTraits<BuildT>::is_offindex);
    const size_t num_points = 3;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3),
                                         nanovdb::Coord(1, 2, 4),
                                         nanovdb::Coord(8, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

#if 0
    nanovdb::CudaPointsToGrid converter;
    auto handle = converter.getHandle<BuildT>(d_coords, num_points);
#else
    auto handle = nanovdb::cudaVoxelsToGrid<BuildT>(d_coords, num_points);
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

TEST(TestNanoVDBCUDA, Basic_CudaPointsToGrid_ValueOnIndexMask)
{
    using BuildT = nanovdb::ValueOnIndexMask;
    using GridT = nanovdb::NanoGrid<BuildT>;
    EXPECT_TRUE(nanovdb::BuildTraits<BuildT>::is_index);
    EXPECT_TRUE(nanovdb::BuildTraits<BuildT>::is_indexmask);
    EXPECT_TRUE(nanovdb::BuildTraits<BuildT>::is_onindex);
    EXPECT_FALSE(nanovdb::BuildTraits<BuildT>::is_offindex);
    const size_t num_points = 3;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3),
                                         nanovdb::Coord(1, 2, 4),
                                         nanovdb::Coord(8, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

#if 0
    nanovdb::CudaPointsToGrid converter;
    auto handle = converter.getHandle<BuildT>(d_coords, num_points);
#else
    auto handle = nanovdb::cudaVoxelsToGrid<BuildT>(d_coords, num_points);
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
    EXPECT_EQ(4u, grid->valueCount());

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

    using OpT = nanovdb::test::AccessLeafMask<BuildT>;
    EXPECT_EQ(false, acc.get<OpT>(nanovdb::Coord(0,2,3)));
    EXPECT_EQ(true,  acc.get<OpT>(nanovdb::Coord(1,2,3)));
    EXPECT_EQ(true,  acc.get<OpT>(nanovdb::Coord(1,2,4)));
    EXPECT_EQ(true,  acc.get<OpT>(nanovdb::Coord(8,2,3)));

    acc.set<OpT>(nanovdb::Coord(1,2,3));
    acc.set<OpT>(nanovdb::Coord(8,2,3));

    EXPECT_EQ(false, acc.get<OpT>(nanovdb::Coord(0,2,3)));
    EXPECT_EQ(true , acc.get<OpT>(nanovdb::Coord(1,2,3)));
    EXPECT_EQ(true,  acc.get<OpT>(nanovdb::Coord(1,2,4)));
    EXPECT_EQ(true,  acc.get<OpT>(nanovdb::Coord(8,2,3)));

    {
        using T = nanovdb::ProbeValue<BuildT>;
        uint64_t value = 0;
        EXPECT_EQ(false, acc.get<T>(nanovdb::Coord(0,2,3), value) );
        EXPECT_EQ(0u,    value);
        EXPECT_EQ(true,  acc.get<T>(nanovdb::Coord(1,2,3), value) );
        EXPECT_EQ(1u,    value);
        EXPECT_EQ(true,  acc.get<T>(nanovdb::Coord(1,2,4), value) );
        EXPECT_EQ(2u,    value);
        EXPECT_EQ(true,  acc.get<T>(nanovdb::Coord(8,2,3), value) );
        EXPECT_EQ(3u,    value);
        EXPECT_EQ(false, acc.get<T>(nanovdb::Coord(-18,2,3), value) );
        EXPECT_EQ(0u,    value);

        EXPECT_EQ(false, grid->tree().get<T>(nanovdb::Coord(0,2,3), value) );
        EXPECT_EQ(0u,    value);
        EXPECT_EQ(true,  grid->tree().get<T>(nanovdb::Coord(1,2,3), value) );
        EXPECT_EQ(1u,    value);
        EXPECT_EQ(true,  grid->tree().get<T>(nanovdb::Coord(1,2,4), value) );
        EXPECT_EQ(2u,    value);
        EXPECT_EQ(true,  grid->tree().get<T>(nanovdb::Coord(8,2,3), value) );
        EXPECT_EQ(3u,    value);
        EXPECT_EQ(false, grid->tree().get<T>(nanovdb::Coord(-18,2,3), value) );
        EXPECT_EQ(0u,    value);
    }

    for (size_t i=0; i<num_points; ++i)  {
        const nanovdb::Coord ijk = coords[i];
        const auto *leaf = acc.get<nanovdb::GetLeaf<BuildT>>(ijk);
        EXPECT_TRUE(leaf);
        const auto offset = leaf->CoordToOffset(ijk);
        EXPECT_EQ(ijk, leaf->offsetToGlobalCoord(offset));
        EXPECT_EQ(leaf->mValueMask, leaf->mMask);
    }
}// Basic_CudaPointsToGrid_ValueOnIndexMask

TEST(TestNanoVDBCUDA, Large_CudaPointsToGrid_old)
{
    using BuildT = nanovdb::ValueOnIndex;
    nanovdb::CpuTimer timer;
    const size_t voxelCount = 1 << 20;// 1048576
    std::vector<nanovdb::Coord> voxels;
    {//generate random voxels
        voxels.reserve(voxelCount);
        std::srand(98765);
        const int max = 512, min = -max;
        auto op = [&](){return rand() % (max - min) + min;};
        timer.start("Creating "+std::to_string(voxelCount)+" random voxels on the CPU");
        while (voxels.size() < voxelCount) voxels.push_back(nanovdb::Coord(op(), op(), op()));
        timer.stop();
        EXPECT_EQ(voxelCount, voxels.size());
    }
#if 0
    {// Build grid on CPU
        nanovdb::build::Grid<float> buildGrid(0.0f);
        timer.start("Building grid on CPU from "+std::to_string(voxels.size())+" points");
        nanovdb::forEach(0, voxelCount, voxelCount >> 6, [&](const nanovdb::Range1D &r){
            auto acc = buildGrid.getWriteAccessor();
            for (size_t i=r.begin(); i!=r.end(); ++i) acc.setValueOn(voxels[i]);
        });
        timer.restart("Converting CPU build::Grid to nanovdb");
        auto handle = nanovdb::createNanoGrid(buildGrid);
        timer.stop();
    }
#endif
    nanovdb::Coord* d_coords;
    const size_t voxelSize = voxels.size() * sizeof(nanovdb::Coord);
    //timer.start("Allocating "+std::to_string(voxelSize >> 20)+" MB on the GPU");
    cudaCheck(cudaMalloc(&d_coords, voxelSize));
    //timer.restart("Copying voxels from CPU to GPU");
    cudaCheck(cudaMemcpy(d_coords, voxels.data(), voxelSize, cudaMemcpyHostToDevice));
    //timer.stop();

    timer.start("Building grid on GPU from "+std::to_string(voxels.size())+" points");
    nanovdb::CudaPointsToGrid<BuildT> converter;
    //converter.setVerbose();
    auto handle = converter.getHandle(d_coords, voxelCount);
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
    nanovdb::forEach(voxels,[&](const nanovdb::Range1D &r){
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
}// Large_CudaPointsToGrid_old

TEST(TestNanoVDBCUDA, mergeSplitGrids)
{
    size_t size1 = 0, size2 = 0;
    std::vector<nanovdb::GridHandle<>> handles1, handles2;
    std::vector<std::string> gridNames;
    nanovdb::CpuTimer timer("create 5 host grids");
    for (int radius = 100; radius<150; radius += 10) {
        gridNames.emplace_back("sphere_" + std::to_string(radius));
        handles1.emplace_back(nanovdb::createLevelSetSphere(radius,nanovdb::Vec3d(0),1,3,
                                                            nanovdb::Vec3d(0), gridNames.back()));
        EXPECT_FALSE(handles1.back().isPadded());
        size1 += handles1.back().size();
    }
    EXPECT_EQ(5u, gridNames.size());
    EXPECT_EQ(5u, handles1.size());
    timer.restart("create 5 host grids");
    for (int radius = 150; radius<200; radius += 10) {
        gridNames.emplace_back("sphere_" + std::to_string(radius));
        handles2.emplace_back(nanovdb::createLevelSetSphere(radius,nanovdb::Vec3d(0),1,3,
                                                            nanovdb::Vec3d(0), gridNames.back()));
        size2 += handles2.back().size();
    }
    EXPECT_EQ(10u, gridNames.size());
    EXPECT_EQ( 5u, handles2.size());
    timer.restart("merging 5 host grids");
    auto mergedHandle = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(handles2);// merge last 5 grid handles
    EXPECT_EQ(size2, mergedHandle.size());
    EXPECT_FALSE(mergedHandle.isPadded());
    EXPECT_TRUE(mergedHandle.data());
    auto *gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(5u, gridData->mGridCount);
    EXPECT_EQ(0u, gridData->mGridIndex);
    EXPECT_EQ(handles2[0].size(), gridData->mGridSize);
    timer.restart("unit-test host grids");
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
    timer.restart("merging 10 host grids");
    mergedHandle = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(handles1);
    EXPECT_EQ(size1 + size2, mergedHandle.size());
    EXPECT_TRUE(mergedHandle.data());
    gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(10u, gridData->mGridCount);
    EXPECT_EQ( 0u, gridData->mGridIndex);
    EXPECT_EQ(handles1[0].size(), gridData->mGridSize);

    timer.restart("splitting host grids");
    auto splitHandles = nanovdb::splitGrids(mergedHandle);
    timer.restart("unit-test split grids");
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
    timer.stop();
}//  mergeSplitGrids

TEST(TestNanoVDBCUDA, mergeSplitDeviceGrids)
{
    using BufferT = nanovdb::CudaDeviceBuffer;
    using HandleT = nanovdb::GridHandle<BufferT>;
    size_t size = 0;
    std::vector<HandleT> handles;
    std::vector<std::string> gridNames;
    nanovdb::CpuTimer timer("create 10 host grids");
    for (int radius = 100; radius<200; radius += 10) {
        gridNames.emplace_back("sphere_" + std::to_string(radius));
        handles.emplace_back(nanovdb::createLevelSetSphere<float, BufferT>(radius,nanovdb::Vec3d(0),1,3,
                                                           nanovdb::Vec3d(0), gridNames.back()));
        EXPECT_FALSE(handles.back().isPadded());
        size += handles.back().size();
    }
    timer.restart("copy grids to device");
    for (auto &h : handles) h.deviceUpload();
    EXPECT_EQ(10u, handles.size());
    timer.restart("merging device grids");
    auto mergedHandle = nanovdb::mergeDeviceGrids<BufferT, std::vector>(handles);
    EXPECT_EQ(size, mergedHandle.size());
    EXPECT_FALSE(mergedHandle.data());
    EXPECT_TRUE(mergedHandle.deviceData());
    EXPECT_FALSE(mergedHandle.isPadded());
    timer.restart("copy grids to host");
    mergedHandle.deviceDownload();
    EXPECT_TRUE(mergedHandle.data());
    EXPECT_TRUE(mergedHandle.deviceData());
    EXPECT_FALSE(mergedHandle.isPadded());
    auto *gridData = mergedHandle.gridData();// first grid
    EXPECT_TRUE(gridData);
    EXPECT_EQ(10u, gridData->mGridCount);
    EXPECT_EQ(0u, gridData->mGridIndex);
    timer.restart("unit-test host grids");
    for (uint32_t i=0; i<10; ++i) {
        gridData = mergedHandle.gridData(i);
        EXPECT_TRUE(gridData);
        EXPECT_EQ(i, gridData->mGridIndex);
        EXPECT_EQ(strcmp(gridNames[i].c_str(), gridData->mGridName),0);
    }
    timer.restart("splitting device grids");
    auto splitHandles = nanovdb::splitDeviceGrids<BufferT, std::vector>(mergedHandle);
    timer.restart("unit-test split grids");
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
    timer.stop();
}//  mergeSplitDeviceGrids

// make -j 4 testNanoVDB && ./unittest/testNanoVDB --gtest_filter="*Cuda*" --gtest_break_on_failure
TEST(TestNanoVDBCUDA, CudaIndexGridToGrid_basic)
{
    using BufferT = nanovdb::CudaDeviceBuffer;
    const float value = 1.23456f, backgroud = 1.0f;
    const nanovdb::Coord ijk(1,2,3);
    nanovdb::GridHandle<BufferT> floatHdl;
    nanovdb::FloatGrid *floatGrid = nullptr;
    //nanovdb::CpuTimer timer;
    {// create float grid with one active voxel
        nanovdb::build::Grid<float> grid(backgroud);
        auto srcAcc = grid.getAccessor();
        srcAcc.setValue(ijk, value);
        auto nodeCount = grid.nodeCount();
        EXPECT_EQ(1u, nodeCount[0]);
        EXPECT_EQ(1u, nodeCount[1]);
        EXPECT_EQ(1u, nodeCount[2]);
        EXPECT_EQ(value, srcAcc.getValue(ijk));
        EXPECT_EQ(value, srcAcc.getValue(1,2,3));
        //timer.start("Create FloatGrid on CPU");
        floatHdl = nanovdb::createNanoGrid<nanovdb::build::Grid<float>, float, BufferT>(grid);
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
    using BufferT = nanovdb::CudaDeviceBuffer;
    auto idxHdl = nanovdb::createNanoGrid<nanovdb::FloatGrid, nanovdb::ValueIndex, BufferT>(*floatGrid, 0u, false, false, 1);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    EXPECT_FALSE(idxHdl.deviceGrid<nanovdb::ValueIndex>());
    idxHdl.deviceUpload();
    EXPECT_TRUE(idxHdl.deviceGrid<nanovdb::ValueIndex>());
    auto *idxGrid = idxHdl.grid<nanovdb::ValueIndex>();
    EXPECT_TRUE(idxGrid);
    //timer.restart("Create value list on CPU");
    EXPECT_EQ(1u + 512u, idxGrid->valueCount());// background + 512 values in one leaf node
    float *values = new float[idxGrid->valueCount()], *d_values = nullptr;
    values[0] = backgroud;
    const float *q = floatGrid->tree().getFirstLeaf()->data()->mValues;
    for (float *p=values+1, *e=p+512;p!=e; ++p) *p = *q++;
    //timer.restart("Allocate and copy values from CPU to GPU");
    cudaCheck(cudaMalloc((void**)&d_values, idxGrid->valueCount()*sizeof(float)));
    EXPECT_TRUE(d_values);
    cudaCheck(cudaMemcpy(d_values, values, idxGrid->valueCount()*sizeof(float), cudaMemcpyHostToDevice));
    EXPECT_FALSE(idxHdl.deviceGrid<float>());
    auto *d_idxGrid = idxHdl.deviceGrid<nanovdb::ValueIndex>();
    EXPECT_TRUE(d_idxGrid);
    //timer.restart("Call CudaIndexToGrid");
    auto hdl = nanovdb::cudaIndexToGrid<float>(d_idxGrid, d_values);
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
    using BufferT = nanovdb::CudaDeviceBuffer;
    //nanovdb::CpuTimer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::createLevelSetSphere<float, BufferT>(100,nanovdb::Vec3d(0),1,3, nanovdb::Vec3d(0), "test");
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    //timer.restart("Create IndexGrid on CPU");
    auto idxHdl = nanovdb::createNanoGrid<nanovdb::FloatGrid, BuildT, BufferT>(*floatGrid);
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
    auto hdl = nanovdb::cudaIndexToGrid<float>(d_idxGrid, d_values);
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
    using BufferT = nanovdb::CudaDeviceBuffer;
    //nanovdb::CpuTimer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::createLevelSetSphere<float, BufferT>(100,nanovdb::Vec3d(0),1,3, nanovdb::Vec3d(0), "test");
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    //timer.restart("Create IndexGrid on CPU");
    auto idxHdl = nanovdb::createNanoGrid<nanovdb::FloatGrid, BuildT, BufferT>(*floatGrid);
    //timer.restart("Copy IndexGrid from CPU to GPU");
    idxHdl.deviceUpload();
    auto *idxGrid = idxHdl.grid<BuildT>();
    EXPECT_TRUE(idxGrid);
    //timer.restart("Create value list on CPU");
    float *values = new float[idxGrid->valueCount()], *d_values = nullptr;
    values[0] = floatGrid->tree().root().background();
    for (auto it = floatGrid->indexBBox().begin(); it; ++it) {
        EXPECT_EQ(acc.isActive(*it), idxGrid->tree().isActive(*it));
        if (acc.isActive(*it)) {
            const uint64_t idx = idxGrid->tree().getValue(*it);
            EXPECT_TRUE(idx < idxGrid->valueCount());
            values[idx] = acc.getValue(*it);
        }
    }
    //timer.restart("Allocate and copy values from CPU to GPU");
    cudaCheck(cudaMalloc((void**)&d_values, idxGrid->valueCount()*sizeof(float)));
    cudaCheck(cudaMemcpy(d_values, values, idxGrid->valueCount()*sizeof(float), cudaMemcpyHostToDevice));
    EXPECT_FALSE(idxHdl.deviceGrid<float>());
    auto *d_idxGrid = idxHdl.deviceGrid<BuildT>();
    EXPECT_TRUE(d_idxGrid);
    //timer.restart("Call CudaIndexToGrid");
    auto hdl = nanovdb::cudaIndexToGrid<float>(d_idxGrid, d_values);
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
        if (acc.isActive(*it)) EXPECT_EQ(acc.getValue(*it), acc2.getValue(*it));
    }
    //timer.stop();
    cudaFree(d_values);
}//  CudaPointToGrid_ValueOnIndex

TEST(TestNanoVDBCUDA, CudaSignedFloodFill)
{
    using BufferT = nanovdb::CudaDeviceBuffer;
    //nanovdb::CpuTimer timer("Create FloatGrid on CPU");
    auto floatHdl = nanovdb::createLevelSetSphere<float, BufferT>(100);
    auto *floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    auto acc = floatGrid->getAccessor();
    EXPECT_FALSE(acc.isActive(nanovdb::Coord(103,0,0)));
    EXPECT_TRUE( acc.isActive(nanovdb::Coord(100,0,0)));
    EXPECT_FALSE(acc.isActive(nanovdb::Coord( 97,0,0)));
    EXPECT_EQ( 3.0f, acc(103,0,0));
    EXPECT_EQ( 0.0f, acc(100,0,0));
    EXPECT_EQ(-3.0f, acc( 97,0,0));
    using OpT = nanovdb::SetVoxel<float>;// only set the voxel value
    acc.set<OpT>(nanovdb::Coord(103,0,0),-1.0f);// flip sign and value of inactive voxel
    acc.set<OpT>(nanovdb::Coord( 97,0,0), 1.0f);// flip sign and value of inactive voxel
    EXPECT_EQ(-1.0f, acc(103,0,0));
    EXPECT_EQ( 0.0f, acc(100,0,0));
    EXPECT_EQ( 1.0f, acc( 97,0,0));
    //timer.restart("Copy FloatGrid from CPU to GPU");
    floatHdl.deviceUpload();// CPU -> GPU
    auto *d_floatGrid = floatHdl.deviceGrid<float>();
    EXPECT_TRUE(d_floatGrid);
    //timer.restart("Signed flood-fill on the GPU");
    //nanovdb::cudaSignedFloodFill(d_floatGrid, true);
    nanovdb::cudaSignedFloodFill(d_floatGrid);
    //timer.restart("Copy FloatGrid from GPU to CPU");
    floatHdl.deviceDownload();// GPU -> CPU
    //timer.stop();
    floatGrid = floatHdl.grid<float>();
    EXPECT_TRUE(floatGrid);
    acc = floatGrid->getAccessor();
    EXPECT_EQ( 3.0f, acc(103,0,0));
    EXPECT_EQ( 0.0f, acc(100,0,0));
    EXPECT_EQ(-3.0f, acc( 97,0,0));
}//  CudaSignedFloodFill

TEST(TestNanoVDBCUDA, OneVoxelToGrid)
{
    using BuildT = float;
    using GridT = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 1;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

    //nanovdb::GpuTimer timer("Create FloatGrid on GPU");
    nanovdb::CudaPointsToGrid<BuildT> converter;
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
    using BuildT = nanovdb::Points;
    using Vec3T  = nanovdb::Vec3f;
    using GridT  = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 3;
    Vec3T points[num_points] = {Vec3T(1, 0, 0),Vec3T(1, 2, 3),Vec3T(1, 2, 3)}, *d_points = nullptr;
    cudaCheck(cudaMalloc(&d_points, num_points * sizeof(Vec3T)));
    cudaCheck(cudaMemcpy(d_points, points, num_points * sizeof(Vec3T), cudaMemcpyHostToDevice));// CPU -> GPU

    //nanovdb::GpuTimer timer("Create FloatGrid on GPU");
    nanovdb::CudaPointsToGrid<BuildT> converter;
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

    //nanovdb::GpuTimer timer("Create FloatGrid on GPU");
    nanovdb::CudaPointsToGrid<BuildT> converter;
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
    using BuildT = nanovdb::Points;//uint32_t;
    using Vec3T = nanovdb::Vec3d;
    //nanovdb::CpuTimer timer;
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
    nanovdb::CudaPointsToGrid<BuildT> converter(voxelSize);// unit map
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
    nanovdb::forEach(points,[&](const nanovdb::Range1D &r){
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
    using BuildT = nanovdb::Points;
    using Vec3T  = nanovdb::Vec3d;
    //nanovdb::CpuTimer timer;
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
    nanovdb::CudaPointsToGrid<BuildT> converter(voxelSize);// unit map
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
    nanovdb::forEach(points,[&](const nanovdb::Range1D &r){
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
                test = nanovdb::isApproxZero<double>( (points[i] - xyz).lengthSqr() );
            }
            EXPECT_TRUE(test);
        }
    });

    //timer.stop();
}// Large_CudaPointsToGrid_World64

TEST(TestNanoVDBCUDA, Sphere_CudaPointsToGrid_World32)
{
    using BuildT = nanovdb::Points;
    using Vec3T  = nanovdb::Vec3f;

    nanovdb::CpuTimer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    timer.stop();

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

    timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    nanovdb::CudaPointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    auto handle = converter.getHandle(d_points, pointCount);
    timer.stop();
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
    nanovdb::forEach(0u, pointCount, 1u,[&](const nanovdb::Range1D &r){
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
    using BuildT = nanovdb::Points;
    using Vec3T  = nanovdb::Vec3f;

    nanovdb::CpuTimer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    timer.stop();

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

    timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    nanovdb::CudaPointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    converter.setPointType(nanovdb::PointType::Voxel32);
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    timer.stop();
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
    nanovdb::forEach(0u, pointCount, 1u,[&](const nanovdb::Range1D &r){
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
    using BuildT = nanovdb::Points;
    using Vec3T  = nanovdb::Vec3f;

    nanovdb::CpuTimer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    timer.stop();

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

    timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    nanovdb::CudaPointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    converter.setPointType(nanovdb::PointType::Voxel16);
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    timer.stop();
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
    nanovdb::forEach(0u, pointCount, 1u,[&](const nanovdb::Range1D &r){
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

    using BuildT = nanovdb::Points;
    using Vec3T  = nanovdb::Vec3f;

    nanovdb::CpuTimer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    std::cerr << "nanovdb::bbox = " << pointGrid->indexBBox() << " voxel count = " << pointGrid->activeVoxelCount() << std::endl;
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

    timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    //auto handle = nanovdb::cudaPointsToGrid(d_points, pointCount, nanovdb::PointType::Voxel8);
    nanovdb::CudaPointsToGrid<BuildT> converter(pointGrid->map());
    //converter.setVerbose();
    converter.setPointType(nanovdb::PointType::Voxel8);
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    timer.stop();
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
    std::cerr << grid->indexBBox() << std::endl;

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
    nanovdb::forEach(0u, pointCount, 1u,[&](const nanovdb::Range1D &r){
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

    using BuildT = nanovdb::Points;
    using Vec3T  = nanovdb::Vec3f;

    nanovdb::CpuTimer timer("Generate sphere with points");
    auto pointsHandle = nanovdb::createPointSphere(8, 100.0, nanovdb::Vec3d(0.0), 0.5);
    timer.stop();

    auto *pointGrid = pointsHandle.grid<uint32_t>();
    EXPECT_TRUE(pointGrid);
    std::cerr << "nanovdb::bbox = " << pointGrid->indexBBox() << " voxel count = " << pointGrid->activeVoxelCount() << std::endl;
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

    timer.start("Building grid on GPU from "+std::to_string(pointCount)+" points");
    /////////////////////////////////////////////////////////////////////////
    //auto handle = nanovdb::cudaPointsToGrid(d_points, pointCount, nanovdb::PointType::Voxel8);
    nanovdb::CudaPointsToGrid<BuildT> converter(pointGrid->map());
    converter.setVerbose(2);
    converter.setPointType(nanovdb::PointType::PointID);
    auto handle = converter.getHandle(d_points, pointCount);
    /////////////////////////////////////////////////////////////////////////
    timer.stop();
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
    nanovdb::forEach(0u, pointCount, 1u,[&](const nanovdb::Range1D &r){
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
    using BuildT = nanovdb::Rgba8;
    using GridT = nanovdb::NanoGrid<BuildT>;
    const size_t num_points = 1;
    nanovdb::Coord coords[num_points] = {nanovdb::Coord(1, 2, 3)}, *d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, num_points * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords, num_points * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));// CPU -> GPU

    nanovdb::CudaPointsToGrid<BuildT> converter;
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
    auto handle = nanovdb::cudaVoxelsToGrid<BuildT>(d_coords, num_points);
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

    nanovdb::GpuTimer timer("cudaAddBlindData");
    auto handle2 = nanovdb::cudaAddBlindData(d_grid, d_blind, num_points);
    cudaCheck(cudaFree(d_blind));
    timer.stop();
    EXPECT_TRUE(handle2.deviceData());// grid only exists on the GPU
    EXPECT_FALSE(handle2.data());// no grid was yet allocated on the CPU
    EXPECT_EQ(handle2.size(), handle.size() + sizeof(nanovdb::GridBlindMetaData) + nanovdb::AlignUp<NANOVDB_DATA_ALIGNMENT>(num_points*sizeof(float)));

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

    auto handle3 = nanovdb::cudaAddBlindData(d_grid2, d_blind2, num_points,
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
