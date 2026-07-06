// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file TestMemoryResource.cu
///
/// @brief Unit tests for the CUDA memory-resource concept (cuda::DeviceResource,
///        cuda::PinnedResource) and the cuda::TempPool / tools::cuda::PointsToGrid
///        resource plumbing.

#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/PinnedResource.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

namespace {

//======================================================================
// Shared test doubles
//======================================================================

/// @brief Resource that counts (non-null) allocations and deallocations so
///        leaks can be asserted. Delegates the actual work to DeviceResource.
struct CountingResource
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    int allocs = 0;
    int deallocs = 0;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        void* p = nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
        if (p) ++allocs;
        return p;
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        if (p) ++deallocs;
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
};

/// @brief Resource that records the stream of every allocation/deallocation,
///        to verify stream-ordered teardown. Delegates work to DeviceResource.
struct StreamRecordingResource
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    std::vector<cudaStream_t> allocStreams;
    std::vector<cudaStream_t> deallocStreams;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        void* p = nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
        if (p) allocStreams.push_back(stream);
        return p;
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        if (p) deallocStreams.push_back(stream);
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
};

//======================================================================
// AsyncResource concept + DeviceResource instance methods
//======================================================================

// Concept satisfaction: DeviceResource must model the async-resource shape,
// and a plainly-unrelated type must not.
static_assert(nanovdb::cuda::is_async_resource<nanovdb::cuda::DeviceResource>::value,
              "DeviceResource must satisfy the AsyncResource concept");
static_assert(!nanovdb::cuda::is_async_resource<int>::value,
              "int must not satisfy the AsyncResource concept");

TEST(TestMemoryResource, DeviceResource_InstanceAllocateFree)
{
    using R = nanovdb::cuda::DeviceResource;
    R res;
    const size_t bytes = 1024;
    void* p = res.allocate_async(bytes, R::DEFAULT_ALIGNMENT, 0);
    ASSERT_NE(p, nullptr);
    res.deallocate_async(p, bytes, R::DEFAULT_ALIGNMENT, 0);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(TestMemoryResource, DefaultResource_AddressStable)
{
    using R = nanovdb::cuda::DeviceResource;
    R& a = nanovdb::cuda::default_resource<R>();
    R& b = nanovdb::cuda::default_resource<R>();
    EXPECT_EQ(&a, &b); // program-lifetime singleton, address-stable
}

//======================================================================
// PinnedResource (host-pinned, host- and device-accessible)
//======================================================================

static_assert(!nanovdb::cuda::is_async_resource<nanovdb::cuda::PinnedResource>::value,
              "PinnedResource is synchronous (no stream-ordered allocate_async)");

TEST(TestMemoryResource, PinnedResource_IsPageLocked)
{
    using P = nanovdb::cuda::PinnedResource;
    P res;
    const size_t bytes = 4096;
    void* host = res.allocate(bytes, P::DEFAULT_ALIGNMENT);
    ASSERT_NE(host, nullptr);

    // Must be genuine page-locked (pinned) host memory, not pageable.
    cudaPointerAttributes attr{};
    ASSERT_EQ(cudaPointerGetAttributes(&attr, host), cudaSuccess);
    EXPECT_EQ(attr.type, cudaMemoryTypeHost);

    // ...and usable as the host side of an asynchronous device->host copy.
    unsigned char* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(d, 0xAB, bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host, d, bytes, cudaMemcpyDeviceToHost, 0), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
    EXPECT_EQ(reinterpret_cast<unsigned char*>(host)[0], 0xABu);
    ASSERT_EQ(cudaFree(d), cudaSuccess);

    res.deallocate(host, bytes, P::DEFAULT_ALIGNMENT);
}

TEST(TestMemoryResource, PinnedResource_DefaultResourceRoundTrip)
{
    using P = nanovdb::cuda::PinnedResource;
    P& res = nanovdb::cuda::default_resource<P>();
    void* p = res.allocate(256, P::DEFAULT_ALIGNMENT);
    ASSERT_NE(p, nullptr);
    res.deallocate(p, 256, P::DEFAULT_ALIGNMENT);
    EXPECT_EQ(&res, &nanovdb::cuda::default_resource<P>()); // address-stable
}

//======================================================================
// TempPool routes allocations through a resource instance and frees on its
// retained stream (the stream of the most recent reallocate), not the null stream.
//======================================================================

TEST(TestMemoryResource, TempPool_FreesOnRetainedStream)
{
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    StreamRecordingResource rec;
    {
        nanovdb::cuda::TempPool<StreamRecordingResource> pool(rec);
        pool.requestedSize() = 1024;
        pool.reallocate(s);            // allocate on stream s
        ASSERT_NE(pool.data(), nullptr);
    }                                  // pool destroyed here -> must free on s
    ASSERT_FALSE(rec.allocStreams.empty());
    EXPECT_EQ(rec.allocStreams.back(), s);
    ASSERT_FALSE(rec.deallocStreams.empty());
    EXPECT_EQ(rec.deallocStreams.back(), s);
    EXPECT_NE(rec.deallocStreams.back(), static_cast<cudaStream_t>(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

TEST(TestMemoryResource, TempPool_NoLeakAcrossGrowth)
{
    CountingResource res;
    {
        nanovdb::cuda::TempPool<CountingResource> pool(res);
        pool.requestedSize() = 256;  pool.reallocate(0);
        pool.requestedSize() = 4096; pool.reallocate(0);   // grow -> free old + alloc new
        pool.requestedSize() = 64;   pool.reallocate(0);   // smaller request -> no realloc
    }                                                      // destroy -> free current block
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
    EXPECT_GT(res.allocs, 0);
    EXPECT_EQ(res.allocs, res.deallocs);                   // every allocation freed
}

TEST(TestMemoryResource, TempPool_DefaultConstructsWithDefaultResource)
{
    // A default-constructed pool uses the default resource and behaves as the
    // existing builders rely on.
    nanovdb::cuda::TempDevicePool pool;
    pool.requestedSize() = 512;
    pool.reallocate(0);
    EXPECT_NE(pool.data(), nullptr);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

//======================================================================
// PointsToGrid routes its internal scratch through an injected resource instance.
//======================================================================

TEST(TestMemoryResource, PointsToGrid_InjectedInstanceSeam)
{
    using BuildT = float;
    const size_t num = 64;
    std::vector<nanovdb::Coord> coords(num);
    for (size_t i = 0; i < num; ++i)
        coords[i] = nanovdb::Coord(int(i), int(i) * 2 + 1, int(i) * 3 + 2);
    nanovdb::Coord* d_coords = nullptr;
    ASSERT_EQ(cudaMalloc(&d_coords, num * sizeof(nanovdb::Coord)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_coords, coords.data(), num * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice), cudaSuccess);

    CountingResource res;
    {
        nanovdb::tools::cuda::PointsToGrid<BuildT, CountingResource> converter(1.0, nanovdb::Vec3d(0.0), 0, res);
        auto handle = converter.getHandle(d_coords, num);
        ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
        EXPECT_TRUE(handle.deviceData());      // grid built on the device
        EXPECT_GT(res.allocs, 0);              // internal scratch + mDeviceData went through the injected instance
    }                                          // converter destroyed -> mDeviceData freed via res
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
    EXPECT_EQ(res.allocs, res.deallocs);       // every allocation through the resource was freed (no leak)
    ASSERT_EQ(cudaFree(d_coords), cudaSuccess);
}

} // unnamed namespace
