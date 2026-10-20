// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file TestBuffer.cu
///
/// @brief Unit tests for the typed, resource-aware CUDA containers
///        (cuda::Buffer, cuda::BufferView), the synchronous Resource trait
///        (cuda::is_resource), and that the GPU grid builders produce
///        byte-identical grids into single-space and dual-space buffers.

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/PinnedResource.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh> // voxelsToGrid / pointsToGrid, and inputs for the builder equivalence tests
#include <nanovdb/cuda/GridHandle.cuh> // for cuda::mergeGridHandles / splitGridHandles
#include <nanovdb/cuda/ManagedResource.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh> // for the single-space entry-point test
#include <nanovdb/cuda/DeviceBuffer.h> // the dual-space default of the prototype forms, compared against
#include <nanovdb/tools/cuda/AddBlindData.cuh>
#include <nanovdb/tools/cuda/CoarsenGrid.cuh>
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/GridChecksum.h> // host checksums: the reference for the device overloads, and validateChecksum on merged headers
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/tools/cuda/IndexToGrid.cuh>
#include <nanovdb/tools/cuda/MergeGrids.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/tools/cuda/RefineGrid.cuh>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm> // for std::max
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>// for the pinned-handle write/read round trip
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Named rather than anonymous: gtest marks each test class's static test_info_
// [[maybe_unused]], an attribute the CUDA front end ignores. Under internal
// linkage it proves the member unreferenced and reports #177, which
// NANOVDB_CUDA_WERROR promotes to an error.
namespace nanovdb_test {

//======================================================================
// Shared test doubles
//======================================================================

/// @brief Bookkeeping shared by all copies of a CountingResource, so counts
///        survive the resource being held by value inside a Buffer.
struct Counters
{
    int    allocs       = 0;
    int    deallocs     = 0;
    size_t allocBytes   = 0; // bytes of the most recent allocation
    size_t deallocBytes = 0; // bytes of the most recent deallocation
};

/// @brief Resource that counts (non-null) allocations and deallocations so
///        leaks can be asserted. Delegates the actual work to DeviceResource.
struct CountingResource
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    Counters* counters = nullptr;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        void* p = nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
        if (p) { ++counters->allocs; counters->allocBytes = bytes; }
        return p;
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        if (p) { ++counters->deallocs; counters->deallocBytes = bytes; }
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
    void* allocate(size_t bytes, size_t alignment) {
        void* p = nanovdb::cuda::DeviceResource{}.allocate(bytes, alignment);
        if (p) { ++counters->allocs; counters->allocBytes = bytes; }
        return p;
    }
    void deallocate(void* p, size_t bytes, size_t alignment) {
        if (p) { ++counters->deallocs; counters->deallocBytes = bytes; }
        nanovdb::cuda::DeviceResource{}.deallocate(p, bytes, alignment);
    }
    // instances sharing counters are interchangeable
    friend bool operator==(const CountingResource& a, const CountingResource& b) { return a.counters == b.counters; }
    friend bool operator!=(const CountingResource& a, const CountingResource& b) { return !(a == b); }
};

/// @brief Streams recorded by all copies of a StreamRecordingResource.
struct StreamLog
{
    std::vector<cudaStream_t> allocStreams;
    std::vector<cudaStream_t> deallocStreams;
};

/// @brief Resource that records the stream of every allocation/deallocation,
///        to verify stream-ordered teardown. Delegates work to DeviceResource.
struct StreamRecordingResource
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    StreamLog* log = nullptr;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        void* p = nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
        if (p) log->allocStreams.push_back(stream);
        return p;
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        if (p) log->deallocStreams.push_back(stream);
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
    void* allocate(size_t bytes, size_t alignment) {
        void* p = nanovdb::cuda::DeviceResource{}.allocate(bytes, alignment);
        if (p) log->allocStreams.push_back(cudaStream_t(0));
        return p;
    }
    void deallocate(void* p, size_t bytes, size_t alignment) {
        if (p) log->deallocStreams.push_back(cudaStream_t(0));
        nanovdb::cuda::DeviceResource{}.deallocate(p, bytes, alignment);
    }
    // instances sharing a log are interchangeable
    friend bool operator==(const StreamRecordingResource& a, const StreamRecordingResource& b) { return a.log == b.log; }
    friend bool operator!=(const StreamRecordingResource& a, const StreamRecordingResource& b) { return !(a == b); }
};

//======================================================================
// Resource concept traits: is_resource (synchronous) / is_async_resource
//======================================================================

/// @brief Local double that provides both the synchronous and the
///        stream-ordered allocation interface, like a CCCL-style resource.
struct DualResource
{
    static constexpr size_t DEFAULT_ALIGNMENT = 256;
    void* allocate(size_t, size_t) { return nullptr; }
    void deallocate(void*, size_t, size_t) {}
    void* allocate_async(size_t, size_t, cudaStream_t) { return nullptr; }
    void deallocate_async(void*, size_t, size_t, cudaStream_t) {}
    // stateless
    friend bool operator==(const DualResource&, const DualResource&) { return true; }
    friend bool operator!=(const DualResource&, const DualResource&) { return false; }
};

// PinnedResource is synchronous-only: allocate/deallocate without a stream.
static_assert(nanovdb::cuda::is_resource<nanovdb::cuda::PinnedResource>::value,
              "PinnedResource must satisfy the synchronous Resource concept");
static_assert(!nanovdb::cuda::is_async_resource<nanovdb::cuda::PinnedResource>::value,
              "PinnedResource must not satisfy the AsyncResource concept");

// DeviceResource models the AsyncResource refinement: both the
// stream-ordered and the synchronous interface (as in CCCL's cuda::mr).
static_assert(nanovdb::cuda::is_async_resource<nanovdb::cuda::DeviceResource>::value,
              "DeviceResource must satisfy the AsyncResource concept");
static_assert(nanovdb::cuda::is_resource<nanovdb::cuda::DeviceResource>::value,
              "DeviceResource must also satisfy the synchronous Resource concept");

/// @brief Local double with only the stream-ordered interface; under the
///        refinement this is NOT an async resource (the sync pair is missing).
struct AsyncOnlyResource
{
    static constexpr size_t DEFAULT_ALIGNMENT = 256;
    void* allocate_async(size_t, size_t, cudaStream_t) { return nullptr; }
    void deallocate_async(void*, size_t, size_t, cudaStream_t) {}
};
static_assert(!nanovdb::cuda::is_async_resource<AsyncOnlyResource>::value,
              "the async pair alone must not satisfy the AsyncResource refinement");
static_assert(!nanovdb::cuda::is_resource<AsyncOnlyResource>::value,
              "the async pair alone must not satisfy the Resource concept");

// An unrelated type satisfies neither concept.
static_assert(!nanovdb::cuda::is_resource<int>::value,
              "int must not satisfy the Resource concept");
static_assert(!nanovdb::cuda::is_async_resource<int>::value,
              "int must not satisfy the AsyncResource concept");

// A resource with both interfaces satisfies both concepts.
static_assert(nanovdb::cuda::is_resource<DualResource>::value,
              "DualResource must satisfy the synchronous Resource concept");
static_assert(nanovdb::cuda::is_async_resource<DualResource>::value,
              "DualResource must satisfy the AsyncResource concept");

//======================================================================
// The members required by the concepts are exercised directly: the trait
// checks above detect them without odr-using them, and an unreferenced
// member of a file-local struct is dead code to the compiler.

TEST(TestBuffer, TestResourcesSynchronousPair)
{
    // The synchronous halves of the counting and stream-recording doubles are
    // required by the AsyncResource refinement; verify they behave like their
    // stream-ordered halves.
    Counters c;
    CountingResource counting{&c};
    void* p = counting.allocate(256, CountingResource::DEFAULT_ALIGNMENT);
    EXPECT_NE(p, nullptr);
    EXPECT_EQ(c.allocs, 1);
    counting.deallocate(p, 256, CountingResource::DEFAULT_ALIGNMENT);
    EXPECT_EQ(c.deallocs, 1);

    StreamLog log;
    StreamRecordingResource recording{&log};
    p = recording.allocate(256, StreamRecordingResource::DEFAULT_ALIGNMENT);
    EXPECT_NE(p, nullptr);
    ASSERT_EQ(log.allocStreams.size(), 1u);
    EXPECT_EQ(log.allocStreams[0], cudaStream_t(0));// sync pair delegates through the null stream
    recording.deallocate(p, 256, StreamRecordingResource::DEFAULT_ALIGNMENT);
    ASSERT_EQ(log.deallocStreams.size(), 1u);
    EXPECT_EQ(log.deallocStreams[0], cudaStream_t(0));
}

TEST(TestBuffer, TraitDoubleStubs)
{
    // DualResource and AsyncOnlyResource exist as concept probes; their stub
    // members allocate nothing and are safe to call with null arguments.
    DualResource dual;
    EXPECT_EQ(dual.allocate(0, DualResource::DEFAULT_ALIGNMENT), nullptr);
    dual.deallocate(nullptr, 0, DualResource::DEFAULT_ALIGNMENT);
    EXPECT_EQ(dual.allocate_async(0, DualResource::DEFAULT_ALIGNMENT, cudaStream_t(0)), nullptr);
    dual.deallocate_async(nullptr, 0, DualResource::DEFAULT_ALIGNMENT, cudaStream_t(0));

    AsyncOnlyResource asyncOnly;
    EXPECT_EQ(asyncOnly.allocate_async(0, AsyncOnlyResource::DEFAULT_ALIGNMENT, cudaStream_t(0)), nullptr);
    asyncOnly.deallocate_async(nullptr, 0, AsyncOnlyResource::DEFAULT_ALIGNMENT, cudaStream_t(0));
}

TEST(TestBuffer, ResourceTraits)
{
    // The static_asserts above are the real test; this anchors them in a
    // runnable suite entry.
    EXPECT_TRUE(nanovdb::cuda::is_resource<nanovdb::cuda::PinnedResource>::value);
    EXPECT_TRUE(nanovdb::cuda::is_async_resource<nanovdb::cuda::DeviceResource>::value);
}

//======================================================================
// Buffer<T, R> with a stream-ordered (async) resource: construction,
// destruction, element counting, value-initialization
//======================================================================

TEST(TestBuffer, AsyncCountingAllocFree)
{
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    Counters c;
    const size_t n = 100;
    {
        nanovdb::cuda::Buffer<float, CountingResource> buf(s, CountingResource{&c}, n, nanovdb::cuda::noInit);
        EXPECT_NE(buf.data(), nullptr);
        EXPECT_EQ(buf.size(), n);                       // elements, not bytes
        EXPECT_EQ(buf.size_bytes(), n * sizeof(float));
        EXPECT_FALSE(buf.empty());
        EXPECT_EQ(c.allocs, 1);                         // exactly one allocation
        EXPECT_EQ(c.allocBytes, n * sizeof(float));
        EXPECT_EQ(c.deallocs, 0);
    }
    EXPECT_EQ(c.deallocs, 1);                           // exactly one deallocation
    EXPECT_EQ(c.deallocBytes, n * sizeof(float));       // with identical size
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

TEST(TestBuffer, EmptyBuffersAllocateNothing)
{
    Counters c;
    {
        nanovdb::cuda::Buffer<float, CountingResource> def; // default ctor: empty
        EXPECT_EQ(def.data(), nullptr);
        EXPECT_EQ(def.size(), 0u);
        EXPECT_EQ(def.size_bytes(), 0u);
        EXPECT_TRUE(def.empty());

        nanovdb::cuda::Buffer<float, CountingResource> zero(0, CountingResource{&c}, 0, nanovdb::cuda::noInit);
        EXPECT_EQ(zero.data(), nullptr);
        EXPECT_TRUE(zero.empty());
    }
    EXPECT_EQ(c.allocs, 0);
    EXPECT_EQ(c.deallocs, 0);
}

TEST(TestBuffer, CountConstructionRequiresNoInit)
{
    // A count without NoInit must not construct: implicit initialization of
    // freshly allocated memory is a hidden fill pass the dominant
    // allocate-then-overwrite pattern wastes (the cuda::buffer rule).
    using DevBuf = nanovdb::cuda::Buffer<int>;
    using PinBuf = nanovdb::cuda::Buffer<int, nanovdb::cuda::PinnedResource>;
    static_assert(!std::is_constructible<DevBuf, cudaStream_t, size_t>::value,
                  "async count c-tor without NoInit must not exist");
    static_assert(!std::is_constructible<DevBuf, cudaStream_t, nanovdb::cuda::DeviceResource, size_t>::value,
                  "async count+resource c-tor without NoInit must not exist");
    static_assert(!std::is_constructible<PinBuf, size_t>::value,
                  "sync count c-tor without NoInit must not exist");
    static_assert(!std::is_constructible<PinBuf, nanovdb::cuda::PinnedResource, size_t>::value,
                  "sync count+resource c-tor without NoInit must not exist");
    static_assert(std::is_constructible<DevBuf, cudaStream_t, size_t, nanovdb::cuda::NoInit>::value,
                  "async count c-tor with NoInit exists");
    static_assert(std::is_constructible<PinBuf, size_t, nanovdb::cuda::NoInit>::value,
                  "sync count c-tor with NoInit exists");
    SUCCEED();
}

TEST(TestBuffer, OversizedCountThrows)
{
    // An element count whose byte size exceeds PTRDIFF_MAX is rejected before
    // anything is allocated.
    const size_t oversized = size_t(PTRDIFF_MAX) / sizeof(int) + 1;
    EXPECT_THROW((nanovdb::cuda::Buffer<int>(0, oversized, nanovdb::cuda::noInit)), std::runtime_error);
    EXPECT_THROW((nanovdb::cuda::Buffer<int, nanovdb::cuda::PinnedResource>(oversized, nanovdb::cuda::noInit)), std::runtime_error);
    {
        nanovdb::cuda::Buffer<int, nanovdb::cuda::PinnedResource> pinned(8, nanovdb::cuda::noInit);
        EXPECT_THROW(pinned.resize(oversized), std::runtime_error);
        EXPECT_EQ(pinned.size(), 8u);
    }
    nanovdb::cuda::Buffer<int> buf(0, 8, nanovdb::cuda::noInit);
    cudaStream_t other = nullptr;
    ASSERT_EQ(cudaStreamCreate(&other), cudaSuccess);
    EXPECT_THROW(buf.resize(other, oversized), std::runtime_error);
    EXPECT_EQ(buf.size(), 8u);                // failed resize leaves the buffer untouched --
    EXPECT_EQ(buf.stream(), cudaStream_t(0)); // including its retained stream
    ASSERT_EQ(cudaStreamDestroy(other), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(TestBuffer, DestroyFreesAndEmpties)
{
    Counters c;
    nanovdb::cuda::Buffer<float, CountingResource> buf(0, CountingResource{&c}, 64, nanovdb::cuda::noInit);
    ASSERT_EQ(c.allocs, 1);
    buf.destroy();
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(c.deallocs, 1);
    buf.destroy();               // idempotent: no second free
    EXPECT_EQ(c.deallocs, 1);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

// A custom stream-ordered resource written the short way: two methods plus the
// mixin, rather than four.
struct MixinResource : nanovdb::cuda::SyncFromAsync<MixinResource>
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    Counters* counters = nullptr;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        void* p = nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
        if (p) { ++counters->allocs; counters->allocBytes = bytes; }
        return p;
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        if (p) { ++counters->deallocs; counters->deallocBytes = bytes; }
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
    // instances sharing counters are interchangeable
    friend bool operator==(const MixinResource& a, const MixinResource& b) { return a.counters == b.counters; }
    friend bool operator!=(const MixinResource& a, const MixinResource& b) { return !(a == b); }
};

// The mixin supplies the synchronous half, so both concepts are satisfied.
static_assert(nanovdb::cuda::is_async_resource<MixinResource>::value,
              "SyncFromAsync user must still model AsyncResource");
static_assert(nanovdb::cuda::is_resource<MixinResource>::value,
              "SyncFromAsync must supply the synchronous half of the concept");

TEST(TestBuffer, SyncFromAsyncSuppliesTheSynchronousPair)
{
    Counters c;
    MixinResource r{{}, &c};
    // the inherited synchronous pair routes through the derived async methods
    void* p = r.allocate(1024, MixinResource::DEFAULT_ALIGNMENT);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(c.allocs, 1);
    r.deallocate(p, 1024, MixinResource::DEFAULT_ALIGNMENT);
    EXPECT_EQ(c.deallocs, 1);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(TestBuffer, BufferWorksOverAMixinResource)
{
    Counters c;
    {
        nanovdb::cuda::Buffer<float, MixinResource> buf(0, MixinResource{{}, &c}, 64, nanovdb::cuda::noInit);
        EXPECT_EQ(c.allocs, 1);
        EXPECT_NE(buf.data(), nullptr);
    }
    EXPECT_EQ(c.deallocs, 1);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

// State held inline, not behind a pointer: copying such a resource strands its
// accounting, which is exactly what ResourceRef exists to avoid.
struct StatefulInlineResource : nanovdb::cuda::SyncFromAsync<StatefulInlineResource>
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    int allocs = 0, deallocs = 0;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        ++allocs; return nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        ++deallocs; nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
    // state is inline, so only the same instance keeps one account
    friend bool operator==(const StatefulInlineResource& a, const StatefulInlineResource& b) { return &a == &b; }
    friend bool operator!=(const StatefulInlineResource& a, const StatefulInlineResource& b) { return !(a == b); }
};

// A ref over an async resource models both tiers; over a synchronous-only
// resource it models only the synchronous one -- it must not misreport.
static_assert(nanovdb::cuda::is_async_resource<nanovdb::cuda::ResourceRef<nanovdb::cuda::DeviceResource>>::value,
              "ref over an async resource must model AsyncResource");
static_assert(nanovdb::cuda::is_resource<nanovdb::cuda::ResourceRef<nanovdb::cuda::DeviceResource>>::value,
              "ref over an async resource must model Resource");
static_assert(!nanovdb::cuda::is_async_resource<nanovdb::cuda::ResourceRef<nanovdb::cuda::PinnedResource>>::value,
              "ref over a synchronous resource must not claim AsyncResource");
static_assert(nanovdb::cuda::is_resource<nanovdb::cuda::ResourceRef<nanovdb::cuda::PinnedResource>>::value,
              "ref over a synchronous resource must model Resource");

TEST(TestBuffer, ResourceRefSharesTheUnderlyingResource)
{
    StatefulInlineResource res;   // the original; a by-value copy would strand these counters
    {
        nanovdb::cuda::Buffer<int, nanovdb::cuda::ResourceRef<StatefulInlineResource>> buf(
            cudaStream_t{0}, nanovdb::cuda::ResourceRef<StatefulInlineResource>(res), 1024, nanovdb::cuda::noInit);
        EXPECT_NE(buf.data(), nullptr);
        EXPECT_EQ(res.allocs, 1);   // traffic reaches the original, not a copy
        EXPECT_EQ(res.deallocs, 0);
    }
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
    EXPECT_EQ(res.allocs, 1);
    EXPECT_EQ(res.deallocs, 1);
}

/// @brief A resource without operator==, which 13.2 deprecates and 14.0
///        rejects. Constructed only under the deprecated-interface suppression below; trait probes elsewhere do not warn.
struct NoEqualityResource : nanovdb::cuda::SyncFromAsync<NoEqualityResource>
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        return nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
};


TEST(TestBuffer, DestroyIsTheSpellingClearDelegatesTo)
{
    Counters c;
    nanovdb::cuda::Buffer<float, CountingResource> buf(0, CountingResource{&c}, 64, nanovdb::cuda::noInit);
    ASSERT_EQ(c.allocs, 1);
    buf.destroy();               // cuda::buffer's spelling
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(c.deallocs, 1);
    buf.destroy();               // idempotent
    EXPECT_EQ(c.deallocs, 1);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(TestBuffer, DestroyOnStreamRetargetsTheFree)
{
    cudaStream_t a, b;
    ASSERT_EQ(cudaStreamCreate(&a), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&b), cudaSuccess);
    {
        StreamLog log;
        nanovdb::cuda::Buffer<float, StreamRecordingResource> buf(a, StreamRecordingResource{&log}, 32, nanovdb::cuda::noInit);
        ASSERT_EQ(log.allocStreams.size(), 1u);
        EXPECT_EQ(log.allocStreams[0], a);   // allocated on a
        buf.destroy(b);                      // explicit stream overload
        ASSERT_EQ(log.deallocStreams.size(), 1u);
        EXPECT_EQ(log.deallocStreams[0], b); // freed on b, not the retained stream a
        EXPECT_EQ(buf.stream(), b);          // b is retained afterwards
    }
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(b), cudaSuccess);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
}

TEST(TestBuffer, SwapExchangesWithoutAllocatingOrFreeing)
{
    Counters c;
    nanovdb::cuda::Buffer<int, CountingResource> x(0, CountingResource{&c}, 128, nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<int, CountingResource> y(0, CountingResource{&c},  64, nanovdb::cuda::noInit);
    ASSERT_EQ(c.allocs, 2);
    auto* px = x.data();
    auto* py = y.data();
    const int allocs = c.allocs, deallocs = c.deallocs;

    x.swap(y);

    EXPECT_EQ(c.allocs,   allocs);  // no allocation
    EXPECT_EQ(c.deallocs, deallocs);// no free
    EXPECT_EQ(x.data(), py);
    EXPECT_EQ(y.data(), px);
    EXPECT_EQ(x.size(), 64u);
    EXPECT_EQ(y.size(), 128u);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

//======================================================================
// Stream retention: the destructor and resize free on the retained stream
// (stream of the most recent allocation, or the one supplied via set_stream)
//======================================================================

TEST(TestBuffer, DestructorFreesOnAllocationStream)
{
    cudaStream_t a = nullptr;
    ASSERT_EQ(cudaStreamCreate(&a), cudaSuccess);
    StreamLog log;
    {
        nanovdb::cuda::Buffer<float, StreamRecordingResource> buf(a, StreamRecordingResource{&log}, 32, nanovdb::cuda::noInit);
        EXPECT_EQ(buf.stream(), a);
    }
    ASSERT_EQ(log.allocStreams.size(), 1u);
    EXPECT_EQ(log.allocStreams.back(), a);
    ASSERT_EQ(log.deallocStreams.size(), 1u);
    EXPECT_EQ(log.deallocStreams.back(), a);
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(a), cudaSuccess);
}

TEST(TestBuffer, SetStreamRedirectsTheFree)
{
    cudaStream_t a = nullptr, b = nullptr;
    ASSERT_EQ(cudaStreamCreate(&a), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&b), cudaSuccess);
    StreamLog log;
    {
        nanovdb::cuda::Buffer<float, StreamRecordingResource> buf(a, StreamRecordingResource{&log}, 32, nanovdb::cuda::noInit);
        buf.set_stream(b); // member update only, no synchronization
        EXPECT_EQ(buf.stream(), b);
    }
    ASSERT_EQ(log.allocStreams.size(), 1u);
    EXPECT_EQ(log.allocStreams.back(), a);
    ASSERT_EQ(log.deallocStreams.size(), 1u);
    EXPECT_EQ(log.deallocStreams.back(), b);
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(b), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(a), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(b), cudaSuccess);
}

TEST(TestBuffer, ResizePreservesPrefixAndRetainsNewStream)
{
    cudaStream_t a = nullptr, c = nullptr;
    ASSERT_EQ(cudaStreamCreate(&a), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&c), cudaSuccess);
    StreamLog log;
    const size_t n = 128, m = 256;
    {
        nanovdb::cuda::Buffer<int, StreamRecordingResource> buf(a, StreamRecordingResource{&log}, n, nanovdb::cuda::noInit);
        std::vector<int> pattern(n);
        for (size_t i = 0; i < n; ++i) pattern[i] = int(i) * 3 + 1;
        ASSERT_EQ(cudaMemcpyAsync(buf.data(), pattern.data(), n * sizeof(int), cudaMemcpyHostToDevice, a), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);

        buf.resize(c, m); // grow on stream c
        EXPECT_EQ(buf.size(), m);
        EXPECT_EQ(buf.stream(), c);
        ASSERT_EQ(log.allocStreams.size(), 2u);
        EXPECT_EQ(log.allocStreams.back(), c);       // new block allocated on c
        ASSERT_EQ(log.deallocStreams.size(), 1u);
        // The prefix copy on c is the old block's last use, so the old block is
        // freed on c as well; freeing it on a would let the allocator recycle
        // it concurrently with the copy.
        EXPECT_EQ(log.deallocStreams.back(), c);

        std::vector<int> readback(m, -1);
        ASSERT_EQ(cudaMemcpyAsync(readback.data(), buf.data(), m * sizeof(int), cudaMemcpyDeviceToHost, c), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(c), cudaSuccess);
        for (size_t i = 0; i < n; ++i)
            ASSERT_EQ(readback[i], pattern[i]) << "prefix element " << i << " lost across resize";

        buf.resize(c, n / 2); // shrink: keeps the min(old,new) prefix
        std::vector<int> shrunk(n / 2, -1);
        ASSERT_EQ(cudaMemcpyAsync(shrunk.data(), buf.data(), (n / 2) * sizeof(int), cudaMemcpyDeviceToHost, c), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(c), cudaSuccess);
        for (size_t i = 0; i < n / 2; ++i)
            ASSERT_EQ(shrunk[i], pattern[i]) << "prefix element " << i << " lost across shrink";
    }
    EXPECT_EQ(log.allocStreams.size(), log.deallocStreams.size()); // every allocation freed
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(c), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(a), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(c), cudaSuccess);
}


//======================================================================
// Resource equality, as cuda::mr's resource concepts require
//======================================================================

template<typename R>
using IsEqualityComparable = nanovdb::cuda::is_equality_comparable_resource<R>;

static_assert(IsEqualityComparable<nanovdb::cuda::DeviceResource>::value, "DeviceResource needs ==");
static_assert(IsEqualityComparable<nanovdb::cuda::MallocResource>::value, "MallocResource needs ==");
static_assert(IsEqualityComparable<nanovdb::cuda::PinnedResource>::value, "PinnedResource needs ==");
static_assert(IsEqualityComparable<nanovdb::cuda::ManagedResource>::value, "ManagedResource needs ==");
static_assert(IsEqualityComparable<nanovdb::cuda::AsyncFromSync<nanovdb::cuda::MallocResource>>::value, "AsyncFromSync needs ==");
// The adapter is only as comparable as what it wraps.
static_assert(!IsEqualityComparable<nanovdb::cuda::AsyncFromSync<NoEqualityResource>>::value,
              "AsyncFromSync over a resource without == must not claim equality");

TEST(TestBuffer, BuiltInResourcesCompareEqual)
{
    // Stateless resources: memory from any instance may be freed through any other.
    EXPECT_TRUE(nanovdb::cuda::DeviceResource{} == nanovdb::cuda::DeviceResource{});
    EXPECT_FALSE(nanovdb::cuda::DeviceResource{} != nanovdb::cuda::DeviceResource{});
    EXPECT_TRUE(nanovdb::cuda::MallocResource{} == nanovdb::cuda::MallocResource{});
    EXPECT_TRUE(nanovdb::cuda::PinnedResource{} == nanovdb::cuda::PinnedResource{});
    EXPECT_TRUE(nanovdb::cuda::ManagedResource{} == nanovdb::cuda::ManagedResource{});
    // A ref compares the referenced resources, as cuda::mr::resource_ref does:
    // refs to two distinct stateless instances are interchangeable, so equal.
    using RefT = nanovdb::cuda::ResourceRef<nanovdb::cuda::MallocResource>;
    nanovdb::cuda::MallocResource ma, mb;
    EXPECT_TRUE(RefT(ma) == RefT(mb));
    // The adapter compares its wrapped resources.
    using Adapted = nanovdb::cuda::AsyncFromSync<RefT>;
    EXPECT_TRUE((Adapted{RefT(ma)} == Adapted{RefT(mb)}));
}

/// @brief Stateful resource whose instances are interchangeable only when
///        they share a pool id, the case where value equality decides.
struct PoolIdResource : nanovdb::cuda::SyncFromAsync<PoolIdResource>
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    int id = 0;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        return nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
    friend bool operator==(const PoolIdResource& a, const PoolIdResource& b) { return a.id == b.id; }
    friend bool operator!=(const PoolIdResource& a, const PoolIdResource& b) { return a.id != b.id; }
};

TEST(TestBuffer, ResourceRefComparesReferencedResources)
{
    PoolIdResource p1{{}, 1}, p1Too{{}, 1}, p2{{}, 2};
    using RefT = nanovdb::cuda::ResourceRef<PoolIdResource>;
    EXPECT_TRUE(RefT(p1) == RefT(p1Too));   // distinct instances, same pool
    EXPECT_TRUE(RefT(p1) != RefT(p2));      // different pools
    EXPECT_FALSE(RefT(p1) == RefT(p2));
}

static_assert(!nanovdb::cuda::is_equality_comparable_resource<int>::value,
              "the trait reports only resources");

/// @brief A resource with operator== alone, which cuda::mr accepts under C++20.
struct EqualityOnlyResource : nanovdb::cuda::SyncFromAsync<EqualityOnlyResource>
{
    static constexpr size_t DEFAULT_ALIGNMENT = nanovdb::cuda::DeviceResource::DEFAULT_ALIGNMENT;
    void* allocate_async(size_t bytes, size_t alignment, cudaStream_t stream) {
        return nanovdb::cuda::DeviceResource{}.allocate_async(bytes, alignment, stream);
    }
    void deallocate_async(void* p, size_t bytes, size_t alignment, cudaStream_t stream) {
        nanovdb::cuda::DeviceResource{}.deallocate_async(p, bytes, alignment, stream);
    }
    friend bool operator==(const EqualityOnlyResource&, const EqualityOnlyResource&) { return true; }
};
static_assert(IsEqualityComparable<EqualityOnlyResource>::value, "operator== alone suffices");

//======================================================================
// Deprecated interfaces, supported until 14.0 removes them
//======================================================================

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#if defined(__CUDACC__)
#pragma nv_diagnostic push
#pragma nv_diag_suppress 1444
#pragma nv_diag_suppress 20199
#endif

TEST(TestBuffer, ResizeDeprecatedArgumentOrderForwards)
{
    StreamLog log;
    cudaStream_t c = nullptr;
    ASSERT_EQ(cudaStreamCreate(&c), cudaSuccess);
    {
        nanovdb::cuda::Buffer<int, StreamRecordingResource> buf(0, StreamRecordingResource{&log}, 8, nanovdb::cuda::noInit);
        buf.resize(size_t(16), c);
        EXPECT_EQ(buf.size(), 16u);
        EXPECT_EQ(buf.stream(), c);
        EXPECT_EQ(log.allocStreams.back(), c);
        buf.resize(0, 0); // two literal zeros: count 0 on the default stream in either order
        EXPECT_TRUE(buf.empty());
        EXPECT_EQ(buf.stream(), cudaStream_t(0));
    }
    ASSERT_EQ(cudaStreamSynchronize(c), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(c), cudaSuccess);
}

TEST(TestBuffer, ResizeRejectsNegativeCountInEitherOrder)
{
    nanovdb::cuda::Buffer<std::byte> buf(cudaStream_t(0), 8, nanovdb::cuda::noInit);
    EXPECT_THROW(buf.resize(-1, cudaStream_t(0)), std::runtime_error);
    EXPECT_THROW(buf.resize(cudaStream_t(0), size_t(-1)), std::runtime_error);
    EXPECT_EQ(8u, buf.size()); // untouched
}

TEST(TestBuffer, ResizeDeprecatedOrderAcceptsBoolCount)
{
    nanovdb::cuda::Buffer<int> buf(cudaStream_t(0), 8, nanovdb::cuda::noInit);
    buf.resize(true, cudaStream_t(0)); // converted to size_t, as the old signature did
    EXPECT_EQ(1u, buf.size());
    buf.resize({4}, cudaStream_t(0));   // a braced count, as the old size_t parameter accepted
    EXPECT_EQ(4u, buf.size());
    buf.resize(2.0, cudaStream_t(0));
    EXPECT_EQ(2u, buf.size());
    EXPECT_THROW(buf.resize(-1.5, cudaStream_t(0)), std::runtime_error);
    EXPECT_EQ(2u, buf.size());
}

TEST(TestBuffer, ResourceRefWithoutResourceEqualityFallsBackToIdentity)
{
    NoEqualityResource a, b;
    nanovdb::cuda::ResourceRef<NoEqualityResource> ra(a), raAgain(a), rb(b);
    EXPECT_TRUE(ra == raAgain); // same underlying resource
    EXPECT_TRUE(ra != rb);      // different underlying resources
}

#if defined(__CUDACC__)
#pragma nv_diagnostic pop
#endif
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

//======================================================================
// Move semantics (Buffer is move-only) and explicit deep copy
//======================================================================

TEST(TestBuffer, MoveConstructorSteals)
{
    Counters c;
    {
        nanovdb::cuda::Buffer<float, CountingResource> src(0, CountingResource{&c}, 64, nanovdb::cuda::noInit);
        float* p = src.data();
        nanovdb::cuda::Buffer<float, CountingResource> dst(std::move(src));
        EXPECT_EQ(dst.data(), p);        // stolen, not reallocated
        EXPECT_EQ(dst.size(), 64u);
        EXPECT_EQ(src.data(), nullptr);  // source emptied
        EXPECT_EQ(src.size(), 0u);
        EXPECT_TRUE(src.empty());
        EXPECT_EQ(c.allocs, 1);          // no allocation on move
    }
    EXPECT_EQ(c.allocs, c.deallocs);     // exactly one free, no double free
    EXPECT_EQ(c.deallocs, 1);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(TestBuffer, MoveAssignFreesDestination)
{
    Counters c;
    {
        nanovdb::cuda::Buffer<float, CountingResource> a(0, CountingResource{&c}, 64, nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<float, CountingResource> b(0, CountingResource{&c}, 32, nanovdb::cuda::noInit);
        ASSERT_EQ(c.allocs, 2);
        float* p = a.data();
        b = std::move(a);                // frees b's old block first
        EXPECT_EQ(c.deallocs, 1);
        EXPECT_EQ(b.data(), p);
        EXPECT_EQ(b.size(), 64u);
        EXPECT_EQ(a.data(), nullptr);
    }
    EXPECT_EQ(c.allocs, c.deallocs);     // both blocks freed exactly once
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(TestBuffer, SelfMoveAssignIsSafe)
{
    Counters c;
    {
        nanovdb::cuda::Buffer<float, CountingResource> buf(0, CountingResource{&c}, 16, nanovdb::cuda::noInit);
        auto& self = buf; // avoids the compiler warning on a literal self-move
        buf = std::move(self);
        EXPECT_NE(buf.data(), nullptr);  // still owns its block
        EXPECT_EQ(buf.size(), 16u);
        EXPECT_EQ(c.deallocs, 0);
    }
    EXPECT_EQ(c.allocs, 1);
    EXPECT_EQ(c.deallocs, 1);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(TestBuffer, CopyIsDeep)
{
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    Counters c;
    const size_t n = 96;
    {
        nanovdb::cuda::Buffer<int, CountingResource> src(s, CountingResource{&c}, n, nanovdb::cuda::noInit);
        std::vector<int> pattern(n);
        for (size_t i = 0; i < n; ++i) pattern[i] = int(i) - 7;
        ASSERT_EQ(cudaMemcpyAsync(src.data(), pattern.data(), n * sizeof(int), cudaMemcpyHostToDevice, s), cudaSuccess);

        auto dup = src.copy(s);
        EXPECT_EQ(c.allocs, 2);              // deep copy allocates its own block
        EXPECT_NE(dup.data(), src.data());
        EXPECT_EQ(dup.size(), n);
        EXPECT_EQ(dup.stream(), s);

        std::vector<int> readback(n, 0);
        ASSERT_EQ(cudaMemcpyAsync(readback.data(), dup.data(), n * sizeof(int), cudaMemcpyDeviceToHost, s), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        EXPECT_EQ(readback, pattern);
    }
    EXPECT_EQ(c.allocs, c.deallocs);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

//======================================================================
// CUDA graph capture: the async path (stream-ordered allocation, no hidden
// synchronization or initialization) must be recordable in a graph. The
// synchronous tier is excluded by construction -- it synchronizes.
//======================================================================

TEST(TestBuffer, AsyncPathIsGraphCapturable)
{
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), cudaSuccess);
    {
        // Allocation, use, and free are all recorded as graph nodes; the
        // buffer is created and destroyed inside the capture so the graph
        // owns the allocation's whole lifetime.
        nanovdb::cuda::Buffer<int> buf(s, 256, nanovdb::cuda::noInit);
        ASSERT_NE(buf.data(), nullptr);
        ASSERT_EQ(cudaMemsetAsync(buf.data(), 0x5A, buf.size_bytes(), s), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamEndCapture(s, &graph), cudaSuccess);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess); // relaunchable
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);

    ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

//======================================================================
// Buffer over a synchronous resource (PinnedResource): no stream anywhere
//======================================================================

template<class B, class = void>
struct HasSetStream : std::false_type {};
template<class B>
struct HasSetStream<B, std::void_t<decltype(std::declval<B&>().set_stream(cudaStream_t{0}))>> : std::true_type {};

template<class B, class = void>
struct HasStreamGetter : std::false_type {};
template<class B>
struct HasStreamGetter<B, std::void_t<decltype(std::declval<const B&>().stream())>> : std::true_type {};

using PinnedBufferF = nanovdb::cuda::Buffer<float, nanovdb::cuda::PinnedResource>;
using DeviceBufferF = nanovdb::cuda::Buffer<float, nanovdb::cuda::DeviceResource>;

// A Buffer over a synchronous resource exposes no stream API at all.
static_assert(!HasSetStream<PinnedBufferF>::value, "sync-resource Buffer must not expose set_stream");
static_assert(!HasStreamGetter<PinnedBufferF>::value, "sync-resource Buffer must not expose stream()");
static_assert(HasSetStream<DeviceBufferF>::value, "async-resource Buffer exposes set_stream");
static_assert(HasStreamGetter<DeviceBufferF>::value, "async-resource Buffer exposes stream()");

TEST(TestBuffer, PinnedBufferIsPageLocked)
{
    const size_t n = 300;
    PinnedBufferF buf(n, nanovdb::cuda::noInit); // synchronous: no stream parameter
    ASSERT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), n);

    cudaPointerAttributes attr{};
    ASSERT_EQ(cudaPointerGetAttributes(&attr, buf.data()), cudaSuccess);
    EXPECT_EQ(attr.type, cudaMemoryTypeHost); // genuine page-locked host memory
}

TEST(TestBuffer, PinnedBufferNoInitResizeAndCopy)
{
    const size_t n = 64;
    nanovdb::cuda::Buffer<int, nanovdb::cuda::PinnedResource> buf(n, nanovdb::cuda::noInit);
    ASSERT_NE(buf.data(), nullptr);
    for (size_t i = 0; i < n; ++i) buf.data()[i] = int(i) + 11;

    buf.resize(2 * n); // synchronous: no stream parameter
    EXPECT_EQ(buf.size(), 2 * n);
    for (size_t i = 0; i < n; ++i)
        ASSERT_EQ(buf.data()[i], int(i) + 11) << "prefix element " << i << " lost across resize";

    auto dup = buf.copy();
    ASSERT_NE(dup.data(), nullptr);
    EXPECT_NE(dup.data(), buf.data()); // deep copy
    EXPECT_EQ(dup.size(), buf.size());
    for (size_t i = 0; i < n; ++i)
        ASSERT_EQ(dup.data()[i], int(i) + 11);
}

//======================================================================
// Raw (untyped) form: Buffer<std::byte, DeviceResource>
//======================================================================

TEST(TestBuffer, RawByteBufferRoundTrip)
{
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    const size_t n = 1000;
    nanovdb::cuda::Buffer<std::byte> buf(s, n, nanovdb::cuda::noInit);
    EXPECT_EQ(buf.size(), n);
    EXPECT_EQ(buf.size_bytes(), n); // one byte per element

    std::vector<std::byte> pattern(n);
    for (size_t i = 0; i < n; ++i) pattern[i] = std::byte(i % 251);
    ASSERT_EQ(cudaMemcpyAsync(buf.data(), pattern.data(), n, cudaMemcpyHostToDevice, s), cudaSuccess);
    std::vector<std::byte> readback(n);
    ASSERT_EQ(cudaMemcpyAsync(readback.data(), buf.data(), n, cudaMemcpyDeviceToHost, s), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    EXPECT_EQ(readback, pattern);
    buf.destroy();
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

//======================================================================
// BufferView: non-owning typed view with span semantics
//======================================================================

static_assert(std::is_trivially_copyable_v<nanovdb::cuda::BufferView<std::byte>>,
              "BufferView must be trivially copyable");
static_assert(std::is_trivially_copyable_v<nanovdb::cuda::BufferView<const std::byte>>,
              "BufferView over const elements must be trivially copyable");

TEST(TestBuffer, BufferViewWrapsHostArray)
{
    float host[5] = {0.f, 1.f, 2.f, 3.f, 4.f};

    nanovdb::cuda::BufferView<float> empty;
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_TRUE(empty.empty());

    nanovdb::cuda::BufferView<float> view(host, 5);
    EXPECT_EQ(view.data(), host);
    EXPECT_EQ(view.size(), 5u);
    EXPECT_EQ(view.size_bytes(), 5 * sizeof(float));
    EXPECT_FALSE(view.empty());
    view.data()[2] = 42.f; // writable through the view
    EXPECT_EQ(host[2], 42.f);

    view.destroy(); // detaches, does not free
    EXPECT_EQ(view.data(), nullptr);
    EXPECT_TRUE(view.empty());
    EXPECT_EQ(host[2], 42.f); // underlying storage untouched

    // A null base pointer is only valid for an empty view.
    EXPECT_THROW((nanovdb::cuda::BufferView<float>(nullptr, 5)), std::runtime_error);
    EXPECT_NO_THROW((nanovdb::cuda::BufferView<float>(nullptr, 0)));
}

TEST(TestBuffer, BufferViewConstElements)
{
    const std::byte host[4] = {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    nanovdb::cuda::BufferView<const std::byte> view(host, 4);
    EXPECT_EQ(view.data(), host);
    EXPECT_EQ(view.size_bytes(), 4u);
    EXPECT_EQ(view.data()[3], std::byte{4}); // read-only access
}

//======================================================================
// Zero-copy GridHandle over a BufferView: a handle that indexes grids in
// storage owned elsewhere
//======================================================================

TEST(TestBuffer, GridHandleOverBufferViewIsZeroCopy)
{
    // Build a small host grid with the default (owning) HostBuffer.
    auto owner = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    ASSERT_NE(owner.data(), nullptr);
    const auto* grid = owner.grid<float>();
    ASSERT_NE(grid, nullptr);

    {
        // Wrap the owning handle's storage in a non-owning view...
        nanovdb::cuda::BufferView<std::byte> view(static_cast<std::byte*>(owner.data()), owner.bufferSize());
        // ...and construct a second, zero-copy handle over the same bytes.
        nanovdb::GridHandle<nanovdb::cuda::BufferView<std::byte>> viewHandle(std::move(view));

        EXPECT_EQ(viewHandle.gridCount(), owner.gridCount());
        EXPECT_EQ(viewHandle.data(), owner.data()); // same bytes, not a copy

        const auto* viewGrid = viewHandle.grid<float>();
        ASSERT_NE(viewGrid, nullptr);
        EXPECT_EQ(viewGrid, grid);                  // zero copy: identical grid address
        EXPECT_EQ(viewGrid->gridType(), grid->gridType());
        EXPECT_EQ(viewGrid->activeVoxelCount(), grid->activeVoxelCount());
        EXPECT_EQ(viewGrid->worldBBox(), grid->worldBBox());
        EXPECT_STREQ(viewGrid->gridName(), grid->gridName());
    } // view handle destroyed first: detaches without freeing

    // The owning handle is still intact and is the single owner of the bytes.
    ASSERT_NE(owner.data(), nullptr);
    EXPECT_NE(owner.grid<float>(), nullptr);
    EXPECT_EQ(owner.grid<float>()->activeVoxelCount(), grid->activeVoxelCount());
} // owner destroyed here: the one and only free

//======================================================================
// Single-space GridHandle over cuda::Buffer (step 3): the handle parses
// grid metadata on the device, exposes device accessors, deep-copies
// device-to-device, and routes every allocation through the buffer's
// resource.
//======================================================================

TEST(TestBuffer, GridHandleSingleSpaceParsesMetaOnDevice)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    ASSERT_NE(host.data(), nullptr);

    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    BufT buf(cudaStream_t(0), host.bufferSize(), nanovdb::cuda::noInit);
    const void* devPtr = buf.data();
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice));

    nanovdb::GridHandle<BufT> handle(std::move(buf));
    EXPECT_EQ(1u, handle.gridCount());
    EXPECT_EQ(nanovdb::GridType::Float, handle.gridType(0));
    EXPECT_EQ(host.gridSize(0), handle.gridSize(0));
    EXPECT_EQ(host.bufferSize(), handle.bufferSize());
    EXPECT_FALSE(handle.isEmpty());
    EXPECT_FALSE(handle.isPadded());
    EXPECT_EQ(devPtr, handle.deviceData());          // single space: deviceData is the buffer
    EXPECT_NE(handle.deviceGrid<float>(), nullptr);  // typed device accessor works
    EXPECT_EQ(handle.deviceGrid<nanovdb::Vec3f>(), nullptr);// wrong type: null, not garbage
}

TEST(TestBuffer, GridHandleSingleSpaceMetaScratchUsesResource)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    ASSERT_NE(host.data(), nullptr);

    Counters counters;
    CountingResource res{&counters};
    using BufT = nanovdb::cuda::Buffer<std::byte, CountingResource>;
    {
        BufT buf(cudaStream_t(0), res, host.bufferSize(), nanovdb::cuda::noInit);// alloc #1: the grid bytes
        ASSERT_EQ(cudaSuccess, cudaMemcpy(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice));

        nanovdb::GridHandle<BufT> handle(std::move(buf));// alloc #2 + free #1: the meta scratch
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(2, counters.allocs);
        EXPECT_EQ(1, counters.deallocs);
        EXPECT_EQ(1u, handle.gridCount());

        handle.reset();// free #2: the grid bytes, through the same resource
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(2, counters.deallocs);
        EXPECT_TRUE(handle.isEmpty());
    }
    EXPECT_EQ(counters.allocs, counters.deallocs);
}

TEST(TestBuffer, GridHandleSingleSpaceCopyIsDeviceDeep)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    ASSERT_NE(host.data(), nullptr);

    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    BufT buf(cudaStream_t(0), host.bufferSize(), nanovdb::cuda::noInit);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice));
    nanovdb::GridHandle<BufT> handle(std::move(buf));

    auto copy = handle.copy<BufT>();
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_NE(copy.deviceData(), handle.deviceData());// deep, not aliasing
    EXPECT_EQ(copy.gridCount(), handle.gridCount());
    EXPECT_EQ(copy.gridSize(0), handle.gridSize(0));
    EXPECT_EQ(copy.bufferSize(), handle.bufferSize());

    std::vector<std::byte> a(handle.bufferSize()), b(copy.bufferSize());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(a.data(), handle.deviceData(), a.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(b.data(), copy.deviceData(), b.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));// same bytes
}

static_assert(!nanovdb::BufferHasDeviceSingle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>>::value,
              "pinned storage is host-accessible, not device-single");
static_assert(nanovdb::BufferHasHostSingle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>>::value,
              "pinned storage is a host-readable single-space buffer");

TEST(TestBuffer, GridHandlePinnedHostReadable)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    ASSERT_NE(host.data(), nullptr);
    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>;
    BufT buf(host.bufferSize(), nanovdb::cuda::noInit);
    std::memcpy(buf.data(), host.data(), host.bufferSize());// pinned memory is host-writable

    nanovdb::GridHandle<BufT> handle(std::move(buf));// metadata parses on the host
    EXPECT_EQ(1u, handle.gridCount());
    EXPECT_EQ(nanovdb::GridType::Float, handle.gridType(0));
    ASSERT_NE(handle.grid<float>(), nullptr);// the host accessors apply
    EXPECT_EQ(std::string("sphere"), handle.grid<float>()->gridName());
    EXPECT_NE(handle.gridData(), nullptr);
    EXPECT_FALSE(handle.isPadded());

    auto copy = handle.copy<BufT>();// host-side deep copy allocating through the buffer's resource
    ASSERT_NE(copy.grid<float>(), nullptr);
    EXPECT_NE(copy.data(), handle.data());
    EXPECT_EQ(0, std::memcmp(copy.data(), handle.data(), handle.bufferSize()));

    std::stringstream ss;
    handle.write(ss);// host-accessible grids serialize directly
    nanovdb::GridHandle<BufT> readBack;
    readBack.read(ss);// and read back, allocating through the buffer's resource
    EXPECT_EQ(1u, readBack.gridCount());
    EXPECT_NE(readBack.grid<float>(), nullptr);
    EXPECT_EQ(0, std::memcmp(readBack.data(), handle.data(), handle.bufferSize()));
}

TEST(TestBuffer, GridHandlePinnedAsyncResource)
{
    // A stream-ordered host-accessible resource (the shape of a pooled pinned
    // allocator) takes the same host-readable handle paths; reads and copies
    // allocate on the pool's retained stream.
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    using ResT = nanovdb::cuda::AsyncFromSync<nanovdb::cuda::PinnedResource>;
    using BufT = nanovdb::cuda::Buffer<std::byte, ResT>;
    static_assert(nanovdb::BufferHasHostSingle<BufT>::value, "an adapted pinned resource stays host-accessible");
    static_assert(nanovdb::BufferHasStream<BufT>::value, "an async resource retains a stream");

    BufT buf(cudaStream_t(0), host.bufferSize(), nanovdb::cuda::noInit);
    std::memcpy(buf.data(), host.data(), host.bufferSize());// synchronous allocation: host-writable immediately
    nanovdb::GridHandle<BufT> handle(std::move(buf));
    EXPECT_EQ(1u, handle.gridCount());
    ASSERT_NE(handle.grid<float>(), nullptr);

    auto copy = handle.copy<BufT>();// allocates via (stream, resource, count, noInit)
    ASSERT_NE(copy.grid<float>(), nullptr);
    EXPECT_EQ(0, std::memcmp(copy.data(), handle.data(), handle.bufferSize()));

    std::stringstream ss;
    handle.write(ss);
    nanovdb::GridHandle<BufT> readBack;
    readBack.read(ss);
    EXPECT_EQ(1u, readBack.gridCount());
    EXPECT_NE(readBack.grid<float>(), nullptr);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));// the buffers free on their retained default stream
}

TEST(TestBuffer, GridHandleCopyToHostAndBack)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    using DevBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    {// the handles' buffers retain the stream and free on it: it must outlive them
        DevBufT buf(stream, host.bufferSize(), nanovdb::cuda::noInit);
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice, stream));
        nanovdb::GridHandle<DevBufT> devHandle(std::move(buf));

        auto hostCopy = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(devHandle, stream);
        EXPECT_EQ(1u, hostCopy.gridCount());// host-readable immediately: copyTo synchronized
        ASSERT_NE(hostCopy.grid<float>(), nullptr);
        EXPECT_EQ(0, std::memcmp(hostCopy.data(), host.data(), host.bufferSize()));

        auto devAgain = nanovdb::cuda::copyTo<DevBufT>(hostCopy, stream);// re-parses (and validates) on the device
        EXPECT_EQ(1u, devAgain.gridCount());
        EXPECT_NE(devAgain.deviceGrid<float>(), nullptr);
        EXPECT_NE(devAgain.deviceData(), devHandle.deviceData());

        using SyncBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::MallocResource>;
        auto syncDev = nanovdb::cuda::copyTo<SyncBufT>(hostCopy, stream);// synchronous destination resource
        EXPECT_NE(syncDev.deviceGrid<float>(), nullptr);

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(TestBuffer, GridHandleCopyToPinnedRoundTrip)
{
    auto a = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "a");
    auto b = nanovdb::tools::createLevelSetSphere<float>(10.0, nanovdb::Vec3d(2), 1.0, 3.0, nanovdb::Vec3d(0), "b");
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> parts;
    parts.push_back(std::move(a));
    parts.push_back(std::move(b));
    auto merged = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(parts);// explicit args: MSVC cannot deduce the template-template VectorT
    ASSERT_EQ(2u, merged.gridCount());

    using DevBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    using PinBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::PinnedResource>;
    auto dev = nanovdb::cuda::copyTo<DevBufT>(merged, cudaStream_t(0));
    EXPECT_EQ(2u, dev.gridCount());// multi-grid chains survive the transfer
    EXPECT_NE(dev.deviceGrid<float>(1), nullptr);

    auto pinned = nanovdb::cuda::copyTo<PinBufT>(dev, cudaStream_t(0));// device -> pinned stays on the transfer stream
    EXPECT_EQ(2u, pinned.gridCount());
    ASSERT_NE(pinned.grid<float>(0), nullptr);
    ASSERT_NE(pinned.grid<float>(1), nullptr);
    EXPECT_EQ(0, std::memcmp(pinned.data(), merged.data(), merged.bufferSize()));

    auto dev2 = nanovdb::cuda::copyTo<DevBufT>(pinned, cudaStream_t(0));// pinned -> device is genuinely asynchronous
    EXPECT_EQ(2u, dev2.gridCount());
    EXPECT_NE(dev2.deviceGrid<float>(1), nullptr);
}

TEST(TestBuffer, GridHandleRejectsMalformedDeviceChain)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    const uint64_t bytes = host.bufferSize();
    using DevBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    // Uploads the valid grid with one header field edited, then constructs a device handle over
    // the raw bytes so the device-side chain parse is what judges them.
    auto forge = [&](auto&& corrupt) {
        std::vector<std::byte> bytesCopy(bytes);
        std::memcpy(bytesCopy.data(), host.data(), bytes);
        corrupt(*reinterpret_cast<nanovdb::GridData*>(bytesCopy.data()));
        DevBufT dev(cudaStream_t(0), bytes, nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpyAsync(dev.data(), bytesCopy.data(), bytes, cudaMemcpyHostToDevice, cudaStream_t(0)));
        return nanovdb::GridHandle<DevBufT>(std::move(dev));
    };
    {// a faithful copy parses and reports the same metadata
        auto ok = forge([](nanovdb::GridData&) {});
        EXPECT_EQ(1u, ok.gridCount());
        EXPECT_EQ(bytes, ok.gridSize(0));
    }
    EXPECT_THROW(forge([](nanovdb::GridData& d) { d.mGridSize *= 2; }), std::runtime_error);// claims twice the allocation
    EXPECT_THROW(forge([](nanovdb::GridData& d) { d.mGridIndex = 7; }), std::runtime_error);// index disagrees with its chain position
    EXPECT_THROW(forge([](nanovdb::GridData& d) { d.mGridCount = 2; }), std::runtime_error);// claims a second grid the buffer does not hold
    EXPECT_THROW(forge([](nanovdb::GridData& d) { d.mGridCount = 2; d.mGridSize = sizeof(nanovdb::GridData) + 1; }), std::runtime_error);// a second header read from an unaligned offset into grid bytes: rejected as invalid, never dereferenced in place
}

TEST(TestBuffer, GridHandleCopyToManagedSynchronizes)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    using ManagedBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ManagedResource>;
    // A managed destination is host-readable, so copyTo synchronizes before
    // returning: the host accessors must be valid immediately, with no
    // synchronization by the caller.
    auto managed = nanovdb::cuda::copyTo<ManagedBufT>(host, cudaStream_t(0));
    EXPECT_EQ(1u, managed.gridCount());
    ASSERT_NE(managed.grid<float>(), nullptr);
    EXPECT_EQ(0, std::memcmp(managed.data(), host.data(), host.bufferSize()));
    EXPECT_NE(managed.deviceGrid<float>(), nullptr);// the same allocation serves the device accessors
}

TEST(TestBuffer, GridHandleCopyToProtoResource)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    Counters counters;
    CountingResource res{&counters};
    using RefT = nanovdb::cuda::ResourceRef<CountingResource>;
    using DevBufT = nanovdb::cuda::Buffer<std::byte, RefT>;
    {
        DevBufT proto(cudaStream_t(0), RefT(res), 16, nanovdb::cuda::noInit);// alloc #1: an exemplar carrying the borrowed resource
        auto dev = nanovdb::cuda::copyTo<DevBufT>(host, cudaStream_t(0), &proto);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(2, counters.allocs); // #2: the grid storage; the metadata is adopted from the source handle, so no scratch
        EXPECT_NE(dev.deviceGrid<float>(), nullptr);

        nanovdb::GridHandle<nanovdb::HostBuffer> empty;
        EXPECT_THROW((nanovdb::cuda::copyTo<DevBufT>(empty, cudaStream_t(0), &proto)), std::runtime_error);// no empty handle over a non-default-constructible buffer
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);

    nanovdb::GridHandle<nanovdb::HostBuffer> empty;
    auto out = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>>(empty, cudaStream_t(0));
    EXPECT_TRUE(out.isEmpty());// empty in, empty out for default-constructible targets
}

TEST(TestBuffer, GridHandleSingleSpaceEmptyAndInvalid)
{
    using BufT = nanovdb::cuda::Buffer<std::byte, CountingResource>;
    Counters counters;
    CountingResource res{&counters};

    {   // empty buffer -> empty handle, no meta parse, no scratch
        nanovdb::GridHandle<BufT> handle{BufT(cudaStream_t(0), res, 0, nanovdb::cuda::noInit)};
        EXPECT_EQ(0u, handle.gridCount());
        EXPECT_EQ(nullptr, handle.deviceData());
        EXPECT_TRUE(handle.isEmpty());
        EXPECT_FALSE(handle.isPadded());
        auto copy = handle.copy<BufT>();// copying an empty handle is a no-op
        EXPECT_TRUE(copy.isEmpty());
        EXPECT_EQ(0, counters.allocs);
    }

    {   // buffer full of zeros is not a valid grid: ctor throws, nothing leaks
        BufT buf(cudaStream_t(0), res, 4096, nanovdb::cuda::noInit);
        ASSERT_EQ(cudaSuccess, cudaMemset(buf.data(), 0, 4096));
        EXPECT_THROW(nanovdb::GridHandle<BufT>{std::move(buf)}, std::runtime_error);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);// the moved-in grid bytes were freed
}

TEST(TestBuffer, GridHandleSingleSpaceSynchronousResource)
{
    // MallocResource is synchronous: the handle must take the stream-less
    // constructor and copy paths.
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::MallocResource>;
    static_assert(nanovdb::BufferHasDeviceSingle<BufT>::value, "cudaMalloc storage is device-resident");

    BufT buf(host.bufferSize(), nanovdb::cuda::noInit);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice));
    nanovdb::GridHandle<BufT> handle(std::move(buf));
    EXPECT_EQ(1u, handle.gridCount());
    EXPECT_NE(handle.deviceGrid<float>(), nullptr);

    auto copy = handle.copy<BufT>();// synchronous D2D copy
    EXPECT_NE(copy.deviceData(), handle.deviceData());
    EXPECT_EQ(copy.gridSize(0), handle.gridSize(0));
}

TEST(TestBuffer, GridHandleSingleSpaceMultiGridAndStream)
{
    // Two grids merged into one buffer exercise the device-side offset walk,
    // and a stream-recording resource proves the parse runs on the buffer's
    // retained stream.
    auto a = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "a");
    auto b = nanovdb::tools::createLevelSetSphere<float>(10.0, nanovdb::Vec3d(2), 1.0, 3.0, nanovdb::Vec3d(0), "b");
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> parts;
    parts.push_back(std::move(a));
    parts.push_back(std::move(b));
    auto merged = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(parts);// explicit args: MSVC cannot deduce the template-template VectorT
    ASSERT_EQ(2u, merged.gridCount());

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    StreamLog recorder;
    StreamRecordingResource res{&recorder};
    using BufT = nanovdb::cuda::Buffer<std::byte, StreamRecordingResource>;
    {
        BufT buf(stream, res, merged.bufferSize(), nanovdb::cuda::noInit);
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(buf.data(), merged.data(), merged.bufferSize(), cudaMemcpyHostToDevice, stream));
        nanovdb::GridHandle<BufT> handle(std::move(buf));
        EXPECT_EQ(2u, handle.gridCount());
        EXPECT_EQ(merged.gridSize(0), handle.gridSize(0));
        EXPECT_EQ(merged.gridSize(1), handle.gridSize(1));
        EXPECT_NE(handle.deviceGrid<float>(0), nullptr);
        EXPECT_NE(handle.deviceGrid<float>(1), nullptr);
        EXPECT_FALSE(handle.isPadded());
        // every allocation (grid bytes + meta scratch) was ordered on the retained stream
        for (auto s2 : recorder.allocStreams) EXPECT_EQ(stream, s2);
        EXPECT_EQ(2u, recorder.allocStreams.size());
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// Trait transitivity: a reference to (or adapter over) a host-accessible
// resource is itself host-accessible, so such buffers are not single-space.
static_assert(!nanovdb::BufferHasDeviceSingle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<nanovdb::cuda::PinnedResource>>>::value,
              "a ref to pinned storage is host-accessible");
static_assert(nanovdb::BufferHasDeviceSingle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<nanovdb::cuda::DeviceResource>>>::value,
              "a ref to device storage is single-space");

TEST(TestBuffer, GridHandleSingleSpaceRejectsForgedChain)
{
    // A valid first header whose mGridCount claims more grids than the buffer
    // holds must be rejected, never walked off the end of the allocation.
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    nanovdb::tools::updateGridCount(reinterpret_cast<nanovdb::GridData*>(host.data()), 0u, 2u);// forge: claims 2 grids

    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    BufT buf(cudaStream_t(0), host.bufferSize(), nanovdb::cuda::noInit);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice));
    EXPECT_THROW(nanovdb::GridHandle<BufT>{std::move(buf)}, std::runtime_error);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(cudaSuccess, cudaGetLastError());// the context was not poisoned

    {   // a corrupt second header in a genuine 2-grid chain must also be rejected
        auto a = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "a");
        auto b = nanovdb::tools::createLevelSetSphere<float>(10.0, nanovdb::Vec3d(2), 1.0, 3.0, nanovdb::Vec3d(0), "b");
        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> parts;
        parts.push_back(std::move(a));
        parts.push_back(std::move(b));
        auto merged = nanovdb::mergeGrids<nanovdb::HostBuffer, std::vector>(parts);// explicit args: MSVC cannot deduce the template-template VectorT
        std::memset(nanovdb::util::PtrAdd(merged.data(), merged.gridSize(0)), 0, 8);// clobber grid 1's magic
        BufT buf2(cudaStream_t(0), merged.bufferSize(), nanovdb::cuda::noInit);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(buf2.data(), merged.data(), merged.bufferSize(), cudaMemcpyHostToDevice));
        EXPECT_THROW(nanovdb::GridHandle<BufT>{std::move(buf2)}, std::runtime_error);
    }
}

TEST(TestBuffer, GridHandleSingleSpaceBorrowedResource)
{
    // A handle over Buffer<byte, ResourceRef<R>> exercises the whole feature
    // with a NON-default-constructible resource: construction, metadata
    // scratch through the borrowed instance, deep copy, and reset.
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    Counters counters;
    CountingResource res{&counters};
    using RefT = nanovdb::cuda::ResourceRef<CountingResource>;
    using BufT = nanovdb::cuda::Buffer<std::byte, RefT>;
    static_assert(nanovdb::BufferHasDeviceSingle<BufT>::value, "a ref to device storage is single-space");
    {
        BufT buf(cudaStream_t(0), RefT(res), host.bufferSize(), nanovdb::cuda::noInit);// alloc #1
        ASSERT_EQ(cudaSuccess, cudaMemcpy(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice));
        nanovdb::GridHandle<BufT> handle(std::move(buf));// alloc #2 + free #1 (meta scratch)
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_NE(handle.deviceGrid<float>(), nullptr);
        EXPECT_EQ(2, counters.allocs);

        auto copy = handle.copy<BufT>();// alloc #3, borrowed through the same instance
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(3, counters.allocs);
        EXPECT_NE(copy.deviceData(), handle.deviceData());
        EXPECT_EQ(copy.gridCount(), 1u);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);
}

TEST(TestBuffer, SingleSpaceNodeManager)
{
    // createNodeManager over an injected resource: the handle's storage and
    // its size scratch both allocate through it, and deviceMgr() maps onto
    // the single-space buffer.
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    BufT buf(cudaStream_t(0), host.bufferSize(), nanovdb::cuda::noInit);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(buf.data(), host.data(), host.bufferSize(), cudaMemcpyHostToDevice));
    nanovdb::GridHandle<BufT> handle(std::move(buf));
    auto *d_grid = handle.deviceGrid<float>();
    ASSERT_NE(d_grid, nullptr);

    Counters counters;
    CountingResource res{&counters};
    {
        auto mgrHandle = nanovdb::cuda::createNodeManager(d_grid, res);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        // a breadth-first grid takes the linear path: storage + size scratch only
        EXPECT_EQ(2, counters.allocs);
        EXPECT_EQ(1, counters.deallocs);// the size scratch
        EXPECT_NE(mgrHandle.deviceMgr<float>(), nullptr);
        EXPECT_EQ(mgrHandle.deviceMgr<nanovdb::Vec3f>(), nullptr);// wrong type: null, not garbage

        mgrHandle.reset();// dispatches to destroy(), through the same resource
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(counters.allocs, counters.deallocs);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);
}

TEST(TestBuffer, SingleSpaceToolEntryPoints)
{
    // The builders allocate their result handle through createDeviceStorage,
    // so a single-space buffer type works wherever a dual-space one does.
    // voxelsToGrid stands in for the whole PointsToGrid family; the pool
    // buffer supplies the resource for the handle storage.
    nanovdb::Coord coords[2] = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8)}, *d_coords = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coords, 2*sizeof(nanovdb::Coord)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords, coords, 2*sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

    Counters counters;
    CountingResource res{&counters};
    using RefT = nanovdb::cuda::ResourceRef<CountingResource>;
    using BufT = nanovdb::cuda::Buffer<std::byte, RefT>;
    {
        BufT pool(cudaStream_t(0), RefT(res), 16, nanovdb::cuda::noInit); // alloc #1: exemplar carrying the borrowed resource
        auto handle = nanovdb::tools::cuda::voxelsToGrid<float, nanovdb::Coord*, BufT>(d_coords, 2, 1.0, pool);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_NE(handle.deviceGrid<float>(), nullptr);
        EXPECT_EQ(3, counters.allocs); // #2: the grid storage, #3: the handle's metadata scratch
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);
    ASSERT_EQ(cudaSuccess, cudaFree(d_coords));
}

namespace {

/// @brief Host copy of every grid header in @a handle, for checking the
///        index, count, name and checksum the device-side merge/split wrote.
template<typename BufT>
nanovdb::GridHandle<nanovdb::HostBuffer> toHost(const nanovdb::GridHandle<BufT>& handle, cudaStream_t stream)
{
    return nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, stream);
}

} // namespace

TEST(TestBuffer, SingleSpaceMergeSplitGridHandles)
{
    // Sources live on one stream, the merge and split run on another, and the
    // sources are released right after each call returns.
    const char* names[3] = {"a", "b", "c"};
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> hosts;
    for (int i = 0; i < 3; ++i) {
        hosts.push_back(nanovdb::tools::createLevelSetSphere<float>(10.0 + 5.0*i, nanovdb::Vec3d(i), 1.0, 3.0, nanovdb::Vec3d(0), names[i]));
        ASSERT_FALSE(hosts.back().gridData()->mChecksum.isEmpty());// so the merge must refresh it
    }
    cudaStream_t srcStream, stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&srcStream, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    Counters counters;
    CountingResource res{&counters};
    using RefT = nanovdb::cuda::ResourceRef<CountingResource>;
    using BufT = nanovdb::cuda::Buffer<std::byte, RefT>;
    {
        const BufT proto(stream, RefT(res), 0, nanovdb::cuda::noInit);
        uint64_t size = 0u;
        auto merged = [&] {
            const BufT srcProto(srcStream, RefT(res), 0, nanovdb::cuda::noInit);
            std::vector<nanovdb::GridHandle<BufT>> sources;
            for (auto &h : hosts) {
                sources.push_back(nanovdb::cuda::copyTo<BufT>(h, srcStream, &srcProto));
                size += h.bufferSize();
            }
            return nanovdb::cuda::mergeGridHandles<BufT, std::vector>(sources, stream, &proto);
        }();// sources freed on return, on srcStream
        ASSERT_EQ(3u, merged.gridCount());
        EXPECT_EQ(size, merged.bufferSize());
        EXPECT_FALSE(merged.isPadded());
        EXPECT_EQ(stream, merged.buffer().stream());

        auto mergedHost = toHost(merged, stream);
        for (uint32_t i = 0; i < 3; ++i) {
            const nanovdb::GridData* data = mergedHost.gridData(i);
            ASSERT_NE(nullptr, data);
            EXPECT_EQ(i, data->mGridIndex);
            EXPECT_EQ(3u, data->mGridCount);
            EXPECT_STREQ(names[i], data->mGridName);
            EXPECT_TRUE(nanovdb::tools::validateChecksum(data, nanovdb::CheckMode::Full));
            EXPECT_EQ(hosts[i].gridSize(), merged.gridSize(i));
            EXPECT_EQ(nanovdb::GridType::Float, merged.gridType(i));
        }

        auto parts = nanovdb::cuda::splitGridHandles<BufT, std::vector>(merged, stream, &proto);
        merged.reset();// released right away
        ASSERT_EQ(3u, parts.size());
        for (uint32_t i = 0; i < 3; ++i) {
            ASSERT_EQ(1u, parts[i].gridCount());
            EXPECT_EQ(hosts[i].bufferSize(), parts[i].bufferSize());
            auto partHost = toHost(parts[i], stream);
            const nanovdb::GridData* data = partHost.gridData();
            ASSERT_NE(nullptr, data);
            EXPECT_EQ(0u, data->mGridIndex);
            EXPECT_EQ(1u, data->mGridCount);
            EXPECT_STREQ(names[i], data->mGridName);
            EXPECT_TRUE(nanovdb::tools::validateChecksum(data, nanovdb::CheckMode::Full));
            EXPECT_EQ(0, std::memcmp(hosts[i].data(), partHost.data(), hosts[i].bufferSize()));
        }
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(srcStream));
    EXPECT_GT(counters.allocs, 0);
    EXPECT_EQ(counters.allocs, counters.deallocs);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(srcStream));
}

TEST(TestBuffer, SingleSpaceMergeSkipsEmptyHandles)
{
    // Empty sources contribute no grids and are not waited on: a reset handle
    // that still retains a stream destroyed since, a moved-from handle that
    // retains a live stream, and a default handle. A non-empty source on
    // another live stream still gets its wait.
    using BufT = nanovdb::cuda::Buffer<std::byte>;
    auto host = nanovdb::tools::createLevelSetSphere<float>(10.0);
    cudaStream_t gone, live, mergeStream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&gone, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&live, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&mergeStream, cudaStreamNonBlocking));
    std::vector<nanovdb::GridHandle<BufT>> sources;
    sources.push_back(nanovdb::cuda::copyTo<BufT>(host, gone));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(gone));
    sources.back().reset();
    EXPECT_EQ(gone, sources.back().buffer().stream());
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(gone));
    auto movedFrom = nanovdb::cuda::copyTo<BufT>(host, live);
    auto source = std::move(movedFrom);
    sources.push_back(std::move(movedFrom));
    sources.emplace_back();
    sources.push_back(std::move(source));// non-empty, on live

    auto merged = nanovdb::cuda::mergeGridHandles<BufT, std::vector>(sources, mergeStream);
    ASSERT_EQ(1u, merged.gridCount());
    auto mergedHost = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(merged, mergeStream);
    EXPECT_EQ(0, std::memcmp(host.data(), mergedHost.data(), host.bufferSize()));
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    sources.clear();
    merged.reset();
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(live));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(mergeStream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(live));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(mergeStream));
}

TEST(TestBuffer, SingleSpaceMergeSplitSynchronousResource)
{
    // A synchronous resource retains no stream, so the results are complete
    // when the calls return.
    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::MallocResource>;
    std::vector<nanovdb::GridHandle<BufT>> sources;
    for (int i = 0; i < 2; ++i) {
        auto host = nanovdb::tools::createLevelSetSphere<float>(10.0 + 5.0*i, nanovdb::Vec3d(i), 1.0, 3.0, nanovdb::Vec3d(0), "s");
        sources.push_back(nanovdb::cuda::copyTo<BufT>(host, cudaStream_t(0)));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));// sources without a retained stream are the caller's to order
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto merged = nanovdb::cuda::mergeGridHandles<BufT, std::vector>(sources, stream);
    EXPECT_EQ(cudaSuccess, cudaStreamQuery(stream));
    ASSERT_EQ(2u, merged.gridCount());
    auto parts = nanovdb::cuda::splitGridHandles<BufT, std::vector>(merged, stream);
    EXPECT_EQ(cudaSuccess, cudaStreamQuery(stream));
    ASSERT_EQ(2u, parts.size());
    for (uint32_t i = 0; i < 2; ++i) {
        auto host = toHost(parts[i], stream);
        EXPECT_EQ(0u, host.gridData()->mGridIndex);
        EXPECT_EQ(1u, host.gridData()->mGridCount);
        EXPECT_TRUE(nanovdb::tools::validateChecksum(host.gridData(), nanovdb::CheckMode::Full));
    }
    EXPECT_TRUE((nanovdb::cuda::splitGridHandles<BufT, std::vector>(nanovdb::GridHandle<BufT>(), stream).empty()));
    EXPECT_TRUE((nanovdb::cuda::mergeGridHandles<BufT, std::vector>(std::vector<nanovdb::GridHandle<BufT>>(), stream).empty()));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(TestBuffer, SingleSpaceMergeSplitBatchBoundary)
{
    // More grids than one renumbering launch covers. Every 7th grid has no
    // checksum, and the renumbering must leave it empty.
    constexpr uint32_t N = 300;
    using BufT = nanovdb::cuda::Buffer<std::byte>;
    std::vector<nanovdb::GridHandle<BufT>> sources;
    for (uint32_t i = 0; i < N; ++i) {
        const auto cMode = i % 7 == 0 ? nanovdb::CheckMode::Disable : nanovdb::CheckMode::Full;
        auto host = nanovdb::tools::createLevelSetSphere<float>(3.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0),
                                                                "g" + std::to_string(i), nanovdb::tools::StatsMode::Default, cMode);
        sources.push_back(nanovdb::cuda::copyTo<BufT>(host, cudaStream_t(0)));
    }
    auto merged = nanovdb::cuda::mergeGridHandles<BufT, std::vector>(sources, cudaStream_t(0));
    sources.clear();
    auto mergedHost = toHost(merged, cudaStream_t(0));
    ASSERT_EQ(N, mergedHost.gridCount());
    for (uint32_t i = 0; i < N; ++i) {
        const nanovdb::GridData* data = mergedHost.gridData(i);
        EXPECT_EQ(i, data->mGridIndex);
        EXPECT_EQ(N, data->mGridCount);
        EXPECT_EQ(i % 7 == 0, data->mChecksum.isEmpty());
        EXPECT_TRUE(nanovdb::tools::validateChecksum(data, nanovdb::CheckMode::Full));
    }
    EXPECT_STREQ("g299", mergedHost.gridData(N - 1)->mGridName);

    auto parts = nanovdb::cuda::splitGridHandles<BufT, std::vector>(merged, cudaStream_t(0));
    merged.reset();
    ASSERT_EQ(N, parts.size());
    for (uint32_t i = 0; i < N; i += 37) {
        auto partHost = toHost(parts[i], cudaStream_t(0));
        const nanovdb::GridData* data = partHost.gridData();
        EXPECT_EQ(0u, data->mGridIndex);
        EXPECT_EQ(1u, data->mGridCount);
        EXPECT_EQ(i % 7 == 0, data->mChecksum.isEmpty());
        EXPECT_TRUE(nanovdb::tools::validateChecksum(data, nanovdb::CheckMode::Full));
    }
    auto lastHost = toHost(parts[N - 1], cudaStream_t(0));
    EXPECT_EQ(0u, lastHost.gridData()->mGridIndex);
    EXPECT_STREQ("g299", lastHost.gridData()->mGridName);
}

TEST(TestBuffer, SingleSpacePointSetEntryPoints)
{
    // The point-set overloads build one grid per set and merge them.
    nanovdb::Coord coords[3] = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8), nanovdb::Coord(-5,0,7)}, *d_coords = nullptr;
    nanovdb::Vec3f points[2] = {nanovdb::Vec3f(0.5f), nanovdb::Vec3f(4.25f, -1.0f, 2.0f)}, *d_points = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coords, sizeof(coords)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_points, sizeof(points)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords, coords, sizeof(coords), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_points, points, sizeof(points), cudaMemcpyHostToDevice));
    using BufT = nanovdb::cuda::Buffer<std::byte>;
    {
        std::vector<std::tuple<const nanovdb::Coord* const, size_t, double>> voxelSets{{d_coords, 2, 1.0}, {d_coords + 2, 1, 0.5}};
        auto handle = nanovdb::tools::cuda::voxelsToGrid<float, const nanovdb::Coord*, BufT>(voxelSets, BufT(), cudaStream_t(0));
        ASSERT_EQ(2u, handle.gridCount());
        auto host = toHost(handle, cudaStream_t(0));
        EXPECT_EQ(2u, host.grid<float>(0)->activeVoxelCount());
        EXPECT_EQ(1u, host.grid<float>(1)->activeVoxelCount());
        EXPECT_EQ(1u, host.gridData(1)->mGridIndex);
        EXPECT_EQ(2u, host.gridData(1)->mGridCount);
    }
    {
        std::vector<std::tuple<nanovdb::Vec3f* const, size_t, double, nanovdb::PointType>> pointSets{
            {d_points, 2, 1.0, nanovdb::PointType::Default}, {d_points, 1, 0.5, nanovdb::PointType::Default}};
        auto handle = nanovdb::tools::cuda::pointsToGrid<nanovdb::Point, nanovdb::Vec3f*, BufT>(pointSets, BufT(), cudaStream_t(0));
        ASSERT_EQ(2u, handle.gridCount());
        auto host = toHost(handle, cudaStream_t(0));
        EXPECT_EQ(2u, host.grid<nanovdb::Point>(0)->pointCount());
        EXPECT_EQ(1u, host.grid<nanovdb::Point>(1)->pointCount());
    }
    {
        auto handle = nanovdb::tools::cuda::pointsToGrid<nanovdb::Vec3f*, BufT>(d_points, 2, 1.0, nanovdb::PointType::Default, BufT(), cudaStream_t(0));
        ASSERT_EQ(1u, handle.gridCount());
        EXPECT_EQ(2u, toHost(handle, cudaStream_t(0)).grid<nanovdb::Point>()->pointCount());
    }
    ASSERT_EQ(cudaSuccess, cudaFree(d_points));
    ASSERT_EQ(cudaSuccess, cudaFree(d_coords));
}

static_assert(nanovdb::BufferHasDeviceSingle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ManagedResource>>::value,
              "managed storage is device-accessible");
static_assert(nanovdb::BufferHasHostSingle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ManagedResource>>::value,
              "managed storage is host-accessible");

TEST(TestBuffer, ManagedBufferBothSpaces)
{
    // A managed-resource buffer serves grids read on both sides: the handle
    // parses metadata on the host and exposes BOTH accessor families over
    // the same allocation.
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ManagedResource>;
    BufT buf(host.bufferSize(), nanovdb::cuda::noInit);
    std::memcpy(buf.data(), host.data(), host.bufferSize()); // managed memory is host-writable

    nanovdb::GridHandle<BufT> handle(std::move(buf));
    EXPECT_EQ(1u, handle.gridCount());
    ASSERT_NE(handle.grid<float>(), nullptr); // host accessor
    ASSERT_NE(handle.deviceGrid<float>(), nullptr); // device accessor, same bytes
    EXPECT_EQ((const void*)handle.grid<float>(), (const void*)handle.deviceGrid<float>());
    EXPECT_EQ(std::string("sphere"), handle.grid<float>()->gridName());

    // built on the device through a tool entry point, read back on the host
    nanovdb::Coord coords[2] = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8)}, *d_coords = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coords, 2*sizeof(nanovdb::Coord)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords, coords, 2*sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto built = nanovdb::tools::cuda::voxelsToGrid<float, nanovdb::Coord*, BufT>(d_coords, 2);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize()); // device writes must land before host reads
    ASSERT_NE(built.grid<float>(), nullptr);
    EXPECT_TRUE(built.grid<float>()->tree().isActive(nanovdb::Coord(1,2,3)));

    nanovdb::cuda::ManagedResource managed;
    auto mgr = nanovdb::cuda::createNodeManager(built.deviceGrid<float>(), managed);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_NE(mgr.deviceMgr<float>(), nullptr); // a managed NodeManager serves the device...
    EXPECT_NE(mgr.mgr<float>(), nullptr); // ...and the host
    ASSERT_EQ(cudaSuccess, cudaFree(d_coords));
}

TEST(TestBuffer, SingleSpaceVoxelBlockManager)
{
    // The VoxelBlockManager entry point allocates through createDeviceStorage
    // and its handle maps the device accessors onto single-space buffers.
    nanovdb::Coord coords[2] = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8)}, *d_coords = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coords, 2*sizeof(nanovdb::Coord)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords, coords, 2*sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto gridHandle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex, nanovdb::Coord*>(d_coords, 2);
    auto* d_grid = gridHandle.deviceGrid<nanovdb::ValueOnIndex>();
    ASSERT_NE(d_grid, nullptr);

    using BufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    auto vbm = nanovdb::tools::cuda::buildVoxelBlockManager<6, BufT>(d_grid);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_GT(vbm.blockCount(), 0u);
    EXPECT_NE(vbm.deviceFirstLeafID(), nullptr);
    EXPECT_NE(vbm.deviceJumpMap(), nullptr);

    Counters counters;
    CountingResource res{&counters};
    using RefT = nanovdb::cuda::ResourceRef<CountingResource>;
    using RefBufT = nanovdb::cuda::Buffer<std::byte, RefT>;
    {
        RefBufT proto(cudaStream_t(0), RefT(res), 16, nanovdb::cuda::noInit); // exemplar carrying the borrowed resource
        auto vbm2 = nanovdb::tools::cuda::buildVoxelBlockManager<6, RefBufT>(d_grid, 0, 0, 0, cudaStream_t(0), &proto);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(3, counters.allocs); // proto + firstLeafID + jumpMap, all through the resource
        EXPECT_NE(vbm2.deviceFirstLeafID(), nullptr);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);
    ASSERT_EQ(cudaSuccess, cudaFree(d_coords));
}

//======================================================================
// The GPU builders write into a single-space buffer exactly what they write
// into the dual-space buffer that the rest of the suite tests against
//======================================================================

using SingleBufT = nanovdb::cuda::Buffer<std::byte>;
using DualBufT   = nanovdb::cuda::DualDeviceBuffer;

/// @brief Host copy of a handle of either buffer family, for byte comparison.
template<typename BufT>
nanovdb::GridHandle<nanovdb::HostBuffer> hostImage(nanovdb::GridHandle<BufT>& handle)
{
    if constexpr (nanovdb::BufferTraits<BufT>::hasDeviceDual) {
        handle.deviceDownload();
        return handle.template copy<nanovdb::HostBuffer>();
    } else {
        return nanovdb::cuda::copyTo<nanovdb::HostBuffer>(handle, cudaStream_t(0));
    }
}

/// @brief Requires two host images to hold the same, non-empty grids, byte for byte.
void expectSameBytes(const nanovdb::GridHandle<nanovdb::HostBuffer>& a, const nanovdb::GridHandle<nanovdb::HostBuffer>& b)
{
    EXPECT_GT(a.gridCount(), 0u);
    EXPECT_EQ(a.gridCount(), b.gridCount());
    EXPECT_EQ(a.bufferSize(), b.bufferSize());
    if (a.bufferSize() == b.bufferSize() && a.bufferSize() > 0)
        EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.bufferSize()));
    if (a.gridCount() > 0) EXPECT_GT(a.gridMetaData()->activeVoxelCount(), 0u); // not trivially equal
}

/// @brief Runs @a build once with a dual-space and once with a single-space
///        prototype buffer, requires byte-identical, non-empty grids, and
///        returns the host image for further checks.
template<typename BuildFn>
nanovdb::GridHandle<nanovdb::HostBuffer> expectSameGridInBothFamilies(BuildFn build)
{
    auto dual   = build(DualBufT());
    auto single = build(SingleBufT());
    static_assert(std::is_same<decltype(single), nanovdb::GridHandle<SingleBufT>>::value,
                  "the builder must return a handle over the prototype's buffer type");
    auto a = hostImage(dual);
    expectSameBytes(a, hostImage(single));
    return a;
}

/// @brief Device copy of host data.
template<typename T>
nanovdb::cuda::Buffer<T> toDevice(const std::vector<T>& host)
{
    nanovdb::cuda::Buffer<T> buf(cudaStream_t(0), host.size(), nanovdb::cuda::noInit);
    if (cudaMemcpy(buf.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("toDevice: upload failed");
    return buf;
}

/// @brief Voxels spanning two root tiles, two upper nodes, several lower
///        nodes and leaves, so the builders' multi-node paths all run.
std::vector<nanovdb::Coord> indexCoords()
{
    return {nanovdb::Coord(1,0,0), nanovdb::Coord(0,1,1), nanovdb::Coord(127,127,127), nanovdb::Coord(4,0,0),
            nanovdb::Coord(0,64,0), nanovdb::Coord(0,0,2048), nanovdb::Coord(-1,-1,-1)};
}

/// @brief Single-space grid over indexCoords(), with the device coordinates it was built from.
struct IndexInput
{
    std::vector<nanovdb::Coord>              coords;
    nanovdb::cuda::Buffer<nanovdb::Coord>    d_coords;
    nanovdb::GridHandle<SingleBufT>          handle;
};

template<typename BuildT>
IndexInput makeIndexInput()
{
    auto coords   = indexCoords();
    auto d_coords = toDevice(coords);
    auto handle   = nanovdb::tools::cuda::voxelsToGrid<BuildT, nanovdb::Coord*, SingleBufT>(d_coords.data(), coords.size(), 1.0, SingleBufT());
    return {std::move(coords), std::move(d_coords), std::move(handle)};
}

TEST(TestBuffer, SingleSpaceDilateMatchesDual)
{
    using BuildT = nanovdb::ValueOnIndex;
    auto in = makeIndexInput<BuildT>();
    auto& input = in.handle;
    expectSameGridInBothFamilies([&](const auto& proto) {
        nanovdb::tools::cuda::DilateGrid<BuildT> op(input.deviceGrid<BuildT>());
        op.setOperation(nanovdb::tools::morphology::NN_FACE_EDGE_VERTEX);
        op.setChecksum(nanovdb::CheckMode::Full);
        return op.getHandle(proto);
    });
}

TEST(TestBuffer, SingleSpaceMergeMatchesDual)
{
    using BuildT = nanovdb::ValueOnIndex;
    auto inA = makeIndexInput<BuildT>();
    auto& a = inA.handle;
    const std::vector<nanovdb::Coord> coordsB = {nanovdb::Coord(-8,3,5), nanovdb::Coord(1,0,0), nanovdb::Coord(300,-40,9)};
    auto d_coordsB = toDevice(coordsB);
    auto b = nanovdb::tools::cuda::voxelsToGrid<BuildT, nanovdb::Coord*, SingleBufT>(d_coordsB.data(), coordsB.size(), 1.0, SingleBufT());
    expectSameGridInBothFamilies([&](const auto& proto) {
        nanovdb::tools::cuda::MergeGrids<BuildT> op(a.deviceGrid<BuildT>(), b.deviceGrid<BuildT>());
        op.setChecksum(nanovdb::CheckMode::Full);
        return op.getHandle(proto);
    });
}

TEST(TestBuffer, SingleSpacePruneMatchesDual)
{
    using BuildT = nanovdb::ValueOnIndex;
    auto in = makeIndexInput<BuildT>();
    auto& input = in.handle;
    const uint32_t leafCount = hostImage(input).grid<BuildT>()->tree().nodeCount(0);
    // Retain only voxel 0 of each leaf. Of the input voxels only (0,64,0) and
    // (0,0,2048) are voxel 0 of their leaf, so the prune keeps those two and
    // removes every other voxel and the leaves left empty.
    std::vector<nanovdb::Mask<3>> masks(leafCount);
    for (auto& m : masks) m.setOn(0);
    auto d_masks = toDevice(masks);
    auto pruned = expectSameGridInBothFamilies([&](const auto& proto) {
        nanovdb::tools::cuda::PruneGrid<BuildT> op(input.deviceGrid<BuildT>(), d_masks.data());
        op.setChecksum(nanovdb::CheckMode::Full);
        return op.getHandle(proto);
    });
    ASSERT_GT(pruned.gridCount(), 0u);
    EXPECT_EQ(2u, pruned.gridMetaData()->activeVoxelCount());
    EXPECT_EQ(2u, pruned.gridMetaData()->nodeCount(0));
}

TEST(TestBuffer, SingleSpaceRefineAndCoarsenMatchDual)
{
    using BuildT = nanovdb::ValueOnIndex;
    auto in = makeIndexInput<BuildT>();
    auto& input = in.handle;
    expectSameGridInBothFamilies([&](const auto& proto) {
        nanovdb::tools::cuda::RefineGrid<BuildT> op(input.deviceGrid<BuildT>());
        op.setChecksum(nanovdb::CheckMode::Full);
        return op.getHandle(proto);
    });
    expectSameGridInBothFamilies([&](const auto& proto) {
        nanovdb::tools::cuda::CoarsenGrid<BuildT> op(input.deviceGrid<BuildT>());
        op.setChecksum(nanovdb::CheckMode::Full);
        return op.getHandle(proto);
    });
}

TEST(TestBuffer, SingleSpaceMeshToGridMatchesDual)
{
    using BuildT = nanovdb::ValueOnIndex;
    const std::vector<nanovdb::Vec3f> points = {{0.f,0.f,0.f}, {1.f,0.f,0.f}, {0.f,1.f,0.f}, {0.f,0.f,1.f}};
    const std::vector<nanovdb::Vec3i> triangles = {{0,1,2}, {0,2,3}, {0,3,1}, {1,2,3}};
    auto d_points = toDevice(points);
    auto d_triangles = toDevice(triangles);
    nanovdb::Map map;
    map.set(0.1, nanovdb::Vec3d(0.0));
    auto makeOp = [&] {
        return nanovdb::tools::cuda::MeshToGrid<BuildT>(d_points.data(), uint32_t(points.size()),
                                                        d_triangles.data(), uint32_t(triangles.size()), map);
    };
    expectSameGridInBothFamilies([&](const auto& proto) { auto op = makeOp(); return op.getHandle(proto); });

    // getHandleAndUDF: the grid and its distance sidecar, in both families
    auto dual   = makeOp().getHandleAndUDF<DualBufT, DualBufT>();
    auto single = makeOp().getHandleAndUDF<SingleBufT, SingleBufT>(SingleBufT(), SingleBufT());
    expectSameBytes(hostImage(dual.first), hostImage(single.first));
    ASSERT_EQ(dual.second.size(), single.second.size_bytes());
    std::vector<std::byte> udfA(dual.second.size()), udfB(single.second.size_bytes());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(udfA.data(), dual.second.deviceData(), udfA.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(udfB.data(), single.second.data(), udfB.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(0, std::memcmp(udfA.data(), udfB.data(), udfA.size()));
}

TEST(TestBuffer, SingleSpaceIndexToGridMatchesDual)
{
    using BuildT = nanovdb::ValueOnIndex;
    auto in = makeIndexInput<BuildT>();
    auto& input = in.handle;
    const uint64_t valueCount = hostImage(input).grid<BuildT>()->valueCount();
    std::vector<float> values(valueCount);
    for (uint64_t i = 0; i < valueCount; ++i) values[i] = 0.5f * float(i);
    auto d_values = toDevice(values);
    auto viaIndexToGrid = expectSameGridInBothFamilies([&](const auto& proto) {
        return nanovdb::tools::cuda::indexToGrid<float, BuildT>(input.deviceGrid<BuildT>(), d_values.data(), proto);
    });
    // tools::cuda::createNanoGrid is a forwarder to indexToGrid: same grid
    auto viaCreateNanoGrid = expectSameGridInBothFamilies([&](const auto& proto) {
        return nanovdb::tools::cuda::createNanoGrid<float, BuildT>(input.deviceGrid<BuildT>(), d_values.data(), proto);
    });
    expectSameBytes(viaIndexToGrid, viaCreateNanoGrid);
}

TEST(TestBuffer, SingleSpaceAddBlindDataMatchesDual)
{
    auto in = makeIndexInput<float>();
    auto& input = in.handle;
    std::vector<float> blind(in.coords.size());
    for (size_t i = 0; i < blind.size(); ++i) blind[i] = 1.25f * float(i);
    auto d_blind = toDevice(blind);
    // Two additions: the second shifts the first's metadata offsets.
    auto twice = expectSameGridInBothFamilies([&](const auto& proto) {
        auto once = nanovdb::tools::cuda::addBlindData(input.deviceGrid<float>(), d_blind.data(), blind.size(),
                                                       nanovdb::GridBlindDataClass::AttributeArray,
                                                       nanovdb::GridBlindDataSemantic::Unknown, "weights", proto);
        return nanovdb::tools::cuda::addBlindData(once.template deviceGrid<float>(), d_blind.data(), blind.size(),
                                                  nanovdb::GridBlindDataClass::AttributeArray,
                                                  nanovdb::GridBlindDataSemantic::Unknown, "more", proto);
    });
    ASSERT_GT(twice.gridCount(), 0u);
    EXPECT_EQ(2u, twice.gridData()->mBlindMetadataCount);
}

TEST(TestBuffer, MaxPointsPerVoxelPointsToGridInBothFamilies)
{
    // The density-targeting overload, which nothing else instantiates.
    // Scattered points near the origin and their mirror at negative
    // coordinates, so the grid spans more than one upper node and root tile.
    // The points come from a fixed linear congruential sequence, identical on
    // every platform, rather than a lattice whose densities jump in steps the
    // voxel-size search cannot land between.
    std::vector<nanovdb::Vec3f> points;
    uint32_t state = 12345u;
    auto next = [&state] { state = 1664525u * state + 1013904223u; return float(state >> 8) / float(1u << 24); };
    for (int i = 0; i < 128; ++i) {
        const nanovdb::Vec3f p(next(), next(), next());
        points.push_back(p);
        points.push_back(-p - nanovdb::Vec3f(0.05f));
    }
    auto d_points = toDevice(points);
    constexpr int maxPointsPerVoxel = 4, tolerance = 1;
    auto host = expectSameGridInBothFamilies([&](const auto& proto) {
        using BufT = std::decay_t<decltype(proto)>;
        return nanovdb::tools::cuda::pointsToGrid<nanovdb::Vec3f*, BufT>(d_points.data(), int(points.size()),
                                                                         maxPointsPerVoxel, tolerance, /*maxIterations=*/10,
                                                                         nanovdb::PointType::Default, proto, cudaStream_t(0));
    });
    // The search shrank the voxels from the initial unit size until the
    // densest voxel holds the target count, within the tolerance.
    const auto* grid = host.grid<nanovdb::Point>();
    ASSERT_TRUE(grid);
    EXPECT_LT(grid->voxelSize()[0], 1.0);
    EXPECT_GE(grid->tree().root().tileCount(), 2u);
    nanovdb::PointAccessor<nanovdb::Vec3f, nanovdb::Point> acc(*grid);
    ASSERT_TRUE(acc);
    uint64_t total = 0, densest = 0;
    const auto* leaf = grid->tree().getFirstLeaf();
    for (uint32_t i = 0; i < grid->tree().nodeCount(0); ++i, ++leaf) {
        for (auto it = leaf->beginValueOn(); it; ++it) {
            const nanovdb::Vec3f *begin = nullptr, *end = nullptr;
            const uint64_t count = acc.voxelPoints(it.getCoord(), begin, end);
            total += count;
            densest = std::max(densest, count);
        }
    }
    EXPECT_EQ(points.size(), total);
    EXPECT_LE(densest, uint64_t(maxPointsPerVoxel + tolerance));
    EXPECT_GE(densest, uint64_t(maxPointsPerVoxel - tolerance));
}

TEST(TestBuffer, TypedEvalChecksumMatchesGridDataForm)
{
    // The typed and GridData* device overloads agree with each other and
    // with the host checksum on current-version grids, the only kind a
    // builder produces.
    auto in = makeIndexInput<float>();
    const nanovdb::NanoGrid<float>* d_grid = in.handle.deviceGrid<float>();
    auto host = hostImage(in.handle);
    for (auto mode : {nanovdb::CheckMode::Half, nanovdb::CheckMode::Full}) {
        const auto typed   = nanovdb::tools::cuda::evalChecksum(d_grid, mode);
        const auto generic = nanovdb::tools::cuda::evalChecksum(reinterpret_cast<const nanovdb::GridData*>(d_grid), mode);
        const auto onHost  = nanovdb::tools::evalChecksum(host.grid<float>(), mode);
        EXPECT_TRUE(typed == generic);
        EXPECT_TRUE(typed == onHost);
        EXPECT_FALSE(typed.isEmpty());
    }
}


//======================================================================
// Resource forms of the GPU tools: output and scratch allocate through the
// caller's resource, beside the unchanged prototype forms
//======================================================================

namespace {

nanovdb::GridHandle<nanovdb::HostBuffer> deviceToHost(const nanovdb::GridHandle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<CountingResource>>>& h)
{
    return nanovdb::cuda::copyTo<nanovdb::HostBuffer>(h, cudaStream_t(0));
}

} // namespace

TEST(TestBuffer, ResourceFormsOfTheGpuTools)
{
    using RefT = nanovdb::cuda::ResourceRef<CountingResource>;
    using OutT = nanovdb::cuda::Buffer<std::byte, RefT>;
    const std::vector<nanovdb::Coord> coords = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8), nanovdb::Coord(-5,0,7), nanovdb::Coord(300,0,0)};
    const std::vector<nanovdb::Vec3f> points = {nanovdb::Vec3f(0.5f), nanovdb::Vec3f(4.25f,-1.0f,2.0f), nanovdb::Vec3f(0.6f), nanovdb::Vec3f(9.0f)};
    nanovdb::cuda::Buffer<nanovdb::Coord> d_coords(cudaStream_t(0), coords.size(), nanovdb::cuda::noInit);
    nanovdb::cuda::Buffer<nanovdb::Vec3f> d_points(cudaStream_t(0), points.size(), nanovdb::cuda::noInit);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords.data(), coords.data(), coords.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_points.data(), points.data(), points.size() * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    Counters counters;
    CountingResource res{&counters};
    {
        // voxelsToGrid: the resource form matches the prototype form, and every
        // allocation (scratch and output) goes through the caller's resource.
        auto proto = nanovdb::tools::cuda::voxelsToGrid<float>(d_coords.data(), coords.size(), 1.0);
        static_assert(std::is_same<decltype(proto), nanovdb::GridHandle<nanovdb::cuda::DualDeviceBuffer>>::value,
                      "a call without a stream keeps selecting the prototype form");
        auto viaRes = nanovdb::tools::cuda::voxelsToGrid<float>(d_coords.data(), coords.size(), 1.0, stream, RefT(res));
        static_assert(std::is_same<decltype(viaRes), nanovdb::GridHandle<OutT>>::value, "the resource form returns a buffer over that resource");
        EXPECT_EQ(stream, viaRes.buffer().stream());
        const int afterVoxels = counters.allocs;
        EXPECT_GT(afterVoxels, 1); // scratch and output
        auto host = deviceToHost(viaRes);
        expectSameBytes(host, hostImage(proto));

        // default resource: no resource argument at all
        auto viaDefault = nanovdb::tools::cuda::voxelsToGrid<float>(d_coords.data(), coords.size(), 1.0, stream);
        static_assert(std::is_same<decltype(viaDefault), nanovdb::GridHandle<nanovdb::cuda::Buffer<std::byte>>>::value,
                      "the resource form defaults to DeviceResource");
        expectSameBytes(host, hostImage(viaDefault));
        EXPECT_EQ(afterVoxels, counters.allocs); // the default resource is not the counting one

        // pointsToGrid, fixed voxel size and maximum density
        auto pFixed = nanovdb::tools::cuda::pointsToGrid(d_points.data(), int(points.size()), 1.0, nanovdb::PointType::Default, stream, RefT(res));
        static_assert(std::is_same<decltype(pFixed), nanovdb::GridHandle<OutT>>::value, "resource form");
        EXPECT_EQ(int(points.size()), int(deviceToHost(pFixed).grid<nanovdb::Point>()->pointCount()));
        auto pFixedProto = nanovdb::tools::cuda::pointsToGrid(d_points.data(), int(points.size()), 1.0);
        expectSameBytes(deviceToHost(pFixed), hostImage(pFixedProto));
        // the prototype forms validate their arguments the same way
        EXPECT_THROW((nanovdb::tools::cuda::voxelsToGrid<float>(d_coords.data(), coords.size(), -1.0)), std::invalid_argument);
        EXPECT_THROW((nanovdb::tools::cuda::pointsToGrid(d_points.data(), -1, 1.0)), std::invalid_argument);
        auto pDense = nanovdb::tools::cuda::pointsToGrid(d_points.data(), int(points.size()), 2, 1, 10, nanovdb::PointType::Default, stream, RefT(res));
        auto pDenseProto = nanovdb::tools::cuda::pointsToGrid(d_points.data(), int(points.size()), 2, 1, 10);
        expectSameBytes(deviceToHost(pDense), hostImage(pDenseProto));

        // indexToGrid over a resource form index grid
        auto idx = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex>(d_coords.data(), coords.size(), 1.0, stream, RefT(res));
        const uint64_t valueCount = deviceToHost(idx).grid<nanovdb::ValueOnIndex>()->valueCount();
        std::vector<float> values(valueCount);
        for (uint64_t i = 0; i < valueCount; ++i) values[i] = 0.25f * float(i);
        nanovdb::cuda::Buffer<float> d_values(cudaStream_t(0), valueCount, nanovdb::cuda::noInit);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_values.data(), values.data(), valueCount * sizeof(float), cudaMemcpyHostToDevice));
        auto floats = nanovdb::tools::cuda::indexToGrid<float>(idx.deviceGrid<nanovdb::ValueOnIndex>(), d_values.data(), stream, RefT(res));
        static_assert(std::is_same<decltype(floats), nanovdb::GridHandle<OutT>>::value, "resource form");
        auto floatsProto = nanovdb::tools::cuda::indexToGrid<float>(idx.deviceGrid<nanovdb::ValueOnIndex>(), d_values.data());
        expectSameBytes(deviceToHost(floats), hostImage(floatsProto));
        // createNanoGrid forwards to indexToGrid in both forms
        auto created = nanovdb::tools::cuda::createNanoGrid<float>(idx.deviceGrid<nanovdb::ValueOnIndex>(), d_values.data(), stream, RefT(res));
        static_assert(std::is_same<decltype(created), nanovdb::GridHandle<OutT>>::value, "resource form");
        expectSameBytes(deviceToHost(created), hostImage(floatsProto));
        auto createdDefault = nanovdb::tools::cuda::createNanoGrid<float>(idx.deviceGrid<nanovdb::ValueOnIndex>(), d_values.data(), stream);
        static_assert(std::is_same<decltype(createdDefault), nanovdb::GridHandle<nanovdb::cuda::Buffer<std::byte>>>::value,
                      "a stream selects the resource form, as for indexToGrid");
        expectSameBytes(deviceToHost(created), hostImage(createdDefault));

        // addBlindData: its scratch used to come from the default resource only,
        // so a non-default-constructible resource could not drive it at all.
        const int beforeBlind = counters.allocs;
        auto withBlind = nanovdb::tools::cuda::addBlindData(viaRes.deviceGrid<float>(), d_values.data(), coords.size(),
                                                            nanovdb::GridBlindDataClass::AttributeArray, nanovdb::GridBlindDataSemantic::Unknown,
                                                            "w", stream, RefT(res));
        static_assert(std::is_same<decltype(withBlind), nanovdb::GridHandle<OutT>>::value, "resource form");
        EXPECT_GE(counters.allocs, beforeBlind + 2); // scratch and output
        EXPECT_EQ(1u, deviceToHost(withBlind).gridData()->mBlindMetadataCount);
        auto withBlindProto = nanovdb::tools::cuda::addBlindData(proto.deviceGrid<float>(), d_values.data(), coords.size(),
                                                                 nanovdb::GridBlindDataClass::AttributeArray, nanovdb::GridBlindDataSemantic::Unknown, "w");
        expectSameBytes(deviceToHost(withBlind), hostImage(withBlindProto));

        // invalid arguments throw instead of overflowing or dereferencing null
        const std::string longName(nanovdb::GridBlindMetaData::MaxNameSize, 'x');
        auto addNamed = [&](const char* name) {
            return nanovdb::tools::cuda::addBlindData(viaRes.deviceGrid<float>(), d_values.data(), coords.size(),
                                                      nanovdb::GridBlindDataClass::AttributeArray, nanovdb::GridBlindDataSemantic::Unknown,
                                                      name, stream, RefT(res));
        };
        EXPECT_THROW(addNamed(nullptr), std::invalid_argument);
        EXPECT_THROW(addNamed(longName.c_str()), std::invalid_argument);
        EXPECT_THROW((nanovdb::tools::cuda::addBlindData(static_cast<const nanovdb::NanoGrid<float>*>(nullptr), d_values.data(), coords.size(),
                                                         nanovdb::GridBlindDataClass::AttributeArray, nanovdb::GridBlindDataSemantic::Unknown,
                                                         "w", stream, RefT(res))), std::invalid_argument);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream)); // the null grid was never launched on
        EXPECT_NO_THROW(addNamed(longName.substr(1).c_str())); // the longest name that fits
        EXPECT_THROW((nanovdb::tools::cuda::voxelsToGrid<float>(d_coords.data(), coords.size(), 0.0, stream)), std::invalid_argument);
        EXPECT_THROW((nanovdb::tools::cuda::pointsToGrid(d_points.data(), -1, 1.0, nanovdb::PointType::Default, stream)), std::invalid_argument);
        EXPECT_THROW((nanovdb::tools::cuda::indexToGrid<float, nanovdb::ValueOnIndex>(nullptr, d_values.data(), stream)), std::invalid_argument);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(TestBuffer, BuilderConstructorsValidateArguments)
{
    // Each builder constructor rejects its arguments before allocating or launching.
    const cudaStream_t stream = 0;
    EXPECT_THROW((nanovdb::tools::cuda::IndexToGrid<nanovdb::ValueOnIndex>(nullptr, stream)), std::invalid_argument);
    EXPECT_THROW((nanovdb::tools::cuda::PointsToGrid<float>(-1.0)), std::invalid_argument);
    EXPECT_THROW((nanovdb::tools::cuda::PointsToGrid<float>(0.0)), std::invalid_argument);
    EXPECT_THROW((nanovdb::tools::cuda::PointsToGrid<float>(std::numeric_limits<double>::infinity())), std::invalid_argument);
    EXPECT_THROW((nanovdb::tools::cuda::PointsToGrid<float>(nanovdb::Map(0.0))), std::invalid_argument);
    EXPECT_THROW((nanovdb::tools::cuda::PointsToGrid<nanovdb::Point>(0)), std::invalid_argument); // maximum points per voxel
    EXPECT_NO_THROW((nanovdb::tools::cuda::PointsToGrid<float>(0.5)));
    Counters counters;
    CountingResource res{&counters};
    EXPECT_THROW((nanovdb::tools::cuda::IndexToGrid<nanovdb::ValueOnIndex, CountingResource>(nullptr, stream, res)), std::invalid_argument);
    EXPECT_THROW((nanovdb::tools::cuda::PointsToGrid<float, CountingResource>(nanovdb::Map(0.0), stream, res)), std::invalid_argument);
    EXPECT_THROW((nanovdb::tools::cuda::PointsToGrid<nanovdb::Point, CountingResource>(0, 1, 10, stream, res)), std::invalid_argument);
    EXPECT_EQ(0, counters.allocs);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
}

TEST(TestBuffer, BuilderBuildAllocatesThroughTheBuilderResource)
{
    // build() is getHandle with the output drawn from the builder's own
    // resource: same grid, and every allocation through that one resource.
    using BuildT = nanovdb::ValueOnIndex;
    using RefT = nanovdb::cuda::ResourceRef<CountingResource>;
    using OutT = nanovdb::cuda::Buffer<std::byte, RefT>;
    const std::vector<nanovdb::Coord> coords = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8), nanovdb::Coord(-5,0,7), nanovdb::Coord(300,0,0)};
    nanovdb::cuda::Buffer<nanovdb::Coord> d_coords(cudaStream_t(0), coords.size(), nanovdb::cuda::noInit);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords.data(), coords.data(), coords.size() * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

    Counters counters;
    CountingResource res{&counters};
    {
        nanovdb::tools::cuda::PointsToGrid<BuildT, CountingResource> p2g(1.0, nanovdb::Vec3d(0), cudaStream_t(0), res);
        auto input = p2g.build(d_coords.data(), coords.size());
        static_assert(std::is_same<decltype(input), nanovdb::GridHandle<OutT>>::value, "build borrows the builder resource");
        auto inputProto = nanovdb::tools::cuda::PointsToGrid<BuildT>().getHandle(d_coords.data(), coords.size());
        expectSameBytes(deviceToHost(input), hostImage(inputProto));
        const nanovdb::NanoGrid<BuildT>* d_grid = input.deviceGrid<BuildT>();

        auto check = [&](auto& op, auto& opProto) {
            const int before = counters.allocs;
            auto out = op.build();
            static_assert(std::is_same<decltype(out), nanovdb::GridHandle<OutT>>::value, "build borrows the builder resource");
            EXPECT_GT(counters.allocs, before); // the output, at least
            auto proto = opProto.getHandle();
            expectSameBytes(deviceToHost(out), hostImage(proto));
        };
        { nanovdb::tools::cuda::DilateGrid<BuildT, CountingResource> op(d_grid, cudaStream_t(0), res);
          nanovdb::tools::cuda::DilateGrid<BuildT> ref(d_grid); check(op, ref); }
        { nanovdb::tools::cuda::MergeGrids<BuildT, CountingResource> op(d_grid, d_grid, cudaStream_t(0), res);
          nanovdb::tools::cuda::MergeGrids<BuildT> ref(d_grid, d_grid); check(op, ref); }
        { nanovdb::tools::cuda::RefineGrid<BuildT, CountingResource> op(d_grid, cudaStream_t(0), res);
          nanovdb::tools::cuda::RefineGrid<BuildT> ref(d_grid); check(op, ref); }
        { nanovdb::tools::cuda::CoarsenGrid<BuildT, CountingResource> op(d_grid, cudaStream_t(0), res);
          nanovdb::tools::cuda::CoarsenGrid<BuildT> ref(d_grid); check(op, ref); }
        {
            const uint32_t leafCount = deviceToHost(input).grid<BuildT>()->tree().nodeCount(0);
            std::vector<nanovdb::Mask<3>> masks(leafCount, nanovdb::Mask<3>(true));
            nanovdb::cuda::Buffer<nanovdb::Mask<3>> d_masks(cudaStream_t(0), leafCount, nanovdb::cuda::noInit);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(d_masks.data(), masks.data(), leafCount * sizeof(nanovdb::Mask<3>), cudaMemcpyHostToDevice));
            nanovdb::tools::cuda::PruneGrid<BuildT, CountingResource> op(d_grid, d_masks.data(), cudaStream_t(0), res);
            nanovdb::tools::cuda::PruneGrid<BuildT> ref(d_grid, d_masks.data());
            check(op, ref);
        }
        {
            const uint64_t valueCount = deviceToHost(input).grid<BuildT>()->valueCount();
            std::vector<float> values(valueCount, 2.5f);
            nanovdb::cuda::Buffer<float> d_values(cudaStream_t(0), valueCount, nanovdb::cuda::noInit);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(d_values.data(), values.data(), valueCount * sizeof(float), cudaMemcpyHostToDevice));
            nanovdb::tools::cuda::IndexToGrid<BuildT, CountingResource> op(d_grid, cudaStream_t(0), res);
            auto out = op.build<float>(d_values.data());
            static_assert(std::is_same<decltype(out), nanovdb::GridHandle<OutT>>::value, "build borrows the builder resource");
            auto proto = nanovdb::tools::cuda::IndexToGrid<BuildT>(d_grid).getHandle<float>(d_values.data());
            expectSameBytes(deviceToHost(out), hostImage(proto));
        }
        {
            const std::vector<nanovdb::Vec3f> points = {{0.f,0.f,0.f}, {1.f,0.f,0.f}, {0.f,1.f,0.f}, {0.f,0.f,1.f}};
            const std::vector<nanovdb::Vec3i> triangles = {{0,1,2}, {0,2,3}, {0,3,1}, {1,2,3}};
            nanovdb::cuda::Buffer<nanovdb::Vec3f> d_points(cudaStream_t(0), points.size(), nanovdb::cuda::noInit);
            nanovdb::cuda::Buffer<nanovdb::Vec3i> d_tris(cudaStream_t(0), triangles.size(), nanovdb::cuda::noInit);
            ASSERT_EQ(cudaSuccess, cudaMemcpy(d_points.data(), points.data(), points.size() * sizeof(nanovdb::Vec3f), cudaMemcpyHostToDevice));
            ASSERT_EQ(cudaSuccess, cudaMemcpy(d_tris.data(), triangles.data(), triangles.size() * sizeof(nanovdb::Vec3i), cudaMemcpyHostToDevice));
            nanovdb::Map map;
            map.set(0.1, nanovdb::Vec3d(0.0));
            nanovdb::tools::cuda::MeshToGrid<BuildT, CountingResource> op(d_points.data(), 4u, d_tris.data(), 4u, map, cudaStream_t(0), res);
            nanovdb::tools::cuda::MeshToGrid<BuildT> ref(d_points.data(), 4u, d_tris.data(), 4u, map);
            check(op, ref);
            nanovdb::tools::cuda::MeshToGrid<BuildT, CountingResource> op2(d_points.data(), 4u, d_tris.data(), 4u, map, cudaStream_t(0), res);
            auto withUdf = op2.buildWithUDF();
            auto refUdf = nanovdb::tools::cuda::MeshToGrid<BuildT>(d_points.data(), 4u, d_tris.data(), 4u, map).getHandleAndUDF();
            expectSameBytes(deviceToHost(withUdf.first), hostImage(refUdf.first));
            ASSERT_EQ(refUdf.second.size(), withUdf.second.size_bytes());
            std::vector<std::byte> a(withUdf.second.size_bytes()), b(refUdf.second.size());
            ASSERT_EQ(cudaSuccess, cudaMemcpy(a.data(), withUdf.second.data(), a.size(), cudaMemcpyDeviceToHost));
            ASSERT_EQ(cudaSuccess, cudaMemcpy(b.data(), refUdf.second.deviceData(), b.size(), cudaMemcpyDeviceToHost));
            EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));
        }
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_GT(counters.allocs, 0);
    EXPECT_EQ(counters.allocs, counters.deallocs);
}

//======================================================================
// GridHandle::adopt
//======================================================================

TEST(TestBuffer, GridHandleAdoptRejectsBadMeta)
{
    auto host = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "sphere");
    using DevBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    auto dev = nanovdb::cuda::copyTo<DevBufT>(host, cudaStream_t(0));
    const uint64_t bytes = dev.bufferSize();
    ASSERT_EQ(1u, dev.metaData().size());
    ASSERT_GT(bytes, 2 * sizeof(nanovdb::GridData) + 16);

    // adopt reads no header, so the layout check is what judges an unrelated allocation
    auto tryAdopt = [&](std::vector<nanovdb::GridHandleMetaData> meta) {
        DevBufT buf(cudaStream_t(0), bytes, nanovdb::cuda::noInit);
        return nanovdb::GridHandle<DevBufT>::adopt(std::move(buf), std::move(meta));
    };
    using Meta = nanovdb::GridHandleMetaData;
    const auto Float = nanovdb::GridType::Float;
    const uint64_t half = nanovdb::math::AlignUp<NANOVDB_DATA_ALIGNMENT>(bytes / 2u);
    // the extra parentheses keep the braced lists' commas from splitting the macro arguments
    EXPECT_THROW((tryAdopt({})), std::runtime_error);// no grids over a non-empty buffer
    EXPECT_NO_THROW((nanovdb::GridHandle<DevBufT>::adopt(DevBufT(), {})));// the empty handle
    EXPECT_THROW((tryAdopt({Meta{0u, sizeof(nanovdb::GridData) - 1u, Float}})), std::runtime_error);// smaller than a header
    EXPECT_THROW((tryAdopt({Meta{0u, bytes + 1u, Float}})), std::runtime_error);// past the end
    EXPECT_THROW((tryAdopt({Meta{8u, bytes - 8u, Float}})), std::runtime_error);// does not start at zero
    EXPECT_THROW((tryAdopt({Meta{0u, half, Float}, Meta{half + 32u, bytes - half - 32u, Float}})), std::runtime_error);// gap
    EXPECT_THROW((tryAdopt({Meta{0u, half, Float}, Meta{half - 32u, bytes - half, Float}})), std::runtime_error);// overlap
    EXPECT_THROW((tryAdopt({Meta{0u, half + 8u, Float}, Meta{half + 8u, bytes - half - 8u, Float}})), std::runtime_error);// misaligned
    EXPECT_NO_THROW((tryAdopt({Meta{0u, half, Float}, Meta{half, bytes - half, Float}})));// contiguous
    EXPECT_NO_THROW((tryAdopt({Meta{0u, half, Float}})));// padded buffer
    EXPECT_THROW((tryAdopt({Meta{0u, bytes, nanovdb::GridType::End}})), std::runtime_error);// invalid grid type

    // the real use: a byte-for-byte copy adopts the source's layout and reads back identical
    DevBufT copy(cudaStream_t(0), bytes, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(copy.data(), dev.deviceData(), bytes, cudaMemcpyDeviceToDevice, cudaStream_t(0)));
    auto twin = nanovdb::GridHandle<DevBufT>::adopt(std::move(copy), dev.metaData());
    EXPECT_EQ(dev.gridCount(), twin.gridCount());
    EXPECT_EQ(dev.gridSize(0), twin.gridSize(0));
    EXPECT_EQ(dev.gridType(0), twin.gridType(0));
    EXPECT_NE(twin.deviceGrid<float>(0), nullptr);
    auto back = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(twin, cudaStream_t(0));
    EXPECT_EQ(0, std::memcmp(back.data(), host.data(), bytes));
}

TEST(TestBuffer, CopyToKeepsUnalignedGrids)
{
    // Two grids where the first grid's size is not a multiple of NANOVDB_DATA_ALIGNMENT,
    // as in Point grids written before their payload was padded. The constructors accept
    // this layout, so copyTo, copy() and splitGridHandles must carry it. Public adopt rejects
    // it, and so does mergeGridHandles, whose output would hold a misaligned device grid.
    auto a = nanovdb::tools::createLevelSetSphere<float>(20.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "a");
    auto b = nanovdb::tools::createLevelSetSphere<float>(10.0, nanovdb::Vec3d(0), 1.0, 3.0, nanovdb::Vec3d(0), "b");
    const uint64_t sizeA = a.bufferSize() + 8u, bytes = sizeA + b.bufferSize();
    auto buffer = nanovdb::HostBuffer::create(bytes);
    std::memset(buffer.data(), 0, bytes);
    std::memcpy(buffer.data(), a.data(), a.bufferSize());
    std::memcpy(nanovdb::util::PtrAdd(buffer.data(), sizeA), b.data(), b.bufferSize());
    auto *first = reinterpret_cast<nanovdb::GridData*>(buffer.data());
    first->mGridSize = sizeA;
    nanovdb::tools::updateGridCount(first, 0u, 2u);
    nanovdb::tools::updateGridCount(nanovdb::util::PtrAdd<nanovdb::GridData>(buffer.data(), sizeA), 1u, 2u);
    nanovdb::GridHandle<nanovdb::HostBuffer> host(std::move(buffer));
    ASSERT_EQ(2u, host.gridCount());
    ASSERT_EQ(sizeA, host.metaData()[1].offset);

    using DevBufT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>;
    nanovdb::GridHandle<DevBufT> dev;
    ASSERT_NO_THROW(dev = nanovdb::cuda::copyTo<DevBufT>(host, cudaStream_t(0)));
    nanovdb::GridHandle<DevBufT> twin;
    ASSERT_NO_THROW(twin = dev.copy<DevBufT>());
    auto back = nanovdb::cuda::copyTo<nanovdb::HostBuffer>(twin, cudaStream_t(0));
    ASSERT_EQ(bytes, back.bufferSize());
    EXPECT_EQ(sizeA, back.metaData()[1].offset);
    EXPECT_EQ(0, std::memcmp(back.data(), host.data(), bytes));

    auto copy = nanovdb::HostBuffer::create(bytes);
    std::memcpy(copy.data(), host.data(), bytes);
    EXPECT_THROW(nanovdb::GridHandle<nanovdb::HostBuffer>::adopt(std::move(copy), host.metaData()), std::runtime_error);

    auto parts = nanovdb::cuda::splitGridHandles<DevBufT, std::vector>(dev, cudaStream_t(0));
    ASSERT_EQ(2u, parts.size());
    EXPECT_EQ(sizeA, parts[0].bufferSize());
    EXPECT_EQ(b.bufferSize(), parts[1].bufferSize());
    std::vector<nanovdb::GridHandle<DevBufT>> devs;
    devs.push_back(std::move(dev));
    EXPECT_THROW((nanovdb::cuda::mergeGridHandles<DevBufT, std::vector>(devs, cudaStream_t(0))), std::runtime_error);
}

} // namespace nanovdb_test
