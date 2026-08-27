// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file TestBuffer.cu
///
/// @brief Unit tests for the typed, resource-aware CUDA containers
///        (cuda::Buffer, cuda::BufferView) and the synchronous Resource
///        trait (cuda::is_resource).

#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/PinnedResource.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>// for the voxelsToGrid entry-point test
#include <nanovdb/cuda/ManagedResource.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>// for the single-space entry-point test

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <sstream>// for the pinned-handle write/read round trip
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

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
    // An element count whose byte size overflows size_t must be rejected up
    // front rather than silently wrapping into a tiny allocation.
    const size_t oversized = std::numeric_limits<size_t>::max() / sizeof(int) + 1;
    EXPECT_THROW((nanovdb::cuda::Buffer<int>(0, oversized, nanovdb::cuda::noInit)), std::runtime_error);
    EXPECT_THROW((nanovdb::cuda::Buffer<int, nanovdb::cuda::PinnedResource>(oversized, nanovdb::cuda::noInit)), std::runtime_error);
    nanovdb::cuda::Buffer<int> buf(0, 8, nanovdb::cuda::noInit);
    cudaStream_t other = nullptr;
    ASSERT_EQ(cudaStreamCreate(&other), cudaSuccess);
    EXPECT_THROW(buf.resize(oversized, other), std::runtime_error);
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

TEST(TestBuffer, ResourceRefEqualityIsIdentity)
{
    StatefulInlineResource a, b;
    nanovdb::cuda::ResourceRef<StatefulInlineResource> ra(a), raAgain(a), rb(b);
    EXPECT_TRUE(ra == raAgain);   // same underlying resource
    EXPECT_TRUE(ra != rb);        // different underlying resources
}

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

        buf.resize(m, c); // grow on stream c
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

        buf.resize(n / 2, c); // shrink: keeps the min(old,new) prefix
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
        EXPECT_EQ(2, counters.allocs);// #2: the grid storage; the metadata is adopted from the source handle, so no scratch
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
        BufT pool(cudaStream_t(0), RefT(res), 16, nanovdb::cuda::noInit);// alloc #1: exemplar carrying the borrowed resource
        auto handle = nanovdb::tools::cuda::voxelsToGrid<float, nanovdb::Coord*, BufT>(d_coords, 2, 1.0, pool);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(1u, handle.gridCount());
        EXPECT_NE(handle.deviceGrid<float>(), nullptr);
        EXPECT_EQ(3, counters.allocs);// #2: the grid storage, #3: the handle's metadata scratch
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);
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
    std::memcpy(buf.data(), host.data(), host.bufferSize());// managed memory is host-writable

    nanovdb::GridHandle<BufT> handle(std::move(buf));
    EXPECT_EQ(1u, handle.gridCount());
    ASSERT_NE(handle.grid<float>(), nullptr);// host accessor
    ASSERT_NE(handle.deviceGrid<float>(), nullptr);// device accessor, same bytes
    EXPECT_EQ((const void*)handle.grid<float>(), (const void*)handle.deviceGrid<float>());
    EXPECT_EQ(std::string("sphere"), handle.grid<float>()->gridName());

    // built on the device through a tool entry point, read back on the host
    nanovdb::Coord coords[2] = {nanovdb::Coord(1,2,3), nanovdb::Coord(10,20,8)}, *d_coords = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_coords, 2*sizeof(nanovdb::Coord)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_coords, coords, 2*sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));
    auto built = nanovdb::tools::cuda::voxelsToGrid<float, nanovdb::Coord*, BufT>(d_coords, 2);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());// device writes must land before host reads
    ASSERT_NE(built.grid<float>(), nullptr);
    EXPECT_TRUE(built.grid<float>()->tree().isActive(nanovdb::Coord(1,2,3)));

    nanovdb::cuda::ManagedResource managed;
    auto mgr = nanovdb::cuda::createNodeManager(built.deviceGrid<float>(), managed);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_NE(mgr.deviceMgr<float>(), nullptr);// a managed NodeManager serves the device...
    EXPECT_NE(mgr.mgr<float>(), nullptr);// ...and the host
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
        RefBufT proto(cudaStream_t(0), RefT(res), 16, nanovdb::cuda::noInit);// exemplar carrying the borrowed resource
        auto vbm2 = nanovdb::tools::cuda::buildVoxelBlockManager<6, RefBufT>(d_grid, 0, 0, 0, cudaStream_t(0), &proto);
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
        EXPECT_EQ(3, counters.allocs);// proto + firstLeafID + jumpMap, all through the resource
        EXPECT_NE(vbm2.deviceFirstLeafID(), nullptr);
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(0));
    EXPECT_EQ(counters.allocs, counters.deallocs);
    ASSERT_EQ(cudaSuccess, cudaFree(d_coords));
}

} // unnamed namespace
