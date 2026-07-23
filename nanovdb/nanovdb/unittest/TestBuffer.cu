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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
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

TEST(TestBuffer, ClearFreesAndEmpties)
{
    Counters c;
    nanovdb::cuda::Buffer<float, CountingResource> buf(0, CountingResource{&c}, 64, nanovdb::cuda::noInit);
    ASSERT_EQ(c.allocs, 1);
    buf.clear();
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(c.deallocs, 1);
    buf.clear();                 // idempotent: no second free
    EXPECT_EQ(c.deallocs, 1);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

//======================================================================
// Stream retention: the destructor and resize free on the retained stream
// (stream of the most recent allocation, or the one supplied via setStream)
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
        buf.setStream(b); // member update only, no synchronization
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
struct HasSetStream<B, std::void_t<decltype(std::declval<B&>().setStream(cudaStream_t{0}))>> : std::true_type {};

template<class B, class = void>
struct HasStreamGetter : std::false_type {};
template<class B>
struct HasStreamGetter<B, std::void_t<decltype(std::declval<const B&>().stream())>> : std::true_type {};

using PinnedBufferF = nanovdb::cuda::Buffer<float, nanovdb::cuda::PinnedResource>;
using DeviceBufferF = nanovdb::cuda::Buffer<float, nanovdb::cuda::DeviceResource>;

// A Buffer over a synchronous resource exposes no stream API at all.
static_assert(!HasSetStream<PinnedBufferF>::value, "sync-resource Buffer must not expose setStream");
static_assert(!HasStreamGetter<PinnedBufferF>::value, "sync-resource Buffer must not expose stream()");
static_assert(HasSetStream<DeviceBufferF>::value, "async-resource Buffer exposes setStream");
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
    buf.clear();
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

    view.clear(); // detaches, does not free
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

} // unnamed namespace
