// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file TestUtilCuda.cu
///
/// @brief Unit tests for the util::cuda allocation wrappers: the cached
///        memory-pool capability query, and an allocate/use/free round-trip
///        through whichever mode the build selects (stream-ordered by default,
///        synchronous under NANOVDB_USE_SYNC_CUDA_MALLOC). This file is compiled
///        into two test binaries -- one per mode -- so both are exercised. On a
///        device without memory pools the default (async) wrappers fail by
///        design; such a device must build with NANOVDB_USE_SYNC_CUDA_MALLOC,
///        so the async round-trip skips when pools are absent.

#include <nanovdb/util/cuda/Util.h>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

namespace {

TEST(TestUtilCuda, MemoryPoolsSupportedMatchesAttribute)
{
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_GT(count, 0);
    for (int i = 0; i < count; ++i) {
        int attr = 0;
        ASSERT_EQ(cudaDeviceGetAttribute(&attr, cudaDevAttrMemoryPoolsSupported, i), cudaSuccess);
        EXPECT_EQ(nanovdb::util::cuda::memoryPoolsSupported(i), attr != 0) << "device " << i;
    }
    // Out-of-range device ids are unsupported, not undefined.
    EXPECT_FALSE(nanovdb::util::cuda::memoryPoolsSupported(-1));
    EXPECT_FALSE(nanovdb::util::cuda::memoryPoolsSupported(count));
}

TEST(TestUtilCuda, MallocAsyncRoundTrip)
{
    // Allocate/use/free through the wrapper in whichever mode the build selects:
    // synchronous under the macro, stream-ordered otherwise. Without the macro
    // the async path requires memory pools, so skip when the device lacks them.
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
#ifndef NANOVDB_USE_SYNC_CUDA_MALLOC
    if (!nanovdb::util::cuda::memoryPoolsSupported(device))
        GTEST_SKIP() << "device " << device << " lacks memory pools; the async wrappers fail by design "
                        "(build with NANOVDB_USE_SYNC_CUDA_MALLOC)";
#endif
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    const size_t n = 4096;
    void* d = nullptr;
    ASSERT_EQ(nanovdb::util::cuda::mallocAsync(&d, n, s), cudaSuccess);
    ASSERT_NE(d, nullptr);
    ASSERT_EQ(cudaMemsetAsync(d, 0x3C, n, s), cudaSuccess);

    std::vector<unsigned char> host(n, 0);
    ASSERT_EQ(cudaMemcpyAsync(host.data(), d, n, cudaMemcpyDeviceToHost, s), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    for (size_t i = 0; i < n; ++i)
        ASSERT_EQ(host[i], 0x3Cu) << "byte " << i;

    ASSERT_EQ(nanovdb::util::cuda::freeAsync(d, s), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

} // unnamed namespace
