// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// This test validates that including the vendored nanovdb_blosc and vendored
// openvdb_blosc while also defining a symbol from an external blosc together
// in the same translation unit does not result in any symbol clashes or
// undefined symbols. Compression round trips are covered by
// TestStreamCompression.

#include <nanovdb_blosc.h>

// Stand-in for an external blosc.h, which uses the same guard and macro names
#ifndef BLOSC_H
#define BLOSC_H
#define BLOSC_MAX_OVERHEAD 32
#define BLOSC_EXPORT
extern "C" int blosc_compress_ctx(int, int, size_t, size_t, const void*, void*,
    size_t, const char*, size_t, int);
#endif

#include <openvdb_blosc.h>

#include <gtest/gtest.h>

#include <type_traits>

// decltype is not evaluated, so this checks the declarations without needing
// either library to export or link the functions.
static_assert(std::is_same<decltype(&nanovdb_blosc_init),
    decltype(&openvdb_blosc_init)>::value, "blosc declarations differ");
static_assert(std::is_same<decltype(&nanovdb_blosc_compress_ctx),
    decltype(&openvdb_blosc_compress_ctx)>::value, "blosc declarations differ");
static_assert(std::is_same<decltype(&nanovdb_blosc_decompress_ctx),
    decltype(&openvdb_blosc_decompress_ctx)>::value, "blosc declarations differ");
static_assert(std::is_same<decltype(&nanovdb_blosc_cbuffer_sizes),
    decltype(&openvdb_blosc_cbuffer_sizes)>::value, "blosc declarations differ");

class TestBloscHeaders: public ::testing::Test
{
};

TEST_F(TestBloscHeaders, testMacrosDoNotCollide)
{
    // The external definition is untouched by the vendored headers
    EXPECT_EQ(32, BLOSC_MAX_OVERHEAD);
    EXPECT_EQ(16, OPENVDB_BLOSC_MAX_OVERHEAD);
    EXPECT_EQ(16, NANOVDB_BLOSC_MAX_OVERHEAD);
}
