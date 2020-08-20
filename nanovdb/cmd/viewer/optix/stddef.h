// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#pragma once

#ifdef NANOVDB_OPTIX_RTC_WIN32
#define __LP64__
#define _WIN64
#endif

#include <limits.h>

namespace detail {
#if __cplusplus >= 201103L
typedef decltype(nullptr) nullptr_t;
#if defined(_MSC_VER)
typedef double max_align_t;
#elif defined(__APPLE__)
typedef long double max_align_t;
#else
// Define max_align_t to match the GCC definition.
typedef struct
{
    long long __max_align_nonce1
        __attribute__((__aligned__(__alignof__(long long))));
    long double __max_align_nonce2
        __attribute__((__aligned__(__alignof__(long double))));
} max_align_t;
#endif
#endif // __cplusplus >= 201103L
#if __cplusplus >= 201703L
enum class byte : unsigned char {};
#endif // __cplusplus >= 201703L
} // namespace detail

namespace std {
// NVRTC provides built-in definitions of ::size_t and ::ptrdiff_t.
using ::size_t;
using ::ptrdiff_t;
//using namespace detail;
} // namespace std

//using namespace detail;
