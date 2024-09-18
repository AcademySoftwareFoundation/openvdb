// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file Assert.h

#ifndef OPENVDB_UTIL_ASSERT_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_ASSERT_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief  Trigger a SIGABRT after printing a formatted assertion message.
///   Effectively performs the same functionality as cassert, but allows
///   VDB code to call this independently of the NDEBUG define.
/// @param assertion  The variable or expression that triggered the assertion
///   as a string
/// @param file  The name of the file the assertion occurred
/// @param line  The line in the file the assertion occurred
/// @param function  The name of the function the assertion occurred in
/// @param msg  An optional descriptive message
[[noreturn]] void assertAbort(
    const char *assertion,
    const char *file,
    const unsigned line,
    const char *function,
    const char* msg = nullptr);

}
}

#ifdef OPENVDB_ENABLE_ASSERTS
#define OPENVDB_ASSERT(X) \
  (OPENVDB_LIKELY(X) ? (void)0 : openvdb::assertAbort(#X, __FILE__, __LINE__, __PRETTY_FUNCTION__))
#define OPENVDB_ASSERT_MESSAGE(X, MSG) \
  (OPENVDB_LIKELY(X) ? (void)0 : openvdb::assertAbort(#X, __FILE__, __LINE__, __PRETTY_FUNCTION__, MSG))
#else
#define OPENVDB_ASSERT(X) (void)0;
#define OPENVDB_ASSERT_MESSAGE(X, MSG) (void)0;
#endif // OPENVDB_ENABLE_ASSERTS

#endif // OPENVDB_UTIL_ASSERT_HAS_BEEN_INCLUDED
