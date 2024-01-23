// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file Platform.h

#include "Assert.h"

#include <cstdio>
#include <cstdlib>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

[[noreturn]] void assertAbort(
    const char *assertion,
    const char *file,
    const unsigned line,
    const char *function,
    const char* msg)
{
    std::fprintf(stderr, "%s:%u:", file, line);
    std::fprintf(stderr, " Assertion failed: ");
    std::fprintf(stderr, "'%s'", assertion);
    std::fprintf(stderr, " in function: ");
    std::fprintf(stderr, "'%s'", function);
    if (msg) std::fprintf(stderr, "\n%s", msg);
    std::fprintf(stderr, "\n");
    // @todo  could make this optional with another compile define
    std::abort();
}

}
}
