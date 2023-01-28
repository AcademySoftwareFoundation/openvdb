// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file util/x86.h

#ifndef OPENVDB_AX_UTIL_X86_HAS_BEEN_INCLUDED
#define OPENVDB_AX_UTIL_X86_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {
namespace x86 {

enum class CpuFlagStatus {
    Unknown, Unsupported, Supported
};

/// @brief  On X86, get the status if a particular CPU instruction
/// @param flag  The flag to check. e.g. avx, bmi, f16c, etc
/// @note   Returns Unknown if the flag was not found. This could either be
///   because the platform is not X86, because the flag is not a valid X86
///   feature or because the feature is too new for this version of AX/LLVM.
OPENVDB_AX_API CpuFlagStatus CheckX86Feature(const std::string& flag);

}
}
}
}

#endif // OPENVDB_AX_UTIL_X86_HAS_BEEN_INCLUDED
