// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file util/x86.cc

#include "x86.h"

#include <llvm/Support/Host.h>
#include <llvm/ADT/StringMap.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {
namespace x86 {

CpuFlagStatus CheckX86Feature(const std::string& flag)
{
    llvm::StringMap<bool> HostFeatures;
    if (!llvm::sys::getHostCPUFeatures(HostFeatures)) {
        return CpuFlagStatus::Unknown;
    }
    if (!HostFeatures.empty()) {
        for (auto& feature : HostFeatures) {
            if (feature.first() == flag) {
                return feature.second ?
                    CpuFlagStatus::Supported :
                    CpuFlagStatus::Unsupported;
            }
        }
    }
    return CpuFlagStatus::Unknown;
}

}
}
}
}
