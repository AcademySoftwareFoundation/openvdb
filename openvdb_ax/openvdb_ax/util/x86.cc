// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file util/x86.cc

#include "x86.h"

#include <llvm/Config/llvm-config.h>
#if LLVM_VERSION_MAJOR < 18
#include <llvm/Support/Host.h>
#else
#include <llvm/TargetParser/Host.h>
#endif
#include <llvm/ADT/StringMap.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {
namespace x86 {

CpuFlagStatus CheckX86Feature(const std::string& flag)
{
#if LLVM_VERSION_MAJOR <= 18
    llvm::StringMap<bool> HostFeatures;
    if (!llvm::sys::getHostCPUFeatures(HostFeatures)) {
        return CpuFlagStatus::Unknown;
    }
#else
    llvm::StringMap<bool> HostFeatures = llvm::sys::getHostCPUFeatures();
#endif
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
