// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/VolumeComputeGenerator.cc

#include "VolumeKernelFunctions.h"

#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

const std::array<std::string, VolumeKernelValue::N_ARGS>&
VolumeKernelValue::argumentKeys()
{
    static const std::array<std::string, VolumeKernelValue::N_ARGS> arguments = {{
        "custom_data",
        "origin",
        "value",
        "active",
        "offset",
        "accessors",
        "transforms",
        "write_index"
    }};

    return arguments;
}

const char* VolumeKernelValue::getDefaultName() { return "ax.compute.voxel.k1"; }

//

const std::array<std::string, VolumeKernelBuffer::N_ARGS>&
VolumeKernelBuffer::argumentKeys()
{
    static const std::array<std::string, VolumeKernelBuffer::N_ARGS> arguments = {{
        "custom_data",
        "origin",
        "value_buffer",
        "active_buffer",
        "buffer_size",
        "mode",
        "accessors",
        "transforms",
        "write_index"
    }};

    return arguments;
}

const char* VolumeKernelBuffer::getDefaultName() { return "ax.compute.voxel.k2"; }

//

const std::array<std::string, VolumeKernelNode::N_ARGS>&
VolumeKernelNode::argumentKeys()
{
    static const std::array<std::string, VolumeKernelNode::N_ARGS> arguments = {{
        "custom_data",
        "coord_is",
        "accessors",
        "transforms",
        "write_index",
        "write_acccessor"
    }};

    return arguments;
}

const char* VolumeKernelNode::getDefaultName() { return "ax.compute.voxel.k3"; }

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

