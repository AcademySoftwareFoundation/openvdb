// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/VolumeKernelFunctions.h
///
/// @authors Nick Avramoussis
///
/// @brief  The definitions of the public functions built by the
///   VolumeComputeGenerator.
///

#ifndef OPENVDB_AX_VOLUME_KERNEL_FUNCTIONS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_VOLUME_KERNEL_FUNCTIONS_HAS_BEEN_INCLUDED

#include "Types.h"
#include <openvdb/version.h>

#include <array>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  The primary volume kernel. This function holds the generated body
///  of AX programs.
/// @details  argument structure is as follows:
///   1) - A void pointer to the ax::CustomData
///   2) - A pointer to an array of three ints representing the
///        current voxel coord being accessed
///   3) - A void pointer to the current value buffer
///   4) - A bool representing the current values active state
///   5) - The index of the current tile in the parent tile's table
///   6) - A void pointer to a vector of void pointers, representing
///        an array of grid accessors
///   7) - A void pointer to a vector of void pointers, representing
///        an array of grid transforms
///   8) - The index of currently executing volume in the list of write
///        accessible volumes.
struct VolumeKernelValue
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3],
             void*,    // value
             bool,     // active
             int64_t,  // index
             void**,   // r accessors
             const void* const*,
             int64_t);

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<std::string, N_ARGS>& argumentKeys();
    static const char* getDefaultName();
};

/// @brief  The second volume kernel, responsible for providing the core
///  layer of SIMD optimisations by invoking this kernel across a range of
///  values.
/// @details  argument structure is as follows:
///   1) - A void pointer to the ax::CustomData
///   2) - A pointer to an array of three ints representing the
///        current voxel coord being accessed
///   3) - A void pointer to the current value buffer
///   4) - A uint64_t pointer to the active word buffer
///   5) - The active state execution mode
///   6) - A void pointer to a vector of void pointers, representing
///        an array of grid accessors
///   7) - A void pointer to a vector of void pointers, representing
///        an array of grid transforms
///   8) - The index of currently executing volume in the list of write
///        accessible volumes.
struct VolumeKernelBuffer
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3],
             void*,    // value buffer
             uint64_t*, // active buffer
             int64_t,  // buffer size
             uint64_t,  // mode (0 = off, 1 = active, 2 = both)
             void**, // read accessors
             const void* const*, // transforms
             int64_t); // write index

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<std::string, N_ARGS>& argumentKeys();
    static const char* getDefaultName();
};

/// @brief  The third volume kernel, providing an agnostic way to modify
///   a single tile value without passing through the buffer states. Note
///   that this kernel is mainly utility and one of the value kernels should
///   almost always be preferred.
/// @details  argument structure is as follows:
///   1) - A void pointer to the ax::CustomData
///   2) - A pointer to an array of three ints representing the
///        current voxel coord being accessed
///   3) - A void pointer to a vector of void pointers, representing
///        an array of grid accessors
///   4) - A void pointer to a vector of void pointers, representing
///        an array of grid transforms
///   5) - The index of currently executing volume in the list of write
///        accessible volumes.
///   5) - A unique write accessor to the target volume.
struct VolumeKernelNode
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3], // index space coord
             void**, // read accessors
             const void* const*, // transforms
             int64_t, // write index
             void*); // write accessor

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<std::string, N_ARGS>& argumentKeys();
    static const char* getDefaultName();
};

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_VOLUME_KERNEL_FUNCTIONS_HAS_BEEN_INCLUDED

