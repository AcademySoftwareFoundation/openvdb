// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/VolumeComputeGenerator.h
///
/// @authors Nick Avramoussis
///
/// @brief  The visitor framework and function definition for volume grid
///   code generation
///

#ifndef OPENVDB_AX_VOLUME_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED
#define OPENVDB_AX_VOLUME_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

#include "ComputeGenerator.h"
#include "FunctionTypes.h"

#include "../compiler/AttributeRegistry.h"

#include <openvdb/version.h>

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

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

namespace codegen_internal {

/// @brief Visitor object which will generate llvm IR for a syntax tree which has been generated
///        from AX that targets volumes.  The IR will represent a single function. It is mainly
///        used by the Compiler class.
struct VolumeComputeGenerator : public ComputeGenerator
{
    /// @brief Constructor
    /// @param module           llvm Module for generating IR
    /// @param options          Options for the function registry behaviour
    /// @param functionRegistry Function registry object which will be used when generating IR
    ///                         for function calls
    /// @param logger           Logger for collecting logical errors and warnings
    VolumeComputeGenerator(llvm::Module& module,
        const FunctionOptions& options,
        FunctionRegistry& functionRegistry,
        Logger& logger);

    ~VolumeComputeGenerator() override = default;

    using ComputeGenerator::traverse;
    using ComputeGenerator::visit;

    AttributeRegistry::Ptr generate(const ast::Tree& node);
    bool visit(const ast::Attribute*) override;

private:
    llvm::Value* accessorHandleFromToken(const std::string&);
    void getAccessorValue(const std::string&, llvm::Value*);

    void computek2(llvm::Function*, const AttributeRegistry&);
    void computek3(llvm::Function*, const AttributeRegistry&);
};

} // namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_VOLUME_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

