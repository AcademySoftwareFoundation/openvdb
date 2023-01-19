// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/PointComputeGenerator.h
///
/// @authors Nick Avramoussis, Matt Warner, Francisco Gochez, Richard Jones
///
/// @brief  The visitor framework and function definition for point data
///   grid code generation
///

#ifndef OPENVDB_AX_POINT_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED
#define OPENVDB_AX_POINT_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

#include "ComputeGenerator.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include "../compiler/AttributeRegistry.h"

#include <openvdb/version.h>
#include <openvdb/points/AttributeArray.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct PointKernelValue
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3],
             Index32*, // leaf value buffer
             bool,     // active
             uint64_t,  // pindex
             void**,   // transforms
             void**,   // values
             uint64_t*, // flags
             const void* const, // attribute set
             void**, // group handles
             void*);  // leaf data

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<const char*, N_ARGS>& argumentKeys();
    static const char* getDefaultName();
};

struct PointKernelAttributeArray
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3],
             Index32*, // leaf value buffer
             bool,     // active
             uint64_t,  // pindex
             void**,   // transforms
             void**,   // arrays
             uint64_t*, // flags
             const void* const, // attribute set
             void**, // group handles
             void*);  // leaf data

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<const char*, N_ARGS>& argumentKeys();
    static const char* getDefaultName();
};

struct PointKernelBuffer
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3],
             Index32*, // leaf value buffer
             bool,     // active
             uint64_t,  // pindex
             void**,   // transforms
             void**,   // buffers
             uint64_t*, // flags
             const void* const, // attribute set
             void**, // group handles
             void*);  // leaf data

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<const char*, N_ARGS>& argumentKeys();
    static const char* getDefaultName();
};

struct PointKernelBufferRange
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3],
             Index32*, // leaf value buffer
             uint64_t*, // active buffer
             int64_t,  // leaf buffer size (512)
             uint64_t,  // mode (0 = off, 1 = active, 2 = both)
             void**,   // transforms
             void**,   // buffers
             uint64_t*, // flags
             const void* const, // attribute set
             void**, // group handles
             void*);  // leaf data

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<const char*, N_ARGS>& argumentKeys();
    static const char* getDefaultName();
};


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

namespace codegen_internal {

/// @brief Visitor object which will generate llvm IR for a syntax tree which has been generated from
///        AX that targets point grids.  The IR will represent  2 functions : one that executes over
///        single points and one that executes over a collection of points.  This is primarily used by the
///        Compiler class.
struct PointComputeGenerator : public ComputeGenerator
{
    /// @brief Constructor
    /// @param module           llvm Module for generating IR
    /// @param options          Options for the function registry behaviour
    /// @param functionRegistry Function registry object which will be used when generating IR
    ///                         for function calls
    /// @param logger           Logger for collecting logical errors and warnings
    PointComputeGenerator(llvm::Module& module,
       const FunctionOptions& options,
       FunctionRegistry& functionRegistry,
       Logger& logger);

    ~PointComputeGenerator() override = default;

    using ComputeGenerator::traverse;
    using ComputeGenerator::visit;

    AttributeRegistry::Ptr generate(const ast::Tree& node);
    bool visit(const ast::Attribute*) override;

private:
    void computePKBR(const AttributeRegistry&);
    void computePKB(const AttributeRegistry&);
    void computePKAA(const AttributeRegistry&);
};

} // namespace namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_POINT_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

