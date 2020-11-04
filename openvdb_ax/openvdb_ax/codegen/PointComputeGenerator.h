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

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  The function definition and signature which is built by the
///         PointComputeGenerator.
///
///         The argument structure is as follows:
///
///           1) - A void pointer to the CustomData
///           2) - A void pointer to the leaf AttributeSet
///           3) - An unsigned integer, representing the leaf relative point
///                id being executed
///           4) - A void pointer to a vector of void pointers, representing an
///                array of attribute handles
///           5) - A void pointer to a vector of void pointers, representing an
///                array of group handles
///           6) - A void pointer to a LeafLocalData object, used to track newly
///                initialized attributes and arrays
///
struct PointKernel
{
    /// The signature of the generated function
    using Signature =
        void(const void* const,
             const void* const,
             uint64_t,
             void**,
             void**,
             void*);

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    /// The argument key names available during code generation
    static const std::array<std::string, N_ARGS>& argumentKeys();
    static std::string getDefaultName();
};

/// @brief  An additonal function built by the PointComputeGenerator.
///         Currently both compute and compute range functions have the same
///         signature
struct PointRangeKernel : public PointKernel
{
    static std::string getDefaultName();
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
    llvm::Value* attributeHandleFromToken(const std::string&);
    void getAttributeValue(const std::string& globalName, llvm::Value* location);
};

} // namespace namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_POINT_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

