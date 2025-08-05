// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

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
#include "Types.h"
#include "Value.h"
#include "../compiler/AttributeRegistry.h"
#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {
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
    void getAccessorValue(const std::string&, Value);

    void computek2(llvm::Function*, const AttributeRegistry&);
    void computek3(llvm::Function*, const AttributeRegistry&);

    // Stores internal ptrs to accessor allocations
    SymbolTable<llvm::Value*> mInternalPtrs;
};

} // namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_VOLUME_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

