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

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  The function definition and signature which is built by the
///         VolumeComputeGenerator.
///
///         The argument structure is as follows:
///
///             1) - A void pointer to the CustomData
///             2) - A pointer to an array of three ints representing the
///                  current voxel coord being accessed
///             3) - An pointer to an array of three floats representing the
///                  current voxel world space coord being accessed
///             4) - A void pointer to a vector of void pointers, representing
///                  an array of grid accessors
///             5) - A void pointer to a vector of void pointers, representing
///                  an array of grid transforms
///
struct VolumeKernel
{
    // The signature of the generated function
    using Signature =
        void(const void* const,
             const int32_t (*)[3],
             const float (*)[3],
             void**,
             void**,
             int64_t,
             void*);

    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    static const std::array<std::string, N_ARGS>& argumentKeys();
    static std::string getDefaultName();
};


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

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
    /// @param warnings         Vector which will hold compiler warnings.  If null, no warnings will
    ///                         be stored.
    VolumeComputeGenerator(llvm::Module& module,
        const FunctionOptions& options,
        FunctionRegistry& functionRegistry,
        std::vector<std::string>* const warnings = nullptr);

    ~VolumeComputeGenerator() override = default;

    using ComputeGenerator::traverse;
    using ComputeGenerator::visit;

    AttributeRegistry::Ptr generate(const ast::Tree& node);
    bool visit(const ast::Attribute*) override;

private:

    void getAccessorValue(const std::string&, llvm::Value*);
    llvm::Value* accessorHandleFromToken(const std::string&);
};

}
}
}
}

#endif // OPENVDB_AX_VOLUME_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

