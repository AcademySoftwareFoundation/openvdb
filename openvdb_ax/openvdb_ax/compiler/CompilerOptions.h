// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/CompilerOptions.h
///
/// @authors Nick Avramoussis
///
/// @brief  OpenVDB AX Compiler Options
///

#ifndef OPENVDB_AX_COMPILER_COMPILER_OPTIONS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_COMPILER_OPTIONS_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

/// @brief Options that control how functions behave
struct OPENVDB_AX_API FunctionOptions
{
    /// @brief  Enable the constant folding of C bindings. Functions may use this setting
    ///         to determine whether they are allowed to be called during code generation
    ///         to evaluate call sites with purely constant arguments and replace the call
    ///         with the result.
    /// @note   This does not impact IR functions which we leave to LLVM's CF during
    ///         IR optimization.
    /// @note   We used to bind IR methods to corresponding C bindings, however it can be
    ///         very easy to implement incorrectly, leading to discrepancies in the CF
    ///         results. Fundamentally, LLVM's support for CF IR is far superior and our
    ///         framework only supports some types of folding (see codegen/ConstantFolding.h)
    bool mConstantFoldCBindings = true;
    /// @brief  When enabled, functions which have IR builder instruction definitions will
    ///         prioritise those over any registered external calls
    bool mPrioritiseIR = true;
    /// @brief  When enabled, the function registry is only populated on a function visit.
    ///         At the end of code generation, only functions which have been instantiated
    ///         will exist in the function map.
    bool mLazyFunctions = true;
};

/// @brief Settings which control how a Compiler class object behaves
struct OPENVDB_AX_API CompilerOptions
{
    /// @brief Controls the llvm compiler optimization level
    enum class OptLevel
    {
        NONE, // Do not run any optimization passes
        O0, // Optimization level 0. Similar to clang -O0
        O1, // Optimization level 1. Similar to clang -O1
        O2, // Optimization level 2. Similar to clang -O2
        Os, // Like -O2 with extra optimizations for size. Similar to clang -Os
        Oz, // Like -Os but reduces code size further. Similar to clang -Oz
        O3  // Optimization level 3. Similar to clang -O3
    };

    OptLevel mOptLevel = OptLevel::O3;

    /// @brief If this flag is true, the generated llvm module will be verified when compilation
    ///        occurs, resulting in an exception being thrown if it is not valid
    bool mVerify = true;
    /// @brief Options for the function registry
    FunctionOptions mFunctionOptions = FunctionOptions();
};

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_FUNCTION_REGISTRY_OPTIONS_HAS_BEEN_INCLUDED

