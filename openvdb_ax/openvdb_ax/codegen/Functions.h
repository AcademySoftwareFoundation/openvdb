// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/Functions.h
///
/// @authors Nick Avramoussis, Richard Jones, Francisco Gochez
///
/// @brief  Contains the function objects that define the functions used in
///   compute function generation, to be inserted into the FunctionRegistry.
///   These define general purpose functions such as math functions.
///

#ifndef OPENVDB_AX_CODEGEN_GENERIC_FUNCTIONS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_GENERIC_FUNCTIONS_HAS_BEEN_INCLUDED

#include "FunctionRegistry.h"

#include "../compiler/CompilerOptions.h"

#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief Creates a registry with the default set of registered functions
///        including math functions, point functions and volume functions
/// @param op The current function options
///
inline FunctionRegistry::UniquePtr createDefaultRegistry(const FunctionOptions* op = nullptr);

/// @brief Populates a function registry with all available "standard" AX
///        library function. This primarily consists of all mathematical ops
///        on AX containers (scalars, vectors, matrices) and other stl built-ins
/// @param reg The function registry to populate
/// @param options The current function options
///
OPENVDB_AX_API void insertStandardFunctions(FunctionRegistry& reg, const FunctionOptions* options = nullptr);

/// @brief Populates a function registry with all available OpenVDB Point AX
///        library function
/// @param reg The function registry to populate
/// @param options The current function options
///
OPENVDB_AX_API void insertVDBPointFunctions(FunctionRegistry& reg, const FunctionOptions* options = nullptr);

/// @brief Populates a function registry with all available OpenVDB Volume AX
///        library function
/// @param reg The function registry to populate
/// @param options The current function options
///
OPENVDB_AX_API void insertVDBVolumeFunctions(FunctionRegistry& reg, const FunctionOptions* options = nullptr);


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


inline FunctionRegistry::UniquePtr createDefaultRegistry(const FunctionOptions* op)
{
    FunctionRegistry::UniquePtr registry(new FunctionRegistry);
    insertStandardFunctions(*registry, op);
    insertVDBPointFunctions(*registry, op);
    insertVDBVolumeFunctions(*registry, op);
    return registry;
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_GENERIC_FUNCTIONS_HAS_BEEN_INCLUDED

