// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file openvdb_ax/Exceptions.h
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  OpenVDB AX Exceptions
///

#ifndef OPENVDB_AX_EXCEPTIONS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_EXCEPTIONS_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Exceptions.h>

#include <sstream>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

#define OPENVDB_AX_EXCEPTION(_classname) \
class _classname: public Exception \
{ \
public: \
    _classname() noexcept: Exception( #_classname ) {} \
    explicit _classname(const std::string& msg) noexcept: Exception( #_classname , &msg) {} \
}

OPENVDB_AX_EXCEPTION(CLIError);

// @note: Compilation errors due to invalid AX code should be collected using a separate logging system.
//   These errors are only thrown upon encountering fatal errors within the compiler/executables themselves
OPENVDB_AX_EXCEPTION(AXTokenError);
OPENVDB_AX_EXCEPTION(AXSyntaxError);
OPENVDB_AX_EXCEPTION(AXCodeGenError);
OPENVDB_AX_EXCEPTION(AXCompilerError);
OPENVDB_AX_EXCEPTION(AXExecutionError);

#undef OPENVDB_AX_EXCEPTION

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_EXCEPTIONS_HAS_BEEN_INCLUDED

