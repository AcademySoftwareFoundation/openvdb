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

#include <openvdb/Exceptions.h>

#include <sstream>
#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

#define OPENVDB_LLVM_EXCEPTION(_classname) \
class OPENVDB_API _classname: public Exception \
{ \
public: \
    _classname() noexcept: Exception( #_classname ) {} \
    explicit _classname(const std::string& msg) noexcept: Exception( #_classname , &msg) {} \
}

#define OPENVDB_AX_EXCEPTION(_classname) \
class OPENVDB_API _classname: public Exception \
{ \
public: \
    _classname() noexcept: Exception( #_classname ) {} \
    explicit _classname(const std::string& msg) noexcept: Exception( #_classname , &msg) {} \
}

/// @todo  The compiler has two levels of runtime errors - The first are internal errors
///        produced either from bad API usage, bad compilation or critical faults. The
///        second are user errors produced from incorrect or unsupported usage of the
///        language. We should be able to catch and determine the instance of the error
///        easily, most likely achieving this with another level of inheritance from
///        openvdb::Exception. We should also introduce naming conventions for these
///        exceptions.

// Runtime internal usage errors

OPENVDB_LLVM_EXCEPTION(LLVMContextError);
OPENVDB_LLVM_EXCEPTION(LLVMInitialisationError);
OPENVDB_LLVM_EXCEPTION(LLVMIRError);
OPENVDB_LLVM_EXCEPTION(LLVMModuleError);
OPENVDB_LLVM_EXCEPTION(LLVMTargetError);
OPENVDB_LLVM_EXCEPTION(LLVMTokenError);

// Runtime user errors

OPENVDB_LLVM_EXCEPTION(LLVMSyntaxError);
OPENVDB_LLVM_EXCEPTION(LLVMArrayError);
OPENVDB_LLVM_EXCEPTION(LLVMBinaryOperationError);
OPENVDB_LLVM_EXCEPTION(LLVMCastError);
OPENVDB_LLVM_EXCEPTION(LLVMLoopError);
OPENVDB_LLVM_EXCEPTION(LLVMKeywordError);
OPENVDB_LLVM_EXCEPTION(LLVMDeclarationError);
OPENVDB_LLVM_EXCEPTION(LLVMFunctionError);
OPENVDB_LLVM_EXCEPTION(LLVMTypeError);
OPENVDB_LLVM_EXCEPTION(LLVMUnaryOperationError);

// Runtime AX usage errors

OPENVDB_AX_EXCEPTION(AXCompilerError);
OPENVDB_AX_EXCEPTION(AXSyntaxError);

// Runtime AX execution errors

OPENVDB_AX_EXCEPTION(AXExecutionError);

#undef OPENVDB_LLVM_EXCEPTION
#undef OPENVDB_AX_EXCEPTION

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_EXCEPTIONS_HAS_BEEN_INCLUDED

