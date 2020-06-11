///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

/// @file Exceptions.h
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

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
