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

#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/codegen/FunctionRegistry.h>

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
void insertStandardFunctions(FunctionRegistry& reg, const FunctionOptions* options = nullptr);

/// @brief Populates a function registry with all available OpenVDB Point AX
///        library function
/// @param reg The function registry to populate
/// @param options The current function options
///
void insertVDBPointFunctions(FunctionRegistry& reg, const FunctionOptions* options = nullptr);

/// @brief Populates a function registry with all available OpenVDB Volume AX
///        library function
/// @param reg The function registry to populate
/// @param options The current function options
///
void insertVDBVolumeFunctions(FunctionRegistry& reg, const FunctionOptions* options = nullptr);


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

}
}
}
}

#endif // OPENVDB_AX_CODEGEN_GENERIC_FUNCTIONS_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
