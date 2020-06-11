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

/// @file codegen/FunctionRegistry.h
///
/// @authors Nick Avramoussis
///
/// @brief  Contains the global function registration definition which
///   described all available user front end functions
///

#ifndef OPENVDB_AX_CODEGEN_FUNCTION_REGISTRY_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_FUNCTION_REGISTRY_HAS_BEEN_INCLUDED

#include "FunctionTypes.h"
#include "Types.h"

#include <openvdb_ax/version.h>
#include <openvdb_ax/compiler/CompilerOptions.h>

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  The global function registry which is used for function code generation.
///         Each time a function is visited within the AST, its identifier is used
///         as a key into this registry for the corresponding function retrieval
///         and execution.
///
///         Functions can be inserted into the registry using insert() with a given
///         identifier and pointer to a function base.
///
class FunctionRegistry
{
public:

    using ConstructorT = FunctionGroup::Ptr(*)(const FunctionOptions&);
    using Ptr = std::shared_ptr<FunctionRegistry>;
    using UniquePtr = std::unique_ptr<FunctionRegistry>;

    /// @brief An object to represent a registered function, storing its constructor,
    ///        a pointer to the function definition and whether it should only be available
    //         internally (i.e. to a developer, not a user)
    ///
    struct RegisteredFunction
    {
        /// @brief Constructor
        /// @param creator The function definition used to create this function
        /// @param internal Whether the function should be only internally accessible
        ///
        RegisteredFunction(const ConstructorT& creator, const bool internal = false)
            : mConstructor(creator), mFunction(), mInternal(internal) {}

        /// @brief Create a function object using this creator of this function
        /// @param op The current function options
        ///
        inline void create(const FunctionOptions& op) { mFunction = mConstructor(op); }

        /// @brief Return a pointer to this function definition
        ///
        inline FunctionGroup::Ptr function() const { return mFunction; }

        /// @brief Check whether this function should be only internally accesible
        ///
        inline bool isInternal() const { return mInternal; }

    private:
        const ConstructorT mConstructor;
        FunctionGroup::Ptr mFunction;
        const bool mInternal;
    };

    using RegistryMap = std::unordered_map<std::string, RegisteredFunction>;

    /// @brief  Insert and register a function base object to a function identifier.
    /// @note   Throws if the identifier is already registered
    ///
    /// @param  identifier  The function identifier to register
    /// @param  creator     The function base to link to the provided identifier
    /// @param  internal    Whether to mark the function as only internally accessible
    ///
    void insert(const std::string& identifier, const ConstructorT creator, const bool internal = false);

    /// @brief  Insert and register a function base object to a function identifier.
    /// @note   Throws if the identifier is already registered
    ///
    /// @param  identifier  The function identifier to register
    /// @param  creator     The function base to link to the provided identifier
    /// @param  op          FunctionOptions to pass the function constructor
    /// @param  internal    Whether to mark the function as only internally accessible
    ///
    void insertAndCreate(const std::string& identifier,
                    const ConstructorT creator,
                    const FunctionOptions& op,
                    const bool internal = false);

    /// @brief  Return the corresponding function object from a provided function identifier
    /// @note   Returns a nullptr if no such function identifier has been registered or if the
    ///         function is marked as internal
    ///
    /// @param  identifier  The function identifier
    /// @param  op          FunctionOptions to pass the function constructor
    /// @param  allowInternalAccess    Whether to look in the 'internal' functions
    ///
    FunctionGroup::Ptr getOrInsert(const std::string& identifier,
                      const FunctionOptions& op,
                      const bool allowInternalAccess);

    /// @brief  Return the corresponding function object from a provided function identifier
    /// @note   Returns a nullptr if no such function identifier has been registered or if the
    ///         function is marked as internal
    ///
    /// @param  identifier  The function identifier
    /// @param  allowInternalAccess    Whether to look in the 'internal' functions
    ///
    FunctionGroup::Ptr get(const std::string& identifier,
                          const bool allowInternalAccess) const;

    /// @brief  Force the (re)creations of all function objects for all registered functions
    /// @param  op  The current function options
    /// @param  verify  Checks functions are created and have valid identifiers/symbols
    ///
    void createAll(const FunctionOptions& op, const bool verify = false);

    /// @brief  Return a const reference to the current registry map
    ///
    inline const RegistryMap& map() const { return mMap; }

    /// @brief  Return whether or not the registry is empty
    ///
    inline bool empty() const { return mMap.empty(); }

    /// @brief  Clear the underlying function registry
    ///
    inline void clear() { mMap.clear(); }

private:
    RegistryMap mMap;
};

}
}
}
}

#endif // OPENVDB_AX_CODEGEN_FUNCTION_REGISTRY_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
