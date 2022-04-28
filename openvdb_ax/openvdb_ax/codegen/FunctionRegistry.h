// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

#include "openvdb_ax/compiler/CompilerOptions.h"

#include <openvdb/version.h>

#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  The function registry which is used for function code generation.
///   Each time a function is visited within the AST, its identifier is used as
///   a key into this registry for the corresponding function retrieval and
///   execution. Functions can be inserted into the registry using insert() with
///   a given identifier and pointer.
class OPENVDB_AX_API FunctionRegistry
{
public:
    using ConstructorT = FunctionGroup::UniquePtr(*)(const FunctionOptions&);
    using Ptr = std::shared_ptr<FunctionRegistry>;
    using UniquePtr = std::unique_ptr<FunctionRegistry>;

    /// @brief An object to represent a registered function, storing its
    ///   constructor, a pointer to the function definition and whether it
    ///   should only be available internally (i.e. to a developer, not a user)
    ///
    struct RegisteredFunction
    {
        /// @brief Constructor
        /// @param creator The function definition used to create this function
        /// @param internal Whether the function should be only internally accessible
        RegisteredFunction(const ConstructorT& creator, const bool internal = false)
            : mConstructor(creator), mFunction(), mInternal(internal) {}

        /// @brief Create a function object using this creator of this function
        /// @param op The current function options
        inline void create(const FunctionOptions& op) { mFunction = mConstructor(op); }

        /// @brief Return a pointer to this function definition
        inline const FunctionGroup* function() const { return mFunction.get(); }

        /// @brief Check whether this function should be only internally accesible
        inline bool isInternal() const { return mInternal; }

    private:
        ConstructorT mConstructor;
        FunctionGroup::Ptr mFunction;
        bool mInternal;
    };

    using RegistryMap = std::unordered_map<std::string, RegisteredFunction>;

    /// @brief Insert and register a function object to a function identifier.
    /// @note  Throws if the identifier is already registered
    ///
    /// @param identifier The function identifier to register
    /// @param creator    The function to link to the provided identifier
    /// @param internal   Whether to mark the function as only internally accessible
    void insert(const std::string& identifier,
        const ConstructorT creator,
        const bool internal = false);

    /// @brief Insert and register a function object to a function identifier.
    /// @note  Throws if the identifier is already registered
    ///
    /// @param identifier The function identifier to register
    /// @param creator    The function to link to the provided identifier
    /// @param op         FunctionOptions to pass the function constructor
    /// @param internal   Whether to mark the function as only internally accessible
    void insertAndCreate(const std::string& identifier,
                    const ConstructorT creator,
                    const FunctionOptions& op,
                    const bool internal = false);

    /// @brief Return the corresponding function from a provided function identifier
    /// @note  Returns a nullptr if no such function identifier has been
    ///   registered or if the function is marked as internal
    ///
    /// @param  identifier  The function identifier
    /// @param  op          FunctionOptions to pass the function constructor
    /// @param  allowInternalAccess  Whether to look in the 'internal' functions
    const FunctionGroup* getOrInsert(const std::string& identifier,
                      const FunctionOptions& op,
                      const bool allowInternalAccess);

    /// @brief Return the corresponding function from a provided function identifier
    /// @note  Returns a nullptr if no such function identifier has been
    ///   registered or if the function is marked as internal
    ///
    /// @param identifier  The function identifier
    /// @param allowInternalAccess  Whether to look in the 'internal' functions
    const FunctionGroup* get(const std::string& identifier,
                          const bool allowInternalAccess) const;

    /// @brief Force the (re)creations of all function objects for all
    ///   registered functions
    /// @param op  The current function options
    /// @param verify  Checks functions are created and have valid identifiers/symbols
    void createAll(const FunctionOptions& op, const bool verify = false);

    /// @brief  Return a const reference to the current registry map
    inline const RegistryMap& map() const { return mMap; }

    /// @brief  Return whether or not the registry is empty
    inline bool empty() const { return mMap.empty(); }

    /// @brief  Clear the underlying function registry
    inline void clear() { mMap.clear(); }

private:
    RegistryMap mMap;
};

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_FUNCTION_REGISTRY_HAS_BEEN_INCLUDED

