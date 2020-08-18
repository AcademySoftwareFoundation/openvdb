// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/FunctionRegistry.cc

#include "FunctionRegistry.h"
#include "Functions.h"
#include "FunctionTypes.h"

#include "../Exceptions.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

void FunctionRegistry::insert(const std::string& identifier,
       const FunctionRegistry::ConstructorT creator, const bool internal)
{
    if (!mMap.emplace(std::piecewise_construct,
              std::forward_as_tuple(identifier),
              std::forward_as_tuple(creator, internal)).second) {
        OPENVDB_THROW(LLVMFunctionError, "A function already exists"
            " with the provided identifier: \"" + identifier + "\"");
    }
}

void FunctionRegistry::insertAndCreate(const std::string& identifier,
                const FunctionRegistry::ConstructorT creator,
                const FunctionOptions& op,
                const bool internal)
{
    auto inserted = mMap.emplace(std::piecewise_construct,
              std::forward_as_tuple(identifier),
              std::forward_as_tuple(creator, internal));
    if (!inserted.second) {
        OPENVDB_THROW(LLVMFunctionError, "A function already exists"
            " with the provided token: \"" + identifier + "\"");
    }
    inserted.first->second.create(op);
}

FunctionGroup::Ptr FunctionRegistry::getOrInsert(const std::string& identifier,
                                                const FunctionOptions& op,
                                                const bool allowInternalAccess)
{
    auto iter = mMap.find(identifier);
    if (iter == mMap.end()) return FunctionGroup::Ptr();
    FunctionRegistry::RegisteredFunction& reg = iter->second;
    if (!allowInternalAccess && reg.isInternal()) return FunctionGroup::Ptr();

    if (!reg.function()) reg.create(op);

    FunctionGroup::Ptr function = reg.function();

    // initialize function dependencies if necessary

    if (op.mLazyFunctions && function) {
        for (const auto& decl : function->list()) {
            const std::vector<const char*>& deps = decl->dependencies();
            for (const auto& dep : deps) {
                // if the function ptr doesn't exist, create it with getOrInsert.
                // This provides internal access and ensures handling of cyclical
                // dependencies do not cause a problem
                FunctionGroup::Ptr internal = this->get(dep, true);
                if (!internal) this->getOrInsert(dep, op, true);
            }
        }
    }

    return function;
}

FunctionGroup::Ptr FunctionRegistry::get(const std::string& identifier, const bool allowInternalAccess) const
{
    auto iter = mMap.find(identifier);
    if (iter == mMap.end()) return FunctionGroup::Ptr();
    if (!allowInternalAccess && iter->second.isInternal()) return FunctionGroup::Ptr();
    return iter->second.function();
}

void FunctionRegistry::createAll(const FunctionOptions& op, const bool verify)
{
    for (auto& it : mMap) it.second.create(op);
    if (!verify) return;

    std::set<std::string> symbols;

    for (auto& it : mMap) {
        const auto& func = it.second.function();
        if (!func) {
            OPENVDB_LOG_WARN("Unable to create function '" << it.first << "'.");
        }
        if (it.first != std::string(func->name())) {
            OPENVDB_LOG_WARN("Registered function identifier does not match function name '" <<
                it.first << "' -> '" << func->name() << "'.");
        }
        if (it.first.empty() || !func->name()) {
            OPENVDB_LOG_WARN("Registered function has no identifier or name.");
        }
        if (func->list().empty()) {
            OPENVDB_LOG_WARN("Function '" << it.first << "' has no declarations.");
        }
        for (const auto& decl : func->list()) {
            if (symbols.count(std::string(decl->symbol()))) {
                OPENVDB_LOG_WARN("Function '" << it.first << "' has a symbol clash. Symbol '" <<
                   decl->symbol() << "' already exists.");
            }
            symbols.insert(std::string(decl->symbol()));
        }
    }
}

}
}
}
}

