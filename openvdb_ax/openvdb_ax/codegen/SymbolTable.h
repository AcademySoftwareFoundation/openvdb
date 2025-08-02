// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/SymbolTable.h
///
/// @authors Nick Avramoussis
///
/// @brief  Contains the symbol table which holds mappings of variables names
///   to llvm::Values.
///

#ifndef OPENVDB_AX_CODEGEN_SYMBOL_TABLE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_SYMBOL_TABLE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include "Value.h"

#include <string>
#include <map>
#include <unordered_map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  A symbol table which can be used to represent a single scoped set of
///         a programs variables. This is simply an unordered map of strings to
///         llvm::Values
/// @note   Consider using llvm's ValueSymbolTable
///
template <typename T>
struct SymbolTable
{
    using MapType = std::unordered_map<std::string, T>;

    SymbolTable() : mMap() {}
    ~SymbolTable() = default;

    /// @brief  Get a Value from this symbol table with the given name
    ///         mapping. It it does not exist, a nullptr is returned.
    /// @param  name  The name of the variable
    ///
    inline const T* get(const std::string& name) const
    {
        const auto iter = mMap.find(name);
        if (iter == mMap.end()) return nullptr;
        return &(iter->second);
    }

    /// @brief  Returns true if a variable exists in this symbol table with the
    ///         given name.
    /// @param  name  The name of the variable
    ///
    inline bool exists(const std::string& name) const
    {
        const auto iter = mMap.find(name);
        return (iter != mMap.end());
    }

    /// @brief  Insert a variable to this symbol table if it does not exist. Returns
    ///         true if successfully, false if a variable already exists with the
    ///         given name.
    /// @param  name   The name of the variable
    /// @param  value  The llvm::Value corresponding to this variable
    ///
    inline bool insert(const std::string& name, const T& value)
    {
        const auto iter = mMap.emplace(name, value);
        return iter.second; // true if inserted, false otherwise
    }

    /// @brief  Replace a variable in this symbol table. Returns true if the variable
    ///         previously existed and false if not. In both cases, the variable is
    ///         inserted.
    /// @param  name   The name of the variable
    /// @param  value  The llvm::Value corresponding to this variable
    ///
    inline bool replace(const std::string& name, const T& value)
    {
        auto iter = mMap.emplace(name, value);
        const bool inserted = iter.second;
        if (!inserted) iter.first->second = value;
        return !inserted;
    }

    /// @brief  Clear all symbols in this table
    ///
    inline void clear() { mMap.clear(); }

    /// @brief  Access to the underlying map
    ///
    inline const MapType& map() const { return mMap; }

private:
    MapType mMap;
};


/// @brief  A map of unique ids to symbol tables which can be used to represent local
///         variables within a program. New scopes can be added and erased where necessary
///         and iterated through using find(). Find assumes that tables are added through
///         parented ascending ids.
///
/// @note   The zero id is used to represent the top block scope, but globals are stored
///         separately.
/// @note   The block symbol table is fairly simple and currently only supports insertion
///         by integer ids. Scopes that exist at the same level are expected to be built
///         in isolation and erase and re-create the desired ids where necessary.
///
struct SymbolTableBlocks
{
    using MapType = std::map<size_t, SymbolTable<Value>>;

    SymbolTableBlocks()
        : mTables({{0, SymbolTable<Value>()}})
        , mGlobals() {}
    ~SymbolTableBlocks() = default;

    /// @brief  Access to the list of global variables which are always accessible
    ///
    inline SymbolTable<llvm::Value*>& globals() { return mGlobals; }
    inline const SymbolTable<llvm::Value*>& globals() const { return mGlobals; }

    /// @brief  Erase a given scoped indexed SymbolTable from the list of held
    ///         SymbolTables. Returns true if the table previously existed.
    ///
    /// @param  index  The SymbolTable index to erase
    ///
    inline bool erase(const size_t index)
    {
        const bool existed = (mTables.find(index) != mTables.end());
        mTables.erase(index);
        return existed;
    }

    /// @brief  Get or insert and get a SymbolTable with a unique index
    ///
    /// @param  index  The SymbolTable index
    ///
    inline SymbolTable<Value>* getOrInsert(const size_t index)
    {
        return &(mTables[index]);
    }

    /// @brief  Get a SymbolTable with a unique index. If it doesn't exist, nullptr is returned
    ///
    /// @param  index  The SymbolTable index
    ///
    inline SymbolTable<Value>* get(const size_t index)
    {
        auto iter = mTables.find(index);
        if (iter == mTables.end()) return nullptr;
        return &(iter->second);
    }

    /// @brief  Find a variable within the program starting at a given table index. If
    ///         the given index does not exist, the next descending index is used.
    /// @note   This function assumes that tables have been added in ascending order
    ///         dictating their nested structure.
    ///
    /// @param  name        The variable name to find
    /// @param  startIndex  The start SymbolTable index
    ///
    inline const Value* find(const std::string& name, const size_t startIndex) const
    {
        // Find the lower bound start index and if necessary, decrement into
        // the first block where the search will be started. Note that this
        // is safe as the top scope block 0 will always exist

        auto it = mTables.lower_bound(startIndex);
        if (it == mTables.end() || it->first != startIndex) --it;

        // reverse the iterator (which also make it point to the preceding
        // value, hence the crement)

        OPENVDB_ASSERT(it != mTables.end());
        MapType::const_reverse_iterator iter(++it);

        for (; iter != mTables.crend(); ++iter) {
            const Value* value = iter->second.get(name);
            if (value) return value;
        }

        return nullptr;
    }

    /// @brief  Find a variable within the program starting at the lowest level
    ///         SymbolTable
    ///
    /// @param  name  The variable name to find
    ///
    inline const Value* find(const std::string& name) const
    {
        return this->find(name, mTables.crbegin()->first);
    }

    /// @brief  Replace the first occurrance of a variable with a given name with a
    ///         replacement value. Returns true if a replacement occurred.
    ///
    /// @param  name   The variable name to find and replace
    /// @param  value  The llvm::Value to replace
    ///
    inline bool replace(const std::string& name, Value value)
    {
        for (auto it = mTables.rbegin(); it != mTables.rend(); ++it) {
            if (it->second.get(name)) {
                it->second.replace(name, value);
                return true;
            }
        }
        return false;
    }

private:
    MapType mTables;
    SymbolTable<llvm::Value*> mGlobals;
};

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_SYMBOL_TABLE_HAS_BEEN_INCLUDED

