// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/CustomData.h
///
/// @authors Nick Avramoussis, Francisco Gochez
///
/// @brief  Access to the CustomData class which can provide custom user
///   user data to the OpenVDB AX Compiler.
///

#ifndef OPENVDB_AX_COMPILER_CUSTOM_DATA_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_CUSTOM_DATA_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Metadata.h>
#include <openvdb/Types.h>

#include <unordered_map>
#include <memory>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

/// @brief The custom data class is a simple container for named openvdb
///   metadata. Its primary use case is passing arbitrary "external" data to an
///   AX executable object when calling Compiler::compile. For example, it is
///   the mechanism by which we pass data held inside of a parent DCC to
///   executable AX code.
class CustomData
{
public:

    using Ptr = std::shared_ptr<CustomData>;
    using ConstPtr = std::shared_ptr<const CustomData>;
    using UniquePtr = std::unique_ptr<CustomData>;

    CustomData() : mData() {}

    static UniquePtr create()
    {
        UniquePtr data(new CustomData);
        return data;
    }

    /// @brief Reset the custom data. This will clear and delete all previously
    ///   added data. This will invalidated any executable which links to this
    ///   custom data.
    inline void reset()
    {
        mData.clear();
    }

    /// @brief  Checks whether or not data of given name has been inserted
    inline bool
    hasData(const Name& name)
    {
        const auto iter = mData.find(name);
        return (iter != mData.end());
    }

    /// @brief  Checks whether or not data of given name and type has been inserted
    template <typename TypedDataCacheT>
    inline bool
    hasData(const Name& name)
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) return false;
        const TypedDataCacheT* const typed =
            dynamic_cast<const TypedDataCacheT* const>(iter->second.get());
        return typed != nullptr;
    }

    /// @brief  Retrieves a const pointer to data of given name.  If it does not
    ///         exist, returns nullptr
    inline const Metadata::ConstPtr
    getData(const Name& name) const
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) return Metadata::ConstPtr();
        return iter->second;
    }

    /// @brief Retrieves a const pointer to data of given name and type.
    ///   If it does not exist, returns nullptr
    /// @param   name Name of the data entry
    /// @returns The metadata. If the type does not match, nullptr is returned.
    template <typename TypedDataCacheT>
    inline const TypedDataCacheT*
    getData(const Name& name) const
    {
        Metadata::ConstPtr data = getData(name);
        if (!data) return nullptr;
        const TypedDataCacheT* const typed =
            dynamic_cast<const TypedDataCacheT* const>(data.get());
        return typed;
    }

    /// @brief  Retrieves or inserts typed metadata. If the data exists, it is
    ///   dynamic-casted to the expected type, which may result in a nullptr. If
    ///   the data does not exist it is guaranteed to be inserted and returned.
    ///   The value of the inserted data can then be modified
    template <typename TypedDataCacheT>
    inline TypedDataCacheT*
    getOrInsertData(const Name& name)
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) {
            Metadata::Ptr data(new TypedDataCacheT());
            mData[name] = data;
            return static_cast<TypedDataCacheT* const>(data.get());
        }
        else {
            return dynamic_cast<TypedDataCacheT* const>(iter->second.get());
        }
    }

    /// @brief  Inserts data of specified type with given name.
    /// @param  name Name of the data
    /// @param  data Shared pointer to the data
    /// @note   If an entry of the given name already exists, will copy the data
    ///   into the existing entry rather than overwriting the pointer
    template <typename TypedDataCacheT>
    inline void
    insertData(const Name& name,
               const typename TypedDataCacheT::Ptr data)
    {
        if (hasData(name)) {
            TypedDataCacheT* const dataToSet =
                getOrInsertData<TypedDataCacheT>(name);
            if (!dataToSet) {
                OPENVDB_THROW(TypeError, "Custom data \"" + name +
                    "\" already exists with a different type.");
            }
            dataToSet->value() = data->value();
        }
        else {
            mData[name] = data->copy();
        }
    }

    /// @brief  Inserts data with given name.
    /// @param  name Name of the data
    /// @param  data The metadata
    /// @note   If an entry of the given name already exists, will copy the data
    ///   into the existing entry rather than overwriting the pointer
    inline void
    insertData(const Name& name,
               const Metadata::Ptr data)
    {
        const auto iter = mData.find(name);
        if (iter == mData.end()) {
            mData[name] = data;
        }
        else {
            iter->second->copy(*data);
        }
    }

private:
    std::unordered_map<Name, Metadata::Ptr> mData;
};

// fwd declare the codegen::String and alias deprecated metadata type
namespace codegen { struct String; }
using AXStringMetadata [[deprecated("The ax::AXStringMetadata type has "
    "been replaced with openvdb::TypedMetadata<ax::codegen::String>. The "
    "new backend string definition can be found in ax/codegen/String.h")]] =
        TypedMetadata<ax::codegen::String>;

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_CUSTOM_DATA_HAS_BEEN_INCLUDED

