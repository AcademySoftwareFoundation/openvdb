// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0/

/// @file compiler/AttributeBindings.h
///
/// @authors Richard Jones
///
/// @brief The Attribute Bindings class is used by the compiled Executables
///   to handle the mapping of AX Attribute names to context dependent data
///   names, i.e point attribute and volume grid names.
///

#ifndef OPENVDB_AX_COMPILER_ATTRIBUTE_BINDINGS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_ATTRIBUTE_BINDINGS_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/util/Name.h>

#include <map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

/// @brief This class wraps an interface for a map of attribute bindings. These map
///   attributes in AX code to context data. These mappings are one-to-one i.e.
///   each AX name can only map to one data name, however each name can appear as either
///   an AX name or data name or both, i.e. the following sets of bindings are valid:
///   axname: a   ->  dataname: a
///   axname: b   ->  dataname: c
///   or
///   axname: a   ->  dataname: b
///   axname: b   ->  dataname: a
class AttributeBindings
{
public:

    AttributeBindings() = default;

    /// @brief Construct a set of attribute bindings from a vector of {ax name, data name} pairs
    /// @param bindings A vector of ax name data name pairs where the first element is the name
    ///        in the AX code, and the second is the name in the context data
    AttributeBindings(const std::vector<std::pair<std::string, std::string>>& bindings)
    {
        set(bindings);
    }

    /// @brief Construct a set of attribute bindings from a vector of {ax name, data name} pairs
    ///        using an initializer list i.e. {{"axname0", "dataname0"}, {"axname1", "dataname0"}}
    /// @param bindings A initializer list of ax name data name pairs where the first element is the
    ///        name in the AX code, and the second is the name in the context data
    AttributeBindings(const std::initializer_list<std::pair<std::string, std::string>>& bindings)
    {
        set(bindings);
    }

    bool operator==(const AttributeBindings& other) const
    {
        return mAXToDataMap == other.mAXToDataMap &&
            mDataToAXMap == other.mDataToAXMap;
    }

    /// @brief Set up a binding. If a data binding exists for this AX name, it will be replaced.
    ///     If another binding exists for the supplied dataname that will be removed.
    /// @param axname The name of the attribute in AX
    /// @param dataname The name of the attribute in the context data
    inline void set(const std::string& axname, const std::string& dataname)
    {
        auto axToData = mAXToDataMap.find(axname);
        if (axToData != mAXToDataMap.end()) {
            // the dataname is already mapped, so update it
            // and remove corresponding map entry in opposite direction
            auto dataToAX = mDataToAXMap.find(axToData->second);
            if (dataToAX != mDataToAXMap.end()) {
                mAXToDataMap.erase(dataToAX->second);
                mDataToAXMap.erase(dataToAX->first);
            }
        }
        auto dataToAX = mDataToAXMap.find(dataname);
        if (dataToAX != mDataToAXMap.end()) {
            mAXToDataMap.erase(dataToAX->second);
        }

        mAXToDataMap[axname] = dataname;
        mDataToAXMap[dataname] = axname;
    }

    /// @brief Set up multiple bindings from a vector of {ax name, data name} pairs.
    ///     If a data binding exists for any AX name, it will be replaced.
    ///     If another binding exists for the supplied dataname that will be removed.
    /// @param bindings Vector of AX name data name pairs
    inline void set(const std::vector<std::pair<std::string, std::string>>& bindings) {
        for (const auto& binding : bindings) {
            this->set(binding.first, binding.second);
        }
    }
    /// @brief Returns a pointer to the data attribute name string that the input AX attribute name
    ///   is bound to, or nullptr if unbound.
    /// @param axname The name of the attribute in AX
    inline const std::string* dataNameBoundTo(const std::string& axname) const
    {
        const auto iter = mAXToDataMap.find(axname);
        if (iter != mAXToDataMap.cend()) {
            return &iter->second;
        }
        return nullptr;
    }

    /// @brief Returns a pointer to the AX attribute name string that a data attribute name
    ///    is bound to, or nullptr if unbound.
    /// @param name The name of the attribute in the context data
    inline const std::string* axNameBoundTo(const std::string& name) const
    {
        const auto iter = mDataToAXMap.find(name);
        if (iter != mDataToAXMap.cend()) {
            return &iter->second;
        }
        return nullptr;
    }

    /// @brief Returns whether the data attribute has been bound to an AX attribute
    /// @param name The name of the attribute in the context data
    inline bool isBoundDataName(const std::string& name) const
    {
        return mDataToAXMap.count(name);
    }

    /// @brief Returns whether the AX attribute has been bound to a data attribute
    /// @param name The name of the attribute in AX
    inline bool isBoundAXName(const std::string& name) const
    {
        return mAXToDataMap.count(name);
    }

    /// @brief Returns the map of AX attribute names to data attribute names
    inline const std::map<std::string, std::string>& axToDataMap() const {
        return mAXToDataMap;
    }

    /// @brief Returns the map of data attribute names to AX attribute names
    inline const std::map<std::string, std::string>& dataToAXMap() const {
        return mDataToAXMap;
    }

private:

    std::map<std::string, std::string> mAXToDataMap;
    std::map<std::string, std::string> mDataToAXMap;
};


inline std::ostream& operator<<(std::ostream& os, const AttributeBindings& bindings)
{
    os << "ax->data map:\n";
    for (const auto& m : bindings.axToDataMap()) {
        os << "  [" << m.first << " -> " << m.second << ']' << '\n';
    }
    os << "data->ax map:\n";
    for (const auto& m : bindings.dataToAXMap()) {
        os << "  [" << m.first << " -> " << m.second  << ']' << '\n';
    }
    return os;
}

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_ATTRIBUTE_BINDINGS_HAS_BEEN_INCLUDED

