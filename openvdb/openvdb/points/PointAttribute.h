// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Dan Bailey, Khang Ngo
///
/// @file points/PointAttribute.h
///
/// @brief  Point attribute manipulation in a VDB Point Grid.

#ifndef OPENVDB_POINTS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/util/Assert.h>

#include "AttributeArrayString.h"
#include "AttributeSet.h"
#include "AttributeGroup.h"
#include "PointDataGrid.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

namespace point_attribute_internal {

template <typename ValueType>
struct Default
{
    static inline ValueType value() { return zeroVal<ValueType>(); }
};

} // namespace point_attribute_internal


/// @brief Appends a new attribute to the VDB tree
/// (this method does not require a templated AttributeType)
///
/// @param tree               the PointDataTree to be appended to.
/// @param name               name for the new attribute.
/// @param type               the type of the attibute.
/// @param strideOrTotalSize  the stride of the attribute
/// @param constantStride     if @c false, stride is interpreted as total size of the array
/// @param defaultValue       metadata default attribute value
/// @param hidden             mark attribute as hidden
/// @param transient          mark attribute as transient
template <typename PointDataTreeT>
inline void appendAttribute(PointDataTreeT& tree,
                            const Name& name,
                            const NamePair& type,
                            const Index strideOrTotalSize = 1,
                            const bool constantStride = true,
                            const Metadata* defaultValue = nullptr,
                            const bool hidden = false,
                            const bool transient = false);

/// @brief Appends a new attribute to the VDB tree.
///
/// @param tree               the PointDataTree to be appended to.
/// @param name               name for the new attribute
/// @param uniformValue       the initial value of the attribute
/// @param strideOrTotalSize  the stride of the attribute
/// @param constantStride     if @c false, stride is interpreted as total size of the array
/// @param defaultValue       metadata default attribute value
/// @param hidden             mark attribute as hidden
/// @param transient          mark attribute as transient
template <typename ValueType,
          typename CodecType = NullCodec,
          typename PointDataTreeT>
inline void appendAttribute(PointDataTreeT& tree,
                            const std::string& name,
                            const ValueType& uniformValue =
                                point_attribute_internal::Default<ValueType>::value(),
                            const Index strideOrTotalSize = 1,
                            const bool constantStride = true,
                            const TypedMetadata<ValueType>* defaultValue = nullptr,
                            const bool hidden = false,
                            const bool transient = false);

/// @brief Collapse the attribute into a uniform value
///
/// @param tree         the PointDataTree in which to collapse the attribute.
/// @param name         name for the attribute.
/// @param uniformValue value of the attribute
template <typename ValueType, typename PointDataTreeT>
inline void collapseAttribute(  PointDataTreeT& tree,
                                const Name& name,
                                const ValueType& uniformValue =
                                    point_attribute_internal::Default<ValueType>::value());

/// @brief Drops attributes from the VDB tree.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param indices       indices of the attributes to drop.
template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<size_t>& indices);

/// @brief Drops attributes from the VDB tree.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param names         names of the attributes to drop.
template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<Name>& names);

/// @brief Drop one attribute from the VDB tree (convenience method).
///
/// @param tree          the PointDataTree to be dropped from.
/// @param index         index of the attribute to drop.
template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const size_t& index);

/// @brief Drop one attribute from the VDB tree (convenience method).
///
/// @param tree          the PointDataTree to be dropped from.
/// @param name          name of the attribute to drop.
template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const Name& name);

/// @brief Rename attributes in a VDB tree.
///
/// @param tree          the PointDataTree.
/// @param oldNames      a list of old attribute names to rename from.
/// @param newNames      a list of new attribute names to rename to.
///
/// @note Number of oldNames must match the number of newNames.
///
/// @note Duplicate names and renaming group attributes are not allowed.
template <typename PointDataTreeT>
inline void renameAttributes(PointDataTreeT& tree,
                            const std::vector<Name>& oldNames,
                            const std::vector<Name>& newNames);

/// @brief Rename an attribute in a VDB tree.
///
/// @param tree          the PointDataTree.
/// @param oldName       the old attribute name to rename from.
/// @param newName       the new attribute name to rename to.
///
/// @note newName must not already exist and must not be a group attribute.
template <typename PointDataTreeT>
inline void renameAttribute(PointDataTreeT& tree,
                            const Name& oldName,
                            const Name& newName);

/// @brief Compact attributes in a VDB tree (if possible).
///
/// @param tree          the PointDataTree.
template <typename PointDataTreeT>
inline void compactAttributes(PointDataTreeT& tree);


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointAttributeImpl.h"

#endif // OPENVDB_POINTS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED
