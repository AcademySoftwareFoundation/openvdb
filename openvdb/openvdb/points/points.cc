// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file points/points.cc

#include "PointDataGrid.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

void
internal::initialize()
{
    // Register attribute arrays with no compression
    TypedAttributeArray<bool>::registerType();
    TypedAttributeArray<int8_t>::registerType();
    TypedAttributeArray<int16_t>::registerType();
    TypedAttributeArray<int32_t>::registerType();
    TypedAttributeArray<int64_t>::registerType();
    TypedAttributeArray<float>::registerType();
    TypedAttributeArray<double>::registerType();
    TypedAttributeArray<math::Vec3<int32_t>>::registerType();
    TypedAttributeArray<math::Vec3<float>>::registerType();
    TypedAttributeArray<math::Vec3<double>>::registerType();

    // Register attribute arrays with group and string attribute
    GroupAttributeArray::registerType();
    StringAttributeArray::registerType();

    // Register attribute arrays with matrix and quaternion attributes
    TypedAttributeArray<math::Mat3<float>>::registerType();
    TypedAttributeArray<math::Mat3<double>>::registerType();
    TypedAttributeArray<math::Mat4<float>>::registerType();
    TypedAttributeArray<math::Mat4<double>>::registerType();
    TypedAttributeArray<math::Quat<float>>::registerType();
    TypedAttributeArray<math::Quat<double>>::registerType();

    // Register attribute arrays with truncate compression
    TypedAttributeArray<float, TruncateCodec>::registerType();
    TypedAttributeArray<math::Vec3<float>, TruncateCodec>::registerType();

    // Register attribute arrays with fixed point compression
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<true>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<false>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<true, PositionRange>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<false, PositionRange>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<true, UnitRange>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<false, UnitRange>>::registerType();

    // Register attribute arrays with unit vector compression
    TypedAttributeArray<math::Vec3<float>, UnitVecCodec>::registerType();

    // Register types associated with point data grids.
    Metadata::registerType(typeNameAsString<PointDataIndex32>(), Int32Metadata::createMetadata);
    Metadata::registerType(typeNameAsString<PointDataIndex64>(), Int64Metadata::createMetadata);

#define OPENVDB_TREE4(T, N1, N2, N3, LeafT) Grid<tree::Tree<tree::RootNode< \
    tree::InternalNode<tree::InternalNode<LeafT<T, N3>, N2>, N1>>>>::registerGrid();

    OPENVDB_TREE4_POINT_DATA_INITIALIZE()

#undef OPENVDB_TREE4
}


void
internal::uninitialize()
{
    AttributeArray::clearRegistry();
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
