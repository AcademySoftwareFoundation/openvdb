// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_OPENVDB_HAS_BEEN_INCLUDED
#define OPENVDB_OPENVDB_HAS_BEEN_INCLUDED

#include "Platform.h"
#include "Types.h"
#include "TypeList.h"
#include "Metadata.h"
#include "math/Maps.h"
#include "math/Transform.h"
#include "Grid.h"
#include "tree/Tree.h"
#include "points/PointDataGrid.h"
#include "io/File.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

/// @brief  Global registration of native Grid, Transform, Metadata and Point
///   attribute types. Also initializes blosc (if enabled).
/// @details  Calling this methods registers factory callbacks for the set of
///   native grid, transform, metadata and point attribute types that OpenVDB
///   supports by default. For most types, calling openvdb::initialize is only
///   required for serialization support. However, openvdb::initialize must be
///   called for PointDataGrid attribute usage as these callbacks are used in
///   various tools.
/// @note  This method is thread safe - it can be concurrently called multiple
///   times, early exiting if it has already been called so long as
///   openvdb::uninitialize() has not been called.
OPENVDB_API void initialize();

/// @brief  Global deregistration of native Grid, Transform, Metadata and Point
///   attribute types.
/// @details  Clears all registered factory callbacks. This includes everything
///   registered by openvdb::initialize() but will also include any manually
///   registered types.
/// @note  This method is thread safe - it can be concurrently called multiple
///   times.
/// @warning  This method does *not* uninitialize blosc. This is to avoid
///   changing program states should another library also be using blosc. If
///   blosc is enabled, consider calling blosc_destroy() in your application.
OPENVDB_API void uninitialize();


// foward declare some default types
namespace io { class DelayedLoadMetadata; }

/// Common tree types
using BoolTree     = tree::Tree4<bool,        5, 4, 3>::Type;
using DoubleTree   = tree::Tree4<double,      5, 4, 3>::Type;
using FloatTree    = tree::Tree4<float,       5, 4, 3>::Type;
using Int32Tree    = tree::Tree4<int32_t,     5, 4, 3>::Type;
using Int64Tree    = tree::Tree4<int64_t,     5, 4, 3>::Type;
using MaskTree     = tree::Tree4<ValueMask,   5, 4, 3>::Type;
using UInt32Tree   = tree::Tree4<uint32_t,    5, 4, 3>::Type;
using Vec2DTree    = tree::Tree4<Vec2d,       5, 4, 3>::Type;
using Vec2ITree    = tree::Tree4<Vec2i,       5, 4, 3>::Type;
using Vec2STree    = tree::Tree4<Vec2s,       5, 4, 3>::Type;
using Vec3DTree    = tree::Tree4<Vec3d,       5, 4, 3>::Type;
using Vec3ITree    = tree::Tree4<Vec3i,       5, 4, 3>::Type;
using Vec3STree    = tree::Tree4<Vec3f,       5, 4, 3>::Type;
using ScalarTree   = FloatTree;
using TopologyTree = MaskTree;
using Vec3dTree    = Vec3DTree;
using Vec3fTree    = Vec3STree;
using VectorTree   = Vec3fTree;

/// Common grid types
using BoolGrid     = Grid<BoolTree>;
using DoubleGrid   = Grid<DoubleTree>;
using FloatGrid    = Grid<FloatTree>;
using Int32Grid    = Grid<Int32Tree>;
using Int64Grid    = Grid<Int64Tree>;
using MaskGrid     = Grid<MaskTree>;
using Vec3DGrid    = Grid<Vec3DTree>;
using Vec3IGrid    = Grid<Vec3ITree>;
using Vec3SGrid    = Grid<Vec3STree>;
using ScalarGrid   = FloatGrid;
using TopologyGrid = MaskGrid;
using Vec3dGrid    = Vec3DGrid;
using Vec3fGrid    = Vec3SGrid;
using VectorGrid   = Vec3fGrid;

/// @name Lists of native Grid Types
/// @{
/// The floating point Grid types which OpenVDB will register by default.
using RealGridTypes   = TypeList<FloatGrid, DoubleGrid>;
/// The integer Grid types which OpenVDB will register by default.
using IntegerGridTypes = TypeList<Int32Grid, Int64Grid>;
/// The scalar Grid types which OpenVDB will register by default. This is a
/// combination of native floating point and integer grid types. Note that
/// this list does not include Bool or Mask Grids.
using NumericGridTypes  = RealGridTypes::Append<IntegerGridTypes>;
/// The Vec3 Grid types which OpenVDB will register by default.
using Vec3GridTypes     = TypeList<Vec3IGrid, Vec3SGrid, Vec3DGrid>;

/// The Grid types which OpenVDB will register by default.
using GridTypes =
    NumericGridTypes::
        Append<Vec3GridTypes>::
        Append<tools::PointIndexGrid>::
    // #define unfortunately required for one of the tests that removes this alias
#ifndef OPENVDB_DISABLE_POINT_DATA_TREE_ALIAS
        Append<points::PointDataGrid>::
#endif
        Append<BoolGrid, MaskGrid>;
/// @}


namespace internal {
template <typename T> using ToTreeType = typename T::TreeType;
}
/// @name Lists of native Tree Types
/// @{
using RealTreeTypes    = RealGridTypes::Transform<internal::ToTreeType>;
using IntegerTreeTypes = IntegerGridTypes::Transform<internal::ToTreeType>;
using NumericTreeTypes = NumericGridTypes::Transform<internal::ToTreeType>;
using Vec3TreeTypes    = Vec3GridTypes::Transform<internal::ToTreeType>;
using TreeTypes        = GridTypes::Transform<internal::ToTreeType>;
/// @}


/// @name Lists of native TypedAttributeArray Types (for PointDataGrids)
/// @{
/// The floating point attribute array types which OpenVDB will register by default.
using RealAttributeTypes = TypeList<
    points::TypedAttributeArray<float>,
    points::TypedAttributeArray<double>,
    points::TypedAttributeArray<float, points::TruncateCodec>,
    points::TypedAttributeArray<float, points::FixedPointCodec<true, points::UnitRange>>,
    points::TypedAttributeArray<float, points::FixedPointCodec<false, points::UnitRange>>
    >;
/// The integer attribute array types which OpenVDB will register by default.
using IntegerAttributeTypes = TypeList<
    points::TypedAttributeArray<int8_t>,
    points::TypedAttributeArray<int16_t>,
    points::TypedAttributeArray<int32_t>,
    points::TypedAttributeArray<int64_t>
    >;
/// The scalar attribute array types which OpenVDB will register by default.
/// This is a combination of native floating point and integer array types.
/// Note that this list does not include bool arrays.
using NumericAttributeTypes =
    RealAttributeTypes::Append<IntegerAttributeTypes>;
/// The Vec3 attribute array types which OpenVDB will register by default.
using Vec3AttributeTypes = TypeList<
    points::TypedAttributeArray<math::Vec3<int32_t>>,
    points::TypedAttributeArray<math::Vec3<float>>,
    points::TypedAttributeArray<math::Vec3<double>>,
    points::TypedAttributeArray<math::Vec3<float>, points::TruncateCodec>,
    points::TypedAttributeArray<math::Vec3<float>, points::FixedPointCodec<true, points::PositionRange>>,
    points::TypedAttributeArray<math::Vec3<float>, points::FixedPointCodec<false, points::PositionRange>>,
    points::TypedAttributeArray<math::Vec3<float>, points::FixedPointCodec<true, points::UnitRange>>,
    points::TypedAttributeArray<math::Vec3<float>, points::FixedPointCodec<false, points::UnitRange>>,
    points::TypedAttributeArray<math::Vec3<float>, points::UnitVecCodec>
    >;
/// The Mat3 attribute array types which OpenVDB will register by default.
using Mat3AttributeTypes = TypeList<
    points::TypedAttributeArray<math::Mat3<float>>,
    points::TypedAttributeArray<math::Mat3<double>>
    >;
/// The Mat4 attribute array types which OpenVDB will register by default.
using Mat4AttributeTypes = TypeList<
    points::TypedAttributeArray<math::Mat4<float>>,
    points::TypedAttributeArray<math::Mat4<double>>
    >;
/// The Quat attribute array types which OpenVDB will register by default.
using QuatAttributeTypes = TypeList<
    points::TypedAttributeArray<math::Quat<float>>,
    points::TypedAttributeArray<math::Quat<double>>
    >;

/// The attribute array types which OpenVDB will register by default.
using AttributeTypes =
    NumericAttributeTypes::
    Append<Vec3AttributeTypes>::
    Append<Mat3AttributeTypes>::
    Append<Mat4AttributeTypes>::
    Append<QuatAttributeTypes>::
    Append<points::GroupAttributeArray>::
    Append<points::StringAttributeArray>::
    Append<points::TypedAttributeArray<bool>>;
/// @}


/// The Map types which OpenVDB will register by default.
using MapTypes = TypeList<
    math::AffineMap,
    math::UnitaryMap,
    math::ScaleMap,
    math::UniformScaleMap,
    math::TranslationMap,
    math::ScaleTranslateMap,
    math::UniformScaleTranslateMap,
    math::NonlinearFrustumMap>;


/// The Metadata types which OpenVDB will register by default.
using MetaTypes = TypeList<
    BoolMetadata,
    DoubleMetadata,
    FloatMetadata,
    Int32Metadata,
    Int64Metadata,
    StringMetadata,
    Vec2IMetadata,
    Vec2SMetadata,
    Vec2DMetadata,
    Vec3IMetadata,
    Vec3SMetadata,
    Vec3DMetadata,
    Vec4IMetadata,
    Vec4SMetadata,
    Vec4DMetadata,
    Mat4SMetadata,
    Mat4DMetadata,
    io::DelayedLoadMetadata>;


} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_OPENVDB_HAS_BEEN_INCLUDED
