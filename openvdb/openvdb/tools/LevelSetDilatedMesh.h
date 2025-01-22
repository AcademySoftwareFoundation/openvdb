// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
///
/// @author Greg Hurst
///
/// @file LevelSetDilatedMesh.h
///
/// @brief Generate a narrow-band level set of a dilated surface mesh.
///
/// @note By definition a level set has a fixed narrow band width
/// (the half width is defined by LEVEL_SET_HALF_WIDTH in Types.h),
/// whereas an SDF can have a variable narrow band width.

#ifndef OPENVDB_TOOLS_LEVELSETDILATEDMESH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETDILATEDMESH_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/NullInterrupter.h>

#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a dilated triangle surface mesh (dilated by a radius in all directions).
///
/// @param vertices    Vertices of the mesh in world units.
/// @param triangles    Triangle indices of the mesh.
/// @param radius    Dilation radius in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the mesh vertex and radius type
/// and must be a floating-point scalar.
/// @note The input mesh is always treated as a surface, and so dilation occurs in every direction.
/// This includes meshes that could represent valid BRep solids, dilation occurs both
/// inward and outward, forming a 'shell' rather than only expanding outward.
template <typename GridType, typename ScalarType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetDilatedMesh(
    const std::vector<math::Vec3<ScalarType>>& vertices, const std::vector<Vec3I>& triangles,
    ScalarType radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a dilated quad surface mesh (dilated by a radius in all directions).
///
/// @param vertices    Vertices of the mesh in world units.
/// @param quads    Quad indices of the mesh.
/// @param radius    Dilation radius in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the mesh vertex and radius type
/// and must be a floating-point scalar.
/// @note The input mesh is always treated as a surface, and so dilation occurs in every direction.
/// This includes meshes that could represent valid BRep solids, dilation occurs both
/// inward and outward, forming a 'shell' rather than only expanding outward.
template <typename GridType, typename ScalarType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetDilatedMesh(
    const std::vector<math::Vec3<ScalarType>>& vertices, const std::vector<Vec4I>& quads,
    ScalarType radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a dilated triangle & quad  surface mesh (dilated by a radius in all directions).
///
/// @param vertices    Vertices of the mesh in world units.
/// @param triangles    Triangle indices of the mesh.
/// @param quads    Quad indices of the mesh.
/// @param radius    Dilation radius in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the mesh vertex and radius type
/// and must be a floating-point scalar.
/// @note The input mesh is always treated as a surface, and so dilation occurs in every direction.
/// This includes meshes that could represent valid BRep solids, dilation occurs both
/// inward and outward, forming a 'shell' rather than only expanding outward.
template <typename GridType, typename ScalarType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetDilatedMesh(const std::vector<math::Vec3<ScalarType>>& vertices,
    const std::vector<Vec3I>& triangles, const std::vector<Vec4I>& quads,
    ScalarType radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr);


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_LEVELSETDILATEDMESH
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetDilatedMesh<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec3I>&, float, float, float, util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetDilatedMesh<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec4I>&, float, float, float, util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetDilatedMesh<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec3I>&, const std::vector<Vec4I>&, float, float, float, \
        util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/LevelSetDilatedMeshImpl.h"

#endif // OPENVDB_TOOLS_LEVELSETDILATEDMESH_HAS_BEEN_INCLUDED
