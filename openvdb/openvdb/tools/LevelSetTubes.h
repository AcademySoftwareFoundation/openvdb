// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
///
/// @author Greg Hurst
///
/// @file LevelSetTubes.h
///
/// @brief Generate a narrow-band level set of a capsule, tapered capsule, and tube complex.
///
/// @note By definition a level set has a fixed narrow band width
/// (the half width is defined by LEVEL_SET_HALF_WIDTH in Types.h),
/// whereas an SDF can have a variable narrow band width.

#ifndef OPENVDB_TOOLS_LEVELSETTUBES_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETTUBES_HAS_BEEN_INCLUDED

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
/// representation of a capsule (tube with constant radius and sphere caps).
///
/// @param pt1    First capsule endpoint in world units.
/// @param pt2    Second capsule endpoint in world units.
/// @param radius    Radius of the capsule in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the capsule endpoint and radius type
/// and must be a floating-point scalar.
template <typename GridType, typename ScalarType, typename InterruptT>
typename GridType::Ptr
createLevelSetCapsule(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
    ScalarType radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr, bool threaded = true);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a capsule (tube with constant radius and sphere caps).
///
/// @param pt1    First capsule endpoint in world units.
/// @param pt2    Second capsule endpoint in world units.
/// @param radius    Radius of the capsule in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the capsule endpoint and radius type
/// and must be a floating-point scalar.
template <typename GridType, typename ScalarType>
typename GridType::Ptr
createLevelSetCapsule(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
    ScalarType radius, float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    bool threaded = true);


/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tapered capsule (tube with sphere caps and different radii at both ends,
/// or equivalently the convex hull of two spheres with possibly different centers and radii).
///
/// @param pt1    First tapered capsule endpoint in world units.
/// @param pt2    Second tapered capsule endpoint in world units.
/// @param radius1    Radius of the tapered capsule at @c pt1 in world units.
/// @param radius2    Radius of the tapered capsule at @c pt2 in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the tapered capsule endpoint and radius type
/// and must be a floating-point scalar.
template <typename GridType, typename ScalarType, typename InterruptT>
typename GridType::Ptr
createLevelSetTaperedCapsule(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
    ScalarType radius1, ScalarType radius2,
    float voxelSize, float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    InterruptT* interrupter = nullptr, bool threaded = true);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tapered capsule (tube with sphere caps and different radii at both ends,
/// or equivalently the convex hull of two spheres with possibly different centers and radii).
///
/// @param pt1    First tapered capsule endpoint in world units.
/// @param pt2    Second tapered capsule endpoint in world units.
/// @param radius1    Radius of the tapered capsule at @c pt1 in world units.
/// @param radius2    Radius of the tapered capsule at @c pt2 in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param threaded     If true multi-threading is enabled (true by default).
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the tapered capsule endpoint and radius type
/// and must be a floating-point scalar.
template <typename GridType, typename ScalarType>
typename GridType::Ptr
createLevelSetTaperedCapsule(const math::Vec3<ScalarType>& pt1, const math::Vec3<ScalarType>& pt2,
    ScalarType radius1, ScalarType radius2, float voxelSize,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH), bool threaded = true);

/// @brief Different policies when creating a tube complex with varying radii
/// @details
/// <dl>
/// <dt><b>TUBE_VERTEX_RADII</b>
/// <dd>Specify that the tube complex radii are per-vertex,
/// meaning each tube has different radii at its two endpoints
/// and the complex is a collection of tapered capsules.
///
/// <dt><b>TUBE_SEGMENT_RADII</b>
/// <dd>Specify that the tube complex radii are per-segment,
/// meaning each tube has a constant radius and the complex is a collection of capsules.
///
/// <dt><b>TUBE_AUTOMATIC</b>
/// <dd>Specify that the only valid setting is to be chosen,
/// defaulting to the per-vertex policy if both are valid.
/// </dl>
enum TubeRadiiPolicy { TUBE_AUTOMATIC = 0, TUBE_VERTEX_RADII, TUBE_SEGMENT_RADII };

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tube complex (a collection of capsules defined by endpoint coordinates and segment indices).
///
/// @param vertices    Endpoint vertices in the tube complex in world units.
/// @param segments    Segment indices in the tube complex.
/// @param radius    Radius of all tubes in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the capsule complex vertex and radius type
/// and must be a floating-point scalar.
template <typename GridType, typename ScalarType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetTubeComplex(const std::vector<math::Vec3<ScalarType>>& vertices,
    const std::vector<Vec2I>& segments, ScalarType radius, float voxelSize,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH), InterruptT* interrupter = nullptr);

/// @brief Return a grid of type @c GridType containing a narrow-band level set
/// representation of a tube complex (a collection of tubes defined by endpoint coordinates, segment indices, and radii).
///
/// @param vertices    Endpoint vertices in the tube complex in world units.
/// @param segments    Segment indices in the tube complex.
/// @param radii    Radii specification for all tubes in world units.
/// @param voxelSize    Voxel size in world units.
/// @param halfWidth    Half the width of the narrow band, in voxel units.
/// @param radii_policy    Policies: per-segment, per-vertex, or automatic (default).
/// @param interrupter    Interrupter adhering to the util::NullInterrupter interface.
///
/// @note @c GridType::ValueType must be a floating-point scalar.
/// @note @c ScalarType represents the capsule complex vertex and radius type
/// and must be a floating-point scalar.
/// @note The automatic @c TubeRadiiPolicy chooses the valid per-segment or per-vertex policy,
/// defaulting to per-vertex if both are valid.
template <typename GridType, typename ScalarType, typename InterruptT = util::NullInterrupter>
typename GridType::Ptr
createLevelSetTubeComplex(const std::vector<math::Vec3<ScalarType>>& vertices,
    const std::vector<Vec2I>& segments, const std::vector<ScalarType>& radii, float voxelSize,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH), TubeRadiiPolicy radii_policy = TUBE_AUTOMATIC,
    InterruptT* interrupter = nullptr);


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_LEVELSETTUBES
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetTubeComplex<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec2I>&, float, float, float, util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetTubeComplex<Grid<TreeT>>(const std::vector<Vec3s>&, \
        const std::vector<Vec2I>&, const std::vector<float>&, float, float, TubeRadiiPolicy, \
        util::NullInterrupter*)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetCapsule<Grid<TreeT>>(const Vec3s&, const Vec3s&, \
        float, float, float, util::NullInterrupter*, bool)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    Grid<TreeT>::Ptr createLevelSetTaperedCapsule<Grid<TreeT>>(const Vec3s&, const Vec3s&, \
        float, float, float, float, util::NullInterrupter*, bool)
OPENVDB_REAL_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/LevelSetTubesImpl.h"

#endif // OPENVDB_TOOLS_LEVELSETTUBES_HAS_BEEN_INCLUDED
