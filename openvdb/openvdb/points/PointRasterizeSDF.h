// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Nick Avramoussis
///
/// @file PointRasterizeSDF.h
///
/// @brief Transfer schemes for rasterizing point positional and radius data to
///  signed distance fields with optional closest point attribute transfers.
///  All methods support arbitrary target linear transformations, fixed or
///  varying point radius, filtering of point data and arbitrary types for
///  attribute transferring.
///
/// @details There are two main transfer implementations; rasterizeSpheres and
///  rasterizeSmoothSpheres. The prior performs trivial narrow band stamping
///  of spheres for each point, where as the latter calculates an averaged
///  position of influence per voxel as described in:
///    [Animating Sand as a Fluid - Zhu Bridson 05].
///
///  rasterizeSpheres() is an extremely fast and efficient way to produce both a
///  valid symmetrical narrow band level set and transfer attributes using
///  closest point lookups.
///
///  rasterizeSmoothSpheres() produces smoother, more blended connections
///  between points which is ideal for generating a more artistically pleasant
///  surface directly from point distributions. It aims to avoid typical post
///  filtering operations used to smooth surface volumes. Note however that
///  rasterizeSmoothSpheres may not necessarily produce a *symmetrical* narrow
///  band level set; the exterior band may be smaller than desired depending on
///  the search radius. The surface can be rebuilt or resized if necessary.
///  The same closet point algorithm is used to transfer attributes.
///
///  In general, it is recommended to consider post rebuilding/renormalizing the
///  generated surface using either tools::levelSetRebuild() or
///  tools::LevelSetTracker::normalize() tools::LevelSetTracker::resize().
///
/// @note These methods use the framework provided in PointTransfer.h
///

#ifndef OPENVDB_POINTS_RASTERIZE_SDF_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_RASTERIZE_SDF_HAS_BEEN_INCLUDED

#include "PointDataGrid.h"
#include "PointTransfer.h"
#include "PointStatistics.h"

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/thread/Threading.h>
#include <openvdb/util/NullInterrupter.h>

#include <unordered_map>

#include <tbb/task_group.h>
#include <tbb/parallel_reduce.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief Narrow band sphere stamping with a uniform radius.
/// @details Rasterizes points into a level set using basic sphere stamping with
///   a uniform radius. The radius parameter is given in world space units and
///   is applied to every point to generate a fixed surface mask and consequent
///   distance values.
/// @param points       the point data grid to rasterize
/// @param radius       the world space radius of every point
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return The signed distance field.
template <typename PointDataGridT,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
typename SdfT::Ptr
rasterizeSpheres(const PointDataGridT& points,
             const Real radius,
             const Real halfband = LEVEL_SET_HALF_WIDTH,
             math::Transform::Ptr transform = nullptr,
             const FilterT& filter = NullFilter(),
             InterrupterT* interrupter = nullptr);

/// @brief Narrow band sphere stamping with a varying radius.
/// @details Rasterizes points into a level set using basic sphere stamping with
///   a variable radius. The radius string parameter expects a point attribute
///   of type RadiusT to exist.
/// @param points       the point data grid to rasterize
/// @param radius       the name of the radius attribute
/// @param scale        an optional scale to apply to each per point radius
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return The signed distance field.
template <typename PointDataGridT,
    typename RadiusT = float,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
typename SdfT::Ptr
rasterizeSpheres(const PointDataGridT& points,
             const std::string& radius,
             const Real scale = 1.0,
             const Real halfband = LEVEL_SET_HALF_WIDTH,
             math::Transform::Ptr transform = nullptr,
             const FilterT& filter = NullFilter(),
             InterrupterT* interrupter = nullptr);

/// @brief Narrow band sphere stamping with a uniform radius and closest point
///   attribute transfer.
/// @details Rasterizes points into a level set using basic sphere stamping with
///   a uniform radius. The radius parameter is given in world space units and
///   is applied to every point to generate a fixed surface mask and consequent
///   distance values. Every voxel's closest point is used to transfer each
///   attribute in the attributes parameter to a new grid of matching topology.
///   The destination types of these grids is equal to the ValueConverter result
///   of the attribute type applied to the PointDataGridT.
/// @note The AttributeTypes template parameter should be a TypeList of the
///   required or possible attributes types. i.e. TypeList<int, float, double>.
///   A runtime error will be thrown if no equivalent type for a given attribute
////  is found in the AttributeTypes TypeList.
/// @param points       the point data grid to rasterize
/// @param radius       the world space radius of every point
/// @param attributes   list of attributes to transfer
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return A vector of grids. The signed distance field is guaranteed to be
///   first and at the type specified by SdfT. Successive grids are the closest
///   point attribute grids. These grids are guaranteed to have a topology
///   and transform equal to the surface.
template <typename PointDataGridT,
    typename AttributeTypes,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
GridPtrVec
rasterizeSpheres(const PointDataGridT& points,
             const Real radius,
             const std::vector<std::string>& attributes,
             const Real halfband = LEVEL_SET_HALF_WIDTH,
             math::Transform::Ptr transform = nullptr,
             const FilterT& filter = NullFilter(),
             InterrupterT* interrupter = nullptr);

/// @brief Narrow band sphere stamping with a varying radius and closest point
///   attribute transfer.
/// @details Rasterizes points into a level set using basic sphere stamping with
///   a variable radius. The radius string parameter expects a point attribute
///   of type RadiusT to exist. Every voxel's closest point is used to transfer
///   each attribute in the attributes parameter to a new grid of matching
///   topology. The destination types of these grids is equal to the
///   ValueConverter result of the attribute type applied to the PointDataGridT.
/// @note The AttributeTypes template parameter should be a TypeList of the
///   required or possible attributes types. i.e. TypeList<int, float, double>.
///   A runtime error will be thrown if no equivalent type for a given attribute
////  is found in the AttributeTypes TypeList.
/// @param points       the point data grid to rasterize
/// @param radius       the name of the radius attribute
/// @param attributes   list of attributes to transfer
/// @param scale        scale to apply to each per point radius
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return A vector of grids. The signed distance field is guaranteed to be
///   first and at the type specified by SdfT. Successive grids are the closest
///   point attribute grids. These grids are guaranteed to have a topology
///   and transform equal to the surface.
template <typename PointDataGridT,
    typename AttributeTypes,
    typename RadiusT = float,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
GridPtrVec
rasterizeSpheres(const PointDataGridT& points,
             const std::string& radius,
             const std::vector<std::string>& attributes,
             const Real scale = 1.0,
             const Real halfband = LEVEL_SET_HALF_WIDTH,
             math::Transform::Ptr transform = nullptr,
             const FilterT& filter = NullFilter(),
             InterrupterT* interrupter = nullptr);

/// @brief Smoothed point distribution based sphere stamping with a uniform radius.
/// @details Rasterizes points into a level set using [Zhu Bridson 05] sphere
///   stamping with a uniform radius. The radius and search radius parameters
///   are given in world space units and are applied to every point to generate
///   a fixed surface mask and consequent distance values. The search radius is
///   each points points maximum contribution to the target level set. The search
///   radius should always have a value equal to or larger than the point radius.
/// @warning The width of the exterior half band *may* be smaller than the
///   specified half band if the search radius is less than the equivalent
///   world space halfband distance.
/// @param points       the point data grid to rasterize
/// @param radius       the world space radius of every point
/// @param searchRadius the maximum search distance of every point
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return The signed distance field.
template <typename PointDataGridT,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
typename SdfT::Ptr
rasterizeSmoothSpheres(const PointDataGridT& points,
             const Real radius,
             const Real searchRadius,
             const Real halfband = LEVEL_SET_HALF_WIDTH,
             math::Transform::Ptr transform = nullptr,
             const FilterT& filter = NullFilter(),
             InterrupterT* interrupter = nullptr);

/// @brief Smoothed point distribution based sphere stamping with a varying radius.
/// @details Rasterizes points into a level set using [Zhu Bridson 05] sphere
///   stamping with a variable radius. The radius string parameter expects a
///   point attribute of type RadiusT to exist. The radiusScale parameter is
///   multiplier for radius values held on the radius attribute. The searchRadius
///   parameter remains a fixed size value which represents each points points
///   maximum contribution to the target level set. The radius scale and search
///   radius parameters are given in world space units and are applied to every
///   point to generate a fixed surface mask and consequent distance values. The
///   search radius should always have a value equal to or larger than the point
///   radii.
/// @warning The width of the exterior half band *may* be smaller than the
///   specified half band if the search radius is less than the equivalent
///   world space halfband distance.
/// @param points       the point data grid to rasterize
/// @param radius       the attribute containing the world space radius
/// @param radiusScale  the scale applied to every world space radius value
/// @param searchRadius the maximum search distance of every point
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return The signed distance field.
template <typename PointDataGridT,
    typename RadiusT = float,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
typename SdfT::Ptr
rasterizeSmoothSpheres(const PointDataGridT& points,
                 const std::string& radius,
                 const Real radiusScale,
                 const Real searchRadius,
                 const Real halfband = LEVEL_SET_HALF_WIDTH,
                 math::Transform::Ptr transform = nullptr,
                 const FilterT& filter = NullFilter(),
                 InterrupterT* interrupter = nullptr);

/// @brief Smoothed point distribution based sphere stamping with a uniform
///   radius and closest point attribute transfer.
/// @details Rasterizes points into a level set using [Zhu Bridson 05] sphere
///   stamping with a uniform radius. The radius and search radius parameters
///   are given in world space units and are applied to every point to generate
///   a fixed surface mask and consequent distance values. The search radius is
///   each points points maximum contribution to the target level set. The
///   search radius should always be larger than the point radius. Every voxel's
///   closest point is used to transfer each attribute in the attributes
///   parameter to a new grid of matching topology. The destination types of
///   these grids is equal to the ValueConverter result of the attribute type
///   applied to the PointDataGridT.
/// @note The AttributeTypes template parameter should be a TypeList of the
///   required or possible attributes types. i.e. TypeList<int, float, double>.
///   A runtime error will be thrown if no equivalent type for a given attribute
///  is found in the AttributeTypes TypeList.
/// @warning The width of the exterior half band *may* be smaller than the
///   specified half band if the search radius is less than the equivalent
///   world space halfband distance.
/// @param points       the point data grid to rasterize
/// @param radius       the world space radius of every point
/// @param searchRadius the maximum search distance of every point
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return A vector of grids. The signed distance field is guaranteed to be
///   first and at the type specified by SdfT. Successive grids are the closest
///   point attribute grids. These grids are guaranteed to have a topology
///   and transform equal to the surface.
template <typename PointDataGridT,
    typename AttributeTypes,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
GridPtrVec
rasterizeSmoothSpheres(const PointDataGridT& points,
                 const Real radius,
                 const Real searchRadius,
                 const std::vector<std::string>& attributes,
                 const Real halfband = LEVEL_SET_HALF_WIDTH,
                 math::Transform::Ptr transform = nullptr,
                 const FilterT& filter = NullFilter(),
                 InterrupterT* interrupter = nullptr);

/// @brief Smoothed point distribution based sphere stamping with a varying
///   radius and closest point attribute transfer.
/// @details Rasterizes points into a level set using [Zhu Bridson 05] sphere
///   stamping with a variable radius. The radius string parameter expects a
///   point attribute of type RadiusT to exist. The radiusScale parameter is
///   multiplier for radius values held on the radius attribute. The searchRadius
///   parameter remains a fixed size value which represents each points points
///   maximum contribution to the target level set. The radius scale and search
///   radius parameters are given in world space units and are applied to every
///   point to generate a fixed surface mask and consequent distance values. The
///   search radius should always have a value equal to or larger than the point
///   radii. Every voxel's closest point is used to transfer each attribute in
///   the attributes parameter to a new grid of matching topology. The
///   destination types of these grids is equal to the ValueConverter result of
///   the attribute type applied to the PointDataGridT.
/// @note The AttributeTypes template parameter should be a TypeList of the
///   required or possible attributes types. i.e. TypeList<int, float, double>.
///   A runtime error will be thrown if no equivalent type for a given attribute
////  is found in the AttributeTypes TypeList.
/// @warning The width of the exterior half band *may* be smaller than the
///   specified half band if the search radius is less than the equivalent
///   world space halfband distance.
/// @param points       the point data grid to rasterize
/// @param radius       the attribute containing the world space radius
/// @param radiusScale  the scale applied to every world space radius value
/// @param searchRadius the maximum search distance of every point
/// @param halfband     the half band width
/// @param transform    the target transform for the surface
/// @param filter       a filter to apply to points
/// @param interrupter  optional interrupter
/// @return A vector of grids. The signed distance field is guaranteed to be
///   first and at the type specified by SdfT. Successive grids are the closest
///   point attribute grids. These grids are guaranteed to have a topology
///   and transform equal to the surface.
template <typename PointDataGridT,
    typename AttributeTypes,
    typename RadiusT = float,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
GridPtrVec
rasterizeSmoothSpheres(const PointDataGridT& points,
                 const std::string& radius,
                 const Real radiusScale,
                 const Real searchRadius,
                 const std::vector<std::string>& attributes,
                 const Real halfband = LEVEL_SET_HALF_WIDTH,
                 math::Transform::Ptr transform = nullptr,
                 const FilterT& filter = NullFilter(),
                 InterrupterT* interrupter = nullptr);

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointRasterizeSDFImpl.h"

#endif //OPENVDB_POINTS_RASTERIZE_SDF_HAS_BEEN_INCLUDED
