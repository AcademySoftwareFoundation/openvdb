// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Nick Avramoussis
///
/// @file PointRasterizeSDF.h
///
/// @brief Various transfer schemes for rasterizing point positions and radius
///  data to signed distance fields with optional closest point attribute
///  transfers. All methods support arbitrary target linear transformations,
///  fixed or varying point radius, filtering of point data and arbitrary types
///  for attribute transferring.
///
/// @details There are currently three main transfer implementations:
///
/// - Rasterize Spheres
///
///     Performs trivial narrow band stamping of spheres for each point. This
///     is an extremely fast and efficient way to produce both a valid
///     symmetrical narrow band level set and transfer attributes using closest
///     point lookups.
///
/// - Rasterize Smooth Spheres
///
///     Calculates an averaged position of influence per voxel as described in:
///       [Animating Sand as a Fluid - Zhu Bridson 2005].
///
///     This technique produces smoother, more blended connections between
///     points which is ideal for generating a more artistically pleasant
///     surface directly from point distributions. It aims to avoid typical
///     post filtering operations used to smooth surface volumes. Note however
///     that this method may not necessarily produce a *symmetrical* narrow
///     band level set; the exterior band may be smaller than desired depending
///     on the search radius - the surface can be rebuilt or resized if
///     necessary. The same closet point algorithm is used to transfer
///     attributes.
///
/// - Rasterize Ellipsoids.
///
///     Rasterizes anisotropic ellipses for each point by analyzing point
///     neighborhood distributions, as described in:
///       [Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic
///        Kernel - Yu Turk 2010].
///
///     This method uses the rotation and affine matrix attributes built from
///     the points::pca() method which model these elliptical distributions
///     using principle component analysis. The ellipses create a much tighter,
///     more fitted surface that better represents the convex hull of the point
///     set. This technique also allows point to smoothly blend from their
///     computed ellipse back to a canonical sphere, as well as allowing
///     isolated points to be rasterized with their own radius scale. Although
///     the rasterization step of this pipeline is relatively fast, it is still
///     the slowest of all three methods and depends on the somewhat expensive
///     points::pca() method. Still, this technique can be far superior at
///     producing fluid surfaces where thin sheets (waterfalls) or sharp edges
///     (wave breaks) are desirable.
///
///
///  In general, it is recommended to consider post rebuilding/renormalizing
///  the generated surface using either tools::levelSetRebuild() or
///  tools::LevelSetTracker::normalize() tools::LevelSetTracker::resize().
///
/// @note These methods use the framework provided in PointTransfer.h

#ifndef OPENVDB_POINTS_RASTERIZE_SDF_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_RASTERIZE_SDF_HAS_BEEN_INCLUDED

#include "PointDataGrid.h"
#include "PointTransfer.h"
#include "PointStatistics.h"

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/points/PrincipleComponentAnalysis.h>
#include <openvdb/thread/Threading.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/util/Simd.h>

#include <unordered_map>

#include <tbb/task_group.h>
#include <tbb/parallel_reduce.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @brief  Perform point rasterzation to produce a signed distance field.
/// @param point     the point data grid to rasterize
/// @param settings  one of the available transfer setting schemes found below
///   in this file.
/// @return A vector of grids. The signed distance field is guaranteed to be
///   first and at the type specified by SdfT. Successive grids are the closest
///   point attribute grids. These grids are guaranteed to have a topology
///   and transform equal to the surface.
///
/// @code
///    points::PointDataGrid g = ...;
///
///    // default settings for sphere stamping with a world space radius of 1
///    SphereSettings<> spheres;
///    FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(points::rasterizeSdf(g, spheres)[0]);
///
///    // custom linear transform of target sdf, world space radius of 5
///    spheres.transform = math::Transform::createLinearTransform(0.3);
///    spheres.radiusScale = 5;
///    FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(points::rasterizeSdf(g, spheres)[0]);
///
///    // smooth sphere rasterization with variable double precision radius
///    // attribute "pscale" scaled by 2
///    SmoothSphereSettings<TypeList<>, double> smooth;
///    smooth.radius = "pscale";
///    smooth.radiusScale = 2;
///    smooth.searchRadius = 3;
///    FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(points::rasterizeSdf(g, smooth)[0]);
///
///    // anisotropic/ellipsoid rasterization with attribute transferring.
///    // requires pca attributes to be initialized using points::pca() first
///    PcaSettings settings;
///    PcaAttributes attribs;
///    points::pca(g, settings, attribs);
///
///    EllipsoidSettings<TypeList<int32_t, Vec3f>> ellips;
///    s.pca = attribs;
///    s.attributes.emplace_back("id");
///    s.attributes.emplace_back("v");
///    GridPtrVec grids = points::rasterizeSdf(g, s);
///    FloatGrid::Ptr sdf = StaticPtrCast<FloatGrid>(grids[0]);
///    Int32Grid::Ptr id  = StaticPtrCast<Int32Grid>(grids[1]);
///    Vec3fGrid::Ptr vel = StaticPtrCast<Vec3fGrid>(grids[2]);
/// @endcode
template <typename PointDataGridT,
    typename SdfT = typename PointDataGridT::template ValueConverter<float>::Type,
    typename SettingsT>
GridPtrVec
rasterizeSdf(const PointDataGridT& points, const SettingsT& settings);

//

/// @brief  Generic settings for narrow band spherical stamping with a uniform
///   or varying radius and optionally with closest point attribute transfer of
///   arbitrary attributes. See the struct member documentation for detailed
///   behavior.
/// @note  There exists other more complex kernels that derive from this struct,
///   but on its own it represents the settings needed to perform basic narrow
///   band sphere stamping. Parameters are interpreted in the same way across
///   derived classes.
template <typename AttributeTs = TypeList<>,
    typename RadiusAttributeT = float,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
struct SphereSettings
{
    using AttributeTypes = AttributeTs;
    using RadiusAttributeType = RadiusAttributeT;
    using FilterType = FilterT;
    using InterrupterType = InterrupterT;

    /// @param radius  the attribute containing the world space radius
    /// @details  if the radius parameter is an empty string then the
    ///   `radiusScale` parameter is used as a uniform world space radius to
    ///   generate a fixed surface mask. Otherwise, a point attribute
    ///   representing the world space radius of each point of type
    ///   `RadiusAttributeT` is expected to exist and radii are scaled by the
    ///   `radiusScale` parameter.
    std::string radius = "";

    /// @param radiusScale  the scale applied to every world space radius value
    /// @note  If no `radius` attribute is provided, this is used as the
    ///   uniform world space radius for every point. Most surfacing operations
    ///   will perform faster if they are able to assume a uniform radius (so
    ///   use this value instead of setting the `radius` parameter if radii are
    ///   uniform).
    Real radiusScale = 1.0;

    /// @param halfband  the half band width of the generated surface.
    Real halfband = LEVEL_SET_HALF_WIDTH;

    /// @param transform  the target transform for the surface. Most surfacing
    ///   operations impose linear restrictions on the target transform.
    math::Transform::Ptr transform = nullptr;

    /// @param attributes   list of attributes to transfer
    /// @details  if the attributes vector is empty, only the surface is built.
    ///   Otherwise, every voxel's closest point is used to transfer each
    ///   attribute in the attributes parameter to a new grid of matching
    ///   topology. The built surface is always the first grid returned from
    ///   the surfacing operation, followed by attribute grids in the order
    ///   that they appear in this vector.
    ///
    ///   The `AttributeTs` template parameter should be a `TypeList` of the
    ///   required or possible attributes types. Example:
    /// @code
    ///   // compile support for int, double and Vec3f attribute transferring
    ///   using SupportedTypes = TypeList<int, double, Vec3f>;
    ///   SphereSettings<SupportedTypes> s;
    ///
    ///   // Produce 4 additional grids from the "v", "Cd", "id" and "density"
    ///   // attributes. Their attribute value types must be available in the
    ///   // provided TypeList
    ///   s.attributes = {"v", "Cd", "id", "density"};
    /// @endcode
    ///
    ///   A runtime error will be thrown if no equivalent type for a given
    ///   attribute is found in the `AttributeTs` TypeList.
    ///
    /// @note The destination types of these grids is equal to the
    ///   `ValueConverter` result of the attribute type applied to the
    ///   PointDataGridT.
    std::vector<std::string> attributes;

    /// @param filter  a filter to apply to points. Only points that evaluate
    ///   to true using this filter are rasterized, regardless of any other
    ///   filtering derived schemes may use.
    const FilterT* filter = nullptr;

    /// @param interrupter  optional interrupter
    InterrupterT* interrupter = nullptr;
};

/// @brief Smoothed point distribution based sphere stamping with a uniform radius
///   or varying radius and optionally with closest point attribute transfer of
///   arbitrary attributes. See the struct member documentation for detailed
///   behavior.
/// @note  Protected inheritance prevents accidental struct slicing
template <typename AttributeTs = TypeList<>,
    typename RadiusAttributeT = float,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
struct SmoothSphereSettings
    : protected SphereSettings<AttributeTs, RadiusAttributeT, FilterT, InterrupterT>
{
    using BaseT = SphereSettings<AttributeTs, RadiusAttributeT, FilterT, InterrupterT>;
    using AttributeTypes = typename BaseT::AttributeTypes;
    using RadiusAttributeType = typename BaseT::RadiusAttributeType;
    using FilterType = typename BaseT::FilterType;
    using InterrupterType = typename BaseT::InterrupterType;

    using BaseT::radius;
    /// @note  See also the searchRadius parameter for SmoothSpehere
    ///   rasterization.
    using BaseT::radiusScale;
    /// @warning The width of the exterior half band *may* be smaller than the
    ///   specified half band if the search radius is less than the equivalent
    ///   world space halfband distance.
    using BaseT::halfband;
    using BaseT::transform;
    using BaseT::attributes;
    using BaseT::filter;
    using BaseT::interrupter;

    /// @param searchRadius  the maximum search distance of every point
    /// @details  The search radius is each points points maximum contribution
    ///   to the target level set. It should always have a value equal to or
    ///   larger than the point radius. Both this and the `radiusScale`
    ///   parameters are given in world space units and are applied to every
    ///   point to generate a surface mask.
    Real searchRadius = 1.0;
};

/// @brief  Anisotropic point rasterization based on the principle component
///   analysis of point neighbours. See the struct member documentation for
///   detailed behavior.
/// @note  Protected inheritance prevents accidental struct slicing
template <typename AttributeTs = TypeList<>,
    typename RadiusAttributeT = float,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
struct EllipsoidSettings
    : protected SphereSettings<AttributeTs, RadiusAttributeT, FilterT, InterrupterT>
{
    using BaseT = SphereSettings<AttributeTs, RadiusAttributeT, FilterT, InterrupterT>;
    using AttributeTypes = typename BaseT::AttributeTypes;
    using RadiusAttributeType = typename BaseT::RadiusAttributeType;
    using FilterType = typename BaseT::FilterType;
    using InterrupterType = typename BaseT::InterrupterType;

    using BaseT::radius;
    using BaseT::radiusScale;
    using BaseT::halfband;
    using BaseT::transform;
    using BaseT::attributes;
    using BaseT::filter;
    using BaseT::interrupter;

    /// @brief  The sphere scale. Points which are not in the inclusion group
    ///   specified by the pca attributes have their world space radius scaled
    ///   by this amount. Typically you'd want this value to be <= 1.0 to
    ///   produce smaller spheres for isolated points.
    Real sphereScale = 1.0;

    /// @brief  The required principle component analysis attributes which are
    ///   required to exist on the points being rasterized. These attributes
    ///   define the rotational and affine transformations which can be used to
    ///   construct ellipsoids for each point. Typically (for our intended
    ///   surfacing) these transformations are built by analysing each points
    ///   neighbourhood distributions and constructing tight ellipsoids that
    ///   orient themselves to follow these point distributions.
    PcaAttributes pca;
};

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#include "impl/PointRasterizeSDFImpl.h"
#include "impl/PointRasterizeEllipsoidsSDFImpl.h"

#endif //OPENVDB_POINTS_RASTERIZE_SDF_HAS_BEEN_INCLUDED
