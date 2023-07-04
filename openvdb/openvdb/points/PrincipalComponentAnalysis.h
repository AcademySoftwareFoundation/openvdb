// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @author Richard Jones, Nick Avramoussis
///
/// @file PrincipalComponentAnalysis.h
///
/// @brief  Provides methods to perform principal component analysis (PCA) over
///   a point set to compute rotational and affine transformations for each
///   point that represent a their neighborhoods anisotropy. The techniques
///   and algorithms used here are described in:
///       [Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic
///        Kernel - Yu Turk 2010].
///   The parameters and results of these methods can be combines with the
///   ellipsoidal surfacing technique found in PointRasterizeSDF.h.

#ifndef OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Coord.h>
#include <openvdb/thread/Threading.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointTransfer.h>
#include <openvdb/points/PointDataGrid.h>

#include <string>
#include <vector>
#include <memory>
#include <cmath> // std::cbrt
#include <algorithm> // std::accumulate

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

struct PcaSettings;
struct PcaAttributes;

/// @brief  Calculate  Calculate ellipsoid transformations from the local point
///   distributions as described in Yu and Turk's 'Reconstructing Fluid Surfaces
///   with Anisotropic Kernels'.
/// @param points   The point data grid to analyses
/// @param settings The PCA settings for controlling the behavior of the
///   neighborhood searches and the resulting ellipsoidal values
/// @param attrs    The PCA attributes to create
/// @param interrupt An optional interrupter.
template <typename PointDataGridT,
    typename FilterT = NullFilter,
    typename InterrupterT = util::NullInterrupter>
inline void
pca(PointDataGridT& points,
    const PcaSettings& settings,
    const PcaAttributes& attrs,
    InterrupterT* interrupt = nullptr);


/// @brief  Various settings for the neighborhood analysis of point
///   distributions.
struct PcaSettings
{
    /// @param searchRadius  the world space search radius of the neighborhood
    ///   around each point. Increasing this value will result in points
    ///   including more of their neighbors into their ellipsoidal calculations.
    ///   This may or may not be desirable depending on the point set's
    ///   distribution and can be tweaked as necessary. Note however that large
    ///   values will cause the PCA calculation to become exponentially more
    ///   expensive. Values equal to or less than 0.0 have undefined results.
    float searchRadius = 1.0f;
    /// @param allowedAnisotropyRatio  the maximum allowed ratio between the
    ///   components in each ellipse' stretch coefficients such that:
    /// @code
    ///     const auto s = stretch.sorted();
    ///     assert(s[0]/s[2] >= allowedAnisotropyRatio);
    /// @endcode
    ///   This parameter effectively clamps the allowed anisotropy, with a
    ///   value of 1.0f resulting in uniform stretch values (representing a
    ///   sphere). Values tending towards zero will allow for greater
    ///   anisotropy i.e. much more exaggerated stretches along the computed
    ///   principal axis and corresponding squashes along the others to
    ///   compensate.
    /// @note Very small values may cause very thing ellipses to be produced,
    ///   so a reasonable minimum should be set. Values equal to or less than
    ///   0.0, or values greater than 1.0 have undefined results.
    float allowedAnisotropyRatio = 0.25f;
    /// @param neighbourThreshold  the number of neighbours a point must have
    ///   to be classified as having an elliptical distribution. Points with
    ///   less neighbours than this will end up with uniform stretch values of
    ///   1.0 and an identity rotation matrix.
    size_t neighbourThreshold = 20;
    /// @param averagePositions  the amount (between 0 and 1) to average out
    ///   positions. All points, whether they end up as ellipses or not,
    ///   can have their positions smoothed to account for their neighbourhood
    ///   distribution. This will have no effect for points with no neighbours.
    /// @warning  This options does NOT modify the P attribute - instead, a
    ///   world space position attribute "pws" (default) is created and stores
    ///   the smoothed position.
    float averagePositions = 1.0f;
};

/// @brief  The persistent attributes created by the PCA methods.
/// @note   These can be passed to points::rasterizeSdf with the
///   EllipsoidSettings to perform ellipsoidal surface construction.
struct PcaAttributes
{
    /// @brief  Settings for the "stretch" attribute, a floating point vector
    ///   attribute which represents the scaling components of each points
    ///   ellipse or (1.0,1.0,1.0) for isolated points.
    using StretchT = math::Vec3<float>;
    std::string stretch = "stretch";

    /// @brief  Settings for the "rotation" attribute, a floating point matrix
    ///   attribute which represents the rotation of each points ellipse or
    ///   the identity matrix for isolated points.
    using RotationT = math::Mat3<float>;
    std::string rotation = "rotation";

    /// @brief  Settings for the world space position of every point. This may
    ///   end up being different to their actual position if the
    ///   PcaSettings::averagePositions value is not exactly 0. This attribute
    ///   is used in place of the point's actual position when calling
    ///   points::rasterizeSdf.
    /// @note  This should always be at least at double precision
    using PosWsT = math::Vec3<double>;
    std::string positionWS = "pws";

    /// @brief A point group to create that represents points which have valid
    ///   ellipsoidal neighborhood. Points outside of this group will have
    ///   their stretch and rotation attributes set to describe a canonical
    ///   sphere. Note however that all points, regardless of this groups
    ///   membership flag, will still contribute to their neighbours and may
    ///   have their world space position deformed in relation to their
    ///   neighboring points.
    std::string ellipses = "ellipsoids";
};

}
}
}

#include "impl/PrincipalComponentAnalysisImpl.h"

#endif // OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_HAS_BEEN_INCLUDED
