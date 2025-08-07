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
#include <openvdb/util/Assert.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointTransfer.h>
#include <openvdb/points/PointDataGrid.h>

#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <cmath> // std::cbrt

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

struct PcaSettings;
struct PcaAttributes;

/// @brief  Calculate ellipsoid transformations from the local point
///   distributions as described in Yu and Turk's 'Reconstructing Fluid Surfaces
///   with Anisotropic Kernels'. The results are stored on the attributes
///   pointed to by the PcaAttributes. See the PcaSettings and PcaAttributes
///   structs for more details.
/// @warning  This method will throw if the 'strech', 'rotation' or 'pws'
///   attributes already exist on this input point set.
/// @param points   The point data grid to analyses
/// @param settings The PCA settings for controlling the behavior of the
///   neighborhood searches and the resulting ellipsoidal values
/// @param attrs    The PCA attributes to create
template <typename PointDataGridT>
inline void
pca(PointDataGridT& points,
    const PcaSettings& settings,
    const PcaAttributes& attrs);


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
    ///   expensive and should be used in conjunction with the max point per
    ///   voxel settings below.
    /// @warning  Valid range is [0, inf). Behaviour is undefined when outside
    ///   this range.
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
    ///   so a reasonable minimum should be set. Valid range is (0, 1].
    ///   Behaviour is undefined when outside this range.
    float allowedAnisotropyRatio = 0.25f;
    /// @param nonAnisotropicStretch  The stretch coefficient that should be
    ///   used for points which have no anisotropic neighbourhood (due to
    ///   being isolated or not having enough neighbours to reach the
    ///   specified @sa neighbourThreshold).
    float nonAnisotropicStretch = 1.0;
    /// @param neighbourThreshold  number of points in a given neighbourhood
    ///   that target points must have to be classified as having an elliptical
    ///   distribution. Points with less neighbours than this will end up with
    ///   uniform stretch values of nonAnisotropicStretch and an identity
    ///   rotation matrix.
    /// @note  This number can include the target point itself.
    /// @warning  Changing this value does not change the size or number of
    ///   point neighborhood lookups. As such, increasing this value only
    ///   speeds up the number of calculated covariance matrices. Valid range
    ///   is [0, inf], where 0 effectively disables all covariance calculations
    ///   and 1 enables them for every point.
    size_t neighbourThreshold = 20;
    /// @param The maximum number of source points to gather for contributions
    ///   to each target point, per voxel. When a voxel contains more points
    ///   than this value, source point are trivially stepped over, with the
    ///   step size calculated as max(1, ppv / neighbourThreshold).
    /// @note  There is no prioritisation of nearest points; for example, voxels
    ///   which partially overlap the search radius may end up selecting point
    ///   ids which all lie outside. This is rarely an issue in practice and
    ///   choosing an appropriate value for this setting can significantly
    ///   increase performance without a large impact to the results.
    /// @warning  Valid range is [1, inf). Behaviour is undefined if this value
    ///   is set to zero.
    size_t maxSourcePointsPerVoxel = 8;
    /// @param The maximum number of target points to write anisotropy values
    ///   to. When a voxel contains more points than this value, target point
    ///   are trivially stepped over, with the step size calculated as:
    ///      max(1, ppv / maxTargetPointsPerVoxel).
    ///   default behaviour is to compute for all points. Any points skipped
    ///   will be treated as being isolated and receive an identity rotation
    ///   and nonAnisotropicStretch.
    /// @note  When using in conjuction with rasterizeSdf for ellipsoids,
    ///   consider choosing a lower value (e.g. same value as
    ///   maxSourcePointsPerVoxel) to speed up iterations and only
    ///   increase if necessary.
    /// @warning  Valid range is [1, inf). Behaviour is undefined if this value
    ///   is set to zero.
    size_t maxTargetPointsPerVoxel = std::numeric_limits<size_t>::max();
    /// @param averagePositions  the amount (between 0 and 1) to average out
    ///   positions. All points, whether they end up as ellipses or not,
    ///   can have their positions smoothed to account for their neighbourhood
    ///   distribution. This will have no effect for points with no neighbours.
    /// @warning  This options does NOT modify the P attribute - instead, a
    ///   world space position attribute "pws" (default) is created and stores
    ///   the smoothed position.
    /// @warning  Valid range is [0, 1]. Behaviour is undefined when outside
    ///   this range.
    float averagePositions = 1.0f;
    /// @param interrupter  optional interrupter
    util::NullInterrupter* interrupter = nullptr;
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
    ///   attribute which represents the orthogonal rotation of each points
    ///   ellipse or the identity matrix for isolated points.
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
