///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file Interpolation.h
///
/// Sampler classes such as PointSampler and BoxSampler that are intended for use
/// with tools::GridTransformer should operate in voxel space and must adhere to
/// the interface described in the example below:
/// @code
/// struct MySampler
/// {
///     // Return a short name that can be used to identify this sampler
///     // in error messages and elsewhere.
///     const char* name() { return "mysampler"; }
///
///     // Return the radius of the sampling kernel in voxels, not including
///     // the center voxel.  This is the number of voxels of padding that
///     // are added to all sides of a volume as a result of resampling.
///     int radius() { return 2; }
///
///     // Return true if scaling by a factor smaller than 0.5 (along any axis)
///     // should be handled via a mipmapping-like scheme of successive halvings
///     // of a grid's resolution, until the remaining scale factor is
///     // greater than or equal to 1/2.  Set this to false only when high-quality
///     // scaling is not required.
///     bool mipmap() { return true; }
///
///     // Specify if sampling at a location that is collocated with a grid point
///     // is guaranteed to return the exact value at that grid point.
///     // For most sampling kernels, this should be false.
///     bool consistent() { return false; }
///
///     // Sample the tree at the given coordinates and return the result in val.
///     // Return true if the sampled value is active.
///     template<class TreeT>
///     bool sample(const TreeT& tree, const Vec3R& coord, typename TreeT::ValueType& val);
/// };
/// @endcode

#ifndef OPENVDB_TOOLS_INTERPOLATION_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_INTERPOLATION_HAS_BEEN_INCLUDED

#include <cmath>
#include <boost/shared_ptr.hpp>
#include <openvdb/version.h> // for OPENVDB_VERSION_NAME
#include <openvdb/Platform.h> // for round()
#include <openvdb/math/Transform.h> // for Transform


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

// The following samplers operate in voxel space.
// When the samplers are applied to grids holding vector or other non-scalar data,
// the data is assumed to be collocated.  For example, using the BoxSampler on a grid
// with ValueType Vec3f assumes that all three elements in a vector can be assigned
// the same physical location.

struct PointSampler
{
    static const char* name() { return "point"; }
    static int radius() { return 0; }
    static bool mipmap() { return false; }
    static bool consistent() { return true; }

    /// @brief Sample @a inTree at the nearest neighbor to @a inCoord
    /// and store the result in @a result.
    /// @return @c true if the sampled value is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
        typename TreeT::ValueType& result);
};


struct BoxSampler
{
    static const char* name() { return "box"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return true; }

    /// @brief Trilinearly reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return @c true if any one of the sampled values is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
        typename TreeT::ValueType& result);

    /// @brief Trilinearly reconstruct @a inTree at @a inCoord.
    /// @return the reconstructed value
    template<class TreeT>
    static typename TreeT::ValueType sample(const TreeT& inTree, const Vec3R& inCoord);

private:
    template<class ValueT, size_t N>
    static inline ValueT trilinearInterpolation(ValueT (& data)[N][N][N], const Vec3R& uvw);
};


struct QuadraticSampler
{
    static const char* name() { return "quadratic"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return false; }

    /// @brief Triquadratically reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return @c true if any one of the sampled values is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
        typename TreeT::ValueType& result);
};


////////////////////////////////////////


// The following samplers operate in voxel space and are designed for Vec3
// staggered grid data (e.g., fluid simulations using the Marker-and-Cell approach
// associate elements of the velocity vector with different physical locations:
// the faces of a cube).

struct StaggeredPointSampler
{
    static const char* name() { return "point"; }
    static int radius() { return 0; }
    static bool mipmap() { return false; }
    static bool consistent() { return false; }

    /// @brief Sample @a inTree at the nearest neighbor to @a inCoord
    /// and store the result in @a result.
    /// @return true if the sampled value is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
        typename TreeT::ValueType& result);
};


struct StaggeredBoxSampler
{
    static const char* name() { return "box"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return false; }

    /// @brief Trilinearly reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return true if any one of the sampled value is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
        typename TreeT::ValueType& result);
};


struct StaggeredQuadraticSampler
{
    static const char* name() { return "quadratic"; }
    static int radius() { return 1; }
    static bool mipmap() { return true; }
    static bool consistent() { return false; }

    /// @brief Triquadratically reconstruct @a inTree at @a inCoord
    /// and store the result in @a result.
    /// @return true if any one of the sampled values is active.
    template<class TreeT>
    static bool sample(const TreeT& inTree, const Vec3R& inCoord,
        typename TreeT::ValueType& result);
};


////////////////////////////////////////


/// @brief Base class that provides the interface for continuous sampling
/// of values in a grid.
/// @details Since grids support only discrete voxel sampling, GridSampler
/// must be used to sample arbitrary continuous points in (world or index) space.
template<typename TreeOrAccessorType, typename SamplerType>
class GridSampler
{
public:
    typedef boost::shared_ptr<GridSampler>          Ptr;
    typedef typename TreeOrAccessorType::ValueType  ValueType;

    /// @param tree  a tree to be sampled, or a ValueAccessor for the tree
    /// @param transform is used when sampling world space locations.
    ///     (by default an identity transform is used)
    explicit GridSampler(const TreeOrAccessorType& tree,
        const math::Transform& transform = math::Transform()):
        mTree(&tree), mTransform(transform) {}

    ~GridSampler() {};

    /// @brief Sample a point in index space in the grid.
    /// @param x x-coordinate of point in index-coordinates of grid
    /// @param y y-coordinate of point in index-coordinates of grid
    /// @param z z-coordinate of point in index-coordinates of grid
    template<typename RealType>
    ValueType sampleVoxel(const RealType& x, const RealType& y, const RealType& z) const
    {
        return isSample(Vec3d(x,y,z));
    }

    /// @brief Sample in index space
    /// @param ispoint the location in index space
    ValueType isSample(const Vec3d& ispoint) const
    {
        ValueType result = zeroVal<ValueType>();
        SamplerType::sample(*mTree, ispoint, result);
        return result;
    }

    /// @brief Sample in world space
    /// @param wspoint the location in world space
    ValueType wsSample(const Vec3d& wspoint) const
    {
        ValueType result = zeroVal<ValueType>();
        SamplerType::sample(*mTree, mTransform.worldToIndex(wspoint), result);
        return result;
    }

private:
    const TreeOrAccessorType* mTree;
    const math::Transform     mTransform;
};


////////////////////////////////////////


namespace local_util {

inline Vec3i
floorVec3(const Vec3R& v)
{
    return Vec3i(int(std::floor(v(0))), int(std::floor(v(1))), int(std::floor(v(2))));
}


inline Vec3i
ceilVec3(const Vec3R& v)
{
    return Vec3i(int(std::ceil(v(0))), int(std::ceil(v(1))), int(std::ceil(v(2))));
}


inline Vec3i
roundVec3(const Vec3R& v)
{
    return Vec3i(int(::round(v(0))), int(::round(v(1))), int(::round(v(2))));
}

} // namespace local_util


////////////////////////////////////////


template<class TreeT>
inline bool
PointSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    Vec3i inIdx = local_util::roundVec3(inCoord);
    return inTree.probeValue(Coord(inIdx), result);
}


////////////////////////////////////////


template<class ValueT, size_t N>
inline ValueT
BoxSampler::trilinearInterpolation(ValueT (& data)[N][N][N], const Vec3R& uvw)
{
    // Trilinear interpolation:
    // The eight surrounding latice values are used to construct the result. \n
    // result(x,y,z) =
    //     v000 (1-x)(1-y)(1-z) + v001 (1-x)(1-y)z + v010 (1-x)y(1-z) + v011 (1-x)yz
    //   + v100 x(1-y)(1-z)     + v101 x(1-y)z     + v110 xy(1-z)     + v111 xyz

    ValueT resultA, resultB;

    resultA = data[0][0][0] + ValueT((data[0][0][1] - data[0][0][0]) * uvw[2]);
    resultB = data[0][1][0] + ValueT((data[0][1][1] - data[0][1][0]) * uvw[2]);
    ValueT result1 = resultA + ValueT((resultB-resultA) * uvw[1]);

    resultA = data[1][0][0] + ValueT((data[1][0][1] - data[1][0][0]) * uvw[2]);
    resultB = data[1][1][0] + ValueT((data[1][1][1] - data[1][1][0]) * uvw[2]);
    ValueT result2 = resultA + ValueT((resultB - resultA) * uvw[1]);

    return result1 + ValueT(uvw[0] * (result2 - result1));
}


template<class TreeT>
inline bool
BoxSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    typedef typename TreeT::ValueType ValueT;

    Vec3i inIdx = local_util::floorVec3(inCoord);
    Vec3R uvw = inCoord - inIdx;

    // Retrieve the values of the eight voxels surrounding the
    // fractional source coordinates.
    ValueT data[2][2][2];

    bool hasActiveValues = false;
    Coord ijk(inIdx);
    hasActiveValues |= inTree.probeValue(ijk, data[0][0][0]);  // i, j, k
    ijk[2] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[0][0][1]);  // i, j, k + 1
    ijk[1] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[0][1][1]); // i, j+1, k + 1
    ijk[2] = inIdx[2];
    hasActiveValues |= inTree.probeValue(ijk, data[0][1][0]);  // i, j+1, k
    ijk[0] += 1;
    ijk[1] = inIdx[1];
    hasActiveValues |= inTree.probeValue(ijk, data[1][0][0]); // i+1, j, k
    ijk[2] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[1][0][1]); // i+1, j, k + 1
    ijk[1] += 1;
    hasActiveValues |= inTree.probeValue(ijk, data[1][1][1]); // i+1, j+1, k + 1
    ijk[2] = inIdx[2];
    hasActiveValues |= inTree.probeValue(ijk, data[1][1][0]); // i+1, j+1, k

    result = trilinearInterpolation(data, uvw);
    return hasActiveValues;
}


template<class TreeT>
inline typename TreeT::ValueType
BoxSampler::sample(const TreeT& inTree, const Vec3R& inCoord)
{
    typedef typename TreeT::ValueType ValueT;

    Vec3i inIdx = local_util::floorVec3(inCoord);
    Vec3R uvw = inCoord - inIdx;

    // Retrieve the values of the eight voxels surrounding the
    // fractional source coordinates.
    ValueT data[2][2][2];

    Coord ijk(inIdx);
    data[0][0][0] = inTree.getValue(ijk);  // i, j, k
    ijk[2] += 1;
    data[0][0][1] = inTree.getValue(ijk);  // i, j, k + 1
    ijk[1] += 1;
    data[0][1][1] = inTree.getValue(ijk); // i, j+1, k + 1
    ijk[2] = inIdx[2];
    data[0][1][0] = inTree.getValue(ijk);  // i, j+1, k
    ijk[0] += 1;
    ijk[1] = inIdx[1];
    data[1][0][0] = inTree.getValue(ijk); // i+1, j, k
    ijk[2] += 1;
    data[1][0][1] = inTree.getValue(ijk); // i+1, j, k + 1
    ijk[1] += 1;
    data[1][1][1] = inTree.getValue(ijk); // i+1, j+1, k + 1
    ijk[2] = inIdx[2];
    data[1][1][0] = inTree.getValue(ijk); // i+1, j+1, k

    return trilinearInterpolation(data, uvw);
}


////////////////////////////////////////


template<class TreeT>
inline bool
QuadraticSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    typedef typename TreeT::ValueType ValueT;

    Vec3i
        inIdx = local_util::floorVec3(inCoord),
        inLoIdx = inIdx - Vec3i(1, 1, 1);
    Vec3R frac = inCoord - inIdx;

    // Retrieve the values of the 27 voxels surrounding the
    // fractional source coordinates.
    bool active = false;
    ValueT v[3][3][3];
    for (int dx = 0, ix = inLoIdx.x(); dx < 3; ++dx, ++ix) {
        for (int dy = 0, iy = inLoIdx.y(); dy < 3; ++dy, ++iy) {
            for (int dz = 0, iz = inLoIdx.z(); dz < 3; ++dz, ++iz) {
                if (inTree.probeValue(Coord(ix, iy, iz), v[dx][dy][dz])) {
                    active = true;
                }
            }
        }
    }

    /// @todo For vector types, interpolate over each component independently.
    ValueT vx[3];
    for (int dx = 0; dx < 3; ++dx) {
        ValueT vy[3];
        for (int dy = 0; dy < 3; ++dy) {
            // Fit a parabola to three contiguous samples in z
            // (at z=-1, z=0 and z=1), then evaluate the parabola at z',
            // where z' is the fractional part of inCoord.z, i.e.,
            // inCoord.z - inIdx.z.  The coefficients come from solving
            //
            // | (-1)^2  -1   1 || a |   | v0 |
            // |    0     0   1 || b | = | v1 |
            // |   1^2    1   1 || c |   | v2 |
            //
            // for a, b and c.
            const ValueT* vz = &v[dx][dy][0];
            const ValueT
                az = static_cast<ValueT>(0.5 * (vz[0] + vz[2]) - vz[1]),
                bz = static_cast<ValueT>(0.5 * (vz[2] - vz[0])),
                cz = static_cast<ValueT>(vz[1]);
            vy[dy] = static_cast<ValueT>(frac.z() * (frac.z() * az + bz) + cz);
        }
        // Fit a parabola to three interpolated samples in y, then
        // evaluate the parabola at y', where y' is the fractional
        // part of inCoord.y.
        const ValueT
            ay = static_cast<ValueT>(0.5 * (vy[0] + vy[2]) - vy[1]),
            by = static_cast<ValueT>(0.5 * (vy[2] - vy[0])),
            cy = static_cast<ValueT>(vy[1]);
        vx[dx] = static_cast<ValueT>(frac.y() * (frac.y() * ay + by) + cy);
    }
    // Fit a parabola to three interpolated samples in x, then
    // evaluate the parabola at the fractional part of inCoord.x.
    const ValueT
        ax = static_cast<ValueT>(0.5 * (vx[0] + vx[2]) - vx[1]),
        bx = static_cast<ValueT>(0.5 * (vx[2] - vx[0])),
        cx = static_cast<ValueT>(vx[1]);
    result = static_cast<ValueT>(frac.x() * (frac.x() * ax + bx) + cx);

    return active;
}


////////////////////////////////////////


template<class TreeT>
inline bool
StaggeredPointSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    typedef typename TreeT::ValueType ValueType;

    ValueType tempX, tempY, tempZ;
    bool active = false;

    active = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0, 0), tempX) || active;
    active = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0.5, 0), tempY) || active;
    active = PointSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0, 0.5), tempZ) || active;

    result.x() = tempX.x();
    result.y() = tempY.y();
    result.z() = tempZ.z();

    return active;
}


////////////////////////////////////////


template<class TreeT>
inline bool
StaggeredBoxSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    typedef typename TreeT::ValueType ValueType;

    ValueType tempX, tempY, tempZ;
    tempX = tempY = tempZ = zeroVal<ValueType>();
    bool active = false;

    active = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0, 0), tempX) || active;
    active = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0.5, 0), tempY) || active;
    active = BoxSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0, 0.5), tempZ) || active;

    result.x() = tempX.x();
    result.y() = tempY.y();
    result.z() = tempZ.z();

    return active;
}


////////////////////////////////////////


template<class TreeT>
inline bool
StaggeredQuadraticSampler::sample(const TreeT& inTree, const Vec3R& inCoord,
    typename TreeT::ValueType& result)
{
    typedef typename TreeT::ValueType ValueType;

    ValueType tempX, tempY, tempZ;
    bool active = false;

    active = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0.5, 0, 0), tempX) || active;
    active = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0.5, 0), tempY) || active;
    active = QuadraticSampler::sample<TreeT>(inTree, inCoord + Vec3R(0, 0, 0.5), tempZ) || active;

    result.x() = tempX.x();
    result.y() = tempY.y();
    result.z() = tempZ.z();

    return active;
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_INTERPOLATION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
