///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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

#ifndef OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h> // for OPENVDB_HAS_CXX11
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/util/Util.h> // for COORD_OFFSETS
#include <openvdb/math/Operators.h> // for ISGradient
#include <openvdb/tools/Morphology.h> // for dilateVoxels()
#include <openvdb/tree/LeafManager.h>
#include "Prune.h" // for pruneInactive

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/utility/enable_if.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vector>
#include <memory> // for auto_ptr/unique_ptr


//////////


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


// Wrapper functions for the VolumeToMesh converter


/// @brief Uniformly mesh any scalar grid that has a continuous isosurface.
///
/// @param grid     a scalar grid to mesh
/// @param points   output list of world space points
/// @param quads    output quad index list
/// @param isovalue determines which isosurface to mesh
///
/// @throw TypeError if @a grid does not have a scalar value type
template<typename GridType>
inline void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& quads,
    double isovalue = 0.0);


/// @brief Adaptively mesh any scalar grid that has a continuous isosurface.
///
/// @param grid         a scalar grid to mesh
/// @param points       output list of world space points
/// @param triangles    output quad index list
/// @param quads        output quad index list
/// @param isovalue     determines which isosurface to mesh
/// @param adaptivity   surface adaptivity threshold [0 to 1]
///
/// @throw TypeError if @a grid does not have a scalar value type
template<typename GridType>
inline void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    std::vector<Vec4I>& quads,
    double isovalue = 0.0,
    double adaptivity = 0.0);


////////////////////////////////////////


/// @brief Polygon flags, used for reference based meshing.
enum { POLYFLAG_EXTERIOR = 0x1, POLYFLAG_FRACTURE_SEAM = 0x2,  POLYFLAG_SUBDIVIDED = 0x4};


/// @brief Collection of quads and triangles
class PolygonPool
{
public:

    inline PolygonPool();
    inline PolygonPool(const size_t numQuads, const size_t numTriangles);


    inline void copy(const PolygonPool& rhs);

    inline void resetQuads(size_t size);
    inline void clearQuads();

    inline void resetTriangles(size_t size);
    inline void clearTriangles();


    // polygon accessor methods

    const size_t& numQuads() const                      { return mNumQuads; }

    openvdb::Vec4I& quad(size_t n)                      { return mQuads[n]; }
    const openvdb::Vec4I& quad(size_t n) const          { return mQuads[n]; }


    const size_t& numTriangles() const                  { return mNumTriangles; }

    openvdb::Vec3I& triangle(size_t n)                  { return mTriangles[n]; }
    const openvdb::Vec3I& triangle(size_t n) const      { return mTriangles[n]; }


    // polygon flags accessor methods

    char& quadFlags(size_t n)                           { return mQuadFlags[n]; }
    const char& quadFlags(size_t n) const               { return mQuadFlags[n]; }

    char& triangleFlags(size_t n)                       { return mTriangleFlags[n]; }
    const char& triangleFlags(size_t n) const           { return mTriangleFlags[n]; }


    // reduce the polygon containers, n has to
    // be smaller than the current container size.

    inline bool trimQuads(const size_t n, bool reallocate = false);
    inline bool trimTrinagles(const size_t n, bool reallocate = false);

private:
    // disallow copy by assignment
    void operator=(const PolygonPool&) {}

    size_t mNumQuads, mNumTriangles;
    boost::scoped_array<openvdb::Vec4I> mQuads;
    boost::scoped_array<openvdb::Vec3I> mTriangles;
    boost::scoped_array<char> mQuadFlags, mTriangleFlags;
};


/// @{
/// @brief Point and primitive list types.
typedef boost::scoped_array<openvdb::Vec3s> PointList;
typedef boost::scoped_array<PolygonPool> PolygonPoolList;
/// @}


////////////////////////////////////////


/// @brief Mesh any scalar grid that has a continuous isosurface.
class VolumeToMesh
{
public:

    /// @param isovalue         Determines which isosurface to mesh.
    /// @param adaptivity       Adaptivity threshold [0 to 1]
    VolumeToMesh(double isovalue = 0, double adaptivity = 0);


    //////////

    // Mesh data accessors

    const size_t& pointListSize() const;
    PointList& pointList();

    const size_t& polygonPoolListSize() const;
    PolygonPoolList& polygonPoolList();
    const PolygonPoolList& polygonPoolList() const;

    std::vector<unsigned char>& pointFlags();
    const std::vector<unsigned char>& pointFlags() const;


    //////////


    /// @brief Main call
    /// @note Call with scalar typed grid.
    template<typename GridT>
    void operator()(const GridT&);


    //////////


    /// @brief  When surfacing fractured SDF fragments, the original unfractured
    ///         SDF grid can be used to eliminate seam lines and tag polygons that are
    ///         coincident with the reference surface with the @c POLYFLAG_EXTERIOR
    ///         flag and polygons that are in proximity to the seam lines with the
    ///         @c POLYFLAG_FRACTURE_SEAM flag. (The performance cost for using this
    ///         reference based scheme compared to the regular meshing scheme is
    ///         approximately 15% for the first fragment and neglect-able for
    ///         subsequent fragments.)
    ///
    /// @note   Attributes from the original asset such as uv coordinates, normals etc.
    ///         are typically transfered to polygons that are marked with the
    ///         @c POLYFLAG_EXTERIOR flag. Polygons that are not marked with this flag
    ///         are interior to reference surface and might need projected UV coordinates
    ///         or a different material. Polygons marked as @c POLYFLAG_FRACTURE_SEAM can
    ///         be used to drive secondary elements such as debris and dust in a FX pipeline.
    ///
    /// @param  grid            reference surface grid of @c GridT type.
    /// @param  secAdaptivity   Secondary adaptivity threshold [0 to 1]. Used in regions
    ///                         that do not exist in the reference grid. (Parts of the
    ///                         fragment surface that are not coincident with the
    ///                         reference surface.)
    void setRefGrid(const GridBase::ConstPtr& grid, double secAdaptivity = 0);


    /// @param mask A boolean grid whose active topology defines the region to mesh.
    /// @param invertMask Toggle to mesh the complement of the mask.
    /// @note The mask's tree configuration has to match @c GridT's tree configuration.
    void setSurfaceMask(const GridBase::ConstPtr& mask, bool invertMask = false);

    /// @param grid A scalar grid used as an spatial multiplier for the adaptivity threshold.
    /// @note The grid's tree configuration has to match @c GridT's tree configuration.
    void setSpatialAdaptivity(const GridBase::ConstPtr& grid);


    /// @param tree A boolean tree whose active topology defines the adaptivity mask.
    /// @note The tree configuration has to match @c GridT's tree configuration.
    void setAdaptivityMask(const TreeBase::ConstPtr& tree);


    /// @brief Subdivide volume and mesh into disjoint parts
    /// @param partitions Number of partitions.
    /// @param activePart Specific partition to mesh, 0 to @c partitions - 1.
    void partition(unsigned partitions = 1, unsigned activePart = 0);

private:

    PointList mPoints;
    PolygonPoolList mPolygons;

    size_t mPointListSize, mSeamPointListSize, mPolygonPoolListSize;
    double mIsovalue, mPrimAdaptivity, mSecAdaptivity;

    GridBase::ConstPtr mRefGrid, mSurfaceMaskGrid, mAdaptivityGrid;
    TreeBase::ConstPtr mAdaptivityMaskTree;

    TreeBase::Ptr mRefSignTree, mRefIdxTree;

    bool mInvertSurfaceMask;
    unsigned mPartitions, mActivePart;

    boost::scoped_array<uint32_t> mQuantizedSeamPoints;

    std::vector<unsigned char> mPointFlags;
};


////////////////////////////////////////


/// @brief  Given a set of tangent elements, @c points with corresponding @c normals,
///         this method returns the intersection point of all tangent elements.
///
/// @note   Used to extract surfaces with sharp edges and corners from volume data,
///         see the following paper for details: "Feature Sensitive Surface
///         Extraction from Volume Data, Kobbelt et al. 2001".
inline Vec3d findFeaturePoint(
    const std::vector<Vec3d>& points,
    const std::vector<Vec3d>& normals)
{
    typedef math::Mat3d Mat3d;

    Vec3d avgPos(0.0);

    if (points.empty()) return avgPos;

    for (size_t n = 0, N = points.size(); n < N; ++n) {
        avgPos += points[n];
    }

    avgPos /= double(points.size());

    // Unique components of the 3x3 A^TA matrix, where A is
    // the matrix of normals.
    double m00=0,m01=0,m02=0,
           m11=0,m12=0,
           m22=0;

    // The rhs vector, A^Tb, where b = n dot p
    Vec3d rhs(0.0);

    for (size_t n = 0, N = points.size(); n < N; ++n) {

        const Vec3d& n_ref = normals[n];

        // A^TA
        m00 += n_ref[0] * n_ref[0]; // diagonal
        m11 += n_ref[1] * n_ref[1];
        m22 += n_ref[2] * n_ref[2];

        m01 += n_ref[0] * n_ref[1]; // Upper-tri
        m02 += n_ref[0] * n_ref[2];
        m12 += n_ref[1] * n_ref[2];

        // A^Tb (centered around the origin)
        rhs += n_ref * n_ref.dot(points[n] - avgPos);
    }

    Mat3d A(m00,m01,m02,
            m01,m11,m12,
            m02,m12,m22);

    /*
    // Inverse
    const double det = A.det();
    if (det > 0.01) {
        Mat3d A_inv = A.adjoint();
        A_inv *= (1.0 / det);

        return avgPos + A_inv * rhs;
    }
    */

    // Compute the pseudo inverse

    math::Mat3d eigenVectors;
    Vec3d eigenValues;

    diagonalizeSymmetricMatrix(A, eigenVectors, eigenValues, 300);

    Mat3d D = Mat3d::identity();


    double tolerance = std::max(std::abs(eigenValues[0]), std::abs(eigenValues[1]));
    tolerance = std::max(tolerance, std::abs(eigenValues[2]));
    tolerance *= 0.01;

    int clamped = 0;
    for (int i = 0; i < 3; ++i ) {
        if (std::abs(eigenValues[i]) < tolerance) {
            D[i][i] = 0.0;
            ++clamped;
        } else {
            D[i][i] = 1.0 / eigenValues[i];
        }
    }

    // Assemble the pseudo inverse and calc. the intersection point
    if (clamped < 3) {
        Mat3d pseudoInv = eigenVectors * D *  eigenVectors.transpose();
        return avgPos + pseudoInv * rhs;
    }

    return avgPos;
}


////////////////////////////////////////


// Internal utility methods
namespace internal {

template<typename T>
struct UniquePtr
{
#ifdef OPENVDB_HAS_CXX11
    typedef std::unique_ptr<T>  type;
#else
    typedef std::auto_ptr<T>    type;
#endif
};


/// @brief  Bit-flags used to classify cells.
enum { SIGNS = 0xFF, EDGES = 0xE00, INSIDE = 0x100,
       XEDGE = 0x200, YEDGE = 0x400, ZEDGE = 0x800, SEAM = 0x1000};


/// @brief Used to quickly determine if a given cell is adaptable.
const bool sAdaptable[256] = {
    1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,
    1,0,1,1,0,0,1,1,0,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,1,0,1,0,0,0,1,
    1,0,0,0,1,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,1,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,1,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,1,0,0,0,1,
    1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,1,
    1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1};


/// @brief  Contains the ambiguous face index for certain cell configuration.
const unsigned char sAmbiguousFace[256] = {
    0,0,0,0,0,5,0,0,0,0,5,0,0,0,0,0,0,0,1,0,0,5,1,0,4,0,0,0,4,0,0,0,
    0,1,0,0,2,0,0,0,0,1,5,0,2,0,0,0,0,0,0,0,2,0,0,0,4,0,0,0,0,0,0,0,
    0,0,2,2,0,5,0,0,3,3,0,0,0,0,0,0,6,6,0,0,6,0,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,4,0,4,3,0,3,0,0,0,5,0,0,0,0,0,0,0,1,0,3,0,0,0,0,0,0,0,0,0,0,0,
    6,0,6,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


/// @brief  Lookup table for different cell sign configurations. The first entry specifies
///         the total number of points that need to be generated inside a cell and the
///         remaining 12 entries indicate different edge groups.
const unsigned char sEdgeGroupTable[256][13] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0},{1,1,0,0,1,0,0,0,0,1,0,0,0},{1,1,1,0,0,0,0,0,0,0,1,0,0},
    {1,0,1,0,1,0,0,0,0,1,1,0,0},{1,0,1,1,0,0,0,0,0,0,0,1,0},{1,1,1,1,1,0,0,0,0,1,0,1,0},
    {1,1,0,1,0,0,0,0,0,0,1,1,0},{1,0,0,1,1,0,0,0,0,1,1,1,0},{1,0,0,1,1,0,0,0,0,0,0,0,1},
    {1,1,0,1,0,0,0,0,0,1,0,0,1},{1,1,1,1,1,0,0,0,0,0,1,0,1},{1,0,1,1,0,0,0,0,0,1,1,0,1},
    {1,0,1,0,1,0,0,0,0,0,0,1,1},{1,1,1,0,0,0,0,0,0,1,0,1,1},{1,1,0,0,1,0,0,0,0,0,1,1,1},
    {1,0,0,0,0,0,0,0,0,1,1,1,1},{1,0,0,0,0,1,0,0,1,1,0,0,0},{1,1,0,0,1,1,0,0,1,0,0,0,0},
    {1,1,1,0,0,1,0,0,1,1,1,0,0},{1,0,1,0,1,1,0,0,1,0,1,0,0},{2,0,1,1,0,2,0,0,2,2,0,1,0},
    {1,1,1,1,1,1,0,0,1,0,0,1,0},{1,1,0,1,0,1,0,0,1,1,1,1,0},{1,0,0,1,1,1,0,0,1,0,1,1,0},
    {1,0,0,1,1,1,0,0,1,1,0,0,1},{1,1,0,1,0,1,0,0,1,0,0,0,1},{2,2,1,1,2,1,0,0,1,2,1,0,1},
    {1,0,1,1,0,1,0,0,1,0,1,0,1},{1,0,1,0,1,1,0,0,1,1,0,1,1},{1,1,1,0,0,1,0,0,1,0,0,1,1},
    {2,1,0,0,1,2,0,0,2,1,2,2,2},{1,0,0,0,0,1,0,0,1,0,1,1,1},{1,0,0,0,0,1,1,0,0,0,1,0,0},
    {1,1,0,0,1,1,1,0,0,1,1,0,0},{1,1,1,0,0,1,1,0,0,0,0,0,0},{1,0,1,0,1,1,1,0,0,1,0,0,0},
    {1,0,1,1,0,1,1,0,0,0,1,1,0},{2,2,2,1,1,1,1,0,0,1,2,1,0},{1,1,0,1,0,1,1,0,0,0,0,1,0},
    {1,0,0,1,1,1,1,0,0,1,0,1,0},{2,0,0,2,2,1,1,0,0,0,1,0,2},{1,1,0,1,0,1,1,0,0,1,1,0,1},
    {1,1,1,1,1,1,1,0,0,0,0,0,1},{1,0,1,1,0,1,1,0,0,1,0,0,1},{1,0,1,0,1,1,1,0,0,0,1,1,1},
    {2,1,1,0,0,2,2,0,0,2,1,2,2},{1,1,0,0,1,1,1,0,0,0,0,1,1},{1,0,0,0,0,1,1,0,0,1,0,1,1},
    {1,0,0,0,0,0,1,0,1,1,1,0,0},{1,1,0,0,1,0,1,0,1,0,1,0,0},{1,1,1,0,0,0,1,0,1,1,0,0,0},
    {1,0,1,0,1,0,1,0,1,0,0,0,0},{1,0,1,1,0,0,1,0,1,1,1,1,0},{2,1,1,2,2,0,2,0,2,0,1,2,0},
    {1,1,0,1,0,0,1,0,1,1,0,1,0},{1,0,0,1,1,0,1,0,1,0,0,1,0},{1,0,0,1,1,0,1,0,1,1,1,0,1},
    {1,1,0,1,0,0,1,0,1,0,1,0,1},{2,1,2,2,1,0,2,0,2,1,0,0,2},{1,0,1,1,0,0,1,0,1,0,0,0,1},
    {2,0,2,0,2,0,1,0,1,2,2,1,1},{2,2,2,0,0,0,1,0,1,0,2,1,1},{2,2,0,0,2,0,1,0,1,2,0,1,1},
    {1,0,0,0,0,0,1,0,1,0,0,1,1},{1,0,0,0,0,0,1,1,0,0,0,1,0},{2,1,0,0,1,0,2,2,0,1,0,2,0},
    {1,1,1,0,0,0,1,1,0,0,1,1,0},{1,0,1,0,1,0,1,1,0,1,1,1,0},{1,0,1,1,0,0,1,1,0,0,0,0,0},
    {1,1,1,1,1,0,1,1,0,1,0,0,0},{1,1,0,1,0,0,1,1,0,0,1,0,0},{1,0,0,1,1,0,1,1,0,1,1,0,0},
    {1,0,0,1,1,0,1,1,0,0,0,1,1},{1,1,0,1,0,0,1,1,0,1,0,1,1},{2,1,2,2,1,0,1,1,0,0,1,2,1},
    {2,0,1,1,0,0,2,2,0,2,2,1,2},{1,0,1,0,1,0,1,1,0,0,0,0,1},{1,1,1,0,0,0,1,1,0,1,0,0,1},
    {1,1,0,0,1,0,1,1,0,0,1,0,1},{1,0,0,0,0,0,1,1,0,1,1,0,1},{1,0,0,0,0,1,1,1,1,1,0,1,0},
    {1,1,0,0,1,1,1,1,1,0,0,1,0},{2,1,1,0,0,2,2,1,1,1,2,1,0},{2,0,2,0,2,1,1,2,2,0,1,2,0},
    {1,0,1,1,0,1,1,1,1,1,0,0,0},{2,2,2,1,1,2,2,1,1,0,0,0,0},{2,2,0,2,0,1,1,2,2,2,1,0,0},
    {2,0,0,1,1,2,2,1,1,0,2,0,0},{2,0,0,1,1,1,1,2,2,1,0,1,2},{2,2,0,2,0,2,2,1,1,0,0,2,1},
    {4,3,2,2,3,4,4,1,1,3,4,2,1},{3,0,2,2,0,1,1,3,3,0,1,2,3},{2,0,2,0,2,2,2,1,1,2,0,0,1},
    {2,1,1,0,0,1,1,2,2,0,0,0,2},{3,1,0,0,1,2,2,3,3,1,2,0,3},{2,0,0,0,0,1,1,2,2,0,1,0,2},
    {1,0,0,0,0,1,0,1,0,0,1,1,0},{1,1,0,0,1,1,0,1,0,1,1,1,0},{1,1,1,0,0,1,0,1,0,0,0,1,0},
    {1,0,1,0,1,1,0,1,0,1,0,1,0},{1,0,1,1,0,1,0,1,0,0,1,0,0},{2,1,1,2,2,2,0,2,0,2,1,0,0},
    {1,1,0,1,0,1,0,1,0,0,0,0,0},{1,0,0,1,1,1,0,1,0,1,0,0,0},{1,0,0,1,1,1,0,1,0,0,1,1,1},
    {2,2,0,2,0,1,0,1,0,1,2,2,1},{2,2,1,1,2,2,0,2,0,0,0,1,2},{2,0,2,2,0,1,0,1,0,1,0,2,1},
    {1,0,1,0,1,1,0,1,0,0,1,0,1},{2,2,2,0,0,1,0,1,0,1,2,0,1},{1,1,0,0,1,1,0,1,0,0,0,0,1},
    {1,0,0,0,0,1,0,1,0,1,0,0,1},{1,0,0,0,0,0,0,1,1,1,1,1,0},{1,1,0,0,1,0,0,1,1,0,1,1,0},
    {1,1,1,0,0,0,0,1,1,1,0,1,0},{1,0,1,0,1,0,0,1,1,0,0,1,0},{1,0,1,1,0,0,0,1,1,1,1,0,0},
    {2,2,2,1,1,0,0,1,1,0,2,0,0},{1,1,0,1,0,0,0,1,1,1,0,0,0},{1,0,0,1,1,0,0,1,1,0,0,0,0},
    {2,0,0,2,2,0,0,1,1,2,2,2,1},{2,1,0,1,0,0,0,2,2,0,1,1,2},{3,2,1,1,2,0,0,3,3,2,0,1,3},
    {2,0,1,1,0,0,0,2,2,0,0,1,2},{2,0,1,0,1,0,0,2,2,1,1,0,2},{2,1,1,0,0,0,0,2,2,0,1,0,2},
    {2,1,0,0,1,0,0,2,2,1,0,0,2},{1,0,0,0,0,0,0,1,1,0,0,0,1},{1,0,0,0,0,0,0,1,1,0,0,0,1},
    {1,1,0,0,1,0,0,1,1,1,0,0,1},{2,1,1,0,0,0,0,2,2,0,1,0,2},{1,0,1,0,1,0,0,1,1,1,1,0,1},
    {1,0,1,1,0,0,0,1,1,0,0,1,1},{2,1,1,2,2,0,0,1,1,1,0,1,2},{1,1,0,1,0,0,0,1,1,0,1,1,1},
    {2,0,0,1,1,0,0,2,2,2,2,2,1},{1,0,0,1,1,0,0,1,1,0,0,0,0},{1,1,0,1,0,0,0,1,1,1,0,0,0},
    {1,1,1,1,1,0,0,1,1,0,1,0,0},{1,0,1,1,0,0,0,1,1,1,1,0,0},{1,0,1,0,1,0,0,1,1,0,0,1,0},
    {1,1,1,0,0,0,0,1,1,1,0,1,0},{1,1,0,0,1,0,0,1,1,0,1,1,0},{1,0,0,0,0,0,0,1,1,1,1,1,0},
    {1,0,0,0,0,1,0,1,0,1,0,0,1},{1,1,0,0,1,1,0,1,0,0,0,0,1},{1,1,1,0,0,1,0,1,0,1,1,0,1},
    {1,0,1,0,1,1,0,1,0,0,1,0,1},{1,0,1,1,0,1,0,1,0,1,0,1,1},{2,2,2,1,1,2,0,2,0,0,0,2,1},
    {2,1,0,1,0,2,0,2,0,1,2,2,1},{2,0,0,2,2,1,0,1,0,0,1,1,2},{1,0,0,1,1,1,0,1,0,1,0,0,0},
    {1,1,0,1,0,1,0,1,0,0,0,0,0},{2,1,2,2,1,2,0,2,0,1,2,0,0},{1,0,1,1,0,1,0,1,0,0,1,0,0},
    {1,0,1,0,1,1,0,1,0,1,0,1,0},{1,1,1,0,0,1,0,1,0,0,0,1,0},{2,2,0,0,2,1,0,1,0,2,1,1,0},
    {1,0,0,0,0,1,0,1,0,0,1,1,0},{1,0,0,0,0,1,1,1,1,0,1,0,1},{2,1,0,0,1,2,1,1,2,2,1,0,1},
    {1,1,1,0,0,1,1,1,1,0,0,0,1},{2,0,2,0,2,1,2,2,1,1,0,0,2},{2,0,1,1,0,1,2,2,1,0,1,2,1},
    {4,1,1,3,3,2,4,4,2,2,1,4,3},{2,2,0,2,0,2,1,1,2,0,0,1,2},{3,0,0,1,1,2,3,3,2,2,0,3,1},
    {1,0,0,1,1,1,1,1,1,0,1,0,0},{2,2,0,2,0,1,2,2,1,1,2,0,0},{2,2,1,1,2,2,1,1,2,0,0,0,0},
    {2,0,1,1,0,2,1,1,2,2,0,0,0},{2,0,2,0,2,2,1,1,2,0,2,1,0},{3,1,1,0,0,3,2,2,3,3,1,2,0},
    {2,1,0,0,1,1,2,2,1,0,0,2,0},{2,0,0,0,0,2,1,1,2,2,0,1,0},{1,0,0,0,0,0,1,1,0,1,1,0,1},
    {1,1,0,0,1,0,1,1,0,0,1,0,1},{1,1,1,0,0,0,1,1,0,1,0,0,1},{1,0,1,0,1,0,1,1,0,0,0,0,1},
    {2,0,2,2,0,0,1,1,0,2,2,1,2},{3,1,1,2,2,0,3,3,0,0,1,3,2},{2,1,0,1,0,0,2,2,0,1,0,2,1},
    {2,0,0,1,1,0,2,2,0,0,0,2,1},{1,0,0,1,1,0,1,1,0,1,1,0,0},{1,1,0,1,0,0,1,1,0,0,1,0,0},
    {2,2,1,1,2,0,1,1,0,2,0,0,0},{1,0,1,1,0,0,1,1,0,0,0,0,0},{2,0,1,0,1,0,2,2,0,1,1,2,0},
    {2,1,1,0,0,0,2,2,0,0,1,2,0},{2,1,0,0,1,0,2,2,0,1,0,2,0},{1,0,0,0,0,0,1,1,0,0,0,1,0},
    {1,0,0,0,0,0,1,0,1,0,0,1,1},{1,1,0,0,1,0,1,0,1,1,0,1,1},{1,1,1,0,0,0,1,0,1,0,1,1,1},
    {2,0,2,0,2,0,1,0,1,1,1,2,2},{1,0,1,1,0,0,1,0,1,0,0,0,1},{2,2,2,1,1,0,2,0,2,2,0,0,1},
    {1,1,0,1,0,0,1,0,1,0,1,0,1},{2,0,0,2,2,0,1,0,1,1,1,0,2},{1,0,0,1,1,0,1,0,1,0,0,1,0},
    {1,1,0,1,0,0,1,0,1,1,0,1,0},{2,2,1,1,2,0,2,0,2,0,2,1,0},{2,0,2,2,0,0,1,0,1,1,1,2,0},
    {1,0,1,0,1,0,1,0,1,0,0,0,0},{1,1,1,0,0,0,1,0,1,1,0,0,0},{1,1,0,0,1,0,1,0,1,0,1,0,0},
    {1,0,0,0,0,0,1,0,1,1,1,0,0},{1,0,0,0,0,1,1,0,0,1,0,1,1},{1,1,0,0,1,1,1,0,0,0,0,1,1},
    {2,2,2,0,0,1,1,0,0,2,1,2,2},{2,0,1,0,1,2,2,0,0,0,2,1,1},{1,0,1,1,0,1,1,0,0,1,0,0,1},
    {2,1,1,2,2,1,1,0,0,0,0,0,2},{2,1,0,1,0,2,2,0,0,1,2,0,1},{2,0,0,2,2,1,1,0,0,0,1,0,2},
    {1,0,0,1,1,1,1,0,0,1,0,1,0},{1,1,0,1,0,1,1,0,0,0,0,1,0},{3,1,2,2,1,3,3,0,0,1,3,2,0},
    {2,0,1,1,0,2,2,0,0,0,2,1,0},{1,0,1,0,1,1,1,0,0,1,0,0,0},{1,1,1,0,0,1,1,0,0,0,0,0,0},
    {2,2,0,0,2,1,1,0,0,2,1,0,0},{1,0,0,0,0,1,1,0,0,0,1,0,0},{1,0,0,0,0,1,0,0,1,0,1,1,1},
    {2,2,0,0,2,1,0,0,1,1,2,2,2},{1,1,1,0,0,1,0,0,1,0,0,1,1},{2,0,1,0,1,2,0,0,2,2,0,1,1},
    {1,0,1,1,0,1,0,0,1,0,1,0,1},{3,1,1,3,3,2,0,0,2,2,1,0,3},{1,1,0,1,0,1,0,0,1,0,0,0,1},
    {2,0,0,2,2,1,0,0,1,1,0,0,2},{1,0,0,1,1,1,0,0,1,0,1,1,0},{2,1,0,1,0,2,0,0,2,2,1,1,0},
    {2,1,2,2,1,1,0,0,1,0,0,2,0},{2,0,1,1,0,2,0,0,2,2,0,1,0},{1,0,1,0,1,1,0,0,1,0,1,0,0},
    {2,1,1,0,0,2,0,0,2,2,1,0,0},{1,1,0,0,1,1,0,0,1,0,0,0,0},{1,0,0,0,0,1,0,0,1,1,0,0,0},
    {1,0,0,0,0,0,0,0,0,1,1,1,1},{1,1,0,0,1,0,0,0,0,0,1,1,1},{1,1,1,0,0,0,0,0,0,1,0,1,1},
    {1,0,1,0,1,0,0,0,0,0,0,1,1},{1,0,1,1,0,0,0,0,0,1,1,0,1},{2,1,1,2,2,0,0,0,0,0,1,0,2},
    {1,1,0,1,0,0,0,0,0,1,0,0,1},{1,0,0,1,1,0,0,0,0,0,0,0,1},{1,0,0,1,1,0,0,0,0,1,1,1,0},
    {1,1,0,1,0,0,0,0,0,0,1,1,0},{2,1,2,2,1,0,0,0,0,1,0,2,0},{1,0,1,1,0,0,0,0,0,0,0,1,0},
    {1,0,1,0,1,0,0,0,0,1,1,0,0},{1,1,1,0,0,0,0,0,0,0,1,0,0},{1,1,0,0,1,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0}};


////////////////////////////////////////

inline bool
isPlanarQuad(
    const Vec3d& p0, const Vec3d& p1,
    const Vec3d& p2, const Vec3d& p3,
    double epsilon = 0.001)
{
    // compute representative plane
    Vec3d normal = (p2-p0).cross(p1-p3);
    normal.normalize();
    const Vec3d centroid = (p0 + p1 + p2 + p3);
    const double d = centroid.dot(normal) * 0.25;


    // test vertice distance to plane
    double absDist = std::abs(p0.dot(normal) - d);
    if (absDist > epsilon) return false;

    absDist = std::abs(p1.dot(normal) - d);
    if (absDist > epsilon) return false;

    absDist = std::abs(p2.dot(normal) - d);
    if (absDist > epsilon) return false;

    absDist = std::abs(p3.dot(normal) - d);
    if (absDist > epsilon) return false;

    return true;
}


////////////////////////////////////////


/// @{
/// @brief  Utility methods for point quantization.

enum {
    MASK_FIRST_10_BITS = 0x000003FF,
    MASK_DIRTY_BIT =     0x80000000,
    MASK_INVALID_BIT =   0x40000000
};

inline uint32_t
packPoint(const Vec3d& v)
{
    uint32_t data = 0;

    // values are expected to be in the [0.0 to 1.0] range.
    assert(!(v.x() > 1.0) && !(v.y() > 1.0) && !(v.z() > 1.0));
    assert(!(v.x() < 0.0) && !(v.y() < 0.0) && !(v.z() < 0.0));

    data |= (uint32_t(v.x() * 1023.0) & MASK_FIRST_10_BITS) << 20;
    data |= (uint32_t(v.y() * 1023.0) & MASK_FIRST_10_BITS) << 10;
    data |= (uint32_t(v.z() * 1023.0) & MASK_FIRST_10_BITS);

    return data;
}

inline Vec3d
unpackPoint(uint32_t data)
{
    Vec3d v;
    v.z() = double(data & MASK_FIRST_10_BITS) * 0.0009775171;
    data = data >> 10;
    v.y() = double(data & MASK_FIRST_10_BITS) * 0.0009775171;
    data = data >> 10;
    v.x() = double(data & MASK_FIRST_10_BITS) * 0.0009775171;

    return v;
}

/// @}

////////////////////////////////////////


/// @brief  General method that computes the cell-sign configuration at the given
///         @c ijk coordinate.
template<typename AccessorT>
inline unsigned char
evalCellSigns(const AccessorT& accessor, const Coord& ijk, typename AccessorT::ValueType iso)
{
    unsigned signs = 0;
    Coord coord = ijk; // i, j, k
    if (accessor.getValue(coord) < iso) signs |= 1u;
    coord[0] += 1; // i+1, j, k
    if (accessor.getValue(coord) < iso) signs |= 2u;
    coord[2] += 1; // i+1, j, k+1
    if (accessor.getValue(coord) < iso) signs |= 4u;
    coord[0] = ijk[0]; // i, j, k+1
    if (accessor.getValue(coord) < iso) signs |= 8u;
    coord[1] += 1; coord[2] = ijk[2]; // i, j+1, k
    if (accessor.getValue(coord) < iso) signs |= 16u;
    coord[0] += 1; // i+1, j+1, k
    if (accessor.getValue(coord) < iso) signs |= 32u;
    coord[2] += 1; // i+1, j+1, k+1
    if (accessor.getValue(coord) < iso) signs |= 64u;
    coord[0] = ijk[0]; // i, j+1, k+1
    if (accessor.getValue(coord) < iso) signs |= 128u;
    return uint8_t(signs);
}


/// @brief  Leaf node optimized method that computes the cell-sign configuration
///         at the given local @c offset
template<typename LeafT>
inline unsigned char
evalCellSigns(const LeafT& leaf, const Index offset, typename LeafT::ValueType iso)
{
    unsigned char signs = 0;

    // i, j, k
    if (leaf.getValue(offset) < iso) signs |= 1u;

    // i, j, k+1
    if (leaf.getValue(offset + 1) < iso) signs |= 8u;

    // i, j+1, k
    if (leaf.getValue(offset + LeafT::DIM) < iso) signs |= 16u;

    // i, j+1, k+1
    if (leaf.getValue(offset + LeafT::DIM + 1) < iso) signs |= 128u;

    // i+1, j, k
    if (leaf.getValue(offset + (LeafT::DIM * LeafT::DIM) ) < iso) signs |= 2u;

    // i+1, j, k+1
    if (leaf.getValue(offset + (LeafT::DIM * LeafT::DIM) + 1) < iso) signs |= 4u;

    // i+1, j+1, k
    if (leaf.getValue(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM) < iso) signs |= 32u;

    // i+1, j+1, k+1
    if (leaf.getValue(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM + 1) < iso) signs |= 64u;

    return signs;
}


/// @brief  Used to correct topological ambiguities related to two adjacent cells
///         that share an ambiguous face.
template<class AccessorT>
inline void
correctCellSigns(unsigned char& signs, unsigned char face,
    const AccessorT& acc, Coord ijk, typename AccessorT::ValueType iso)
{
    if (face == 1) {
        ijk[2] -= 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 3) signs = uint8_t(~signs);
    } else if (face == 3) {
        ijk[2] += 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 1) signs = uint8_t(~signs);
    } else if (face == 2) {
        ijk[0] += 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 4) signs = uint8_t(~signs);
    } else if (face == 4) {
        ijk[0] -= 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 2) signs = uint8_t(~signs);
    } else if (face == 5) {
        ijk[1] -= 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 6) signs = uint8_t(~signs);
    } else if (face == 6) {
        ijk[1] += 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 5) signs = uint8_t(~signs);
    }
}


template<class AccessorT>
inline bool
isNonManifold(const AccessorT& accessor, const Coord& ijk,
    typename AccessorT::ValueType isovalue, const int dim)
{
    int hDim = dim >> 1;
    bool m, p[8]; // Corner signs

    Coord coord = ijk; // i, j, k
    p[0] = accessor.getValue(coord) < isovalue;
    coord[0] += dim; // i+dim, j, k
    p[1] = accessor.getValue(coord) < isovalue;
    coord[2] += dim; // i+dim, j, k+dim
    p[2] = accessor.getValue(coord) < isovalue;
    coord[0] = ijk[0]; // i, j, k+dim
    p[3] = accessor.getValue(coord) < isovalue;
    coord[1] += dim; coord[2] = ijk[2]; // i, j+dim, k
    p[4] = accessor.getValue(coord) < isovalue;
    coord[0] += dim; // i+dim, j+dim, k
    p[5] = accessor.getValue(coord) < isovalue;
    coord[2] += dim; // i+dim, j+dim, k+dim
    p[6] = accessor.getValue(coord) < isovalue;
    coord[0] = ijk[0]; // i, j+dim, k+dim
    p[7] = accessor.getValue(coord) < isovalue;

    // Check if the corner sign configuration is ambiguous
    unsigned signs = 0;
    if (p[0]) signs |= 1u;
    if (p[1]) signs |= 2u;
    if (p[2]) signs |= 4u;
    if (p[3]) signs |= 8u;
    if (p[4]) signs |= 16u;
    if (p[5]) signs |= 32u;
    if (p[6]) signs |= 64u;
    if (p[7]) signs |= 128u;
    if (!sAdaptable[signs]) return true;

    // Manifold check

    // Evaluate edges
    int i = ijk[0], ip = ijk[0] + hDim, ipp = ijk[0] + dim;
    int j = ijk[1], jp = ijk[1] + hDim, jpp = ijk[1] + dim;
    int k = ijk[2], kp = ijk[2] + hDim, kpp = ijk[2] + dim;

    // edge 1
    coord.reset(ip, j, k);
    m = accessor.getValue(coord) < isovalue;
    if (p[0] != m && p[1] != m) return true;

    // edge 2
    coord.reset(ipp, j, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[1] != m && p[2] != m) return true;

    // edge 3
    coord.reset(ip, j, kpp);
    m = accessor.getValue(coord) < isovalue;
    if (p[2] != m && p[3] != m) return true;

    // edge 4
    coord.reset(i, j, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[0] != m && p[3] != m) return true;

    // edge 5
    coord.reset(ip, jpp, k);
    m = accessor.getValue(coord) < isovalue;
    if (p[4] != m && p[5] != m) return true;

    // edge 6
    coord.reset(ipp, jpp, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[5] != m && p[6] != m) return true;

    // edge 7
    coord.reset(ip, jpp, kpp);
    m = accessor.getValue(coord) < isovalue;
    if (p[6] != m && p[7] != m) return true;

    // edge 8
    coord.reset(i, jpp, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[7] != m && p[4] != m) return true;

    // edge 9
    coord.reset(i, jp, k);
    m = accessor.getValue(coord) < isovalue;
    if (p[0] != m && p[4] != m) return true;

    // edge 10
    coord.reset(ipp, jp, k);
    m = accessor.getValue(coord) < isovalue;
    if (p[1] != m && p[5] != m) return true;

    // edge 11
    coord.reset(ipp, jp, kpp);
    m = accessor.getValue(coord) < isovalue;
    if (p[2] != m && p[6] != m) return true;


    // edge 12
    coord.reset(i, jp, kpp);
    m = accessor.getValue(coord) < isovalue;
    if (p[3] != m && p[7] != m) return true;


    // Evaluate faces

    // face 1
    coord.reset(ip, jp, k);
    m = accessor.getValue(coord) < isovalue;
    if (p[0] != m && p[1] != m && p[4] != m && p[5] != m) return true;

    // face 2
    coord.reset(ipp, jp, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[1] != m && p[2] != m && p[5] != m && p[6] != m) return true;

    // face 3
    coord.reset(ip, jp, kpp);
    m = accessor.getValue(coord) < isovalue;
    if (p[2] != m && p[3] != m && p[6] != m && p[7] != m) return true;

    // face 4
    coord.reset(i, jp, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[0] != m && p[3] != m && p[4] != m && p[7] != m) return true;

    // face 5
    coord.reset(ip, j, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[0] != m && p[1] != m && p[2] != m && p[3] != m) return true;

    // face 6
    coord.reset(ip, jpp, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[4] != m && p[5] != m && p[6] != m && p[7] != m) return true;

    // test cube center
    coord.reset(ip, jp, kp);
    m = accessor.getValue(coord) < isovalue;
    if (p[0] != m && p[1] != m && p[2] != m && p[3] != m &&
        p[4] != m && p[5] != m && p[6] != m && p[7] != m) return true;

    return false;
}


////////////////////////////////////////


template <class LeafType>
inline void
mergeVoxels(LeafType& leaf, const Coord& start, int dim, int regionId)
{
    Coord ijk, end = start;
    end[0] += dim;
    end[1] += dim;
    end[2] += dim;

    for (ijk[0] = start[0]; ijk[0] < end[0]; ++ijk[0]) {
        for (ijk[1] = start[1]; ijk[1] < end[1]; ++ijk[1]) {
            for (ijk[2] = start[2]; ijk[2] < end[2]; ++ijk[2]) {
                leaf.setValueOnly(ijk, regionId);
            }
        }
    }
}


// Note that we must use ValueType::value_type or else Visual C++ gets confused
// thinking that it is a constructor.
template <class LeafType>
inline bool
isMergable(LeafType& leaf, const Coord& start, int dim,
    typename LeafType::ValueType::value_type adaptivity)
{
    if (adaptivity < 1e-6) return false;

    typedef typename LeafType::ValueType VecT;
    Coord ijk, end = start;
    end[0] += dim;
    end[1] += dim;
    end[2] += dim;

    std::vector<VecT> norms;
    for (ijk[0] = start[0]; ijk[0] < end[0]; ++ijk[0]) {
        for (ijk[1] = start[1]; ijk[1] < end[1]; ++ijk[1]) {
            for (ijk[2] = start[2]; ijk[2] < end[2]; ++ijk[2]) {

                if(!leaf.isValueOn(ijk)) continue;
                norms.push_back(leaf.getValue(ijk));
            }
        }
    }

    size_t N = norms.size();
    for (size_t ni = 0; ni < N; ++ni) {
        VecT n_i = norms[ni];
        for (size_t nj = 0; nj < N; ++nj) {
            VecT n_j = norms[nj];
            if ((1.0 - n_i.dot(n_j)) > adaptivity) return false;
        }
    }
    return true;
}


////////////////////////////////////////


template<typename TreeT, typename LeafManagerT>
class SignData
{
public:
    typedef typename TreeT::ValueType ValueT;
    typedef tree::ValueAccessor<const TreeT> AccessorT;

    typedef typename TreeT::template ValueConverter<int>::Type IntTreeT;
    typedef tree::ValueAccessor<IntTreeT> IntAccessorT;

    typedef typename TreeT::template ValueConverter<Int16>::Type Int16TreeT;
    typedef tree::ValueAccessor<Int16TreeT> Int16AccessorT;

    //////////


    SignData(const TreeT& distTree, const LeafManagerT& leafs, ValueT iso);

    void run(bool threaded = true);

    typename Int16TreeT::Ptr signTree() const { return mSignTree; }
    typename IntTreeT::Ptr idxTree() const { return mIdxTree; }

    //////////

    SignData(SignData&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(const SignData& rhs)
    {
        mSignTree->merge(*rhs.mSignTree);
        mIdxTree->merge(*rhs.mIdxTree);
    }

private:

    const TreeT& mDistTree;
    AccessorT mDistAcc;

    const LeafManagerT& mLeafs;
    ValueT mIsovalue;

    typename Int16TreeT::Ptr mSignTree;
    Int16AccessorT mSignAcc;

    typename IntTreeT::Ptr mIdxTree;
    IntAccessorT mIdxAcc;

};


template<typename TreeT, typename LeafManagerT>
SignData<TreeT, LeafManagerT>::SignData(const TreeT& distTree,
    const LeafManagerT& leafs, ValueT iso)
    : mDistTree(distTree)
    , mDistAcc(mDistTree)
    , mLeafs(leafs)
    , mIsovalue(iso)
    , mSignTree(new Int16TreeT(0))
    , mSignAcc(*mSignTree)
    , mIdxTree(new IntTreeT(int(util::INVALID_IDX)))
    , mIdxAcc(*mIdxTree)
{
}


template<typename TreeT, typename LeafManagerT>
SignData<TreeT, LeafManagerT>::SignData(SignData& rhs, tbb::split)
    : mDistTree(rhs.mDistTree)
    , mDistAcc(mDistTree)
    , mLeafs(rhs.mLeafs)
    , mIsovalue(rhs.mIsovalue)
    , mSignTree(new Int16TreeT(0))
    , mSignAcc(*mSignTree)
    , mIdxTree(new IntTreeT(int(util::INVALID_IDX)))
    , mIdxAcc(*mIdxTree)
{
}


template<typename TreeT, typename LeafManagerT>
void
SignData<TreeT, LeafManagerT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mLeafs.getRange(), *this);
    else (*this)(mLeafs.getRange());
}

template<typename TreeT, typename LeafManagerT>
void
SignData<TreeT, LeafManagerT>::operator()(const tbb::blocked_range<size_t>& range)
{
    typedef typename Int16TreeT::LeafNodeType Int16LeafT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typename LeafManagerT::TreeType::LeafNodeType::ValueOnCIter iter;
    unsigned char signs, face;
    Coord ijk, coord;

    typename internal::UniquePtr<Int16LeafT>::type signLeafPt(new Int16LeafT(ijk, 0));

    for (size_t n = range.begin(); n != range.end(); ++n) {

        bool collectedData = false;

        coord = mLeafs.leaf(n).origin();

        if (!signLeafPt.get()) signLeafPt.reset(new Int16LeafT(coord, 0));
        else signLeafPt->setOrigin(coord);

        const typename TreeT::LeafNodeType *leafPt = mDistAcc.probeConstLeaf(coord);

        coord.offset(TreeT::LeafNodeType::DIM - 1);

        for (iter = mLeafs.leaf(n).cbeginValueOn(); iter; ++iter) {

            ijk = iter.getCoord();

            if (leafPt && ijk[0] < coord[0] && ijk[1] < coord[1] && ijk[2] < coord[2]) {
                signs = evalCellSigns(*leafPt, iter.pos(), mIsovalue);
            } else {
                signs = evalCellSigns(mDistAcc, ijk, mIsovalue);
            }

            if (signs != 0 && signs != 0xFF) {
                Int16 flags = (signs & 0x1) ? INSIDE : 0;

                if (bool(signs & 0x1) != bool(signs & 0x2)) flags |= XEDGE;
                if (bool(signs & 0x1) != bool(signs & 0x10)) flags |= YEDGE;
                if (bool(signs & 0x1) != bool(signs & 0x8)) flags |= ZEDGE;

                face = internal::sAmbiguousFace[signs];
                if (face != 0) correctCellSigns(signs, face, mDistAcc, ijk, mIsovalue);

                flags = Int16(flags | Int16(signs));

                signLeafPt->setValue(ijk, flags);
                collectedData = true;
            }
        }

        if (collectedData) {

            IntLeafT* idxLeaf = mIdxAcc.touchLeaf(coord);
            idxLeaf->topologyUnion(*signLeafPt);
            typename IntLeafT::ValueOnIter it = idxLeaf->beginValueOn();
            for (; it; ++it) {
                it.setValue(0);
            }

            mSignAcc.addLeaf(signLeafPt.release());
        }
    }
}


////////////////////////////////////////


/// @brief Counts the total number of points per leaf, accounts for cells with multiple points.
class CountPoints
{
public:
    CountPoints(std::vector<size_t>& pointList) : mPointList(pointList) {}

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t leafIndex) const
    {
        size_t points = 0;

        typename LeafNodeType::ValueOnCIter iter = leaf.cbeginValueOn();
        for (; iter; ++iter) {
            points += size_t(sEdgeGroupTable[(SIGNS & iter.getValue())][0]);
        }

        mPointList[leafIndex] = points;
    }

private:
    std::vector<size_t>& mPointList;
};


/// @brief Computes the point list indices for the index tree.
template<typename Int16TreeT>
class MapPoints
{
public:
    typedef tree::ValueAccessor<const Int16TreeT> Int16AccessorT;

    MapPoints(std::vector<size_t>& pointList, const Int16TreeT& signTree)
        : mPointList(pointList)
        , mSignAcc(signTree)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t leafIndex) const
    {
        size_t ptnIdx = mPointList[leafIndex];
        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();

        const typename Int16TreeT::LeafNodeType *signLeafPt =
            mSignAcc.probeConstLeaf(leaf.origin());

        for (; iter; ++iter) {
            iter.setValue(static_cast<typename LeafNodeType::ValueType>(ptnIdx));
            unsigned signs = SIGNS & signLeafPt->getValue(iter.pos());
            ptnIdx += size_t(sEdgeGroupTable[signs][0]);
        }
    }

private:
    std::vector<size_t>& mPointList;
    Int16AccessorT mSignAcc;
};


/// @brief Counts the total number of points per collapsed region
template<typename IntTreeT>
class CountRegions
{
public:
    typedef tree::ValueAccessor<IntTreeT> IntAccessorT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;

    CountRegions(IntTreeT& idxTree, std::vector<size_t>& regions)
    : mIdxAcc(idxTree)
    , mRegions(regions)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t leafIndex) const
    {

        size_t regions = 0;

        IntLeafT tmpLeaf(*mIdxAcc.probeConstLeaf(leaf.origin()));

        typename IntLeafT::ValueOnIter iter = tmpLeaf.beginValueOn();
        for (; iter; ++iter) {
            if(iter.getValue() == 0) {
                iter.setValueOff();
                regions += size_t(sEdgeGroupTable[(SIGNS & leaf.getValue(iter.pos()))][0]);
            }
        }

        int onVoxelCount = int(tmpLeaf.onVoxelCount());
        while (onVoxelCount > 0) {
            ++regions;
            iter = tmpLeaf.beginValueOn();
            int regionId = iter.getValue();
            for (; iter; ++iter) {
                if (iter.getValue() == regionId) {
                    iter.setValueOff();
                    --onVoxelCount;
                }
            }
        }

        mRegions[leafIndex] = regions;
    }

private:
    IntAccessorT mIdxAcc;
    std::vector<size_t>& mRegions;
};


////////////////////////////////////////


// @brief linear interpolation.
inline double evalRoot(double v0, double v1, double iso) { return (iso - v0) / (v1 - v0); }


/// @brief Extracts the eight corner values for leaf inclusive cells.
template<typename LeafT>
inline void
collectCornerValues(const LeafT& leaf, const Index offset, std::vector<double>& values)
{
    values[0] = double(leaf.getValue(offset)); // i, j, k
    values[3] = double(leaf.getValue(offset + 1)); // i, j, k+1
    values[4] = double(leaf.getValue(offset + LeafT::DIM)); // i, j+1, k
    values[7] = double(leaf.getValue(offset + LeafT::DIM + 1)); // i, j+1, k+1
    values[1] = double(leaf.getValue(offset + (LeafT::DIM * LeafT::DIM))); // i+1, j, k
    values[2] = double(leaf.getValue(offset + (LeafT::DIM * LeafT::DIM) + 1)); // i+1, j, k+1
    values[5] = double(leaf.getValue(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM)); // i+1, j+1, k
    values[6] = double(leaf.getValue(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM + 1)); // i+1, j+1, k+1
}


/// @brief Extracts the eight corner values for a cell starting at the given @ijk coordinate.
template<typename AccessorT>
inline void
collectCornerValues(const AccessorT& acc, const Coord& ijk, std::vector<double>& values)
{
    Coord coord = ijk;
    values[0] = double(acc.getValue(coord)); // i, j, k

    coord[0] += 1;
    values[1] = double(acc.getValue(coord)); // i+1, j, k

    coord[2] += 1;
    values[2] = double(acc.getValue(coord)); // i+i, j, k+1

    coord[0] = ijk[0];
    values[3] = double(acc.getValue(coord)); // i, j, k+1

    coord[1] += 1; coord[2] = ijk[2];
    values[4] = double(acc.getValue(coord)); // i, j+1, k

    coord[0] += 1;
    values[5] = double(acc.getValue(coord)); // i+1, j+1, k

    coord[2] += 1;
    values[6] = double(acc.getValue(coord)); // i+1, j+1, k+1

    coord[0] = ijk[0];
    values[7] = double(acc.getValue(coord)); // i, j+1, k+1
}


/// @brief Computes the average cell point for a given edge group.
inline Vec3d
computePoint(const std::vector<double>& values, unsigned char signs,
    unsigned char edgeGroup, double iso)
{
    Vec3d avg(0.0, 0.0, 0.0);
    int samples = 0;

    if (sEdgeGroupTable[signs][1] == edgeGroup) { // Edged: 0 - 1
        avg[0] += evalRoot(values[0], values[1], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][2] == edgeGroup) { // Edged: 1 - 2
        avg[0] += 1.0;
        avg[2] += evalRoot(values[1], values[2], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][3] == edgeGroup) { // Edged: 3 - 2
        avg[0] += evalRoot(values[3], values[2], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][4] == edgeGroup) { // Edged: 0 - 3
        avg[2] += evalRoot(values[0], values[3], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][5] == edgeGroup) { // Edged: 4 - 5
        avg[0] += evalRoot(values[4], values[5], iso);
        avg[1] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][6] == edgeGroup) { // Edged: 5 - 6
        avg[0] += 1.0;
        avg[1] += 1.0;
        avg[2] += evalRoot(values[5], values[6], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][7] == edgeGroup) { // Edged: 7 - 6
        avg[0] += evalRoot(values[7], values[6], iso);
        avg[1] += 1.0;
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][8] == edgeGroup) { // Edged: 4 - 7
        avg[1] += 1.0;
        avg[2] += evalRoot(values[4], values[7], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][9] == edgeGroup) { // Edged: 0 - 4
        avg[1] += evalRoot(values[0], values[4], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][10] == edgeGroup) { // Edged: 1 - 5
        avg[0] += 1.0;
        avg[1] += evalRoot(values[1], values[5], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][11] == edgeGroup) { // Edged: 2 - 6
        avg[0] += 1.0;
        avg[1] += evalRoot(values[2], values[6], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][12] == edgeGroup) { // Edged: 3 - 7
        avg[1] += evalRoot(values[3], values[7], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (samples > 1) {
        double w = 1.0 / double(samples);
        avg[0] *= w;
        avg[1] *= w;
        avg[2] *= w;
    }

    return avg;
}


/// @brief  Computes the average cell point for a given edge group, ignoring edge
///         samples present in the @c signsMask configuration.
inline int
computeMaskedPoint(Vec3d& avg, const std::vector<double>& values, unsigned char signs,
    unsigned char signsMask, unsigned char edgeGroup, double iso)
{
    avg = Vec3d(0.0, 0.0, 0.0);
    int samples = 0;

    if (sEdgeGroupTable[signs][1] == edgeGroup
        && sEdgeGroupTable[signsMask][1] == 0) { // Edged: 0 - 1
        avg[0] += evalRoot(values[0], values[1], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][2] == edgeGroup
        && sEdgeGroupTable[signsMask][2] == 0) { // Edged: 1 - 2
        avg[0] += 1.0;
        avg[2] += evalRoot(values[1], values[2], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][3] == edgeGroup
        && sEdgeGroupTable[signsMask][3] == 0) { // Edged: 3 - 2
        avg[0] += evalRoot(values[3], values[2], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][4] == edgeGroup
        && sEdgeGroupTable[signsMask][4] == 0) { // Edged: 0 - 3
        avg[2] += evalRoot(values[0], values[3], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][5] == edgeGroup
        && sEdgeGroupTable[signsMask][5] == 0) { // Edged: 4 - 5
        avg[0] += evalRoot(values[4], values[5], iso);
        avg[1] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][6] == edgeGroup
        && sEdgeGroupTable[signsMask][6] == 0) { // Edged: 5 - 6
        avg[0] += 1.0;
        avg[1] += 1.0;
        avg[2] += evalRoot(values[5], values[6], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][7] == edgeGroup
        && sEdgeGroupTable[signsMask][7] == 0) { // Edged: 7 - 6
        avg[0] += evalRoot(values[7], values[6], iso);
        avg[1] += 1.0;
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][8] == edgeGroup
        && sEdgeGroupTable[signsMask][8] == 0) { // Edged: 4 - 7
        avg[1] += 1.0;
        avg[2] += evalRoot(values[4], values[7], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][9] == edgeGroup
        && sEdgeGroupTable[signsMask][9] == 0) { // Edged: 0 - 4
        avg[1] += evalRoot(values[0], values[4], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][10] == edgeGroup
        && sEdgeGroupTable[signsMask][10] == 0) { // Edged: 1 - 5
        avg[0] += 1.0;
        avg[1] += evalRoot(values[1], values[5], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][11] == edgeGroup
        && sEdgeGroupTable[signsMask][11] == 0) { // Edged: 2 - 6
        avg[0] += 1.0;
        avg[1] += evalRoot(values[2], values[6], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][12] == edgeGroup
        && sEdgeGroupTable[signsMask][12] == 0) { // Edged: 3 - 7
        avg[1] += evalRoot(values[3], values[7], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (samples > 1) {
        double w = 1.0 / double(samples);
        avg[0] *= w;
        avg[1] *= w;
        avg[2] *= w;
    }

    return samples;
}


/// @brief  Computes the average cell point for a given edge group, by computing
///         convex weights based on the distance from the sample point @c p.
inline Vec3d
computeWeightedPoint(const Vec3d& p, const std::vector<double>& values,
    unsigned char signs, unsigned char edgeGroup, double iso)
{
    std::vector<Vec3d> samples;
    samples.reserve(8);

    std::vector<double> weights;
    weights.reserve(8);

    Vec3d avg(0.0, 0.0, 0.0);

    if (sEdgeGroupTable[signs][1] == edgeGroup) { // Edged: 0 - 1
        avg[0] = evalRoot(values[0], values[1], iso);
        avg[1] = 0.0;
        avg[2] = 0.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][2] == edgeGroup) { // Edged: 1 - 2
        avg[0] = 1.0;
        avg[1] = 0.0;
        avg[2] = evalRoot(values[1], values[2], iso);

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][3] == edgeGroup) { // Edged: 3 - 2
        avg[0] = evalRoot(values[3], values[2], iso);
        avg[1] = 0.0;
        avg[2] = 1.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][4] == edgeGroup) { // Edged: 0 - 3
        avg[0] = 0.0;
        avg[1] = 0.0;
        avg[2] = evalRoot(values[0], values[3], iso);

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][5] == edgeGroup) { // Edged: 4 - 5
        avg[0] = evalRoot(values[4], values[5], iso);
        avg[1] = 1.0;
        avg[2] = 0.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][6] == edgeGroup) { // Edged: 5 - 6
        avg[0] = 1.0;
        avg[1] = 1.0;
        avg[2] = evalRoot(values[5], values[6], iso);

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][7] == edgeGroup) { // Edged: 7 - 6
        avg[0] = evalRoot(values[7], values[6], iso);
        avg[1] = 1.0;
        avg[2] = 1.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][8] == edgeGroup) { // Edged: 4 - 7
        avg[0] = 0.0;
        avg[1] = 1.0;
        avg[2] = evalRoot(values[4], values[7], iso);

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][9] == edgeGroup) { // Edged: 0 - 4
        avg[0] = 0.0;
        avg[1] = evalRoot(values[0], values[4], iso);
        avg[2] = 0.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][10] == edgeGroup) { // Edged: 1 - 5
        avg[0] = 1.0;
        avg[1] = evalRoot(values[1], values[5], iso);
        avg[2] = 0.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][11] == edgeGroup) { // Edged: 2 - 6
        avg[0] = 1.0;
        avg[1] = evalRoot(values[2], values[6], iso);
        avg[2] = 1.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }

    if (sEdgeGroupTable[signs][12] == edgeGroup) { // Edged: 3 - 7
        avg[0] = 0.0;
        avg[1] = evalRoot(values[3], values[7], iso);
        avg[2] = 1.0;

        samples.push_back(avg);
        weights.push_back((avg-p).lengthSqr());
    }


    double minWeight = std::numeric_limits<double>::max();
    double maxWeight = -std::numeric_limits<double>::max();

    for (size_t i = 0, I = weights.size(); i < I; ++i) {
        minWeight = std::min(minWeight, weights[i]);
        maxWeight = std::max(maxWeight, weights[i]);
    }

    const double offset = maxWeight + minWeight * 0.1;
    for (size_t i = 0, I = weights.size(); i < I; ++i) {
        weights[i] = offset - weights[i];
    }


    double weightSum = 0.0;
    for (size_t i = 0, I = weights.size(); i < I; ++i) {
        weightSum += weights[i];
    }

    avg[0] = 0.0;
    avg[1] = 0.0;
    avg[2] = 0.0;

    if (samples.size() > 1) {
        for (size_t i = 0, I = samples.size(); i < I; ++i) {
            avg += samples[i] * (weights[i] / weightSum);
        }
    } else {
        avg = samples.front();
    }

    return avg;
}


/// @brief  Computes the average cell points defined by the sign configuration
///         @c signs and the given corner values @c values.
inline void
computeCellPoints(std::vector<Vec3d>& points,
    const std::vector<double>& values, unsigned char signs, double iso)
{
    for (size_t n = 1, N = sEdgeGroupTable[signs][0] + 1; n < N; ++n) {
        points.push_back(computePoint(values, signs, uint8_t(n), iso));
    }
}


/// @brief  Given a sign configuration @c lhsSigns and an edge group @c groupId,
///         finds the corresponding edge group in a different sign configuration
///         @c rhsSigns. Returns -1 if no match is found.
inline int
matchEdgeGroup(unsigned char groupId, unsigned char lhsSigns, unsigned char rhsSigns)
{
    int id = -1;
    for (size_t i = 1; i <= 12; ++i) {
        if (sEdgeGroupTable[lhsSigns][i] == groupId && sEdgeGroupTable[rhsSigns][i] != 0) {
            id = sEdgeGroupTable[rhsSigns][i];
            break;
        }
    }
    return id;
}


/// @brief  Computes the average cell points defined by the sign configuration
///         @c signs and the given corner values @c values. Combines data from
///         two different level sets to eliminate seam lines when meshing
///         fractured segments.
inline void
computeCellPoints(std::vector<Vec3d>& points, std::vector<bool>& weightedPointMask,
    const std::vector<double>& lhsValues, const std::vector<double>& rhsValues,
    unsigned char lhsSigns, unsigned char rhsSigns,
    double iso, size_t pointIdx, const boost::scoped_array<uint32_t>& seamPoints)
{
    for (size_t n = 1, N = sEdgeGroupTable[lhsSigns][0] + 1; n < N; ++n) {

        int id = matchEdgeGroup(uint8_t(n), lhsSigns, rhsSigns);

        if (id != -1) {

            const unsigned char e = uint8_t(id);
            uint32_t& quantizedPoint = seamPoints[pointIdx + (id - 1)];

            if ((quantizedPoint & MASK_DIRTY_BIT) && !(quantizedPoint & MASK_INVALID_BIT)) {
                Vec3d p = unpackPoint(quantizedPoint);
                points.push_back(computeWeightedPoint(p, rhsValues, rhsSigns, e, iso));
                weightedPointMask.push_back(true);
            } else {
                points.push_back(computePoint(rhsValues, rhsSigns, e, iso));
                weightedPointMask.push_back(false);
            }

        } else {
            points.push_back(computePoint(lhsValues, lhsSigns, uint8_t(n), iso));
            weightedPointMask.push_back(false);
        }
    }
}


template <typename TreeT, typename LeafManagerT>
class GenPoints
{
public:
    typedef tree::ValueAccessor<const TreeT> AccessorT;

    typedef typename TreeT::template ValueConverter<int>::Type IntTreeT;
    typedef tree::ValueAccessor<IntTreeT> IntAccessorT;
    typedef tree::ValueAccessor<const IntTreeT> IntCAccessorT;

    typedef typename TreeT::template ValueConverter<Int16>::Type Int16TreeT;
    typedef tree::ValueAccessor<const Int16TreeT> Int16CAccessorT;

    typedef boost::scoped_array<uint32_t> QuantizedPointList;

    //////////


    GenPoints(const LeafManagerT& signLeafs, const TreeT& distTree,
        IntTreeT& idxTree, PointList& points, std::vector<size_t>& indices,
        const math::Transform& xform, double iso);

    void run(bool threaded = true);

    void setRefData(const Int16TreeT* refSignTree = NULL, const TreeT* refDistTree = NULL,
        IntTreeT* refIdxTree = NULL, const QuantizedPointList* seamPoints = NULL,
        std::vector<unsigned char>* mSeamPointMaskPt = NULL);

    //////////


    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    const LeafManagerT& mSignLeafs;

    AccessorT mDistAcc;
    IntTreeT& mIdxTree;

    PointList& mPoints;
    std::vector<size_t>& mIndices;
    const math::Transform& mTransform;
    const double mIsovalue;

    // reference data
    const Int16TreeT *mRefSignTreePt;
    const TreeT* mRefDistTreePt;
    const IntTreeT* mRefIdxTreePt;
    const QuantizedPointList* mSeamPointsPt;
    std::vector<unsigned char>* mSeamPointMaskPt;
};


template <typename TreeT, typename LeafManagerT>
GenPoints<TreeT, LeafManagerT>::GenPoints(const LeafManagerT& signLeafs,
    const TreeT& distTree, IntTreeT& idxTree, PointList& points,
    std::vector<size_t>& indices, const math::Transform& xform, double iso)
    : mSignLeafs(signLeafs)
    , mDistAcc(distTree)
    , mIdxTree(idxTree)
    , mPoints(points)
    , mIndices(indices)
    , mTransform(xform)
    , mIsovalue(iso)
    , mRefSignTreePt(NULL)
    , mRefDistTreePt(NULL)
    , mRefIdxTreePt(NULL)
    , mSeamPointsPt(NULL)
    , mSeamPointMaskPt(NULL)
{
}


template <typename TreeT, typename LeafManagerT>
void
GenPoints<TreeT, LeafManagerT>::run(bool threaded)
{
    if (threaded) tbb::parallel_for(mSignLeafs.getRange(), *this);
    else (*this)(mSignLeafs.getRange());
}


template <typename TreeT, typename LeafManagerT>
void
GenPoints<TreeT, LeafManagerT>::setRefData(
    const Int16TreeT *refSignTree,
    const TreeT *refDistTree,
    IntTreeT* refIdxTree,
    const QuantizedPointList* seamPoints,
    std::vector<unsigned char>* seamPointMask)
{
    mRefSignTreePt = refSignTree;
    mRefDistTreePt = refDistTree;
    mRefIdxTreePt = refIdxTree;
    mSeamPointsPt = seamPoints;
    mSeamPointMaskPt = seamPointMask;
}


template <typename TreeT, typename LeafManagerT>
void
GenPoints<TreeT, LeafManagerT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typename IntTreeT::LeafNodeType::ValueOnIter iter;
    unsigned char signs, refSigns;
    Index offset;
    Coord ijk, coord;
    std::vector<Vec3d> points(4);
    std::vector<bool> weightedPointMask(4);
    std::vector<double> values(8), refValues(8);


    IntAccessorT idxAcc(mIdxTree);

    // reference data accessors
    boost::scoped_ptr<Int16CAccessorT> refSignAcc;
    if (mRefSignTreePt) refSignAcc.reset(new Int16CAccessorT(*mRefSignTreePt));

    boost::scoped_ptr<IntCAccessorT> refIdxAcc;
    if (mRefIdxTreePt) refIdxAcc.reset(new IntCAccessorT(*mRefIdxTreePt));

    boost::scoped_ptr<AccessorT> refDistAcc;
    if (mRefDistTreePt) refDistAcc.reset(new AccessorT(*mRefDistTreePt));


    for (size_t n = range.begin(); n != range.end(); ++n) {

        coord = mSignLeafs.leaf(n).origin();

        const typename TreeT::LeafNodeType *leafPt = mDistAcc.probeConstLeaf(coord);
        typename IntTreeT::LeafNodeType *idxLeafPt = idxAcc.probeLeaf(coord);


        // reference data leafs
        const typename Int16TreeT::LeafNodeType *refSignLeafPt = NULL;
        if (refSignAcc) refSignLeafPt = refSignAcc->probeConstLeaf(coord);

        const typename IntTreeT::LeafNodeType *refIdxLeafPt = NULL;
        if (refIdxAcc) refIdxLeafPt = refIdxAcc->probeConstLeaf(coord);

        const typename TreeT::LeafNodeType *refDistLeafPt = NULL;
        if (refDistAcc) refDistLeafPt = refDistAcc->probeConstLeaf(coord);


        // generate cell points
        size_t ptnIdx = mIndices[n];
        coord.offset(TreeT::LeafNodeType::DIM - 1);



        for (iter = idxLeafPt->beginValueOn(); iter; ++iter) {

            if(iter.getValue() != 0) continue;

            iter.setValue(static_cast<typename IntTreeT::ValueType>(ptnIdx));
            iter.setValueOff();
            offset = iter.pos();
            ijk = iter.getCoord();

            const bool inclusiveCell = ijk[0] < coord[0] && ijk[1] < coord[1] && ijk[2] < coord[2];

            const Int16& flags = mSignLeafs.leaf(n).getValue(offset);
            signs    = uint8_t(SIGNS & flags);
            refSigns = 0;

            if ((flags & SEAM) && refSignLeafPt && refIdxLeafPt) {
                if (refSignLeafPt->isValueOn(offset)) {
                    refSigns = uint8_t(SIGNS & refSignLeafPt->getValue(offset));
                }
            }


            if (inclusiveCell) collectCornerValues(*leafPt, offset, values);
            else collectCornerValues(mDistAcc, ijk, values);


            points.clear();
            weightedPointMask.clear();

            if (refSigns == 0) {
                computeCellPoints(points, values, signs, mIsovalue);
            } else {

                if (inclusiveCell) collectCornerValues(*refDistLeafPt, offset, refValues);
                else collectCornerValues(*refDistAcc, ijk, refValues);

                computeCellPoints(points, weightedPointMask, values, refValues, signs, refSigns,
                    mIsovalue, refIdxLeafPt->getValue(offset), *mSeamPointsPt);
            }


            for (size_t i = 0, I = points.size(); i < I; ++i) {

                // offset by cell-origin
                points[i][0] += double(ijk[0]);
                points[i][1] += double(ijk[1]);
                points[i][2] += double(ijk[2]);


                points[i] = mTransform.indexToWorld(points[i]);

                mPoints[ptnIdx][0] = float(points[i][0]);
                mPoints[ptnIdx][1] = float(points[i][1]);
                mPoints[ptnIdx][2] = float(points[i][2]);

                if (mSeamPointMaskPt && !weightedPointMask.empty() && weightedPointMask[i]) {
                    (*mSeamPointMaskPt)[ptnIdx] = 1;
                }

                ++ptnIdx;
            }
        }

        // generate collapsed region points
        int onVoxelCount = int(idxLeafPt->onVoxelCount());
        while (onVoxelCount > 0) {

            iter = idxLeafPt->beginValueOn();
            int regionId = iter.getValue(), count = 0;

            Vec3d avg(0.0), point;

            for (; iter; ++iter) {
                if (iter.getValue() != regionId) continue;

                iter.setValue(static_cast<typename IntTreeT::ValueType>(ptnIdx));
                iter.setValueOff();
                --onVoxelCount;

                ijk = iter.getCoord();
                offset = iter.pos();

                signs = uint8_t(SIGNS & mSignLeafs.leaf(n).getValue(offset));

                if (ijk[0] < coord[0] && ijk[1] < coord[1] && ijk[2] < coord[2]) {
                    collectCornerValues(*leafPt, offset, values);
                } else {
                    collectCornerValues(mDistAcc, ijk, values);
                }

                points.clear();
                computeCellPoints(points, values, signs, mIsovalue);

                avg[0] += double(ijk[0]) + points[0][0];
                avg[1] += double(ijk[1]) + points[0][1];
                avg[2] += double(ijk[2]) + points[0][2];

                ++count;
            }


            if (count > 1) {
                double w = 1.0 / double(count);
                avg[0] *= w;
                avg[1] *= w;
                avg[2] *= w;
            }

            avg = mTransform.indexToWorld(avg);

            mPoints[ptnIdx][0] = float(avg[0]);
            mPoints[ptnIdx][1] = float(avg[1]);
            mPoints[ptnIdx][2] = float(avg[2]);

            ++ptnIdx;
        }
    }
}


////////////////////////////////////////


template<typename TreeT>
class SeamWeights
{
public:
    typedef tree::ValueAccessor<const TreeT> AccessorT;

    typedef typename TreeT::template ValueConverter<int>::Type IntTreeT;
    typedef tree::ValueAccessor<const IntTreeT> IntAccessorT;

    typedef typename TreeT::template ValueConverter<Int16>::Type Int16TreeT;
    typedef tree::ValueAccessor<const Int16TreeT> Int16AccessorT;

    typedef boost::scoped_array<uint32_t> QuantizedPointList;

    //////////

    SeamWeights(const TreeT& distTree, const Int16TreeT& refSignTree,
        IntTreeT& refIdxTree, QuantizedPointList& points, double iso);

    template <typename LeafNodeType>
    void operator()(LeafNodeType &signLeaf, size_t leafIndex) const;

private:
    AccessorT mDistAcc;
    Int16AccessorT mRefSignAcc;
    IntAccessorT mRefIdxAcc;

    QuantizedPointList& mPoints;
    const double mIsovalue;
};


template<typename TreeT>
SeamWeights<TreeT>::SeamWeights(const TreeT& distTree, const Int16TreeT& refSignTree,
    IntTreeT& refIdxTree, QuantizedPointList& points, double iso)
    : mDistAcc(distTree)
    , mRefSignAcc(refSignTree)
    , mRefIdxAcc(refIdxTree)
    , mPoints(points)
    , mIsovalue(iso)
{
}


template<typename TreeT>
template <typename LeafNodeType>
void
SeamWeights<TreeT>::operator()(LeafNodeType &signLeaf, size_t /*leafIndex*/) const
{
    Coord coord = signLeaf.origin();
    const typename Int16TreeT::LeafNodeType *refSignLeafPt = mRefSignAcc.probeConstLeaf(coord);

    if (!refSignLeafPt) return;

    const typename TreeT::LeafNodeType *distLeafPt = mDistAcc.probeConstLeaf(coord);
    const typename IntTreeT::LeafNodeType *refIdxLeafPt = mRefIdxAcc.probeConstLeaf(coord);

    std::vector<double> values(8);
    unsigned char lhsSigns, rhsSigns;
    Vec3d point;
    Index offset;

    Coord ijk;
    coord.offset(TreeT::LeafNodeType::DIM - 1);

    typename LeafNodeType::ValueOnCIter iter = signLeaf.cbeginValueOn();
    for (; iter; ++iter) {

        offset = iter.pos();
        ijk = iter.getCoord();

        const bool inclusiveCell = ijk[0] < coord[0] && ijk[1] < coord[1] && ijk[2] < coord[2];

        if ((iter.getValue() & SEAM) && refSignLeafPt->isValueOn(offset)) {

            lhsSigns = uint8_t(SIGNS & iter.getValue());
            rhsSigns = uint8_t(SIGNS & refSignLeafPt->getValue(offset));


            if (inclusiveCell) {
                collectCornerValues(*distLeafPt, offset, values);
            } else {
                collectCornerValues(mDistAcc, ijk, values);
            }


            for (size_t n = 1, N = sEdgeGroupTable[lhsSigns][0] + 1; n < N; ++n) {

                int id = matchEdgeGroup(uint8_t(n), lhsSigns, rhsSigns);

                if (id != -1) {

                    uint32_t& data = mPoints[refIdxLeafPt->getValue(offset) + (id - 1)];

                    if (!(data & MASK_DIRTY_BIT)) {

                        int smaples = computeMaskedPoint(
                            point, values, lhsSigns, rhsSigns, uint8_t(n), mIsovalue);

                        if (smaples > 0) data = packPoint(point);
                        else data = MASK_INVALID_BIT;

                        data |= MASK_DIRTY_BIT;
                    }
                }
            }
        }
    }
}


////////////////////////////////////////


template <typename TreeT, typename LeafManagerT>
class MergeVoxelRegions
{
public:
    typedef typename TreeT::ValueType ValueT;
    typedef tree::ValueAccessor<const TreeT> AccessorT;

    typedef typename TreeT::template ValueConverter<int>::Type IntTreeT;
    typedef tree::ValueAccessor<IntTreeT> IntAccessorT;

    typedef typename TreeT::template ValueConverter<bool>::Type BoolTreeT;

    typedef typename LeafManagerT::TreeType::template ValueConverter<Int16>::Type Int16TreeT;
    typedef tree::ValueAccessor<const Int16TreeT> Int16AccessorT;

    typedef typename TreeT::template ValueConverter<float>::Type FloatTreeT;
    typedef Grid<FloatTreeT> FloatGridT;


    //////////

    MergeVoxelRegions(const LeafManagerT& signLeafs, const Int16TreeT& signTree,
        const TreeT& distTree, IntTreeT& idxTree, ValueT iso, ValueT adaptivity);

    void run(bool threaded = true);

    void setSpatialAdaptivity(
        const math::Transform& distGridXForm, const FloatGridT& adaptivityField);

    void setAdaptivityMask(const BoolTreeT* mask);

    void setRefData(const Int16TreeT* signTree, ValueT adaptivity);

    //////////


    void operator()(const tbb::blocked_range<size_t>&) const;

private:

    const LeafManagerT& mSignLeafs;

    const Int16TreeT& mSignTree;
    Int16AccessorT mSignAcc;

    const TreeT& mDistTree;
    AccessorT mDistAcc;

    IntTreeT& mIdxTree;
    ValueT mIsovalue, mSurfaceAdaptivity, mInternalAdaptivity;

    const math::Transform* mTransform;
    const FloatGridT* mAdaptivityGrid;
    const BoolTreeT* mAdaptivityMask;

    const Int16TreeT* mRefSignTree;
};


template <typename TreeT, typename LeafManagerT>
MergeVoxelRegions<TreeT, LeafManagerT>::MergeVoxelRegions(
    const LeafManagerT& signLeafs, const Int16TreeT& signTree,
    const TreeT& distTree, IntTreeT& idxTree, ValueT iso, ValueT adaptivity)
    : mSignLeafs(signLeafs)
    , mSignTree(signTree)
    , mSignAcc(mSignTree)
    , mDistTree(distTree)
    , mDistAcc(mDistTree)
    , mIdxTree(idxTree)
    , mIsovalue(iso)
    , mSurfaceAdaptivity(adaptivity)
    , mInternalAdaptivity(adaptivity)
    , mTransform(NULL)
    , mAdaptivityGrid(NULL)
    , mAdaptivityMask(NULL)
    , mRefSignTree(NULL)
{
}


template <typename TreeT, typename LeafManagerT>
void
MergeVoxelRegions<TreeT, LeafManagerT>::run(bool threaded)
{
    if (threaded) tbb::parallel_for(mSignLeafs.getRange(), *this);
    else (*this)(mSignLeafs.getRange());
}


template <typename TreeT, typename LeafManagerT>
void
MergeVoxelRegions<TreeT, LeafManagerT>::setSpatialAdaptivity(
    const math::Transform& distGridXForm, const FloatGridT& adaptivityField)
{
    mTransform = &distGridXForm;
    mAdaptivityGrid = &adaptivityField;
}


template <typename TreeT, typename LeafManagerT>
void
MergeVoxelRegions<TreeT, LeafManagerT>::setAdaptivityMask(const BoolTreeT* mask)
{
    mAdaptivityMask = mask;
}

template <typename TreeT, typename LeafManagerT>
void
MergeVoxelRegions<TreeT, LeafManagerT>::setRefData(const Int16TreeT* signTree, ValueT adaptivity)
{
    mRefSignTree = signTree;
    mInternalAdaptivity = adaptivity;
}


template <typename TreeT, typename LeafManagerT>
void
MergeVoxelRegions<TreeT, LeafManagerT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typedef math::Vec3<ValueT> Vec3T;

    typedef typename TreeT::LeafNodeType LeafT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    typedef typename LeafT::template ValueConverter<Vec3T>::Type Vec3LeafT;

    const int LeafDim = LeafT::DIM;

    IntAccessorT idxAcc(mIdxTree);

    typename LeafManagerT::TreeType::LeafNodeType::ValueOnCIter iter;

    typedef typename tree::ValueAccessor<const FloatTreeT> FloatTreeCAccessorT;
    boost::scoped_ptr<FloatTreeCAccessorT> adaptivityAcc;
    if (mAdaptivityGrid) {
        adaptivityAcc.reset(new FloatTreeCAccessorT(mAdaptivityGrid->tree()));
    }

    typedef typename tree::ValueAccessor<const Int16TreeT> Int16TreeCAccessorT;
    boost::scoped_ptr<Int16TreeCAccessorT> refAcc;
    if (mRefSignTree) {
        refAcc.reset(new Int16TreeCAccessorT(*mRefSignTree));
    }

    typedef typename tree::ValueAccessor<const BoolTreeT> BoolTreeCAccessorT;
    boost::scoped_ptr<BoolTreeCAccessorT> maskAcc;
    if (mAdaptivityMask) {
        maskAcc.reset(new BoolTreeCAccessorT(*mAdaptivityMask));
    }


    BoolLeafT mask;
    Vec3LeafT gradients;
    Coord ijk, end;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        mask.setValuesOff();

        const Coord& origin = mSignLeafs.leaf(n).origin();

        ValueT adaptivity = (refAcc && !refAcc->probeConstLeaf(origin)) ?
            mInternalAdaptivity : mSurfaceAdaptivity;

        IntLeafT& idxLeaf = *idxAcc.probeLeaf(origin);

        end[0] = origin[0] + LeafDim;
        end[1] = origin[1] + LeafDim;
        end[2] = origin[2] + LeafDim;

        // Mask off seam line adjacent voxels
        if (maskAcc) {
            const BoolLeafT* maskLeaf = maskAcc->probeConstLeaf(origin);
            if (maskLeaf != NULL) {
                typename BoolLeafT::ValueOnCIter it;
                for (it = maskLeaf->cbeginValueOn(); it; ++it) {
                    mask.setActiveState(it.getCoord() & ~1u, true);
                }
            }
        }

        // Set region adaptivity
        LeafT adaptivityLeaf(origin, adaptivity);
        if (mAdaptivityGrid) {
            for (Index offset = 0; offset < LeafT::NUM_VALUES; ++offset) {
                ijk = adaptivityLeaf.offsetToGlobalCoord(offset);
                Vec3d xyz = mAdaptivityGrid->transform().worldToIndex(
                    mTransform->indexToWorld(ijk));
                ValueT tmpA = ValueT(adaptivityAcc->getValue(util::nearestCoord(xyz)));
                adaptivityLeaf.setValueOnly(offset, tmpA * adaptivity);
            }
        }

        // Mask off ambiguous voxels
        for (iter = mSignLeafs.leaf(n).cbeginValueOn(); iter; ++iter) {
            unsigned char signs = static_cast<unsigned char>(SIGNS & int(iter.getValue()));
            if (!sAdaptable[signs] || sEdgeGroupTable[signs][0] > 1) {
                mask.setActiveState(iter.getCoord() & ~1u, true);
            }
        }

        // Mask off topologically ambiguous 2x2x2 voxel sub-blocks
        int dim = 2;
        for (ijk[0] = origin[0]; ijk[0] < end[0]; ijk[0] += dim) {
            for (ijk[1] = origin[1]; ijk[1] < end[1]; ijk[1] += dim) {
                for (ijk[2] = origin[2]; ijk[2] < end[2]; ijk[2] += dim) {
                    if (!mask.isValueOn(ijk) & isNonManifold(mDistAcc, ijk, mIsovalue, dim)) {
                        mask.setActiveState(ijk, true);
                    }
                }
            }
        }

        // Compute the gradient for the remaining voxels
        gradients.setValuesOff();
        for (iter = mSignLeafs.leaf(n).cbeginValueOn(); iter; ++iter) {
            ijk = iter.getCoord();
            if(!mask.isValueOn(ijk & ~1u)) {
                Vec3T dir(math::ISGradient<math::CD_2ND>::result(mDistAcc, ijk));
                dir.normalize();
                gradients.setValueOn(iter.pos(), dir);
            }
        }

        // Merge regions
        int regionId = 1;
        for ( ; dim <= LeafDim; dim = dim << 1) {
            const unsigned coordMask = ~((dim << 1) - 1);
            for (ijk[0] = origin[0]; ijk[0] < end[0]; ijk[0] += dim) {
                for (ijk[1] = origin[1]; ijk[1] < end[1]; ijk[1] += dim) {
                    for (ijk[2] = origin[2]; ijk[2] < end[2]; ijk[2] += dim) {

                        adaptivity = adaptivityLeaf.getValue(ijk);

                        if (mask.isValueOn(ijk) || isNonManifold(mDistAcc, ijk, mIsovalue, dim)
                            || !isMergable(gradients, ijk, dim, adaptivity)) {
                            mask.setActiveState(ijk & coordMask, true);
                        } else {
                            mergeVoxels(idxLeaf, ijk, dim, regionId++);
                        }
                    }
                }
            }
        }
    }
}


////////////////////////////////////////


// Constructs qudas
struct UniformPrimBuilder
{
    UniformPrimBuilder(): mIdx(0), mPolygonPool(NULL) {}

    void init(const size_t upperBound, PolygonPool& quadPool)
    {
        mPolygonPool = &quadPool;
        mPolygonPool->resetQuads(upperBound);
        mIdx = 0;
    }

    void addPrim(const Vec4I& verts, bool reverse, char flags = 0)
    {
        if (!reverse) {
            mPolygonPool->quad(mIdx) = verts;
        } else {
            Vec4I& quad = mPolygonPool->quad(mIdx);
            quad[0] = verts[3];
            quad[1] = verts[2];
            quad[2] = verts[1];
            quad[3] = verts[0];
        }
        mPolygonPool->quadFlags(mIdx) = flags;
        ++mIdx;
    }

    void done()
    {
        mPolygonPool->trimQuads(mIdx);
    }

private:
    size_t mIdx;
    PolygonPool* mPolygonPool;
};


// Constructs qudas and triangles
struct AdaptivePrimBuilder
{
    AdaptivePrimBuilder() : mQuadIdx(0), mTriangleIdx(0), mPolygonPool(NULL) {}

    void init(const size_t upperBound, PolygonPool& polygonPool)
    {
        mPolygonPool = &polygonPool;
        mPolygonPool->resetQuads(upperBound);
        mPolygonPool->resetTriangles(upperBound);

        mQuadIdx = 0;
        mTriangleIdx = 0;
    }

    void addPrim(const Vec4I& verts, bool reverse, char flags = 0)
    {
        if (verts[0] != verts[1] && verts[0] != verts[2] && verts[0] != verts[3]
            && verts[1] != verts[2] && verts[1] != verts[3] && verts[2] != verts[3]) {
            mPolygonPool->quadFlags(mQuadIdx) = flags;
            addQuad(verts, reverse);
        } else if (
            verts[0] == verts[3] &&
            verts[1] != verts[2] &&
            verts[1] != verts[0] &&
            verts[2] != verts[0]) {
            mPolygonPool->triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[1], verts[2], reverse);
        } else if (
            verts[1] == verts[2] &&
            verts[0] != verts[3] &&
            verts[0] != verts[1] &&
            verts[3] != verts[1]) {
            mPolygonPool->triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[1], verts[3], reverse);
        } else if (
            verts[0] == verts[1] &&
            verts[2] != verts[3] &&
            verts[2] != verts[0] &&
            verts[3] != verts[0]) {
            mPolygonPool->triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[2], verts[3], reverse);
        } else if (
            verts[2] == verts[3] &&
            verts[0] != verts[1] &&
            verts[0] != verts[2] &&
            verts[1] != verts[2]) {
            mPolygonPool->triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[1], verts[2], reverse);
        }
    }


    void done()
    {
        mPolygonPool->trimQuads(mQuadIdx, /*reallocate=*/true);
        mPolygonPool->trimTrinagles(mTriangleIdx, /*reallocate=*/true);
    }

private:

    void addQuad(const Vec4I& verts, bool reverse)
    {
        if (!reverse) {
            mPolygonPool->quad(mQuadIdx) = verts;
        } else {
            Vec4I& quad = mPolygonPool->quad(mQuadIdx);
            quad[0] = verts[3];
            quad[1] = verts[2];
            quad[2] = verts[1];
            quad[3] = verts[0];
        }
        ++mQuadIdx;
    }

    void addTriangle(unsigned v0, unsigned v1, unsigned v2, bool reverse)
    {
        Vec3I& prim = mPolygonPool->triangle(mTriangleIdx);

        prim[1] = v1;

        if (!reverse) {
            prim[0] = v0;
            prim[2] = v2;
        } else {
            prim[0] = v2;
            prim[2] = v0;
        }
        ++mTriangleIdx;
    }

    size_t mQuadIdx, mTriangleIdx;
    PolygonPool *mPolygonPool;
};


template<typename SignAccT, typename IdxAccT, typename PrimBuilder>
inline void
constructPolygons(Int16 flags, Int16 refFlags, const Vec4i& offsets, const Coord& ijk,
    const SignAccT& signAcc, const IdxAccT& idxAcc, PrimBuilder& mesher, Index32 pointListSize)
{
    const Index32 v0 = idxAcc.getValue(ijk);
    if (v0 == util::INVALID_IDX) return;

    char tag[2];
    tag[0] = (flags & SEAM) ? POLYFLAG_FRACTURE_SEAM : 0;
    tag[1] = tag[0] | char(POLYFLAG_EXTERIOR);

    const bool isInside = flags & INSIDE;

    Coord coord;
    openvdb::Vec4I quad;
    unsigned char cell;
    Index32 tmpIdx = 0;

    if (flags & XEDGE) {

        quad[0] = v0 + offsets[0];

        // i, j-1, k
        coord[0] = ijk[0];
        coord[1] = ijk[1] - 1;
        coord[2] = ijk[2];

        quad[1] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[1] + Index32(sEdgeGroupTable[cell][5] - 1);
            if (tmpIdx < pointListSize) quad[1] = tmpIdx;
        }

        // i, j-1, k-1
        coord[2] -= 1;

        quad[2] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[2] + Index32(sEdgeGroupTable[cell][7] - 1);
            if (tmpIdx < pointListSize) quad[2] = tmpIdx;
        }

        // i, j, k-1
        coord[1] = ijk[1];

        quad[3] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[3] + Index32(sEdgeGroupTable[cell][3] - 1);
            if (tmpIdx < pointListSize) quad[3] = tmpIdx;
        }

        if (quad[1] != util::INVALID_IDX &&
            quad[2] != util::INVALID_IDX && quad[3] != util::INVALID_IDX) {
            mesher.addPrim(quad, isInside, tag[bool(refFlags & XEDGE)]);
        }
    }


    if (flags & YEDGE) {

        quad[0] = v0 + offsets[1];

        // i, j, k-1
        coord[0] = ijk[0];
        coord[1] = ijk[1];
        coord[2] = ijk[2] - 1;

        quad[1] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[1] + Index32(sEdgeGroupTable[cell][12] - 1);
            if (tmpIdx < pointListSize) quad[1] = tmpIdx;
        }

        // i-1, j, k-1
        coord[0] -= 1;

        quad[2] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[2] + Index32(sEdgeGroupTable[cell][11] - 1);
            if (tmpIdx < pointListSize) quad[2] = tmpIdx;
        }

        // i-1, j, k
        coord[2] = ijk[2];

        quad[3] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[3] + Index32(sEdgeGroupTable[cell][10] - 1);
            if (tmpIdx < pointListSize) quad[3] = tmpIdx;
        }

        if (quad[1] != util::INVALID_IDX &&
            quad[2] != util::INVALID_IDX && quad[3] != util::INVALID_IDX) {
            mesher.addPrim(quad, isInside, tag[bool(refFlags & YEDGE)]);
        }
    }

    if (flags & ZEDGE) {

        quad[0] = v0 + offsets[2];

        // i, j-1, k
        coord[0] = ijk[0];
        coord[1] = ijk[1] - 1;
        coord[2] = ijk[2];

        quad[1] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[1] + Index32(sEdgeGroupTable[cell][8] - 1);
            if (tmpIdx < pointListSize) quad[1] = tmpIdx;
        }

        // i-1, j-1, k
        coord[0] -= 1;

        quad[2] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[2] + Index32(sEdgeGroupTable[cell][6] - 1);
            if (tmpIdx < pointListSize) quad[2] = tmpIdx;
        }

        // i-1, j, k
        coord[1] = ijk[1];

        quad[3] = idxAcc.getValue(coord);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        if (sEdgeGroupTable[cell][0] > 1) {
            tmpIdx = quad[3] + Index32(sEdgeGroupTable[cell][2] - 1);
            if (tmpIdx < pointListSize) quad[3] = tmpIdx;
        }

        if (quad[1] != util::INVALID_IDX &&
            quad[2] != util::INVALID_IDX && quad[3] != util::INVALID_IDX) {
            mesher.addPrim(quad, !isInside, tag[bool(refFlags & ZEDGE)]);
        }
    }
}


////////////////////////////////////////


template<typename LeafManagerT, typename PrimBuilder>
class GenPolygons
{
public:
    typedef typename LeafManagerT::TreeType::template ValueConverter<int>::Type IntTreeT;
    typedef typename LeafManagerT::TreeType::template ValueConverter<Int16>::Type Int16TreeT;

    typedef tree::ValueAccessor<const IntTreeT> IntAccessorT;
    typedef tree::ValueAccessor<const Int16TreeT> Int16AccessorT;

    //////////


    GenPolygons(const LeafManagerT& signLeafs, const Int16TreeT& signTree,
        const IntTreeT& idxTree, PolygonPoolList& polygons, Index32 pointListSize);

    void run(bool threaded = true);


    void setRefSignTree(const Int16TreeT *r) { mRefSignTree = r; }

    //////////


    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    const LeafManagerT& mSignLeafs;
    const Int16TreeT& mSignTree;
    const IntTreeT& mIdxTree;
    const PolygonPoolList& mPolygonPoolList;
    const Index32 mPointListSize;

    const Int16TreeT *mRefSignTree;
 };


template<typename LeafManagerT, typename PrimBuilder>
GenPolygons<LeafManagerT, PrimBuilder>::GenPolygons(const LeafManagerT& signLeafs,
    const Int16TreeT& signTree, const IntTreeT& idxTree, PolygonPoolList& polygons,
    Index32 pointListSize)
    : mSignLeafs(signLeafs)
    , mSignTree(signTree)
    , mIdxTree(idxTree)
    , mPolygonPoolList(polygons)
    , mPointListSize(pointListSize)
    , mRefSignTree(NULL)
{
}

template<typename LeafManagerT, typename PrimBuilder>
void
GenPolygons<LeafManagerT, PrimBuilder>::run(bool threaded)
{
    if (threaded) tbb::parallel_for(mSignLeafs.getRange(), *this);
    else (*this)(mSignLeafs.getRange());
}

template<typename LeafManagerT, typename PrimBuilder>
void
GenPolygons<LeafManagerT, PrimBuilder>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    typename LeafManagerT::TreeType::LeafNodeType::ValueOnCIter iter;
    IntAccessorT idxAcc(mIdxTree);
    Int16AccessorT signAcc(mSignTree);


    PrimBuilder mesher;
    size_t edgeCount;
    Coord ijk, origin;


    // reference data
    boost::scoped_ptr<Int16AccessorT> refSignAcc;
    if (mRefSignTree) refSignAcc.reset(new Int16AccessorT(*mRefSignTree));


    for (size_t n = range.begin(); n != range.end(); ++n) {

        origin = mSignLeafs.leaf(n).origin();

        // Get an upper bound on the number of primitives.
        edgeCount = 0;
        iter = mSignLeafs.leaf(n).cbeginValueOn();
        for (; iter; ++iter) {
            if (iter.getValue() & XEDGE) ++edgeCount;
            if (iter.getValue() & YEDGE) ++edgeCount;
            if (iter.getValue() & ZEDGE) ++edgeCount;
        }

        if(edgeCount == 0) continue;

        mesher.init(edgeCount, mPolygonPoolList[n]);

        const typename Int16TreeT::LeafNodeType *signleafPt = signAcc.probeConstLeaf(origin);
        const typename IntTreeT::LeafNodeType *idxLeafPt = idxAcc.probeConstLeaf(origin);

        if (!signleafPt || !idxLeafPt) continue;


        const typename Int16TreeT::LeafNodeType *refSignLeafPt = NULL;
        if (refSignAcc) refSignLeafPt = refSignAcc->probeConstLeaf(origin);

        Vec4i offsets;

        iter = mSignLeafs.leaf(n).cbeginValueOn();
        for (; iter; ++iter) {
            ijk = iter.getCoord();

            Int16 flags = iter.getValue();

            if (!(flags & 0xE00)) continue;

            Int16 refFlags = 0;
            if (refSignLeafPt) {
                refFlags = refSignLeafPt->getValue(iter.pos());
            }

            offsets[0] = 0;
            offsets[1] = 0;
            offsets[2] = 0;

            const unsigned char cell = uint8_t(SIGNS & flags);

            if (sEdgeGroupTable[cell][0] > 1) {
                offsets[0] = (sEdgeGroupTable[cell][1] - 1);
                offsets[1] = (sEdgeGroupTable[cell][9] - 1);
                offsets[2] = (sEdgeGroupTable[cell][4] - 1);
            }

            if (ijk[0] > origin[0] && ijk[1] > origin[1] && ijk[2] > origin[2]) {
                constructPolygons(flags, refFlags, offsets, ijk,
                    *signleafPt, *idxLeafPt, mesher, mPointListSize);
            } else {
                constructPolygons(flags, refFlags, offsets, ijk,
                    signAcc, idxAcc, mesher, mPointListSize);
            }
        }

        mesher.done();
    }
}


////////////////////////////////////////

// Masking and mesh partitioning

struct PartOp
{

    PartOp(size_t leafCount, size_t partitions, size_t activePart)
    {
        size_t leafSegments = leafCount / partitions;
        mStart = leafSegments * activePart;
        mEnd = activePart >= (partitions - 1) ? leafCount : mStart + leafSegments;
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t leafIndex) const
    {
        if (leafIndex < mStart || leafIndex >= mEnd) leaf.setValuesOff();
    }

private:
    size_t mStart, mEnd;
};


////////////////////////////////////////


template<typename SrcTreeT>
class PartGen
{
public:
    typedef tree::LeafManager<const SrcTreeT> LeafManagerT;
    typedef typename SrcTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef tree::ValueAccessor<BoolTreeT> BoolAccessorT;

    //////////


    PartGen(const LeafManagerT& leafs, size_t partitions, size_t activePart);

    void run(bool threaded = true);

    BoolTreeT&  tree() { return mTree; }


    //////////

    PartGen(PartGen&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(PartGen& rhs) { mTree.merge(rhs.mTree); }

private:
    const LeafManagerT& mLeafManager;
    BoolTreeT mTree;
    size_t mStart, mEnd;
};

template<typename SrcTreeT>
PartGen<SrcTreeT>::PartGen(const LeafManagerT& leafs, size_t partitions, size_t activePart)
    : mLeafManager(leafs)
    , mTree(false)
    , mStart(0)
    , mEnd(0)
{
    size_t leafCount = leafs.leafCount();
    size_t leafSegments = leafCount / partitions;
    mStart = leafSegments * activePart;
    mEnd = activePart >= (partitions - 1) ? leafCount : mStart + leafSegments;
}

template<typename SrcTreeT>
PartGen<SrcTreeT>::PartGen(PartGen& rhs, tbb::split)
    : mLeafManager(rhs.mLeafManager)
    , mTree(false)
    , mStart(rhs.mStart)
    , mEnd(rhs.mEnd)
{
}


template<typename SrcTreeT>
void
PartGen<SrcTreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mLeafManager.getRange(), *this);
    else (*this)(mLeafManager.getRange());
}


template<typename SrcTreeT>
void
PartGen<SrcTreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    Coord ijk;
    BoolAccessorT acc(mTree);

    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    typename SrcTreeT::LeafNodeType::ValueOnCIter iter;

    for (size_t n = range.begin(); n != range.end(); ++n) {
        if (n < mStart || n >= mEnd) continue;
        BoolLeafT* leaf = acc.touchLeaf(mLeafManager.leaf(n).origin());
        leaf->topologyUnion(mLeafManager.leaf(n));
    }
}


////////////////////////////////////////


template<typename TreeT, typename LeafManagerT>
class GenSeamMask
{
public:
    typedef typename TreeT::template ValueConverter<bool>::Type BoolTreeT;

    //////////

    GenSeamMask(const LeafManagerT& leafs, const TreeT& tree);

    void run(bool threaded = true);

    BoolTreeT& mask() { return mMaskTree; }

    //////////

    GenSeamMask(GenSeamMask&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(GenSeamMask& rhs) { mMaskTree.merge(rhs.mMaskTree); }

private:

    const LeafManagerT& mLeafManager;
    const TreeT& mTree;

    BoolTreeT mMaskTree;
};


template<typename TreeT, typename LeafManagerT>
GenSeamMask<TreeT, LeafManagerT>::GenSeamMask(const LeafManagerT& leafs, const TreeT& tree)
    : mLeafManager(leafs)
    , mTree(tree)
    , mMaskTree(false)
{
}


template<typename TreeT, typename LeafManagerT>
GenSeamMask<TreeT, LeafManagerT>::GenSeamMask(GenSeamMask& rhs, tbb::split)
    : mLeafManager(rhs.mLeafManager)
    , mTree(rhs.mTree)
    , mMaskTree(false)
{
}


template<typename TreeT, typename LeafManagerT>
void
GenSeamMask<TreeT, LeafManagerT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mLeafManager.getRange(), *this);
    else (*this)(mLeafManager.getRange());
}


template<typename TreeT, typename LeafManagerT>
void
GenSeamMask<TreeT, LeafManagerT>::operator()(const tbb::blocked_range<size_t>& range)
{
    Coord ijk;
    tree::ValueAccessor<const TreeT> acc(mTree);
    tree::ValueAccessor<BoolTreeT> maskAcc(mMaskTree);

    typename LeafManagerT::TreeType::LeafNodeType::ValueOnCIter it;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        it = mLeafManager.leaf(n).cbeginValueOn();

        for (; it; ++it) {

            ijk = it.getCoord();

            unsigned char rhsSigns = uint8_t(acc.getValue(ijk) & SIGNS);

            if (sEdgeGroupTable[rhsSigns][0] > 0) {
                unsigned char lhsSigns = uint8_t(it.getValue() & SIGNS);
                if (rhsSigns != lhsSigns) {
                    maskAcc.setValueOn(ijk);
                }
            }
        }
    }
}


////////////////////////////////////////


template<typename TreeT>
class TagSeamEdges
{
public:
    typedef tree::ValueAccessor<const TreeT> AccessorT;

    TagSeamEdges(const TreeT& tree) : mAcc(tree) {}

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t/*leafIndex*/) const
    {
        const typename TreeT::LeafNodeType *maskLeaf =
            mAcc.probeConstLeaf(leaf.origin());

        if (!maskLeaf) return;

        typename LeafNodeType::ValueOnIter it = leaf.beginValueOn();

        for (; it; ++it) {

            if (maskLeaf->isValueOn(it.pos())) {
                it.setValue(it.getValue() | SEAM);
            }
        }
    }

private:
    AccessorT mAcc;
};



template<typename BoolTreeT>
struct MaskEdges
{
    typedef tree::ValueAccessor<const BoolTreeT> BoolAccessorT;

    MaskEdges(const BoolTreeT& valueMask) : mMaskAcc(valueMask) {}

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t /*leafIndex*/) const
    {
        typename LeafNodeType::ValueOnIter it = leaf.beginValueOn();

        const typename BoolTreeT::LeafNodeType * maskLeaf =
            mMaskAcc.probeConstLeaf(leaf.origin());

        if (maskLeaf) {
            for (; it; ++it) {
                if (!maskLeaf->isValueOn(it.pos())) {
                    it.setValue(0x1FF & it.getValue());
                }
            }
        } else {
            for (; it; ++it) {
                it.setValue(0x1FF & it.getValue());
            }
        }
    }

private:
    BoolAccessorT mMaskAcc;
};


class FlagUsedPoints
{
public:
    //////////

    FlagUsedPoints(const PolygonPoolList& polygons, size_t polyListCount,
        std::vector<unsigned char>& usedPointMask)
        : mPolygons(polygons)
        , mPolyListCount(polyListCount)
        , mUsedPointMask(usedPointMask)
    {
    }

    void run(bool threaded = true)
    {
        if (threaded) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mPolyListCount), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, mPolyListCount));
        }
    }

    //////////

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        // Concurrent writes to same memory address can occur, but
        // all threads are writing the same value and char is atomic.
        for (size_t n = range.begin(); n != range.end(); ++n) {
            const PolygonPool& polygons = mPolygons[n];
            for (size_t i = 0; i < polygons.numQuads(); ++i) {
                const Vec4I& quad = polygons.quad(i);
                mUsedPointMask[quad[0]] = 1;
                mUsedPointMask[quad[1]] = 1;
                mUsedPointMask[quad[2]] = 1;
                mUsedPointMask[quad[3]] = 1;
            }

            for (size_t i = 0; i < polygons.numTriangles(); ++i) {
                const Vec3I& triangle = polygons.triangle(i);
                mUsedPointMask[triangle[0]] = 1;
                mUsedPointMask[triangle[1]] = 1;
                mUsedPointMask[triangle[2]] = 1;
            }
        }
    }


private:
    const PolygonPoolList& mPolygons;
    size_t mPolyListCount;
    std::vector<unsigned char>& mUsedPointMask;
};

class RemapIndices
{
public:
    //////////

    RemapIndices(PolygonPoolList& polygons,
        size_t polyListCount, const std::vector<unsigned>& indexMap)
        : mPolygons(polygons)
        , mPolyListCount(polyListCount)
        , mIndexMap(indexMap)
    {
    }

    void run(bool threaded = true)
    {
        if (threaded) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mPolyListCount), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, mPolyListCount));
        }
    }

    //////////

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(); n != range.end(); ++n) {
            PolygonPool& polygons = mPolygons[n];
            for (size_t i = 0; i < polygons.numQuads(); ++i) {
                Vec4I& quad = polygons.quad(i);
                quad[0] = mIndexMap[quad[0]];
                quad[1] = mIndexMap[quad[1]];
                quad[2] = mIndexMap[quad[2]];
                quad[3] = mIndexMap[quad[3]];
            }

            for (size_t i = 0; i < polygons.numTriangles(); ++i) {
                Vec3I& triangle = polygons.triangle(i);
                triangle[0] = mIndexMap[triangle[0]];
                triangle[1] = mIndexMap[triangle[1]];
                triangle[2] = mIndexMap[triangle[2]];
            }
        }
    }


private:
    PolygonPoolList& mPolygons;
    size_t mPolyListCount;
    const std::vector<unsigned>& mIndexMap;
};


class MovePoints
{
public:
    //////////

    MovePoints(
        internal::UniquePtr<openvdb::Vec3s>::type& newPointList,
        const PointList& oldPointList,
        const std::vector<unsigned>& indexMap,
        const std::vector<unsigned char>& usedPointMask)
        : mNewPointList(newPointList)
        , mOldPointList(oldPointList)
        , mIndexMap(indexMap)
        , mUsedPointMask(usedPointMask)
    {
    }

    void run(bool threaded = true)
    {
        if (threaded) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mIndexMap.size()), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, mIndexMap.size()));
        }
    }

    //////////

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(); n != range.end(); ++n) {
            if (mUsedPointMask[n]) {
                const size_t index = mIndexMap[n];
                mNewPointList.get()[index] = mOldPointList[n];
            }
        }
    }

private:
    internal::UniquePtr<openvdb::Vec3s>::type& mNewPointList;
    const PointList& mOldPointList;
    const std::vector<unsigned>& mIndexMap;
    const std::vector<unsigned char>& mUsedPointMask;
};


////////////////////////////////////////


template<typename SrcTreeT>
class GenTopologyMask
{
public:
    typedef tree::LeafManager<const SrcTreeT> LeafManagerT;
    typedef typename SrcTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef tree::ValueAccessor<const SrcTreeT> SrcAccessorT;
    typedef tree::ValueAccessor<BoolTreeT> BoolAccessorT;
    typedef Grid<BoolTreeT> BoolGridT;


    //////////


    GenTopologyMask(const BoolGridT& mask, const LeafManagerT& srcLeafs,
        const math::Transform& srcXForm, bool invertMask);

    void run(bool threaded = true);

    BoolTreeT& tree() { return mTree; }


    //////////

    GenTopologyMask(GenTopologyMask&, tbb::split);

    void operator()(const tbb::blocked_range<size_t>&);

    void join(GenTopologyMask& rhs) { mTree.merge(rhs.mTree); }

private:

    const BoolGridT& mMask;
    const LeafManagerT& mLeafManager;
    const math::Transform& mSrcXForm;
    bool mInvertMask;
    BoolTreeT mTree;
};


template<typename SrcTreeT>
GenTopologyMask<SrcTreeT>::GenTopologyMask(const BoolGridT& mask, const LeafManagerT& srcLeafs,
    const math::Transform& srcXForm, bool invertMask)
    : mMask(mask)
    , mLeafManager(srcLeafs)
    , mSrcXForm(srcXForm)
    , mInvertMask(invertMask)
    , mTree(false)
{
}


template<typename SrcTreeT>
GenTopologyMask<SrcTreeT>::GenTopologyMask(GenTopologyMask& rhs, tbb::split)
    : mMask(rhs.mMask)
    , mLeafManager(rhs.mLeafManager)
    , mSrcXForm(rhs.mSrcXForm)
    , mInvertMask(rhs.mInvertMask)
    , mTree(false)
{
}


template<typename SrcTreeT>
void
GenTopologyMask<SrcTreeT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(mLeafManager.getRange(), *this);
    } else {
        (*this)(mLeafManager.getRange());
    }
}


template<typename SrcTreeT>
void
GenTopologyMask<SrcTreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    Coord ijk;
    Vec3d xyz;
    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    const math::Transform& maskXForm = mMask.transform();
    tree::ValueAccessor<const BoolTreeT> maskAcc(mMask.tree());
    tree::ValueAccessor<BoolTreeT> acc(mTree);

    typename SrcTreeT::LeafNodeType::ValueOnCIter iter;
    for (size_t n = range.begin(); n != range.end(); ++n) {

        ijk = mLeafManager.leaf(n).origin();
        BoolLeafT* leaf = new BoolLeafT(ijk, false);
        bool addLeaf = false;

        if (maskXForm == mSrcXForm) {

            const BoolLeafT* maskLeaf = maskAcc.probeConstLeaf(ijk);

            if (maskLeaf) {

                for (iter = mLeafManager.leaf(n).cbeginValueOn(); iter; ++iter) {
                    Index pos = iter.pos();
                    if(maskLeaf->isValueOn(pos) != mInvertMask) {
                        leaf->setValueOn(pos);
                        addLeaf = true;
                    }
                }

            } else if (maskAcc.isValueOn(ijk) != mInvertMask) {
                leaf->topologyUnion(mLeafManager.leaf(n));
                addLeaf = true;
            }

        } else {
            for (iter = mLeafManager.leaf(n).cbeginValueOn(); iter; ++iter) {
                ijk = iter.getCoord();
                xyz = maskXForm.worldToIndex(mSrcXForm.indexToWorld(ijk));
                if(maskAcc.isValueOn(util::nearestCoord(xyz)) != mInvertMask) {
                    leaf->setValueOn(iter.pos());
                    addLeaf = true;
                }
            }
        }

        if (addLeaf) acc.addLeaf(leaf);
        else delete leaf;
    }
}


////////////////////////////////////////


template<typename SrcTreeT>
class GenBoundaryMask
{
public:
    typedef typename SrcTreeT::template ValueConverter<int>::Type IntTreeT;
    typedef typename SrcTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef tree::LeafManager<const SrcTreeT> LeafManagerT;

    //////////

    GenBoundaryMask(const LeafManagerT& leafs, const BoolTreeT&, const IntTreeT&);

    void run(bool threaded = true);

    BoolTreeT&  tree() { return mTree; }

    //////////

    GenBoundaryMask(GenBoundaryMask&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(GenBoundaryMask& rhs) { mTree.merge(rhs.mTree); }

private:
    // This typedef is needed for Windows
    typedef tree::ValueAccessor<const IntTreeT> IntTreeAccessorT;

    bool neighboringLeaf(const Coord&, const IntTreeAccessorT&) const;

    const LeafManagerT& mLeafManager;
    const BoolTreeT& mMaskTree;
    const IntTreeT& mIdxTree;
    BoolTreeT mTree;
    CoordBBox mLeafBBox;
};


template<typename SrcTreeT>
GenBoundaryMask<SrcTreeT>::GenBoundaryMask(const LeafManagerT& leafs,
    const BoolTreeT& maskTree, const IntTreeT& auxTree)
    : mLeafManager(leafs)
    , mMaskTree(maskTree)
    , mIdxTree(auxTree)
    , mTree(false)
{
    mIdxTree.evalLeafBoundingBox(mLeafBBox);
    mLeafBBox.expand(IntTreeT::LeafNodeType::DIM);
}


template<typename SrcTreeT>
GenBoundaryMask<SrcTreeT>::GenBoundaryMask(GenBoundaryMask& rhs, tbb::split)
    : mLeafManager(rhs.mLeafManager)
    , mMaskTree(rhs.mMaskTree)
    , mIdxTree(rhs.mIdxTree)
    , mTree(false)
    , mLeafBBox(rhs.mLeafBBox)
{
}


template<typename SrcTreeT>
void
GenBoundaryMask<SrcTreeT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(mLeafManager.getRange(), *this);
    } else {
        (*this)(mLeafManager.getRange());
    }
}


template<typename SrcTreeT>
bool
GenBoundaryMask<SrcTreeT>::neighboringLeaf(const Coord& ijk, const IntTreeAccessorT& acc) const
{
    if (acc.probeConstLeaf(ijk)) return true;

    const int dim = IntTreeT::LeafNodeType::DIM;

    // face adjacent neghbours
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1], ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1], ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1] + dim, ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1] - dim, ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1], ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1], ijk[2] - dim))) return true;

    // edge adjacent neighbors
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1], ijk[2] - dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1], ijk[2] - dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1], ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1], ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1] + dim, ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1] + dim, ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1] - dim, ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1] - dim, ijk[2]))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1] - dim, ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1] - dim, ijk[2] - dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1] + dim, ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0], ijk[1] + dim, ijk[2] - dim))) return true;

    // corner adjacent neighbors
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1] - dim, ijk[2] - dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1] - dim, ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1] - dim, ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1] - dim, ijk[2] - dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1] + dim, ijk[2] - dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] - dim, ijk[1] + dim, ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1] + dim, ijk[2] + dim))) return true;
    if (acc.probeConstLeaf(Coord(ijk[0] + dim, ijk[1] + dim, ijk[2] - dim))) return true;

    return false;
}


template<typename SrcTreeT>
void
GenBoundaryMask<SrcTreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    Coord ijk;
    tree::ValueAccessor<const BoolTreeT> maskAcc(mMaskTree);
    tree::ValueAccessor<const IntTreeT> idxAcc(mIdxTree);
    tree::ValueAccessor<BoolTreeT> acc(mTree);

    typename SrcTreeT::LeafNodeType::ValueOnCIter iter;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const typename SrcTreeT::LeafNodeType&
            leaf = mLeafManager.leaf(n);

        ijk = leaf.origin();

        if (!mLeafBBox.isInside(ijk) || !neighboringLeaf(ijk, idxAcc)) continue;

        const typename BoolTreeT::LeafNodeType*
            maskLeaf = maskAcc.probeConstLeaf(ijk);

        if (!maskLeaf || !leaf.hasSameTopology(maskLeaf)) {
            acc.touchLeaf(ijk)->topologyUnion(leaf);
        }
    }
}


////////////////////////////////////////


template<typename TreeT>
class GenTileMask
{
public:
    typedef typename TreeT::template ValueConverter<bool>::Type BoolTreeT;

    typedef typename TreeT::ValueType ValueT;

    //////////

    GenTileMask(const std::vector<Vec4i>& tiles, const TreeT& distTree, ValueT iso);

    void run(bool threaded = true);

    BoolTreeT& tree() { return mTree; }

    //////////

    GenTileMask(GenTileMask&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(GenTileMask& rhs) { mTree.merge(rhs.mTree); }

private:

    const std::vector<Vec4i>& mTiles;
    const TreeT& mDistTree;
    ValueT mIsovalue;

    BoolTreeT mTree;
};


template<typename TreeT>
GenTileMask<TreeT>::GenTileMask(
    const std::vector<Vec4i>& tiles, const TreeT& distTree, ValueT iso)
    : mTiles(tiles)
    , mDistTree(distTree)
    , mIsovalue(iso)
    , mTree(false)
{
}


template<typename TreeT>
GenTileMask<TreeT>::GenTileMask(GenTileMask& rhs, tbb::split)
    : mTiles(rhs.mTiles)
    , mDistTree(rhs.mDistTree)
    , mIsovalue(rhs.mIsovalue)
    , mTree(false)
{
}


template<typename TreeT>
void
GenTileMask<TreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mTiles.size()), *this);
    else (*this)(tbb::blocked_range<size_t>(0, mTiles.size()));
}


template<typename TreeT>
void
GenTileMask<TreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    tree::ValueAccessor<const TreeT> distAcc(mDistTree);
    CoordBBox region, bbox;
    Coord ijk, nijk;
    bool processRegion = true;
    ValueT value;


    for (size_t n = range.begin(); n != range.end(); ++n) {

        const Vec4i& tile = mTiles[n];

        bbox.min()[0] = tile[0];
        bbox.min()[1] = tile[1];
        bbox.min()[2] = tile[2];

        bbox.max() = bbox.min();
        bbox.max().offset(tile[3]);

        const bool thisInside = (distAcc.getValue(bbox.min()) < mIsovalue);
        const int thisDepth = distAcc.getValueDepth(bbox.min());

        // eval x-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[0];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(nijk)) {
            processRegion = thisInside != (distAcc.getValue(nijk) < mIsovalue);
        }


        if (processRegion) {
            region = bbox;
            region.min()[0] = region.max()[0] = ijk[0];
            mTree.fill(region, true);
        }


        ijk = bbox.min();
        --ijk[0];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processRegion = !distAcc.probeValue(ijk, value) && thisInside != (value < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[0] = region.max()[0] = ijk[0];
            mTree.fill(region, true);
        }


        // eval y-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[1];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(nijk)) {
            processRegion = thisInside != (distAcc.getValue(nijk) < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[1] = region.max()[1] = ijk[1];
            mTree.fill(region, true);
        }


        ijk = bbox.min();
        --ijk[1];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processRegion = !distAcc.probeValue(ijk, value) && thisInside != (value < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[1] = region.max()[1] = ijk[1];
            mTree.fill(region, true);
        }


        // eval z-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[2];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(nijk)) {
            processRegion = thisInside != (distAcc.getValue(nijk) < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[2] = region.max()[2] = ijk[2];
            mTree.fill(region, true);
        }

        ijk = bbox.min();
        --ijk[2];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processRegion = !distAcc.probeValue(ijk, value) && thisInside != (value < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[2] = region.max()[2] = ijk[2];
            mTree.fill(region, true);
        }


        ijk = bbox.min();
        --ijk[1];
        --ijk[2];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processRegion = !distAcc.probeValue(ijk, value) && thisInside != (value < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[1] = region.max()[1] = ijk[1];
            region.min()[2] = region.max()[2] = ijk[2];
            mTree.fill(region, true);
        }


        ijk = bbox.min();
        --ijk[0];
        --ijk[1];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processRegion = !distAcc.probeValue(ijk, value) && thisInside != (value < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[1] = region.max()[1] = ijk[1];
            region.min()[0] = region.max()[0] = ijk[0];
            mTree.fill(region, true);
        }

        ijk = bbox.min();
        --ijk[0];
        --ijk[2];

        processRegion = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processRegion = !distAcc.probeValue(ijk, value) && thisInside != (value < mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.min()[2] = region.max()[2] = ijk[2];
            region.min()[0] = region.max()[0] = ijk[0];
            mTree.fill(region, true);
        }
    }
}


////////////////////////////////////////


template<class DistTreeT, class SignTreeT, class IdxTreeT>
inline void
tileData(const DistTreeT& distTree, SignTreeT& signTree, IdxTreeT& idxTree, double iso)
{
    typename DistTreeT::ValueOnCIter tileIter(distTree);
    tileIter.setMaxDepth(DistTreeT::ValueOnCIter::LEAF_DEPTH - 1);

    if (!tileIter) return; // volume has no active tiles.

    size_t tileCount = 0;
    for ( ; tileIter; ++tileIter) {
        ++tileCount;
    }

    std::vector<Vec4i> tiles(tileCount);

    tileCount = 0;
    tileIter = distTree.cbeginValueOn();
    tileIter.setMaxDepth(DistTreeT::ValueOnCIter::LEAF_DEPTH - 1);

    CoordBBox bbox;
    for (; tileIter; ++tileIter) {
        Vec4i& tile = tiles[tileCount++];
        tileIter.getBoundingBox(bbox);
        tile[0] = bbox.min()[0];
        tile[1] = bbox.min()[1];
        tile[2] = bbox.min()[2];
        tile[3] = bbox.max()[0] - bbox.min()[0];
    }

    typename DistTreeT::ValueType isovalue = typename DistTreeT::ValueType(iso);

    GenTileMask<DistTreeT> tileMask(tiles, distTree, isovalue);
    tileMask.run();

    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef tree::LeafManager<BoolTreeT> BoolLeafManagerT;

    BoolLeafManagerT leafs(tileMask.tree());


    internal::SignData<DistTreeT, BoolLeafManagerT> op(distTree, leafs, isovalue);
    op.run();

    signTree.merge(*op.signTree());
    idxTree.merge(*op.idxTree());
}


////////////////////////////////////////


// Utility class for the volumeToMesh wrapper
class PointListCopy
{
public:
    PointListCopy(const PointList& pointsIn, std::vector<Vec3s>& pointsOut)
        : mPointsIn(pointsIn) , mPointsOut(pointsOut)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(); n < range.end(); ++n) {
            mPointsOut[n] = mPointsIn[n];
        }
    }

private:
    const PointList& mPointsIn;
    std::vector<Vec3s>& mPointsOut;
};


// Checks if the isovalue is in proximity to the active voxel boundary.
template <typename LeafManagerT>
inline bool
needsActiveVoxePadding(const LeafManagerT& leafs, double iso, double voxelSize)
{
    double interiorWidth = 0.0, exteriorWidth = 0.0;
    {
        typename LeafManagerT::TreeType::LeafNodeType::ValueOffCIter it;
        bool foundInterior = false, foundExterior = false;
        for (size_t n = 0, N = leafs.leafCount(); n < N; ++n) {

            for (it = leafs.leaf(n).cbeginValueOff(); it; ++it) {
                double value = double(it.getValue());
                if (value < 0.0) {
                    interiorWidth = value;
                    foundInterior = true;
                } else if (value > 0.0) {
                    exteriorWidth = value;
                    foundExterior = true;
                }

                if (foundInterior && foundExterior) break;
            }

            if (foundInterior && foundExterior) break;
        }

    }

    double minDist = std::min(std::abs(interiorWidth - iso), std::abs(exteriorWidth - iso));
    return !(minDist > (2.0 * voxelSize));
}


} // end namespace internal


////////////////////////////////////////


inline
PolygonPool::PolygonPool()
    : mNumQuads(0)
    , mNumTriangles(0)
    , mQuads(NULL)
    , mTriangles(NULL)
    , mQuadFlags(NULL)
    , mTriangleFlags(NULL)
{
}


inline
PolygonPool::PolygonPool(const size_t numQuads, const size_t numTriangles)
    : mNumQuads(numQuads)
    , mNumTriangles(numTriangles)
    , mQuads(new openvdb::Vec4I[mNumQuads])
    , mTriangles(new openvdb::Vec3I[mNumTriangles])
    , mQuadFlags(new char[mNumQuads])
    , mTriangleFlags(new char[mNumTriangles])
{
}


inline void
PolygonPool::copy(const PolygonPool& rhs)
{
    resetQuads(rhs.numQuads());
    resetTriangles(rhs.numTriangles());

    for (size_t i = 0; i < mNumQuads; ++i) {
        mQuads[i] = rhs.mQuads[i];
        mQuadFlags[i] = rhs.mQuadFlags[i];
    }

    for (size_t i = 0; i < mNumTriangles; ++i) {
        mTriangles[i] = rhs.mTriangles[i];
        mTriangleFlags[i] = rhs.mTriangleFlags[i];
    }
}


inline void
PolygonPool::resetQuads(size_t size)
{
    mNumQuads = size;
    mQuads.reset(new openvdb::Vec4I[mNumQuads]);
    mQuadFlags.reset(new char[mNumQuads]);
}


inline void
PolygonPool::clearQuads()
{
    mNumQuads = 0;
    mQuads.reset(NULL);
    mQuadFlags.reset(NULL);
}


inline void
PolygonPool::resetTriangles(size_t size)
{
    mNumTriangles = size;
    mTriangles.reset(new openvdb::Vec3I[mNumTriangles]);
    mTriangleFlags.reset(new char[mNumTriangles]);
}


inline void
PolygonPool::clearTriangles()
{
    mNumTriangles = 0;
    mTriangles.reset(NULL);
    mTriangleFlags.reset(NULL);
}


inline bool
PolygonPool::trimQuads(const size_t n, bool reallocate)
{
    if (!(n < mNumQuads)) return false;

    if (reallocate) {

        if (n == 0) {
            mQuads.reset(NULL);
        } else {

            boost::scoped_array<openvdb::Vec4I> quads(new openvdb::Vec4I[n]);
            boost::scoped_array<char> flags(new char[n]);

            for (size_t i = 0; i < n; ++i) {
                quads[i] = mQuads[i];
                flags[i] = mQuadFlags[i];
            }

            mQuads.swap(quads);
            mQuadFlags.swap(flags);
        }
    }

    mNumQuads = n;
    return true;
}


inline bool
PolygonPool::trimTrinagles(const size_t n, bool reallocate)
{
    if (!(n < mNumTriangles)) return false;

    if (reallocate) {

        if (n == 0) {
            mTriangles.reset(NULL);
        } else {

            boost::scoped_array<openvdb::Vec3I> triangles(new openvdb::Vec3I[n]);
            boost::scoped_array<char> flags(new char[n]);

            for (size_t i = 0; i < n; ++i) {
                triangles[i] = mTriangles[i];
                flags[i] = mTriangleFlags[i];
            }

            mTriangles.swap(triangles);
            mTriangleFlags.swap(flags);
        }
    }

    mNumTriangles = n;
    return true;
}


////////////////////////////////////////


inline VolumeToMesh::VolumeToMesh(double isovalue, double adaptivity)
    : mPoints(NULL)
    , mPolygons()
    , mPointListSize(0)
    , mSeamPointListSize(0)
    , mPolygonPoolListSize(0)
    , mIsovalue(isovalue)
    , mPrimAdaptivity(adaptivity)
    , mSecAdaptivity(0.0)
    , mRefGrid(GridBase::ConstPtr())
    , mSurfaceMaskGrid(GridBase::ConstPtr())
    , mAdaptivityGrid(GridBase::ConstPtr())
    , mAdaptivityMaskTree(TreeBase::ConstPtr())
    , mRefSignTree(TreeBase::Ptr())
    , mRefIdxTree(TreeBase::Ptr())
    , mInvertSurfaceMask(false)
    , mPartitions(1)
    , mActivePart(0)
    , mQuantizedSeamPoints(NULL)
    , mPointFlags(0)
{
}


inline PointList&
VolumeToMesh::pointList()
{
    return mPoints;
}


inline const size_t&
VolumeToMesh::pointListSize() const
{
    return mPointListSize;
}


inline PolygonPoolList&
VolumeToMesh::polygonPoolList()
{
    return mPolygons;
}


inline const PolygonPoolList&
VolumeToMesh::polygonPoolList() const
{
    return mPolygons;
}


inline const size_t&
VolumeToMesh::polygonPoolListSize() const
{
    return mPolygonPoolListSize;
}


inline void
VolumeToMesh::setRefGrid(const GridBase::ConstPtr& grid, double secAdaptivity)
{
    mRefGrid = grid;
    mSecAdaptivity = secAdaptivity;

    // Clear out old auxiliary data
    mRefSignTree = TreeBase::Ptr();
    mRefIdxTree = TreeBase::Ptr();
    mSeamPointListSize = 0;
    mQuantizedSeamPoints.reset(NULL);
}


inline void
VolumeToMesh::setSurfaceMask(const GridBase::ConstPtr& mask, bool invertMask)
{
    mSurfaceMaskGrid = mask;
    mInvertSurfaceMask = invertMask;
}


inline void
VolumeToMesh::setSpatialAdaptivity(const GridBase::ConstPtr& grid)
{
    mAdaptivityGrid = grid;
}


inline void
VolumeToMesh::setAdaptivityMask(const TreeBase::ConstPtr& tree)
{
   mAdaptivityMaskTree = tree;
}


inline void
VolumeToMesh::partition(unsigned partitions, unsigned activePart)
{
    mPartitions = std::max(partitions, unsigned(1));
    mActivePart = std::min(activePart, mPartitions-1);
}


inline std::vector<unsigned char>&
VolumeToMesh::pointFlags()
{
    return mPointFlags;
}


inline const std::vector<unsigned char>&
VolumeToMesh::pointFlags() const
{
    return mPointFlags;
}


template<typename GridT>
inline void
VolumeToMesh::operator()(const GridT& distGrid)
{
    typedef typename GridT::TreeType DistTreeT;
    typedef tree::LeafManager<const DistTreeT> DistLeafManagerT;
    typedef typename DistTreeT::ValueType DistValueT;

    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef tree::LeafManager<BoolTreeT> BoolLeafManagerT;
    typedef Grid<BoolTreeT> BoolGridT;

    typedef typename DistTreeT::template ValueConverter<Int16>::Type Int16TreeT;
    typedef tree::LeafManager<Int16TreeT> Int16LeafManagerT;

    typedef typename DistTreeT::template ValueConverter<int>::Type IntTreeT;
    typedef typename DistTreeT::template ValueConverter<float>::Type FloatTreeT;
    typedef Grid<FloatTreeT> FloatGridT;


    const openvdb::math::Transform& transform = distGrid.transform();
    const DistTreeT& distTree = distGrid.tree();
    const DistValueT isovalue = DistValueT(mIsovalue);

    typename Int16TreeT::Ptr signTreePt;
    typename IntTreeT::Ptr idxTreePt;
    typename BoolTreeT::Ptr pointMask;

    BoolTreeT valueMask(false), seamMask(false);
    const bool adaptive = mPrimAdaptivity > 1e-7 || mSecAdaptivity > 1e-7;
    bool maskEdges = false;


    const BoolGridT * surfaceMask = NULL;
    if (mSurfaceMaskGrid && mSurfaceMaskGrid->type() == BoolGridT::gridType()) {
        surfaceMask = static_cast<const BoolGridT*>(mSurfaceMaskGrid.get());
    }

    const FloatGridT * adaptivityField = NULL;
    if (mAdaptivityGrid && mAdaptivityGrid->type() == FloatGridT::gridType()) {
        adaptivityField = static_cast<const FloatGridT*>(mAdaptivityGrid.get());
    }

    if (mAdaptivityMaskTree && mAdaptivityMaskTree->type() == BoolTreeT::treeType()) {
        const BoolTreeT *adaptivityMaskPt =
            static_cast<const BoolTreeT*>(mAdaptivityMaskTree.get());
        seamMask.topologyUnion(*adaptivityMaskPt);
    }


    // Collect auxiliary data
    {
        DistLeafManagerT distLeafs(distTree);

        // Check if the isovalue is in proximity to the active voxel boundary.
        bool padActiveVoxels = false;
        int padVoxels = 3;

        if (distGrid.getGridClass() != GRID_LEVEL_SET) {
            padActiveVoxels = true;
        } else {
            padActiveVoxels = internal::needsActiveVoxePadding(distLeafs,
                mIsovalue, transform.voxelSize()[0]);
        }

        // always pad the active region for small volumes (the performance hit is neglectable).
        if (!padActiveVoxels) {
            Coord dim;
            distTree.evalActiveVoxelDim(dim);
            int maxDim = std::max(std::max(dim[0], dim[1]), dim[2]);
            if (maxDim < 1000) {
                padActiveVoxels = true;
                padVoxels = 1;
            }
        }

        if (surfaceMask || mPartitions > 1) {

            maskEdges = true;

            if (surfaceMask) {

                { // Mask
                    internal::GenTopologyMask<DistTreeT> masking(
                        *surfaceMask, distLeafs, transform, mInvertSurfaceMask);
                    masking.run();
                    valueMask.merge(masking.tree());
                }

                if (mPartitions > 1) { // Partition
                    tree::LeafManager<BoolTreeT> leafs(valueMask);
                    leafs.foreach(internal::PartOp(leafs.leafCount() , mPartitions, mActivePart));
                    tools::pruneInactive(valueMask);
                }

            } else { // Partition

                internal::PartGen<DistTreeT> partitioner(distLeafs, mPartitions, mActivePart);
                partitioner.run();
                valueMask.merge(partitioner.tree());
            }

            {
                if (padActiveVoxels) tools::dilateVoxels(valueMask, padVoxels);
                BoolLeafManagerT leafs(valueMask);

                internal::SignData<DistTreeT, BoolLeafManagerT>
                    signDataOp(distTree, leafs, isovalue);
                signDataOp.run();

                signTreePt = signDataOp.signTree();
                idxTreePt = signDataOp.idxTree();
            }

            {
                internal::GenBoundaryMask<DistTreeT> boundary(distLeafs, valueMask, *idxTreePt);
                boundary.run();

                BoolLeafManagerT bleafs(boundary.tree());

                internal::SignData<DistTreeT, BoolLeafManagerT>
                    signDataOp(distTree, bleafs, isovalue);
                signDataOp.run();

                signTreePt->merge(*signDataOp.signTree());
                idxTreePt->merge(*signDataOp.idxTree());
            }

        } else {

            // Collect voxel-sign configurations
            if (padActiveVoxels) {

                BoolTreeT regionMask(false);
                regionMask.topologyUnion(distTree);
                tools::dilateVoxels(regionMask, padVoxels);

                BoolLeafManagerT leafs(regionMask);

                internal::SignData<DistTreeT, BoolLeafManagerT>
                    signDataOp(distTree, leafs, isovalue);
                signDataOp.run();

                signTreePt = signDataOp.signTree();
                idxTreePt = signDataOp.idxTree();
            } else {

                internal::SignData<DistTreeT, DistLeafManagerT>
                    signDataOp(distTree, distLeafs, isovalue);
                signDataOp.run();

                signTreePt = signDataOp.signTree();
                idxTreePt = signDataOp.idxTree();
            }
        }

    }


    // Collect auxiliary data from active tiles
    internal::tileData(distTree, *signTreePt, *idxTreePt, static_cast<double>(isovalue));

    // Optionally collect auxiliary data from a reference level set.
    Int16TreeT *refSignTreePt = NULL;
    IntTreeT *refIdxTreePt = NULL;
    const DistTreeT *refDistTreePt = NULL;

    if (mRefGrid && mRefGrid->type() == GridT::gridType()) {

        const GridT* refGrid = static_cast<const GridT*>(mRefGrid.get());
        refDistTreePt = &refGrid->tree();

        // Collect and cache auxiliary data from the reference grid.
        if (!mRefSignTree && !mRefIdxTree) {

            DistLeafManagerT refDistLeafs(*refDistTreePt);
            internal::SignData<DistTreeT, DistLeafManagerT>
                signDataOp(*refDistTreePt, refDistLeafs, isovalue);

            signDataOp.run();

            mRefSignTree = signDataOp.signTree();
            mRefIdxTree = signDataOp.idxTree();
        }

        // Get cached auxiliary data
        if (mRefSignTree && mRefIdxTree) {
            refSignTreePt = static_cast<Int16TreeT*>(mRefSignTree.get());
            refIdxTreePt = static_cast<IntTreeT*>(mRefIdxTree.get());
        }
    }


    // Process auxiliary data
    Int16LeafManagerT signLeafs(*signTreePt);

    if (maskEdges) {
        signLeafs.foreach(internal::MaskEdges<BoolTreeT>(valueMask));
        valueMask.clear();
    }


    // Generate the seamline mask
    if (refSignTreePt) {
        internal::GenSeamMask<Int16TreeT, Int16LeafManagerT> seamOp(signLeafs, *refSignTreePt);
        seamOp.run();

        tools::dilateVoxels(seamOp.mask(), 3);
        signLeafs.foreach(internal::TagSeamEdges<BoolTreeT>(seamOp.mask()));

        seamMask.merge(seamOp.mask());
    }


    std::vector<size_t> regions(signLeafs.leafCount(), 0);
    if (regions.empty()) return;

    if (adaptive) {

        internal::MergeVoxelRegions<DistTreeT, Int16LeafManagerT> merge(
            signLeafs, *signTreePt, distTree, *idxTreePt, isovalue, DistValueT(mPrimAdaptivity));

        if (adaptivityField) {
            merge.setSpatialAdaptivity(transform, *adaptivityField);
        }

        if (refSignTreePt || mAdaptivityMaskTree) {
            merge.setAdaptivityMask(&seamMask);
        }

        if (refSignTreePt) {
            merge.setRefData(refSignTreePt, DistValueT(mSecAdaptivity));
        }

        merge.run();

        signLeafs.foreach(internal::CountRegions<IntTreeT>(*idxTreePt, regions));

    } else {
        signLeafs.foreach(internal::CountPoints(regions));
    }


    {
        mPointListSize = 0;
        size_t tmp = 0;
        for (size_t n = 0, N = regions.size(); n < N; ++n) {
            tmp = regions[n];
            regions[n] = mPointListSize;
            mPointListSize += tmp;
        }
    }


    // Generate the unique point list
    mPoints.reset(new openvdb::Vec3s[mPointListSize]);
    mPointFlags.clear();

    // Generate seam line sample points
    if (refSignTreePt && refIdxTreePt) {

        if (mSeamPointListSize == 0) {

            std::vector<size_t> pointMap;

            {
                Int16LeafManagerT refSignLeafs(*refSignTreePt);
                pointMap.resize(refSignLeafs.leafCount(), 0);

                refSignLeafs.foreach(internal::CountPoints(pointMap));

                size_t tmp = 0;
                for (size_t n = 0, N = pointMap.size(); n < N; ++n) {
                    tmp = pointMap[n];
                    pointMap[n] = mSeamPointListSize;
                    mSeamPointListSize += tmp;
                }
            }

            if (!pointMap.empty() && mSeamPointListSize != 0) {

                mQuantizedSeamPoints.reset(new uint32_t[mSeamPointListSize]);
                memset(mQuantizedSeamPoints.get(), 0, sizeof(uint32_t) * mSeamPointListSize);

                typedef tree::LeafManager<IntTreeT> IntLeafManagerT;

                IntLeafManagerT refIdxLeafs(*refIdxTreePt);
                refIdxLeafs.foreach(internal::MapPoints<Int16TreeT>(pointMap, *refSignTreePt));
            }
        }

        if (mSeamPointListSize != 0) {
            signLeafs.foreach(internal::SeamWeights<DistTreeT>(
                distTree, *refSignTreePt, *refIdxTreePt, mQuantizedSeamPoints, mIsovalue));
        }
    }


    internal::GenPoints<DistTreeT, Int16LeafManagerT>
        pointOp(signLeafs, distTree, *idxTreePt, mPoints, regions, transform, mIsovalue);


    if (mSeamPointListSize != 0) {
        mPointFlags.resize(mPointListSize);
        pointOp.setRefData(refSignTreePt, refDistTreePt, refIdxTreePt,
            &mQuantizedSeamPoints, &mPointFlags);
    }

    pointOp.run();


    mPolygonPoolListSize = signLeafs.leafCount();
    mPolygons.reset(new PolygonPool[mPolygonPoolListSize]);


    if (adaptive) {

        internal::GenPolygons<Int16LeafManagerT, internal::AdaptivePrimBuilder>
            mesher(signLeafs, *signTreePt, *idxTreePt, mPolygons, Index32(mPointListSize));

        mesher.setRefSignTree(refSignTreePt);
        mesher.run();

    } else {

        internal::GenPolygons<Int16LeafManagerT, internal::UniformPrimBuilder>
            mesher(signLeafs, *signTreePt, *idxTreePt, mPolygons, Index32(mPointListSize));

        mesher.setRefSignTree(refSignTreePt);
        mesher.run();
    }

    // Clean up unused points, only necessary if masking and/or
    // automatic mesh partitioning is enabled.
    if ((surfaceMask || mPartitions > 1) && mPointListSize > 0) {

        // Flag used points
        std::vector<unsigned char> usedPointMask(mPointListSize, 0);

        internal::FlagUsedPoints flagPoints(mPolygons, mPolygonPoolListSize, usedPointMask);
        flagPoints.run();

        // Create index map
        std::vector<unsigned> indexMap(mPointListSize);
        size_t usedPointCount = 0;
        for (size_t p = 0; p < mPointListSize; ++p) {
            if (usedPointMask[p]) indexMap[p] = static_cast<unsigned>(usedPointCount++);
        }

        if (usedPointCount < mPointListSize) {

            // move points
            internal::UniquePtr<openvdb::Vec3s>::type
                newPointList(new openvdb::Vec3s[usedPointCount]);

            internal::MovePoints movePoints(newPointList, mPoints, indexMap, usedPointMask);
            movePoints.run();

            mPointListSize = usedPointCount;
            mPoints.reset(newPointList.release());

            // update primitives
            internal::RemapIndices remap(mPolygons, mPolygonPoolListSize, indexMap);
            remap.run();
        }
    }


    // Subdivide nonplanar quads near the seamline edges
    // todo: thread and clean up
    if (refSignTreePt || refIdxTreePt || refDistTreePt) {
        std::vector<Vec3s> newPoints;

        for (size_t n = 0; n <  mPolygonPoolListSize; ++n) {

            PolygonPool& polygons = mPolygons[n];

            std::vector<size_t> nonPlanarQuads;
            nonPlanarQuads.reserve(polygons.numQuads());

            for (size_t i = 0; i < polygons.numQuads(); ++i) {

                char& flags = polygons.quadFlags(i);

                if ((flags & POLYFLAG_FRACTURE_SEAM) && !(flags & POLYFLAG_EXTERIOR)) {

                    openvdb::Vec4I& quad = polygons.quad(i);

                    const bool edgePoly = mPointFlags[quad[0]] || mPointFlags[quad[1]]
                        || mPointFlags[quad[2]] || mPointFlags[quad[3]];

                    if (!edgePoly) continue;

                    const Vec3s& p0 = mPoints[quad[0]];
                    const Vec3s& p1 = mPoints[quad[1]];
                    const Vec3s& p2 = mPoints[quad[2]];
                    const Vec3s& p3 = mPoints[quad[3]];

                    if (!internal::isPlanarQuad(p0, p1, p2, p3, 1e-6f)) {
                        nonPlanarQuads.push_back(i);
                    }
                }
            }


            if (!nonPlanarQuads.empty()) {

                PolygonPool tmpPolygons;

                tmpPolygons.resetQuads(polygons.numQuads() - nonPlanarQuads.size());
                tmpPolygons.resetTriangles(polygons.numTriangles() + 4 * nonPlanarQuads.size());

                size_t triangleIdx = 0;
                for (size_t i = 0; i < nonPlanarQuads.size(); ++i) {

                    size_t& quadIdx = nonPlanarQuads[i];

                    openvdb::Vec4I& quad = polygons.quad(quadIdx);
                    char& quadFlags = polygons.quadFlags(quadIdx);
                    //quadFlags |= POLYFLAG_SUBDIVIDED;

                    Vec3s centroid = (mPoints[quad[0]] + mPoints[quad[1]] +
                        mPoints[quad[2]] + mPoints[quad[3]]) * 0.25;

                    size_t pointIdx = newPoints.size() + mPointListSize;

                    newPoints.push_back(centroid);


                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[0];
                        triangle[1] = static_cast<unsigned>(pointIdx);
                        triangle[2] = quad[3];

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;

                        if (mPointFlags[triangle[0]] || mPointFlags[triangle[2]]) {
                            tmpPolygons.triangleFlags(triangleIdx) |= POLYFLAG_SUBDIVIDED;
                        }
                    }

                    ++triangleIdx;

                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[0];
                        triangle[1] = quad[1];
                        triangle[2] = static_cast<unsigned>(pointIdx);

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;

                        if (mPointFlags[triangle[0]] || mPointFlags[triangle[1]]) {
                            tmpPolygons.triangleFlags(triangleIdx) |= POLYFLAG_SUBDIVIDED;
                        }
                    }

                    ++triangleIdx;

                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[1];
                        triangle[1] = quad[2];
                        triangle[2] = static_cast<unsigned>(pointIdx);

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;

                        if (mPointFlags[triangle[0]] || mPointFlags[triangle[1]]) {
                            tmpPolygons.triangleFlags(triangleIdx) |= POLYFLAG_SUBDIVIDED;
                        }
                    }


                    ++triangleIdx;

                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[2];
                        triangle[1] = quad[3];
                        triangle[2] = static_cast<unsigned>(pointIdx);

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;

                        if (mPointFlags[triangle[0]] || mPointFlags[triangle[1]]) {
                            tmpPolygons.triangleFlags(triangleIdx) |= POLYFLAG_SUBDIVIDED;
                        }
                    }

                    ++triangleIdx;

                    quad[0] = util::INVALID_IDX;
                }


                for (size_t i = 0; i < polygons.numTriangles(); ++i) {
                    tmpPolygons.triangle(triangleIdx) = polygons.triangle(i);
                    tmpPolygons.triangleFlags(triangleIdx) = polygons.triangleFlags(i);
                    ++triangleIdx;
                }


                size_t quadIdx = 0;
                for (size_t i = 0; i < polygons.numQuads(); ++i) {
                    openvdb::Vec4I& quad = polygons.quad(i);

                    if (quad[0] != util::INVALID_IDX) {
                        tmpPolygons.quad(quadIdx) = quad;
                        tmpPolygons.quadFlags(quadIdx) = polygons.quadFlags(i);
                        ++quadIdx;
                    }
                }


                polygons.copy(tmpPolygons);
            }

        }


        if (!newPoints.empty()) {

            size_t newPointCount = newPoints.size() + mPointListSize;

            internal::UniquePtr<openvdb::Vec3s>::type
                newPointList(new openvdb::Vec3s[newPointCount]);

            for (size_t i = 0; i < mPointListSize; ++i) {
                newPointList.get()[i] = mPoints[i];
            }

            for (size_t i = mPointListSize; i < newPointCount; ++i) {
                newPointList.get()[i] = newPoints[i - mPointListSize];
            }

            mPointListSize = newPointCount;
            mPoints.reset(newPointList.release());
            mPointFlags.resize(mPointListSize, 0);
        }
    }
}


////////////////////////////////////////


/// @internal This overload is enabled only for grids with a scalar ValueType.
template<typename GridType>
inline typename boost::enable_if<boost::is_scalar<typename GridType::ValueType>, void>::type
doVolumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    std::vector<Vec4I>& quads,
    double isovalue,
    double adaptivity)
{
    VolumeToMesh mesher(isovalue, adaptivity);
    mesher(grid);

    // Preallocate the point list
    points.clear();
    points.resize(mesher.pointListSize());

    { // Copy points
        internal::PointListCopy ptnCpy(mesher.pointList(), points);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), ptnCpy);
        mesher.pointList().reset(NULL);
    }

    PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

    { // Preallocate primitive lists
        size_t numQuads = 0, numTriangles = 0;
        for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
            numTriangles += polygons.numTriangles();
            numQuads += polygons.numQuads();
        }

        triangles.clear();
        triangles.resize(numTriangles);
        quads.clear();
        quads.resize(numQuads);
    }

    // Copy primitives
    size_t qIdx = 0, tIdx = 0;
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
            quads[qIdx++] = polygons.quad(i);
        }

        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
            triangles[tIdx++] = polygons.triangle(i);
        }
    }
}

/// @internal This overload is enabled only for grids that do not have a scalar ValueType.
template<typename GridType>
inline typename boost::disable_if<boost::is_scalar<typename GridType::ValueType>, void>::type
doVolumeToMesh(
    const GridType&,
    std::vector<Vec3s>&,
    std::vector<Vec3I>&,
    std::vector<Vec4I>&,
    double,
    double)
{
    OPENVDB_THROW(TypeError, "volume to mesh conversion is supported only for scalar grids");
}


template<typename GridType>
inline void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    std::vector<Vec4I>& quads,
    double isovalue,
    double adaptivity)
{
    doVolumeToMesh(grid, points, triangles, quads, isovalue, adaptivity);
}


template<typename GridType>
inline void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& quads,
    double isovalue)
{
    std::vector<Vec3I> triangles;
    doVolumeToMesh(grid, points, triangles, quads, isovalue, 0.0);
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
