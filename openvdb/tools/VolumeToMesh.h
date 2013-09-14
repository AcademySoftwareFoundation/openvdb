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

#ifndef OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED

#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/util/Util.h> // for COORD_OFFSETS
#include <openvdb/math/Operators.h> // for ISGradient
#include <openvdb/tools/Morphology.h> // for dilateVoxels()
#include <openvdb/tree/LeafManager.h>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vector>
#include <memory> // for auto_pointer


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
template<typename GridType>
void
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
template<typename GridType>
void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    std::vector<Vec4I>& quads,
    double isovalue = 0.0,
    double adaptivity = 0.0);


////////////////////////////////////////


/// @brief Polygon flags, used for reference based meshing.
enum { POLYFLAG_EXTERIOR = 0x1, POLYFLAG_FRACTURE_SEAM = 0x2 };


/// @brief Collection of quads and triangles
class PolygonPool
{
public:
    PolygonPool()
    : mNumQuads(0)
    , mNumTriangles(0)
    , mQuads(NULL)
    , mTriangles(NULL)
    , mQuadFlags(NULL)
    , mTriangleFlags(NULL)
    {
    }

    void resetQuads(size_t size)
    {
        mNumQuads = size;
        mQuads.reset(new openvdb::Vec4I[mNumQuads]);
        mQuadFlags.reset(new char[mNumQuads]);
    }

    void clearQuads()
    {
        mNumQuads = 0;
        mQuads.reset(NULL);
        mQuadFlags.reset(NULL);
    }

    void resetTriangles(size_t size)
    {
        mNumTriangles = size;
        mTriangles.reset(new openvdb::Vec3I[mNumTriangles]);
        mTriangleFlags.reset(new char[mNumTriangles]);
    }

    void clearTriangles()
    {
        mNumTriangles = 0;
        mTriangles.reset(NULL);
        mTriangleFlags.reset(NULL);
    }

    const size_t& numQuads() const { return mNumQuads; }
    const size_t& numTriangles() const { return mNumTriangles; }

    // polygon accessor methods
    openvdb::Vec4I& quad(size_t n) { return mQuads[n]; }
    const openvdb::Vec4I& quad(size_t n) const { return mQuads[n]; }

    openvdb::Vec3I& triangle(size_t n) { return mTriangles[n]; }
    const openvdb::Vec3I& triangle(size_t n) const { return mTriangles[n]; }

    // polygon flags accessor methods
    char& quadFlags(size_t n) { return mQuadFlags[n]; }
    const char& quadFlags(size_t n) const { return mQuadFlags[n]; }

    char& triangleFlags(size_t n) { return mTriangleFlags[n]; }
    const char& triangleFlags(size_t n) const { return mTriangleFlags[n]; }

private:
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

    PointList& pointList();
    const size_t& pointListSize() const;

    PolygonPoolList& polygonPoolList();
    const PolygonPoolList& polygonPoolList() const;
    const size_t& polygonPoolListSize() const;

    /// @brief main call
    /// @note Call with scalar typed grid.
    template<typename GridT>
    void operator()(const GridT&);


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


    boost::scoped_array<bool> markedPoints;
    boost::scoped_array<unsigned> cases;

private:

    PointList mPoints;
    PolygonPoolList mPolygons;

    size_t mPointListSize, mPolygonPoolListSize;
    double mIsovalue, mPrimAdaptivity, mSecAdaptivity;

    GridBase::ConstPtr mRefGrid, mSurfaceMaskGrid, mAdaptivityGrid;
    TreeBase::ConstPtr mAdaptivityMaskTree;

    TreeBase::Ptr mRefSignTree; //, mRefTopologyMaskTree, mSeamPointTree;

    bool mSmoothSeams, mInvertSurfaceMask;
    unsigned mPartitions, mActivePart;
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
    return signs;
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


template<class AccessorT>
inline void
correctCellSigns(unsigned char& signs, unsigned char face,
    const AccessorT& acc, Coord ijk, typename AccessorT::ValueType iso)
{
    if (face == 1) {
        ijk[2] -= 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 3) signs = ~signs;
    } else if (face == 3) {
        ijk[2] += 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 1) signs = ~signs;
    } else if (face == 2) {
        ijk[0] += 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 4) signs = ~signs;
    } else if (face == 4) {
        ijk[0] -= 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 2) signs = ~signs;
    } else if (face == 5) {
        ijk[1] -= 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 6) signs = ~signs;
    } else if (face == 6) {
        ijk[1] += 1;
        if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 5) signs = ~signs;
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
                if(leaf.isValueOn(ijk)) leaf.setValue(ijk, regionId);
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
    , mIdxTree(new IntTreeT(0))
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
    , mIdxTree(new IntTreeT(0))
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
    typename LeafManagerT::TreeType::LeafNodeType::ValueOnCIter iter;
    unsigned char signs, face;
    Coord ijk, coord;

    std::auto_ptr<Int16LeafT> signLeafPt(new Int16LeafT(ijk, 0)); 

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

                flags |= Int16(signs);

                signLeafPt->setValue(ijk, flags);
                collectedData = true; 
            }
        }


        if (collectedData) {
            mIdxAcc.touchLeaf(coord)->topologyUnion(*signLeafPt);
            mSignAcc.addLeaf(signLeafPt.release());
        }
    }
}


////////////////////////////////////////


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

inline double evalRoot(double v0, double v1, double iso) { return (iso - v0) / (v1 - v0); }


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


inline void
computePoints(const std::vector<double>& values, unsigned char signs,
    std::vector<Vec3d>& points, double iso)
{
    Vec3d avg;
    for (size_t n = 1, N = sEdgeGroupTable[signs][0] + 1; n < N; ++n) {

        int samples = 0;
        avg[0] = 0.0;
        avg[1] = 0.0;
        avg[2] = 0.0;

        if (sEdgeGroupTable[signs][1] == n) { // Edged: 0 - 1
            avg[0] += evalRoot(values[0], values[1], iso);
            ++samples;
        }

        if (sEdgeGroupTable[signs][2] == n) { // Edged: 1 - 2
            avg[0] += 1.0;
            avg[2] += evalRoot(values[1], values[2], iso);
            ++samples;
        }

        if (sEdgeGroupTable[signs][3] == n) { // Edged: 3 - 2
            avg[0] += evalRoot(values[3], values[2], iso);
            avg[2] += 1.0;
            ++samples;
        }

        if (sEdgeGroupTable[signs][4] == n) { // Edged: 0 - 3
            avg[2] += evalRoot(values[0], values[3], iso);
            ++samples;
        }

        if (sEdgeGroupTable[signs][5] == n) { // Edged: 4 - 5
            avg[0] += evalRoot(values[4], values[5], iso);
            avg[1] += 1.0;
            ++samples;
        }

        if (sEdgeGroupTable[signs][6] == n) { // Edged: 5 - 6
            avg[0] += 1.0;
            avg[1] += 1.0;
            avg[2] += evalRoot(values[5], values[6], iso);
            ++samples;
        }

        if (sEdgeGroupTable[signs][7] == n) { // Edged: 7 - 6
            avg[0] += evalRoot(values[7], values[6], iso);
            avg[1] += 1.0;
            avg[2] += 1.0;
            ++samples;
        }

        if (sEdgeGroupTable[signs][8] == n) { // Edged: 4 - 7
            avg[1] += 1.0;
            avg[2] += evalRoot(values[4], values[7], iso);
            ++samples;
        }

        if (sEdgeGroupTable[signs][9] == n) { // Edged: 0 - 4
            avg[1] += evalRoot(values[0], values[4], iso);
            ++samples;
        }

        if (sEdgeGroupTable[signs][10] == n) { // Edged: 1 - 5
            avg[0] += 1.0;
            avg[1] += evalRoot(values[1], values[5], iso);
            ++samples;
        }

        if (sEdgeGroupTable[signs][11] == n) { // Edged: 2 - 6
            avg[0] += 1.0;
            avg[1] += evalRoot(values[2], values[6], iso);
            avg[2] += 1.0;
            ++samples;
        }

        if (sEdgeGroupTable[signs][12] == n) { // Edged: 3 - 7
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

        points.push_back(avg);
    }
}

template <typename TreeT, typename LeafManagerT>
class GenPoints
{
public:
    typedef tree::ValueAccessor<const TreeT> AccessorT;

    typedef typename TreeT::template ValueConverter<int>::Type IntTreeT;
    typedef tree::ValueAccessor<IntTreeT> IntAccessorT;


    //////////


    GenPoints(const LeafManagerT& signLeafs, const TreeT& distTree,
        IntTreeT& idxTree, PointList& points, std::vector<size_t>& indices,
        const math::Transform& xform, double iso);

    void run(bool threaded = true);

    //////////


    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    const LeafManagerT& mSignLeafs;

    const TreeT& mDistTree;
    AccessorT mDistAcc;

    IntTreeT& mIdxTree;

    PointList& mPoints;
    std::vector<size_t>& mIndices;
    const math::Transform& mTransform;
    const double mIsovalue;
};


template <typename TreeT, typename LeafManagerT>
GenPoints<TreeT, LeafManagerT>::GenPoints(const LeafManagerT& signLeafs,
    const TreeT& distTree, IntTreeT& idxTree, PointList& points,
    std::vector<size_t>& indices, const math::Transform& xform, double iso)
    : mSignLeafs(signLeafs)
    , mDistTree(distTree)
    , mDistAcc(mDistTree)
    , mIdxTree(idxTree)
    , mPoints(points)
    , mIndices(indices)
    , mTransform(xform)
    , mIsovalue(iso)
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
GenPoints<TreeT, LeafManagerT>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    typename IntTreeT::LeafNodeType::ValueOnIter iter;
    IntAccessorT idxAcc(mIdxTree);

    unsigned char signs;
    Index offset;
    Coord ijk, coord;
    std::vector<Vec3d> points(4);
    std::vector<double> values(8);

    for (size_t n = range.begin(); n != range.end(); ++n) {

        coord = mSignLeafs.leaf(n).origin();
        
        const typename TreeT::LeafNodeType *leafPt = mDistAcc.probeConstLeaf(coord);
        typename IntTreeT::LeafNodeType *idxLeafPt = idxAcc.probeLeaf(coord);

        // generate points
        size_t ptnIdx = mIndices[n];
        coord.offset(TreeT::LeafNodeType::DIM - 1);

        for (iter = idxLeafPt->beginValueOn(); iter; ++iter) {

            if(iter.getValue() != 0) continue;

            iter.setValue(ptnIdx);
            iter.setValueOff();
            offset = iter.pos();
            ijk = iter.getCoord();

            points.clear();
            signs = (SIGNS & mSignLeafs.leaf(n).getValue(offset));

            if (ijk[0] < coord[0] && ijk[1] < coord[1] && ijk[2] < coord[2]) {
                collectCornerValues(*leafPt, offset, values);
            } else {
                collectCornerValues(mDistAcc, ijk, values);
            }

            computePoints(values, signs, points, mIsovalue);

            for (size_t i = 0, I = points.size(); i < I; ++i) {

                // offset by cell-origin
                points[i][0] += double(ijk[0]);
                points[i][1] += double(ijk[1]);
                points[i][2] += double(ijk[2]);

                points[i] = mTransform.indexToWorld(points[i]);

                mPoints[ptnIdx][0] = float(points[i][0]);
                mPoints[ptnIdx][1] = float(points[i][1]);
                mPoints[ptnIdx][2] = float(points[i][2]);

                ++ptnIdx;
            }

        }


        int onVoxelCount = int(idxLeafPt->onVoxelCount());
        while (onVoxelCount > 0) {

            iter = idxLeafPt->beginValueOn();
            int regionId = iter.getValue(), count = 0;

            Vec3d avg(0.0), point;

            for (; iter; ++iter) {
                if (iter.getValue() != regionId) continue;

                iter.setValue(ptnIdx);
                iter.setValueOff();
                --onVoxelCount;

                ijk = iter.getCoord();
                offset = iter.pos();

                signs = (SIGNS & mSignLeafs.leaf(n).getValue(offset));
                
                if (ijk[0] < coord[0] && ijk[1] < coord[1] && ijk[2] < coord[2]) {
                    collectCornerValues(*leafPt, offset, values);
                } else {
                    collectCornerValues(mDistAcc, ijk, values);
                }

                points.clear();
                computePoints(values, signs, points, mIsovalue);

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
    const BoolTreeT* mMask;

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
    , mMask(NULL)
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
    mMask = mask;
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
    if (mMask) {
        maskAcc.reset(new BoolTreeCAccessorT(*mMask));
    }

    // Allocate reusable leaf buffers
    BoolLeafT mask;
    Vec3LeafT gradientBuffer;
    Coord ijk, nijk, coord, end;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const Coord& origin = mSignLeafs.leaf(n).origin();

        ValueT adaptivity = mSurfaceAdaptivity;

        if (refAcc && refAcc->probeConstLeaf(origin) == NULL) {
            adaptivity = mInternalAdaptivity;
        }

        IntLeafT& idxLeaf = *idxAcc.probeLeaf(origin); 

        end[0] = origin[0] + LeafDim;
        end[1] = origin[1] + LeafDim;
        end[2] = origin[2] + LeafDim;

        mask.setValuesOff();

        // Mask off seam line adjacent voxels
        if (maskAcc) {
            const BoolLeafT* maskLeaf = maskAcc->probeConstLeaf(origin);
            if (maskLeaf != NULL) {
                typename BoolLeafT::ValueOnCIter it;
                for (it = maskLeaf->cbeginValueOn(); it; ++it) {
                    ijk = it.getCoord();
                    coord[0] = ijk[0] - (ijk[0] % 2);
                    coord[1] = ijk[1] - (ijk[1] % 2);
                    coord[2] = ijk[2] - (ijk[2] % 2);
                    mask.setActiveState(coord, true);
                }
            }
        }


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
            ijk = iter.getCoord();
            coord[0] = ijk[0] - (ijk[0] % 2);
            coord[1] = ijk[1] - (ijk[1] % 2);
            coord[2] = ijk[2] - (ijk[2] % 2);
            if(mask.isValueOn(coord)) continue;



            int flags = int(iter.getValue());
            unsigned char signs = SIGNS & flags;
            if ((flags & SEAM) || !sAdaptable[signs] || sEdgeGroupTable[signs][0] > 1) {
                mask.setActiveState(coord, true);
                continue;
            }

            for (int i = 0; i < 26; ++i) {
                nijk = ijk + util::COORD_OFFSETS[i];
                signs = SIGNS & mSignAcc.getValue(nijk);
                if (!sAdaptable[signs] || sEdgeGroupTable[signs][0] > 1) {
                    mask.setActiveState(coord, true);
                    break;
                }
            }
        }

        int dim = 2;
        // Mask off topologically ambiguous 2x2x2 voxel sub-blocks
        for (ijk[0] = origin[0]; ijk[0] < end[0]; ijk[0] += dim) {
            for (ijk[1] = origin[1]; ijk[1] < end[1]; ijk[1] += dim) {
                for (ijk[2] = origin[2]; ijk[2] < end[2]; ijk[2] += dim) {
                    if (isNonManifold(mDistAcc, ijk, mIsovalue, dim)) {
                        mask.setActiveState(ijk, true);
                    }
                }
            }
        }

        // Compute the gradient for the remaining voxels
        gradientBuffer.setValuesOff();
        for (iter = mSignLeafs.leaf(n).cbeginValueOn(); iter; ++iter) {

            ijk = iter.getCoord();
            coord[0] = ijk[0] - (ijk[0] % dim);
            coord[1] = ijk[1] - (ijk[1] % dim);
            coord[2] = ijk[2] - (ijk[2] % dim);
            if(mask.isValueOn(coord)) continue;

            Vec3T norm(math::ISGradient<math::CD_2ND>::result(mDistAcc, ijk));
            // Normalize (Vec3's normalize uses isApproxEqual, which uses abs and does more work)
            ValueT length = norm.length();
            if (length > ValueT(1.0e-7)) {
                norm *= ValueT(1.0) / length;
            }
            gradientBuffer.setValue(ijk, norm);
        }

        int regionId = 1, next_dim = dim << 1;

        // Process the first adaptivity level.
         for (ijk[0] = 0; ijk[0] < LeafDim; ijk[0] += dim) {
            coord[0] = ijk[0] - (ijk[0] % next_dim);
            for (ijk[1] = 0; ijk[1] < LeafDim; ijk[1] += dim) {
                coord[1] = ijk[1] - (ijk[1] % next_dim);
                for (ijk[2] = 0; ijk[2] < LeafDim; ijk[2] += dim) {
                    coord[2] = ijk[2] - (ijk[2] % next_dim);
                    adaptivity = adaptivityLeaf.getValue(ijk);
                    if(mask.isValueOn(ijk) || !isMergable(gradientBuffer, ijk, dim, adaptivity)) {
                        mask.setActiveState(coord, true);
                        continue;
                    }
                    mergeVoxels(idxLeaf, ijk, dim, regionId++);
                }
            }
        }


        // Process remaining adaptivity levels
       for (dim = 4; dim < LeafDim; dim = dim << 1) {
            next_dim = dim << 1;
            coord[0] = ijk[0] - (ijk[0] % next_dim);
            for (ijk[0] = origin[0]; ijk[0] < end[0]; ijk[0] += dim) {
                coord[1] = ijk[1] - (ijk[1] % next_dim);
                for (ijk[1] = origin[1]; ijk[1] < end[1]; ijk[1] += dim) {
                    coord[2] = ijk[2] - (ijk[2] % next_dim);
                    for (ijk[2] = origin[2]; ijk[2] < end[2]; ijk[2] += dim) {
                        adaptivity = adaptivityLeaf.getValue(ijk);
                        if (mask.isValueOn(ijk) || isNonManifold(mDistAcc, ijk, mIsovalue, dim) ||
                            !isMergable(gradientBuffer, ijk, dim, adaptivity)) {
                            mask.setActiveState(coord, true);
                            continue;
                        }
                        mergeVoxels(idxLeaf, ijk, dim, regionId++);
                    }
                }
            }
        }

        adaptivity = adaptivityLeaf.getValue(origin);
        if (!(mask.isValueOn(origin) || isNonManifold(mDistAcc, origin, mIsovalue, LeafDim))
            && isMergable(gradientBuffer, origin, LeafDim, adaptivity)) {
            mergeVoxels(idxLeaf, origin, LeafDim, regionId++);
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

    void done() {}

private:
    size_t mIdx;
    PolygonPool* mPolygonPool;
};


// Constructs qudas and triangles
struct AdaptivePrimBuilder
{
    AdaptivePrimBuilder()
        : mQuadIdx(0), mTriangleIdx(0), mPolygonPool(NULL), mTmpPolygonPool() {}

    void init(const size_t upperBound, PolygonPool& polygonPool)
    {
        mPolygonPool = &polygonPool;

        mTmpPolygonPool.resetQuads(upperBound);
        mTmpPolygonPool.resetTriangles(upperBound);

        mQuadIdx = 0;
        mTriangleIdx = 0;
    }

    void addPrim(const Vec4I& verts, bool reverse, char flags = 0)
    {
        if (verts[0] != verts[1] && verts[0] != verts[2] && verts[0] != verts[3]
            && verts[1] != verts[2] && verts[1] != verts[3] && verts[2] != verts[3]) {
            mTmpPolygonPool.quadFlags(mQuadIdx) = flags;
            addQuad(verts, reverse);
        } else if (
            verts[0] == verts[3] &&
            verts[1] != verts[2] &&
            verts[1] != verts[0] &&
            verts[2] != verts[0]) {
            mTmpPolygonPool.triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[1], verts[2], reverse);
        } else if (
            verts[1] == verts[2] &&
            verts[0] != verts[3] &&
            verts[0] != verts[1] &&
            verts[3] != verts[1]) {
            mTmpPolygonPool.triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[1], verts[3], reverse);
        } else if (
            verts[0] == verts[1] &&
            verts[2] != verts[3] &&
            verts[2] != verts[0] &&
            verts[3] != verts[0]) {
            mTmpPolygonPool.triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[2], verts[3], reverse);
        } else if (
            verts[2] == verts[3] &&
            verts[0] != verts[1] &&
            verts[0] != verts[2] &&
            verts[1] != verts[2]) {
            mTmpPolygonPool.triangleFlags(mTriangleIdx) = flags;
            addTriangle(verts[0], verts[1], verts[2], reverse);
        }
    }


    void done()
    {
        mPolygonPool->resetQuads(mQuadIdx);
        for (size_t  i = 0; i < mQuadIdx; ++i) {
            mPolygonPool->quad(i) = mTmpPolygonPool.quad(i);
            mPolygonPool->quadFlags(i) = mTmpPolygonPool.quadFlags(i);
        }
        mTmpPolygonPool.clearQuads();

        mPolygonPool->resetTriangles(mTriangleIdx);
        for (size_t  i = 0; i < mTriangleIdx; ++i) {
            mPolygonPool->triangle(i) = mTmpPolygonPool.triangle(i);
            mPolygonPool->triangleFlags(i) = mTmpPolygonPool.triangleFlags(i);
        }
        mTmpPolygonPool.clearTriangles();
    }

private:

    void addQuad(const Vec4I& verts, bool reverse)
    {
        if (!reverse) {
            mTmpPolygonPool.quad(mQuadIdx) = verts;
        } else {
            Vec4I& quad = mTmpPolygonPool.quad(mQuadIdx);
            quad[0] = verts[3];
            quad[1] = verts[2];
            quad[2] = verts[1];
            quad[3] = verts[0];
        }
        ++mQuadIdx;
    }

    void addTriangle(unsigned v0, unsigned v1, unsigned v2, bool reverse)
    {
        Vec3I& prim = mTmpPolygonPool.triangle(mTriangleIdx);

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
    PolygonPool mTmpPolygonPool;
};


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
        const IntTreeT& idxTree, PolygonPoolList& polygons);

    void run(bool threaded = true);


    void setRefSignTree(const Int16TreeT *r) { mRefSignTree = r; }

    //////////


    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    const LeafManagerT& mSignLeafs;
    const Int16TreeT& mSignTree;
    const IntTreeT& mIdxTree;
    const PolygonPoolList& mPolygonPoolList;

    const Int16TreeT *mRefSignTree;
 };


template<typename LeafManagerT, typename PrimBuilder>
GenPolygons<LeafManagerT, PrimBuilder>::GenPolygons(const LeafManagerT& signLeafs,
    const Int16TreeT& signTree, const IntTreeT& idxTree, PolygonPoolList& polygons)
    : mSignLeafs(signLeafs)
    , mSignTree(signTree)
    , mIdxTree(idxTree)
    , mPolygonPoolList(polygons)
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

template<typename SignAccT, typename IdxAccT, typename PrimBuilder>
inline void
constructPolygons(Int16 flags, Int16 refFlags, const Vec4i& offsets, const Coord& ijk,
    const SignAccT& signAcc, const IdxAccT& idxAcc, PrimBuilder& mesher)
{
    char tag[2];
    tag[0] = (flags & SEAM) ? POLYFLAG_FRACTURE_SEAM : 0;
    tag[1] = tag[0] | char(POLYFLAG_EXTERIOR);
   
    const bool isInside = flags & INSIDE;
    const int v0 = idxAcc.getValue(ijk);
    Coord coord;
    openvdb::Vec4I quad;
    unsigned char cell;

    if (flags & XEDGE) {


        quad[0] = v0 + offsets[0];

        coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2]; // i, j-1, k
        quad[1] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[1] += (sEdgeGroupTable[cell][5] - 1);
        }

        coord[2] -= 1; // i, j-1, k-1
        quad[2] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[2] += (sEdgeGroupTable[cell][7] - 1);
        }


        coord[1] = ijk[1]; // i, j, k-1
        quad[3] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[3] += (sEdgeGroupTable[cell][3] - 1);
        }

        mesher.addPrim(quad, isInside, tag[bool(refFlags & XEDGE)]);

    }


    if (flags & YEDGE) {

        quad[0] = v0 + offsets[1];
        coord[0] = ijk[0]; coord[1] = ijk[1]; coord[2] = ijk[2]-1; // i, j, k-1
        quad[1] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[1] += (sEdgeGroupTable[cell][12] - 1);
        }

        coord[0] -= 1; // i-1, j, k-1
        quad[2] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[2] += (sEdgeGroupTable[cell][11] - 1);
        }

        coord[2] = ijk[2]; // i-1, j, k
        quad[3] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[3] += (sEdgeGroupTable[cell][10] - 1);
        }

        mesher.addPrim(quad, isInside, tag[bool(refFlags & YEDGE)]);
    }

    if (flags & ZEDGE) {

        quad[0] = v0 + offsets[2];
        coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2]; // i, j-1, k
        quad[1] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[1] += (sEdgeGroupTable[cell][8] - 1);
        }

        coord[0] -= 1; // i-1, j-1, k
        quad[2] = idxAcc.getValue(coord);


        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[2] += (sEdgeGroupTable[cell][6] - 1);
        }

        coord[1] = ijk[1]; // i-1, j, k
        quad[3] = idxAcc.getValue(coord);

        cell = SIGNS & signAcc.getValue(coord);
        if (sEdgeGroupTable[cell][0] > 1) {
            quad[3] += (sEdgeGroupTable[cell][2] - 1);
        }

        mesher.addPrim(quad, !isInside, tag[bool(refFlags & ZEDGE)]);
    }
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

            const unsigned char cell = (SIGNS & flags);

            if (sEdgeGroupTable[cell][0] > 1) {
                offsets[0] = (sEdgeGroupTable[cell][1] - 1);
                offsets[1] = (sEdgeGroupTable[cell][9] - 1);
                offsets[2] = (sEdgeGroupTable[cell][4] - 1);
            }

            if (ijk[0] > origin[0] && ijk[1] > origin[1] && ijk[2] > origin[2]) {
                constructPolygons(flags, refFlags, offsets, ijk, *signleafPt, *idxLeafPt, mesher);
            } else {
                constructPolygons(flags, refFlags, offsets, ijk, signAcc, idxAcc, mesher);
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

            unsigned char rhsSigns = acc.getValue(ijk) & SIGNS;

            if (sEdgeGroupTable[rhsSigns][0] > 0) {
                unsigned char lhsSigns = it.getValue() & SIGNS;
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
        std::auto_ptr<openvdb::Vec3s>& newPointList,
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
    std::auto_ptr<openvdb::Vec3s>& mNewPointList;
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
    bool neighboringLeaf(const Coord&,
        const tree::ValueAccessor<const IntTreeT>&) const;

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
GenBoundaryMask<SrcTreeT>::neighboringLeaf(const Coord& ijk,
    const tree::ValueAccessor<const IntTreeT>& acc) const
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


    // edge adjacent neghbours

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

    // corner adjacent neghbours

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

        // Eval x-edges

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


        // Eval y-edges

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


        // Eval z-edges

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

} // namespace internal


////////////////////////////////////////


inline VolumeToMesh::VolumeToMesh(double isovalue, double adaptivity)
    : mPointListSize(0)
    , mPolygonPoolListSize(0)
    , mIsovalue(isovalue)
    , mPrimAdaptivity(adaptivity)
    , mSecAdaptivity(0.0)
    , mRefGrid(GridBase::ConstPtr())
    , mSurfaceMaskGrid(GridBase::ConstPtr())
    , mAdaptivityGrid(GridBase::ConstPtr())
    , mAdaptivityMaskTree(TreeBase::ConstPtr())
    , mRefSignTree(TreeBase::Ptr())
    , mSmoothSeams(false)
    , mInvertSurfaceMask(false)
    , mPartitions(1)
    , mActivePart(0)
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
        const BoolTreeT *adaptivityMaskPt = static_cast<const BoolTreeT*>(mAdaptivityMaskTree.get());
        seamMask.topologyUnion(*adaptivityMaskPt);
    }



    // Collect auxiliary data
    {
        DistLeafManagerT distLeafs(distTree);

        // Check if the isovalue is in proximity to the active voxel boundary.
        bool padActiveVoxels = false;

        if (distGrid.getGridClass() != GRID_LEVEL_SET) {
            padActiveVoxels = true;
        } else {
            padActiveVoxels = internal::needsActiveVoxePadding(distLeafs,
                mIsovalue, transform.voxelSize()[0]);
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
                    valueMask.pruneInactive();
                }

            } else { // Partition

                internal::PartGen<DistTreeT> partitioner(distLeafs, mPartitions, mActivePart);
                partitioner.run();
                valueMask.merge(partitioner.tree());
            }

            {
                if (padActiveVoxels) tools::dilateVoxels(valueMask, 3);

                BoolLeafManagerT leafs(valueMask);

                internal::SignData<DistTreeT, BoolLeafManagerT> signDataOp(distTree, leafs, isovalue);
                signDataOp.run();

                signTreePt = signDataOp.signTree();
                idxTreePt = signDataOp.idxTree();
            }

            {
                internal::GenBoundaryMask<DistTreeT> boundary(distLeafs, valueMask, *idxTreePt);
                boundary.run();

                BoolLeafManagerT bleafs(boundary.tree());

                internal::SignData<DistTreeT, BoolLeafManagerT> signDataOp(distTree, bleafs, isovalue);
                signDataOp.run();

                signTreePt->merge(*signDataOp.signTree());
                idxTreePt->merge(*signDataOp.idxTree());
            }

        } else {
            // Collect voxel-sign configurations
            
            if (padActiveVoxels) {

                BoolTreeT regionMask(false);
                regionMask.topologyUnion(distTree);
                tools::dilateVoxels(regionMask, 3);

                BoolLeafManagerT leafs(regionMask);

                internal::SignData<DistTreeT, BoolLeafManagerT> signDataOp(distTree, leafs, isovalue);
                signDataOp.run();
                
                signTreePt = signDataOp.signTree();
                idxTreePt = signDataOp.idxTree();

            } else {

                internal::SignData<DistTreeT, DistLeafManagerT> signDataOp(distTree, distLeafs, isovalue);
                signDataOp.run();

                signTreePt = signDataOp.signTree();
                idxTreePt = signDataOp.idxTree();
            }
        }

    }

    // Collect auxiliary data from active tiles
    internal::tileData(distTree, *signTreePt, *idxTreePt, isovalue);
  

    // Optionally collect auxiliary data from a reference level set.

    const Int16TreeT *refSignTreePt = NULL;

    if (mRefGrid && mRefGrid->type() == GridT::gridType()) {


        // Collect and cache auxiliary data from the reference grid. 
        if (!mRefSignTree) { 

            const GridT* refGrid = static_cast<const GridT*>(mRefGrid.get());
            const DistTreeT *refDistTreePt = &refGrid->tree();

            DistLeafManagerT refDistLeafs(*refDistTreePt);
            internal::SignData<DistTreeT, DistLeafManagerT> signDataOp(*refDistTreePt, refDistLeafs, isovalue);
            signDataOp.run();
            mRefSignTree = signDataOp.signTree();
        }

        // Get cached auxiliary data
        if (mRefSignTree) {
            refSignTreePt = static_cast<Int16TreeT*>(mRefSignTree.get());
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

        internal::MergeVoxelRegions<DistTreeT, Int16LeafManagerT>
            merge(signLeafs, *signTreePt, distTree, *idxTreePt, isovalue, DistValueT(mPrimAdaptivity));

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

    internal::GenPoints<DistTreeT, Int16LeafManagerT>
        pointOp(signLeafs, distTree, *idxTreePt, mPoints, regions, transform, mIsovalue);

    pointOp.run();




    mPolygonPoolListSize = signLeafs.leafCount();
    mPolygons.reset(new PolygonPool[mPolygonPoolListSize]);

    if (adaptive) {
        internal::GenPolygons<Int16LeafManagerT, internal::AdaptivePrimBuilder>
            mesher(signLeafs, *signTreePt, *idxTreePt, mPolygons);

        mesher.setRefSignTree(refSignTreePt);

        mesher.run();

    } else {

        internal::GenPolygons<Int16LeafManagerT, internal::UniformPrimBuilder>
            mesher(signLeafs, *signTreePt, *idxTreePt, mPolygons);

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
            if (usedPointMask[p]) indexMap[p] = usedPointCount++;
        }

        if (usedPointCount < mPointListSize) {

            // move points
            std::auto_ptr<openvdb::Vec3s> newPointList(new openvdb::Vec3s[usedPointCount]);

            internal::MovePoints movePoints(newPointList, mPoints, indexMap, usedPointMask);
            movePoints.run();

            mPointListSize = usedPointCount;
            mPoints.reset(newPointList.release());

            // update primitives
            internal::RemapIndices remap(mPolygons, mPolygonPoolListSize, indexMap);
            remap.run();
        }
    }
}


////////////////////////////////////////


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


template<typename GridType>
void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& quads,
    double isovalue)
{
    std::vector<Vec3I> triangles(0);
    volumeToMesh(grid,points, triangles, quads, isovalue, 0.0);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
