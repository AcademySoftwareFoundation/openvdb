// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file   VolumeToMesh.h
///
/// @brief  Extract polygonal surfaces from scalar volumes.
///
/// @author Mihai Alden

#ifndef OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/math/Operators.h> // for ISGradient
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/util/Util.h> // for INVALID_IDX
#include <openvdb/openvdb.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <cmath> // for std::isfinite()
#include <cstring> // for std::memset
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>


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
void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& quads,
    double isovalue = 0.0);


/// @brief Adaptively mesh any scalar grid that has a continuous isosurface.
/// @details When converting to polygons, the adaptivity threshold determines
///   how closely the isosurface is matched by the resulting mesh. Higher
///   thresholds will allow more variation in polygon size, using fewer
///   polygons to express the surface. Triangles will only be created for
///   areas of the mesh which hit the adaptivity threshold and can't be
///   represented as quads.
/// @note Do not use this method just to get a triangle mesh - use the above
///   method and post process the quad index list.
///
/// @param grid                       a scalar grid to mesh
/// @param points                     output list of world space points
/// @param triangles                  output triangle index list
/// @param quads                      output quad index list
/// @param isovalue                   determines which isosurface to mesh
/// @param adaptivity                 surface adaptivity threshold [0 to 1]
/// @param relaxDisorientedTriangles  toggle relaxing disoriented triangles during
///                                   adaptive meshing.
///
/// @throw TypeError if @a grid does not have a scalar value type
template<typename GridType>
void
volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    std::vector<Vec4I>& quads,
    double isovalue = 0.0,
    double adaptivity = 0.0,
    bool relaxDisorientedTriangles = true);


////////////////////////////////////////


/// @brief Polygon flags, used for reference based meshing.
enum { POLYFLAG_EXTERIOR = 0x1, POLYFLAG_FRACTURE_SEAM = 0x2,  POLYFLAG_SUBDIVIDED = 0x4 };


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
    std::unique_ptr<openvdb::Vec4I[]> mQuads;
    std::unique_ptr<openvdb::Vec3I[]> mTriangles;
    std::unique_ptr<char[]> mQuadFlags, mTriangleFlags;
};


/// @{
/// @brief Point and primitive list types.
using PointList = std::unique_ptr<openvdb::Vec3s[]>;
using PolygonPoolList = std::unique_ptr<PolygonPool[]>;
/// @}


////////////////////////////////////////


/// @brief Mesh any scalar grid that has a continuous isosurface.
struct VolumeToMesh
{

    /// @param isovalue                   Determines which isosurface to mesh.
    /// @param adaptivity                 Adaptivity threshold [0 to 1]
    /// @param relaxDisorientedTriangles  Toggle relaxing disoriented triangles during
    ///                                   adaptive meshing.
    VolumeToMesh(double isovalue = 0, double adaptivity = 0, bool relaxDisorientedTriangles = true);

    //////////

    /// @{
    // Mesh data accessors

    size_t pointListSize() const { return mPointListSize; }
    PointList& pointList() { return mPoints; }
    const PointList& pointList() const { return mPoints; }

    size_t polygonPoolListSize() const { return mPolygonPoolListSize; }
    PolygonPoolList& polygonPoolList() { return mPolygons; }
    const PolygonPoolList& polygonPoolList() const { return mPolygons; }

    std::vector<uint8_t>& pointFlags() { return mPointFlags; }
    const std::vector<uint8_t>& pointFlags() const { return mPointFlags; }
    /// @}


    //////////


    /// @brief Main call
    /// @note Call with scalar typed grid.
    template<typename InputGridType>
    void operator()(const InputGridType&);


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
    ///         are typically transferred to polygons that are marked with the
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

    /// @param grid A scalar grid used as a spatial multiplier for the adaptivity threshold.
    /// @note The grid's tree configuration has to match @c GridT's tree configuration.
    void setSpatialAdaptivity(const GridBase::ConstPtr& grid);


    /// @param tree A boolean tree whose active topology defines the adaptivity mask.
    /// @note The tree configuration has to match @c GridT's tree configuration.
    void setAdaptivityMask(const TreeBase::ConstPtr& tree);

private:
    // Disallow copying
    VolumeToMesh(const VolumeToMesh&);
    VolumeToMesh& operator=(const VolumeToMesh&);


    PointList mPoints;
    PolygonPoolList mPolygons;

    size_t mPointListSize, mSeamPointListSize, mPolygonPoolListSize;
    double mIsovalue, mPrimAdaptivity, mSecAdaptivity;

    GridBase::ConstPtr mRefGrid, mSurfaceMaskGrid, mAdaptivityGrid;
    TreeBase::ConstPtr mAdaptivityMaskTree;

    TreeBase::Ptr mRefSignTree, mRefIdxTree;

    bool mInvertSurfaceMask, mRelaxDisorientedTriangles;

    std::unique_ptr<uint32_t[]> mQuantizedSeamPoints;
    std::vector<uint8_t> mPointFlags;
}; // struct VolumeToMesh


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
    using Mat3d = math::Mat3d;

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


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


// Internal utility objects and implementation details

/// @cond OPENVDB_DOCS_INTERNAL

namespace volume_to_mesh_internal {

template<typename ValueType>
struct FillArray
{
    FillArray(ValueType* array, const ValueType& v) : mArray(array), mValue(v) { }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        const ValueType v = mValue;
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            mArray[n] = v;
        }
    }

    ValueType * const mArray;
    const ValueType mValue;
};


template<typename ValueType>
inline void
fillArray(ValueType* array, const ValueType& val, const size_t length)
{
    const auto grainSize = std::max<size_t>(
        length / tbb::this_task_arena::max_concurrency(), 1024);
    const tbb::blocked_range<size_t> range(0, length, grainSize);
    tbb::parallel_for(range, FillArray<ValueType>(array, val), tbb::simple_partitioner());
}


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
    const double epsilon = 0.001)
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

template<typename T>
inline bool isBoolValue() { return false; }

template<>
inline bool isBoolValue<bool>() { return true; }

template<typename T>
inline bool isInsideValue(T value, T isovalue) { return value < isovalue; }

template<>
inline bool isInsideValue<bool>(bool value, bool /*isovalue*/) { return value; }


/// @brief  Minor wrapper around the Leaf API to avoid atomic access with
///   delayed loading.
template <typename LeafT,
    bool IsBool = std::is_same<typename LeafT::ValueType, bool>::value>
struct LeafBufferAccessor
{
    using T = typename LeafT::ValueType;
    LeafBufferAccessor(const LeafT& leaf) : mData(leaf.buffer().data()) {}
    inline T get(const Index idx) const { return mData[idx]; }
    const T* const mData;
};

template <typename LeafT>
struct LeafBufferAccessor<LeafT, true>
{
    using T = bool;
    LeafBufferAccessor(const LeafT& leaf) : mLeaf(leaf) {}
    inline T get(const Index idx) const { return mLeaf.getValue(idx); }
    const LeafT& mLeaf;
};


/// @brief  Whether a coordinate does not lie at the positive edge of a leaf node.
template <typename LeafT>
bool isInternalLeafCoord(const Coord& ijk)
{
    return
        ijk[0] < int(LeafT::DIM - 1) &&
        ijk[1] < int(LeafT::DIM - 1) &&
        ijk[2] < int(LeafT::DIM - 1);
}


/// @brief Extracts the eight corner values for a cell starting at the given @ijk coordinate.
template<typename AccessorT, typename ValueT>
inline void
getCellVertexValues(const AccessorT& accessor,
    Coord ijk,
    std::array<ValueT, 8>& values)
{
    values[0] = ValueT(accessor.getValue(ijk)); // i, j, k
    ++ijk[0];
    values[1] = ValueT(accessor.getValue(ijk)); // i+1, j, k
    ++ijk[2];
    values[2] = ValueT(accessor.getValue(ijk)); // i+1, j, k+1
    --ijk[0];
    values[3] = ValueT(accessor.getValue(ijk)); // i, j, k+1
    --ijk[2]; ++ijk[1];
    values[4] = ValueT(accessor.getValue(ijk)); // i, j+1, k
    ++ijk[0];
    values[5] = ValueT(accessor.getValue(ijk)); // i+1, j+1, k
    ++ijk[2];
    values[6] = ValueT(accessor.getValue(ijk)); // i+1, j+1, k+1
    --ijk[0];
    values[7] = ValueT(accessor.getValue(ijk)); // i, j+1, k+1
}


/// @brief Extracts the eight corner values for a cell starting at the given @ijk coordinate.
template<typename LeafT, typename ValueT>
inline void
getCellVertexValues(const LeafT& leaf,
    const Index offset,
    std::array<ValueT, 8>& values)
{
    const LeafBufferAccessor<LeafT> acc(leaf);

    values[0] = ValueT(acc.get(offset));                                              // i, j, k
    values[3] = ValueT(acc.get(offset + 1));                                          // i, j, k+1
    values[4] = ValueT(acc.get(offset + LeafT::DIM));                                 // i, j+1, k
    values[7] = ValueT(acc.get(offset + LeafT::DIM + 1));                             // i, j+1, k+1
    values[1] = ValueT(acc.get(offset + (LeafT::DIM * LeafT::DIM)));                  // i+1, j, k
    values[2] = ValueT(acc.get(offset + (LeafT::DIM * LeafT::DIM) + 1));              // i+1, j, k+1
    values[5] = ValueT(acc.get(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM));     // i+1, j+1, k
    values[6] = ValueT(acc.get(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM + 1)); // i+1, j+1, k+1
}


template<typename ValueType>
inline uint8_t
computeSignFlags(const std::array<ValueType, 8>& values, const ValueType iso)
{
    unsigned signs = 0;
    signs |= isInsideValue(values[0], iso) ?   1u : 0u;
    signs |= isInsideValue(values[1], iso) ?   2u : 0u;
    signs |= isInsideValue(values[2], iso) ?   4u : 0u;
    signs |= isInsideValue(values[3], iso) ?   8u : 0u;
    signs |= isInsideValue(values[4], iso) ?  16u : 0u;
    signs |= isInsideValue(values[5], iso) ?  32u : 0u;
    signs |= isInsideValue(values[6], iso) ?  64u : 0u;
    signs |= isInsideValue(values[7], iso) ? 128u : 0u;
    return uint8_t(signs);
}


/// @brief  General method that computes the cell-sign configuration at the given
///         @c ijk coordinate.
template<typename AccessorT>
inline uint8_t
evalCellSigns(const AccessorT& accessor, const Coord& ijk, typename AccessorT::ValueType iso)
{
    unsigned signs = 0;
    Coord coord = ijk; // i, j, k
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 1u;
    coord[0] += 1; // i+1, j, k
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 2u;
    coord[2] += 1; // i+1, j, k+1
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 4u;
    coord[0] = ijk[0]; // i, j, k+1
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 8u;
    coord[1] += 1; coord[2] = ijk[2]; // i, j+1, k
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 16u;
    coord[0] += 1; // i+1, j+1, k
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 32u;
    coord[2] += 1; // i+1, j+1, k+1
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 64u;
    coord[0] = ijk[0]; // i, j+1, k+1
    if (isInsideValue(accessor.getValue(coord), iso)) signs |= 128u;
    return uint8_t(signs);
}


/// @brief  Leaf node optimized method that computes the cell-sign configuration
///         at the given local @c offset
template<typename LeafT>
inline uint8_t
evalCellSigns(const LeafT& leaf, const Index offset, typename LeafT::ValueType iso)
{
    const LeafBufferAccessor<LeafT> acc(leaf);

    unsigned signs = 0;
    if (isInsideValue(acc.get(offset), iso))                                               signs |= 1u; // i, j, k
    if (isInsideValue(acc.get(offset + (LeafT::DIM * LeafT::DIM)), iso))                   signs |= 2u; // i+1, j, k
    if (isInsideValue(acc.get(offset + (LeafT::DIM * LeafT::DIM) + 1), iso))               signs |= 4u; // i+1, j, k+1
    if (isInsideValue(acc.get(offset + 1), iso))                                           signs |= 8u; // i, j, k+1
    if (isInsideValue(acc.get(offset + LeafT::DIM), iso))                                  signs |= 16u; // i, j+1, k
    if (isInsideValue(acc.get(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM), iso))      signs |= 32u; // i+1, j+1, k
    if (isInsideValue(acc.get(offset + (LeafT::DIM * LeafT::DIM) + LeafT::DIM + 1), iso))  signs |= 64u; // i+1, j+1, k+1
    if (isInsideValue(acc.get(offset + LeafT::DIM + 1), iso))                              signs |= 128u; // i, j+1, k+1
    return uint8_t(signs);
}


/// @brief  Used to correct topological ambiguities related to two adjacent cells
///         that share an ambiguous face.
template<class AccessorT>
inline void
correctCellSigns(uint8_t& signs,
    const uint8_t face,
    const AccessorT& acc,
    Coord ijk,
    const typename AccessorT::ValueType iso)
{
    switch (int(face)) {
        case 1:
            ijk[2] -= 1;
            if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 3) signs = uint8_t(~signs);
            break;
        case 2:
            ijk[0] += 1;
            if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 4) signs = uint8_t(~signs);
            break;
        case 3:
            ijk[2] += 1;
            if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 1) signs = uint8_t(~signs);
            break;
        case 4:
            ijk[0] -= 1;
            if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 2) signs = uint8_t(~signs);
            break;
        case 5:
            ijk[1] -= 1;
            if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 6) signs = uint8_t(~signs);
            break;
        case 6:
            ijk[1] += 1;
            if (sAmbiguousFace[evalCellSigns(acc, ijk, iso)] == 5) signs = uint8_t(~signs);
            break;
        default:
            break;
    }
}


template<class AccessorT>
inline bool
isNonManifold(const AccessorT& accessor, const Coord& ijk,
    typename AccessorT::ValueType isovalue, const int dim)
{
    const int hDim = dim >> 1;
    bool m, p[8]; // Corner signs

    Coord coord = ijk; // i, j, k
    p[0] = isInsideValue(accessor.getValue(coord), isovalue);
    coord[0] += dim; // i+dim, j, k
    p[1] = isInsideValue(accessor.getValue(coord), isovalue);
    coord[2] += dim; // i+dim, j, k+dim
    p[2] = isInsideValue(accessor.getValue(coord), isovalue);
    coord[0] = ijk[0]; // i, j, k+dim
    p[3] = isInsideValue(accessor.getValue(coord), isovalue);
    coord[1] += dim; coord[2] = ijk[2]; // i, j+dim, k
    p[4] = isInsideValue(accessor.getValue(coord), isovalue);
    coord[0] += dim; // i+dim, j+dim, k
    p[5] = isInsideValue(accessor.getValue(coord), isovalue);
    coord[2] += dim; // i+dim, j+dim, k+dim
    p[6] = isInsideValue(accessor.getValue(coord), isovalue);
    coord[0] = ijk[0]; // i, j+dim, k+dim
    p[7] = isInsideValue(accessor.getValue(coord), isovalue);

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
    const int i = ijk[0], ip = ijk[0] + hDim, ipp = ijk[0] + dim;
    const int j = ijk[1], jp = ijk[1] + hDim, jpp = ijk[1] + dim;
    const int k = ijk[2], kp = ijk[2] + hDim, kpp = ijk[2] + dim;

    // edge 1
    coord.reset(ip, j, k);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[0] != m && p[1] != m) return true;

    // edge 2
    coord.reset(ipp, j, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[1] != m && p[2] != m) return true;

    // edge 3
    coord.reset(ip, j, kpp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[2] != m && p[3] != m) return true;

    // edge 4
    coord.reset(i, j, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[0] != m && p[3] != m) return true;

    // edge 5
    coord.reset(ip, jpp, k);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[4] != m && p[5] != m) return true;

    // edge 6
    coord.reset(ipp, jpp, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[5] != m && p[6] != m) return true;

    // edge 7
    coord.reset(ip, jpp, kpp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[6] != m && p[7] != m) return true;

    // edge 8
    coord.reset(i, jpp, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[7] != m && p[4] != m) return true;

    // edge 9
    coord.reset(i, jp, k);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[0] != m && p[4] != m) return true;

    // edge 10
    coord.reset(ipp, jp, k);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[1] != m && p[5] != m) return true;

    // edge 11
    coord.reset(ipp, jp, kpp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[2] != m && p[6] != m) return true;


    // edge 12
    coord.reset(i, jp, kpp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[3] != m && p[7] != m) return true;


    // Evaluate faces

    // face 1
    coord.reset(ip, jp, k);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[0] != m && p[1] != m && p[4] != m && p[5] != m) return true;

    // face 2
    coord.reset(ipp, jp, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[1] != m && p[2] != m && p[5] != m && p[6] != m) return true;

    // face 3
    coord.reset(ip, jp, kpp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[2] != m && p[3] != m && p[6] != m && p[7] != m) return true;

    // face 4
    coord.reset(i, jp, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[0] != m && p[3] != m && p[4] != m && p[7] != m) return true;

    // face 5
    coord.reset(ip, j, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[0] != m && p[1] != m && p[2] != m && p[3] != m) return true;

    // face 6
    coord.reset(ip, jpp, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[4] != m && p[5] != m && p[6] != m && p[7] != m) return true;

    // test cube center
    coord.reset(ip, jp, kp);
    m = isInsideValue(accessor.getValue(coord), isovalue);
    if (p[0] != m && p[1] != m && p[2] != m && p[3] != m &&
        p[4] != m && p[5] != m && p[6] != m && p[7] != m) return true;

    return false;
}


////////////////////////////////////////


template <class LeafType>
inline void
mergeVoxels(LeafType& leaf, const Coord& start, const int dim, const int regionId)
{
    Coord ijk;
    const Coord end = start.offsetBy(dim);

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
isMergeable(const LeafType& leaf,
    const Coord& start,
    const int dim,
    typename LeafType::ValueType::value_type adaptivity)
{
    if (adaptivity < 1e-6) return false;

    using VecT = typename LeafType::ValueType;
    Coord ijk;
    const Coord end = start.offsetBy(dim);

    std::vector<VecT> norms;
    for (ijk[0] = start[0]; ijk[0] < end[0]; ++ijk[0]) {
        for (ijk[1] = start[1]; ijk[1] < end[1]; ++ijk[1]) {
            for (ijk[2] = start[2]; ijk[2] < end[2]; ++ijk[2]) {
                if (!leaf.isValueOn(ijk)) continue;
                norms.push_back(leaf.getValue(ijk));
            }
        }
    }

    const size_t N = norms.size();
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


/// linear interpolation.
inline double evalZeroCrossing(double v0, double v1, double iso) { return (iso - v0) / (v1 - v0); }


/// @brief Computes the average cell point for a given edge group.
inline Vec3d
computePoint(const std::array<double, 8>& values,
    const unsigned char signs,
    const unsigned char edgeGroup,
    const double iso)
{
    Vec3d avg(0.0, 0.0, 0.0);
    int samples = 0;

    if (sEdgeGroupTable[signs][1] == edgeGroup) { // Edged: 0 - 1
        avg[0] += evalZeroCrossing(values[0], values[1], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][2] == edgeGroup) { // Edged: 1 - 2
        avg[0] += 1.0;
        avg[2] += evalZeroCrossing(values[1], values[2], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][3] == edgeGroup) { // Edged: 3 - 2
        avg[0] += evalZeroCrossing(values[3], values[2], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][4] == edgeGroup) { // Edged: 0 - 3
        avg[2] += evalZeroCrossing(values[0], values[3], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][5] == edgeGroup) { // Edged: 4 - 5
        avg[0] += evalZeroCrossing(values[4], values[5], iso);
        avg[1] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][6] == edgeGroup) { // Edged: 5 - 6
        avg[0] += 1.0;
        avg[1] += 1.0;
        avg[2] += evalZeroCrossing(values[5], values[6], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][7] == edgeGroup) { // Edged: 7 - 6
        avg[0] += evalZeroCrossing(values[7], values[6], iso);
        avg[1] += 1.0;
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][8] == edgeGroup) { // Edged: 4 - 7
        avg[1] += 1.0;
        avg[2] += evalZeroCrossing(values[4], values[7], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][9] == edgeGroup) { // Edged: 0 - 4
        avg[1] += evalZeroCrossing(values[0], values[4], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][10] == edgeGroup) { // Edged: 1 - 5
        avg[0] += 1.0;
        avg[1] += evalZeroCrossing(values[1], values[5], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][11] == edgeGroup) { // Edged: 2 - 6
        avg[0] += 1.0;
        avg[1] += evalZeroCrossing(values[2], values[6], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][12] == edgeGroup) { // Edged: 3 - 7
        avg[1] += evalZeroCrossing(values[3], values[7], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (samples > 1) {
        const double w = 1.0 / double(samples);
        avg *= w;
    }

    return avg;
}


/// @brief  Computes the average cell point for a given edge group, ignoring edge
///         samples present in the @c signsMask configuration.
inline int
computeMaskedPoint(Vec3d& avg,
    const std::array<double, 8>& values,
    const unsigned char signs,
    const unsigned char signsMask,
    const unsigned char edgeGroup,
    const double iso)
{
    avg = Vec3d(0.0, 0.0, 0.0);
    int samples = 0;

    if (sEdgeGroupTable[signs][1] == edgeGroup &&
        sEdgeGroupTable[signsMask][1] == 0) { // Edged: 0 - 1
        avg[0] += evalZeroCrossing(values[0], values[1], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][2] == edgeGroup &&
        sEdgeGroupTable[signsMask][2] == 0) { // Edged: 1 - 2
        avg[0] += 1.0;
        avg[2] += evalZeroCrossing(values[1], values[2], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][3] == edgeGroup &&
        sEdgeGroupTable[signsMask][3] == 0) { // Edged: 3 - 2
        avg[0] += evalZeroCrossing(values[3], values[2], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][4] == edgeGroup &&
        sEdgeGroupTable[signsMask][4] == 0) { // Edged: 0 - 3
        avg[2] += evalZeroCrossing(values[0], values[3], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][5] == edgeGroup &&
        sEdgeGroupTable[signsMask][5] == 0) { // Edged: 4 - 5
        avg[0] += evalZeroCrossing(values[4], values[5], iso);
        avg[1] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][6] == edgeGroup &&
        sEdgeGroupTable[signsMask][6] == 0) { // Edged: 5 - 6
        avg[0] += 1.0;
        avg[1] += 1.0;
        avg[2] += evalZeroCrossing(values[5], values[6], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][7] == edgeGroup &&
        sEdgeGroupTable[signsMask][7] == 0) { // Edged: 7 - 6
        avg[0] += evalZeroCrossing(values[7], values[6], iso);
        avg[1] += 1.0;
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][8] == edgeGroup &&
        sEdgeGroupTable[signsMask][8] == 0) { // Edged: 4 - 7
        avg[1] += 1.0;
        avg[2] += evalZeroCrossing(values[4], values[7], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][9] == edgeGroup &&
        sEdgeGroupTable[signsMask][9] == 0) { // Edged: 0 - 4
        avg[1] += evalZeroCrossing(values[0], values[4], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][10] == edgeGroup &&
        sEdgeGroupTable[signsMask][10] == 0) { // Edged: 1 - 5
        avg[0] += 1.0;
        avg[1] += evalZeroCrossing(values[1], values[5], iso);
        ++samples;
    }

    if (sEdgeGroupTable[signs][11] == edgeGroup &&
        sEdgeGroupTable[signsMask][11] == 0) { // Edged: 2 - 6
        avg[0] += 1.0;
        avg[1] += evalZeroCrossing(values[2], values[6], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (sEdgeGroupTable[signs][12] == edgeGroup &&
        sEdgeGroupTable[signsMask][12] == 0) { // Edged: 3 - 7
        avg[1] += evalZeroCrossing(values[3], values[7], iso);
        avg[2] += 1.0;
        ++samples;
    }

    if (samples > 1) {
        const double w = 1.0 / double(samples);
        avg *= w;
    }

    return samples;
}


/// @brief  Computes the average cell point for a given edge group, by computing
///         convex weights based on the distance from the sample point @c p.
inline Vec3d
computeWeightedPoint(const Vec3d& p,
    const std::array<double, 8>& values,
    const unsigned char signs,
    const unsigned char edgeGroup,
    const double iso)
{
    std::vector<Vec3d> samples;
    samples.reserve(8);

    Vec3d avg(0.0, 0.0, 0.0);

    if (sEdgeGroupTable[signs][1] == edgeGroup) { // Edged: 0 - 1
        avg[0] = evalZeroCrossing(values[0], values[1], iso);
        avg[1] = 0.0;
        avg[2] = 0.0;

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][2] == edgeGroup) { // Edged: 1 - 2
        avg[0] = 1.0;
        avg[1] = 0.0;
        avg[2] = evalZeroCrossing(values[1], values[2], iso);

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][3] == edgeGroup) { // Edged: 3 - 2
        avg[0] = evalZeroCrossing(values[3], values[2], iso);
        avg[1] = 0.0;
        avg[2] = 1.0;

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][4] == edgeGroup) { // Edged: 0 - 3
        avg[0] = 0.0;
        avg[1] = 0.0;
        avg[2] = evalZeroCrossing(values[0], values[3], iso);

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][5] == edgeGroup) { // Edged: 4 - 5
        avg[0] = evalZeroCrossing(values[4], values[5], iso);
        avg[1] = 1.0;
        avg[2] = 0.0;

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][6] == edgeGroup) { // Edged: 5 - 6
        avg[0] = 1.0;
        avg[1] = 1.0;
        avg[2] = evalZeroCrossing(values[5], values[6], iso);

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][7] == edgeGroup) { // Edged: 7 - 6
        avg[0] = evalZeroCrossing(values[7], values[6], iso);
        avg[1] = 1.0;
        avg[2] = 1.0;

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][8] == edgeGroup) { // Edged: 4 - 7
        avg[0] = 0.0;
        avg[1] = 1.0;
        avg[2] = evalZeroCrossing(values[4], values[7], iso);

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][9] == edgeGroup) { // Edged: 0 - 4
        avg[0] = 0.0;
        avg[1] = evalZeroCrossing(values[0], values[4], iso);
        avg[2] = 0.0;

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][10] == edgeGroup) { // Edged: 1 - 5
        avg[0] = 1.0;
        avg[1] = evalZeroCrossing(values[1], values[5], iso);
        avg[2] = 0.0;

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][11] == edgeGroup) { // Edged: 2 - 6
        avg[0] = 1.0;
        avg[1] = evalZeroCrossing(values[2], values[6], iso);
        avg[2] = 1.0;

        samples.push_back(avg);
    }

    if (sEdgeGroupTable[signs][12] == edgeGroup) { // Edged: 3 - 7
        avg[0] = 0.0;
        avg[1] = evalZeroCrossing(values[3], values[7], iso);
        avg[2] = 1.0;

        samples.push_back(avg);
    }

    assert(!samples.empty());
    if (samples.size() == 1) {
        return samples.front();
    }

    std::vector<double> weights;
    weights.reserve(samples.size());

    for (const Vec3d& s : samples) {
        weights.emplace_back((s-p).lengthSqr());
    }

    double minWeight = weights.front();
    double maxWeight = weights.front();

    for (size_t i = 1, I = weights.size(); i < I; ++i) {
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

    avg.setZero();
    for (size_t i = 0, I = samples.size(); i < I; ++i) {
        avg += samples[i] * (weights[i] / weightSum);
    }

    return avg;
}


/// @brief  Computes the average cell points defined by the sign configuration
///         @c signs and the given corner values @c values.
inline size_t
computeCellPoints(std::array<Vec3d, 4>& points,
    const std::array<double, 8>& values,
    const unsigned char signs,
    const double iso)
{
    size_t offset = 0;
    for (size_t n = 1, N = sEdgeGroupTable[signs][0] + 1; n < N; ++n, ++offset) {
        assert(offset < 4);
        points[offset] = computePoint(values, signs, uint8_t(n), iso);
    }
    return offset;
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
inline size_t
computeCellPoints(std::array<Vec3d, 4>& points,
    std::array<bool, 4>& weightedPointMask,
    const std::array<double, 8>& lhsValues,
    const std::array<double, 8>& rhsValues,
    const unsigned char lhsSigns,
    const unsigned char rhsSigns,
    const double iso,
    const size_t pointIdx,
    const uint32_t * seamPointArray)
{
    size_t offset = 0;
    for (size_t n = 1, N = sEdgeGroupTable[lhsSigns][0] + 1; n < N; ++n, ++offset)
    {
        assert(offset < 4);
        const int id = matchEdgeGroup(uint8_t(n), lhsSigns, rhsSigns);

        if (id != -1) {

            const unsigned char e = uint8_t(id);
            const uint32_t quantizedPoint = seamPointArray[pointIdx + (id - 1)];

            if ((quantizedPoint & MASK_DIRTY_BIT) && !(quantizedPoint & MASK_INVALID_BIT)) {
                const Vec3d p = unpackPoint(quantizedPoint);
                points[offset] = computeWeightedPoint(p, rhsValues, rhsSigns, e, iso);
                weightedPointMask[offset] = true;
            } else {
                points[offset] = computePoint(rhsValues, rhsSigns, e, iso);
                weightedPointMask[offset] = false;
            }

        } else {
            points[offset] = computePoint(lhsValues, lhsSigns, uint8_t(n), iso);
            weightedPointMask[offset] = false;
        }
    }
    return offset;
}


template <typename InputTreeType>
struct ComputePoints
{
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using InputValueType = typename InputLeafNodeType::ValueType;

    using Int16TreeType = typename InputTreeType::template ValueConverter<Int16>::Type;
    using Int16LeafNodeType = typename Int16TreeType::LeafNodeType;

    using Index32TreeType = typename InputTreeType::template ValueConverter<Index32>::Type;
    using Index32LeafNodeType = typename Index32TreeType::LeafNodeType;

    ComputePoints(Vec3s * pointArray,
        const InputTreeType& inputTree,
        const std::vector<Index32LeafNodeType*>& pointIndexLeafNodes,
        const std::vector<Int16LeafNodeType*>& signFlagsLeafNodes,
        const std::unique_ptr<Index32[]>& leafNodeOffsets,
        const math::Transform& xform,
        const double iso);

    void setRefData(const InputTreeType& refInputTree,
        const Index32TreeType& refPointIndexTree,
        const Int16TreeType& refSignFlagsTree,
        const uint32_t * quantizedSeamLinePoints,
        uint8_t * seamLinePointsFlags);

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    Vec3s                             * const mPoints;
    InputTreeType               const * const mInputTree;
    Index32LeafNodeType       * const * const mPointIndexNodes;
    Int16LeafNodeType   const * const * const mSignFlagsNodes;
    Index32                     const * const mNodeOffsets;
    math::Transform                     const mTransform;
    double                              const mIsovalue;
    // reference meshing data
    InputTreeType               const *       mRefInputTree;
    Index32TreeType             const *       mRefPointIndexTree;
    Int16TreeType               const *       mRefSignFlagsTree;
    uint32_t                    const *       mQuantizedSeamLinePoints;
    uint8_t                           *       mSeamLinePointsFlags;
}; // struct ComputePoints


template <typename InputTreeType>
ComputePoints<InputTreeType>::ComputePoints(
    Vec3s * pointArray,
    const InputTreeType& inputTree,
    const std::vector<Index32LeafNodeType*>& pointIndexLeafNodes,
    const std::vector<Int16LeafNodeType*>& signFlagsLeafNodes,
    const std::unique_ptr<Index32[]>& leafNodeOffsets,
    const math::Transform& xform,
    const double iso)
    : mPoints(pointArray)
    , mInputTree(&inputTree)
    , mPointIndexNodes(pointIndexLeafNodes.data())
    , mSignFlagsNodes(signFlagsLeafNodes.data())
    , mNodeOffsets(leafNodeOffsets.get())
    , mTransform(xform)
    , mIsovalue(iso)
    , mRefInputTree(nullptr)
    , mRefPointIndexTree(nullptr)
    , mRefSignFlagsTree(nullptr)
    , mQuantizedSeamLinePoints(nullptr)
    , mSeamLinePointsFlags(nullptr)
{
}

template <typename InputTreeType>
void
ComputePoints<InputTreeType>::setRefData(
    const InputTreeType& refInputTree,
    const Index32TreeType& refPointIndexTree,
    const Int16TreeType& refSignFlagsTree,
    const uint32_t * quantizedSeamLinePoints,
    uint8_t * seamLinePointsFlags)
{
    mRefInputTree = &refInputTree;
    mRefPointIndexTree = &refPointIndexTree;
    mRefSignFlagsTree = &refSignFlagsTree;
    mQuantizedSeamLinePoints = quantizedSeamLinePoints;
    mSeamLinePointsFlags = seamLinePointsFlags;
}

template <typename InputTreeType>
void
ComputePoints<InputTreeType>::operator()(const tbb::blocked_range<size_t>& range) const
{
    using InputTreeAccessor = tree::ValueAccessor<const InputTreeType>;
    using Index32TreeAccessor = tree::ValueAccessor<const Index32TreeType>;
    using Int16TreeAccessor = tree::ValueAccessor<const Int16TreeType>;

    using IndexType = typename Index32TreeType::ValueType;

    using IndexArray = std::vector<Index>;
    using IndexArrayMap = std::map<IndexType, IndexArray>;

    InputTreeAccessor inputAcc(*mInputTree);

    Vec3d xyz;
    Coord ijk;
    std::array<Vec3d, 4> points;
    std::array<bool, 4> weightedPointMask;
    std::array<double, 8> values, refValues;
    const double iso = mIsovalue;

    // reference data accessors

    std::unique_ptr<InputTreeAccessor> refInputAcc;
    std::unique_ptr<Index32TreeAccessor> refPointIndexAcc;
    std::unique_ptr<Int16TreeAccessor> refSignFlagsAcc;

    const bool hasReferenceData = mRefInputTree && mRefPointIndexTree && mRefSignFlagsTree;

    if (hasReferenceData) {
        refInputAcc.reset(new InputTreeAccessor(*mRefInputTree));
        refPointIndexAcc.reset(new Index32TreeAccessor(*mRefPointIndexTree));
        refSignFlagsAcc.reset(new Int16TreeAccessor(*mRefSignFlagsTree));
    }

    for (size_t n = range.begin(), N = range.end(); n != N; ++n)
    {
        Index32LeafNodeType& pointIndexNode = *mPointIndexNodes[n];
        const Coord& origin = pointIndexNode.origin();

        const Int16LeafNodeType& signFlagsNode = *mSignFlagsNodes[n];
        const InputLeafNodeType * inputNode = inputAcc.probeConstLeaf(origin);

        // get reference data
        const InputLeafNodeType * refInputNode = nullptr;
        const Index32LeafNodeType * refPointIndexNode = nullptr;
        const Int16LeafNodeType * refSignFlagsNode = nullptr;

        if (hasReferenceData) {
            refInputNode = refInputAcc->probeConstLeaf(origin);
            refPointIndexNode = refPointIndexAcc->probeConstLeaf(origin);
            refSignFlagsNode = refSignFlagsAcc->probeConstLeaf(origin);
        }

        IndexType pointOffset = IndexType(mNodeOffsets[n]);
        IndexArrayMap regions;

        auto*       const pidxData = pointIndexNode.buffer().data();
        const auto* const sfData = signFlagsNode.buffer().data();

        for (auto it = pointIndexNode.beginValueOn(); it; ++it)
        {
            const Index offset = it.pos();
            IndexType& id = pidxData[offset];

            if (id != 0) {
                if (id != IndexType(util::INVALID_IDX)) {
                    regions[id].push_back(offset);
                }
                continue;
            }

            id = pointOffset;

            const Int16 flags = sfData[offset];
            const uint8_t signs = uint8_t(SIGNS & flags);
            uint8_t refSigns = 0;

            if ((flags & SEAM) && refPointIndexNode && refSignFlagsNode) {
                if (refSignFlagsNode->isValueOn(offset)) {
                    refSigns = uint8_t(SIGNS & refSignFlagsNode->getValue(offset));
                }
            }

            ijk = Index32LeafNodeType::offsetToLocalCoord(offset);

            const bool inclusiveCell = inputNode && isInternalLeafCoord<InputLeafNodeType>(ijk);

            ijk += origin;

            if (inclusiveCell) getCellVertexValues(*inputNode, offset, values);
            else               getCellVertexValues(inputAcc, ijk, values);

            size_t count, weightcount;

            if (refSigns == 0) {
                count = computeCellPoints(points, values, signs, iso);
                weightcount = 0;
            } else {
                if (inclusiveCell && refInputNode) {
                    getCellVertexValues(*refInputNode, offset, refValues);
                } else {
                    getCellVertexValues(*refInputAcc, ijk, refValues);
                }
                count = computeCellPoints(points, weightedPointMask, values, refValues, signs, refSigns,
                    iso, refPointIndexNode->getValue(offset), mQuantizedSeamLinePoints);
                weightcount = count;
            }

            xyz = ijk.asVec3d();

            for (size_t i = 0; i < count; ++i) {

                Vec3d& point = points[i];

                // Checks for both NaN and inf vertex positions, i.e. any value that is not finite.
                if (!std::isfinite(point[0]) ||
                    !std::isfinite(point[1]) ||
                    !std::isfinite(point[2]))
                {
                    OPENVDB_THROW(ValueError,
                        "VolumeToMesh encountered NaNs or infs in the input VDB!"
                        " Hint: Check the input and consider using the \"Diagnostics\" tool "
                        "to detect and resolve the NaNs.");
                }

                point += xyz;
                point = mTransform.indexToWorld(point);

                Vec3s& pos = mPoints[pointOffset];
                pos[0] = float(point[0]);
                pos[1] = float(point[1]);
                pos[2] = float(point[2]);

                if (mSeamLinePointsFlags && weightcount && weightedPointMask[i]) {
                    mSeamLinePointsFlags[pointOffset] = uint8_t(1);
                }

                ++pointOffset;
            }
        }

        // generate collapsed region points
        for (auto it = regions.begin(); it != regions.end(); ++it)
        {
            Vec3d avg(0.0);

            const IndexArray& voxels = it->second;
            for (size_t i = 0, I = voxels.size(); i < I; ++i) {

                const Index offset = voxels[i];
                ijk = Index32LeafNodeType::offsetToLocalCoord(offset);

                const bool inclusiveCell = inputNode && isInternalLeafCoord<InputLeafNodeType>(ijk);

                ijk += origin;

                pidxData[offset] = pointOffset;

                const uint8_t signs = uint8_t(SIGNS & sfData[offset]);

                if (inclusiveCell) getCellVertexValues(*inputNode, offset, values);
                else               getCellVertexValues(inputAcc, ijk, values);

                computeCellPoints(points, values, signs, iso);

                avg[0] += double(ijk[0]) + points[0][0];
                avg[1] += double(ijk[1]) + points[0][1];
                avg[2] += double(ijk[2]) + points[0][2];
            }

            if (voxels.size() > 1) {
                const double w = 1.0 / double(voxels.size());
                avg *= w;
            }

            avg = mTransform.indexToWorld(avg);

            Vec3s& pos = mPoints[pointOffset];
            pos[0] = float(avg[0]);
            pos[1] = float(avg[1]);
            pos[2] = float(avg[2]);

            ++pointOffset;
        }
    }
} // ComputePoints::operator()


////////////////////////////////////////


template <typename InputTreeType>
struct SeamLineWeights
{
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using InputValueType = typename InputLeafNodeType::ValueType;

    using Int16TreeType = typename InputTreeType::template ValueConverter<Int16>::Type;
    using Int16LeafNodeType = typename Int16TreeType::LeafNodeType;

    using Index32TreeType = typename InputTreeType::template ValueConverter<Index32>::Type;
    using Index32LeafNodeType = typename Index32TreeType::LeafNodeType;

    SeamLineWeights(const std::vector<Int16LeafNodeType*>& signFlagsLeafNodes,
        const InputTreeType& inputTree,
        const Index32TreeType& refPointIndexTree,
        const Int16TreeType& refSignFlagsTree,
        uint32_t * quantizedPoints,
        InputValueType iso)
        : mSignFlagsNodes(signFlagsLeafNodes.data())
        , mInputTree(&inputTree)
        , mRefPointIndexTree(&refPointIndexTree)
        , mRefSignFlagsTree(&refSignFlagsTree)
        , mQuantizedPoints(quantizedPoints)
        , mIsovalue(iso)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        tree::ValueAccessor<const InputTreeType> inputTreeAcc(*mInputTree);
        tree::ValueAccessor<const Index32TreeType> pointIndexTreeAcc(*mRefPointIndexTree);
        tree::ValueAccessor<const Int16TreeType> signFlagsTreeAcc(*mRefSignFlagsTree);

        std::array<double, 8> values;
        const double iso = double(mIsovalue);
        Coord ijk;
        Vec3d pos;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const Int16LeafNodeType& signFlagsNode = *mSignFlagsNodes[n];
            const Coord& origin = signFlagsNode.origin();

            const Int16LeafNodeType * refSignNode = signFlagsTreeAcc.probeConstLeaf(origin);
            if (!refSignNode) continue;

            const Index32LeafNodeType* refPointIndexNode =
                pointIndexTreeAcc.probeConstLeaf(origin);
            if (!refPointIndexNode) continue;

            const InputLeafNodeType* inputNode = inputTreeAcc.probeConstLeaf(origin);

            const auto* const sfData    = signFlagsNode.buffer().data();
            const auto* const rfIdxData = refPointIndexNode->buffer().data();
            const auto* const rsData    = refSignNode->buffer().data();

            for (auto it = signFlagsNode.cbeginValueOn(); it; ++it)
            {
                const Index offset = it.pos();
                const Int16 flags = sfData[offset];

                ijk = Index32LeafNodeType::offsetToLocalCoord(offset);

                const bool inclusiveCell = inputNode && isInternalLeafCoord<InputLeafNodeType>(ijk);

                ijk += origin;

                if ((flags & SEAM) && refSignNode->isValueOn(offset)) {

                    const uint8_t lhsSigns = uint8_t(SIGNS & flags);
                    const uint8_t rhsSigns = uint8_t(SIGNS & rsData[offset]);

                    if (inclusiveCell) getCellVertexValues(*inputNode, offset, values);
                    else               getCellVertexValues(inputTreeAcc, ijk, values);

                    for (unsigned i = 1, I = sEdgeGroupTable[lhsSigns][0] + 1; i < I; ++i) {

                        const int id = matchEdgeGroup(uint8_t(i), lhsSigns, rhsSigns);

                        if (id != -1) {

                            uint32_t& data = mQuantizedPoints[rfIdxData[offset] + (id - 1)];

                            if (!(data & MASK_DIRTY_BIT)) {

                                const int samples = computeMaskedPoint(
                                    pos, values, lhsSigns, rhsSigns, uint8_t(i), iso);

                                if (samples > 0) data = packPoint(pos);
                                else             data = MASK_INVALID_BIT;

                                data |= MASK_DIRTY_BIT;
                            }
                        }
                    } // end point group loop
                }
            } // end value on loop
        } // end leaf node loop
    }

private:
    Int16LeafNodeType   const * const * const mSignFlagsNodes;
    InputTreeType               const * const mInputTree;
    Index32TreeType             const * const mRefPointIndexTree;
    Int16TreeType               const * const mRefSignFlagsTree;
    uint32_t                          * const mQuantizedPoints;
    InputValueType                      const mIsovalue;
}; // struct SeamLineWeights


template <typename TreeType>
struct SetSeamLineFlags
{
    using LeafNodeType = typename TreeType::LeafNodeType;

    SetSeamLineFlags(const std::vector<LeafNodeType*>& signFlagsLeafNodes,
        const TreeType& refSignFlagsTree)
        : mSignFlagsNodes(signFlagsLeafNodes.data())
        , mRefSignFlagsTree(&refSignFlagsTree)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        tree::ValueAccessor<const TreeType> refSignFlagsTreeAcc(*mRefSignFlagsTree);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            LeafNodeType& signFlagsNode = *mSignFlagsNodes[n];
            const Coord& origin = signFlagsNode.origin();

            const LeafNodeType* refSignNode = refSignFlagsTreeAcc.probeConstLeaf(origin);
            if (!refSignNode) continue;

            const auto* const rsData = refSignNode->buffer().data();
            auto* const sfData = signFlagsNode.buffer().data();

            for (auto it = signFlagsNode.cbeginValueOn(); it; ++it) {
                const Index offset = it.pos();

                const uint8_t rhsSigns = uint8_t(rsData[offset] & SIGNS);

                if (sEdgeGroupTable[rhsSigns][0] > 0) {

                    typename LeafNodeType::ValueType& value = sfData[offset];
                    const uint8_t lhsSigns = uint8_t(value & SIGNS);

                    if (rhsSigns != lhsSigns) {
                        value |= SEAM;
                    }
                }

            } // end value on loop

        } // end leaf node loop
    }

private:
    LeafNodeType * const * const mSignFlagsNodes;
    TreeType       const * const mRefSignFlagsTree;
}; // struct SetSeamLineFlags


template <typename BoolTreeType, typename SignDataType>
struct TransferSeamLineFlags
{
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    using SignDataTreeType = typename BoolTreeType::template ValueConverter<SignDataType>::Type;
    using SignDataLeafNodeType = typename SignDataTreeType::LeafNodeType;

    TransferSeamLineFlags(const std::vector<SignDataLeafNodeType*>& signFlagsLeafNodes,
        const BoolTreeType& maskTree)
        : mSignFlagsNodes(signFlagsLeafNodes.data())
        , mMaskTree(&maskTree)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        tree::ValueAccessor<const BoolTreeType> maskAcc(*mMaskTree);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            SignDataLeafNodeType& signFlagsNode = *mSignFlagsNodes[n];
            const Coord& origin = signFlagsNode.origin();

            const BoolLeafNodeType * maskNode = maskAcc.probeConstLeaf(origin);
            if (!maskNode) continue;

            auto* const sfData = signFlagsNode.buffer().data();

            for (auto it = signFlagsNode.cbeginValueOn(); it; ++it) {
                const Index offset = it.pos();

                if (maskNode->isValueOn(offset)) {
                    sfData[offset] |= SEAM;
                }
            } // end value on loop
        } // end leaf node loop
    }

private:
    SignDataLeafNodeType * const * const mSignFlagsNodes;
    BoolTreeType           const * const mMaskTree;
}; // struct TransferSeamLineFlags


template <typename TreeType>
struct MaskSeamLineVoxels
{
    using LeafNodeType = typename TreeType::LeafNodeType;
    using BoolTreeType = typename TreeType::template ValueConverter<bool>::Type;

    MaskSeamLineVoxels(const std::vector<LeafNodeType*>& signFlagsLeafNodes,
        const TreeType& signFlagsTree,
        BoolTreeType& mask)
        : mSignFlagsNodes(signFlagsLeafNodes.data())
        , mSignFlagsTree(&signFlagsTree)
        , mTempMask(false)
        , mMask(&mask)
    {
    }

    MaskSeamLineVoxels(MaskSeamLineVoxels& rhs, tbb::split)
        : mSignFlagsNodes(rhs.mSignFlagsNodes)
        , mSignFlagsTree(rhs.mSignFlagsTree)
        , mTempMask(false)
        , mMask(&mTempMask)
    {
    }

    void join(MaskSeamLineVoxels& rhs) { mMask->merge(*rhs.mMask); }

    void operator()(const tbb::blocked_range<size_t>& range)
    {
        using ValueOnCIter = typename LeafNodeType::ValueOnCIter;
        using ValueType = typename LeafNodeType::ValueType;

        tree::ValueAccessor<const TreeType> signFlagsAcc(*mSignFlagsTree);
        tree::ValueAccessor<BoolTreeType> maskAcc(*mMask);
        Coord ijk;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            LeafNodeType& signFlagsNode = *mSignFlagsNodes[n];
            auto* const sfData = signFlagsNode.buffer().data();

            for (ValueOnCIter it = signFlagsNode.cbeginValueOn(); it; ++it) {

                const ValueType flags = sfData[it.pos()];

                if (!(flags & SEAM) && (flags & EDGES)) {

                    ijk = it.getCoord();

                    bool isSeamLineVoxel = false;

                    if (flags & XEDGE) {
                        ijk[1] -= 1;
                        isSeamLineVoxel = (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[2] -= 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[1] += 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[2] += 1;
                    }

                    if (!isSeamLineVoxel && flags & YEDGE) {
                        ijk[2] -= 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[0] -= 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[2] += 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[0] += 1;
                    }

                    if (!isSeamLineVoxel && flags & ZEDGE) {
                        ijk[1] -= 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[0] -= 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[1] += 1;
                        isSeamLineVoxel = isSeamLineVoxel || (signFlagsAcc.getValue(ijk) & SEAM);
                        ijk[0] += 1;
                    }

                    if (isSeamLineVoxel) {
                        maskAcc.setValue(it.getCoord(), true);
                    }
                }
            } // end value on loop

        } // end leaf node loop
    }

private:
    LeafNodeType * const * const mSignFlagsNodes;
    TreeType       const * const mSignFlagsTree;
    BoolTreeType                 mTempMask;
    BoolTreeType         * const mMask;
}; // struct MaskSeamLineVoxels


template<typename SignDataTreeType>
inline void
markSeamLineData(SignDataTreeType& signFlagsTree, const SignDataTreeType& refSignFlagsTree)
{
    using SignDataType = typename SignDataTreeType::ValueType;
    using SignDataLeafNodeType = typename SignDataTreeType::LeafNodeType;
    using BoolTreeType = typename SignDataTreeType::template ValueConverter<bool>::Type;

    std::vector<SignDataLeafNodeType*> signFlagsLeafNodes;
    signFlagsTree.getNodes(signFlagsLeafNodes);

    const tbb::blocked_range<size_t> nodeRange(0, signFlagsLeafNodes.size());

    tbb::parallel_for(nodeRange,
        SetSeamLineFlags<SignDataTreeType>(signFlagsLeafNodes, refSignFlagsTree));

    BoolTreeType seamLineMaskTree(false);

    MaskSeamLineVoxels<SignDataTreeType>
        maskSeamLine(signFlagsLeafNodes, signFlagsTree, seamLineMaskTree);

    tbb::parallel_reduce(nodeRange, maskSeamLine);

    tbb::parallel_for(nodeRange,
        TransferSeamLineFlags<BoolTreeType, SignDataType>(signFlagsLeafNodes, seamLineMaskTree));
}


////////////////////////////////////////


template <typename InputGridType>
struct MergeVoxelRegions
{
    using InputTreeType = typename InputGridType::TreeType;
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using InputValueType = typename InputLeafNodeType::ValueType;

    using FloatTreeType = typename InputTreeType::template ValueConverter<float>::Type;
    using FloatLeafNodeType = typename FloatTreeType::LeafNodeType;
    using FloatGridType = Grid<FloatTreeType>;

    using Int16TreeType = typename InputTreeType::template ValueConverter<Int16>::Type;
    using Int16LeafNodeType = typename Int16TreeType::LeafNodeType;

    using Index32TreeType = typename InputTreeType::template ValueConverter<Index32>::Type;
    using Index32LeafNodeType = typename Index32TreeType::LeafNodeType;

    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    using MaskTreeType = typename InputTreeType::template ValueConverter<ValueMask>::Type;
    using MaskLeafNodeType = typename MaskTreeType::LeafNodeType;

    MergeVoxelRegions(const InputGridType& inputGrid,
        const Index32TreeType& pointIndexTree,
        const std::vector<Index32LeafNodeType*>& pointIndexLeafNodes,
        const std::vector<Int16LeafNodeType*>& signFlagsLeafNodes,
        InputValueType iso,
        float adaptivity,
        bool invertSurfaceOrientation);

    void setSpatialAdaptivity(const FloatGridType& grid)
    {
        mSpatialAdaptivityTree = &grid.tree();
        mSpatialAdaptivityTransform = &grid.transform();
    }

    void setAdaptivityMask(const BoolTreeType& mask)
    {
        mMaskTree = &mask;
    }

    void setRefSignFlagsData(const Int16TreeType& signFlagsData, float internalAdaptivity)
    {
        mRefSignFlagsTree = &signFlagsData;
        mInternalAdaptivity = internalAdaptivity;
    }

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    InputTreeType               const * const mInputTree;
    math::Transform             const * const mInputTransform;

    Index32TreeType             const * const mPointIndexTree;
    Index32LeafNodeType       * const * const mPointIndexNodes;
    Int16LeafNodeType   const * const * const mSignFlagsNodes;

    InputValueType mIsovalue;
    float mSurfaceAdaptivity, mInternalAdaptivity;
    bool mInvertSurfaceOrientation;

    FloatTreeType               const *       mSpatialAdaptivityTree;
    BoolTreeType                const *       mMaskTree;
    Int16TreeType               const *       mRefSignFlagsTree;
    math::Transform             const *       mSpatialAdaptivityTransform;
}; // struct MergeVoxelRegions


template <typename InputGridType>
MergeVoxelRegions<InputGridType>::MergeVoxelRegions(
    const InputGridType& inputGrid,
    const Index32TreeType& pointIndexTree,
    const std::vector<Index32LeafNodeType*>& pointIndexLeafNodes,
    const std::vector<Int16LeafNodeType*>& signFlagsLeafNodes,
    InputValueType iso,
    float adaptivity,
    bool invertSurfaceOrientation)
    : mInputTree(&inputGrid.tree())
    , mInputTransform(&inputGrid.transform())
    , mPointIndexTree(&pointIndexTree)
    , mPointIndexNodes(pointIndexLeafNodes.data())
    , mSignFlagsNodes(signFlagsLeafNodes.data())
    , mIsovalue(iso)
    , mSurfaceAdaptivity(adaptivity)
    , mInternalAdaptivity(adaptivity)
    , mInvertSurfaceOrientation(invertSurfaceOrientation)
    , mSpatialAdaptivityTree(nullptr)
    , mMaskTree(nullptr)
    , mRefSignFlagsTree(nullptr)
    , mSpatialAdaptivityTransform(nullptr)
{
}


template <typename InputGridType>
void
MergeVoxelRegions<InputGridType>::operator()(const tbb::blocked_range<size_t>& range) const
{
    using Vec3sType = math::Vec3<float>;
    using Vec3sLeafNodeType = typename InputLeafNodeType::template ValueConverter<Vec3sType>::Type;

    using InputTreeAccessor = tree::ValueAccessor<const InputTreeType>;
    using FloatTreeAccessor = tree::ValueAccessor<const FloatTreeType>;
    using Index32TreeAccessor = tree::ValueAccessor<const Index32TreeType>;
    using Int16TreeAccessor = tree::ValueAccessor<const Int16TreeType>;
    using BoolTreeAccessor = tree::ValueAccessor<const BoolTreeType>;

    std::unique_ptr<FloatTreeAccessor> spatialAdaptivityAcc;
    if (mSpatialAdaptivityTree && mSpatialAdaptivityTransform) {
        spatialAdaptivityAcc.reset(new FloatTreeAccessor(*mSpatialAdaptivityTree));
    }

    std::unique_ptr<BoolTreeAccessor> maskAcc;
    if (mMaskTree) {
        maskAcc.reset(new BoolTreeAccessor(*mMaskTree));
    }

    std::unique_ptr<Int16TreeAccessor> refSignFlagsAcc;
    if (mRefSignFlagsTree) {
        refSignFlagsAcc.reset(new Int16TreeAccessor(*mRefSignFlagsTree));
    }

    InputTreeAccessor inputAcc(*mInputTree);
    Index32TreeAccessor pointIndexAcc(*mPointIndexTree);

    MaskLeafNodeType mask;

    const bool invertGradientDir = mInvertSurfaceOrientation || isBoolValue<InputValueType>();
    std::unique_ptr<Vec3sLeafNodeType> gradientNode;

    Coord ijk, end;
    const int LeafDim = InputLeafNodeType::DIM;

    for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

        mask.setValuesOff();

        const Int16LeafNodeType& signFlagsNode = *mSignFlagsNodes[n];
        Index32LeafNodeType& pointIndexNode = *mPointIndexNodes[n];

        const Coord& origin = pointIndexNode.origin();
        end = origin.offsetBy(LeafDim);

        // Mask off seam line adjacent voxels
        if (maskAcc) {
            const BoolLeafNodeType* maskLeaf = maskAcc->probeConstLeaf(origin);
            if (maskLeaf != nullptr) {
                for (auto it = maskLeaf->cbeginValueOn(); it; ++it)
                {
                    mask.setActiveState(it.getCoord() & ~1u, true);
                }
            }
        }

        float adaptivity = (refSignFlagsAcc && !refSignFlagsAcc->probeConstLeaf(origin)) ?
            mInternalAdaptivity : mSurfaceAdaptivity;

        bool useGradients = adaptivity < 1.0f;

        // Set region adaptivity
        FloatLeafNodeType adaptivityLeaf(origin, adaptivity);

        if (spatialAdaptivityAcc) {
            useGradients = false;
            for (Index offset = 0; offset < FloatLeafNodeType::NUM_VALUES; ++offset) {
                ijk = adaptivityLeaf.offsetToGlobalCoord(offset);
                ijk = mSpatialAdaptivityTransform->worldToIndexCellCentered(
                    mInputTransform->indexToWorld(ijk));
                float weight = spatialAdaptivityAcc->getValue(ijk);
                float adaptivityValue = weight * adaptivity;
                if (adaptivityValue < 1.0f) useGradients = true;
                adaptivityLeaf.setValueOnly(offset, adaptivityValue);
            }
        }

        // Mask off ambiguous voxels
        for (auto it = signFlagsNode.cbeginValueOn(); it; ++it) {
            const Int16 flags = it.getValue();
            const unsigned char signs = static_cast<unsigned char>(SIGNS & int(flags));

            if ((flags & SEAM) || !sAdaptable[signs] || sEdgeGroupTable[signs][0] > 1) {

                mask.setActiveState(it.getCoord() & ~1u, true);

            } else if (flags & EDGES) {

                bool maskRegion = false;

                ijk = it.getCoord();
                if (!pointIndexAcc.isValueOn(ijk)) maskRegion = true;

                if (!maskRegion && flags & XEDGE) {
                    ijk[1] -= 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[2] -= 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[1] += 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[2] += 1;
                }

                if (!maskRegion && flags & YEDGE) {
                    ijk[2] -= 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[0] -= 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[2] += 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[0] += 1;
                }

                if (!maskRegion && flags & ZEDGE) {
                    ijk[1] -= 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[0] -= 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[1] += 1;
                    if (!maskRegion && !pointIndexAcc.isValueOn(ijk)) maskRegion = true;
                    ijk[0] += 1;
                }

                if (maskRegion) {
                    mask.setActiveState(it.getCoord() & ~1u, true);
                }
            }
        }

        // Mask off topologically ambiguous 2x2x2 voxel sub-blocks
        int dim = 2;
        for (ijk[0] = origin[0]; ijk[0] < end[0]; ijk[0] += dim) {
            for (ijk[1] = origin[1]; ijk[1] < end[1]; ijk[1] += dim) {
                for (ijk[2] = origin[2]; ijk[2] < end[2]; ijk[2] += dim) {
                    if (!mask.isValueOn(ijk) && isNonManifold(inputAcc, ijk, mIsovalue, dim)) {
                        mask.setActiveState(ijk, true);
                    }
                }
            }
        }

        // Compute the gradient for the remaining voxels

        if (useGradients) {

            if (gradientNode) {
                gradientNode->setValuesOff();
            } else {
                gradientNode.reset(new Vec3sLeafNodeType());
            }

            for (auto it = signFlagsNode.cbeginValueOn(); it; ++it)
            {
                ijk = it.getCoord();
                if (!mask.isValueOn(ijk & ~1u))
                {
                    Vec3sType dir(math::ISGradient<math::CD_2ND>::result(inputAcc, ijk));
                    dir.normalize();

                    if (invertGradientDir) {
                        dir = -dir;
                    }

                    gradientNode->setValueOn(it.pos(), dir);
                }
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

                        if (mask.isValueOn(ijk)
                            || isNonManifold(inputAcc, ijk, mIsovalue, dim)
                            || (useGradients && !isMergeable(*gradientNode, ijk, dim, adaptivity)))
                        {
                            mask.setActiveState(ijk & coordMask, true);
                        } else {
                            mergeVoxels(pointIndexNode, ijk, dim, regionId++);
                        }
                    }
                }
            }
        }
    }
} // MergeVoxelRegions::operator()


////////////////////////////////////////


// Constructs qudas
struct UniformPrimBuilder
{
    UniformPrimBuilder(): mIdx(0), mPolygonPool(nullptr) {}

    void init(const size_t upperBound, PolygonPool& quadPool)
    {
        mPolygonPool = &quadPool;
        mPolygonPool->resetQuads(upperBound);
        mIdx = 0;
    }

    template<typename IndexType>
    void addPrim(const math::Vec4<IndexType>& verts, bool reverse, char flags = 0)
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
    AdaptivePrimBuilder() : mQuadIdx(0), mTriangleIdx(0), mPolygonPool(nullptr) {}

    void init(const size_t upperBound, PolygonPool& polygonPool)
    {
        mPolygonPool = &polygonPool;
        mPolygonPool->resetQuads(upperBound);
        mPolygonPool->resetTriangles(upperBound);

        mQuadIdx = 0;
        mTriangleIdx = 0;
    }

    template<typename IndexType>
    void addPrim(const math::Vec4<IndexType>& verts, bool reverse, char flags = 0)
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

    template<typename IndexType>
    void addQuad(const math::Vec4<IndexType>& verts, bool reverse)
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
constructPolygons(
    bool invertSurfaceOrientation,
    Int16 flags,
    Int16 refFlags,
    const Vec3i& offsets,
    const Coord& ijk,
    const SignAccT& signAcc,
    const IdxAccT& idxAcc,
    PrimBuilder& mesher)
{
    using IndexType = typename IdxAccT::ValueType;

    IndexType v0 = IndexType(util::INVALID_IDX);
    const bool isActive = idxAcc.probeValue(ijk, v0);
    if (isActive == false || v0 == IndexType(util::INVALID_IDX)) return;

    char tag[2];
    tag[0] = (flags & SEAM) ? POLYFLAG_FRACTURE_SEAM : 0;
    tag[1] = tag[0] | char(POLYFLAG_EXTERIOR);

    bool isInside = flags & INSIDE;

    isInside = invertSurfaceOrientation ? !isInside : isInside;

    Coord coord = ijk;
    math::Vec4<IndexType> quad(0,0,0,0);

    if (flags & XEDGE) {

        quad[0] = v0 + offsets[0];

        // i, j-1, k
        coord[1]--;
        bool activeValues = idxAcc.probeValue(coord, quad[1]);
        uint8_t cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[1] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][5] - 1 : 0;

        // i, j-1, k-1
        coord[2]--;
        activeValues = activeValues && idxAcc.probeValue(coord, quad[2]);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[2] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][7] - 1 : 0;

        // i, j, k-1
        coord[1]++;
        activeValues = activeValues && idxAcc.probeValue(coord, quad[3]);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[3] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][3] - 1 : 0;

        if (activeValues) {
            mesher.addPrim(quad, isInside, tag[bool(refFlags & XEDGE)]);
        }

        coord[2]++; // i, j, k
    }


    if (flags & YEDGE) {

        quad[0] = v0 + offsets[1];

        // i, j, k-1
        coord[2]--;
        bool activeValues = idxAcc.probeValue(coord, quad[1]);
        uint8_t cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[1] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][12] - 1 : 0;

        // i-1, j, k-1
        coord[0]--;
        activeValues = activeValues && idxAcc.probeValue(coord, quad[2]);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[2] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][11] - 1 : 0;

        // i-1, j, k
        coord[2]++;
        activeValues = activeValues && idxAcc.probeValue(coord, quad[3]);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[3] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][10] - 1 : 0;

        if (activeValues) {
            mesher.addPrim(quad, isInside, tag[bool(refFlags & YEDGE)]);
        }

        coord[0]++; // i, j, k
    }


    if (flags & ZEDGE) {

        quad[0] = v0 + offsets[2];

        // i, j-1, k
        coord[1]--;
        bool activeValues = idxAcc.probeValue(coord, quad[1]);
        uint8_t cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[1] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][8] - 1 : 0;

        // i-1, j-1, k
        coord[0]--;
        activeValues = activeValues && idxAcc.probeValue(coord, quad[2]);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[2] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][6] - 1 : 0;

        // i-1, j, k
        coord[1]++;
        activeValues = activeValues && idxAcc.probeValue(coord, quad[3]);
        cell = uint8_t(SIGNS & signAcc.getValue(coord));
        quad[3] += sEdgeGroupTable[cell][0] > 1 ? sEdgeGroupTable[cell][2] - 1 : 0;

        if (activeValues) {
            mesher.addPrim(quad, !isInside, tag[bool(refFlags & ZEDGE)]);
        }
    }
}


////////////////////////////////////////


template<typename InputTreeType>
struct MaskTileBorders
{
    using InputValueType = typename InputTreeType::ValueType;
    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;


    MaskTileBorders(const InputTreeType& inputTree, InputValueType iso,
        BoolTreeType& mask, const Vec4i* tileArray)
        : mInputTree(&inputTree)
        , mIsovalue(iso)
        , mTempMask(false)
        , mMask(&mask)
        , mTileArray(tileArray)
    {
    }

    MaskTileBorders(MaskTileBorders& rhs, tbb::split)
        : mInputTree(rhs.mInputTree)
        , mIsovalue(rhs.mIsovalue)
        , mTempMask(false)
        , mMask(&mTempMask)
        , mTileArray(rhs.mTileArray)
    {
    }

    void join(MaskTileBorders& rhs) { mMask->merge(*rhs.mMask); }

    void operator()(const tbb::blocked_range<size_t>&);

private:
    InputTreeType   const * const mInputTree;
    InputValueType          const mIsovalue;
    BoolTreeType                  mTempMask;
    BoolTreeType          * const mMask;
    Vec4i           const * const mTileArray;
}; // MaskTileBorders


template<typename InputTreeType>
void
MaskTileBorders<InputTreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    tree::ValueAccessor<const InputTreeType> inputTreeAcc(*mInputTree);

    CoordBBox region, bbox;
    Coord ijk, nijk;

    for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

        const Vec4i& tile = mTileArray[n];

        bbox.min()[0] = tile[0];
        bbox.min()[1] = tile[1];
        bbox.min()[2] = tile[2];
        bbox.max() = bbox.min();
        bbox.max().offset(tile[3]);

        InputValueType value = mInputTree->background();

        const bool isInside = isInsideValue(inputTreeAcc.getValue(bbox.min()), mIsovalue);
        const int valueDepth = inputTreeAcc.getValueDepth(bbox.min());

        // eval x-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[0];

        bool processRegion = true;
        if (valueDepth >= inputTreeAcc.getValueDepth(nijk)) {
            processRegion = isInside != isInsideValue(inputTreeAcc.getValue(nijk), mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.expand(1);
            region.min()[0] = region.max()[0] = ijk[0];
            mMask->fill(region, false);
        }


        ijk = bbox.min();
        --ijk[0];

        processRegion = true;
        if (valueDepth >= inputTreeAcc.getValueDepth(ijk)) {
            processRegion = (!inputTreeAcc.probeValue(ijk, value)
                && isInside != isInsideValue(value, mIsovalue));
        }

        if (processRegion) {
            region = bbox;
            region.expand(1);
            region.min()[0] = region.max()[0] = ijk[0];
            mMask->fill(region, false);
        }


        // eval y-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[1];

        processRegion = true;
        if (valueDepth >= inputTreeAcc.getValueDepth(nijk)) {
            processRegion = isInside != isInsideValue(inputTreeAcc.getValue(nijk), mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.expand(1);
            region.min()[1] = region.max()[1] = ijk[1];
            mMask->fill(region, false);
        }


        ijk = bbox.min();
        --ijk[1];

        processRegion = true;
        if (valueDepth >= inputTreeAcc.getValueDepth(ijk)) {
            processRegion = (!inputTreeAcc.probeValue(ijk, value)
                && isInside != isInsideValue(value, mIsovalue));
        }

        if (processRegion) {
            region = bbox;
            region.expand(1);
            region.min()[1] = region.max()[1] = ijk[1];
            mMask->fill(region, false);
        }


        // eval z-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[2];

        processRegion = true;
        if (valueDepth >= inputTreeAcc.getValueDepth(nijk)) {
            processRegion = isInside != isInsideValue(inputTreeAcc.getValue(nijk), mIsovalue);
        }

        if (processRegion) {
            region = bbox;
            region.expand(1);
            region.min()[2] = region.max()[2] = ijk[2];
            mMask->fill(region, false);
        }

        ijk = bbox.min();
        --ijk[2];

        processRegion = true;
        if (valueDepth >= inputTreeAcc.getValueDepth(ijk)) {
            processRegion = (!inputTreeAcc.probeValue(ijk, value)
                && isInside != isInsideValue(value, mIsovalue));
        }

        if (processRegion) {
            region = bbox;
            region.expand(1);
            region.min()[2] = region.max()[2] = ijk[2];
            mMask->fill(region, false);
        }
    }
} // MaskTileBorders::operator()


template<typename InputTreeType>
inline void
maskActiveTileBorders(const InputTreeType& inputTree,
    const typename InputTreeType::ValueType iso,
    typename InputTreeType::template ValueConverter<bool>::Type& mask)
{
    typename InputTreeType::ValueOnCIter tileIter(inputTree);
    tileIter.setMaxDepth(InputTreeType::ValueOnCIter::LEAF_DEPTH - 1);

    size_t tileCount = 0;
    for ( ; tileIter; ++tileIter) {
        ++tileCount;
    }

    if (tileCount > 0) {
        std::unique_ptr<Vec4i[]> tiles(new Vec4i[tileCount]);

        CoordBBox bbox;
        size_t index = 0;

        tileIter = inputTree.cbeginValueOn();
        tileIter.setMaxDepth(InputTreeType::ValueOnCIter::LEAF_DEPTH - 1);

        for (; tileIter; ++tileIter) {
            Vec4i& tile = tiles[index++];
            tileIter.getBoundingBox(bbox);
            tile[0] = bbox.min()[0];
            tile[1] = bbox.min()[1];
            tile[2] = bbox.min()[2];
            tile[3] = bbox.max()[0] - bbox.min()[0];
        }

        MaskTileBorders<InputTreeType> op(inputTree, iso, mask, tiles.get());
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, tileCount), op);
    }
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


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


struct LeafNodeVoxelOffsets
{
    using IndexVector = std::vector<Index>;

    template<typename LeafNodeType>
    void constructOffsetList();

    /// Return internal core voxel offsets.
    const IndexVector& core() const { return mCore; }


    /// Return front face voxel offsets.
    const IndexVector& minX() const { return mMinX; }

    /// Return back face voxel offsets.
    const IndexVector& maxX() const { return mMaxX; }


    /// Return bottom face voxel offsets.
    const IndexVector& minY() const { return mMinY; }

    /// Return top face voxel offsets.
    const IndexVector& maxY() const { return mMaxY; }


    /// Return left face voxel offsets.
    const IndexVector& minZ() const { return mMinZ; }

    /// Return right face voxel offsets.
    const IndexVector& maxZ() const { return mMaxZ; }


    /// Return voxel offsets with internal neighbours in x + 1.
    const IndexVector& internalNeighborsX() const { return mInternalNeighborsX; }

    /// Return voxel offsets with internal neighbours in y + 1.
    const IndexVector& internalNeighborsY() const { return mInternalNeighborsY; }

    /// Return voxel offsets with internal neighbours in z + 1.
    const IndexVector& internalNeighborsZ() const { return mInternalNeighborsZ; }


private:
    IndexVector mCore, mMinX, mMaxX, mMinY, mMaxY, mMinZ, mMaxZ,
        mInternalNeighborsX, mInternalNeighborsY, mInternalNeighborsZ;
}; // struct LeafNodeOffsets


template<typename LeafNodeType>
inline void
LeafNodeVoxelOffsets::constructOffsetList()
{
    // internal core voxels
    mCore.clear();
    mCore.reserve((LeafNodeType::DIM - 2) * (LeafNodeType::DIM - 2));

    for (Index x = 1; x < (LeafNodeType::DIM - 1); ++x) {
        const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
        for (Index y = 1; y < (LeafNodeType::DIM - 1); ++y) {
            const Index offsetXY = offsetX + (y << LeafNodeType::LOG2DIM);
            for (Index z = 1; z < (LeafNodeType::DIM - 1); ++z) {
                mCore.push_back(offsetXY + z);
            }
        }
    }

    // internal neighbors in x + 1
    mInternalNeighborsX.clear();
    mInternalNeighborsX.reserve(LeafNodeType::SIZE - (LeafNodeType::DIM * LeafNodeType::DIM));

    for (Index x = 0; x < (LeafNodeType::DIM - 1); ++x) {
        const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
        for (Index y = 0; y < LeafNodeType::DIM; ++y) {
            const Index offsetXY = offsetX + (y << LeafNodeType::LOG2DIM);
            for (Index z = 0; z < LeafNodeType::DIM; ++z) {
                mInternalNeighborsX.push_back(offsetXY + z);
            }
        }
    }

    // internal neighbors in y + 1
    mInternalNeighborsY.clear();
    mInternalNeighborsY.reserve(LeafNodeType::SIZE - (LeafNodeType::DIM * LeafNodeType::DIM));

    for (Index x = 0; x < LeafNodeType::DIM; ++x) {
        const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
        for (Index y = 0; y < (LeafNodeType::DIM - 1); ++y) {
            const Index offsetXY = offsetX + (y << LeafNodeType::LOG2DIM);
            for (Index z = 0; z < LeafNodeType::DIM; ++z) {
                mInternalNeighborsY.push_back(offsetXY + z);
            }
        }
    }

    // internal neighbors in z + 1
    mInternalNeighborsZ.clear();
    mInternalNeighborsZ.reserve(LeafNodeType::SIZE - (LeafNodeType::DIM * LeafNodeType::DIM));

    for (Index x = 0; x < LeafNodeType::DIM; ++x) {
        const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
        for (Index y = 0; y < LeafNodeType::DIM; ++y) {
            const Index offsetXY = offsetX + (y << LeafNodeType::LOG2DIM);
            for (Index z = 0; z < (LeafNodeType::DIM - 1); ++z) {
                mInternalNeighborsZ.push_back(offsetXY + z);
            }
        }
    }

    // min x
    mMinX.clear();
    mMinX.reserve(LeafNodeType::DIM * LeafNodeType::DIM);
    {
        for (Index y = 0; y < LeafNodeType::DIM; ++y) {
            const Index offsetXY = (y << LeafNodeType::LOG2DIM);
            for (Index z = 0; z < LeafNodeType::DIM; ++z) {
                mMinX.push_back(offsetXY + z);
            }
        }
    }

    // max x
    mMaxX.clear();
    mMaxX.reserve(LeafNodeType::DIM * LeafNodeType::DIM);
    {
        const Index offsetX = (LeafNodeType::DIM - 1) << (2 * LeafNodeType::LOG2DIM);
        for (Index y = 0; y < LeafNodeType::DIM; ++y) {
            const Index offsetXY = offsetX + (y << LeafNodeType::LOG2DIM);
            for (Index z = 0; z < LeafNodeType::DIM; ++z) {
                mMaxX.push_back(offsetXY + z);
            }
        }
    }

    // min y
    mMinY.clear();
    mMinY.reserve(LeafNodeType::DIM * LeafNodeType::DIM);
    {
        for (Index x = 0; x < LeafNodeType::DIM; ++x) {
            const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
            for (Index z = 0; z < (LeafNodeType::DIM - 1); ++z) {
                mMinY.push_back(offsetX + z);
            }
        }
    }

    // max y
    mMaxY.clear();
    mMaxY.reserve(LeafNodeType::DIM * LeafNodeType::DIM);
    {
        const Index offsetY = (LeafNodeType::DIM - 1) << LeafNodeType::LOG2DIM;
        for (Index x = 0; x < LeafNodeType::DIM; ++x) {
            const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
            for (Index z = 0; z < (LeafNodeType::DIM - 1); ++z) {
                mMaxY.push_back(offsetX + offsetY + z);
            }
        }
    }

    // min z
    mMinZ.clear();
    mMinZ.reserve(LeafNodeType::DIM * LeafNodeType::DIM);
    {
        for (Index x = 0; x < LeafNodeType::DIM; ++x) {
            const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
            for (Index y = 0; y < LeafNodeType::DIM; ++y) {
                const Index offsetXY = offsetX + (y << LeafNodeType::LOG2DIM);
                mMinZ.push_back(offsetXY);
            }
        }
    }

    // max z
    mMaxZ.clear();
    mMaxZ.reserve(LeafNodeType::DIM * LeafNodeType::DIM);
    {
        for (Index x = 0; x < LeafNodeType::DIM; ++x) {
            const Index offsetX = x << (2 * LeafNodeType::LOG2DIM);
            for (Index y = 0; y < LeafNodeType::DIM; ++y) {
                const Index offsetXY = offsetX + (y << LeafNodeType::LOG2DIM);
                mMaxZ.push_back(offsetXY + (LeafNodeType::DIM - 1));
            }
        }
    }
}


////////////////////////////////////////


/// Utility method to marks all voxels that share an edge.
template<typename AccessorT, int _AXIS>
struct VoxelEdgeAccessor
{
    enum { AXIS = _AXIS };
    AccessorT& acc;

    VoxelEdgeAccessor(AccessorT& _acc) : acc(_acc) {}

    void set(Coord ijk) {
        if (_AXIS == 0) {                   // x + 1 edge
            acc.setActiveState(ijk);
            --ijk[1]; // set i, j-1, k
            acc.setActiveState(ijk);
            --ijk[2]; // set i, j-1, k-1
            acc.setActiveState(ijk);
            ++ijk[1]; // set i, j, k-1
            acc.setActiveState(ijk);
        } else if (_AXIS == 1) {            // y + 1 edge
            acc.setActiveState(ijk);
            --ijk[2]; // set i, j, k-1
            acc.setActiveState(ijk);
            --ijk[0]; // set i-1, j, k-1
            acc.setActiveState(ijk);
            ++ijk[2]; // set i-1, j, k
            acc.setActiveState(ijk);
        } else {                            // z + 1 edge
            acc.setActiveState(ijk);
            --ijk[1]; // set i, j-1, k
            acc.setActiveState(ijk);
            --ijk[0]; // set i-1, j-1, k
            acc.setActiveState(ijk);
            ++ijk[1]; // set i-1, j, k
            acc.setActiveState(ijk);
        }
    }
};


/// Utility method to check for sign changes along the x + 1, y + 1 or z + 1 directions.
/// The direction is determined by the @a edgeAcc parameter. Only voxels that have internal
/// neighbours are evaluated.
template<typename VoxelEdgeAcc, typename LeafNodeT>
void
evalInternalVoxelEdges(VoxelEdgeAcc& edgeAcc,
    const LeafNodeT& leafnode,
    const LeafNodeVoxelOffsets& voxels,
    const typename LeafNodeT::ValueType iso)
{
    Index nvo = 1; // neighbour voxel offset, z + 1 direction assumed initially.
    const std::vector<Index>* offsets = &voxels.internalNeighborsZ();

    if (VoxelEdgeAcc::AXIS == 0) { // x + 1 direction
        nvo = LeafNodeT::DIM * LeafNodeT::DIM;
        offsets = &voxels.internalNeighborsX();
    } else if (VoxelEdgeAcc::AXIS == 1) { // y + 1 direction
        nvo = LeafNodeT::DIM;
        offsets = &voxels.internalNeighborsY();
    }

    const LeafBufferAccessor<LeafNodeT> lhsAcc(leafnode);

    for (size_t n = 0, N = offsets->size(); n < N; ++n) {
        const Index& pos = (*offsets)[n];
        const bool isActive = leafnode.isValueOn(pos) || leafnode.isValueOn(pos + nvo);
        if (isActive && (isInsideValue(lhsAcc.get(pos), iso) !=
                isInsideValue(lhsAcc.get((pos + nvo)), iso))) {
            edgeAcc.set(leafnode.offsetToGlobalCoord(pos));
        }
    }
}


/// Utility method to check for sign changes along the x + 1, y + 1 or z + 1 directions.
/// The direction is determined by the @a edgeAcc parameter. All voxels that reside in the
/// specified leafnode face: back, top or right are evaluated.
template<typename LeafNodeT, typename TreeAcc, typename VoxelEdgeAcc>
void
evalExternalVoxelEdges(VoxelEdgeAcc& edgeAcc,
    const TreeAcc& acc,
    const LeafNodeT& lhsNode,
    const LeafNodeVoxelOffsets& voxels,
    const typename LeafNodeT::ValueType iso)
{
    const std::vector<Index>* lhsOffsets = &voxels.maxX();
    const std::vector<Index>* rhsOffsets = &voxels.minX();
    Coord ijk = lhsNode.origin();

    if (VoxelEdgeAcc::AXIS == 0) { // back leafnode face
        ijk[0] += LeafNodeT::DIM;
    } else if (VoxelEdgeAcc::AXIS == 1) { // top leafnode face
        ijk[1] += LeafNodeT::DIM;
        lhsOffsets = &voxels.maxY();
        rhsOffsets = &voxels.minY();
    } else if (VoxelEdgeAcc::AXIS == 2) { // right leafnode face
        ijk[2] += LeafNodeT::DIM;
        lhsOffsets = &voxels.maxZ();
        rhsOffsets = &voxels.minZ();
    }

    typename LeafNodeT::ValueType value;
    const LeafNodeT* rhsNodePt = acc.probeConstLeaf(ijk);

    const LeafBufferAccessor<LeafNodeT> lhsAcc(lhsNode);

    if (rhsNodePt) {
        const LeafBufferAccessor<LeafNodeT> rhsAcc(*rhsNodePt);

        for (size_t n = 0, N = lhsOffsets->size(); n < N; ++n) {
            const Index& pos = (*lhsOffsets)[n];
            bool isActive = lhsNode.isValueOn(pos) || rhsNodePt->isValueOn((*rhsOffsets)[n]);
            if (isActive && (isInsideValue(lhsAcc.get(pos), iso) !=
                    isInsideValue(rhsAcc.get((*rhsOffsets)[n]), iso))) {
                edgeAcc.set(lhsNode.offsetToGlobalCoord(pos));
            }
        }
    } else if (!acc.probeValue(ijk, value)) {
        const bool inside = isInsideValue(value, iso);
        for (size_t n = 0, N = lhsOffsets->size(); n < N; ++n) {
            const Index& pos = (*lhsOffsets)[n];
            if (lhsNode.isValueOn(pos) && (inside != isInsideValue(lhsAcc.get(pos), iso))) {
                edgeAcc.set(lhsNode.offsetToGlobalCoord(pos));
            }
        }
    }
}


/// Utility method to check for sign changes along the x - 1, y - 1 or z - 1 directions.
/// The direction is determined by the @a edgeAcc parameter. All voxels that reside in the
/// specified leafnode face: front, bottom or left are evaluated.
template<typename LeafNodeT, typename TreeAcc, typename VoxelEdgeAcc>
void
evalExternalVoxelEdgesInv(VoxelEdgeAcc& edgeAcc,
    const TreeAcc& acc,
    const LeafNodeT& leafnode,
    const LeafNodeVoxelOffsets& voxels,
    const typename LeafNodeT::ValueType iso)
{
    Coord ijk = leafnode.origin();
    if      (VoxelEdgeAcc::AXIS == 0) --ijk[0]; // front leafnode face
    else if (VoxelEdgeAcc::AXIS == 1) --ijk[1]; // bottom leafnode face
    else if (VoxelEdgeAcc::AXIS == 2) --ijk[2]; // left leafnode face

    typename LeafNodeT::ValueType value;
    if (!acc.probeConstLeaf(ijk) && !acc.probeValue(ijk, value)) {

        const std::vector<Index>* offsets = &voxels.internalNeighborsX();
        if      (VoxelEdgeAcc::AXIS == 1) offsets = &voxels.internalNeighborsY();
        else if (VoxelEdgeAcc::AXIS == 2) offsets = &voxels.internalNeighborsZ();

        const LeafBufferAccessor<LeafNodeT> lhsAcc(leafnode);

        const bool inside = isInsideValue(value, iso);
        for (size_t n = 0, N = offsets->size(); n < N; ++n) {

            const Index& pos = (*offsets)[n];
            if (leafnode.isValueOn(pos)
                && (inside != isInsideValue(lhsAcc.get(pos), iso)))
            {
                ijk = leafnode.offsetToGlobalCoord(pos);
                if      (VoxelEdgeAcc::AXIS == 0) --ijk[0];
                else if (VoxelEdgeAcc::AXIS == 1) --ijk[1];
                else if (VoxelEdgeAcc::AXIS == 2) --ijk[2];

                edgeAcc.set(ijk);
            }
        }
    }
}


template<typename InputTreeType>
struct IdentifyIntersectingVoxels
{
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using InputValueType = typename InputLeafNodeType::ValueType;

    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;

    IdentifyIntersectingVoxels(
        const InputTreeType& inputTree,
        const std::vector<const InputLeafNodeType*>& inputLeafNodes,
        const LeafNodeVoxelOffsets& offsets,
        BoolTreeType& intersectionTree,
        InputValueType iso);

    IdentifyIntersectingVoxels(IdentifyIntersectingVoxels&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(const IdentifyIntersectingVoxels& rhs) {
        mIntersectionAccessor.tree().merge(rhs.mIntersectionAccessor.tree());
    }

private:
    tree::ValueAccessor<const InputTreeType>    mInputAccessor;
    InputLeafNodeType const * const * const     mInputNodes;

    BoolTreeType                        mIntersectionTree;
    tree::ValueAccessor<BoolTreeType>   mIntersectionAccessor;

    const LeafNodeVoxelOffsets& mOffsets;
    const InputValueType        mIsovalue;
}; // struct IdentifyIntersectingVoxels


template<typename InputTreeType>
IdentifyIntersectingVoxels<InputTreeType>::IdentifyIntersectingVoxels(
    const InputTreeType& inputTree,
    const std::vector<const InputLeafNodeType*>& inputLeafNodes,
    const LeafNodeVoxelOffsets& offsets,
    BoolTreeType& intersectionTree,
    InputValueType iso)
    : mInputAccessor(inputTree)
    , mInputNodes(inputLeafNodes.data())
    , mIntersectionTree(false)
    , mIntersectionAccessor(intersectionTree)
    , mOffsets(offsets)
    , mIsovalue(iso)
{
}


template<typename InputTreeType>
IdentifyIntersectingVoxels<InputTreeType>::IdentifyIntersectingVoxels(
    IdentifyIntersectingVoxels& rhs, tbb::split)
    : mInputAccessor(rhs.mInputAccessor.tree())
    , mInputNodes(rhs.mInputNodes)
    , mIntersectionTree(false)
    , mIntersectionAccessor(mIntersectionTree) // use local tree.
    , mOffsets(rhs.mOffsets)
    , mIsovalue(rhs.mIsovalue)
{
}


template<typename InputTreeType>
void
IdentifyIntersectingVoxels<InputTreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    VoxelEdgeAccessor<tree::ValueAccessor<BoolTreeType>, 0> xEdgeAcc(mIntersectionAccessor);
    VoxelEdgeAccessor<tree::ValueAccessor<BoolTreeType>, 1> yEdgeAcc(mIntersectionAccessor);
    VoxelEdgeAccessor<tree::ValueAccessor<BoolTreeType>, 2> zEdgeAcc(mIntersectionAccessor);

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const InputLeafNodeType& node = *mInputNodes[n];

        // internal x + 1 voxel edges
        evalInternalVoxelEdges(xEdgeAcc, node, mOffsets, mIsovalue);
        // internal y + 1 voxel edges
        evalInternalVoxelEdges(yEdgeAcc, node, mOffsets, mIsovalue);
        // internal z + 1 voxel edges
        evalInternalVoxelEdges(zEdgeAcc, node, mOffsets, mIsovalue);

        // external x + 1 voxels edges (back face)
        evalExternalVoxelEdges(xEdgeAcc, mInputAccessor, node, mOffsets, mIsovalue);
        // external y + 1 voxels edges (top face)
        evalExternalVoxelEdges(yEdgeAcc, mInputAccessor, node, mOffsets, mIsovalue);
        // external z + 1 voxels edges (right face)
        evalExternalVoxelEdges(zEdgeAcc, mInputAccessor, node, mOffsets, mIsovalue);

        // The remaining edges are only checked if the leafnode neighbour, in the
        // corresponding direction, is an inactive tile.

        // external x - 1 voxels edges (front face)
        evalExternalVoxelEdgesInv(xEdgeAcc, mInputAccessor, node, mOffsets, mIsovalue);
        // external y - 1 voxels edges (bottom face)
        evalExternalVoxelEdgesInv(yEdgeAcc, mInputAccessor, node, mOffsets, mIsovalue);
        // external z - 1 voxels edges (left face)
        evalExternalVoxelEdgesInv(zEdgeAcc, mInputAccessor, node, mOffsets, mIsovalue);
    }
} // IdentifyIntersectingVoxels::operator()


template<typename InputTreeType>
inline void
identifySurfaceIntersectingVoxels(
    typename InputTreeType::template ValueConverter<bool>::Type& intersectionTree,
    const InputTreeType& inputTree,
    typename InputTreeType::ValueType isovalue)
{
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;

    std::vector<const InputLeafNodeType*> inputLeafNodes;
    inputTree.getNodes(inputLeafNodes);

    LeafNodeVoxelOffsets offsets;
    offsets.constructOffsetList<InputLeafNodeType>();

    IdentifyIntersectingVoxels<InputTreeType> op(
        inputTree, inputLeafNodes, offsets, intersectionTree, isovalue);

    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, inputLeafNodes.size()), op);

    maskActiveTileBorders(inputTree, isovalue, intersectionTree);
}


////////////////////////////////////////


template<typename InputTreeType>
struct MaskIntersectingVoxels
{
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using InputValueType = typename InputLeafNodeType::ValueType;

    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    MaskIntersectingVoxels(
        const InputTreeType& inputTree,
        const std::vector<BoolLeafNodeType*>& nodes,
        BoolTreeType& intersectionTree,
        InputValueType iso);

    MaskIntersectingVoxels(MaskIntersectingVoxels&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(const MaskIntersectingVoxels& rhs) {
        mIntersectionAccessor.tree().merge(rhs.mIntersectionAccessor.tree());
    }

private:
    tree::ValueAccessor<const InputTreeType>    mInputAccessor;
    BoolLeafNodeType const * const * const      mNodes;

    BoolTreeType                        mIntersectionTree;
    tree::ValueAccessor<BoolTreeType>   mIntersectionAccessor;

    const InputValueType mIsovalue;
}; // struct MaskIntersectingVoxels


template<typename InputTreeType>
MaskIntersectingVoxels<InputTreeType>::MaskIntersectingVoxels(
    const InputTreeType& inputTree,
    const std::vector<BoolLeafNodeType*>& nodes,
    BoolTreeType& intersectionTree,
    InputValueType iso)
    : mInputAccessor(inputTree)
    , mNodes(nodes.data())
    , mIntersectionTree(false)
    , mIntersectionAccessor(intersectionTree)
    , mIsovalue(iso)
{
}


template<typename InputTreeType>
MaskIntersectingVoxels<InputTreeType>::MaskIntersectingVoxels(
    MaskIntersectingVoxels& rhs, tbb::split)
    : mInputAccessor(rhs.mInputAccessor.tree())
    , mNodes(rhs.mNodes)
    , mIntersectionTree(false)
    , mIntersectionAccessor(mIntersectionTree) // use local tree.
    , mIsovalue(rhs.mIsovalue)
{
}


template<typename InputTreeType>
void
MaskIntersectingVoxels<InputTreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    VoxelEdgeAccessor<tree::ValueAccessor<BoolTreeType>, 0> xEdgeAcc(mIntersectionAccessor);
    VoxelEdgeAccessor<tree::ValueAccessor<BoolTreeType>, 1> yEdgeAcc(mIntersectionAccessor);
    VoxelEdgeAccessor<tree::ValueAccessor<BoolTreeType>, 2> zEdgeAcc(mIntersectionAccessor);

    Coord ijk;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const BoolLeafNodeType& node = *mNodes[n];

        for (auto it = node.cbeginValueOn(); it; ++it) {

            if (!it.getValue()) {

                ijk = it.getCoord();

                const bool inside = isInsideValue(mInputAccessor.getValue(ijk), mIsovalue);

                if (inside != isInsideValue(mInputAccessor.getValue(ijk.offsetBy(1, 0, 0)), mIsovalue)) {
                    xEdgeAcc.set(ijk);
                }

                if (inside != isInsideValue(mInputAccessor.getValue(ijk.offsetBy(0, 1, 0)), mIsovalue)) {
                    yEdgeAcc.set(ijk);
                }

                if (inside != isInsideValue(mInputAccessor.getValue(ijk.offsetBy(0, 0, 1)), mIsovalue)) {
                    zEdgeAcc.set(ijk);
                }
            }
        }
    }
} // MaskIntersectingVoxels::operator()


template<typename BoolTreeType>
struct MaskBorderVoxels
{
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    MaskBorderVoxels(const BoolTreeType& maskTree,
        const std::vector<BoolLeafNodeType*>& maskNodes,
        BoolTreeType& borderTree)
        : mMaskTree(&maskTree)
        , mMaskNodes(maskNodes.data())
        , mTmpBorderTree(false)
        , mBorderTree(&borderTree) {}

    MaskBorderVoxels(MaskBorderVoxels& rhs, tbb::split)
        : mMaskTree(rhs.mMaskTree)
        , mMaskNodes(rhs.mMaskNodes)
        , mTmpBorderTree(false)
        , mBorderTree(&mTmpBorderTree) {}

    void join(MaskBorderVoxels& rhs) { mBorderTree->merge(*rhs.mBorderTree); }

    void operator()(const tbb::blocked_range<size_t>& range)
    {
        tree::ValueAccessor<const BoolTreeType> maskAcc(*mMaskTree);
        tree::ValueAccessor<BoolTreeType> borderAcc(*mBorderTree);
        Coord ijk;

        for (size_t n = range.begin(); n != range.end(); ++n) {

            const BoolLeafNodeType& node = *mMaskNodes[n];

            for (auto it = node.cbeginValueOn(); it; ++it) {

                ijk = it.getCoord();

                const bool lhs = it.getValue();
                bool rhs = lhs;

                bool isEdgeVoxel = false;

                ijk[2] += 1; // i, j, k+1
                isEdgeVoxel = (maskAcc.probeValue(ijk, rhs) && lhs != rhs);

                ijk[1] += 1; // i, j+1, k+1
                isEdgeVoxel = isEdgeVoxel || (maskAcc.probeValue(ijk, rhs) && lhs != rhs);

                ijk[0] += 1; // i+1, j+1, k+1
                isEdgeVoxel = isEdgeVoxel || (maskAcc.probeValue(ijk, rhs) && lhs != rhs);

                ijk[1] -= 1; // i+1, j, k+1
                isEdgeVoxel = isEdgeVoxel || (maskAcc.probeValue(ijk, rhs) && lhs != rhs);


                ijk[2] -= 1; // i+1, j, k
                isEdgeVoxel = isEdgeVoxel || (maskAcc.probeValue(ijk, rhs) && lhs != rhs);

                ijk[1] += 1; // i+1, j+1, k
                isEdgeVoxel = isEdgeVoxel || (maskAcc.probeValue(ijk, rhs) && lhs != rhs);

                ijk[0] -= 1; // i, j+1, k
                isEdgeVoxel = isEdgeVoxel || (maskAcc.probeValue(ijk, rhs) && lhs != rhs);

                if (isEdgeVoxel) {
                    ijk[1] -= 1; // i, j, k
                    borderAcc.setValue(ijk, true);
                }
            }
        }
    }

private:
    BoolTreeType             const * const mMaskTree;
    BoolLeafNodeType const * const * const mMaskNodes;

    BoolTreeType                           mTmpBorderTree;
    BoolTreeType                   * const mBorderTree;
}; // struct MaskBorderVoxels


template<typename BoolTreeType>
struct SyncMaskValues
{
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    SyncMaskValues(const std::vector<BoolLeafNodeType*>& nodes, const BoolTreeType& mask)
        : mNodes(nodes.data())
        , mMaskTree(&mask) {}

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        using ValueOnIter = typename BoolLeafNodeType::ValueOnIter;

        tree::ValueAccessor<const BoolTreeType> maskTreeAcc(*mMaskTree);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            BoolLeafNodeType& node = *mNodes[n];

            const BoolLeafNodeType * maskNode = maskTreeAcc.probeConstLeaf(node.origin());

            if (maskNode) {
                for (ValueOnIter it = node.beginValueOn(); it; ++it) {
                    const Index pos = it.pos();
                    if (maskNode->getValue(pos)) {
                        node.setValueOnly(pos, true);
                    }
                }
            }
        }
    }

private:
    BoolLeafNodeType * const * const mNodes;
    BoolTreeType       const * const mMaskTree;
}; // struct SyncMaskValues


////////////////////////////////////////


template<typename BoolTreeType>
struct MaskSurface
{
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    MaskSurface(const std::vector<BoolLeafNodeType*>& nodes,
        const BoolTreeType& mask,
        const math::Transform& inputTransform,
        const math::Transform& maskTransform,
        const bool invert)
        : mNodes(nodes.data())
        , mMaskTree(&mask)
        , mInputTransform(inputTransform)
        , mMaskTransform(maskTransform)
        , mMatchingTransforms(mInputTransform == mMaskTransform)
        , mInvertMask(invert) {}

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        using ValueOnIter = typename BoolLeafNodeType::ValueOnIter;

        tree::ValueAccessor<const BoolTreeType> maskTreeAcc(*mMaskTree);

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            BoolLeafNodeType& node = *mNodes[n];

            if (mMatchingTransforms) {

                const BoolLeafNodeType * maskNode = maskTreeAcc.probeConstLeaf(node.origin());

                if (maskNode) {

                    for (ValueOnIter it = node.beginValueOn(); it; ++it) {
                        const Index pos = it.pos();
                        if (maskNode->isValueOn(pos) == mInvertMask) {
                            node.setValueOnly(pos, true);
                        }
                    }

                } else {

                    if (maskTreeAcc.isValueOn(node.origin()) == mInvertMask) {
                        for (ValueOnIter it = node.beginValueOn(); it; ++it) {
                            node.setValueOnly(it.pos(), true);
                        }
                    }

                }

            } else {

                Coord ijk;

                for (ValueOnIter it = node.beginValueOn(); it; ++it) {

                    ijk = mMaskTransform.worldToIndexCellCentered(
                            mInputTransform.indexToWorld(it.getCoord()));

                    if (maskTreeAcc.isValueOn(ijk) == mInvertMask) {
                        node.setValueOnly(it.pos(), true);
                    }
                }

            }
        }
    }

private:
    BoolLeafNodeType * const * const mNodes;
    BoolTreeType       const * const mMaskTree;
    const math::Transform&           mInputTransform;
    const math::Transform&           mMaskTransform;
    const bool                       mMatchingTransforms;
    const bool                       mInvertMask;
}; // struct MaskSurface


template<typename InputGridType>
inline void
applySurfaceMask(
    typename InputGridType::TreeType::template ValueConverter<bool>::Type& intersectionTree,
    typename InputGridType::TreeType::template ValueConverter<bool>::Type& borderTree,
    const InputGridType& inputGrid,
    const GridBase::ConstPtr& maskGrid,
    const bool invertMask,
    const typename InputGridType::ValueType isovalue)
{
    using InputTreeType = typename InputGridType::TreeType;
    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;
    using BoolGridType = Grid<BoolTreeType>;

    if (!maskGrid)                                    return;
    if (maskGrid->type() != BoolGridType::gridType()) return;

    const math::Transform& transform = inputGrid.transform();
    const InputTreeType& inputTree = inputGrid.tree();

    const BoolGridType * surfaceMask = static_cast<const BoolGridType*>(maskGrid.get());

    const BoolTreeType& maskTree = surfaceMask->tree();
    const math::Transform& maskTransform = surfaceMask->transform();

    // mark masked voxels

    std::vector<BoolLeafNodeType*> intersectionLeafNodes;
    intersectionTree.getNodes(intersectionLeafNodes);

    const tbb::blocked_range<size_t> intersectionRange(0, intersectionLeafNodes.size());

    tbb::parallel_for(intersectionRange,
        MaskSurface<BoolTreeType>(
            intersectionLeafNodes, maskTree, transform, maskTransform, invertMask));


    // mask surface-mask border

    MaskBorderVoxels<BoolTreeType> borderOp(
        intersectionTree, intersectionLeafNodes, borderTree);
    tbb::parallel_reduce(intersectionRange, borderOp);


    // recompute isosurface intersection mask

    BoolTreeType tmpIntersectionTree(false);

    MaskIntersectingVoxels<InputTreeType> op(
        inputTree, intersectionLeafNodes, tmpIntersectionTree, isovalue);

    tbb::parallel_reduce(intersectionRange, op);

    std::vector<BoolLeafNodeType*> tmpIntersectionLeafNodes;
    tmpIntersectionTree.getNodes(tmpIntersectionLeafNodes);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, tmpIntersectionLeafNodes.size()),
        SyncMaskValues<BoolTreeType>(tmpIntersectionLeafNodes, intersectionTree));

    intersectionTree.clear();
    intersectionTree.merge(tmpIntersectionTree);
}


////////////////////////////////////////


template<typename InputTreeType>
struct ComputeAuxiliaryData
{
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using InputValueType = typename InputLeafNodeType::ValueType;

    using BoolLeafNodeType = tree::LeafNode<bool, InputLeafNodeType::LOG2DIM>;

    using Int16TreeType = typename InputTreeType::template ValueConverter<Int16>::Type;
    using Index32TreeType = typename InputTreeType::template ValueConverter<Index32>::Type;


    ComputeAuxiliaryData(const InputTreeType& inputTree,
        const std::vector<const BoolLeafNodeType*>& intersectionLeafNodes,
        Int16TreeType& signFlagsTree,
        Index32TreeType& pointIndexTree,
        InputValueType iso);

    ComputeAuxiliaryData(ComputeAuxiliaryData&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(const ComputeAuxiliaryData& rhs) {
        mSignFlagsAccessor.tree().merge(rhs.mSignFlagsAccessor.tree());
        mPointIndexAccessor.tree().merge(rhs.mPointIndexAccessor.tree());
    }

private:
    tree::ValueAccessor<const InputTreeType>    mInputAccessor;
    BoolLeafNodeType const * const * const      mIntersectionNodes;

    Int16TreeType                           mSignFlagsTree;
    tree::ValueAccessor<Int16TreeType>      mSignFlagsAccessor;
    Index32TreeType                         mPointIndexTree;
    tree::ValueAccessor<Index32TreeType>    mPointIndexAccessor;

    const InputValueType mIsovalue;
};


template<typename InputTreeType>
ComputeAuxiliaryData<InputTreeType>::ComputeAuxiliaryData(
    const InputTreeType& inputTree,
    const std::vector<const BoolLeafNodeType*>& intersectionLeafNodes,
    Int16TreeType& signFlagsTree,
    Index32TreeType& pointIndexTree,
    InputValueType iso)
    : mInputAccessor(inputTree)
    , mIntersectionNodes(intersectionLeafNodes.data())
    , mSignFlagsTree(0)
    , mSignFlagsAccessor(signFlagsTree)
    , mPointIndexTree(std::numeric_limits<Index32>::max())
    , mPointIndexAccessor(pointIndexTree)
    , mIsovalue(iso)
{
    pointIndexTree.root().setBackground(std::numeric_limits<Index32>::max(), false);
}


template<typename InputTreeType>
ComputeAuxiliaryData<InputTreeType>::ComputeAuxiliaryData(ComputeAuxiliaryData& rhs, tbb::split)
    : mInputAccessor(rhs.mInputAccessor.tree())
    , mIntersectionNodes(rhs.mIntersectionNodes)
    , mSignFlagsTree(0)
    , mSignFlagsAccessor(mSignFlagsTree)
    , mPointIndexTree(std::numeric_limits<Index32>::max())
    , mPointIndexAccessor(mPointIndexTree)
    , mIsovalue(rhs.mIsovalue)
{
}


template<typename InputTreeType>
void
ComputeAuxiliaryData<InputTreeType>::operator()(const tbb::blocked_range<size_t>& range)
{
    using Int16LeafNodeType = typename Int16TreeType::LeafNodeType;

    Coord ijk;
    std::array<InputValueType, 8> cellVertexValues;
    std::unique_ptr<Int16LeafNodeType> signsNodePt(nullptr);

    for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

        const BoolLeafNodeType& maskNode = *mIntersectionNodes[n];
        const Coord& origin = maskNode.origin();

        const InputLeafNodeType* leafPt = mInputAccessor.probeConstLeaf(origin);

        if (!signsNodePt.get()) signsNodePt.reset(new Int16LeafNodeType(origin, 0));
        else                    signsNodePt->setOrigin(origin);

        bool updatedNode = false;

        for (auto it = maskNode.cbeginValueOn(); it; ++it) {

            const Index pos = it.pos();
            ijk = BoolLeafNodeType::offsetToLocalCoord(pos);

            const bool inclusiveCell = leafPt && isInternalLeafCoord<InputLeafNodeType>(ijk);

            if (inclusiveCell) getCellVertexValues(*leafPt, pos, cellVertexValues);
            else               getCellVertexValues(mInputAccessor, origin + ijk, cellVertexValues);

            uint8_t signFlags = computeSignFlags(cellVertexValues, mIsovalue);

            if (signFlags != 0 && signFlags != 0xFF) {

                const bool inside = signFlags & 0x1;

                int edgeFlags = inside ? INSIDE : 0;

                if (!it.getValue()) {
                    edgeFlags |= inside != ((signFlags & 0x02) != 0) ? XEDGE : 0;
                    edgeFlags |= inside != ((signFlags & 0x10) != 0) ? YEDGE : 0;
                    edgeFlags |= inside != ((signFlags & 0x08) != 0) ? ZEDGE : 0;
                }

                const uint8_t ambiguousCellFlags = sAmbiguousFace[signFlags];
                if (ambiguousCellFlags != 0) {
                    correctCellSigns(signFlags, ambiguousCellFlags, mInputAccessor,
                        origin + ijk, mIsovalue);
                }

                edgeFlags |= int(signFlags);

                signsNodePt->setValueOn(pos, Int16(edgeFlags));
                updatedNode = true;
            }
        }

        if (updatedNode) {
            typename Index32TreeType::LeafNodeType* idxNode =
                mPointIndexAccessor.touchLeaf(origin);
            idxNode->topologyUnion(*signsNodePt);

            // zero fill
            auto* const idxData = idxNode->buffer().data();
            for (auto it = idxNode->beginValueOn(); it; ++it) {
                idxData[it.pos()] = 0;
            }

            mSignFlagsAccessor.addLeaf(signsNodePt.release());
        }
    }
} // ComputeAuxiliaryData::operator()


template<typename InputTreeType>
inline void
computeAuxiliaryData(
    typename InputTreeType::template ValueConverter<Int16>::Type& signFlagsTree,
    typename InputTreeType::template ValueConverter<Index32>::Type& pointIndexTree,
    const typename InputTreeType::template ValueConverter<bool>::Type& intersectionTree,
    const InputTreeType& inputTree,
    typename InputTreeType::ValueType isovalue)
{
    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;
    using BoolLeafNodeType = typename BoolTreeType::LeafNodeType;

    std::vector<const BoolLeafNodeType*> intersectionLeafNodes;
    intersectionTree.getNodes(intersectionLeafNodes);

    ComputeAuxiliaryData<InputTreeType> op(
        inputTree, intersectionLeafNodes, signFlagsTree, pointIndexTree, isovalue);

    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, intersectionLeafNodes.size()), op);
}


////////////////////////////////////////


template<Index32 LeafNodeLog2Dim>
struct LeafNodePointCount
{
    using Int16LeafNodeType = tree::LeafNode<Int16, LeafNodeLog2Dim>;

    LeafNodePointCount(const std::vector<Int16LeafNodeType*>& leafNodes,
        std::unique_ptr<Index32[]>& leafNodeCount)
        : mLeafNodes(leafNodes.data())
        , mData(leafNodeCount.get()) {}

    void operator()(const tbb::blocked_range<size_t>& range) const {

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            Index32 count = 0;

            Int16 const * p = mLeafNodes[n]->buffer().data();
            Int16 const * const endP = p + Int16LeafNodeType::SIZE;

            while (p < endP) {
                count += Index32(sEdgeGroupTable[(SIGNS & *p)][0]);
                ++p;
            }

            mData[n] = count;
        }
    }

private:
    Int16LeafNodeType * const * const mLeafNodes;
    Index32 *mData;
}; // struct LeafNodePointCount


template<typename PointIndexLeafNode>
struct AdaptiveLeafNodePointCount
{
    using Int16LeafNodeType = tree::LeafNode<Int16, PointIndexLeafNode::LOG2DIM>;

    AdaptiveLeafNodePointCount(const std::vector<PointIndexLeafNode*>& pointIndexNodes,
        const std::vector<Int16LeafNodeType*>& signDataNodes,
        std::unique_ptr<Index32[]>& leafNodeCount)
        : mPointIndexNodes(pointIndexNodes.data())
        , mSignDataNodes(signDataNodes.data())
        , mData(leafNodeCount.get()) {}

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        using IndexType = typename PointIndexLeafNode::ValueType;

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const PointIndexLeafNode& node = *mPointIndexNodes[n];
            const Int16LeafNodeType& signNode = *mSignDataNodes[n];

            size_t count = 0;

            std::set<IndexType> uniqueRegions;

            for (typename PointIndexLeafNode::ValueOnCIter it = node.cbeginValueOn(); it; ++it) {

                IndexType id = it.getValue();

                if (id == 0) {
                    count += size_t(sEdgeGroupTable[(SIGNS & signNode.getValue(it.pos()))][0]);
                } else if (id != IndexType(util::INVALID_IDX)) {
                    uniqueRegions.insert(id);
                }
            }

            mData[n] = Index32(count + uniqueRegions.size());
        }
    }

private:
    PointIndexLeafNode const * const * const mPointIndexNodes;
    Int16LeafNodeType  const * const * const mSignDataNodes;
    Index32 *mData;
}; // struct AdaptiveLeafNodePointCount


template<typename PointIndexLeafNode>
struct MapPoints
{
    using Int16LeafNodeType = tree::LeafNode<Int16, PointIndexLeafNode::LOG2DIM>;

    MapPoints(const std::vector<PointIndexLeafNode*>& pointIndexNodes,
        const std::vector<Int16LeafNodeType*>& signDataNodes,
        std::unique_ptr<Index32[]>& leafNodeCount)
        : mPointIndexNodes(pointIndexNodes.data())
        , mSignDataNodes(signDataNodes.data())
        , mData(leafNodeCount.get()) {}

    void operator()(const tbb::blocked_range<size_t>& range) const {

        for (size_t n = range.begin(), N = range.end(); n != N; ++n) {

            const Int16LeafNodeType& signNode = *mSignDataNodes[n];
            PointIndexLeafNode& indexNode = *mPointIndexNodes[n];

            Index32 pointOffset = mData[n];

            for (auto it = indexNode.beginValueOn(); it; ++it) {
                const Index pos = it.pos();
                indexNode.setValueOnly(pos, pointOffset);
                const int signs = SIGNS & int(signNode.getValue(pos));
                pointOffset += Index32(sEdgeGroupTable[signs][0]);
            }
        }
    }

private:
    PointIndexLeafNode       * const * const mPointIndexNodes;
    Int16LeafNodeType  const * const * const mSignDataNodes;
    Index32                          * const mData;
}; // struct MapPoints


template<typename TreeType, typename PrimBuilder>
struct ComputePolygons
{
    using Int16TreeType = typename TreeType::template ValueConverter<Int16>::Type;
    using Int16LeafNodeType = typename Int16TreeType::LeafNodeType;

    using Index32TreeType = typename TreeType::template ValueConverter<Index32>::Type;
    using Index32LeafNodeType = typename Index32TreeType::LeafNodeType;

    ComputePolygons(
        const std::vector<Int16LeafNodeType*>& signFlagsLeafNodes,
        const Int16TreeType& signFlagsTree,
        const Index32TreeType& idxTree,
        PolygonPoolList& polygons,
        bool invertSurfaceOrientation);

    void setRefSignTree(const Int16TreeType * r) { mRefSignFlagsTree = r; }

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    Int16LeafNodeType * const * const mSignFlagsLeafNodes;
    Int16TreeType       const * const mSignFlagsTree;
    Int16TreeType       const *       mRefSignFlagsTree;
    Index32TreeType     const * const mIndexTree;
    PolygonPoolList           * const mPolygonPoolList;
    bool                        const mInvertSurfaceOrientation;
}; // struct ComputePolygons


template<typename TreeType, typename PrimBuilder>
ComputePolygons<TreeType, PrimBuilder>::ComputePolygons(
    const std::vector<Int16LeafNodeType*>& signFlagsLeafNodes,
    const Int16TreeType& signFlagsTree,
    const Index32TreeType& idxTree,
    PolygonPoolList& polygons,
    bool invertSurfaceOrientation)
    : mSignFlagsLeafNodes(signFlagsLeafNodes.data())
    , mSignFlagsTree(&signFlagsTree)
    , mRefSignFlagsTree(nullptr)
    , mIndexTree(&idxTree)
    , mPolygonPoolList(&polygons)
    , mInvertSurfaceOrientation(invertSurfaceOrientation)
{
}

template<typename InputTreeType, typename PrimBuilder>
void
ComputePolygons<InputTreeType, PrimBuilder>::operator()(const tbb::blocked_range<size_t>& range) const
{
    using Int16ValueAccessor = tree::ValueAccessor<const Int16TreeType>;
    Int16ValueAccessor signAcc(*mSignFlagsTree);

    tree::ValueAccessor<const Index32TreeType> idxAcc(*mIndexTree);

    const bool invertSurfaceOrientation = mInvertSurfaceOrientation;

    PrimBuilder mesher;
    size_t edgeCount;
    Coord ijk, origin;

    // reference data
    std::unique_ptr<Int16ValueAccessor> refSignAcc;
    if (mRefSignFlagsTree) refSignAcc.reset(new Int16ValueAccessor(*mRefSignFlagsTree));

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const Int16LeafNodeType& node = *mSignFlagsLeafNodes[n];
        origin = node.origin();

        // Get an upper bound on the number of primitives.
        edgeCount = 0;
        typename Int16LeafNodeType::ValueOnCIter iter = node.cbeginValueOn();
        for (; iter; ++iter) {
            if (iter.getValue() & XEDGE) ++edgeCount;
            if (iter.getValue() & YEDGE) ++edgeCount;
            if (iter.getValue() & ZEDGE) ++edgeCount;
        }

        if (edgeCount == 0) continue;

        mesher.init(edgeCount, (*mPolygonPoolList)[n]);

        const Int16LeafNodeType *signleafPt = signAcc.probeConstLeaf(origin);
        const Index32LeafNodeType *idxLeafPt = idxAcc.probeConstLeaf(origin);

        if (!signleafPt || !idxLeafPt) continue;

        const Int16LeafNodeType *refSignLeafPt = nullptr;
        if (refSignAcc) refSignLeafPt = refSignAcc->probeConstLeaf(origin);

        Vec3i offsets;

        for (iter = node.cbeginValueOn(); iter; ++iter) {
            ijk = iter.getCoord();

            const Int16 flags = iter.getValue();
            if (!(flags & 0xE00)) continue;

            Int16 refFlags = 0;
            if (refSignLeafPt) {
                refFlags = refSignLeafPt->getValue(iter.pos());
            }

            const uint8_t cell = uint8_t(SIGNS & flags);

            if (sEdgeGroupTable[cell][0] > 1) {
                offsets[0] = (sEdgeGroupTable[cell][1] - 1);
                offsets[1] = (sEdgeGroupTable[cell][9] - 1);
                offsets[2] = (sEdgeGroupTable[cell][4] - 1);
            }
            else {
                offsets.setZero();
            }

            if (ijk[0] > origin[0] && ijk[1] > origin[1] && ijk[2] > origin[2]) {
                constructPolygons(invertSurfaceOrientation,
                    flags, refFlags, offsets, ijk, *signleafPt, *idxLeafPt, mesher);
            } else {
                constructPolygons(invertSurfaceOrientation,
                    flags, refFlags, offsets, ijk, signAcc, idxAcc, mesher);
            }
        }

        mesher.done();
    }

} // ComputePolygons::operator()


////////////////////////////////////////


template<typename T>
struct CopyArray
{
    CopyArray(T * outputArray, const T * inputArray, size_t outputOffset = 0)
        : mOutputArray(outputArray), mInputArray(inputArray), mOutputOffset(outputOffset)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& inputArrayRange) const
    {
        const size_t offset = mOutputOffset;
        for (size_t n = inputArrayRange.begin(), N = inputArrayRange.end(); n < N; ++n) {
            mOutputArray[offset + n] = mInputArray[n];
        }
    }

private:
    T             * const mOutputArray;
    T       const * const mInputArray;
    size_t          const mOutputOffset;
}; // struct CopyArray


struct FlagAndCountQuadsToSubdivide
{
    FlagAndCountQuadsToSubdivide(PolygonPoolList& polygons,
        const std::vector<uint8_t>& pointFlags,
        std::unique_ptr<openvdb::Vec3s[]>& points,
        std::unique_ptr<unsigned[]>& numQuadsToDivide)
        : mPolygonPoolList(&polygons)
        , mPointFlags(pointFlags.data())
        , mPoints(points.get())
        , mNumQuadsToDivide(numQuadsToDivide.get())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            PolygonPool& polygons = (*mPolygonPoolList)[n];

            unsigned count = 0;

            // count and tag nonplanar seam line quads.
            for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {

                char& flags = polygons.quadFlags(i);

                if ((flags & POLYFLAG_FRACTURE_SEAM) && !(flags & POLYFLAG_EXTERIOR)) {

                    Vec4I& quad = polygons.quad(i);

                    const bool edgePoly = mPointFlags[quad[0]] || mPointFlags[quad[1]]
                        || mPointFlags[quad[2]] || mPointFlags[quad[3]];

                    if (!edgePoly) continue;

                    const Vec3s& p0 = mPoints[quad[0]];
                    const Vec3s& p1 = mPoints[quad[1]];
                    const Vec3s& p2 = mPoints[quad[2]];
                    const Vec3s& p3 = mPoints[quad[3]];

                    if (!isPlanarQuad(p0, p1, p2, p3, 1e-6f)) {
                        flags |= POLYFLAG_SUBDIVIDED;
                        count++;
                    }
                }
            }

            mNumQuadsToDivide[n] = count;
        }
    }

private:
    PolygonPoolList       * const mPolygonPoolList;
    uint8_t         const * const mPointFlags;
    Vec3s           const * const mPoints;
    unsigned              * const mNumQuadsToDivide;
}; // struct FlagAndCountQuadsToSubdivide


struct SubdivideQuads
{
    SubdivideQuads(PolygonPoolList& polygons,
        const std::unique_ptr<openvdb::Vec3s[]>& points,
        size_t pointCount,
        std::unique_ptr<openvdb::Vec3s[]>& centroids,
        std::unique_ptr<unsigned[]>& numQuadsToDivide,
        std::unique_ptr<unsigned[]>& centroidOffsets)
        : mPolygonPoolList(&polygons)
        , mPoints(points.get())
        , mCentroids(centroids.get())
        , mNumQuadsToDivide(numQuadsToDivide.get())
        , mCentroidOffsets(centroidOffsets.get())
        , mPointCount(pointCount)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            PolygonPool& polygons = (*mPolygonPoolList)[n];

            const size_t nonplanarCount = size_t(mNumQuadsToDivide[n]);

            if (nonplanarCount > 0) {

                PolygonPool tmpPolygons;
                tmpPolygons.resetQuads(polygons.numQuads() - nonplanarCount);
                tmpPolygons.resetTriangles(polygons.numTriangles() + size_t(4) * nonplanarCount);

                size_t offset = mCentroidOffsets[n];

                size_t triangleIdx = 0;

                for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {

                    const char quadFlags = polygons.quadFlags(i);
                    if (!(quadFlags & POLYFLAG_SUBDIVIDED)) continue;

                    unsigned newPointIdx = unsigned(offset + mPointCount);

                    openvdb::Vec4I& quad = polygons.quad(i);

                    mCentroids[offset] = (mPoints[quad[0]] + mPoints[quad[1]] +
                        mPoints[quad[2]] + mPoints[quad[3]]) * 0.25f;

                    ++offset;

                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[0];
                        triangle[1] = newPointIdx;
                        triangle[2] = quad[3];

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;
                    }

                    ++triangleIdx;

                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[0];
                        triangle[1] = quad[1];
                        triangle[2] = newPointIdx;

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;
                    }

                    ++triangleIdx;

                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[1];
                        triangle[1] = quad[2];
                        triangle[2] = newPointIdx;

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;
                    }


                    ++triangleIdx;

                    {
                        Vec3I& triangle = tmpPolygons.triangle(triangleIdx);

                        triangle[0] = quad[2];
                        triangle[1] = quad[3];
                        triangle[2] = newPointIdx;

                        tmpPolygons.triangleFlags(triangleIdx) = quadFlags;
                    }

                    ++triangleIdx;

                    quad[0] = util::INVALID_IDX; // mark for deletion
                }


                for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
                    tmpPolygons.triangle(triangleIdx) = polygons.triangle(i);
                    tmpPolygons.triangleFlags(triangleIdx) = polygons.triangleFlags(i);
                    ++triangleIdx;
                }

                size_t quadIdx = 0;
                for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
                    openvdb::Vec4I& quad = polygons.quad(i);

                    if (quad[0] != util::INVALID_IDX) { // ignore invalid quads
                        tmpPolygons.quad(quadIdx) = quad;
                        tmpPolygons.quadFlags(quadIdx) = polygons.quadFlags(i);
                        ++quadIdx;
                    }
                }

                polygons.copy(tmpPolygons);
            }
        }
    }

private:
    PolygonPoolList       * const mPolygonPoolList;
    Vec3s           const * const mPoints;
    Vec3s                 * const mCentroids;
    unsigned              * const mNumQuadsToDivide;
    unsigned              * const mCentroidOffsets;
    size_t                  const mPointCount;
}; // struct SubdivideQuads


inline void
subdivideNonplanarSeamLineQuads(
    PolygonPoolList& polygonPoolList,
    size_t polygonPoolListSize,
    PointList& pointList,
    size_t& pointListSize,
    std::vector<uint8_t>& pointFlags)
{
    const tbb::blocked_range<size_t> polygonPoolListRange(0, polygonPoolListSize);

    std::unique_ptr<unsigned[]> numQuadsToDivide(new unsigned[polygonPoolListSize]);

    tbb::parallel_for(polygonPoolListRange,
        FlagAndCountQuadsToSubdivide(polygonPoolList, pointFlags, pointList, numQuadsToDivide));

    std::unique_ptr<unsigned[]> centroidOffsets(new unsigned[polygonPoolListSize]);

    size_t centroidCount = 0;

    {
        unsigned sum = 0;
        for (size_t n = 0, N = polygonPoolListSize; n < N; ++n) {
            centroidOffsets[n] = sum;
            sum += numQuadsToDivide[n];
        }
        centroidCount = size_t(sum);
    }

    std::unique_ptr<Vec3s[]> centroidList(new Vec3s[centroidCount]);

    tbb::parallel_for(polygonPoolListRange,
        SubdivideQuads(polygonPoolList, pointList, pointListSize,
            centroidList, numQuadsToDivide, centroidOffsets));

    if (centroidCount > 0) {

        const size_t newPointListSize = centroidCount + pointListSize;

        std::unique_ptr<openvdb::Vec3s[]> newPointList(new openvdb::Vec3s[newPointListSize]);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, pointListSize),
            CopyArray<Vec3s>(newPointList.get(), pointList.get()));

        tbb::parallel_for(tbb::blocked_range<size_t>(0, newPointListSize - pointListSize),
            CopyArray<Vec3s>(newPointList.get(), centroidList.get(), pointListSize));

        pointListSize = newPointListSize;
        pointList.swap(newPointList);
        pointFlags.resize(pointListSize, 0);
    }
}


struct ReviseSeamLineFlags
{
    ReviseSeamLineFlags(PolygonPoolList& polygons,
        const std::vector<uint8_t>& pointFlags)
        : mPolygonPoolList(&polygons)
        , mPointFlags(pointFlags.data())
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            PolygonPool& polygons = (*mPolygonPoolList)[n];

            for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {

                char& flags = polygons.quadFlags(i);

                if (flags & POLYFLAG_FRACTURE_SEAM) {

                    openvdb::Vec4I& verts = polygons.quad(i);

                    const bool hasSeamLinePoint =
                        mPointFlags[verts[0]] || mPointFlags[verts[1]] ||
                        mPointFlags[verts[2]] || mPointFlags[verts[3]];

                    if (!hasSeamLinePoint) {
                        flags &= ~POLYFLAG_FRACTURE_SEAM;
                    }
                }
            } // end quad loop

            for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {

                char& flags = polygons.triangleFlags(i);

                if (flags & POLYFLAG_FRACTURE_SEAM) {

                    openvdb::Vec3I& verts = polygons.triangle(i);

                    const bool hasSeamLinePoint =
                        mPointFlags[verts[0]] || mPointFlags[verts[1]] || mPointFlags[verts[2]];

                    if (!hasSeamLinePoint) {
                        flags &= ~POLYFLAG_FRACTURE_SEAM;
                    }

                }
            } // end triangle loop

        } // end polygon pool loop
    }

private:
    PolygonPoolList       * const mPolygonPoolList;
    uint8_t         const * const mPointFlags;
}; // struct ReviseSeamLineFlags


inline void
reviseSeamLineFlags(PolygonPoolList& polygonPoolList, size_t polygonPoolListSize,
    std::vector<uint8_t>& pointFlags)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, polygonPoolListSize),
        ReviseSeamLineFlags(polygonPoolList, pointFlags));
}


////////////////////////////////////////


template<typename InputTreeType>
struct MaskDisorientedTrianglePoints
{
    MaskDisorientedTrianglePoints(const InputTreeType& inputTree, const PolygonPoolList& polygons,
        const PointList& pointList, std::unique_ptr<uint8_t[]>& pointMask,
        const math::Transform& transform, bool invertSurfaceOrientation)
        : mInputTree(&inputTree)
        , mPolygonPoolList(&polygons)
        , mPointList(&pointList)
        , mPointMask(pointMask.get())
        , mTransform(transform)
        , mInvertSurfaceOrientation(invertSurfaceOrientation)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) const
    {
        using ValueType = typename InputTreeType::LeafNodeType::ValueType;

        tree::ValueAccessor<const InputTreeType> inputAcc(*mInputTree);
        Vec3s centroid, normal;
        Coord ijk;

        const bool invertGradientDir = mInvertSurfaceOrientation || isBoolValue<ValueType>();

        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {

            const PolygonPool& polygons = (*mPolygonPoolList)[n];

            for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {

                const Vec3I& verts = polygons.triangle(i);

                const Vec3s& v0 = (*mPointList)[verts[0]];
                const Vec3s& v1 = (*mPointList)[verts[1]];
                const Vec3s& v2 = (*mPointList)[verts[2]];

                normal = (v2 - v0).cross((v1 - v0));
                normal.normalize();

                centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
                ijk = mTransform.worldToIndexCellCentered(centroid);

                Vec3s dir( math::ISGradient<math::CD_2ND>::result(inputAcc, ijk) );
                dir.normalize();

                if (invertGradientDir) {
                    dir = -dir;
                }

                // check if the angle is obtuse
                if (dir.dot(normal) < -0.5f) {
                    // Concurrent writes to same memory address can occur, but
                    // all threads are writing the same value and char is atomic.
                    // (It is extremely rare that disoriented triangles share points,
                    // false sharing related performance impacts are not a concern.)
                    mPointMask[verts[0]] = 1;
                    mPointMask[verts[1]] = 1;
                    mPointMask[verts[2]] = 1;
                }

            } // end triangle loop

        } // end polygon pool loop
    }

private:
    InputTreeType   const * const mInputTree;
    PolygonPoolList const * const mPolygonPoolList;
    PointList       const * const mPointList;
    uint8_t               * const mPointMask;
    math::Transform         const mTransform;
    bool                    const mInvertSurfaceOrientation;
}; // struct MaskDisorientedTrianglePoints


template<typename InputTree>
inline void
relaxDisorientedTriangles(
    bool invertSurfaceOrientation,
    const InputTree& inputTree,
    const math::Transform& transform,
    PolygonPoolList& polygonPoolList,
    size_t polygonPoolListSize,
    PointList& pointList,
    const size_t pointListSize)
{
    const tbb::blocked_range<size_t> polygonPoolListRange(0, polygonPoolListSize);

    std::unique_ptr<uint8_t[]> pointMask(new uint8_t[pointListSize]);
    fillArray(pointMask.get(), uint8_t(0), pointListSize);

    tbb::parallel_for(polygonPoolListRange,
        MaskDisorientedTrianglePoints<InputTree>(
            inputTree, polygonPoolList, pointList, pointMask, transform, invertSurfaceOrientation));

    std::unique_ptr<uint8_t[]> pointUpdates(new uint8_t[pointListSize]);
    fillArray(pointUpdates.get(), uint8_t(0), pointListSize);

    std::unique_ptr<Vec3s[]> newPoints(new Vec3s[pointListSize]);
    fillArray(newPoints.get(), Vec3s(0.0f, 0.0f, 0.0f), pointListSize);

    for (size_t n = 0, N = polygonPoolListSize; n < N; ++n) {

        PolygonPool& polygons = polygonPoolList[n];

        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {
            openvdb::Vec4I& verts = polygons.quad(i);

            for (int v = 0; v < 4; ++v) {

                const unsigned pointIdx = verts[v];

                if (pointMask[pointIdx] == 1) {

                    newPoints[pointIdx] +=
                        pointList[verts[0]] + pointList[verts[1]] +
                        pointList[verts[2]] + pointList[verts[3]];

                    pointUpdates[pointIdx] = uint8_t(pointUpdates[pointIdx] + 4);
                }
            }
        }

        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {
            openvdb::Vec3I& verts = polygons.triangle(i);

            for (int v = 0; v < 3; ++v) {

                const unsigned pointIdx = verts[v];

                if (pointMask[pointIdx] == 1) {
                    newPoints[pointIdx] +=
                        pointList[verts[0]] + pointList[verts[1]] + pointList[verts[2]];

                    pointUpdates[pointIdx] = uint8_t(pointUpdates[pointIdx] + 3);
                }
            }
        }
    }

    for (size_t n = 0, N = pointListSize; n < N; ++n) {
        if (pointUpdates[n] > 0) {
            const double weight = 1.0 / double(pointUpdates[n]);
            pointList[n] = newPoints[n] * float(weight);
        }
    }
}


template<typename GridType>
void
doVolumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    std::vector<Vec4I>& quads,
    double isovalue,
    double adaptivity,
    bool relaxDisorientedTriangles)
{
    static_assert(std::is_scalar<typename GridType::ValueType>::value,
        "volume to mesh conversion is supported only for scalar grids");

    VolumeToMesh mesher(isovalue, adaptivity, relaxDisorientedTriangles);
    mesher(grid);

    // Preallocate the point list
    points.clear();
    points.resize(mesher.pointListSize());

    { // Copy points
        volume_to_mesh_internal::PointListCopy ptnCpy(mesher.pointList(), points);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size()), ptnCpy);
        mesher.pointList().reset(nullptr);
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


} // volume_to_mesh_internal namespace

/// @endcond

////////////////////////////////////////


inline
PolygonPool::PolygonPool()
    : mNumQuads(0)
    , mNumTriangles(0)
    , mQuads(nullptr)
    , mTriangles(nullptr)
    , mQuadFlags(nullptr)
    , mTriangleFlags(nullptr)
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
    mQuads.reset(nullptr);
    mQuadFlags.reset(nullptr);
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
    mTriangles.reset(nullptr);
    mTriangleFlags.reset(nullptr);
}


inline bool
PolygonPool::trimQuads(const size_t n, bool reallocate)
{
    if (!(n < mNumQuads)) return false;

    if (reallocate) {

        if (n == 0) {
            mQuads.reset(nullptr);
        } else {

            std::unique_ptr<openvdb::Vec4I[]> quads(new openvdb::Vec4I[n]);
            std::unique_ptr<char[]> flags(new char[n]);

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
            mTriangles.reset(nullptr);
        } else {

            std::unique_ptr<openvdb::Vec3I[]> triangles(new openvdb::Vec3I[n]);
            std::unique_ptr<char[]> flags(new char[n]);

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


inline
VolumeToMesh::VolumeToMesh(double isovalue, double adaptivity, bool relaxDisorientedTriangles)
    : mPoints(nullptr)
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
    , mRelaxDisorientedTriangles(relaxDisorientedTriangles)
    , mQuantizedSeamPoints(nullptr)
    , mPointFlags(0)
{
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
    mQuantizedSeamPoints.reset(nullptr);
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


template<typename InputGridType>
inline void
VolumeToMesh::operator()(const InputGridType& inputGrid)
{
    // input data types

    using InputTreeType = typename InputGridType::TreeType;
    using InputLeafNodeType = typename InputTreeType::LeafNodeType;
    using InputValueType = typename InputLeafNodeType::ValueType;

    // auxiliary data types

    using FloatTreeType = typename InputTreeType::template ValueConverter<float>::Type;
    using FloatGridType = Grid<FloatTreeType>;
    using BoolTreeType = typename InputTreeType::template ValueConverter<bool>::Type;
    using Int16TreeType = typename InputTreeType::template ValueConverter<Int16>::Type;
    using Int16LeafNodeType = typename Int16TreeType::LeafNodeType;
    using Index32TreeType = typename InputTreeType::template ValueConverter<Index32>::Type;
    using Index32LeafNodeType = typename Index32TreeType::LeafNodeType;

    // clear old data
    mPointListSize = 0;
    mPoints.reset();
    mPolygonPoolListSize = 0;
    mPolygons.reset();
    mPointFlags.clear();

    // settings

    const math::Transform& transform = inputGrid.transform();
    const InputValueType isovalue = InputValueType(mIsovalue);
    const float adaptivityThreshold = float(mPrimAdaptivity);
    const bool adaptive = mPrimAdaptivity > 1e-7 || mSecAdaptivity > 1e-7;

    // The default surface orientation is setup for level set and bool/mask grids.
    // Boolean grids are handled correctly by their value type.  Signed distance fields,
    // unsigned distance fields and fog volumes have the same value type but use different
    // inside value classifications.
    const bool invertSurfaceOrientation = (!volume_to_mesh_internal::isBoolValue<InputValueType>()
        && (inputGrid.getGridClass() != openvdb::GRID_LEVEL_SET));

    // references, masks and auxiliary data

    const InputTreeType& inputTree = inputGrid.tree();

    BoolTreeType intersectionTree(false), adaptivityMask(false);

    if (mAdaptivityMaskTree && mAdaptivityMaskTree->type() == BoolTreeType::treeType()) {
        const BoolTreeType *refAdaptivityMask=
            static_cast<const BoolTreeType*>(mAdaptivityMaskTree.get());
        adaptivityMask.topologyUnion(*refAdaptivityMask);
    }

    Int16TreeType signFlagsTree(0);
    Index32TreeType pointIndexTree(std::numeric_limits<Index32>::max());


    // collect auxiliary data

    volume_to_mesh_internal::identifySurfaceIntersectingVoxels(
        intersectionTree, inputTree, isovalue);

    volume_to_mesh_internal::applySurfaceMask(intersectionTree, adaptivityMask,
        inputGrid, mSurfaceMaskGrid, mInvertSurfaceMask, isovalue);

    if (intersectionTree.empty()) return;

    volume_to_mesh_internal::computeAuxiliaryData(
         signFlagsTree, pointIndexTree, intersectionTree, inputTree, isovalue);

    intersectionTree.clear();

    std::vector<Index32LeafNodeType*> pointIndexLeafNodes;
    pointIndexTree.getNodes(pointIndexLeafNodes);

    std::vector<Int16LeafNodeType*> signFlagsLeafNodes;
    signFlagsTree.getNodes(signFlagsLeafNodes);

    const tbb::blocked_range<size_t> auxiliaryLeafNodeRange(0, signFlagsLeafNodes.size());


    // optionally collect auxiliary data from a reference volume.

    Int16TreeType* refSignFlagsTree = nullptr;
    Index32TreeType* refPointIndexTree = nullptr;
    InputTreeType const* refInputTree = nullptr;

    if (mRefGrid && mRefGrid->type() == InputGridType::gridType()) {

        const InputGridType* refGrid = static_cast<const InputGridType*>(mRefGrid.get());
        refInputTree = &refGrid->tree();

        if (!mRefSignTree && !mRefIdxTree) {

            // first time, collect and cache auxiliary data.

            typename Int16TreeType::Ptr refSignFlagsTreePt(new Int16TreeType(0));
            typename Index32TreeType::Ptr refPointIndexTreePt(
                new Index32TreeType(std::numeric_limits<Index32>::max()));

            BoolTreeType refIntersectionTree(false);

            volume_to_mesh_internal::identifySurfaceIntersectingVoxels(
                refIntersectionTree, *refInputTree, isovalue);

            volume_to_mesh_internal::computeAuxiliaryData(*refSignFlagsTreePt,
                *refPointIndexTreePt, refIntersectionTree, *refInputTree, isovalue);

            mRefSignTree = refSignFlagsTreePt;
            mRefIdxTree = refPointIndexTreePt;
        }

        if (mRefSignTree && mRefIdxTree) {

            // get cached auxiliary data

            refSignFlagsTree = static_cast<Int16TreeType*>(mRefSignTree.get());
            refPointIndexTree = static_cast<Index32TreeType*>(mRefIdxTree.get());
        }


        if (refSignFlagsTree && refPointIndexTree) {

            // generate seam line sample points

            volume_to_mesh_internal::markSeamLineData(signFlagsTree, *refSignFlagsTree);

            if (mSeamPointListSize == 0) {

                // count unique points on reference surface

                std::vector<Int16LeafNodeType*> refSignFlagsLeafNodes;
                refSignFlagsTree->getNodes(refSignFlagsLeafNodes);

                std::unique_ptr<Index32[]> leafNodeOffsets(
                    new Index32[refSignFlagsLeafNodes.size()]);

                tbb::parallel_for(tbb::blocked_range<size_t>(0, refSignFlagsLeafNodes.size()),
                    volume_to_mesh_internal::LeafNodePointCount<Int16LeafNodeType::LOG2DIM>(
                        refSignFlagsLeafNodes, leafNodeOffsets));

                {
                    Index32 count = 0;
                    for (size_t n = 0, N = refSignFlagsLeafNodes.size(); n < N; ++n) {
                        const Index32 tmp = leafNodeOffsets[n];
                        leafNodeOffsets[n] = count;
                        count += tmp;
                    }
                    mSeamPointListSize = size_t(count);
                }

                if (mSeamPointListSize != 0) {

                    mQuantizedSeamPoints.reset(new uint32_t[mSeamPointListSize]);

                    std::memset(mQuantizedSeamPoints.get(), 0, sizeof(uint32_t) * mSeamPointListSize);

                    std::vector<Index32LeafNodeType*> refPointIndexLeafNodes;
                    refPointIndexTree->getNodes(refPointIndexLeafNodes);

                    tbb::parallel_for(tbb::blocked_range<size_t>(0, refPointIndexLeafNodes.size()),
                        volume_to_mesh_internal::MapPoints<Index32LeafNodeType>(
                            refPointIndexLeafNodes, refSignFlagsLeafNodes, leafNodeOffsets));
                }
            }

            if (mSeamPointListSize != 0) {

                tbb::parallel_for(auxiliaryLeafNodeRange,
                    volume_to_mesh_internal::SeamLineWeights<InputTreeType>(
                        signFlagsLeafNodes, inputTree, *refPointIndexTree, *refSignFlagsTree,
                            mQuantizedSeamPoints.get(), isovalue));
            }
        }
    }

    const bool referenceMeshing = refSignFlagsTree && refPointIndexTree && refInputTree;


    // adapt and count unique points

    std::unique_ptr<Index32[]> leafNodeOffsets(new Index32[signFlagsLeafNodes.size()]);

    if (adaptive) {
        volume_to_mesh_internal::MergeVoxelRegions<InputGridType> mergeOp(
            inputGrid, pointIndexTree, pointIndexLeafNodes, signFlagsLeafNodes,
            isovalue, adaptivityThreshold, invertSurfaceOrientation);

        if (mAdaptivityGrid && mAdaptivityGrid->type() == FloatGridType::gridType()) {
            const FloatGridType* adaptivityGrid =
                static_cast<const FloatGridType*>(mAdaptivityGrid.get());
            mergeOp.setSpatialAdaptivity(*adaptivityGrid);
        }

        if (!adaptivityMask.empty()) {
            mergeOp.setAdaptivityMask(adaptivityMask);
        }

        if (referenceMeshing) {
            mergeOp.setRefSignFlagsData(*refSignFlagsTree, float(mSecAdaptivity));
        }

        tbb::parallel_for(auxiliaryLeafNodeRange, mergeOp);

        volume_to_mesh_internal::AdaptiveLeafNodePointCount<Index32LeafNodeType>
            op(pointIndexLeafNodes, signFlagsLeafNodes, leafNodeOffsets);

        tbb::parallel_for(auxiliaryLeafNodeRange, op);

    } else {

        volume_to_mesh_internal::LeafNodePointCount<Int16LeafNodeType::LOG2DIM>
            op(signFlagsLeafNodes, leafNodeOffsets);

        tbb::parallel_for(auxiliaryLeafNodeRange, op);
    }


    {
        Index32 pointCount = 0;
        for (size_t n = 0, N = signFlagsLeafNodes.size(); n < N; ++n) {
            const Index32 tmp = leafNodeOffsets[n];
            leafNodeOffsets[n] = pointCount;
            pointCount += tmp;
        }

        mPointListSize = size_t(pointCount);
        mPoints.reset(new openvdb::Vec3s[mPointListSize]);
        mPointFlags.clear();
    }


    // compute points

    {
        volume_to_mesh_internal::ComputePoints<InputTreeType>
            op(mPoints.get(), inputTree, pointIndexLeafNodes,
                signFlagsLeafNodes, leafNodeOffsets, transform, mIsovalue);

        if (referenceMeshing) {
            mPointFlags.resize(mPointListSize);
            op.setRefData(*refInputTree, *refPointIndexTree, *refSignFlagsTree,
                mQuantizedSeamPoints.get(), mPointFlags.data());
        }

        tbb::parallel_for(auxiliaryLeafNodeRange, op);
    }


    // compute polygons

    mPolygonPoolListSize = signFlagsLeafNodes.size();
    mPolygons.reset(new PolygonPool[mPolygonPoolListSize]);

    if (adaptive) {

        using PrimBuilder = volume_to_mesh_internal::AdaptivePrimBuilder;

        volume_to_mesh_internal::ComputePolygons<Int16TreeType, PrimBuilder>
            op(signFlagsLeafNodes, signFlagsTree, pointIndexTree,
                mPolygons, invertSurfaceOrientation);

        if (referenceMeshing) {
            op.setRefSignTree(refSignFlagsTree);
        }

        tbb::parallel_for(auxiliaryLeafNodeRange, op);

    } else {

        using PrimBuilder = volume_to_mesh_internal::UniformPrimBuilder;

        volume_to_mesh_internal::ComputePolygons<Int16TreeType, PrimBuilder>
            op(signFlagsLeafNodes, signFlagsTree, pointIndexTree,
                mPolygons, invertSurfaceOrientation);

        if (referenceMeshing) {
            op.setRefSignTree(refSignFlagsTree);
        }

        tbb::parallel_for(auxiliaryLeafNodeRange, op);
    }


    signFlagsTree.clear();
    pointIndexTree.clear();


    if (adaptive && mRelaxDisorientedTriangles) {
        volume_to_mesh_internal::relaxDisorientedTriangles(invertSurfaceOrientation,
            inputTree, transform, mPolygons, mPolygonPoolListSize, mPoints, mPointListSize);
    }


    if (referenceMeshing) {
        volume_to_mesh_internal::subdivideNonplanarSeamLineQuads(
            mPolygons, mPolygonPoolListSize, mPoints, mPointListSize, mPointFlags);

        volume_to_mesh_internal::reviseSeamLineFlags(mPolygons, mPolygonPoolListSize, mPointFlags);
    }

}


////////////////////////////////////////


template<typename GridType>
void volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec3I>& triangles,
    std::vector<Vec4I>& quads,
    double isovalue,
    double adaptivity,
    bool relaxDisorientedTriangles)
{
    volume_to_mesh_internal::doVolumeToMesh(grid, points, triangles, quads,
        isovalue, adaptivity, relaxDisorientedTriangles);
}

template<typename GridType>
void volumeToMesh(
    const GridType& grid,
    std::vector<Vec3s>& points,
    std::vector<Vec4I>& quads,
    double isovalue)
{
    std::vector<Vec3I> triangles;
    volumeToMesh(grid, points, triangles, quads, isovalue, 0.0, true);
}


////////////////////////////////////////


// Explicit Template Instantiation

#ifdef OPENVDB_USE_EXPLICIT_INSTANTIATION

#ifdef OPENVDB_INSTANTIATE_VOLUMETOMESH
#include <openvdb/util/ExplicitInstantiation.h>
#endif

#define _FUNCTION(TreeT) \
    void volumeToMesh(const Grid<TreeT>&, std::vector<Vec3s>&, std::vector<Vec4I>&, double)
OPENVDB_NUMERIC_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#define _FUNCTION(TreeT) \
    void volumeToMesh(const Grid<TreeT>&, std::vector<Vec3s>&, std::vector<Vec3I>&, std::vector<Vec4I>&, double, double, bool)
OPENVDB_NUMERIC_TREE_INSTANTIATE(_FUNCTION)
#undef _FUNCTION

#endif // OPENVDB_USE_EXPLICIT_INSTANTIATION


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_VOLUME_TO_MESH_HAS_BEEN_INCLUDED
