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
enum {
    POLYFLAG_EXTERIOR = 0x1, POLYFLAG_FRACTURE_SEAM = 0x2
};


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
    /// @param  smoothSeams     toggle to smooth seam line edges during mesh extraction,
    ///                         removes staircase artifacts.
    void setRefGrid(const GridBase::ConstPtr& grid, double secAdaptivity = 0,
        bool smoothSeams = true);


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

    size_t mPointListSize, mPolygonPoolListSize;
    double mIsovalue, mPrimAdaptivity, mSecAdaptivity;

    GridBase::ConstPtr mRefGrid, mSurfaceMaskGrid, mAdaptivityGrid;
    TreeBase::ConstPtr mAdaptivityMaskTree;

    TreeBase::Ptr mRefEdgeTree, mRefTopologyMaskTree, mSeamPointTree;

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


// Bit-flags
enum { INSIDE = 0x1, XEDGE = 0x2, YEDGE = 0x4, ZEDGE = 0x8 };

const bool sAmbiguous[256] =
   {0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,
    0,0,1,0,1,1,1,0,
    1,0,1,0,1,0,1,0,
    0,1,0,0,1,1,0,0,
    1,1,1,0,1,1,0,0,
    0,0,0,0,1,1,0,0,
    1,0,1,0,1,1,1,0,
    0,1,1,1,0,1,0,0,
    1,1,1,1,0,0,0,0,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    0,1,0,0,0,1,0,0,
    1,1,1,1,0,1,0,0,
    0,0,0,0,0,1,0,0,
    1,1,1,1,1,1,1,0,
    0,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,
    0,0,1,0,1,1,1,1,
    0,0,1,0,0,0,1,0,
    1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    0,0,0,0,1,1,1,1,
    0,0,1,0,1,1,1,0,
    0,1,1,1,0,1,0,1,
    0,0,1,1,0,0,0,0,
    0,0,1,1,0,1,1,1,
    0,0,1,1,0,0,1,0,
    0,1,0,1,0,1,0,1,
    0,1,1,1,0,1,0,0,
    0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0};


template<class AccessorT>
inline bool isAmbiguous(const AccessorT& accessor, const Coord& ijk,
    typename AccessorT::ValueType isovalue, int dim)
{
    unsigned signs = 0;
    Coord coord = ijk; // i, j, k
    if (accessor.getValue(coord) < isovalue) signs |= 1u;
    coord[0] += dim; // i+dim, j, k
    if (accessor.getValue(coord) < isovalue) signs |= 2u;
    coord[2] += dim; // i+dim, j, k+dim
    if (accessor.getValue(coord) < isovalue) signs |= 4u;
    coord[0] = ijk[0]; // i, j, k+dim
    if (accessor.getValue(coord) < isovalue) signs |= 8u;
    coord[1] += dim; coord[2] = ijk[2]; // i, j+dim, k
    if (accessor.getValue(coord) < isovalue) signs |= 16u;
    coord[0] += dim; // i+dim, j+dim, k
    if (accessor.getValue(coord) < isovalue) signs |= 32u;
    coord[2] += dim; // i+dim, j+dim, k+dim
    if (accessor.getValue(coord) < isovalue) signs |= 64u;
    coord[0] = ijk[0]; // i, j+dim, k+dim
    if (accessor.getValue(coord) < isovalue) signs |= 128u;
    return sAmbiguous[signs];
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
    if (sAmbiguous[signs]) return true;

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


template <class TreeT>
class LeafCPtrList
{
public:
    typedef std::vector<const typename TreeT::LeafNodeType *> ListT;

    LeafCPtrList(const TreeT& tree)
    {
        mLeafNodes.reserve(tree.leafCount());
        typename TreeT::LeafCIter iter = tree.cbeginLeaf();
        for ( ; iter; ++iter) mLeafNodes.push_back(iter.getLeaf());
    }

    size_t size() const { return mLeafNodes.size(); }

    const typename TreeT::LeafNodeType* operator[](size_t n) const
        { return mLeafNodes[n]; }

    tbb::blocked_range<size_t> getRange() const
        { return tbb::blocked_range<size_t>(0, mLeafNodes.size()); }

    const ListT& getList() const { return mLeafNodes; }

private:
    ListT mLeafNodes;
};


template <class TreeT>
class LeafPtrList
{
public:
    typedef std::vector<typename TreeT::LeafNodeType *> ListT;

    LeafPtrList(TreeT& tree)
    {
        mLeafNodes.reserve(tree.leafCount());
        typename TreeT::LeafIter iter = tree.beginLeaf();
        for ( ; iter; ++iter) mLeafNodes.push_back(iter.getLeaf());
    }

    size_t size() const { return mLeafNodes.size(); }

    typename TreeT::LeafNodeType* operator[](size_t n) const
        { return mLeafNodes[n]; }

    tbb::blocked_range<size_t> getRange() const
        { return tbb::blocked_range<size_t>(0, mLeafNodes.size()); }

    const ListT& getList() const { return mLeafNodes; }

private:
    ListT mLeafNodes;
};


////////////////////////////////////////

template<typename DistTreeT>
struct ReferenceData
{
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename DistTreeT::template ValueConverter<char>::Type CharTreeT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename DistTreeT::template ValueConverter<Vec3s>::Type Vec3sTreeT;

    ReferenceData()
        : mDistTree(NULL)
        , mEdgeTree(NULL)
        , mTopologyMaskTree(NULL)
        , mSeamPointTree(NULL)
        , mSeamMaskTree(typename BoolTreeT::Ptr())
        , mSmoothingMaskTree(typename BoolTreeT::Ptr())
        , mInternalAdaptivity(DistValueT(0.0))
    {
    }

    bool isValid() const
    {
        return mDistTree && mEdgeTree && mTopologyMaskTree && mSeamMaskTree;
    }

    const DistTreeT* mDistTree;
    const CharTreeT* mEdgeTree;
    BoolTreeT* mTopologyMaskTree;
    Vec3sTreeT* mSeamPointTree;
    typename BoolTreeT::Ptr mSeamMaskTree, mSmoothingMaskTree;
    DistValueT mInternalAdaptivity;
};


////////////////////////////////////////


template <class DistTreeT>
class Count
{
public:
    Count(const LeafPtrList<DistTreeT>&, std::vector<size_t>&);
    inline Count(const Count<DistTreeT>&);

    void runParallel();
    void runSerial();

    inline void operator()(const tbb::blocked_range<size_t>&) const;
private:
    const LeafPtrList<DistTreeT>& mLeafNodes;
    std::vector<size_t>& mLeafRegionCount;
};


template <class DistTreeT>
Count<DistTreeT>::Count(
    const LeafPtrList<DistTreeT>& leafs,
    std::vector<size_t>& leafRegionCount)
    : mLeafNodes(leafs)
    , mLeafRegionCount(leafRegionCount)
{
}


template <class DistTreeT>
inline
Count<DistTreeT>::Count(const Count<DistTreeT>& rhs)
    : mLeafNodes(rhs.mLeafNodes)
    , mLeafRegionCount(rhs.mLeafRegionCount)
{
}


template <class DistTreeT>
void
Count<DistTreeT>::runParallel()
{
    tbb::parallel_for(mLeafNodes.getRange(), *this);
}


template <class DistTreeT>
void
Count<DistTreeT>::runSerial()
{
    (*this)(mLeafNodes.getRange());
}


template <class DistTreeT>
inline void
Count<DistTreeT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    for (size_t n = range.begin(); n != range.end(); ++n) {
        mLeafRegionCount[n] = size_t(mLeafNodes[n]->onVoxelCount());
    }
}


////////////////////////////////////////


template <class DistTreeT>
class Merge
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename DistTreeT::template ValueConverter<int>::Type IntTreeT;
    typedef typename DistTreeT::template ValueConverter<float>::Type FloatTreeT;
    typedef Grid<FloatTreeT> FloatGridT;

    Merge(
        const DistTreeT& distTree,
        LeafPtrList<IntTreeT>& auxLeafs,
        const DistValueT iso,
        const DistValueT adaptivity);

    inline Merge(const Merge<DistTreeT>&);

    void setRefData(const ReferenceData<DistTreeT>&);

    void setSpatialAdaptivity(
        const math::Transform& distGridXForm, const FloatGridT& adaptivityField);

    void run(bool threaded = true);

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    const DistTreeT& mDistTree;
    LeafPtrList<IntTreeT>& mAuxLeafs;
    const DistValueT mIsovalue, mAdaptivity;
    const ReferenceData<DistTreeT>* mRefData;

    const math::Transform * mTransform;
    const FloatGridT * mAdaptivityGrid;
};


template <class DistTreeT>
Merge<DistTreeT>::Merge(
    const DistTreeT& distTree,
    LeafPtrList<IntTreeT>& auxLeafs,
    const DistValueT iso,
    const DistValueT adaptivity)
    : mDistTree(distTree)
    , mAuxLeafs(auxLeafs)
    , mIsovalue(iso)
    , mAdaptivity(adaptivity)
    , mRefData(NULL)
    , mTransform(NULL)
    , mAdaptivityGrid(NULL)
{
}


template <class DistTreeT>
inline
Merge<DistTreeT>::Merge(const Merge<DistTreeT>& rhs)
    : mDistTree(rhs.mDistTree)
    , mAuxLeafs(rhs.mAuxLeafs)
    , mIsovalue(rhs.mIsovalue)
    , mAdaptivity(rhs.mAdaptivity)
    , mRefData(rhs.mRefData)
    , mTransform(rhs.mTransform)
    , mAdaptivityGrid(rhs.mAdaptivityGrid)
{
}


template <class DistTreeT>
void
Merge<DistTreeT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(mAuxLeafs.getRange(), *this);
    } else {
        (*this)(mAuxLeafs.getRange());
    }
}


template <class DistTreeT>
void
Merge<DistTreeT>::setRefData(const ReferenceData<DistTreeT>& refData)
{
    mRefData = &refData;
}


template <class DistTreeT>
void
Merge<DistTreeT>::setSpatialAdaptivity(
    const math::Transform& distGridXForm, const FloatGridT& adaptivityField)
{
    mTransform = &distGridXForm;
    mAdaptivityGrid = &adaptivityField;
}


template <class DistTreeT>
void
Merge<DistTreeT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typedef math::Vec3<DistValueT> Vec3T;
    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typedef typename BoolLeafT::template ValueConverter<Vec3T>::Type Vec3LeafT;
    typedef typename BoolLeafT::template ValueConverter<DistValueT>::Type DistLeafT;

    typedef typename IntLeafT::ValueOnIter IntIterT;
    typedef typename BoolLeafT::ValueOnCIter BoolCIterT;

    typedef typename tree::ValueAccessor<BoolTreeT> BoolTreeAccessorT;
    typedef typename tree::ValueAccessor<const BoolTreeT> BoolTreeCAccessorT;
    typedef typename tree::ValueAccessor<const FloatTreeT> FloatTreeCAccessorT;

    boost::scoped_ptr<FloatTreeCAccessorT> adaptivityAcc;
    if (mAdaptivityGrid) {
        adaptivityAcc.reset(new FloatTreeCAccessorT(mAdaptivityGrid->tree()));
    }

    boost::scoped_ptr<BoolTreeAccessorT> seamMaskAcc;
    boost::scoped_ptr<BoolTreeCAccessorT> topologyMaskAcc;
    if (mRefData && mRefData->isValid()) {
        seamMaskAcc.reset(new BoolTreeAccessorT(*mRefData->mSeamMaskTree.get()));
        topologyMaskAcc.reset(new BoolTreeCAccessorT(*mRefData->mTopologyMaskTree));
    }
    const bool hasRefData = seamMaskAcc && topologyMaskAcc;

    const int LeafDim = BoolLeafT::DIM;
    tree::ValueAccessor<const DistTreeT> distAcc(mDistTree);

    // Allocate reusable leaf buffers
    BoolLeafT mask;
    Vec3LeafT gradientBuffer;
    Coord ijk, coord, end;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        DistValueT adaptivity = mAdaptivity;
        IntLeafT& auxLeaf = *mAuxLeafs[n];

        const Coord& origin = auxLeaf.origin();
        end[0] = origin[0] + LeafDim;
        end[1] = origin[1] + LeafDim;
        end[2] = origin[2] + LeafDim;

        mask.setValuesOff();

        // Mask off seam line adjacent voxels
        if (hasRefData) {
            const BoolLeafT* seamMask = seamMaskAcc->probeConstLeaf(origin);
            if (seamMask != NULL) {
                for (BoolCIterT it = seamMask->cbeginValueOn(); it; ++it) {
                    ijk = it.getCoord();
                    coord[0] = ijk[0] - (ijk[0] % 2);
                    coord[1] = ijk[1] - (ijk[1] % 2);
                    coord[2] = ijk[2] - (ijk[2] % 2);
                    mask.setActiveState(coord, true);
                }
            }
            if (topologyMaskAcc->probeConstLeaf(origin) == NULL) {
                adaptivity = mRefData->mInternalAdaptivity;
            }
        }


        DistLeafT adaptivityLeaf(origin, adaptivity);

        if (mAdaptivityGrid) {
            for (Index offset = 0; offset < DistLeafT::NUM_VALUES; ++offset) {
                ijk = adaptivityLeaf.offset2globalCoord(offset);
                Vec3d xyz = mAdaptivityGrid->transform().worldToIndex(
                    mTransform->indexToWorld(ijk));
                DistValueT tmpA = DistValueT(adaptivityAcc->getValue(util::nearestCoord(xyz)));
                adaptivityLeaf.setValueOnly(offset, tmpA * adaptivity);
            }
        }

        // Mask off ambiguous voxels
        for (IntIterT it = auxLeaf.beginValueOn(); it; ++it) {
            ijk = it.getCoord();
            coord[0] = ijk[0] - (ijk[0] % 2);
            coord[1] = ijk[1] - (ijk[1] % 2);
            coord[2] = ijk[2] - (ijk[2] % 2);
            if(mask.isValueOn(coord)) continue;
            mask.setActiveState(coord, isAmbiguous(distAcc, ijk, mIsovalue, 1));
        }

        int dim = 2;
        // Mask off topologically ambiguous 2x2x2 voxel sub-blocks
        for (ijk[0] = origin[0]; ijk[0] < end[0]; ijk[0] += dim) {
            for (ijk[1] = origin[1]; ijk[1] < end[1]; ijk[1] += dim) {
                for (ijk[2] = origin[2]; ijk[2] < end[2]; ijk[2] += dim) {
                    if (isNonManifold(distAcc, ijk, mIsovalue, dim)) {
                        mask.setActiveState(ijk, true);
                    }
                }
            }
        }

        // Compute the gradient for the remaining voxels
        gradientBuffer.setValuesOff();

        for (IntIterT it = auxLeaf.beginValueOn(); it; ++it) {

            ijk = it.getCoord();
            coord[0] = ijk[0] - (ijk[0] % dim);
            coord[1] = ijk[1] - (ijk[1] % dim);
            coord[2] = ijk[2] - (ijk[2] % dim);
            if(mask.isValueOn(coord)) continue;

            Vec3T norm(math::ISGradient<math::CD_2ND>::result(distAcc, ijk));
            // Normalize (Vec3's normalize uses isApproxEqual, which uses abs and does more work)
            DistValueT length = norm.length();
            if (length > DistValueT(1.0e-7)) {
                norm *= DistValueT(1.0) / length;
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
                    mergeVoxels(auxLeaf, ijk, dim, regionId++);
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
                        if (mask.isValueOn(ijk) || isNonManifold(distAcc, ijk, mIsovalue, dim) ||
                            !isMergable(gradientBuffer, ijk, dim, adaptivity)) {
                            mask.setActiveState(coord, true);
                            continue;
                        }
                        mergeVoxels(auxLeaf, ijk, dim, regionId++);
                    }
                }
            }
        }

        adaptivity = adaptivityLeaf.getValue(origin);
        if (!(mask.isValueOn(origin) || isNonManifold(distAcc, origin, mIsovalue, LeafDim))
            && isMergable(gradientBuffer, origin, LeafDim, adaptivity)) {
            mergeVoxels(auxLeaf, origin, LeafDim, regionId++);
        }
    }
}
////////////////////////////////////////


template <typename IntTreeT>
class RegionCount
{
public:
    typedef typename IntTreeT::template ValueConverter<bool>::Type BoolTreeT;

    RegionCount(
        LeafPtrList<IntTreeT>& auxLeafs,
        std::vector<size_t>& leafRegionCount,
        const BoolTreeT * pointMask = NULL);

    inline RegionCount(const RegionCount&);

    void run(bool threaded = true);

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    LeafPtrList<IntTreeT>& mAuxLeafs;
    std::vector<size_t>& mLeafRegionCount;
    const BoolTreeT * mPointMask;
};


template <typename IntTreeT>
RegionCount<IntTreeT>::RegionCount(
    LeafPtrList<IntTreeT>& auxLeafs,
    std::vector<size_t>& leafRegionCount,
    const BoolTreeT * pointMask)
    : mAuxLeafs(auxLeafs)
    , mLeafRegionCount(leafRegionCount)
    , mPointMask(pointMask)
{
}


template <typename IntTreeT>
inline
RegionCount<IntTreeT>::RegionCount(const RegionCount& rhs)
    : mAuxLeafs(rhs.mAuxLeafs)
    , mLeafRegionCount(rhs.mLeafRegionCount)
    , mPointMask(rhs.mPointMask)
{
}


template <typename IntTreeT>
void
RegionCount<IntTreeT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(mAuxLeafs.getRange(), *this);
    } else {
        (*this)(mAuxLeafs.getRange());
    }
}

template <typename IntTreeT>
void
RegionCount<IntTreeT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typedef typename IntLeafT::ValueOnIter IntIterT;

    typedef typename tree::ValueAccessor<const BoolTreeT> BoolTreeCAccessorT;

    Coord ijk;
    int regionId;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        IntLeafT& auxLeaf = *mAuxLeafs[n];

        if (mPointMask) {
            const BoolLeafT * ptnMaskLeaf = mPointMask->probeConstLeaf(auxLeaf.origin());

            if (ptnMaskLeaf) { // remove unused regions

                for (IntIterT it = auxLeaf.beginValueOn(); it; ++it) {
                    if (it.getValue() == 0 && !ptnMaskLeaf->isValueOn(it.pos())) {
                        it.setValueOff();
                    }
                }

                IntLeafT tmpLeaf(auxLeaf);

                while (tmpLeaf.onVoxelCount() > 0) {

                    IntIterT it = tmpLeaf.beginValueOn();
                    regionId = it.getValue();
                    bool removeRegion = true;
                    for (; it; ++it) {
                        if (it.getValue() == regionId) {
                            if (ptnMaskLeaf->isValueOn(it.pos())) {
                                removeRegion = false;
                            }
                            it.setValueOff();
                        }
                    }

                    if (removeRegion) {
                        it = auxLeaf.beginValueOn();
                        for (; it; ++it) {
                            if (it.getValue() == regionId) {
                                it.setValueOff();
                            }
                        }
                    }
                }
            } else {
                mLeafRegionCount[n] = 0;
                auxLeaf.setValuesOff();
                continue;
            }
        }

        // Count unique regions
        size_t numVoxels = 0;
        IntLeafT tmpLeaf(auxLeaf);
        for (IntIterT it = tmpLeaf.beginValueOn(); it; ++it) {
            if(it.getValue() == 0) {
                it.setValueOff();
                ++numVoxels;
            }
        }

        while (tmpLeaf.onVoxelCount() > 0) {
            ++numVoxels;
            IntIterT it = tmpLeaf.beginValueOn();
            regionId = it.getValue();
            for (; it; ++it) {
                if (it.getValue() == regionId) it.setValueOff();
            }
        }

        mLeafRegionCount[n] = numVoxels;
    }
}


////////////////////////////////////////


template <class DistTreeT>
class PointGen
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef tree::ValueAccessor<const DistTreeT> DistTreeAccessorT;
    typedef typename DistTreeT::template ValueConverter<int>::Type IntTreeT;

    PointGen(
        const DistTreeT& distTree,
        const LeafPtrList<IntTreeT>& auxLeafs,
        std::vector<size_t>& leafIndices,
        const openvdb::math::Transform& xform,
        PointList& points,
        double iso = 0.0);

    PointGen(const PointGen<DistTreeT>&);

    void setRefData(const ReferenceData<DistTreeT>&);

    void runParallel();
    void runSerial();

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    const DistTreeT& mDistTree;
    const LeafPtrList<IntTreeT>& mAuxLeafs;
    const std::vector<size_t>& mLeafIndices;
    const openvdb::math::Transform& mTransform;
    const PointList& mPoints;
    const double mIsovalue;

    const ReferenceData<DistTreeT>* mRefData;

    double root(double v0, double v1) const { return (mIsovalue - v0) / (v1 - v0); }
    int calcAvgPoint(DistTreeAccessorT&, const Coord&, openvdb::Vec3d&) const;
};


template <class DistTreeT>
PointGen<DistTreeT>::PointGen(
    const DistTreeT& distTree,
    const LeafPtrList<IntTreeT>& auxLeafs,
    std::vector<size_t>& leafIndices,
    const openvdb::math::Transform& xform,
    PointList& points,
    double iso)
    : mDistTree(distTree)
    , mAuxLeafs(auxLeafs)
    , mLeafIndices(leafIndices)
    , mTransform(xform)
    , mPoints(points)
    , mIsovalue(iso)
    , mRefData(NULL)
{
}


template <class DistTreeT>
PointGen<DistTreeT>::PointGen(const PointGen<DistTreeT>& rhs)
    : mDistTree(rhs.mDistTree)
    , mAuxLeafs(rhs.mAuxLeafs)
    , mLeafIndices(rhs.mLeafIndices)
    , mTransform(rhs.mTransform)
    , mPoints(rhs.mPoints)
    , mIsovalue(rhs.mIsovalue)
    , mRefData(rhs.mRefData)
{
}


template <class DistTreeT>
void
PointGen<DistTreeT>::setRefData(
    const ReferenceData<DistTreeT>& refData)
{
    mRefData = &refData;
}

template <class DistTreeT>
void
PointGen<DistTreeT>::runParallel()
{
    tbb::parallel_for(mAuxLeafs.getRange(), *this);
}


template <class DistTreeT>
void
PointGen<DistTreeT>::runSerial()
{
    (*this)(mAuxLeafs.getRange());
}


template <class DistTreeT>
void
PointGen<DistTreeT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename DistTreeT::template ValueConverter<Vec3s>::Type Vec3sTreeT;

    typedef tree::ValueAccessor<BoolTreeT> BoolTreeAccessorT;
    typedef tree::ValueAccessor<Vec3sTreeT> Vec3sTreeAccessorT;

    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typedef typename Vec3sTreeT::LeafNodeType Vec3sLeafT;

    boost::scoped_ptr<DistTreeAccessorT> refDistAcc;
    boost::scoped_ptr<BoolTreeAccessorT> refMaskAcc, refSmoothMaskAcc;
    boost::scoped_ptr<Vec3sTreeAccessorT> refPtnAcc;

    if (mRefData && mRefData->isValid()) {
        refDistAcc.reset(new DistTreeAccessorT(*mRefData->mDistTree));
        refMaskAcc.reset(new BoolTreeAccessorT(*mRefData->mTopologyMaskTree));
        refSmoothMaskAcc.reset(new BoolTreeAccessorT(*mRefData->mSmoothingMaskTree));
        refPtnAcc.reset(new Vec3sTreeAccessorT(*mRefData->mSeamPointTree));
    }


    const bool hasRefData = refDistAcc && refMaskAcc;
    typename IntTreeT::LeafNodeType::ValueOnIter auxIter;
    DistTreeAccessorT distAcc(mDistTree);

    Coord ijk;
    openvdb::Vec3d avg, tmp;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        size_t idx = mLeafIndices[n];
        IntLeafT& auxLeaf = *mAuxLeafs[n];

        if (auxLeaf.isEmpty()) continue;

        BoolLeafT* maskLeaf = NULL;
        BoolLeafT* smoothMaskLeaf = NULL;
        Vec3sLeafT* ptnLeaf = NULL;
        if (hasRefData) {
            maskLeaf = refMaskAcc->probeLeaf(auxLeaf.origin());
            smoothMaskLeaf = refSmoothMaskAcc->probeLeaf(auxLeaf.origin());
            ptnLeaf = refPtnAcc->probeLeaf(auxLeaf.origin());
        }

        for (auxIter = auxLeaf.beginValueOn(); auxIter; ++auxIter) {

            if(auxIter.getValue() == 0) {

                auxIter.setValue(idx);
                auxIter.setValueOff();
                ijk = auxIter.getCoord();

                if (hasRefData && maskLeaf && maskLeaf->isValueOn(ijk)) {

                    if (ptnLeaf && ptnLeaf->isValueOn(ijk)) {
                        avg = ptnLeaf->getValue(ijk);
                    } else {
                        int e1 = calcAvgPoint(*refDistAcc.get(), ijk, avg);

                        if (e1 != (XEDGE|YEDGE|ZEDGE)) {
                            int e2 = calcAvgPoint(distAcc, ijk, tmp);
                            if((e2 & (~e1)) != 0) smoothMaskLeaf->setValueOn(ijk);
                        }
                    }
                } else {
                    calcAvgPoint(distAcc, ijk, avg);
                }

                openvdb::Vec3s& ptn = mPoints[idx];
                ptn[0] = float(avg[0]);
                ptn[1] = float(avg[1]);
                ptn[2] = float(avg[2]);

                ++idx;
            }
        }

        while(auxLeaf.onVoxelCount() > 0) {

            avg[0] = 0;
            avg[1] = 0;
            avg[2] = 0;

            auxIter = auxLeaf.beginValueOn();
            int regionId = auxIter.getValue(), points = 0;

            for (; auxIter; ++auxIter) {
                if(auxIter.getValue() == regionId) {

                    auxIter.setValue(idx);
                    auxIter.setValueOff();
                    ijk = auxIter.getCoord();

                    if (hasRefData && maskLeaf && maskLeaf->isValueOn(ijk)) {
                        calcAvgPoint(*refDistAcc.get(), ijk, tmp);
                    } else {
                        calcAvgPoint(distAcc, ijk, tmp);
                    }

                    avg += tmp;
                    ++points;
                }
            }


            if (points > 1) {
                double w = 1.0 / double(points);
                avg[0] *= w;
                avg[1] *= w;
                avg[2] *= w;
            }

            openvdb::Vec3s& ptn = mPoints[idx];
            ptn[0] = float(avg[0]);
            ptn[1] = float(avg[1]);
            ptn[2] = float(avg[2]);
            ++idx;
        }
    }
}

template <class DistTreeT>
int
PointGen<DistTreeT>::calcAvgPoint(DistTreeAccessorT& acc,
    const Coord& ijk, openvdb::Vec3d& avg) const
{
    double values[8];
    bool signMask[8];
    Coord coord;

    // Sample corner values
    coord = ijk;
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

    // init sign mask
    for (int n = 0; n < 8; ++n) signMask[n] = (values[n] < mIsovalue);

    int samples = 0, edgeFlags = 0;
    avg[0] = 0.0;
    avg[1] = 0.0;
    avg[2] = 0.0;

    if (signMask[0] != signMask[1]) { // Edged: 0 - 1
        avg[0] += root(values[0], values[1]);
        ++samples;
        edgeFlags |= XEDGE;
    }

    if (signMask[1] != signMask[2]) { // Edged: 1 - 2
        avg[0] += 1.0;
        avg[2] += root(values[1], values[2]);
        ++samples;
        edgeFlags |= ZEDGE;
    }

    if (signMask[3] != signMask[2]) { // Edged: 3 - 2
        avg[0] += root(values[3], values[2]);
        avg[2] += 1.0;
        ++samples;
        edgeFlags |= XEDGE;
    }

    if (signMask[0] != signMask[3]) { // Edged: 0 - 3
        avg[2] += root(values[0], values[3]);
        ++samples;
        edgeFlags |= ZEDGE;
    }

    if (signMask[4] != signMask[5]) { // Edged: 4 - 5
        avg[0] += root(values[4], values[5]);
        avg[1] += 1.0;
        ++samples;
        edgeFlags |= XEDGE;
    }

    if (signMask[5] != signMask[6]) { // Edged: 5 - 6
        avg[0] += 1.0;
        avg[1] += 1.0;
        avg[2] += root(values[5], values[6]);
        ++samples;
        edgeFlags |= ZEDGE;
    }

    if (signMask[7] != signMask[6]) { // Edged: 7 - 6
        avg[0] += root(values[7], values[6]);
        avg[1] += 1.0;
        avg[2] += 1.0;
        ++samples;
        edgeFlags |= XEDGE;
    }

    if (signMask[4] != signMask[7]) { // Edged: 4 - 7
        avg[1] += 1.0;
        avg[2] += root(values[4], values[7]);
        ++samples;
        edgeFlags |= ZEDGE;
    }

    if (signMask[0] != signMask[4]) { // Edged: 0 - 4
        avg[1] += root(values[0], values[4]);
        ++samples;
        edgeFlags |= YEDGE;
    }

    if (signMask[1] != signMask[5]) { // Edged: 1 - 5
        avg[0] += 1.0;
        avg[1] += root(values[1], values[5]);
        ++samples;
        edgeFlags |= YEDGE;
    }

    if (signMask[2] != signMask[6]) { // Edged: 2 - 6
        avg[0] += 1.0;
        avg[1] += root(values[2], values[6]);
        avg[2] += 1.0;
        ++samples;
        edgeFlags |= YEDGE;
    }

    if (signMask[3] != signMask[7]) { // Edged: 3 - 7
        avg[1] += root(values[3], values[7]);
        avg[2] += 1.0;
        ++samples;
        edgeFlags |= YEDGE;
    }

    if (samples > 1) {
        double w = 1.0 / double(samples);
        avg[0] *= w;
        avg[1] *= w;
        avg[2] *= w;
    }

    // offset by cell-origin
    avg[0] += double(ijk[0]);
    avg[1] += double(ijk[1]);
    avg[2] += double(ijk[2]);

    avg = mTransform.indexToWorld(avg);

    return edgeFlags;
}


////////////////////////////////////////


struct QuadMeshOp
{
    QuadMeshOp(): mIdx(0), mPolygonPool(NULL) {}

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


struct AdaptiveMeshOp
{
    AdaptiveMeshOp(): mQuadIdx(0), mTriangleIdx(0), mPolygonPool(NULL), mTmpPolygonPool() {}

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


template<class DistTreeT, class MeshingOp>
class MeshGen
{
public:
    typedef typename DistTreeT::template ValueConverter<char>::Type CharTreeT;
    typedef typename DistTreeT::template ValueConverter<int>::Type IntTreeT;

    MeshGen(const LeafCPtrList<CharTreeT>& edgeLeafs, const IntTreeT& auxTree, PolygonPoolList&);
    MeshGen(const MeshGen<DistTreeT, MeshingOp>&);

    void setRefData(const ReferenceData<DistTreeT>&);

    void runParallel();
    void runSerial();

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    const LeafCPtrList<CharTreeT>& mEdgeLeafs;
    const IntTreeT& mAuxTree;
    const PolygonPoolList& mPolygonPoolList;
    size_t mID;
    const ReferenceData<DistTreeT>* mRefData;
};


template<class DistTreeT, class MeshingOp>
MeshGen<DistTreeT, MeshingOp>::MeshGen(const LeafCPtrList<CharTreeT>& edgeLeafs,
    const IntTreeT& auxTree, PolygonPoolList& polygonPoolList)
    : mEdgeLeafs(edgeLeafs)
    , mAuxTree(auxTree)
    , mPolygonPoolList(polygonPoolList)
    , mRefData(NULL)
{
}


template<class DistTreeT, class MeshingOp>
MeshGen<DistTreeT, MeshingOp>::MeshGen(const MeshGen<DistTreeT, MeshingOp>& rhs)
    : mEdgeLeafs(rhs.mEdgeLeafs)
    , mAuxTree(rhs.mAuxTree)
    , mPolygonPoolList(rhs.mPolygonPoolList)
    , mRefData(rhs.mRefData)
{
}


template<class DistTreeT, class MeshingOp>
void
MeshGen<DistTreeT, MeshingOp>::setRefData(
    const ReferenceData<DistTreeT>& refData)
{
    mRefData = &refData;
}


template<class DistTreeT, class MeshingOp>
void
MeshGen<DistTreeT, MeshingOp>::runParallel()
{
    tbb::parallel_for(mEdgeLeafs.getRange(), *this);
}


template<class DistTreeT, class MeshingOp>
void
MeshGen<DistTreeT, MeshingOp>::runSerial()
{
    (*this)(mEdgeLeafs.getRange());
}


template<class DistTreeT, class MeshingOp>
void
MeshGen<DistTreeT, MeshingOp>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    typedef typename CharTreeT::LeafNodeType CharLeafT;

    typedef openvdb::tree::ValueAccessor<const CharTreeT> CharTreeAccessorT;
    typedef openvdb::tree::ValueAccessor<const IntTreeT> IntTreeAccessorT;
    typedef openvdb::tree::ValueAccessor<const BoolTreeT> BoolTreeAccessorT;

    boost::scoped_ptr<CharTreeAccessorT> refEdgeAcc;
    boost::scoped_ptr<BoolTreeAccessorT> refMaskAcc;
    if (mRefData && mRefData->isValid()) {
        refEdgeAcc.reset(new CharTreeAccessorT(*mRefData->mEdgeTree));
        refMaskAcc.reset(new BoolTreeAccessorT(*mRefData->mSeamMaskTree.get()));
    }
    const bool hasRefData = refEdgeAcc && refMaskAcc;


    typename CharTreeT::LeafNodeType::ValueOnCIter iter;
    IntTreeAccessorT auxAcc(mAuxTree);

    Coord ijk, coord;
    char refEdgeFlags, isSemLinePoly;
    const char isExteriorPoly[2] = {0, char(POLYFLAG_EXTERIOR)};
    openvdb::Vec4I quad;
    size_t edgeCount;

    MeshingOp mesher;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const Coord origin = mEdgeLeafs[n]->origin();

        // Get an upper bound on the number of primitives.
        edgeCount = 0;
        iter = mEdgeLeafs[n]->cbeginValueOn();
        for (; iter; ++iter) {
            char edgeFlags = iter.getValue() >> 1;
            edgeCount += edgeFlags & 0x1;

            edgeFlags = edgeFlags >> 1;
            edgeCount += edgeFlags & 0x1;

            edgeFlags = edgeFlags >> 1;
            edgeCount += edgeFlags & 0x1;
        }

        mesher.init(edgeCount, mPolygonPoolList[n]);

        const CharLeafT* refEdgeLeaf = NULL;
        const BoolLeafT* refMaskLeaf = NULL;

        if (hasRefData) {
            refEdgeLeaf = refEdgeAcc->probeConstLeaf(origin);
            refMaskLeaf = refMaskAcc->probeConstLeaf(origin);
        }

        iter = mEdgeLeafs[n]->cbeginValueOn();
        for (; iter; ++iter) {
            ijk = iter.getCoord();
            const char& edgeFlags = iter.getValue();

            const bool isInside = edgeFlags & INSIDE;

            refEdgeFlags = 0;
            isSemLinePoly = 0;
            if (hasRefData) {
                if(refEdgeLeaf) refEdgeFlags = refEdgeLeaf->getValue(ijk);
                if (refMaskLeaf && refMaskLeaf->isValueOn(ijk)) {
                    isSemLinePoly = char(POLYFLAG_FRACTURE_SEAM);
                }
            }


            int v0 = auxAcc.getValue(ijk);

            if (edgeFlags & XEDGE) {

                quad[0] = v0;
                coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2]; // i, j-1, k
                quad[1] = auxAcc.getValue(coord);
                coord[2] -= 1; // i, j-1, k-1
                quad[2] = auxAcc.getValue(coord);
                coord[1] = ijk[1]; // i, j, k-1
                quad[3] = auxAcc.getValue(coord);

                mesher.addPrim(quad, isInside,
                    (isSemLinePoly | isExteriorPoly[bool(refEdgeFlags & XEDGE)]));
            }


            if (edgeFlags & YEDGE) {

                quad[0] = v0;
                coord[0] = ijk[0]; coord[1] = ijk[1]; coord[2] = ijk[2]-1; // i, j, k-1
                quad[1] = auxAcc.getValue(coord);
                coord[0] -= 1; // i-1, j, k-1
                quad[2] = auxAcc.getValue(coord);
                coord[2] = ijk[2]; // i-1, j, k
                quad[3] = auxAcc.getValue(coord);

                mesher.addPrim(quad, isInside,
                    (isSemLinePoly | isExteriorPoly[bool(refEdgeFlags & YEDGE)]));
            }

            if (edgeFlags & ZEDGE) {

                quad[0] = v0;
                coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2]; // i, j-1, k
                quad[1] = auxAcc.getValue(coord);
                coord[0] -= 1; // i-1, j-1, k
                quad[2] = auxAcc.getValue(coord);
                coord[1] = ijk[1]; // i, j, k
                quad[3] = auxAcc.getValue(coord);

                mesher.addPrim(quad, !isInside,
                     (isSemLinePoly | isExteriorPoly[bool(refEdgeFlags & ZEDGE)]));
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
    if (threaded) {
        tbb::parallel_reduce(mLeafManager.getRange(), *this);
    } else {
        (*this)(mLeafManager.getRange());
    }
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


template<typename SrcTreeT>
class MaskGen
{
public:
    typedef tree::LeafManager<const SrcTreeT> LeafManagerT;
    typedef typename SrcTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef tree::ValueAccessor<const SrcTreeT> SrcAccessorT;
    typedef tree::ValueAccessor<BoolTreeT> BoolAccessorT;
    typedef Grid<BoolTreeT> BoolGridT;


    //////////


    MaskGen(const BoolGridT& mask, const LeafManagerT& srcLeafs,
        const math::Transform& srcXForm, bool invertMask);

    void run(bool threaded = true);

    BoolTreeT&  tree() { return mTree; }


    //////////

    MaskGen(MaskGen&, tbb::split);

    void operator()(const tbb::blocked_range<size_t>&);

    void join(MaskGen& rhs) { mTree.merge(rhs.mTree); }

private:

    const BoolGridT& mMask;
    const LeafManagerT& mLeafManager;
    const math::Transform& mSrcXForm;
    bool mInvertMask;
    BoolTreeT mTree;
};

template<typename SrcTreeT>
MaskGen<SrcTreeT>::MaskGen(const BoolGridT& mask, const LeafManagerT& srcLeafs,
    const math::Transform& srcXForm, bool invertMask)
    : mMask(mask)
    , mLeafManager(srcLeafs)
    , mSrcXForm(srcXForm)
    , mInvertMask(invertMask)
    , mTree(false)
{
}

template<typename SrcTreeT>
MaskGen<SrcTreeT>::MaskGen(MaskGen& rhs, tbb::split)
    : mMask(rhs.mMask)
    , mLeafManager(rhs.mLeafManager)
    , mSrcXForm(rhs.mSrcXForm)
    , mInvertMask(rhs.mInvertMask)
    , mTree(false)
{
}


template<typename SrcTreeT>
void
MaskGen<SrcTreeT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(mLeafManager.getRange(), *this);
    } else {
        (*this)(mLeafManager.getRange());
    }
}


template<typename SrcTreeT>
void
MaskGen<SrcTreeT>::operator()(const tbb::blocked_range<size_t>& range)
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
class BoundaryMaskGen
{
public:
    typedef typename SrcTreeT::template ValueConverter<int>::Type IntTreeT;
    typedef typename SrcTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef tree::LeafManager<const SrcTreeT> LeafManagerT;

    //////////

    BoundaryMaskGen(const LeafManagerT& leafs, const BoolTreeT&, const IntTreeT&);

    void run(bool threaded = true);

    BoolTreeT&  tree() { return mTree; }

    //////////

    BoundaryMaskGen(BoundaryMaskGen&, tbb::split);
    void operator()(const tbb::blocked_range<size_t>&);
    void join(BoundaryMaskGen& rhs) { mTree.merge(rhs.mTree); }

private:
    bool neighboringLeaf(const Coord&,
        const tree::ValueAccessor<const IntTreeT>&) const;

    const LeafManagerT& mLeafManager;
    const BoolTreeT& mMaskTree;
    const IntTreeT& mAuxTree;
    BoolTreeT mTree;
    CoordBBox mLeafBBox;
};


template<typename SrcTreeT>
BoundaryMaskGen<SrcTreeT>::BoundaryMaskGen(const LeafManagerT& leafs,
    const BoolTreeT& maskTree, const IntTreeT& auxTree)
    : mLeafManager(leafs)
    , mMaskTree(maskTree)
    , mAuxTree(auxTree)
    , mTree(false)
{
    mAuxTree.evalLeafBoundingBox(mLeafBBox);
    mLeafBBox.expand(IntTreeT::LeafNodeType::DIM);
}


template<typename SrcTreeT>
BoundaryMaskGen<SrcTreeT>::BoundaryMaskGen(BoundaryMaskGen& rhs, tbb::split)
    : mLeafManager(rhs.mLeafManager)
    , mMaskTree(rhs.mMaskTree)
    , mAuxTree(rhs.mAuxTree)
    , mTree(false)
    , mLeafBBox(rhs.mLeafBBox)
{
}


template<typename SrcTreeT>
void
BoundaryMaskGen<SrcTreeT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(mLeafManager.getRange(), *this);
    } else {
        (*this)(mLeafManager.getRange());
    }
}


template<typename SrcTreeT>
bool
BoundaryMaskGen<SrcTreeT>::neighboringLeaf(const Coord& ijk,
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
BoundaryMaskGen<SrcTreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    Coord ijk;
    tree::ValueAccessor<const BoolTreeT> maskAcc(mMaskTree);
    tree::ValueAccessor<const IntTreeT> auxAcc(mAuxTree);
    tree::ValueAccessor<BoolTreeT> acc(mTree);

    typename SrcTreeT::LeafNodeType::ValueOnCIter iter;

    for (size_t n = range.begin(); n != range.end(); ++n) {

        const typename SrcTreeT::LeafNodeType&
            leaf = mLeafManager.leaf(n);

        ijk = leaf.origin();

        if (!mLeafBBox.isInside(ijk) || !neighboringLeaf(ijk, auxAcc)) continue;

        const typename BoolTreeT::LeafNodeType*
            maskLeaf = maskAcc.probeConstLeaf(ijk);

        if (!maskLeaf || !leaf.hasSameTopology(maskLeaf)) {
            acc.touchLeaf(ijk)->topologyUnion(leaf);
        }
    }
}


////////////////////////////////////////


template<
    typename SrcTreeT,
    typename LeafManagerT = tree::LeafManager<const SrcTreeT>,
    typename AuxDataT = int>
class AuxiliaryData
{
public:
    typedef typename SrcTreeT::ValueType SrcValueT;
    typedef typename SrcTreeT::template ValueConverter<char>::Type CharTreeT;
    typedef typename SrcTreeT::template ValueConverter<AuxDataT>::Type AuxTreeT;
    typedef tree::ValueAccessor<const SrcTreeT> SrcAccessorT;
    typedef tree::ValueAccessor<CharTreeT> CharAccessorT;
    typedef tree::ValueAccessor<AuxTreeT> AuxAccessorT;

    //////////


    AuxiliaryData(const SrcTreeT&, const LeafManagerT&,
        double iso, bool extraCheck = false);

    void run(bool threaded = true);

    typename CharTreeT::Ptr edgeTree() const { return mEdgeTree; }
    typename AuxTreeT::Ptr auxTree() const { return mAuxTree; }


    //////////

    AuxiliaryData(AuxiliaryData&, tbb::split);

    void operator()(const tbb::blocked_range<size_t>&);

    void join(const AuxiliaryData& rhs)
    {
        mEdgeTree->merge(*rhs.mEdgeTree);
        mAuxTree->merge(*rhs.mAuxTree);
    }

private:

    int edgeCheck(const Coord& ijk, const bool thisInside);

    const LeafManagerT& mLeafManager;

    const SrcTreeT& mSourceTree;
    SrcAccessorT mSourceAccessor;

    typename CharTreeT::Ptr mEdgeTree;
    CharAccessorT mEdgeAccessor;

    typename AuxTreeT::Ptr mAuxTree;
    AuxAccessorT mAuxAccessor;

    const double mIsovalue;
    const bool mExtraCheck;
};

template<typename SrcTreeT, typename LeafManagerT, typename AuxDataT>
AuxiliaryData<SrcTreeT, LeafManagerT, AuxDataT>::AuxiliaryData(const SrcTreeT& tree,
    const LeafManagerT& leafs, double iso, bool extraCheck)
    : mLeafManager(leafs)
    , mSourceTree(tree)
    , mSourceAccessor(mSourceTree)
    , mEdgeTree(new CharTreeT(0))
    , mEdgeAccessor(*mEdgeTree)
    , mAuxTree(new AuxTreeT(AuxDataT(0)))
    , mAuxAccessor(*mAuxTree)
    , mIsovalue(iso)
    , mExtraCheck(extraCheck)
{
}

template<typename SrcTreeT, typename LeafManagerT, typename AuxDataT>
AuxiliaryData<SrcTreeT, LeafManagerT, AuxDataT>::AuxiliaryData(AuxiliaryData& rhs, tbb::split)
    : mLeafManager(rhs.mLeafManager)
    , mSourceTree(rhs.mSourceTree)
    , mSourceAccessor(mSourceTree)
    , mEdgeTree(new CharTreeT(0))
    , mEdgeAccessor(*mEdgeTree)
    , mAuxTree(new AuxTreeT(AuxDataT(0)))
    , mAuxAccessor(*mAuxTree)
    , mIsovalue(rhs.mIsovalue)
    , mExtraCheck(rhs.mExtraCheck)
{
}


template<typename SrcTreeT, typename LeafManagerT, typename AuxDataT>
void
AuxiliaryData<SrcTreeT, LeafManagerT, AuxDataT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(mLeafManager.getRange(), *this);
    } else {
        (*this)(mLeafManager.getRange());
    }
}


template<typename SrcTreeT, typename LeafManagerT, typename AuxDataT>
void
AuxiliaryData<SrcTreeT, LeafManagerT, AuxDataT>::operator()(const tbb::blocked_range<size_t>& range)
{
    typename LeafManagerT::TreeType::LeafNodeType::ValueOnCIter iter;
    Coord ijk;
    bool thisInside;
    int edgeFlags;
    SrcValueT val;

    if (!mExtraCheck) {
        for (size_t n = range.begin(); n != range.end(); ++n) {


            for (iter = mLeafManager.leaf(n).cbeginValueOn(); iter; ++iter) {

                ijk = iter.getCoord();

                thisInside = mSourceAccessor.getValue(ijk) < mIsovalue;
                edgeFlags = edgeCheck(ijk, thisInside);

                if (edgeFlags != 0) {
                    edgeFlags |= int(thisInside);
                    mEdgeAccessor.setValue(ijk, char(edgeFlags));
                }
            }
        }
    } else {
        for (size_t n = range.begin(); n != range.end(); ++n) {

            for (iter = mLeafManager.leaf(n).cbeginValueOn(); iter; ++iter) {

                ijk = iter.getCoord();

                thisInside = mSourceAccessor.getValue(ijk) < mIsovalue;
                edgeFlags = edgeCheck(ijk, thisInside);

                if (edgeFlags != 0) {
                    edgeFlags |= int(thisInside);
                    mEdgeAccessor.setValue(ijk, char(edgeFlags));
                }

                --ijk[0];
                if (!mSourceAccessor.probeValue(ijk, val)) {
                    thisInside = val < mIsovalue;
                    edgeFlags = edgeCheck(ijk, thisInside);

                    if (edgeFlags != 0) {
                        edgeFlags |= int(thisInside);
                        mEdgeAccessor.setValue(ijk, char(edgeFlags));
                    }
                }

                ++ijk[0];
                --ijk[1];
                if (!mSourceAccessor.probeValue(ijk, val)) {
                    thisInside = val < mIsovalue;
                    edgeFlags = edgeCheck(ijk, thisInside);

                    if (edgeFlags != 0) {
                        edgeFlags |= int(thisInside);
                        mEdgeAccessor.setValue(ijk, char(edgeFlags));
                    }
                }

                ++ijk[1];
                --ijk[2];
                if (!mSourceAccessor.probeValue(ijk, val)) {
                    thisInside = val < mIsovalue;
                    edgeFlags = edgeCheck(ijk, thisInside);

                    if (edgeFlags != 0) {
                        edgeFlags |= int(thisInside);
                        mEdgeAccessor.setValue(ijk, char(edgeFlags));
                    }
                }
            }
        }
    }
}

template<typename SrcTreeT, typename LeafManagerT, typename AuxDataT>
inline int
AuxiliaryData<SrcTreeT, LeafManagerT, AuxDataT>::edgeCheck(const Coord& ijk, const bool thisInside)
{
    int edgeFlags = 0;
    Coord n_ijk, coord;

    // Eval upwind x-edge
    n_ijk = ijk; ++n_ijk[0];
    bool otherInside = (mSourceAccessor.getValue(n_ijk) < mIsovalue);
    if (otherInside != thisInside) {

        edgeFlags = XEDGE;

        mAuxAccessor.setActiveState(ijk, true);

        coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2];
        mAuxAccessor.setActiveState(coord, true);

        coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2]-1;
        mAuxAccessor.setActiveState(coord, true);

        coord[0] = ijk[0]; coord[1] = ijk[1]; coord[2] = ijk[2]-1;
        mAuxAccessor.setActiveState(coord, true);
    }

    // Eval upwind y-edge
    n_ijk[0] = ijk[0]; ++n_ijk[1];
    otherInside = (mSourceAccessor.getValue(n_ijk) < mIsovalue);
    if (otherInside != thisInside) {

        edgeFlags |= YEDGE;

        mAuxAccessor.setActiveState(ijk, true);

        coord[0] = ijk[0]; coord[1] = ijk[1]; coord[2] = ijk[2]-1;
        mAuxAccessor.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]; coord[2] = ijk[2];
        mAuxAccessor.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]; coord[2] = ijk[2]-1;
        mAuxAccessor.setActiveState(coord, true);
    }

    // Eval upwind z-edge
    n_ijk[1] = ijk[1]; ++n_ijk[2];
    otherInside = (mSourceAccessor.getValue(n_ijk) < mIsovalue);
    if (otherInside != thisInside) {

        edgeFlags |= ZEDGE;

        mAuxAccessor.setActiveState(ijk, true);

        coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2];
        mAuxAccessor.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]; coord[2] = ijk[2];
        mAuxAccessor.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]-1; coord[2] = ijk[2];
        mAuxAccessor.setActiveState(coord, true);
    }
    return edgeFlags;
}




////////////////////////////////////////


template <class DistTreeT>
class SeamMaskGen
{
public:
    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename DistTreeT::template ValueConverter<int>::Type  IntTreeT;
    typedef tree::ValueAccessor<const BoolTreeT> BoolTreeAccessorT;
    typedef tree::ValueAccessor<const IntTreeT> IntTreeAccessorT;

    SeamMaskGen(LeafPtrList<BoolTreeT>& seamMaskLeafs,
        const BoolTreeT& topologyMaskTree, const IntTreeT& auxTree);

    SeamMaskGen(const SeamMaskGen<DistTreeT>&);

    void runParallel();
    void runSerial();

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    LeafPtrList<BoolTreeT>& mSeamMaskLeafs;
    const BoolTreeT& mTopologyMaskTree;
    BoolTreeAccessorT mTopologyMaskAcc;
    const IntTreeT& mAuxTree;
    IntTreeAccessorT mAuxAcc;
};


template <class DistTreeT>
SeamMaskGen<DistTreeT>::SeamMaskGen(LeafPtrList<BoolTreeT>& seamMaskLeafs,
    const BoolTreeT& topologyMaskTree, const IntTreeT& auxTree)
    : mSeamMaskLeafs(seamMaskLeafs)
    , mTopologyMaskTree(topologyMaskTree)
    , mTopologyMaskAcc(mTopologyMaskTree)
    , mAuxTree(auxTree)
    , mAuxAcc(mAuxTree)
{
}

template <class DistTreeT>
SeamMaskGen<DistTreeT>::SeamMaskGen(const SeamMaskGen<DistTreeT>& rhs)
    : mSeamMaskLeafs(rhs.mSeamMaskLeafs)
    , mTopologyMaskTree(rhs.mTopologyMaskTree)
    , mTopologyMaskAcc(mTopologyMaskTree)
    , mAuxTree(rhs.mAuxTree)
    , mAuxAcc(mAuxTree)
{
}

template <class DistTreeT>
void
SeamMaskGen<DistTreeT>::runParallel()
{
    tbb::parallel_for(mSeamMaskLeafs.getRange(), *this);
}

template <class DistTreeT>
void
SeamMaskGen<DistTreeT>::runSerial()
{
    (*this)(mSeamMaskLeafs.getRange());
}

template <class DistTreeT>
void
SeamMaskGen<DistTreeT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typedef typename BoolTreeT::LeafNodeType::ValueOnIter ValueOnIterT;
    Coord ijk, n_ijk;
    for (size_t leafIdx = range.begin(); leafIdx != range.end(); ++leafIdx) {
        ValueOnIterT it = mSeamMaskLeafs[leafIdx]->beginValueOn();
        for (; it; ++it) {
            ijk = it.getCoord();
            if (!mTopologyMaskAcc.isValueOn(ijk)) {
                it.setValueOff();
            } else {
                bool turnOff = true;
                for (size_t n = 0; n < 6; ++n) {
                    n_ijk = ijk + util::COORD_OFFSETS[n];
                    if (!mAuxTree.isValueOn(n_ijk) && mTopologyMaskAcc.isValueOn(n_ijk)) {
                        turnOff = false;
                        break;
                    }
                }
                if (turnOff) it.setValueOff();
            }
        }
    }
}

////////////////////////////////////////


template <class DistTreeT>
class EdgeSmooth
{
public:
    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename DistTreeT::template ValueConverter<int>::Type  IntTreeT;
    typedef tree::ValueAccessor<const BoolTreeT> BoolTreeAccessorT;
    typedef tree::ValueAccessor<const IntTreeT> IntTreeAccessorT;

    EdgeSmooth(
        LeafPtrList<BoolTreeT>& leafs,
        const BoolTreeT& edgeMaskTree,
        const IntTreeT& auxTree,
        PointList& points,
        const math::Transform& xform);

    EdgeSmooth(const EdgeSmooth<DistTreeT>&);

    void runParallel(const size_t iterations);
    void runSerial(const size_t iterations);

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    LeafPtrList<BoolTreeT>& mLeafs;
    const BoolTreeT& mEdgeMaskTree;
    const IntTreeT& mAuxTree;
    PointList& mPoints;
    const math::Transform& mTransform;

    bool pointInAABB(const Vec3s& p, const Vec3s& bmin, const Vec3s& bmax) const
    {
        for (int i = 0; i < 3; ++i) {
            if (p[i] < bmin[i] || p[i] > bmax[i]) {
                return false;
            }
        }
        return true;
    }

};


template <class DistTreeT>
EdgeSmooth<DistTreeT>::EdgeSmooth(
    LeafPtrList<BoolTreeT>& leafs,
    const BoolTreeT& edgeMaskTree,
    const IntTreeT& auxTree,
    PointList& points,
    const math::Transform& xform)
    : mLeafs(leafs)
    , mEdgeMaskTree(edgeMaskTree)
    , mAuxTree(auxTree)
    , mPoints(points)
    , mTransform(xform)
{
}

template <class DistTreeT>
EdgeSmooth<DistTreeT>::EdgeSmooth(const EdgeSmooth<DistTreeT>& rhs)
    : mLeafs(rhs.mLeafs)
    , mEdgeMaskTree(rhs.mEdgeMaskTree)
    , mAuxTree(rhs.mAuxTree)
    , mPoints(rhs.mPoints)
    , mTransform(rhs.mTransform)
{
}

template <class DistTreeT>
void
EdgeSmooth<DistTreeT>::runParallel(const size_t iterations)
{
    for (size_t i = 0; i < iterations; ++i) {
        tbb::parallel_for(mLeafs.getRange(), *this);
    }
}

template <class DistTreeT>
void
EdgeSmooth<DistTreeT>::runSerial(const size_t iterations)
{
    for (size_t i = 0; i < iterations; ++i) {
        (*this)(mLeafs.getRange());
    }
}

template <class DistTreeT>
void
EdgeSmooth<DistTreeT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typedef typename BoolTreeT::LeafNodeType::ValueOnIter ValueOnIterT;
    BoolTreeAccessorT maskAcc(mEdgeMaskTree);
    IntTreeAccessorT auxAcc(mAuxTree);

    float dx = float(mTransform.voxelSize()[0]);
    openvdb::Vec3s avg, bmin, bmax;
    Coord ijk, nijk;
    int count;

    for (size_t leafIdx = range.begin(); leafIdx != range.end(); ++leafIdx) {
        typename BoolTreeT::LeafNodeType::ValueOnIter valueIt = mLeafs[leafIdx]->beginValueOn();
        for ( ; valueIt; ++valueIt) {

            ijk = valueIt.getCoord();
            openvdb::Vec3s& ptn = mPoints[auxAcc.getValue(ijk)];

            avg = ptn;
            count = 1;
            for (int n = 0; n < 26; ++n) {
                nijk = ijk + util::COORD_OFFSETS[n];
                if (maskAcc.isValueOn(nijk)) {
                    avg += mPoints[auxAcc.getValue(nijk)];
                    ++count;
                }
            }

            if (count > 2) {
                avg *= (1.0 / float(count));

                // Constrain to current cell
                bmin = openvdb::Vec3s(mTransform.indexToWorld(ijk));
                bmax = bmin + dx;

                bool inCell = false;
                for (int i = 0; i < 10; ++i) {

                    inCell = pointInAABB(avg, bmin, bmax);

                    if (inCell) {
                        break;
                    }

                    avg += ptn;
                    avg *= 0.5;
                }

                if (inCell) ptn = avg;

                /*Vec3d a = mTransform.worldToIndex(ptn);
                Vec3d b = mTransform.worldToIndex(avg);

                bmin[0] = float(ijk[0]);
                bmin[1] = float(ijk[1]);
                bmin[2] = float(ijk[2]);
                bmax = bmin + 1.0;

                if(!pointInAABB(a, bmin, bmax)) continue;

                while (true) {
                    if(pointInAABB(b, bmin, bmax)) break;

                    b += a;
                    b *= 0.5;
                }

                ptn = mTransform.indexToWorld(b);*/
            }
        }
    }
}


////////////////////////////////////////


template<class CharAccessorT, typename AuxAccessorT>
class AuxDataGenerator
{
public:
    AuxDataGenerator(CharAccessorT& edgeAcc, AuxAccessorT& auxAcc)
        : mEdgeAcc(edgeAcc), mAuxAcc(auxAcc) {}


    void setXEdge(char edgeFlags, const Coord& ijk)
    {
        mEdgeAcc.setValue(ijk, edgeFlags | XEDGE);

        mAuxAcc.setActiveState(ijk, true);

        Coord coord = ijk;
        coord[1] = ijk[1]-1;
        mAuxAcc.setActiveState(coord, true);

        coord[0] = ijk[0]; coord[1] = ijk[1]-1; coord[2] = ijk[2]-1;
        mAuxAcc.setActiveState(coord, true);

        coord[0] = ijk[0]; coord[1] = ijk[1]; coord[2] = ijk[2]-1;
        mAuxAcc.setActiveState(coord, true);
    }

    void setYEdge(char edgeFlags, const Coord& ijk)
    {
        mEdgeAcc.setValue(ijk, edgeFlags | YEDGE);

        mAuxAcc.setActiveState(ijk, true);

        Coord coord = ijk;
        coord[2] = ijk[2]-1;
        mAuxAcc.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]; coord[2] = ijk[2];
        mAuxAcc.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]; coord[2] = ijk[2]-1;
        mAuxAcc.setActiveState(coord, true);
    }

    void setZEdge(char edgeFlags, const Coord& ijk)
    {
        mEdgeAcc.setValue(ijk, edgeFlags | ZEDGE);

        mAuxAcc.setActiveState(ijk, true);

        Coord coord = ijk;
        coord[1] = ijk[1]-1;
        mAuxAcc.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]; coord[2] = ijk[2];
        mAuxAcc.setActiveState(coord, true);

        coord[0] = ijk[0]-1; coord[1] = ijk[1]-1; coord[2] = ijk[2];
        mAuxAcc.setActiveState(coord, true);
    }

private:
    CharAccessorT& mEdgeAcc;
    AuxAccessorT& mAuxAcc;
};


////////////////////////////////////////


template<class DistTreeT, class AuxTreeT, class CharTreeT>
inline void
tileAuxiliaryData(
    const DistTreeT& distTree, CharTreeT& edgeTree, AuxTreeT& auxTree,
    double iso)
{
    typedef tree::ValueAccessor<const DistTreeT> DistAccessorT;
    typedef tree::ValueAccessor<AuxTreeT> AuxTreeAccessorT;
    typedef tree::ValueAccessor<CharTreeT> CharTreeAccessorT;

    typename DistTreeT::ValueType isoValue = typename DistTreeT::ValueType(iso);

    DistAccessorT distAcc(distTree);
    CharTreeAccessorT edgeAcc(edgeTree);
    AuxTreeAccessorT auxAcc(auxTree);

    AuxDataGenerator<CharTreeAccessorT, AuxTreeAccessorT> auxData(edgeAcc, auxAcc);

    Coord ijk, nijk;
    typename DistTreeT::ValueType value;
    CoordBBox bbox;
    bool processTileFace;

    typename DistTreeT::ValueOnCIter tileIter(distTree);
    tileIter.setMaxDepth(DistTreeT::ValueOnCIter::LEAF_DEPTH - 1);

    for ( ; tileIter; ++tileIter) {
        tileIter.getBoundingBox(bbox);

        const bool thisInside = (distAcc.getValue(bbox.min()) < isoValue);
        const int thisDepth = distAcc.getValueDepth(bbox.min());


        // Eval x-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[0];

        processTileFace = true;
        if (thisDepth >= distAcc.getValueDepth(nijk)) {
            processTileFace = thisInside != (distAcc.getValue(nijk) < isoValue);
        }

        if (processTileFace) {
            for (ijk[1] = bbox.min()[1]; ijk[1] <= bbox.max()[1]; ++ijk[1]) {
                nijk[1] = ijk[1];
                for (ijk[2] = bbox.min()[2]; ijk[2] <= bbox.max()[2]; ++ijk[2]) {
                    nijk[2] = ijk[2];
                    if ((distAcc.getValue(nijk) < isoValue) != thisInside) {
                        auxData.setXEdge(edgeAcc.getValue(ijk) | char(thisInside), ijk);
                    }
                }
            }
        }

        ijk = bbox.min();
        --ijk[0];

        processTileFace = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processTileFace = !distAcc.probeValue(ijk, value) && thisInside != (value < isoValue);
        }

        if (processTileFace) {
            for (ijk[1] = bbox.min()[1]; ijk[1] <= bbox.max()[1]; ++ijk[1]) {
                for (ijk[2] = bbox.min()[2]; ijk[2] <= bbox.max()[2]; ++ijk[2]) {
                    if (!distAcc.probeValue(ijk, value) && (value < isoValue) != thisInside) {
                         auxData.setXEdge(edgeAcc.getValue(ijk) | char(!thisInside), ijk);
                    }
                }
            }
        }


        // Eval y-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[1];

        processTileFace = true;
        if (thisDepth >= distAcc.getValueDepth(nijk)) {
            processTileFace = thisInside != (distAcc.getValue(nijk) < isoValue);
        }

        if (processTileFace) {
            for (ijk[0] = bbox.min()[0]; ijk[0] <= bbox.max()[0]; ++ijk[0]) {
                nijk[0] = ijk[0];
                for (ijk[2] = bbox.min()[2]; ijk[2] <= bbox.max()[2]; ++ijk[2]) {
                    nijk[2] = ijk[2];

                    if ((distAcc.getValue(nijk) < isoValue) != thisInside) {
                        auxData.setYEdge(edgeAcc.getValue(ijk) | char(thisInside), ijk);
                    }
                }
            }
        }


        ijk = bbox.min();
        --ijk[1];

        processTileFace = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processTileFace = !distAcc.probeValue(ijk, value) && thisInside != (value < isoValue);
        }

        if (processTileFace) {
            for (ijk[0] = bbox.min()[0]; ijk[0] <= bbox.max()[0]; ++ijk[0]) {
                for (ijk[2] = bbox.min()[2]; ijk[2] <= bbox.max()[2]; ++ijk[2]) {

                    if (!distAcc.probeValue(ijk, value) && (value < isoValue) != thisInside) {
                        auxData.setYEdge(edgeAcc.getValue(ijk) | char(!thisInside), ijk);
                    }
                }
            }
        }


        // Eval z-edges

        ijk = bbox.max();
        nijk = ijk;
        ++nijk[2];

        processTileFace = true;
        if (thisDepth >= distAcc.getValueDepth(nijk)) {
            processTileFace = thisInside != (distAcc.getValue(nijk) < isoValue);
        }

        if (processTileFace) {
            for (ijk[0] = bbox.min()[0]; ijk[0] <= bbox.max()[0]; ++ijk[0]) {
                nijk[0] = ijk[0];
                for (ijk[1] = bbox.min()[1]; ijk[1] <= bbox.max()[1]; ++ijk[1]) {
                    nijk[1] = ijk[1];

                    if ((distAcc.getValue(nijk) < isoValue) != thisInside) {
                        auxData.setZEdge(edgeAcc.getValue(ijk) | char(thisInside), ijk);
                    }
                }
            }
        }

        ijk = bbox.min();
        --ijk[2];

        processTileFace = true;
        if (thisDepth >= distAcc.getValueDepth(ijk)) {
            processTileFace = !distAcc.probeValue(ijk, value) && thisInside != (value < isoValue);
        }

        if (processTileFace) {
            for (ijk[0] = bbox.min()[0]; ijk[0] <= bbox.max()[0]; ++ijk[0]) {
                for (ijk[1] = bbox.min()[1]; ijk[1] <= bbox.max()[1]; ++ijk[1]) {

                    if (!distAcc.probeValue(ijk, value) && (value < isoValue) != thisInside) {
                        auxData.setZEdge(edgeAcc.getValue(ijk) | char(!thisInside), ijk);
                    }
                }
            }
        }
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
    , mRefEdgeTree(TreeBase::Ptr())
    , mRefTopologyMaskTree(TreeBase::Ptr())
    , mSeamPointTree(TreeBase::Ptr())
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
VolumeToMesh::setRefGrid(const GridBase::ConstPtr& grid, double secAdaptivity, bool smoothSeams)
{
    mRefGrid = grid;
    mSecAdaptivity = secAdaptivity;
    // Clear out old auxiliary data
    mRefEdgeTree = TreeBase::Ptr();
    mRefTopologyMaskTree = TreeBase::Ptr();
    mSeamPointTree = TreeBase::Ptr();
    mSmoothSeams = smoothSeams;
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
    typedef typename DistTreeT::ValueType DistValueT;

    typedef typename DistTreeT::template ValueConverter<char>::Type CharTreeT;
    typedef typename DistTreeT::template ValueConverter<int>::Type IntTreeT;
    typedef typename DistTreeT::template ValueConverter<Vec3s>::Type Vec3sTreeT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef Grid<BoolTreeT> BoolGridT;

    typedef typename DistTreeT::template ValueConverter<float>::Type FloatTreeT;
    typedef Grid<FloatTreeT> FloatGridT;

    const openvdb::math::Transform& transform = distGrid.transform();
    const DistTreeT& distTree = distGrid.tree();
    typename CharTreeT::Ptr edgeTree; // edge flags
    typename IntTreeT::Ptr auxTree; // auxiliary data
    typename BoolTreeT::Ptr pointMask;

    const bool nonLevelSetGrid = distGrid.getGridClass() != GRID_LEVEL_SET;
    const bool noAdaptivity = mPrimAdaptivity < 1e-6 && mSecAdaptivity < 1e-6;

    const bool extraSignCheck = nonLevelSetGrid ||
        (std::abs(mIsovalue - double(distGrid.background())) < (1.5 * transform.voxelSize()[0]));

    const BoolGridT * surfaceMask = NULL;
    if (mSurfaceMaskGrid) {
        surfaceMask = static_cast<const BoolGridT*>(mSurfaceMaskGrid.get());
    }

    const FloatGridT * adaptivityField = NULL;
    if (mAdaptivityGrid) {
        adaptivityField = static_cast<const FloatGridT*>(mAdaptivityGrid.get());
    }

    // Collect auxiliary data
    {
        if (surfaceMask || mPartitions > 1) {

            BoolTreeT valueMask(false);

            if (surfaceMask) {

                // Mask
                {
                    tree::LeafManager<const DistTreeT> leafs(distTree);
                    internal::MaskGen<DistTreeT> masking(
                        *surfaceMask, leafs, transform, mInvertSurfaceMask);
                    masking.run();
                    valueMask.merge(masking.tree());
                }

                // Partition
                if (mPartitions > 1) {
                    tree::LeafManager<BoolTreeT> leafs(valueMask);
                    leafs.foreach(internal::PartOp(leafs.leafCount() , mPartitions, mActivePart));
                    valueMask.pruneInactive();
                }

            } else {

                // Partition
                tree::LeafManager<const DistTreeT> leafs(distTree);
                internal::PartGen<DistTreeT> partitioner(leafs, mPartitions, mActivePart);
                partitioner.run();
                valueMask.merge(partitioner.tree());
            }

            {
                tree::LeafManager<BoolTreeT> leafs(valueMask);
                internal::AuxiliaryData<DistTreeT, tree::LeafManager<BoolTreeT> >
                    op(distTree, leafs, mIsovalue, extraSignCheck);
                op.run();

                edgeTree = op.edgeTree();
                auxTree = op.auxTree();
            }


            if (!noAdaptivity) {

                pointMask = typename BoolTreeT::Ptr(new BoolTreeT(false));
                pointMask->topologyUnion(*auxTree);

                tree::LeafManager<const DistTreeT> distLeafs(distTree);
                internal::BoundaryMaskGen<DistTreeT> boundary(distLeafs, valueMask, *auxTree);
                boundary.run();

                {
                    tree::LeafManager<BoolTreeT> leafs(boundary.tree());
                    internal::AuxiliaryData<DistTreeT, tree::LeafManager<BoolTreeT> >
                        op(distTree, leafs, mIsovalue, extraSignCheck);
                    op.run();

                    op.auxTree()->merge(*auxTree);
                    auxTree = op.auxTree();
                }

            }

        } else {

            tree::LeafManager<const DistTreeT> leafs(distTree);
            internal::AuxiliaryData<DistTreeT> op(distTree, leafs, mIsovalue, extraSignCheck);
            op.run();

            edgeTree = op.edgeTree();
            auxTree = op.auxTree();
        }

        // Collect auxiliary data from active tiles
        if (nonLevelSetGrid) {
            internal::tileAuxiliaryData(distTree, *edgeTree, *auxTree, mIsovalue);
        }
    }

    // Optionally collect auxiliary data from a reference surface.
    internal::ReferenceData<DistTreeT> refData;
    if (mRefGrid) {
        const GridT* refGrid = static_cast<const GridT*>(mRefGrid.get());
        if (refGrid && refGrid->activeVoxelCount() > 0) {

            refData.mDistTree = &refGrid->tree();
            refData.mInternalAdaptivity = DistValueT(mSecAdaptivity);

            // Cache reference data for reuse.
            if (!mRefEdgeTree && !mRefTopologyMaskTree) {
                tree::LeafManager<const DistTreeT> leafs(*refData.mDistTree);
                internal::AuxiliaryData<DistTreeT, tree::LeafManager<const DistTreeT>, bool> op(
                    *refData.mDistTree, leafs, mIsovalue, extraSignCheck);
                op.run();
                mRefEdgeTree = op.edgeTree();
                mRefTopologyMaskTree = op.auxTree();
                mSeamPointTree = typename Vec3sTreeT::Ptr(new Vec3sTreeT(Vec3s(0.0)));
            }

            if (mRefEdgeTree && mRefTopologyMaskTree) {
                refData.mEdgeTree = static_cast<CharTreeT*>(mRefEdgeTree.get());
                refData.mTopologyMaskTree =  static_cast<BoolTreeT*>(mRefTopologyMaskTree.get());
                refData.mSeamPointTree = static_cast<Vec3sTreeT*>(mSeamPointTree.get());
                refData.mSeamMaskTree = typename BoolTreeT::Ptr(new BoolTreeT(false));
                refData.mSmoothingMaskTree = typename BoolTreeT::Ptr(new BoolTreeT(false));
            }
        }
    }

    BoolTreeT edgeMaskTree(0.0);

    // Generate the seamline mask
    if (refData.mSeamMaskTree) {
        refData.mSeamMaskTree->topologyUnion(*auxTree.get());

        internal::LeafPtrList<BoolTreeT> leafs(*refData.mSeamMaskTree.get());
        internal::SeamMaskGen<DistTreeT> op(leafs, *refData.mTopologyMaskTree, *auxTree.get());
        op.runParallel();

        refData.mSeamMaskTree->pruneInactive();

        edgeMaskTree.topologyUnion(*refData.mSeamMaskTree);

        dilateVoxels(*refData.mSeamMaskTree, 3);

        if (mAdaptivityMaskTree) {

            const BoolTreeT* refGrid = static_cast<const BoolTreeT*>(
                 mAdaptivityMaskTree.get());

            refData.mSeamMaskTree->topologyUnion(*refGrid);
        }
    }

    // Process auxiliary data
    {
        internal::LeafPtrList<IntTreeT> auxLeafs(*auxTree);
        std::vector<size_t> regions(auxLeafs.size(), 0);

        {
            if (noAdaptivity) {
                internal::Count<IntTreeT> count(auxLeafs, regions);
                count.runParallel();
            } else {

                internal::Merge<DistTreeT> merge(distTree, auxLeafs,
                    DistValueT(mIsovalue), DistValueT(mPrimAdaptivity));

                merge.setRefData(refData);

                if (adaptivityField) {
                    merge.setSpatialAdaptivity(transform, *adaptivityField);
                }

                merge.run();

                internal::RegionCount<IntTreeT> count(auxLeafs, regions, pointMask.get());
                count.run();
         }


            mPointListSize = 0;
            size_t tmp = 0;
            for (size_t n = 0, N = regions.size(); n < N; ++n) {
                tmp = regions[n];
                regions[n] = mPointListSize;
                mPointListSize += tmp;
            }
        }

        if (refData.isValid()) { // match leaf topology
            tree::ValueAccessor<BoolTreeT> acc(*refData.mSmoothingMaskTree);
            for (size_t n = 0, N = auxLeafs.size(); n < N; ++n) {
                acc.touchLeaf(auxLeafs[n]->origin());
            }
        }

        // Generate the unique point list
        mPoints.reset(new openvdb::Vec3s[mPointListSize]);

        internal::PointGen<DistTreeT>
            pointGen(distTree, auxLeafs, regions, transform, mPoints, mIsovalue);
        pointGen.setRefData(refData);
        pointGen.runParallel();
    }

    // Smooth seam line edges
    if (mSmoothSeams && refData.isValid()) {

        refData.mSmoothingMaskTree->pruneInactive();
        internal::LeafPtrList<BoolTreeT> leafs(*refData.mSmoothingMaskTree);
        internal::EdgeSmooth<DistTreeT> op(leafs, edgeMaskTree, *auxTree, mPoints, transform);
        op.runParallel(3);

        // Cache shared points
        tree::ValueAccessor<Vec3sTreeT> ptnAcc(*refData.mSeamPointTree);
        tree::ValueAccessor<IntTreeT> auxAcc(*auxTree);
        Coord ijk;

        typename BoolTreeT::LeafIter leafIt = refData.mSeamMaskTree->beginLeaf();
        for ( ; leafIt; ++leafIt) {
            typename BoolTreeT::LeafNodeType::ValueOnIter valueIt = leafIt->beginValueOn();
            for ( ; valueIt; ++valueIt) {
                ijk = valueIt.getCoord();
                const int idx = auxAcc.getValue(ijk);
                if (idx != 0 && !ptnAcc.isValueOn(ijk)) {
                    ptnAcc.setValue(ijk, mPoints[idx]);
                }
            }
        }
    }

    // Generate mesh
    {
        internal::LeafCPtrList<CharTreeT> edgeLeafs(*edgeTree);
        mPolygonPoolListSize = edgeLeafs.size();
        mPolygons.reset(new PolygonPool[mPolygonPoolListSize]);

        if (noAdaptivity) {
            internal::MeshGen<DistTreeT, internal::QuadMeshOp>
                meshGen(edgeLeafs, *auxTree, mPolygons);
            meshGen.setRefData(refData);
            meshGen.runParallel();
        } else {

            internal::MeshGen<DistTreeT, internal::AdaptiveMeshOp>
                meshGen(edgeLeafs, *auxTree, mPolygons);
            meshGen.setRefData(refData);
            meshGen.runParallel();
        }
    }

    // Clean up unused points, only necessary if adaptivity and
    // automatic mesh partitioning is enabled.
    if (mPartitions > 1 && !noAdaptivity) {

        boost::scoped_array<bool> usedPointMask;
        if (mPointListSize > 0) {
            usedPointMask.reset(new bool[mPointListSize]);
            for (size_t n = 0; n < mPointListSize; ++n) {
                usedPointMask[n] = false;
            }
        }

        for (size_t n = 0; n < mPolygonPoolListSize; ++n) {
            const PolygonPool& polygons = mPolygons[n];

            for (size_t i = 0; i < polygons.numQuads(); ++i) {
                const Vec4I& quad = polygons.quad(i);
                usedPointMask[quad[0]] = true;
                usedPointMask[quad[1]] = true;
                usedPointMask[quad[2]] = true;
                usedPointMask[quad[3]] = true;
            }

            for (size_t i = 0; i < polygons.numTriangles(); ++i) {
                const Vec3I& triangle = polygons.triangle(i);
                usedPointMask[triangle[0]] = true;
                usedPointMask[triangle[1]] = true;
                usedPointMask[triangle[2]] = true;
            }
        }


        size_t removedPoints = 0;
        for (size_t p = 0; p < mPointListSize; ++p) {

            if (!usedPointMask[p]) {
                for (size_t n = 0; n < mPolygonPoolListSize; ++n) {
                    PolygonPool& polygons = mPolygons[n];

                    for (size_t i = 0; i < polygons.numQuads(); ++i) {
                        Vec4I& quad = polygons.quad(i);
                        if (quad[0] > (p - removedPoints)) quad[0] -= 1;
                        if (quad[1] > (p - removedPoints)) quad[1] -= 1;
                        if (quad[2] > (p - removedPoints)) quad[2] -= 1;
                        if (quad[3] > (p - removedPoints)) quad[3] -= 1;
                    }

                    for (size_t i = 0; i < polygons.numTriangles(); ++i) {
                        Vec3I& triangle = polygons.triangle(i);
                        if (triangle[0] > (p - removedPoints)) triangle[0] -= 1;
                        if (triangle[1] > (p - removedPoints)) triangle[1] -= 1;
                        if (triangle[2] > (p - removedPoints)) triangle[2] -= 1;
                    }
                }
                removedPoints++;
            }
        }

        if (removedPoints > 0) {
            size_t oldPointListSize = mPointListSize;
            mPointListSize = mPointListSize - removedPoints;
            openvdb::Vec3s* newPointList = new openvdb::Vec3s[mPointListSize];

            size_t index = 0;
            for (size_t p = 0; p < oldPointListSize; ++p) {
                if (usedPointMask[p]) {
                    newPointList[index++] = mPoints[p];
                }
            }
            mPoints.reset(newPointList);
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
