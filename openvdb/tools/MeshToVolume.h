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

#ifndef OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/math/FiniteDifference.h>
#include <openvdb/math/Operators.h> // for ISGradientNormSqrd
#include <openvdb/math/Proximity.h> // for closestPointOnTriangleToPoint()
#include <openvdb/tools/LevelSetUtil.h> // for LeafTransformer
#include <openvdb/tools/Morphology.h> // for dilateVoxels()
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/util/Util.h> // for nearestCoord()

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <deque>
#include <limits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////


// Wrapper functions for the MeshToVolume converter


/// @brief Convert a triangle mesh to a level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));


/// @brief Convert a quad mesh to a level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param quads        quad index list
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec4I>& quads,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));


/// @brief Convert a triangle and quad mesh to a level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param quads        quad index list
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));


/// @brief Convert a triangle and quad mesh to a signed distance field
///        with an asymmetrical narrow band.
///
/// @return a grid of type @c GridType containing a narrow-band signed
///         distance field representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param quads        quad index list
/// @param exBandWidth  the exterior narrow-band width in voxel units
/// @param inBandWidth  the interior narrow-band width in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToSignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth);


/// @brief Convert a triangle and quad mesh to an unsigned distance field.
///
/// @return a grid of type @c GridType containing a narrow-band unsigned
///         distance field representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Does not requires a closed surface.
///
/// @param xform        transform for the output grid
/// @param points       list of world space point positions
/// @param triangles    triangle index list
/// @param quads        quad index list
/// @param bandWidth    the width of the narrow band, in voxel units
template<typename GridType>
inline typename GridType::Ptr
meshToUnsignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float bandWidth);


////////////////////////////////////////


/// Conversion flags, used to control the MeshToVolume output
enum { GENERATE_PRIM_INDEX_GRID = 0x1 };


// MeshToVolume
template<typename DistGridT, typename InterruptT = util::NullInterrupter>
class MeshToVolume
{
public:
    typedef typename DistGridT::TreeType DistTreeT;
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename DistTreeT::template ValueConverter<Int32>::Type IndexTreeT;
    typedef Grid<IndexTreeT> IndexGridT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef Grid<StencilTreeT> StencilGridT;

    MeshToVolume(openvdb::math::Transform::Ptr&, int conversionFlags = 0,
        InterruptT *interrupter = NULL, int signSweeps = 1);

    /// @brief  Mesh to Level Set / Signed Distance Field conversion
    ///
    /// @note   Requires a closed surface but not necessarily a manifold surface.
    ///         Supports surfaces with self intersections, degenerate faces and
    ///         is independent of mesh surface normals.
    ///
    /// @param pointList    List of points in grid index space, preferably unique
    ///                     and shared by different polygons.
    /// @param polygonList  List of triangles and/or quads.
    /// @param exBandWidth  The exterior narrow-band width in voxel units.
    /// @param inBandWidth  The interior narrow-band width in voxel units.
    void convertToLevelSet(
        const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList,
        DistValueT exBandWidth = DistValueT(LEVEL_SET_HALF_WIDTH),
        DistValueT inBandWidth = DistValueT(LEVEL_SET_HALF_WIDTH));

    /// @brief Mesh to Unsigned Distance Field conversion
    ///
    /// @note Does not requires a closed surface.
    ///
    /// @param pointList    List of points in grid index space, preferably unique
    ///                     and shared by different polygons.
    /// @param polygonList  List of triangles and/or quads.
    /// @param exBandWidth  The narrow-band width in voxel units.
    void convertToUnsignedDistanceField(const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList, DistValueT exBandWidth);

    void clear();

    /// Returns a narrow-band (signed) distance field / level set grid.
    typename DistGridT::Ptr distGridPtr() const { return mDistGrid; }

    /// Returns a grid containing the closest-primitive index for each
    /// voxel in the narrow-band.
    typename IndexGridT::Ptr indexGridPtr() const { return mIndexGrid; }

private:
    // disallow copy by assignment
    void operator=(const MeshToVolume<DistGridT, InterruptT>&) {}

    void doConvert(const std::vector<Vec3s>&, const std::vector<Vec4I>&,
        DistValueT exBandWidth, DistValueT inBandWidth, bool unsignedDistField = false);

    openvdb::math::Transform::Ptr mTransform;
    int mConversionFlags, mSignSweeps;

    typename DistGridT::Ptr     mDistGrid;
    typename IndexGridT::Ptr    mIndexGrid;
    typename StencilGridT::Ptr  mIntersectingVoxelsGrid;

    InterruptT *mInterrupter;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


// Internal utility objects and implementation details

namespace internal {


class PointTransform
{
public:
    PointTransform(const std::vector<Vec3s>& pointsIn, std::vector<Vec3s>& pointsOut,
        const math::Transform& xform)
        : mPointsIn(pointsIn)
        , mPointsOut(&pointsOut)
        , mXform(xform)
    {
    }

    void runParallel()
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mPointsOut->size()), *this);
    }

    void runSerial()
    {
        (*this)(tbb::blocked_range<size_t>(0, mPointsOut->size()));
    }

    inline void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(); n < range.end(); ++n) {
            (*mPointsOut)[n] = mXform.worldToIndex(mPointsIn[n]);
        }
    }

private:
    const std::vector<Vec3s>& mPointsIn;
    std::vector<Vec3s> * const mPointsOut;
    const math::Transform& mXform;
};


template<typename ValueType>
struct Tolerance
{
    static ValueType epsilon() { return ValueType(1e-7); }
    static ValueType minNarrowBandWidth() { return ValueType(1.0 + 1e-6); }
};


template<typename DistTreeT, typename IndexTreeT>
inline void
combine(DistTreeT& lhsDist, IndexTreeT& lhsIndex, DistTreeT& rhsDist, IndexTreeT& rhsIndex)
{
    typedef typename DistTreeT::ValueType DistValueT;
    typename tree::ValueAccessor<DistTreeT> lhsDistAccessor(lhsDist);
    typename tree::ValueAccessor<IndexTreeT> lhsIndexAccessor(lhsIndex);
    typename tree::ValueAccessor<IndexTreeT> rhsIndexAccessor(rhsIndex);
    typename DistTreeT::LeafCIter iter = rhsDist.cbeginLeaf();

    DistValueT rhsValue;
    Coord ijk;

    for ( ; iter; ++iter) {
        typename DistTreeT::LeafNodeType::ValueOnCIter it = iter->cbeginValueOn();

        for ( ; it; ++it) {

            ijk = it.getCoord();
            rhsValue = it.getValue();
            DistValueT& lhsValue = const_cast<DistValueT&>(lhsDistAccessor.getValue(ijk));

            if (-rhsValue < std::abs(lhsValue)) {
                lhsValue = rhsValue;
                lhsIndexAccessor.setValue(ijk, rhsIndexAccessor.getValue(ijk));
            }
        }
    }
}


////////////////////////////////////////


/// MeshVoxelizer
/// @brief TBB body object to voxelize a mesh of triangles and/or quads into a collection
/// of VDB grids, namely a squared distance grid, a closest primitive grid and an
/// intersecting voxels grid (the voxels intersect the mesh)
/// @note Only the leaf nodes that intersect the mesh are allocated, and only voxels in
/// a narrow band (of two to three voxels in proximity to the mesh's surface) are activated.
/// They are populated with distance values and primitive indices.
template<typename DistTreeT, typename InterruptT = util::NullInterrupter>
class MeshVoxelizer
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef typename DistTreeT::template ValueConverter<Int32>::Type IndexTreeT;
    typedef typename tree::ValueAccessor<IndexTreeT> IndexAccessorT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<StencilTreeT> StencilAccessorT;

    MeshVoxelizer(const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList, InterruptT *interrupter = NULL);

    ~MeshVoxelizer() {}

    void runParallel();
    void runSerial();

    MeshVoxelizer(MeshVoxelizer<DistTreeT, InterruptT>& rhs, tbb::split);
    void operator() (const tbb::blocked_range<size_t> &range);
    void join(MeshVoxelizer<DistTreeT, InterruptT>& rhs);

    DistTreeT& sqrDistTree() { return mSqrDistTree; }
    IndexTreeT& primIndexTree() { return mPrimIndexTree; }
    StencilTreeT& intersectionTree() { return mIntersectionTree; }

private:
    // disallow copy by assignment
    void operator=(const MeshVoxelizer<DistTreeT, InterruptT>&) {}

    inline bool shortEdge(const Vec3d&, const Vec3d&, const Vec3d&) const;
    bool evalVoxel(const Coord& ijk, const Int32 polyIdx);

    std::vector<Vec3s> const * const mPointList;
    std::vector<Vec4I> const * const mPolygonList;

    DistTreeT mSqrDistTree;
    DistAccessorT mSqrDistAccessor;

    IndexTreeT mPrimIndexTree;
    IndexAccessorT mPrimIndexAccessor;

    StencilTreeT mIntersectionTree;
    StencilAccessorT mIntersectionAccessor;

    // Used internally for acceleration
    IndexTreeT mLastPrimTree;
    IndexAccessorT mLastPrimAccessor;

    InterruptT *mInterrupter;
};


template<typename DistTreeT, typename InterruptT>
void
MeshVoxelizer<DistTreeT, InterruptT>::runParallel()
{
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mPolygonList->size()), *this);
}

template<typename DistTreeT, typename InterruptT>
void
MeshVoxelizer<DistTreeT, InterruptT>::runSerial()
{
    (*this)(tbb::blocked_range<size_t>(0, mPolygonList->size()));
}

template<typename DistTreeT, typename InterruptT>
MeshVoxelizer<DistTreeT, InterruptT>::MeshVoxelizer(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    InterruptT *interrupter)
    : mPointList(&pointList)
    , mPolygonList(&polygonList)
    , mSqrDistTree(std::numeric_limits<DistValueT>::max())
    , mSqrDistAccessor(mSqrDistTree)
    , mPrimIndexTree(Int32(util::INVALID_IDX))
    , mPrimIndexAccessor(mPrimIndexTree)
    , mIntersectionTree(false)
    , mIntersectionAccessor(mIntersectionTree)
    , mLastPrimTree(Int32(util::INVALID_IDX))
    , mLastPrimAccessor(mLastPrimTree)
    , mInterrupter(interrupter)
{
}

template<typename DistTreeT, typename InterruptT>
MeshVoxelizer<DistTreeT, InterruptT>::MeshVoxelizer(
    MeshVoxelizer<DistTreeT, InterruptT>& rhs, tbb::split)
    : mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mSqrDistTree(std::numeric_limits<DistValueT>::max())
    , mSqrDistAccessor(mSqrDistTree)
    , mPrimIndexTree(Int32(util::INVALID_IDX))
    , mPrimIndexAccessor(mPrimIndexTree)
    , mIntersectionTree(false)
    , mIntersectionAccessor(mIntersectionTree)
    , mLastPrimTree(Int32(util::INVALID_IDX))
    , mLastPrimAccessor(mLastPrimTree)
    , mInterrupter(rhs.mInterrupter)
{
}

template<typename DistTreeT, typename InterruptT>
inline bool
MeshVoxelizer<DistTreeT, InterruptT>::shortEdge(
    const Vec3d& v0, const Vec3d& v1, const Vec3d& v2) const
{
    double edge_max = std::abs(v1[0] - v0[0]);
    edge_max = std::max(edge_max, std::abs(v1[1] - v0[1]));
    edge_max = std::max(edge_max, std::abs(v1[2] - v0[2]));
    edge_max = std::max(edge_max, std::abs(v0[0] - v2[0]));
    edge_max = std::max(edge_max, std::abs(v0[1] - v2[1]));
    edge_max = std::max(edge_max, std::abs(v0[2] - v2[2]));
    return edge_max < 200.0;
}

template<typename DistTreeT, typename InterruptT>
void
MeshVoxelizer<DistTreeT, InterruptT>::operator()(const tbb::blocked_range<size_t> &range)
{
    std::deque<Coord> coordList;
    StencilTreeT auxTree(false);
    StencilAccessorT auxAcc(auxTree);
    Coord ijk, n_ijk;

    for (size_t n = range.begin(); n < range.end(); ++n) {

        if (mInterrupter && mInterrupter->wasInterrupted()) {
            tbb::task::self().cancel_group_execution();
            break;
        }

        const Int32 primIdx = n;
        const Vec4I verts = (*mPolygonList)[n];

        Vec3d p0((*mPointList)[verts[0]]);
        Vec3d p1((*mPointList)[verts[1]]);
        Vec3d p2((*mPointList)[verts[2]]);

        if (shortEdge(p0, p1, p2)) {
            coordList.clear();


            ijk = util::nearestCoord(p0);
            evalVoxel(ijk, primIdx);
            coordList.push_back(ijk);


            ijk = util::nearestCoord(p1);
            evalVoxel(ijk, primIdx);
            coordList.push_back(ijk);


            ijk = util::nearestCoord(p2);
            evalVoxel(ijk, primIdx);
            coordList.push_back(ijk);

            if (util::INVALID_IDX != verts[3]) {
                Vec3d p3((*mPointList)[verts[3]]);
                ijk = util::nearestCoord(p3);
                evalVoxel(ijk, primIdx);
                coordList.push_back(ijk);
            }

            while (!coordList.empty()) {
                if (mInterrupter && mInterrupter->wasInterrupted()) {
                    break;
                }

                ijk = coordList.back();
                coordList.pop_back();

                mIntersectionAccessor.setActiveState(ijk, true);

                for (Int32 i = 0; i < 26; ++i) {
                    n_ijk = ijk + util::COORD_OFFSETS[i];

                    if (primIdx != mLastPrimAccessor.getValue(n_ijk)) {
                        mLastPrimAccessor.setValue(n_ijk, n);
                        if(evalVoxel(n_ijk, n)) coordList.push_back(n_ijk);
                    }
                }
            }

        } else {

            ijk = util::nearestCoord(p0);
            evalVoxel(ijk, primIdx);

            mLastPrimAccessor.setValue(ijk, primIdx);
            auxAcc.setActiveState(ijk, true);

            while (!auxTree.empty()) {

                if (mInterrupter && mInterrupter->wasInterrupted()) {
                    break;
                }

                typename StencilTreeT::LeafIter leafIter = auxTree.beginLeaf();
                for (; leafIter; leafIter.next()) {

                    if (mInterrupter && mInterrupter->wasInterrupted()) {
                        break;
                    }

                    typename StencilTreeT::LeafNodeType::ValueOnIter iter =
                        leafIter->beginValueOn();
                    for (; iter; iter.next()) {
                        ijk = iter.getCoord();
                        iter.setValueOff();

                        mIntersectionAccessor.setActiveState(ijk, true);

                        for (Int32 i = 0; i < 26; ++i) {
                            n_ijk = ijk + util::COORD_OFFSETS[i];

                            if (primIdx != mLastPrimAccessor.getValue(n_ijk)) {
                                mLastPrimAccessor.setValue(n_ijk, n);
                                if (evalVoxel(n_ijk, n)) auxAcc.setActiveState(n_ijk, true);
                            }
                        }
                    }
                }

                auxTree.pruneInactive();
            }
        }
    }
}

template<typename DistTreeT, typename InterruptT>
bool
MeshVoxelizer<DistTreeT, InterruptT>::evalVoxel(const Coord& ijk, const Int32 polyIdx)
{
    Vec3d voxelCenter(ijk[0], ijk[1], ijk[2]);
    const Vec4I& verts = (*mPolygonList)[polyIdx];
    const std::vector<Vec3s>& points = *mPointList;

    // Evaluate first triangle
    const Vec3d a(points[verts[0]]);
    const Vec3d b(points[verts[1]]);
    const Vec3d c(points[verts[2]]);

    Vec3d uvw;
    double dist = (voxelCenter -
        closestPointOnTriangleToPoint(a, c, b, voxelCenter, uvw)).lengthSqr();

    // Split-up quad into a second triangle and calac distance.
    if (util::INVALID_IDX != verts[3]) {
        const Vec3d d(points[verts[3]]);

        double secondDist = (voxelCenter -
            closestPointOnTriangleToPoint(a, d, c, voxelCenter, uvw)).lengthSqr();

        if (secondDist < dist) dist = secondDist;
    }

    const DistValueT tmp(dist);
    if (tmp < std::abs(mSqrDistAccessor.getValue(ijk))) {
        mSqrDistAccessor.setValue(ijk, -tmp);
        mPrimIndexAccessor.setValue(ijk, polyIdx);
    }

    return (dist < 0.86602540378443861);
}

template<typename DistTreeT, typename InterruptT>
void
MeshVoxelizer<DistTreeT, InterruptT>::join(MeshVoxelizer<DistTreeT, InterruptT>& rhs)
{
    typename DistTreeT::LeafCIter iter = rhs.mSqrDistTree.cbeginLeaf();
    DistValueT rhsDist;
    Coord ijk;

    for ( ; iter; ++iter) {
        typename DistTreeT::LeafNodeType::ValueOnCIter it = iter->cbeginValueOn();

        for ( ; it; ++it) {

            ijk = it.getCoord();
            rhsDist = it.getValue();
            DistValueT lhsDist = mSqrDistAccessor.getValue(ijk);

            if (-rhsDist < std::abs(lhsDist)) {
                mSqrDistAccessor.setValue(ijk, rhsDist);
                mPrimIndexAccessor.setValue(ijk, rhs.mPrimIndexAccessor.getValue(ijk));
            }
        }
    }

    mIntersectionTree.merge(rhs.mIntersectionTree);
}


////////////////////////////////////////


// ContourTracer
/// @brief TBB body object that partitions a volume into 2D slices that can be processed
/// in parallel and marks the exterior contour of disjoint voxel sets in each slice
template<typename DistTreeT, typename InterruptT = util::NullInterrupter>
class ContourTracer
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<const StencilTreeT> StencilAccessorT;

    ContourTracer(DistTreeT&, const StencilTreeT&, InterruptT *interrupter = NULL);
    ~ContourTracer() {}

    void runParallel();
    void runSerial();

    ContourTracer(const ContourTracer<DistTreeT, InterruptT>& rhs);
    void operator()(const tbb::blocked_range<int> &range) const;

private:
    void operator=(const ContourTracer<DistTreeT, InterruptT>&) {}

    int sparseScan(int slice) const;

    DistTreeT& mDistTree;
    DistAccessorT mDistAccessor;

    const StencilTreeT& mIntersectionTree;
    StencilAccessorT mIntersectionAccessor;

    CoordBBox mBBox;

    /// List of value-depth dependant step sizes.
    std::vector<Index> mStepSize;

    InterruptT *mInterrupter;
};

template<typename DistTreeT, typename InterruptT>
void
ContourTracer<DistTreeT, InterruptT>::runParallel()
{
    tbb::parallel_for(tbb::blocked_range<int>(mBBox.min()[0], mBBox.max()[0]+1), *this);
}

template<typename DistTreeT, typename InterruptT>
void
ContourTracer<DistTreeT, InterruptT>::runSerial()
{
    (*this)(tbb::blocked_range<int>(mBBox.min()[0], mBBox.max()[0]+1));
}

template<typename DistTreeT, typename InterruptT>
ContourTracer<DistTreeT, InterruptT>::ContourTracer(
    DistTreeT& distTree, const StencilTreeT& intersectionTree, InterruptT *interrupter)
    : mDistTree(distTree)
    , mDistAccessor(mDistTree)
    , mIntersectionTree(intersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
    , mBBox(CoordBBox())
    , mStepSize(0)
    , mInterrupter(interrupter)
{
    // Build the step size table for different tree value depths.
    std::vector<Index> dims;
    mDistTree.getNodeLog2Dims(dims);

    mStepSize.resize(dims.size()+1, 1);
    Index exponent = 0;
    for (int idx = static_cast<int>(dims.size()) - 1; idx > -1; --idx) {
        exponent += dims[idx];
        mStepSize[idx] = 1 << exponent;
    }

    mDistTree.evalLeafBoundingBox(mBBox);

    // Make sure that mBBox coincides with the min and max corners of the internal nodes.
    const int tileDim = mStepSize[0];

    for (size_t i = 0; i < 3; ++i) {

        int n;
        double diff = std::abs(double(mBBox.min()[i])) / double(tileDim);

        if (mBBox.min()[i] <= tileDim) {
            n = int(std::ceil(diff));
            mBBox.min()[i] = - n * tileDim;
        } else {
            n = int(std::floor(diff));
            mBBox.min()[i] = n * tileDim;
        }

        n = int(std::ceil(std::abs(double(mBBox.max()[i] - mBBox.min()[i])) / double(tileDim)));
        mBBox.max()[i] =  mBBox.min()[i] + n * tileDim;
    }
}

template<typename DistTreeT, typename InterruptT>
ContourTracer<DistTreeT, InterruptT>::ContourTracer(
    const ContourTracer<DistTreeT, InterruptT> &rhs)
    : mDistTree(rhs.mDistTree)
    , mDistAccessor(mDistTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
    , mBBox(rhs.mBBox)
    , mStepSize(rhs.mStepSize)
    , mInterrupter(rhs.mInterrupter)
{
}

template<typename DistTreeT, typename InterruptT>
void
ContourTracer<DistTreeT, InterruptT>::operator()(const tbb::blocked_range<int> &range) const
{
    // Slice up the volume and trace contours.
    int iStep = 1;
    for (int n = range.begin(); n < range.end(); n += iStep) {

        if (mInterrupter && mInterrupter->wasInterrupted()) {
            tbb::task::self().cancel_group_execution();
            break;
        }

        iStep = sparseScan(n);
    }
}

template<typename DistTreeT, typename InterruptT>
int
ContourTracer<DistTreeT, InterruptT>::sparseScan(int slice) const
{
    bool lastVoxelWasOut = true;
    int last_k;

    Coord ijk(slice, mBBox.min()[1], mBBox.min()[2]);
    Coord step(mStepSize[mDistAccessor.getValueDepth(ijk) + 1]);
    Coord n_ijk;

    for (ijk[1] = mBBox.min()[1]; ijk[1] <= mBBox.max()[1]; ijk[1] += step[1]) { // j

        if (mInterrupter && mInterrupter->wasInterrupted()) {
            break;
        }

        step[1] = mStepSize[mDistAccessor.getValueDepth(ijk) + 1];
        step[0] = std::min(step[0], step[1]);

        for (ijk[2] = mBBox.min()[2]; ijk[2] <= mBBox.max()[2]; ijk[2] += step[2]) { // k

            step[2] = mStepSize[mDistAccessor.getValueDepth(ijk) + 1];
            step[1] = std::min(step[1], step[2]);
            step[0] = std::min(step[0], step[2]);

            // If the current voxel is set?
            if (mDistAccessor.isValueOn(ijk)) {

                // Is this a boundary voxel?
                if (mIntersectionAccessor.isValueOn(ijk)) {

                    lastVoxelWasOut = false;
                    last_k = ijk[2];

                } else if (lastVoxelWasOut) {

                    DistValueT& val = const_cast<DistValueT&>(mDistAccessor.getValue(ijk));
                    val = -val; // flip sign

                } else {

                    DistValueT val;
                    for (Int32 n = 3; n < 6; n += 2) {
                        n_ijk = ijk + util::COORD_OFFSETS[n];

                        if (mDistAccessor.probeValue(n_ijk, val) && val > 0) {
                            lastVoxelWasOut = true;
                            break;
                        }
                    }

                    if (lastVoxelWasOut) {

                        DistValueT& v = const_cast<DistValueT&>(mDistAccessor.getValue(ijk));
                        v = -v; // flip sign

                        const int tmp_k = ijk[2];

                        // backtrace
                        for (--ijk[2]; ijk[2] >= last_k; --ijk[2]) {
                            if (mIntersectionAccessor.isValueOn(ijk)) break;
                            DistValueT& vb = const_cast<DistValueT&>(mDistAccessor.getValue(ijk));
                            if(vb < DistValueT(0.0)) vb = -vb; // flip sign
                        }

                        last_k = tmp_k;
                        ijk[2] = tmp_k;

                    } else {
                        last_k = std::min(ijk[2], last_k);
                    }

                }

            } // end isValueOn check
        } // end k
    } // end j
    return step[0];
}


////////////////////////////////////////


// IntersectingVoxelSign
/// @brief TBB body object that traversers all intersecting voxels (defined by the
/// intersectingVoxelsGrid) and potentially flips their sign, by comparing the "closest point"
/// directions of outside-marked and non-intersecting neighboring voxels
template<typename DistTreeT>
class IntersectingVoxelSign
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef typename DistTreeT::template ValueConverter<Int32>::Type IndexTreeT;
    typedef typename tree::ValueAccessor<IndexTreeT> IndexAccessorT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<StencilTreeT> StencilAccessorT;
    typedef tree::LeafManager<StencilTreeT> StencilArrayT;

    IntersectingVoxelSign(
        const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList,
        DistTreeT& distTree,
        IndexTreeT& indexTree,
        StencilTreeT& intersectionTree,
        StencilArrayT& leafs);

    ~IntersectingVoxelSign() {}

    void runParallel();
    void runSerial();

    IntersectingVoxelSign(const IntersectingVoxelSign<DistTreeT> &rhs);
    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    void operator=(const IntersectingVoxelSign<DistTreeT>&) {}

    void evalVoxel(const Coord& ijk) const;
    Vec3d getClosestPointDir(const Coord& ijk) const;

    std::vector<Vec3s> const * const mPointList;
    std::vector<Vec4I> const * const mPolygonList;

    DistTreeT& mDistTree;
    DistAccessorT mDistAccessor;

    IndexTreeT& mIndexTree;
    IndexAccessorT mIndexAccessor;

    StencilTreeT& mIntersectionTree;
    StencilAccessorT mIntersectionAccessor;
    StencilArrayT& mLeafs;
};

template<typename DistTreeT>
void
IntersectingVoxelSign<DistTreeT>::runParallel()
{
    tbb::parallel_for(mLeafs.getRange(), *this);
}

template<typename DistTreeT>
void
IntersectingVoxelSign<DistTreeT>::runSerial()
{
    (*this)(mLeafs.getRange());
}

template<typename DistTreeT>
IntersectingVoxelSign<DistTreeT>::IntersectingVoxelSign(
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList,
    DistTreeT& distTree,
    IndexTreeT& indexTree,
    StencilTreeT& intersectionTree,
    StencilArrayT& leafs)
    : mPointList(&pointList)
    , mPolygonList(&polygonList)
    , mDistTree(distTree)
    , mDistAccessor(mDistTree)
    , mIndexTree(indexTree)
    , mIndexAccessor(mIndexTree)
    , mIntersectionTree(intersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
    , mLeafs(leafs)
{
}

template<typename DistTreeT>
IntersectingVoxelSign<DistTreeT>::IntersectingVoxelSign(
    const IntersectingVoxelSign<DistTreeT> &rhs)
    : mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mDistTree(rhs.mDistTree)
    , mDistAccessor(mDistTree)
    , mIndexTree(rhs.mIndexTree)
    , mIndexAccessor(mIndexTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
    , mLeafs(rhs.mLeafs)
{
}

template<typename DistTreeT>
void
IntersectingVoxelSign<DistTreeT>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    typename StencilTreeT::LeafNodeType::ValueOnCIter iter;

    for (size_t n = range.begin(); n < range.end(); ++n) {
        iter = mLeafs.leaf(n).cbeginValueOn();
        for (; iter; ++iter) {
            evalVoxel(iter.getCoord());
        }
    }
}

template<typename DistTreeT>
void
IntersectingVoxelSign<DistTreeT>::evalVoxel(const Coord& ijk) const
{
    const DistValueT val = mDistAccessor.getValue(ijk), zeroVal(0.0);

    if(!(val < zeroVal)) return;

    Vec3d dir = getClosestPointDir(ijk), n_dir;
    DistValueT n_val;
    Coord n_ijk;

    // Check voxel-face adjacent neighbors.
    for (Int32 n = 0; n < 26; ++n) {
        n_ijk = ijk + util::COORD_OFFSETS[n];

        if (mIntersectionAccessor.isValueOn(n_ijk)) continue;
        if (!mDistAccessor.probeValue(n_ijk, n_val)) continue;
        if (n_val < zeroVal) continue;

        n_dir = getClosestPointDir(n_ijk);

        if (n_dir.dot(dir) > 0.0 ) {
            const_cast<IntersectingVoxelSign<DistTreeT> *>(this)->
                mDistAccessor.setValue(ijk, -val);
            break;
        }
    }
}

template<typename DistTreeT>
Vec3d
IntersectingVoxelSign<DistTreeT>::getClosestPointDir(const Coord& ijk) const
{
    Vec3d voxelCenter(ijk[0], ijk[1], ijk[2]);
    const Vec4I& prim = (*mPolygonList)[mIndexAccessor.getValue(ijk)];
    const std::vector<Vec3s>& points = *mPointList;

    // Evaluate first triangle
    const Vec3d a(points[prim[0]]);
    const Vec3d b(points[prim[1]]);
    const Vec3d c(points[prim[2]]);

    Vec3d uvw;
    Vec3d cpt = closestPointOnTriangleToPoint(a, c, b, voxelCenter, uvw);
    Vec3d diff = voxelCenter - cpt;

    // Evaluate second triangle if quad.
    if (prim[3] != util::INVALID_IDX) {
        const Vec3d d(points[prim[3]]);

        cpt = closestPointOnTriangleToPoint(a, d, c, voxelCenter, uvw);
        Vec3d tmpDiff = voxelCenter - cpt;

        if (tmpDiff.lengthSqr() < diff.lengthSqr()) {
            diff = tmpDiff;
        }
    }

    diff.normalize();
    return diff;
}


////////////////////////////////////////


// IntersectingVoxelCleaner
/// @brief TBB body object that removes intersecting voxels that were set via
/// voxelization of self-intersecting parts of a mesh
template<typename DistTreeT>
class IntersectingVoxelCleaner
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef typename DistTreeT::template ValueConverter<Int32>::Type IndexTreeT;
    typedef typename tree::ValueAccessor<IndexTreeT> IndexAccessorT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<StencilTreeT> StencilAccessorT;
    typedef tree::LeafManager<StencilTreeT> StencilArrayT;

    IntersectingVoxelCleaner(DistTreeT& distTree, IndexTreeT& indexTree,
        StencilTreeT& intersectionTree, StencilArrayT& leafs);

    ~IntersectingVoxelCleaner() {}

    void runParallel();
    void runSerial();

    IntersectingVoxelCleaner(const IntersectingVoxelCleaner<DistTreeT> &rhs);
    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    void operator=(const IntersectingVoxelCleaner<DistTreeT>&) {}

    DistTreeT& mDistTree;
    DistAccessorT mDistAccessor;

    IndexTreeT& mIndexTree;
    IndexAccessorT mIndexAccessor;

    StencilTreeT& mIntersectionTree;
    StencilAccessorT mIntersectionAccessor;
    StencilArrayT& mLeafs;
};

template<typename DistTreeT>
void
IntersectingVoxelCleaner<DistTreeT>::runParallel()
{
    tbb::parallel_for(mLeafs.getRange(), *this);
    mIntersectionTree.pruneInactive();
}

template<typename DistTreeT>
void
IntersectingVoxelCleaner<DistTreeT>::runSerial()
{
    (*this)(mLeafs.getRange());
    mIntersectionTree.pruneInactive();
}

template<typename DistTreeT>
IntersectingVoxelCleaner<DistTreeT>::IntersectingVoxelCleaner(
    DistTreeT& distTree,
    IndexTreeT& indexTree,
    StencilTreeT& intersectionTree,
    StencilArrayT& leafs)
    : mDistTree(distTree)
    , mDistAccessor(mDistTree)
    , mIndexTree(indexTree)
    , mIndexAccessor(mIndexTree)
    , mIntersectionTree(intersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
    , mLeafs(leafs)
{
}

template<typename DistTreeT>
IntersectingVoxelCleaner<DistTreeT>::IntersectingVoxelCleaner(
    const IntersectingVoxelCleaner<DistTreeT>& rhs)
    : mDistTree(rhs.mDistTree)
    , mDistAccessor(mDistTree)
    , mIndexTree(rhs.mIndexTree)
    , mIndexAccessor(mIndexTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
    , mLeafs(rhs.mLeafs)
{
}

template<typename DistTreeT>
void
IntersectingVoxelCleaner<DistTreeT>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    Coord ijk, m_ijk;
    bool turnOff;
    DistValueT value, bg = mDistTree.background();

    typename StencilTreeT::LeafNodeType::ValueOnCIter iter;

    for (size_t n = range.begin(); n < range.end(); ++n) {

        typename StencilTreeT::LeafNodeType& leaf = mLeafs.leaf(n);
        iter = leaf.cbeginValueOn();

        for (; iter; ++iter) {

            ijk = iter.getCoord();

            turnOff = true;
            for (Int32 m = 0; m < 26; ++m) {
                m_ijk = ijk + util::COORD_OFFSETS[m];
                if (mDistAccessor.probeValue(m_ijk, value)) {
                    if (value > 0.0) {
                        turnOff = false;
                        break;
                    }
                }
            }

            if (turnOff) leaf.setValueOff(ijk, bg);
        }
    }
}


////////////////////////////////////////


// ShellVoxelCleaner
/// @brief TBB body object that removes non-intersecting voxels that where set by rasterizing
/// self-intersecting parts of the mesh.
template<typename DistTreeT>
class ShellVoxelCleaner
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef tree::LeafManager<DistTreeT> DistArrayT;
    typedef typename DistTreeT::template ValueConverter<Int32>::Type IndexTreeT;
    typedef typename tree::ValueAccessor<IndexTreeT> IndexAccessorT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<StencilTreeT> StencilAccessorT;

    ShellVoxelCleaner(DistTreeT& distTree, DistArrayT& leafs, IndexTreeT& indexTree,
        StencilTreeT& intersectionTree);

    ~ShellVoxelCleaner() {}

    void runParallel();
    void runSerial();

    ShellVoxelCleaner(const ShellVoxelCleaner<DistTreeT> &rhs);
    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    void operator=(const ShellVoxelCleaner<DistTreeT>&) {}

    DistTreeT& mDistTree;
    DistArrayT& mLeafs;
    DistAccessorT mDistAccessor;

    IndexTreeT& mIndexTree;
    IndexAccessorT mIndexAccessor;

    StencilTreeT& mIntersectionTree;
    StencilAccessorT mIntersectionAccessor;
};

template<typename DistTreeT>
void
ShellVoxelCleaner<DistTreeT>::runParallel()
{
    tbb::parallel_for(mLeafs.getRange(), *this);
    mDistTree.pruneInactive();
    mIndexTree.pruneInactive();
}

template<typename DistTreeT>
void
ShellVoxelCleaner<DistTreeT>::runSerial()
{
    (*this)(mLeafs.getRange());
    mDistTree.pruneInactive();
    mIndexTree.pruneInactive();
}

template<typename DistTreeT>
ShellVoxelCleaner<DistTreeT>::ShellVoxelCleaner(
    DistTreeT& distTree,
    DistArrayT& leafs,
    IndexTreeT& indexTree,
    StencilTreeT& intersectionTree)
    : mDistTree(distTree)
    , mLeafs(leafs)
    , mDistAccessor(mDistTree)
    , mIndexTree(indexTree)
    , mIndexAccessor(mIndexTree)
    , mIntersectionTree(intersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
{
}

template<typename DistTreeT>
ShellVoxelCleaner<DistTreeT>::ShellVoxelCleaner(
    const ShellVoxelCleaner<DistTreeT> &rhs)
    : mDistTree(rhs.mDistTree)
    , mLeafs(rhs.mLeafs)
    , mDistAccessor(mDistTree)
    , mIndexTree(rhs.mIndexTree)
    , mIndexAccessor(mIndexTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
{
}

template<typename DistTreeT>
void
ShellVoxelCleaner<DistTreeT>::operator()(
    const tbb::blocked_range<size_t>& range) const
{

    Coord ijk, m_ijk;
    bool turnOff;
    DistValueT value;

    const DistValueT distC = -0.86602540378443861;
    const DistValueT distBG = mDistTree.background();
    const Int32 indexBG = mIntersectionTree.background();

    typename DistTreeT::LeafNodeType::ValueOnCIter iter;
    for (size_t n = range.begin(); n < range.end(); ++n) {

        typename DistTreeT::LeafNodeType& leaf = mLeafs.leaf(n);
        iter = leaf.cbeginValueOn();
        for (; iter; ++iter) {

            value = iter.getValue();
            if(value > 0.0) continue;

            ijk = iter.getCoord();
            if (mIntersectionAccessor.isValueOn(ijk)) continue;

            turnOff = true;
            for (Int32 m = 0; m < 18; ++m) {
                m_ijk = ijk + util::COORD_OFFSETS[m];

                if (mIntersectionAccessor.isValueOn(m_ijk)) {
                    turnOff = false;
                    break;
                }
            }

            if (turnOff) {
                leaf.setValueOff(ijk, distBG);

                const_cast<ShellVoxelCleaner<DistTreeT> *>(this)->
                    mIndexAccessor.setValueOff(ijk, indexBG);

            } else {
                if (value > distC) leaf.setValue(ijk, distC);
            }
        }
    }
}


////////////////////////////////////////


// ExpandNB
/// @brief TBB body object to expand the level set narrow band
/// @note The interior and exterior widths should be in world space units and squared.
template<typename DistTreeT>
class ExpandNB
{
public:
    typedef typename DistTreeT::ValueType DistValueT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef typename DistTreeT::template ValueConverter<Int32>::Type IndexTreeT;
    typedef typename tree::ValueAccessor<IndexTreeT> IndexAccessorT;
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<StencilTreeT> StencilAccessorT;
    typedef tree::LeafManager<StencilTreeT> StencilArrayT;

    ExpandNB(const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
        DistTreeT& distTree, IndexTreeT& indexTree, StencilTreeT& maskTree, StencilArrayT& leafs,
        DistValueT exteriorBandWidth, DistValueT interiorBandWidth, DistValueT voxelSize);

    ExpandNB(const ExpandNB<DistTreeT>& rhs, tbb::split);

    ~ExpandNB() {}

    void runParallel();
    void runSerial();

    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    void operator=(const ExpandNB<DistTreeT>&) {}
    double getDist(const Coord&, DistAccessorT&, IndexAccessorT&, Int32& primIndex) const;
    double getDistToPrim(const Coord& ijk, const Int32 polyIdx) const;

    std::vector<Vec3s> const * const mPointList;
    std::vector<Vec4I> const * const mPolygonList;

    DistTreeT& mDistTree;
    IndexTreeT& mIndexTree;
    StencilTreeT& mMaskTree;
    StencilArrayT& mLeafs;

    const DistValueT mExteriorBandWidth, mInteriorBandWidth, mVoxelSize;
};

template<typename DistTreeT>
void
ExpandNB<DistTreeT>::runParallel()
{
    tbb::parallel_for(mLeafs.getRange(), *this);
    mMaskTree.pruneInactive();
}

template<typename DistTreeT>
void
ExpandNB<DistTreeT>::runSerial()
{
    (*this)(mLeafs.getRange());
    mMaskTree.pruneInactive();
}

template<typename DistTreeT>
ExpandNB<DistTreeT>::ExpandNB(
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList,
    DistTreeT& distTree,
    IndexTreeT& indexTree,
    StencilTreeT& maskTree,
    StencilArrayT& leafs,
    DistValueT exteriorBandWidth, DistValueT interiorBandWidth,
    DistValueT voxelSize)
    : mPointList(&pointList)
    , mPolygonList(&polygonList)
    , mDistTree(distTree)
    , mIndexTree(indexTree)
    , mMaskTree(maskTree)
    , mLeafs(leafs)
    , mExteriorBandWidth(exteriorBandWidth)
    , mInteriorBandWidth(interiorBandWidth)
    , mVoxelSize(voxelSize)
{
}

template<typename DistTreeT>
ExpandNB<DistTreeT>::ExpandNB(const ExpandNB<DistTreeT>& rhs, tbb::split)
    : mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mDistTree(rhs.mDistTree)
    , mIndexTree(rhs.mIndexTree)
    , mMaskTree(rhs.mMaskTree)
    , mLeafs(rhs.mLeafs)
    , mExteriorBandWidth(rhs.mExteriorBandWidth)
    , mInteriorBandWidth(rhs.mInteriorBandWidth)
    , mVoxelSize(rhs.mVoxelSize)
{
}

template<typename DistTreeT>
void
ExpandNB<DistTreeT>::operator()(const tbb::blocked_range<size_t>& range) const
{
    typedef typename DistTreeT::LeafNodeType DistLeafT;
    typedef typename IndexTreeT::LeafNodeType IndexLeafT;
    typedef typename StencilTreeT::LeafNodeType StencilLeafT;

    Coord ijk, n_ijk;
    Int32 closestPrimIndex = 0;

    DistAccessorT distAcc(mDistTree);
    IndexAccessorT indexAcc(mIndexTree);

    for (size_t n = range.begin(); n < range.end(); ++n) {

        StencilLeafT& maskLeaf = mLeafs.leaf(n);

        const Coord origin = maskLeaf.getOrigin();

        DistLeafT* distLeafPt = distAcc.probeLeaf(origin);
        IndexLeafT* indexLeafPt = indexAcc.probeLeaf(origin);

        if (distLeafPt != NULL && indexLeafPt != NULL) {

            DistLeafT& distLeaf = *distLeafPt;
            IndexLeafT& indexLeaf = *indexLeafPt;


            typename StencilLeafT::ValueOnIter iter = maskLeaf.beginValueOn();
            for (; iter; ++iter) {

                const Index pos = iter.pos();

                if (distLeaf.isValueOn(pos)) {
                    iter.setValueOff();
                    continue;
                }

                DistValueT distance = Tolerance<DistValueT>::epsilon() +
                    getDist(iter.getCoord(), distAcc, indexAcc, closestPrimIndex);

                const bool inside = distLeaf.getValue(pos) < DistValueT(0.0);

                if (!inside && distance < mExteriorBandWidth) {
                    distLeaf.setValueOn(pos, distance);
                    indexLeaf.setValueOn(pos, closestPrimIndex);
                } else if (inside && distance < mInteriorBandWidth) {
                    distLeaf.setValueOn(pos, -distance);
                    indexLeaf.setValueOn(pos, closestPrimIndex);
                } else {
                    iter.setValueOff();
                }
            }

        } else {
            maskLeaf.setValuesOff();
        }
    }
}

template<typename DistTreeT>
double
ExpandNB<DistTreeT>::getDist(const Coord& ijk, DistAccessorT& distAcc,
    IndexAccessorT& indexAcc, Int32& primIndex) const
{
    Vec3d voxelCenter(ijk[0], ijk[1], ijk[2]);
    DistValueT nDist, dist = std::numeric_limits<double>::max();

    // Find neighbor with closest face point
    Coord n_ijk;
    for (Int32 n = 0; n < 18; ++n) {
        n_ijk = ijk + util::COORD_OFFSETS[n];
        if (distAcc.probeValue(n_ijk, nDist)) {
            nDist = std::abs(nDist);
            if (nDist < dist) {
                dist = nDist;
                primIndex = indexAcc.getValue(n_ijk);
            }
        }
    }

    // Calc. this voxels distance to the closest primitive.
    return getDistToPrim(ijk, primIndex);
}


template<typename DistTreeT>
double
ExpandNB<DistTreeT>::getDistToPrim(const Coord& ijk, const Int32 polyIdx) const
{
    Vec3d voxelCenter(ijk[0], ijk[1], ijk[2]);
    const Vec4I& verts = (*mPolygonList)[polyIdx];
    const std::vector<Vec3s>& points = *mPointList;

    // Evaluate first triangle
    const Vec3d a(points[verts[0]]);
    const Vec3d b(points[verts[1]]);
    const Vec3d c(points[verts[2]]);

    Vec3d uvw;
    double dist = (voxelCenter -
        closestPointOnTriangleToPoint(a, c, b, voxelCenter, uvw)).lengthSqr();

    // Split-up quad into a second triangle and calac distance.
    if (util::INVALID_IDX != verts[3]) {
        const Vec3d d(points[verts[3]]);

        double secondDist = (voxelCenter -
            closestPointOnTriangleToPoint(a, d, c, voxelCenter, uvw)).lengthSqr();

        if (secondDist < dist) dist = secondDist;
    }

    return std::sqrt(dist) * double(mVoxelSize);
}


////////////////////////////////////////


// Helper methods

/// @brief Surface tracing method that flips the sign of interior marked voxels
/// and will not cross the boundary defined by the intersecting voxels
///
/// @param seed             the coordinates of a marked interior seed point
/// @param distTree         the distance field to operate on
/// @param intersectionTree tree that defines the surface boundary
template<typename DistTreeT>
inline void
surfaceTracer(const Coord &seed, DistTreeT& distTree,
    typename DistTreeT::template ValueConverter<bool>::Type& intersectionTree)
{
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<StencilTreeT> StencilAccessorT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef typename DistTreeT::ValueType DistValueT;

    StencilAccessorT intrAccessor(intersectionTree);
    DistAccessorT distAccessor(distTree);

    std::deque<Coord> coordList;
    coordList.push_back(seed);
    Coord ijk, n_ijk;

    while (!coordList.empty()) {
        ijk = coordList.back();
        coordList.pop_back();

        if (!distAccessor.isValueOn(ijk)) continue;

        DistValueT& dist = const_cast<DistValueT&>(distAccessor.getValue(ijk));
        if (!(dist < 0.0)) continue;
        dist = -dist; // flip sign

        for (int n = 0; n < 6; ++n) {
            n_ijk = ijk + util::COORD_OFFSETS[n];

            if (!intrAccessor.isValueOn(n_ijk)) {    // Don't cross the interface
                if (distAccessor.isValueOn(n_ijk)) {                 // Is part of the narrow band
                    if (distAccessor.getValue(n_ijk) < 0.0) {        // Marked as outside.
                        coordList.push_back(n_ijk);
                    }
                }
            }

        } // END neighbor voxel loop.
    } // END coordList loop.
}


/// @brief Iterate sparsely over a distance grid to find regions with inconsistent sign
/// information and use internal::surfaceTracer to resolve those inconsistencies.
///
/// @param distTree         signed distance field to operate on
/// @param intersectionTree tree that defines the surface boundary for the surface tracer
/// @param interrupter      an object that implements the util::NullInterrupter interface
template<typename DistTreeT, typename InterruptT>
inline void
propagateSign(DistTreeT& distTree,
    typename DistTreeT::template ValueConverter<bool>::Type& intersectionTree,
    InterruptT *interrupter = NULL)
{
    typedef typename DistTreeT::template ValueConverter<bool>::Type StencilTreeT;
    typedef typename tree::ValueAccessor<StencilTreeT> StencilAccessorT;
    typedef typename tree::ValueAccessor<DistTreeT> DistAccessorT;
    typedef typename DistTreeT::ValueType DistValueT;

    StencilAccessorT intrAccessor(intersectionTree);
    DistAccessorT distAccessor(distTree);
    Coord ijk, n_ijk;

    typename DistTreeT::LeafIter leafIter = distTree.beginLeaf();
    for (; leafIter; leafIter.next()) {

        if (interrupter && interrupter->wasInterrupted()) break;

        typename DistTreeT::LeafNodeType::ValueOnIter iter = leafIter->beginValueOn();
        for (; iter; iter.next()) {

            ijk = iter.getCoord();

            // Ignore intersecting voxels.
            if (intrAccessor.isValueOn(ijk)) continue;

            if (iter.getValue() < 0.0) {
                for (Int32 n = 0; n < 6; ++n) {
                    n_ijk = ijk + util::COORD_OFFSETS[n];

                    if (distAccessor.isValueOn(n_ijk) && distAccessor.getValue(n_ijk) > 0.0) {
                        surfaceTracer(ijk, distTree, intersectionTree);
                        break;
                    }
                }
            }

        } // END voxel iteration
    } // END leaf iteration
}


template<typename ValueType>
struct SqrtAndScaleOp
{
    SqrtAndScaleOp(ValueType voxelSize, bool unsignedDist = false)
        : mVoxelSize(voxelSize)
        , mUnsigned(unsignedDist)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t/*leafIndex*/) const
    {
        ValueType w[2];
        w[0] = mVoxelSize;
        w[1] = -mVoxelSize;

        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();
        for (; iter; ++iter) {
            ValueType& val = const_cast<ValueType&>(iter.getValue());
            val = w[!mUnsigned && int(val < ValueType(0.0))] * std::sqrt(std::abs(val));
        }
    }

private:
    ValueType mVoxelSize;
    const bool mUnsigned;
};


template<typename ValueType>
struct VoxelSignOp
{
    VoxelSignOp(ValueType exBandWidth, ValueType inBandWidth)
        : mExBandWidth(exBandWidth)
        , mInBandWidth(inBandWidth)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t/*leafIndex*/) const
    {
        ValueType bgValues[2];
        bgValues[0] = mExBandWidth;
        bgValues[1] = -mInBandWidth;

        typename LeafNodeType::ValueOffIter iter = leaf.beginValueOff();

        for (; iter; ++iter) {
            ValueType& val = const_cast<ValueType&>(iter.getValue());
            val = bgValues[int(val < ValueType(0.0))];
        }
    }

private:
    ValueType mExBandWidth, mInBandWidth;
};


template<typename ValueType>
struct TrimOp
{
    TrimOp(ValueType exBandWidth, ValueType inBandWidth)
        : mExBandWidth(exBandWidth)
        , mInBandWidth(inBandWidth)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t/*leafIndex*/) const
    {
        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();

        for (; iter; ++iter) {
            const ValueType& val = iter.getValue();
            const bool inside = val < ValueType(0.0);

            if (inside && !(val > -mInBandWidth)) {
                iter.setValue(-mInBandWidth);
                iter.setValueOff();
            } else if (!inside && !(val < mInBandWidth)) {
                iter.setValue(mExBandWidth);
                iter.setValueOff();
            }
        }
    }

private:
    ValueType mExBandWidth, mInBandWidth;
};


template<typename ValueType>
struct OffsetOp
{
    OffsetOp(ValueType offset): mOffset(offset) {}

    void resetOffset(ValueType offset) { mOffset = offset; }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t/*leafIndex*/) const
    {
        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();
        for (; iter; ++iter) {
            ValueType& val = const_cast<ValueType&>(iter.getValue());
            val += mOffset;
        }
    }

private:
    ValueType mOffset;
};


template<typename GridType, typename ValueType>
struct RenormOp
{
    typedef math::BIAS_SCHEME<math::FIRST_BIAS> Scheme;
    typedef typename Scheme::template ISStencil<GridType>::StencilType Stencil;
    typedef tree::LeafManager<typename GridType::TreeType> LeafManagerType;
    typedef typename LeafManagerType::BufferType BufferType;

    RenormOp(GridType& grid, LeafManagerType& leafs, ValueType voxelSize, ValueType cfl = 1.0)
        : mGrid(grid)
        , mLeafs(leafs)
        , mVoxelSize(voxelSize)
        , mCFL(cfl)
    {
    }

    void resetCFL(ValueType cfl) { mCFL = cfl; }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t leafIndex) const
    {
        const ValueType dt = mCFL * mVoxelSize, one(1.0), invDx = one / mVoxelSize;
        Stencil stencil(mGrid);

        BufferType& buffer = mLeafs.getBuffer(leafIndex, 1);

        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();
        for (; iter; ++iter) {
            stencil.moveTo(iter);

            const ValueType normSqGradPhi =
                math::ISGradientNormSqrd<math::FIRST_BIAS>::result(stencil);

            const ValueType phi0 = stencil.getValue();
            const ValueType diff = math::Sqrt(normSqGradPhi) * invDx - one;
            const ValueType S = phi0 / (math::Sqrt(math::Pow2(phi0) + normSqGradPhi));

            buffer.setValue(iter.pos(), phi0 - dt * S * diff);
        }
    }

private:
    GridType& mGrid;
    LeafManagerType& mLeafs;
    ValueType mVoxelSize, mCFL;
};


template<typename TreeType, typename ValueType>
struct MinOp
{
    typedef tree::LeafManager<TreeType> LeafManagerType;
    typedef typename LeafManagerType::BufferType BufferType;

    MinOp(LeafManagerType& leafs): mLeafs(leafs) {}

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t leafIndex) const
    {
        BufferType& buffer = mLeafs.getBuffer(leafIndex, 1);

        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();
        for (; iter; ++iter) {
            ValueType& val = const_cast<ValueType&>(iter.getValue());
            val = std::min(val, buffer.getValue(iter.pos()));
        }
    }

private:
    LeafManagerType& mLeafs;
};


template<typename TreeType, typename ValueType>
struct MergeBufferOp
{
    typedef tree::LeafManager<TreeType> LeafManagerType;
    typedef typename LeafManagerType::BufferType BufferType;

    MergeBufferOp(LeafManagerType& leafs, size_t bufferIndex = 1)
        : mLeafs(leafs)
        , mBufferIndex(bufferIndex)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t leafIndex) const
    {
        BufferType& buffer = mLeafs.getBuffer(leafIndex, mBufferIndex);

        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();
        for (; iter; ++iter) {
            leaf.setValueOnly(iter.pos(), buffer.getValue(iter.pos()));
        }
    }

private:
    LeafManagerType& mLeafs;
    const size_t mBufferIndex;
};

} // internal namespace


////////////////////////////////////////


// MeshToVolume

template<typename DistGridT, typename InterruptT>
MeshToVolume<DistGridT, InterruptT>::MeshToVolume(
    openvdb::math::Transform::Ptr& transform, int conversionFlags,
    InterruptT *interrupter, int signSweeps)
    : mTransform(transform)
    , mConversionFlags(conversionFlags)
    , mSignSweeps(signSweeps)
    , mInterrupter(interrupter)
{
    clear();
    mSignSweeps = std::min(mSignSweeps, 1);
}


template<typename DistGridT, typename InterruptT>
void
MeshToVolume<DistGridT, InterruptT>::clear()
{
    mDistGrid = DistGridT::create(std::numeric_limits<DistValueT>::max());
    mIndexGrid = IndexGridT::create(Int32(util::INVALID_IDX));
    mIntersectingVoxelsGrid = StencilGridT::create(false);
}


template<typename DistGridT, typename InterruptT>
inline void
MeshToVolume<DistGridT, InterruptT>::convertToLevelSet(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    DistValueT exBandWidth, DistValueT inBandWidth)
{
    // The narrow band width is exclusive, the shortest valid distance has to be > 1 voxel
    exBandWidth = std::max(internal::Tolerance<DistValueT>::minNarrowBandWidth(), exBandWidth);
    inBandWidth = std::max(internal::Tolerance<DistValueT>::minNarrowBandWidth(), inBandWidth);
    const DistValueT vs = mTransform->voxelSize()[0];
    doConvert(pointList, polygonList, vs * exBandWidth, vs * inBandWidth);
    mDistGrid->setGridClass(GRID_LEVEL_SET);
}


template<typename DistGridT, typename InterruptT>
inline void
MeshToVolume<DistGridT, InterruptT>::convertToUnsignedDistanceField(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    DistValueT exBandWidth)
{
    // The narrow band width is exclusive, the shortest valid distance has to be > 1 voxel
    exBandWidth = std::max(internal::Tolerance<DistValueT>::minNarrowBandWidth(), exBandWidth);
    const DistValueT vs = mTransform->voxelSize()[0];
    doConvert(pointList, polygonList, vs * exBandWidth, 0.0, true);
    mDistGrid->setGridClass(GRID_UNKNOWN);
}


template<typename DistGridT, typename InterruptT>
void
MeshToVolume<DistGridT, InterruptT>::doConvert(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    DistValueT exBandWidth, DistValueT inBandWidth, bool unsignedDistField)
{
    mDistGrid->setTransform(mTransform);
    mIndexGrid->setTransform(mTransform);

    if (mInterrupter && mInterrupter->wasInterrupted()) return;

    // Voxelize mesh
    {
        internal::MeshVoxelizer<DistTreeT, InterruptT>
            voxelizer(pointList, polygonList, mInterrupter);

        voxelizer.runParallel();

        if (mInterrupter && mInterrupter->wasInterrupted()) return;

        mDistGrid->tree().merge(voxelizer.sqrDistTree());
        mIndexGrid->tree().merge(voxelizer.primIndexTree());
        mIntersectingVoxelsGrid->tree().merge(voxelizer.intersectionTree());
    }

    if (!unsignedDistField) {

        // Determine the inside/outside state for the narrow band of voxels.
        {
            // Slices up the volume and label the exterior contour of each slice in parallel.
            internal::ContourTracer<DistTreeT, InterruptT> trace(
                mDistGrid->tree(), mIntersectingVoxelsGrid->tree(), mInterrupter);

            for (int i = 0; i < mSignSweeps; ++i) {

                if (mInterrupter && mInterrupter->wasInterrupted()) break;
                trace.runParallel();

                if (mInterrupter && mInterrupter->wasInterrupted()) break;

                // Propagate sign information between the slices.
                internal::propagateSign<DistTreeT, InterruptT>
                    (mDistGrid->tree(), mIntersectingVoxelsGrid->tree(), mInterrupter);
            }
        }

        if (mInterrupter && mInterrupter->wasInterrupted()) return;

        {
            tree::LeafManager<StencilTreeT> leafs(mIntersectingVoxelsGrid->tree());

            // Determine the sign of the mesh intersecting voxels.
            internal::IntersectingVoxelSign<DistTreeT> sign(pointList, polygonList,
                mDistGrid->tree(), mIndexGrid->tree(), mIntersectingVoxelsGrid->tree(), leafs);

            sign.runParallel();

            if (mInterrupter && mInterrupter->wasInterrupted()) return;

            // Remove mesh intersecting voxels that where set by rasterizing
            // self-intersecting portions of the mesh.
            internal::IntersectingVoxelCleaner<DistTreeT> cleaner(mDistGrid->tree(),
                mIndexGrid->tree(), mIntersectingVoxelsGrid->tree(), leafs);

            cleaner.runParallel();
        }

        if (mInterrupter && mInterrupter->wasInterrupted()) return;

        {
            // Remove shell voxels that where set by rasterizing
            // self-intersecting portions of the mesh.

            tree::LeafManager<DistTreeT> leafs(mDistGrid->tree());

            internal::ShellVoxelCleaner<DistTreeT> cleaner(mDistGrid->tree(),
                leafs, mIndexGrid->tree(), mIntersectingVoxelsGrid->tree());

            cleaner.runParallel();
        }

        if (mInterrupter && mInterrupter->wasInterrupted()) return;

    } else { // if unsigned dist. field
        inBandWidth = DistValueT(0.0);
    }

    if (mDistGrid->activeVoxelCount() == 0) return;

    const DistValueT voxelSize(mTransform->voxelSize()[0]);

    // Transform values (world space scaling etc.)
    {
        typedef internal::SqrtAndScaleOp<DistValueT> XForm;
        tree::LeafManager<DistTreeT> leafs(mDistGrid->tree());
        XForm op(voxelSize, unsignedDistField);
        LeafTransformer<DistTreeT, XForm> transform(leafs, op);

        if (mInterrupter && mInterrupter->wasInterrupted()) return;

        transform.runParallel();
    }

    if (mInterrupter && mInterrupter->wasInterrupted()) return;

    if (!unsignedDistField) {
        // Propagate sign information to inactive values.
        mDistGrid->tree().signedFloodFill();

        if (mInterrupter && mInterrupter->wasInterrupted()) return;

        // Update the background value (inactive values)
        {
            tree::LeafManager<DistTreeT> leafs(mDistGrid->tree());

            typedef internal::VoxelSignOp<DistValueT> SignXForm;
            SignXForm op(exBandWidth, inBandWidth);

            LeafTransformer<DistTreeT, SignXForm> transform(leafs, op);
            transform.runParallel();

            if (mInterrupter && mInterrupter->wasInterrupted()) return;

            DistValueT bgValues[2];
            bgValues[0] = exBandWidth;
            bgValues[1] = -inBandWidth;

            typename DistTreeT::ValueAllIter tileIt(mDistGrid->tree());
            tileIt.setMaxDepth(DistTreeT::ValueAllIter::LEAF_DEPTH - 1);

            for ( ; tileIt; ++tileIt) {
                DistValueT& val = const_cast<DistValueT&>(tileIt.getValue());
                val = bgValues[int(val < DistValueT(0.0))];
            }

            if (mInterrupter && mInterrupter->wasInterrupted()) return;

            // fast bg value swap
            typename DistTreeT::Ptr newTree(new DistTreeT(/*background=*/exBandWidth));
            newTree->merge(mDistGrid->tree());
            mDistGrid->setTree(newTree);
        }

        // Smooth out bumps caused by self-intersecting and
        // overlapping portions of the mesh and renormalize the level set.
        {
            typedef internal::OffsetOp<DistValueT> OffsetOp;
            typedef internal::RenormOp<DistGridT, DistValueT> RenormOp;
            typedef internal::MinOp<DistTreeT, DistValueT> MinOp;
            typedef internal::MergeBufferOp<DistTreeT, DistValueT> MergeBufferOp;

            tree::LeafManager<DistTreeT> leafs(mDistGrid->tree(), 1);

            const DistValueT offset = 0.8 * voxelSize;

            if (mInterrupter && mInterrupter->wasInterrupted()) return;

            OffsetOp offsetOp(-offset);
            LeafTransformer<DistTreeT, OffsetOp> offsetXform(leafs, offsetOp);
            offsetXform.runParallel();

            if (mInterrupter && mInterrupter->wasInterrupted()) return;

            RenormOp renormOp(*mDistGrid, leafs, voxelSize);
            LeafTransformer<DistTreeT, RenormOp> renormXform(leafs, renormOp);
            renormXform.runParallel();

            MinOp minOp(leafs);
            LeafTransformer<DistTreeT, MinOp> minXform(leafs, minOp);
            minXform.runParallel();

            if (mInterrupter && mInterrupter->wasInterrupted()) return;

            offsetOp.resetOffset(offset - internal::Tolerance<DistValueT>::epsilon());
            offsetXform.runParallel();
        }

        mIntersectingVoxelsGrid->clear();
    }

    if (mInterrupter && mInterrupter->wasInterrupted()) return;

    // Narrow-band dilation
    const DistValueT minWidth = voxelSize * 2.0;
    if (inBandWidth > minWidth || exBandWidth > minWidth) {

        // Create the initial voxel mask.
        StencilTreeT maskTree(false);
        tree::ValueAccessor<StencilTreeT> acc(maskTree);
        maskTree.topologyUnion(mDistGrid->tree());

        // Preallocate leafs.
        {
            typedef typename DistTreeT::LeafNodeType DistLeafType;

            std::vector<DistLeafType*> distLeafs;
            distLeafs.reserve(mDistGrid->tree().leafCount());

            typename DistTreeT::LeafIter iter = mDistGrid->tree().beginLeaf();
            for ( ; iter; ++iter) distLeafs.push_back(iter.getLeaf());

            tree::ValueAccessor<DistTreeT> distAcc(mDistGrid->tree());

            DistValueT leafSize = DistValueT(DistLeafType::DIM - 1) * voxelSize;

            const double inLeafsRatio = double(inBandWidth) / double(leafSize);
            size_t inLeafs = std::numeric_limits<size_t>::max();
            if (double(inLeafs) > (inLeafsRatio + 1.0)) {
                inLeafs = size_t(std::ceil(inLeafsRatio)) + 1;
            }
            size_t exLeafs = size_t(std::ceil(exBandWidth / leafSize)) + 1;
            size_t numLeafs = std::max(inLeafs, exLeafs);

            for (size_t i = 0; i < numLeafs; ++i) {

                if (mInterrupter && mInterrupter->wasInterrupted()) return;

                std::vector<DistLeafType*> newDistLeafs;
                newDistLeafs.reserve(2 * distLeafs.size());

                for (size_t n = 0, N = distLeafs.size(); n < N; ++n) {

                    Coord ijk = distLeafs[n]->getOrigin();

                    const bool inside = distLeafs[n]->getValue(ijk) < DistValueT(0.0);

                    if (inside && !(i < inLeafs)) continue;
                    else if (!inside && !(i < exLeafs)) continue;

                    ijk[0] -= 1;
                    if (distAcc.probeLeaf(ijk) == NULL) {
                        newDistLeafs.push_back(distAcc.touchLeaf(ijk));
                    }

                    ijk[0] += 1;
                    ijk[1] -= 1;
                    if (distAcc.probeLeaf(ijk) == NULL) {
                        newDistLeafs.push_back(distAcc.touchLeaf(ijk));
                    }

                    ijk[1] += 1;
                    ijk[2] -= 1;
                    if (distAcc.probeLeaf(ijk) == NULL) {
                        newDistLeafs.push_back(distAcc.touchLeaf(ijk));
                    }

                    ijk[2] += 1;
                    ijk[0] += DistLeafType::DIM;
                    if (distAcc.probeLeaf(ijk) == NULL) {
                        newDistLeafs.push_back(distAcc.touchLeaf(ijk));
                    }

                    ijk[0] -= DistLeafType::DIM;
                    ijk[1] += DistLeafType::DIM;
                    if (distAcc.probeLeaf(ijk) == NULL) {
                        newDistLeafs.push_back(distAcc.touchLeaf(ijk));
                    }

                    ijk[1] -= DistLeafType::DIM;
                    ijk[2] += DistLeafType::DIM;
                    if (distAcc.probeLeaf(ijk) == NULL) {
                        newDistLeafs.push_back(distAcc.touchLeaf(ijk));
                    }
                }

                if (newDistLeafs.empty()) break;
                distLeafs.swap(newDistLeafs);
            }
        }

        if (mInterrupter && mInterrupter->wasInterrupted()) return;

        mIndexGrid->tree().topologyUnion(mDistGrid->tree());

        while (maskTree.activeVoxelCount() > 0) {

            if (mInterrupter && mInterrupter->wasInterrupted()) break;

            openvdb::tools::dilateVoxels(maskTree);
            tree::LeafManager<StencilTreeT> leafs(maskTree);

            internal::ExpandNB<DistTreeT> expand(pointList, polygonList, mDistGrid->tree(),
                mIndexGrid->tree(), maskTree, leafs, exBandWidth, inBandWidth, voxelSize);

            expand.runParallel();
        }
    }

    if (!bool(GENERATE_PRIM_INDEX_GRID & mConversionFlags)) mIndexGrid->clear();

    const DistValueT minTrimWidth = voxelSize * 4.0;
    if (inBandWidth < minTrimWidth || exBandWidth < minTrimWidth) {

        // If the narrow band was not expanded, we might need to trim off
        // some of the active voxels in order to respect the narrow band limits.
        // (The mesh voxelization step generates some extra 'shell' voxels)

        tree::LeafManager<DistTreeT> leafs(mDistGrid->tree());

        typedef internal::TrimOp<DistValueT> TrimOp;

        TrimOp op(exBandWidth, unsignedDistField ? exBandWidth : inBandWidth);
        LeafTransformer<DistTreeT, TrimOp> transform(leafs, op);
        transform.runParallel();
    }

    if (mInterrupter && mInterrupter->wasInterrupted()) return;

    mDistGrid->tree().pruneLevelSet();
}


////////////////////////////////////////


/// @internal This overload is enabled only for grids with a scalar, floating-point ValueType.
template<typename GridType>
inline typename boost::enable_if<boost::is_floating_point<typename GridType::ValueType>,
typename GridType::Ptr>::type
doMeshConversion(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth,
    bool unsignedDistanceField = false)
{
    std::vector<Vec3s> indexSpacePoints(points.size());

    { // Copy and transform (required for MeshToVolume) points to grid space.
        internal::PointTransform ptnXForm(points, indexSpacePoints, xform);
        ptnXForm.runParallel();
    }

    // Copy primitives
    std::vector<Vec4I> primitives(triangles.size() + quads.size());

    for (size_t n = 0, N = triangles.size(); n < N; ++n) {
        Vec4I& prim = primitives[n];
        const Vec3I& triangle = triangles[n];
        prim[0] = triangle[0];
        prim[1] = triangle[1];
        prim[2] = triangle[2];
        prim[3] = util::INVALID_IDX;
    }

    for (size_t n = 0, N = quads.size(); n < N; ++n) {
        primitives[n + triangles.size()] = quads[n];
    }

    typename GridType::ValueType exWidth(exBandWidth);
    typename GridType::ValueType inWidth(inBandWidth);


    math::Transform::Ptr transform = xform.copy();
    MeshToVolume<GridType> vol(transform);

    if (!unsignedDistanceField) {
        vol.convertToLevelSet(indexSpacePoints, primitives, exWidth, inWidth);
    } else {
        vol.convertToUnsignedDistanceField(indexSpacePoints, primitives, exWidth);
    }

    return vol.distGridPtr();
}


/// @internal This overload is enabled only for grids that do not have a scalar,
/// floating-point ValueType.
template<typename GridType>
inline typename boost::disable_if<boost::is_floating_point<typename GridType::ValueType>,
typename GridType::Ptr>::type
doMeshConversion(
    const math::Transform& /*xform*/,
    const std::vector<Vec3s>& /*points*/,
    const std::vector<Vec3I>& /*triangles*/,
    const std::vector<Vec4I>& /*quads*/,
    float /*exBandWidth*/,
    float /*inBandWidth*/,
    bool unsignedDistanceField = false)
{
    OPENVDB_THROW(TypeError,
        "mesh to volume conversion is supported only for scalar, floating-point grids");
}


////////////////////////////////////////


template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    float halfWidth)
{
    std::vector<Vec4I> quads(0);
    return doMeshConversion<GridType>(xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec4I>& quads,
    float halfWidth)
{
    std::vector<Vec3I> triangles(0);
    return doMeshConversion<GridType>(xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToLevelSet(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float halfWidth)
{
    return doMeshConversion<GridType>(xform, points, triangles, quads,
        halfWidth, halfWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToSignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float exBandWidth,
    float inBandWidth)
{
    return doMeshConversion<GridType>(xform, points, triangles,
        quads, exBandWidth, inBandWidth);
}


template<typename GridType>
inline typename GridType::Ptr
meshToUnsignedDistanceField(
    const openvdb::math::Transform& xform,
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads,
    float bandWidth)
{
    return doMeshConversion<GridType>(xform, points, triangles, quads,
        bandWidth, bandWidth, true);
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
