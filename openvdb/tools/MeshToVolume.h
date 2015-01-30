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

#ifndef OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/math/FiniteDifference.h>
#include <openvdb/math/Operators.h> // for ISGradientNormSqrd
#include <openvdb/math/Proximity.h> // for closestPointOnTriangleToPoint()
#include <openvdb/tools/Morphology.h> // for dilateVoxels()
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/util/Util.h> // for nearestCoord()
#include "ChangeBackground.h"
#include "Prune.h"// for pruneInactive and pruneLevelSet
#include "SignedFloodFill.h" // for signedFloodFillWithValues

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <boost/math/special_functions/fpclassify.hpp> // for isfinite()

#include <deque>
#include <limits>
#include <sstream>

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
enum { GENERATE_PRIM_INDEX_GRID = 0x1, OUTPUT_RAW_DATA = 0x2};


// MeshToVolume
template<typename FloatGridT, typename InterruptT = util::NullInterrupter>
class MeshToVolume
{
public:
    typedef typename FloatGridT::TreeType FloatTreeT;
    typedef typename FloatTreeT::ValueType FloatValueT;
    typedef typename FloatTreeT::template ValueConverter<Int32>::Type IntTreeT;
    typedef Grid<IntTreeT> IntGridT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef Grid<BoolTreeT> BoolGridT;

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
        FloatValueT exBandWidth = FloatValueT(LEVEL_SET_HALF_WIDTH),
        FloatValueT inBandWidth = FloatValueT(LEVEL_SET_HALF_WIDTH));

    /// @brief Mesh to Unsigned Distance Field conversion
    ///
    /// @note Does not requires a closed surface.
    ///
    /// @param pointList    List of points in grid index space, preferably unique
    ///                     and shared by different polygons.
    /// @param polygonList  List of triangles and/or quads.
    /// @param exBandWidth  The narrow-band width in voxel units.
    void convertToUnsignedDistanceField(const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList, FloatValueT exBandWidth);

    void clear();

    /// Returns a narrow-band (signed) distance field / level set grid.
    typename FloatGridT::Ptr distGridPtr() const { return mDistGrid; }

    /// Returns a grid containing the closest-primitive index for each
    /// voxel in the narrow-band.
    typename IntGridT::Ptr indexGridPtr() const { return mIndexGrid; }

private:
    // disallow copy by assignment
    void operator=(const MeshToVolume<FloatGridT, InterruptT>&) {}

    void doConvert(const std::vector<Vec3s>&, const std::vector<Vec4I>&,
        FloatValueT exBandWidth, FloatValueT inBandWidth, bool unsignedDistField = false);

    bool wasInterrupted(int percent = -1) const {
        return mInterrupter && mInterrupter->wasInterrupted(percent);
    }

    openvdb::math::Transform::Ptr mTransform;
    int mConversionFlags, mSignSweeps;

    typename FloatGridT::Ptr mDistGrid;
    typename IntGridT::Ptr mIndexGrid;
    typename BoolGridT::Ptr mIntersectingVoxelsGrid;

    InterruptT *mInterrupter;
};


////////////////////////////////////////


/// @brief  Extracts and stores voxel edge intersection data from a mesh.
class MeshToVoxelEdgeData
{
public:

    //////////

    ///@brief Internal edge data type.
    struct EdgeData {
        EdgeData(float dist = 1.0)
            : mXDist(dist), mYDist(dist), mZDist(dist)
            , mXPrim(util::INVALID_IDX)
            , mYPrim(util::INVALID_IDX)
            , mZPrim(util::INVALID_IDX)
        {
        }

        //@{
        /// Required by several of the tree nodes
        /// @note These methods don't perform meaningful operations.
        bool operator< (const EdgeData&) const { return false; }
        bool operator> (const EdgeData&) const { return false; }
        template<class T> EdgeData operator+(const T&) const { return *this; }
        template<class T> EdgeData operator-(const T&) const { return *this; }
        EdgeData operator-() const { return *this; }
        //@}

        bool operator==(const EdgeData& rhs) const
        {
            return mXPrim == rhs.mXPrim && mYPrim == rhs.mYPrim && mZPrim == rhs.mZPrim;
        }

        float mXDist, mYDist, mZDist;
        Index32 mXPrim, mYPrim, mZPrim;
    };

    typedef tree::Tree4<EdgeData, 5, 4, 3>::Type    TreeType;
    typedef tree::ValueAccessor<TreeType>           Accessor;


    //////////


    MeshToVoxelEdgeData();


    /// @brief  Threaded method to extract voxel edge data, the closest
    ///         intersection point and corresponding primitive index,
    ///         from the given mesh.
    ///
    /// @param pointList    List of points in grid index space, preferably unique
    ///                     and shared by different polygons.
    /// @param polygonList  List of triangles and/or quads.
    void convert(const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList);


    /// @brief  Returns intersection points with corresponding primitive
    ///         indices for the given @c ijk voxel.
    void getEdgeData(Accessor& acc, const Coord& ijk,
        std::vector<Vec3d>& points, std::vector<Index32>& primitives);

    /// @return An accessor of @c MeshToVoxelEdgeData::Accessor type that
    ///         provides random read access to the internal tree.
    Accessor getAccessor() { return Accessor(mTree); }

private:
    void operator=(const MeshToVoxelEdgeData&) {}
    TreeType mTree;
    class GenEdgeData;
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

    void run(bool threaded = true)
    {
        if (threaded) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mPointsOut->size()), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, mPointsOut->size()));
        }
    }

    inline void operator()(const tbb::blocked_range<size_t>& range) const
    {
        for (size_t n = range.begin(); n < range.end(); ++n) {
            (*mPointsOut)[n] = Vec3s(mXform.worldToIndex(mPointsIn[n]));
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


template<typename FloatTreeT, typename IntTreeT>
inline void
combine(FloatTreeT& lhsDist, IntTreeT& lhsIndex, FloatTreeT& rhsDist, IntTreeT& rhsIndex)
{
    typedef typename FloatTreeT::ValueType FloatValueT;
    typename tree::ValueAccessor<FloatTreeT> lhsDistAccessor(lhsDist);
    typename tree::ValueAccessor<IntTreeT> lhsIndexAccessor(lhsIndex);
    typename tree::ValueAccessor<IntTreeT> rhsIndexAccessor(rhsIndex);
    typename FloatTreeT::LeafCIter iter = rhsDist.cbeginLeaf();

    FloatValueT rhsValue;
    Coord ijk;

    for ( ; iter; ++iter) {
        typename FloatTreeT::LeafNodeType::ValueOnCIter it = iter->cbeginValueOn();

        for ( ; it; ++it) {

            ijk = it.getCoord();
            rhsValue = it.getValue();
            FloatValueT& lhsValue = const_cast<FloatValueT&>(lhsDistAccessor.getValue(ijk));

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
/// intersecting voxels grid (masks the mesh intersecting voxels)
/// @note Only the leaf nodes that intersect the mesh are allocated, and only voxels in
/// a narrow band (of two to three voxels in proximity to the mesh's surface) are activated.
/// They are populated with distance values and primitive indices.
template<typename FloatTreeT, typename InterruptT = util::NullInterrupter>
class MeshVoxelizer
{
public:
    typedef typename FloatTreeT::ValueType                              FloatValueT;
    typedef typename FloatTreeT::LeafNodeType                           FloatLeafT;
    typedef typename tree::ValueAccessor<FloatTreeT>                    FloatAccessorT;
    typedef typename FloatTreeT::template ValueConverter<Int32>::Type   IntTreeT;
    typedef typename IntTreeT::LeafNodeType                             IntLeafT;
    typedef typename tree::ValueAccessor<IntTreeT>                      IntAccessorT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type    BoolTreeT;
    typedef typename BoolTreeT::LeafNodeType                            BoolLeafT;
    typedef typename tree::ValueAccessor<BoolTreeT>                     BoolAccessorT;


    MeshVoxelizer(const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList, InterruptT *interrupter = NULL);

    ~MeshVoxelizer() {}

    void run(bool threaded = true);

    MeshVoxelizer(MeshVoxelizer<FloatTreeT, InterruptT>& rhs, tbb::split);
    void operator() (const tbb::blocked_range<size_t> &range);
    void join(MeshVoxelizer<FloatTreeT, InterruptT>& rhs);

    FloatTreeT& sqrDistTree() { return mSqrDistTree; }
    IntTreeT& primIndexTree() { return mPrimIndexTree; }
    BoolTreeT& intersectionTree() { return mIntersectionTree; }

private:
    // disallow copy by assignment
    void operator=(const MeshVoxelizer<FloatTreeT, InterruptT>&) {}
    bool wasInterrupted() const { return mInterrupter && mInterrupter->wasInterrupted(); }

    bool evalVoxel(const Coord& ijk, const Int32 polyIdx);

    const std::vector<Vec3s>& mPointList;
    const std::vector<Vec4I>& mPolygonList;

    FloatTreeT mSqrDistTree;
    FloatAccessorT mSqrDistAccessor;

    IntTreeT mPrimIndexTree;
    IntAccessorT mPrimIndexAccessor;

    BoolTreeT mIntersectionTree;
    BoolAccessorT mIntersectionAccessor;

    // Used internally for acceleration
    IntTreeT mLastPrimTree;
    IntAccessorT mLastPrimAccessor;

    InterruptT *mInterrupter;


    struct Primitive { Vec3d a, b, c, d; Int32 index; };

    template<bool IsQuad>
    bool evalPrimitive(const Coord&, const Primitive&);

    template<bool IsQuad>
    void voxelize(const Primitive&);
};


template<typename FloatTreeT, typename InterruptT>
void
MeshVoxelizer<FloatTreeT, InterruptT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mPolygonList.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mPolygonList.size()));
    }
}

template<typename FloatTreeT, typename InterruptT>
MeshVoxelizer<FloatTreeT, InterruptT>::MeshVoxelizer(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    InterruptT *interrupter)
    : mPointList(pointList)
    , mPolygonList(polygonList)
    , mSqrDistTree(std::numeric_limits<FloatValueT>::max())
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

template<typename FloatTreeT, typename InterruptT>
MeshVoxelizer<FloatTreeT, InterruptT>::MeshVoxelizer(
    MeshVoxelizer<FloatTreeT, InterruptT>& rhs, tbb::split)
    : mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mSqrDistTree(std::numeric_limits<FloatValueT>::max())
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


template<typename FloatTreeT, typename InterruptT>
void
MeshVoxelizer<FloatTreeT, InterruptT>::operator()(const tbb::blocked_range<size_t> &range)
{
    Primitive prim;

    for (size_t n = range.begin(); n < range.end(); ++n) {

        if (mInterrupter && mInterrupter->wasInterrupted()) {
            tbb::task::self().cancel_group_execution();
            break;
        }

        const Vec4I& verts = mPolygonList[n];

        prim.index = Int32(n);
        prim.a = Vec3d(mPointList[verts[0]]);
        prim.b = Vec3d(mPointList[verts[1]]);
        prim.c = Vec3d(mPointList[verts[2]]);

        if (util::INVALID_IDX != verts[3]) {
            prim.d = Vec3d(mPointList[verts[3]]);
            voxelize<true>(prim);
        } else {
            voxelize<false>(prim);
        }
    }
}


template<typename FloatTreeT, typename InterruptT>
template<bool IsQuad>
void
MeshVoxelizer<FloatTreeT, InterruptT>::voxelize(const Primitive& prim)
{
    std::deque<Coord> coordList;
    Coord ijk, nijk;

    ijk = util::nearestCoord(prim.a);
    coordList.push_back(ijk);

    evalPrimitive<IsQuad>(ijk, prim);

    while (!coordList.empty()) {
        if(wasInterrupted()) break;

        ijk = coordList.back();
        coordList.pop_back();

        mIntersectionAccessor.setActiveState(ijk, true);

        for (Int32 i = 0; i < 26; ++i) {
            nijk = ijk + util::COORD_OFFSETS[i];
            if (prim.index != mLastPrimAccessor.getValue(nijk)) {
                mLastPrimAccessor.setValue(nijk, prim.index);
                if(evalPrimitive<IsQuad>(nijk, prim)) coordList.push_back(nijk);
            }
        }
    }
}


template<typename FloatTreeT, typename InterruptT>
template<bool IsQuad>
bool
MeshVoxelizer<FloatTreeT, InterruptT>::evalPrimitive(const Coord& ijk, const Primitive& prim)
{
    Vec3d uvw, voxelCenter(ijk[0], ijk[1], ijk[2]);

    // Evaluate first triangle
    FloatValueT dist = FloatValueT((voxelCenter -
        closestPointOnTriangleToPoint(prim.a, prim.c, prim.b, voxelCenter, uvw)).lengthSqr());

    if (IsQuad) {
        // Split quad into a second triangle and calculate distance.
        FloatValueT secondDist = FloatValueT((voxelCenter -
            closestPointOnTriangleToPoint(prim.a, prim.d, prim.c, voxelCenter, uvw)).lengthSqr());

        if (secondDist < dist) dist = secondDist;
    }

    FloatValueT oldDist = std::abs(mSqrDistAccessor.getValue(ijk));

    if (dist < oldDist) {
        mSqrDistAccessor.setValue(ijk, -dist);
        mPrimIndexAccessor.setValue(ijk, prim.index);
    } else if (math::isExactlyEqual(dist, oldDist)) {
        // makes reduction deterministic when different polygons
        // produce the same distance value.
        mPrimIndexAccessor.setValue(ijk, std::min(prim.index, mPrimIndexAccessor.getValue(ijk)));
    }

    return (dist < 0.86602540378443861);
}


template<typename FloatTreeT, typename InterruptT>
void
MeshVoxelizer<FloatTreeT, InterruptT>::join(MeshVoxelizer<FloatTreeT, InterruptT>& rhs)
{
    typedef typename FloatTreeT::RootNodeType FloatRootNodeT;
    typedef typename FloatRootNodeT::NodeChainType  FloatNodeChainT;
    BOOST_STATIC_ASSERT(boost::mpl::size<FloatNodeChainT>::value > 1);
    typedef typename boost::mpl::at<FloatNodeChainT, boost::mpl::int_<1> >::type FloatInternalNodeT;

    typedef typename IntTreeT::RootNodeType IntRootNodeT;
    typedef typename IntRootNodeT::NodeChainType  IntNodeChainT;
    BOOST_STATIC_ASSERT(boost::mpl::size<IntNodeChainT>::value > 1);
    typedef typename boost::mpl::at<IntNodeChainT, boost::mpl::int_<1> >::type IntInternalNodeT;

    const FloatValueT background = std::numeric_limits<FloatValueT>::max();

    Coord ijk;
    Index offset;

    rhs.mSqrDistTree.clearAllAccessors();
    rhs.mPrimIndexTree.clearAllAccessors();

    typename FloatTreeT::LeafIter leafIt = rhs.mSqrDistTree.beginLeaf();
    for ( ; leafIt; ++leafIt) {

        ijk = leafIt->origin();
        FloatLeafT* lhsDistLeafPt = mSqrDistAccessor.probeLeaf(ijk);

        if (!lhsDistLeafPt) {

            // Steals leaf nodes through their parent, always the last internal-node
            // stored in the ValueAccessor's node chain, avoiding the overhead of
            // the root node. This is significantly faster than going through the
            // tree or root node.
            mSqrDistAccessor.addLeaf(rhs.mSqrDistAccessor.probeLeaf(ijk));
            FloatInternalNodeT* floatNode =
                rhs.mSqrDistAccessor.template getNode<FloatInternalNodeT>();
            floatNode->template stealNode<FloatLeafT>(ijk, background, false);

            mPrimIndexAccessor.addLeaf(rhs.mPrimIndexAccessor.probeLeaf(ijk));
            IntInternalNodeT* intNode =
                rhs.mPrimIndexAccessor.template getNode<IntInternalNodeT>();
            intNode->template stealNode<IntLeafT>(ijk, util::INVALID_IDX, false);

        } else {

            IntLeafT* lhsIdxLeafPt = mPrimIndexAccessor.probeLeaf(ijk);
            IntLeafT* rhsIdxLeafPt = rhs.mPrimIndexAccessor.probeLeaf(ijk);
            FloatValueT lhsValue, rhsValue;

            typename FloatLeafT::ValueOnCIter it = leafIt->cbeginValueOn();
            for ( ; it; ++it) {

                offset = it.pos();

                lhsValue = std::abs(lhsDistLeafPt->getValue(offset));
                rhsValue = std::abs(it.getValue());

                if (rhsValue < lhsValue) {
                    lhsDistLeafPt->setValueOn(offset, it.getValue());
                    lhsIdxLeafPt->setValueOn(offset, rhsIdxLeafPt->getValue(offset));
                } else if (math::isExactlyEqual(rhsValue, lhsValue)) {
                    lhsIdxLeafPt->setValueOn(offset,
                        std::min(lhsIdxLeafPt->getValue(offset), rhsIdxLeafPt->getValue(offset)));
                }
            }
        }
    }

    mIntersectionTree.merge(rhs.mIntersectionTree);

    rhs.mSqrDistTree.clear();
    rhs.mPrimIndexTree.clear();
    rhs.mIntersectionTree.clear();
}


////////////////////////////////////////


// ContourTracer
/// @brief TBB body object that partitions a volume into 2D slices that can be processed
/// in parallel and marks the exterior contour of disjoint voxel sets in each slice
template<typename FloatTreeT, typename InterruptT = util::NullInterrupter>
class ContourTracer
{
public:
    typedef typename FloatTreeT::ValueType FloatValueT;
    typedef typename tree::ValueAccessor<FloatTreeT> DistAccessorT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename tree::ValueAccessor<const BoolTreeT> BoolAccessorT;

    ContourTracer(FloatTreeT&, const BoolTreeT&, InterruptT *interrupter = NULL);
    ~ContourTracer() {}

    void run(bool threaded = true);

    ContourTracer(const ContourTracer<FloatTreeT, InterruptT>& rhs);
    void operator()(const tbb::blocked_range<int> &range) const;

private:
    void operator=(const ContourTracer<FloatTreeT, InterruptT>&) {}

    int sparseScan(int slice) const;

    FloatTreeT& mDistTree;
    DistAccessorT mDistAccessor;

    const BoolTreeT& mIntersectionTree;
    BoolAccessorT mIntersectionAccessor;

    CoordBBox mBBox;

    /// List of value-depth dependant step sizes.
    std::vector<Index> mStepSize;

    InterruptT *mInterrupter;
};


template<typename FloatTreeT, typename InterruptT>
void
ContourTracer<FloatTreeT, InterruptT>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(tbb::blocked_range<int>(mBBox.min()[0], mBBox.max()[0]+1), *this);
    } else {
        (*this)(tbb::blocked_range<int>(mBBox.min()[0], mBBox.max()[0]+1));
    }
}


template<typename FloatTreeT, typename InterruptT>
ContourTracer<FloatTreeT, InterruptT>::ContourTracer(
    FloatTreeT& distTree, const BoolTreeT& intersectionTree, InterruptT *interrupter)
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


template<typename FloatTreeT, typename InterruptT>
ContourTracer<FloatTreeT, InterruptT>::ContourTracer(
    const ContourTracer<FloatTreeT, InterruptT> &rhs)
    : mDistTree(rhs.mDistTree)
    , mDistAccessor(mDistTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mIntersectionAccessor(mIntersectionTree)
    , mBBox(rhs.mBBox)
    , mStepSize(rhs.mStepSize)
    , mInterrupter(rhs.mInterrupter)
{
}


template<typename FloatTreeT, typename InterruptT>
void
ContourTracer<FloatTreeT, InterruptT>::operator()(const tbb::blocked_range<int> &range) const
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


template<typename FloatTreeT, typename InterruptT>
int
ContourTracer<FloatTreeT, InterruptT>::sparseScan(int slice) const
{
    bool lastVoxelWasOut = true;
    int last_k = mBBox.min()[2];

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

                    FloatValueT& val = const_cast<FloatValueT&>(mDistAccessor.getValue(ijk));
                    val = -val; // flip sign

                } else {

                    FloatValueT val;
                    for (Int32 n = 3; n < 6; n += 2) {
                        n_ijk = ijk + util::COORD_OFFSETS[n];

                        if (mDistAccessor.probeValue(n_ijk, val) && val > 0) {
                            lastVoxelWasOut = true;
                            break;
                        }
                    }

                    if (lastVoxelWasOut) {

                        FloatValueT& v = const_cast<FloatValueT&>(mDistAccessor.getValue(ijk));
                        v = -v; // flip sign

                        const int tmp_k = ijk[2];

                        // backtrace
                        for (--ijk[2]; ijk[2] >= last_k; --ijk[2]) {
                            if (mIntersectionAccessor.isValueOn(ijk)) break;
                            FloatValueT& vb =
                                const_cast<FloatValueT&>(mDistAccessor.getValue(ijk));
                            if (vb < FloatValueT(0.0)) vb = -vb; // flip sign
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


/// @brief TBB body object that that finds seed points for the parallel flood fill.
template<typename FloatTreeT, typename InterruptT = util::NullInterrupter>
class SignMask
{
public:
    typedef typename FloatTreeT::ValueType                              FloatValueT;
    typedef typename FloatTreeT::LeafNodeType                           FloatLeafT;
    typedef tree::LeafManager<FloatTreeT>                               FloatLeafManager;
    typedef typename tree::ValueAccessor<const FloatTreeT>              FloatAccessorT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type    BoolTreeT;
    typedef typename BoolTreeT::LeafNodeType                            BoolLeafT;
    typedef typename tree::ValueAccessor<BoolTreeT>                     BoolAccessorT;
    typedef typename tree::ValueAccessor<const BoolTreeT>               BoolConstAccessorT;


    SignMask(const FloatLeafManager&, const FloatTreeT&, const BoolTreeT&,
        InterruptT *interrupter = NULL);

    ~SignMask() {}

    void run(bool threaded = true);

    SignMask(SignMask<FloatTreeT, InterruptT>& rhs, tbb::split);
    void operator() (const tbb::blocked_range<size_t> &range);
    void join(SignMask<FloatTreeT, InterruptT>& rhs);

    BoolTreeT& signMaskTree() { return mSignMaskTree; }

private:
    // disallow copy by assignment
    void operator=(const SignMask<FloatTreeT, InterruptT>&) {}
    bool wasInterrupted() const { return mInterrupter && mInterrupter->wasInterrupted(); }

    const FloatLeafManager& mDistLeafs;
    const FloatTreeT& mDistTree;
    const BoolTreeT& mIntersectionTree;

    BoolTreeT mSignMaskTree;

    InterruptT *mInterrupter;
}; // class SignMask


template<typename FloatTreeT, typename InterruptT>
SignMask<FloatTreeT, InterruptT>::SignMask(
    const FloatLeafManager& distLeafs, const FloatTreeT& distTree,
    const BoolTreeT& intersectionTree, InterruptT *interrupter)
    : mDistLeafs(distLeafs)
    , mDistTree(distTree)
    , mIntersectionTree(intersectionTree)
    , mSignMaskTree(false)
    , mInterrupter(interrupter)
{
}


template<typename FloatTreeT, typename InterruptT>
SignMask<FloatTreeT, InterruptT>::SignMask(
    SignMask<FloatTreeT, InterruptT>& rhs, tbb::split)
    : mDistLeafs(rhs.mDistLeafs)
    , mDistTree(rhs.mDistTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mSignMaskTree(false)
    , mInterrupter(rhs.mInterrupter)
{
}


template<typename FloatTreeT, typename InterruptT>
void
SignMask<FloatTreeT, InterruptT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mDistLeafs.getRange(), *this);
    else (*this)(mDistLeafs.getRange());
}


template<typename FloatTreeT, typename InterruptT>
void
SignMask<FloatTreeT, InterruptT>::operator()(const tbb::blocked_range<size_t> &range)
{
    FloatAccessorT distAcc(mDistTree);
    BoolConstAccessorT intersectionAcc(mIntersectionTree);
    BoolAccessorT maskAcc(mSignMaskTree);

    FloatValueT value;
    CoordBBox bbox;
    Coord& maxCoord = bbox.max();
    Coord& minCoord = bbox.min();
    Coord ijk;
    const int extent = BoolLeafT::DIM - 1;

    for (size_t n = range.begin(); n < range.end(); ++n) {

        const FloatLeafT& distLeaf = mDistLeafs.leaf(n);

        minCoord = distLeaf.origin();
        maxCoord[0] = minCoord[0] + extent;
        maxCoord[1] = minCoord[1] + extent;
        maxCoord[2] = minCoord[2] + extent;

        const BoolLeafT* intersectionLeaf = intersectionAcc.probeConstLeaf(minCoord);

        BoolLeafT* maskLeafPt = new BoolLeafT(minCoord, false);
        BoolLeafT& maskLeaf = *maskLeafPt;
        bool addLeaf = false;

        bbox.expand(-1);

        typename FloatLeafT::ValueOnCIter it = distLeaf.cbeginValueOn();
        for (; it; ++it) {
            if (intersectionLeaf && intersectionLeaf->isValueOn(it.pos())) continue;
            if (it.getValue() < FloatValueT(0.0)) {
                ijk = it.getCoord();
                if (bbox.isInside(ijk)) {
                    for (size_t i   = 0; i < 6; ++i) {
                        if (distLeaf.probeValue(ijk+util::COORD_OFFSETS[i], value) && value>0.0) {
                            maskLeaf.setValueOn(ijk);
                            addLeaf = true;
                            break;
                        }
                    }
                } else {
                    for (size_t i = 0; i < 6; ++i) {
                        if (distAcc.probeValue(ijk+util::COORD_OFFSETS[i], value) && value>0.0) {
                            maskLeaf.setValueOn(ijk);
                            addLeaf = true;
                            break;
                        }
                    }
                }
            }
        }

        if (addLeaf) maskAcc.addLeaf(maskLeafPt);
        else delete maskLeafPt;
    }
}


template<typename FloatTreeT, typename InterruptT>
void
SignMask<FloatTreeT, InterruptT>::join(SignMask<FloatTreeT, InterruptT>& rhs)
{
    mSignMaskTree.merge(rhs.mSignMaskTree);
}


////////////////////////////////////////


/// @brief TBB body object that performs a parallel flood fill
template<typename FloatTreeT, typename InterruptT = util::NullInterrupter>
class PropagateSign
{
public:
    typedef typename FloatTreeT::ValueType                              FloatValueT;
    typedef typename FloatTreeT::LeafNodeType                           FloatLeafT;
    typedef typename tree::ValueAccessor<FloatTreeT>                    FloatAccessorT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type    BoolTreeT;
    typedef typename BoolTreeT::LeafNodeType                            BoolLeafT;
    typedef tree::LeafManager<BoolTreeT>                                BoolLeafManager;
    typedef typename tree::ValueAccessor<BoolTreeT>                     BoolAccessorT;
    typedef typename tree::ValueAccessor<const BoolTreeT>               BoolConstAccessorT;

    PropagateSign(BoolLeafManager&, FloatTreeT&, const BoolTreeT&, InterruptT *interrupter = NULL);

    ~PropagateSign() {}

    void run(bool threaded = true);

    PropagateSign(PropagateSign<FloatTreeT, InterruptT>& rhs, tbb::split);
    void operator() (const tbb::blocked_range<size_t> &range);
    void join(PropagateSign<FloatTreeT, InterruptT>& rhs);

    BoolTreeT& signMaskTree() { return mSignMaskTree; }

private:
    // disallow copy by assignment
    void operator=(const PropagateSign<FloatTreeT, InterruptT>&);
    bool wasInterrupted() const { return mInterrupter && mInterrupter->wasInterrupted(); }

    BoolLeafManager& mOldSignMaskLeafs;
    FloatTreeT& mDistTree;
    const BoolTreeT& mIntersectionTree;

    BoolTreeT mSignMaskTree;
    InterruptT *mInterrupter;
};


template<typename FloatTreeT, typename InterruptT>
PropagateSign<FloatTreeT, InterruptT>::PropagateSign(BoolLeafManager& signMaskLeafs,
    FloatTreeT& distTree, const BoolTreeT& intersectionTree, InterruptT *interrupter)
    : mOldSignMaskLeafs(signMaskLeafs)
    , mDistTree(distTree)
    , mIntersectionTree(intersectionTree)
    , mSignMaskTree(false)
    , mInterrupter(interrupter)
{
}


template<typename FloatTreeT, typename InterruptT>
PropagateSign<FloatTreeT, InterruptT>::PropagateSign(
    PropagateSign<FloatTreeT, InterruptT>& rhs, tbb::split)
    : mOldSignMaskLeafs(rhs.mOldSignMaskLeafs)
    , mDistTree(rhs.mDistTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mSignMaskTree(false)
    , mInterrupter(rhs.mInterrupter)
{
}


template<typename FloatTreeT, typename InterruptT>
void
PropagateSign<FloatTreeT, InterruptT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mOldSignMaskLeafs.getRange(), *this);
    else (*this)(mOldSignMaskLeafs.getRange());
}


template<typename FloatTreeT, typename InterruptT>
void
PropagateSign<FloatTreeT, InterruptT>::operator()(const tbb::blocked_range<size_t> &range)
{
    FloatAccessorT distAcc(mDistTree);
    BoolConstAccessorT intersectionAcc(mIntersectionTree);
    BoolAccessorT maskAcc(mSignMaskTree);

    std::deque<Coord> coordList;

    FloatValueT value;
    CoordBBox bbox;
    Coord& maxCoord = bbox.max();
    Coord& minCoord = bbox.min();
    Coord ijk, nijk;
    const int extent = BoolLeafT::DIM - 1;

    for (size_t n = range.begin(); n < range.end(); ++n) {
        BoolLeafT& oldMaskLeaf = mOldSignMaskLeafs.leaf(n);

        minCoord = oldMaskLeaf.origin();
        maxCoord[0] = minCoord[0] + extent;
        maxCoord[1] = minCoord[1] + extent;
        maxCoord[2] = minCoord[2] + extent;

        FloatLeafT& distLeaf = *distAcc.probeLeaf(minCoord);
        const BoolLeafT* intersectionLeaf = intersectionAcc.probeConstLeaf(minCoord);

        typename BoolLeafT::ValueOnCIter it = oldMaskLeaf.cbeginValueOn();
        for (; it; ++it) {
            coordList.push_back(it.getCoord());

            while (!coordList.empty()) {

                ijk = coordList.back();
                coordList.pop_back();

                FloatValueT& dist = const_cast<FloatValueT&>(distLeaf.getValue(ijk));
                if (dist < FloatValueT(0.0)) {
                    dist = -dist; // flip sign

                    for (size_t i = 0; i < 6; ++i) {
                        nijk = ijk + util::COORD_OFFSETS[i];
                        if (bbox.isInside(nijk)) {
                            if (intersectionLeaf && intersectionLeaf->isValueOn(nijk)) continue;

                            if (distLeaf.probeValue(nijk, value) && value < 0.0) {
                                coordList.push_back(nijk);
                            }

                        } else {
                            if(!intersectionAcc.isValueOn(nijk) &&
                                distAcc.probeValue(nijk, value) && value < 0.0) {
                                maskAcc.setValueOn(nijk);
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename FloatTreeT, typename InterruptT>
void
PropagateSign<FloatTreeT, InterruptT>::join(PropagateSign<FloatTreeT, InterruptT>& rhs)
{
    mSignMaskTree.merge(rhs.mSignMaskTree);
}


////////////////////////////////////////


// IntersectingVoxelSign
/// @brief TBB body object that traversers all intersecting voxels (defined by the
/// intersectingVoxelsGrid) and potentially flips their sign, by comparing the "closest point"
/// directions of outside-marked and non-intersecting neighboring voxels
template<typename FloatTreeT>
class IntersectingVoxelSign
{
public:
    typedef typename FloatTreeT::ValueType                              FloatValueT;
    typedef typename tree::ValueAccessor<FloatTreeT>                    FloatAccessorT;
    typedef typename FloatTreeT::template ValueConverter<Int32>::Type   IntTreeT;
    typedef typename tree::ValueAccessor<IntTreeT>                      IntAccessorT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type    BoolTreeT;
    typedef typename tree::ValueAccessor<BoolTreeT>                     BoolAccessorT;
    typedef tree::LeafManager<BoolTreeT>                                BoolLeafManager;

    IntersectingVoxelSign(
        const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList,
        FloatTreeT& distTree,
        IntTreeT& indexTree,
        BoolTreeT& intersectionTree,
        BoolLeafManager& leafs);

    ~IntersectingVoxelSign() {}

    void run(bool threaded = true);

    IntersectingVoxelSign(const IntersectingVoxelSign<FloatTreeT> &rhs);
    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    void operator=(const IntersectingVoxelSign<FloatTreeT>&) {}

    Vec3d getClosestPoint(const Coord& ijk, const Vec4I& prim) const;

    std::vector<Vec3s> const * const mPointList;
    std::vector<Vec4I> const * const mPolygonList;

    FloatTreeT& mDistTree;
    IntTreeT&   mIndexTree;
    BoolTreeT&  mIntersectionTree;

    BoolLeafManager& mLeafs;
};


template<typename FloatTreeT>
void
IntersectingVoxelSign<FloatTreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_for(mLeafs.getRange(), *this);
    else (*this)(mLeafs.getRange());
}


template<typename FloatTreeT>
IntersectingVoxelSign<FloatTreeT>::IntersectingVoxelSign(
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList,
    FloatTreeT& distTree,
    IntTreeT& indexTree,
    BoolTreeT& intersectionTree,
    BoolLeafManager& leafs)
    : mPointList(&pointList)
    , mPolygonList(&polygonList)
    , mDistTree(distTree)
    , mIndexTree(indexTree)
    , mIntersectionTree(intersectionTree)
    , mLeafs(leafs)
{
}


template<typename FloatTreeT>
IntersectingVoxelSign<FloatTreeT>::IntersectingVoxelSign(
    const IntersectingVoxelSign<FloatTreeT> &rhs)
    : mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mDistTree(rhs.mDistTree)
    , mIndexTree(rhs.mIndexTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mLeafs(rhs.mLeafs)
{
}


template<typename FloatTreeT>
void
IntersectingVoxelSign<FloatTreeT>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    Coord ijk, nijk;

    FloatAccessorT distAcc(mDistTree);
    BoolAccessorT maskAcc(mIntersectionTree);
    IntAccessorT idxAcc(mIndexTree);

    FloatValueT tmpValue;
    Vec3d cpt, center, dir1, dir2;

    typename BoolTreeT::LeafNodeType::ValueOnCIter iter;
    for (size_t n = range.begin(); n < range.end(); ++n) {
        iter = mLeafs.leaf(n).cbeginValueOn();
        for (; iter; ++iter) {

            ijk = iter.getCoord();

            FloatValueT value = distAcc.getValue(ijk);

            if (!(value < FloatValueT(0.0))) continue;

            center = Vec3d(ijk[0], ijk[1], ijk[2]);

            for (Int32 i = 0; i < 26; ++i) {
                nijk = ijk + util::COORD_OFFSETS[i];

                if (!maskAcc.isValueOn(nijk) && distAcc.probeValue(nijk, tmpValue)) {
                    if (tmpValue < FloatValueT(0.0)) continue;

                    const Vec4I& prim = (*mPolygonList)[idxAcc.getValue(nijk)];

                    cpt = getClosestPoint(nijk, prim);

                    dir1 = center - cpt;
                    dir1.normalize();

                    dir2 = Vec3d(nijk[0], nijk[1], nijk[2]) - cpt;
                    dir2.normalize();

                    if (dir2.dot(dir1) > 0.0) {
                        distAcc.setValue(ijk, -value);
                        break;
                    }
                }
            }
        }
    }
}


template<typename FloatTreeT>
Vec3d
IntersectingVoxelSign<FloatTreeT>::getClosestPoint(const Coord& ijk, const Vec4I& prim) const
{
    Vec3d voxelCenter(ijk[0], ijk[1], ijk[2]);

    // Evaluate first triangle
    const Vec3d a((*mPointList)[prim[0]]);
    const Vec3d b((*mPointList)[prim[1]]);
    const Vec3d c((*mPointList)[prim[2]]);

    Vec3d uvw;
    Vec3d cpt1 = closestPointOnTriangleToPoint(a, c, b, voxelCenter, uvw);

    // Evaluate second triangle if quad.
    if (prim[3] != util::INVALID_IDX) {

        Vec3d diff1 = voxelCenter - cpt1;

        const Vec3d d((*mPointList)[prim[3]]);

        Vec3d cpt2 = closestPointOnTriangleToPoint(a, d, c, voxelCenter, uvw);
        Vec3d diff2 = voxelCenter - cpt2;

        if (diff2.lengthSqr() < diff1.lengthSqr()) {
            return cpt2;
        }
    }

    return cpt1;
}


////////////////////////////////////////


// IntersectingVoxelCleaner
/// @brief TBB body object that removes intersecting voxels that were set via
/// voxelization of self-intersecting parts of a mesh
template<typename FloatTreeT>
class IntersectingVoxelCleaner
{
public:
    typedef typename FloatTreeT::ValueType FloatValueT;
    typedef typename tree::ValueAccessor<FloatTreeT> DistAccessorT;
    typedef typename FloatTreeT::LeafNodeType DistLeafT;
    typedef typename FloatTreeT::template ValueConverter<Int32>::Type IntTreeT;
    typedef typename tree::ValueAccessor<IntTreeT> IntAccessorT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename tree::ValueAccessor<BoolTreeT> BoolAccessorT;
    typedef typename BoolTreeT::LeafNodeType BoolLeafT;
    typedef tree::LeafManager<BoolTreeT> BoolLeafManager;

    IntersectingVoxelCleaner(FloatTreeT& distTree, IntTreeT& indexTree,
        BoolTreeT& intersectionTree, BoolLeafManager& leafs);

    ~IntersectingVoxelCleaner() {}

    void run(bool threaded = true);

    IntersectingVoxelCleaner(const IntersectingVoxelCleaner<FloatTreeT> &rhs);
    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    void operator=(const IntersectingVoxelCleaner<FloatTreeT>&) {}

    FloatTreeT& mDistTree;
    IntTreeT& mIndexTree;
    BoolTreeT& mIntersectionTree;
    BoolLeafManager& mLeafs;
};


template<typename FloatTreeT>
void
IntersectingVoxelCleaner<FloatTreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_for(mLeafs.getRange(), *this);
    else (*this)(mLeafs.getRange());

    tools::pruneInactive(mIntersectionTree, threaded);
}


template<typename FloatTreeT>
IntersectingVoxelCleaner<FloatTreeT>::IntersectingVoxelCleaner(
    FloatTreeT& distTree,
    IntTreeT& indexTree,
    BoolTreeT& intersectionTree,
    BoolLeafManager& leafs)
    : mDistTree(distTree)
    , mIndexTree(indexTree)
    , mIntersectionTree(intersectionTree)
    , mLeafs(leafs)
{
}


template<typename FloatTreeT>
IntersectingVoxelCleaner<FloatTreeT>::IntersectingVoxelCleaner(
    const IntersectingVoxelCleaner<FloatTreeT>& rhs)
    : mDistTree(rhs.mDistTree)
    , mIndexTree(rhs.mIndexTree)
    , mIntersectionTree(rhs.mIntersectionTree)
    , mLeafs(rhs.mLeafs)
{
}


template<typename FloatTreeT>
void
IntersectingVoxelCleaner<FloatTreeT>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    Coord ijk, m_ijk;
    bool turnOff;
    FloatValueT value;
    Index offset;

    typename BoolLeafT::ValueOnCIter iter;

    IntAccessorT      indexAcc(mIndexTree);
    DistAccessorT       distAcc(mDistTree);
    BoolAccessorT    maskAcc(mIntersectionTree);

    for (size_t n = range.begin(); n < range.end(); ++n) {

        BoolLeafT& maskLeaf = mLeafs.leaf(n);

        ijk = maskLeaf.origin();

        DistLeafT * distLeaf = distAcc.probeLeaf(ijk);
        if (distLeaf) {
            iter = maskLeaf.cbeginValueOn();
            for (; iter; ++iter) {

                offset = iter.pos();

                if(distLeaf->getValue(offset) > 0.0) continue;

                ijk = iter.getCoord();
                turnOff = true;
                for (Int32 m = 0; m < 26; ++m) {
                    m_ijk = ijk + util::COORD_OFFSETS[m];
                    if (distAcc.probeValue(m_ijk, value)) {
                        if (value > 0.0) {
                            turnOff = false;
                            break;
                        }
                    }
                }

                if (turnOff) {
                    maskLeaf.setValueOff(offset);
                    distLeaf->setValueOn(offset, FloatValueT(-0.86602540378443861));
                }
            }
        }
    }
}


////////////////////////////////////////


// ShellVoxelCleaner
/// @brief TBB body object that removes non-intersecting voxels that where set by rasterizing
/// self-intersecting parts of the mesh.
template<typename FloatTreeT>
class ShellVoxelCleaner
{
public:
    typedef typename FloatTreeT::ValueType FloatValueT;
    typedef typename tree::ValueAccessor<FloatTreeT> DistAccessorT;
    typedef typename FloatTreeT::LeafNodeType DistLeafT;
    typedef tree::LeafManager<FloatTreeT> DistArrayT;
    typedef typename FloatTreeT::template ValueConverter<Int32>::Type IntTreeT;
    typedef typename tree::ValueAccessor<IntTreeT> IntAccessorT;
    typedef typename IntTreeT::LeafNodeType IntLeafT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type BoolTreeT;
    typedef typename tree::ValueAccessor<BoolTreeT> BoolAccessorT;
    typedef typename BoolTreeT::LeafNodeType BoolLeafT;

    ShellVoxelCleaner(FloatTreeT& distTree, DistArrayT& leafs, IntTreeT& indexTree,
        BoolTreeT& intersectionTree);

    ~ShellVoxelCleaner() {}

    void run(bool threaded = true);

    ShellVoxelCleaner(const ShellVoxelCleaner<FloatTreeT> &rhs);
    void operator()(const tbb::blocked_range<size_t>&) const;

private:
    void operator=(const ShellVoxelCleaner<FloatTreeT>&) {}

    FloatTreeT& mDistTree;
    DistArrayT& mLeafs;
    IntTreeT& mIndexTree;
    BoolTreeT& mIntersectionTree;
};


template<typename FloatTreeT>
void
ShellVoxelCleaner<FloatTreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_for(mLeafs.getRange(), *this);
    else (*this)(mLeafs.getRange());

    tools::pruneInactive(mDistTree,  threaded);
    tools::pruneInactive(mIndexTree, threaded);
}


template<typename FloatTreeT>
ShellVoxelCleaner<FloatTreeT>::ShellVoxelCleaner(
    FloatTreeT& distTree,
    DistArrayT& leafs,
    IntTreeT& indexTree,
    BoolTreeT& intersectionTree)
    : mDistTree(distTree)
    , mLeafs(leafs)
    , mIndexTree(indexTree)
    , mIntersectionTree(intersectionTree)
{
}


template<typename FloatTreeT>
ShellVoxelCleaner<FloatTreeT>::ShellVoxelCleaner(
    const ShellVoxelCleaner<FloatTreeT> &rhs)
    : mDistTree(rhs.mDistTree)
    , mLeafs(rhs.mLeafs)
    , mIndexTree(rhs.mIndexTree)
    , mIntersectionTree(rhs.mIntersectionTree)
{
}


template<typename FloatTreeT>
void
ShellVoxelCleaner<FloatTreeT>::operator()(
    const tbb::blocked_range<size_t>& range) const
{
    Coord ijk, m_ijk;
    bool turnOff;
    FloatValueT value;
    Index offset;

    typename DistLeafT::ValueOnCIter iter;
    const FloatValueT distBG = mDistTree.background();
    const Int32 indexBG = mIntersectionTree.background();

    IntAccessorT      indexAcc(mIndexTree);
    DistAccessorT       distAcc(mDistTree);
    BoolAccessorT    maskAcc(mIntersectionTree);


    for (size_t n = range.begin(); n < range.end(); ++n) {

        DistLeafT& distLeaf = mLeafs.leaf(n);

        ijk = distLeaf.origin();

        const BoolLeafT* maskLeaf = maskAcc.probeConstLeaf(ijk);
        IntLeafT& indexLeaf = *indexAcc.probeLeaf(ijk);

        iter = distLeaf.cbeginValueOn();
        for (; iter; ++iter) {

            value = iter.getValue();
            if(value > 0.0) continue;

            offset = iter.pos();
            if (maskLeaf && maskLeaf->isValueOn(offset)) continue;

            ijk = iter.getCoord();
            turnOff = true;
            for (Int32 m = 0; m < 26; ++m) {
                m_ijk = ijk + util::COORD_OFFSETS[m];
                if (maskAcc.isValueOn(m_ijk)) {
                    turnOff = false;
                    break;
                }
            }

            if (turnOff) {
                distLeaf.setValueOff(offset, distBG);
                indexLeaf.setValueOff(offset, indexBG);
            }
        }
    }
}


////////////////////////////////////////


template<typename TreeType>
struct CopyActiveVoxelsOp
{
    typedef typename tree::ValueAccessor<TreeType> AccessorT;

    CopyActiveVoxelsOp(TreeType& tree) : mAcc(tree) { }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t) const
    {
        LeafNodeType* rhsLeaf = const_cast<LeafNodeType*>(mAcc.probeLeaf(leaf.origin()));
        typename LeafNodeType::ValueOnIter iter = leaf.beginValueOn();
        for (; iter; ++iter) {
            rhsLeaf->setValueOnly(iter.pos(), iter.getValue());
        }
    }

private:
    AccessorT mAcc;
};


// ExpandNB
/// @brief TBB body object to expand the level set narrow band
/// @note The interior and exterior widths should be in world space units and squared.
template<typename FloatTreeT>
class ExpandNB
{
public:
    typedef typename FloatTreeT::ValueType                              FloatValueT;
    typedef typename FloatTreeT::LeafNodeType                           FloatLeafT;
    typedef typename tree::ValueAccessor<FloatTreeT>                    FloatAccessorT;
    typedef typename FloatTreeT::template ValueConverter<Int32>::Type   IntTreeT;
    typedef typename IntTreeT::LeafNodeType                             IntLeafT;
    typedef typename tree::ValueAccessor<IntTreeT>                      IntAccessorT;
    typedef typename FloatTreeT::template ValueConverter<bool>::Type    BoolTreeT;
    typedef typename BoolTreeT::LeafNodeType                            BoolLeafT;
    typedef tree::LeafManager<BoolTreeT>                                BoolLeafManager;
    typedef typename tree::ValueAccessor<BoolTreeT>                     BoolAccessorT;

    ExpandNB(BoolLeafManager& leafs,
        FloatTreeT& distTree, IntTreeT& indexTree, BoolTreeT& maskTree,
        FloatValueT exteriorBandWidth, FloatValueT interiorBandWidth, FloatValueT voxelSize,
        const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList);

    void run(bool threaded = true);

    void operator()(const tbb::blocked_range<size_t>&);
    void join(ExpandNB<FloatTreeT>&);
    ExpandNB(const ExpandNB<FloatTreeT>&, tbb::split);
    ~ExpandNB() {}

private:
    void operator=(const ExpandNB<FloatTreeT>&) {}

    double evalVoxelDist(const Coord&, FloatAccessorT&, IntAccessorT&,
        BoolAccessorT&, std::vector<Int32>&, Int32&) const;

    double evalVoxelDist(const Coord&, FloatLeafT&, IntLeafT&,
        BoolLeafT&, std::vector<Int32>&, Int32&) const;

    double closestPrimDist(const Coord&, std::vector<Int32>&, Int32&) const;

    BoolLeafManager& mMaskLeafs;

    FloatTreeT& mDistTree;
    IntTreeT& mIndexTree;
    BoolTreeT& mMaskTree;

    const FloatValueT mExteriorBandWidth, mInteriorBandWidth, mVoxelSize;
    const std::vector<Vec3s>& mPointList;
    const std::vector<Vec4I>& mPolygonList;

    FloatTreeT mNewDistTree;
    IntTreeT mNewIndexTree;
    BoolTreeT mNewMaskTree;
};


template<typename FloatTreeT>
ExpandNB<FloatTreeT>::ExpandNB(
    BoolLeafManager& leafs,
    FloatTreeT& distTree,
    IntTreeT& indexTree,
    BoolTreeT& maskTree,
    FloatValueT exteriorBandWidth,
    FloatValueT interiorBandWidth,
    FloatValueT voxelSize,
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList)
    : mMaskLeafs(leafs)
    , mDistTree(distTree)
    , mIndexTree(indexTree)
    , mMaskTree(maskTree)
    , mExteriorBandWidth(exteriorBandWidth)
    , mInteriorBandWidth(interiorBandWidth)
    , mVoxelSize(voxelSize)
    , mPointList(pointList)
    , mPolygonList(polygonList)
    , mNewDistTree(std::numeric_limits<FloatValueT>::max())
    , mNewIndexTree(Int32(util::INVALID_IDX))
    , mNewMaskTree(false)
{
}


template<typename FloatTreeT>
ExpandNB<FloatTreeT>::ExpandNB(const ExpandNB<FloatTreeT>& rhs, tbb::split)
    : mMaskLeafs(rhs.mMaskLeafs)
    , mDistTree(rhs.mDistTree)
    , mIndexTree(rhs.mIndexTree)
    , mMaskTree(rhs.mMaskTree)
    , mExteriorBandWidth(rhs.mExteriorBandWidth)
    , mInteriorBandWidth(rhs.mInteriorBandWidth)
    , mVoxelSize(rhs.mVoxelSize)
    , mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mNewDistTree(std::numeric_limits<FloatValueT>::max())
    , mNewIndexTree(Int32(util::INVALID_IDX))
    , mNewMaskTree(false)
{
}


template<typename FloatTreeT>
void
ExpandNB<FloatTreeT>::run(bool threaded)
{
    if (threaded) tbb::parallel_reduce(mMaskLeafs.getRange(), *this);
    else (*this)(mMaskLeafs.getRange());

    // Copy only the active voxels (tree::merge does branch stealing
    // which also moves indicative values).
    mDistTree.topologyUnion(mNewDistTree);
    tree::LeafManager<FloatTreeT> leafs(mNewDistTree);
    leafs.foreach(CopyActiveVoxelsOp<FloatTreeT>(mDistTree));

    mIndexTree.merge(mNewIndexTree);

    mMaskTree.clear();
    mMaskTree.merge(mNewMaskTree);

}


template<typename FloatTreeT>
void
ExpandNB<FloatTreeT>::operator()(const tbb::blocked_range<size_t>& range)
{
    Coord ijk;
    Int32 closestPrim = 0;
    Index pos = 0;
    FloatValueT distance;
    bool inside;

    FloatAccessorT newDistAcc(mNewDistTree);
    IntAccessorT newIndexAcc(mNewIndexTree);
    BoolAccessorT newMaskAcc(mNewMaskTree);

    FloatAccessorT distAcc(mDistTree);
    IntAccessorT indexAcc(mIndexTree);
    BoolAccessorT maskAcc(mMaskTree);

    CoordBBox bbox;
    std::vector<Int32> primitives(18);

    for (size_t n = range.begin(); n < range.end(); ++n) {

        BoolLeafT& maskLeaf = mMaskLeafs.leaf(n);

        if (maskLeaf.isEmpty()) continue;

        ijk = maskLeaf.origin();

        FloatLeafT* distLeafPt = distAcc.probeLeaf(ijk);

        if (!distLeafPt) {
            distLeafPt = new FloatLeafT(ijk, distAcc.getValue(ijk));
            newDistAcc.addLeaf(distLeafPt);
        }

        IntLeafT* indexLeafPt = indexAcc.probeLeaf(ijk);
        if (!indexLeafPt) indexLeafPt =  newIndexAcc.touchLeaf(ijk);

        bbox = maskLeaf.getNodeBoundingBox();
        bbox.expand(-1);

        typename BoolLeafT::ValueOnIter iter = maskLeaf.beginValueOn();
        for (; iter; ++iter) {

            ijk = iter.getCoord();

            if (bbox.isInside(ijk)) {
                distance = FloatValueT(evalVoxelDist(ijk, *distLeafPt, *indexLeafPt, maskLeaf,
                    primitives, closestPrim));
            } else {
                distance = FloatValueT(evalVoxelDist(ijk, distAcc, indexAcc, maskAcc,
                    primitives, closestPrim));
            }

            pos = iter.pos();

            inside = distLeafPt->getValue(pos) < FloatValueT(0.0);

            if (!inside && distance < mExteriorBandWidth) {
                distLeafPt->setValueOn(pos, distance);
                indexLeafPt->setValueOn(pos, closestPrim);
            } else if (inside && distance < mInteriorBandWidth) {
                distLeafPt->setValueOn(pos, -distance);
                indexLeafPt->setValueOn(pos, closestPrim);
            } else {
                continue;
            }

            for (Int32 i = 0; i < 6; ++i) {
                newMaskAcc.setValueOn(ijk + util::COORD_OFFSETS[i]);
            }
        }
    }
}


template<typename FloatTreeT>
double
ExpandNB<FloatTreeT>::evalVoxelDist(
    const Coord& ijk,
    FloatAccessorT& distAcc,
    IntAccessorT& indexAcc,
    BoolAccessorT& maskAcc,
    std::vector<Int32>& prims,
    Int32& closestPrim) const
{
    FloatValueT tmpDist, minDist = std::numeric_limits<FloatValueT>::max();
    prims.clear();

    // Collect primitive indices from active neighbors and min distance.
    Coord n_ijk;
    for (Int32 n = 0; n < 18; ++n) {
        n_ijk = ijk + util::COORD_OFFSETS[n];
        if (!maskAcc.isValueOn(n_ijk) && distAcc.probeValue(n_ijk, tmpDist)) {
            prims.push_back(indexAcc.getValue(n_ijk));
            tmpDist = std::abs(tmpDist);
            if (tmpDist < minDist) minDist = tmpDist;
        }
    }

    // Calc. this voxels distance to the closest primitive.
    tmpDist = FloatValueT(closestPrimDist(ijk, prims, closestPrim));

    // Forces the gradient to be monotonic for non-manifold
    // polygonal models with self-intersections.
    return tmpDist > minDist ? tmpDist : minDist + mVoxelSize;
}


// Leaf specialized version.
template<typename FloatTreeT>
double
ExpandNB<FloatTreeT>::evalVoxelDist(
    const Coord& ijk,
    FloatLeafT& distLeaf,
    IntLeafT& indexLeaf,
    BoolLeafT& maskLeaf,
    std::vector<Int32>& prims,
    Int32& closestPrim) const
{
    FloatValueT tmpDist, minDist = std::numeric_limits<FloatValueT>::max();
    prims.clear();

    Index pos;
    for (Int32 n = 0; n < 18; ++n) {
        pos = FloatLeafT::coordToOffset(ijk + util::COORD_OFFSETS[n]);
        if (!maskLeaf.isValueOn(pos) && distLeaf.probeValue(pos, tmpDist)) {
            prims.push_back(indexLeaf.getValue(pos));
            tmpDist = std::abs(tmpDist);
            if (tmpDist < minDist) minDist = tmpDist;
        }
    }

    tmpDist = FloatValueT(closestPrimDist(ijk, prims, closestPrim));
    return tmpDist > minDist ? tmpDist : minDist + mVoxelSize;
}


template<typename FloatTreeT>
double
ExpandNB<FloatTreeT>::closestPrimDist(const Coord& ijk,
    std::vector<Int32>& prims, Int32& closestPrim) const
{
    std::sort(prims.begin(), prims.end());

    Int32 lastPrim = -1;
    Vec3d uvw, voxelCenter(ijk[0], ijk[1], ijk[2]);
    double primDist, tmpDist, dist = std::numeric_limits<double>::max();

    for (size_t n = 0, N = prims.size(); n < N; ++n) {
        if (prims[n] == lastPrim) continue;

        lastPrim = prims[n];

        const Vec4I& verts = mPolygonList[lastPrim];

        // Evaluate first triangle
        const Vec3d a(mPointList[verts[0]]);
        const Vec3d b(mPointList[verts[1]]);
        const Vec3d c(mPointList[verts[2]]);

        primDist = (voxelCenter -
            closestPointOnTriangleToPoint(a, c, b, voxelCenter, uvw)).lengthSqr();

        // Split-up quad into a second triangle and calac distance.
        if (util::INVALID_IDX != verts[3]) {
            const Vec3d d(mPointList[verts[3]]);

            tmpDist = (voxelCenter -
                closestPointOnTriangleToPoint(a, d, c, voxelCenter, uvw)).lengthSqr();

            if (tmpDist < primDist) primDist = tmpDist;
        }

        if (primDist < dist) {
            dist = primDist;
            closestPrim = lastPrim;
        }
    }

    return std::sqrt(dist) * double(mVoxelSize);
}


template<typename FloatTreeT>
void
ExpandNB<FloatTreeT>::join(ExpandNB<FloatTreeT>& rhs)
{
    mNewDistTree.merge(rhs.mNewDistTree);
    mNewIndexTree.merge(rhs.mNewIndexTree);
    mNewMaskTree.merge(rhs.mNewMaskTree);
}


////////////////////////////////////////


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
            ValueType& val = const_cast<ValueType&>(iter.getValue());
            const bool inside = val < ValueType(0.0);

            if (inside && !(val > -mInBandWidth)) {
                val = -mInBandWidth;
                iter.setValueOff();
            } else if (!inside && !(val < mExBandWidth)) {
                val = mExBandWidth;
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

            const ValueType phi0 = iter.getValue();
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
        Index offset;

        for (; iter; ++iter) {
            offset = iter.pos();
            leaf.setValueOnly(offset, buffer.getValue(offset));
        }
    }

private:
    LeafManagerType& mLeafs;
    const size_t mBufferIndex;
};


template<typename TreeType>
struct LeafTopologyDiffOp
{
    typedef typename tree::ValueAccessor<TreeType> AccessorT;
    typedef typename TreeType::LeafNodeType LeafNodeT;

    LeafTopologyDiffOp(TreeType& tree) : mAcc(tree) { }

    template <typename LeafNodeType>
    void operator()(LeafNodeType &leaf, size_t) const
    {
        const LeafNodeT* rhsLeaf = mAcc.probeConstLeaf(leaf.origin());
        if (rhsLeaf) leaf.topologyDifference(*rhsLeaf, false);
    }

private:
    AccessorT mAcc;
};

} // internal namespace


////////////////////////////////////////


// MeshToVolume

template<typename FloatGridT, typename InterruptT>
MeshToVolume<FloatGridT, InterruptT>::MeshToVolume(
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


template<typename FloatGridT, typename InterruptT>
void
MeshToVolume<FloatGridT, InterruptT>::clear()
{
    mDistGrid = FloatGridT::create(std::numeric_limits<FloatValueT>::max());
    mIndexGrid = IntGridT::create(Int32(util::INVALID_IDX));
    mIntersectingVoxelsGrid = BoolGridT::create(false);
}


template<typename FloatGridT, typename InterruptT>
inline void
MeshToVolume<FloatGridT, InterruptT>::convertToLevelSet(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    FloatValueT exBandWidth, FloatValueT inBandWidth)
{
    // The narrow band width is exclusive, the shortest valid distance has to be > 1 voxel
    exBandWidth = std::max(internal::Tolerance<FloatValueT>::minNarrowBandWidth(), exBandWidth);
    inBandWidth = std::max(internal::Tolerance<FloatValueT>::minNarrowBandWidth(), inBandWidth);
    const FloatValueT vs = FloatValueT(mTransform->voxelSize()[0]);

    // Convert from index space units to world-space units. To fill the
    // interior, inBandWidth is passed FLOAT_MAX. Don't multiply with vs if so.
    exBandWidth *= vs;
    if (inBandWidth < std::numeric_limits<FloatValueT>::max()) {
        inBandWidth *= vs;
    }

    doConvert(pointList, polygonList, exBandWidth, inBandWidth);
    mDistGrid->setGridClass(GRID_LEVEL_SET);
}


template<typename FloatGridT, typename InterruptT>
inline void
MeshToVolume<FloatGridT, InterruptT>::convertToUnsignedDistanceField(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    FloatValueT exBandWidth)
{
    // The narrow band width is exclusive, the shortest valid distance has to be > 1 voxel
    exBandWidth = std::max(internal::Tolerance<FloatValueT>::minNarrowBandWidth(), exBandWidth);
    const FloatValueT vs = FloatValueT(mTransform->voxelSize()[0]);
    doConvert(pointList, polygonList, vs * exBandWidth, 0.0, true);
    mDistGrid->setGridClass(GRID_UNKNOWN);
}


template<typename FloatGridT, typename InterruptT>
void
MeshToVolume<FloatGridT, InterruptT>::doConvert(
    const std::vector<Vec3s>& pointList, const std::vector<Vec4I>& polygonList,
    FloatValueT exBandWidth, FloatValueT inBandWidth, bool unsignedDistField)
{
    mDistGrid->setTransform(mTransform);
    mIndexGrid->setTransform(mTransform);
    const bool rawData = OUTPUT_RAW_DATA & mConversionFlags;

    // Note that inBandWidth is allowed to be infinite when filling the interior.
    if (!boost::math::isfinite(exBandWidth) || boost::math::isnan(inBandWidth)) {
        std::stringstream msg;
        msg << "Illegal narrow band width: exterior = " << exBandWidth
            << ", interior = " << inBandWidth;
        OPENVDB_THROW(ValueError, msg.str());
    }


    // The progress estimates given to the interrupter are based on the
    // observed average time for each stage and therefore not alway
    // accurate. The goal is to give some progression feedback to the user.

    if (wasInterrupted(1)) return;

    // Voxelize mesh
    {
        internal::MeshVoxelizer<FloatTreeT, InterruptT>
            voxelizer(pointList, polygonList, mInterrupter);

        voxelizer.run();

        if (wasInterrupted(18)) return;

        mDistGrid->tree().merge(voxelizer.sqrDistTree());
        mIndexGrid->tree().merge(voxelizer.primIndexTree());
        mIntersectingVoxelsGrid->tree().merge(voxelizer.intersectionTree());
    }

    if (!unsignedDistField) {
        // Determine the inside/outside state for the narrow band of voxels.
        {
            // Slices up the volume and label the exterior contour of each slice in parallel.
            internal::ContourTracer<FloatTreeT, InterruptT> trace(
                mDistGrid->tree(), mIntersectingVoxelsGrid->tree(), mInterrupter);
            for (int i = 0; i < mSignSweeps; ++i) {

                if (wasInterrupted(19)) return;

                trace.run();

                if (wasInterrupted(24)) return;

                // Propagate sign information between the slices.
                BoolTreeT signMaskTree(false);
                {
                    tree::LeafManager<FloatTreeT> leafs(mDistGrid->tree());
                    internal::SignMask<FloatTreeT, InterruptT> signMaskOp(leafs,
                        mDistGrid->tree(), mIntersectingVoxelsGrid->tree(), mInterrupter);
                    signMaskOp.run();
                    signMaskTree.merge(signMaskOp.signMaskTree());
                }

                if (wasInterrupted(25)) return;

                while (true) {
                    tree::LeafManager<BoolTreeT> leafs(signMaskTree);
                    if(leafs.leafCount() == 0) break;

                    internal::PropagateSign<FloatTreeT, InterruptT> sign(leafs,
                        mDistGrid->tree(), mIntersectingVoxelsGrid->tree(), mInterrupter);

                    sign.run();

                    signMaskTree.clear();
                    signMaskTree.merge(sign.signMaskTree());
                }
            }
        }


        if (wasInterrupted(28)) return;
        {
            tree::LeafManager<BoolTreeT> leafs(mIntersectingVoxelsGrid->tree());

            // Determine the sign of the mesh intersecting voxels.
            internal::IntersectingVoxelSign<FloatTreeT> sign(pointList, polygonList,
                mDistGrid->tree(), mIndexGrid->tree(), mIntersectingVoxelsGrid->tree(), leafs);

            sign.run();

            if (wasInterrupted(34)) return;

            // Remove mesh intersecting voxels that where set by rasterizing
            // self-intersecting portions of the mesh.
            internal::IntersectingVoxelCleaner<FloatTreeT> cleaner(mDistGrid->tree(),
                mIndexGrid->tree(), mIntersectingVoxelsGrid->tree(), leafs);
            cleaner.run();
        }

        // Remove shell voxels that where set by rasterizing
        // self-intersecting portions of the mesh.
        {
            tree::LeafManager<FloatTreeT> leafs(mDistGrid->tree());

            internal::ShellVoxelCleaner<FloatTreeT> cleaner(mDistGrid->tree(),
                leafs, mIndexGrid->tree(), mIntersectingVoxelsGrid->tree());

            cleaner.run();
        }

        if (wasInterrupted(38)) return;

    } else { // if unsigned dist. field
        inBandWidth = FloatValueT(0.0);
    }

    if (mDistGrid->activeVoxelCount() == 0) {
        tools::changeBackground(mDistGrid->tree(), exBandWidth);
        return;
    }

    mIntersectingVoxelsGrid->clear();
    const FloatValueT voxelSize = FloatValueT(mTransform->voxelSize()[0]);

    { // Transform values (world space scaling etc.)
        tree::LeafManager<FloatTreeT> leafs(mDistGrid->tree());
        leafs.foreach(internal::SqrtAndScaleOp<FloatValueT>(voxelSize, unsignedDistField));
    }

    if (wasInterrupted(40)) return;

    if (!unsignedDistField) { // Propagate sign information to inactive values.
        mDistGrid->tree().root().setBackground(exBandWidth, /*updateChildNodes=*/false);
        tools::signedFloodFillWithValues(mDistGrid->tree(), exBandWidth, -inBandWidth);
    }

    if (wasInterrupted(46)) return;

    // Narrow-band dilation
    const FloatValueT minWidth = FloatValueT(voxelSize * 2.0);
    if (inBandWidth > minWidth || exBandWidth > minWidth) {

        // Create the initial voxel mask.
        BoolTreeT maskTree(false);
        maskTree.topologyUnion(mDistGrid->tree());

        if (wasInterrupted(48)) return;

        internal::LeafTopologyDiffOp<FloatTreeT> diffOp(mDistGrid->tree());
        openvdb::tools::dilateVoxels(maskTree);

        unsigned maxIterations = std::numeric_limits<unsigned>::max();
        float progress = 48, step = 0.0;
        // progress estimation..
        double estimated =
            2.0 * std::ceil((std::max(inBandWidth, exBandWidth) - minWidth) / voxelSize);
        if (estimated < double(maxIterations)) {
            maxIterations = unsigned(estimated);
            step = 42.f / float(maxIterations);
        }

        unsigned count = 0;
        while (true) {

            if (wasInterrupted(int(progress))) return;

            tree::LeafManager<BoolTreeT> leafs(maskTree);

            if (leafs.leafCount() == 0) break;

            leafs.foreach(diffOp);

            internal::ExpandNB<FloatTreeT> expand(
                leafs, mDistGrid->tree(), mIndexGrid->tree(), maskTree,
                exBandWidth, inBandWidth, voxelSize, pointList, polygonList);

            expand.run();

            if ((++count) >= maxIterations) break;
            progress += step;
        }
    }

    if (!bool(GENERATE_PRIM_INDEX_GRID & mConversionFlags)) mIndexGrid->clear();

    if (wasInterrupted(80)) return;

    // Renormalize distances to smooth out bumps caused by self-intersecting
    // and overlapping portions of the mesh and renormalize the level set.
    if (!unsignedDistField && !rawData) {

        tools::pruneLevelSet(mDistGrid->tree(), exBandWidth, -inBandWidth);

        tree::LeafManager<FloatTreeT> leafs(mDistGrid->tree(), 1);

        const FloatValueT offset = FloatValueT(0.8 * voxelSize);
        if (wasInterrupted(82)) return;

        internal::OffsetOp<FloatValueT> offsetOp(-offset);

        leafs.foreach(offsetOp);

        if (wasInterrupted(84)) return;

        leafs.foreach(internal::RenormOp<FloatGridT, FloatValueT>(*mDistGrid, leafs, voxelSize));

        leafs.foreach(internal::MinOp<FloatTreeT, FloatValueT>(leafs));

        if (wasInterrupted(95)) return;

        offsetOp.resetOffset(offset - internal::Tolerance<FloatValueT>::epsilon());
        leafs.foreach(offsetOp);
    }

    if (wasInterrupted(98)) return;

    const FloatValueT minTrimWidth = FloatValueT(voxelSize * 4.0);
    if (inBandWidth < minTrimWidth || exBandWidth < minTrimWidth) {

        // If the narrow band was not expanded, we might need to trim off
        // some of the active voxels in order to respect the narrow band limits.
        // (The mesh voxelization step generates some extra 'shell' voxels)

        tree::LeafManager<FloatTreeT> leafs(mDistGrid->tree());
        leafs.foreach(internal::TrimOp<FloatValueT>(
            exBandWidth, unsignedDistField ? exBandWidth : inBandWidth));

        tools::pruneLevelSet(mDistGrid->tree(), exBandWidth, unsignedDistField ? -exBandWidth : -inBandWidth);
    }
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
        ptnXForm.run();
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
    bool /*unsignedDistanceField*/ = false)
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


////////////////////////////////////////////////////////////////////////////////


// Required by several of the tree nodes
inline std::ostream&
operator<<(std::ostream& ostr, const MeshToVoxelEdgeData::EdgeData& rhs)
{
    ostr << "{[ " << rhs.mXPrim << ", " << rhs.mXDist << "]";
    ostr << " [ " << rhs.mYPrim << ", " << rhs.mYDist << "]";
    ostr << " [ " << rhs.mZPrim << ", " << rhs.mZDist << "]}";
    return ostr;
}

// Required by math::Abs
inline MeshToVoxelEdgeData::EdgeData
Abs(const MeshToVoxelEdgeData::EdgeData& x)
{
    return x;
}


////////////////////////////////////////


class MeshToVoxelEdgeData::GenEdgeData
{
public:

    GenEdgeData(
        const std::vector<Vec3s>& pointList,
        const std::vector<Vec4I>& polygonList);

    void run(bool threaded = true);

    GenEdgeData(GenEdgeData& rhs, tbb::split);
    inline void operator() (const tbb::blocked_range<size_t> &range);
    inline void join(GenEdgeData& rhs);

    inline TreeType& tree() { return mTree; }

private:
    void operator=(const GenEdgeData&) {}

    struct Primitive { Vec3d a, b, c, d; Int32 index; };

    template<bool IsQuad>
    inline void voxelize(const Primitive&);

    template<bool IsQuad>
    inline bool evalPrimitive(const Coord&, const Primitive&);

    inline bool rayTriangleIntersection( const Vec3d& origin, const Vec3d& dir,
        const Vec3d& a, const Vec3d& b, const Vec3d& c, double& t);


    TreeType mTree;
    Accessor mAccessor;

    const std::vector<Vec3s>& mPointList;
    const std::vector<Vec4I>& mPolygonList;

    // Used internally for acceleration
    typedef TreeType::ValueConverter<Int32>::Type IntTreeT;
    IntTreeT mLastPrimTree;
    tree::ValueAccessor<IntTreeT> mLastPrimAccessor;
}; // class MeshToVoxelEdgeData::GenEdgeData


inline
MeshToVoxelEdgeData::GenEdgeData::GenEdgeData(
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList)
    : mTree(EdgeData())
    , mAccessor(mTree)
    , mPointList(pointList)
    , mPolygonList(polygonList)
    , mLastPrimTree(Int32(util::INVALID_IDX))
    , mLastPrimAccessor(mLastPrimTree)
{
}


inline
MeshToVoxelEdgeData::GenEdgeData::GenEdgeData(GenEdgeData& rhs, tbb::split)
    : mTree(EdgeData())
    , mAccessor(mTree)
    , mPointList(rhs.mPointList)
    , mPolygonList(rhs.mPolygonList)
    , mLastPrimTree(Int32(util::INVALID_IDX))
    , mLastPrimAccessor(mLastPrimTree)
{
}


inline void
MeshToVoxelEdgeData::GenEdgeData::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, mPolygonList.size()), *this);
    } else {
        (*this)(tbb::blocked_range<size_t>(0, mPolygonList.size()));
    }
}


inline void
MeshToVoxelEdgeData::GenEdgeData::join(GenEdgeData& rhs)
{
    typedef TreeType::RootNodeType       RootNodeType;
    typedef RootNodeType::NodeChainType  NodeChainType;
    BOOST_STATIC_ASSERT(boost::mpl::size<NodeChainType>::value > 1);
    typedef boost::mpl::at<NodeChainType, boost::mpl::int_<1> >::type InternalNodeType;

    Coord ijk;
    Index offset;

    rhs.mTree.clearAllAccessors();

    TreeType::LeafIter leafIt = rhs.mTree.beginLeaf();
    for ( ; leafIt; ++leafIt) {
        ijk = leafIt->origin();

        TreeType::LeafNodeType* lhsLeafPt = mTree.probeLeaf(ijk);

        if (!lhsLeafPt) {

            mAccessor.addLeaf(rhs.mAccessor.probeLeaf(ijk));
            InternalNodeType* node = rhs.mAccessor.getNode<InternalNodeType>();
            node->stealNode<TreeType::LeafNodeType>(ijk, EdgeData(), false);
            rhs.mAccessor.clear();

        } else {

            TreeType::LeafNodeType::ValueOnCIter it = leafIt->cbeginValueOn();
            for ( ; it; ++it) {

                offset = it.pos();
                const EdgeData& rhsValue = it.getValue();

                if (!lhsLeafPt->isValueOn(offset)) {
                    lhsLeafPt->setValueOn(offset, rhsValue);
                } else {

                    EdgeData& lhsValue = const_cast<EdgeData&>(lhsLeafPt->getValue(offset));

                    if (rhsValue.mXDist < lhsValue.mXDist) {
                        lhsValue.mXDist = rhsValue.mXDist;
                        lhsValue.mXPrim = rhsValue.mXPrim;
                    }

                    if (rhsValue.mYDist < lhsValue.mYDist) {
                        lhsValue.mYDist = rhsValue.mYDist;
                        lhsValue.mYPrim = rhsValue.mYPrim;
                    }

                    if (rhsValue.mZDist < lhsValue.mZDist) {
                        lhsValue.mZDist = rhsValue.mZDist;
                        lhsValue.mZPrim = rhsValue.mZPrim;
                    }

                }
            } // end value iteration
        }
    } // end leaf iteration
}


inline void
MeshToVoxelEdgeData::GenEdgeData::operator()(const tbb::blocked_range<size_t> &range)
{
    Primitive prim;

    for (size_t n = range.begin(); n < range.end(); ++n) {

        const Vec4I& verts = mPolygonList[n];

        prim.index = Int32(n);
        prim.a = Vec3d(mPointList[verts[0]]);
        prim.b = Vec3d(mPointList[verts[1]]);
        prim.c = Vec3d(mPointList[verts[2]]);

        if (util::INVALID_IDX != verts[3]) {
            prim.d = Vec3d(mPointList[verts[3]]);
            voxelize<true>(prim);
        } else {
            voxelize<false>(prim);
        }
    }
}


template<bool IsQuad>
inline void
MeshToVoxelEdgeData::GenEdgeData::voxelize(const Primitive& prim)
{
    std::deque<Coord> coordList;
    Coord ijk, nijk;

    ijk = util::nearestCoord(prim.a);
    coordList.push_back(ijk);

    evalPrimitive<IsQuad>(ijk, prim);

    while (!coordList.empty()) {

        ijk = coordList.back();
        coordList.pop_back();

        for (Int32 i = 0; i < 26; ++i) {
            nijk = ijk + util::COORD_OFFSETS[i];

            if (prim.index != mLastPrimAccessor.getValue(nijk)) {
                mLastPrimAccessor.setValue(nijk, prim.index);
                if(evalPrimitive<IsQuad>(nijk, prim)) coordList.push_back(nijk);
            }
        }
    }
}


template<bool IsQuad>
inline bool
MeshToVoxelEdgeData::GenEdgeData::evalPrimitive(const Coord& ijk, const Primitive& prim)
{
    Vec3d uvw, org(ijk[0], ijk[1], ijk[2]);
    bool intersecting = false;
    double t;

    EdgeData edgeData;
    mAccessor.probeValue(ijk, edgeData);

    // Evaluate first triangle
    double dist = (org -
        closestPointOnTriangleToPoint(prim.a, prim.c, prim.b, org, uvw)).lengthSqr();

    if (rayTriangleIntersection(org, Vec3d(1.0, 0.0, 0.0), prim.a, prim.c, prim.b, t)) {
        if (t < edgeData.mXDist) {
            edgeData.mXDist = float(t);
            edgeData.mXPrim = prim.index;
            intersecting = true;
        }
    }

    if (rayTriangleIntersection(org, Vec3d(0.0, 1.0, 0.0), prim.a, prim.c, prim.b, t)) {
        if (t < edgeData.mYDist) {
            edgeData.mYDist = float(t);
            edgeData.mYPrim = prim.index;
            intersecting = true;
        }
    }

    if (rayTriangleIntersection(org, Vec3d(0.0, 0.0, 1.0), prim.a, prim.c, prim.b, t)) {
        if (t < edgeData.mZDist) {
            edgeData.mZDist = float(t);
            edgeData.mZPrim = prim.index;
            intersecting = true;
        }
    }

    if (IsQuad) {
        // Split quad into a second triangle and calculate distance.
        double secondDist = (org -
            closestPointOnTriangleToPoint(prim.a, prim.d, prim.c, org, uvw)).lengthSqr();

        if (secondDist < dist) dist = secondDist;

        if (rayTriangleIntersection(org, Vec3d(1.0, 0.0, 0.0), prim.a, prim.d, prim.c, t)) {
            if (t < edgeData.mXDist) {
                edgeData.mXDist = float(t);
                edgeData.mXPrim = prim.index;
                intersecting = true;
            }
        }

        if (rayTriangleIntersection(org, Vec3d(0.0, 1.0, 0.0), prim.a, prim.d, prim.c, t)) {
            if (t < edgeData.mYDist) {
                edgeData.mYDist = float(t);
                edgeData.mYPrim = prim.index;
                intersecting = true;
            }
        }

        if (rayTriangleIntersection(org, Vec3d(0.0, 0.0, 1.0), prim.a, prim.d, prim.c, t)) {
            if (t < edgeData.mZDist) {
                edgeData.mZDist = float(t);
                edgeData.mZPrim = prim.index;
                intersecting = true;
            }
        }
    }

    if (intersecting) mAccessor.setValue(ijk, edgeData);

    return (dist < 0.86602540378443861);
}


inline bool
MeshToVoxelEdgeData::GenEdgeData::rayTriangleIntersection(
    const Vec3d& origin, const Vec3d& dir,
    const Vec3d& a, const Vec3d& b, const Vec3d& c,
    double& t)
{
    // Check if ray is parallel with triangle

    Vec3d e1 = b - a;
    Vec3d e2 = c - a;
    Vec3d s1 = dir.cross(e2);

    double divisor = s1.dot(e1);
    if (!(std::abs(divisor) > 0.0)) return false;

    // Compute barycentric coordinates

    double inv_divisor = 1.0 / divisor;
    Vec3d d = origin - a;
    double b1 = d.dot(s1) * inv_divisor;

    if (b1 < 0.0 || b1 > 1.0) return false;

    Vec3d s2 = d.cross(e1);
    double b2 = dir.dot(s2) * inv_divisor;

    if (b2 < 0.0 || (b1 + b2) > 1.0) return false;

    // Compute distance to intersection point

    t = e2.dot(s2) * inv_divisor;
    return (t < 0.0) ? false : true;
}


////////////////////////////////////////


inline
MeshToVoxelEdgeData::MeshToVoxelEdgeData()
    : mTree(EdgeData())
{
}


inline void
MeshToVoxelEdgeData::convert(
    const std::vector<Vec3s>& pointList,
    const std::vector<Vec4I>& polygonList)
{
    GenEdgeData converter(pointList, polygonList);
    converter.run();

    mTree.clear();
    mTree.merge(converter.tree());
}


inline void
MeshToVoxelEdgeData::getEdgeData(
    Accessor& acc,
    const Coord& ijk,
    std::vector<Vec3d>& points,
    std::vector<Index32>& primitives)
{
    EdgeData data;
    Vec3d point;

    Coord coord = ijk;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }

        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }

    }

    coord[0] += 1;

    if (acc.probeValue(coord, data)) {

        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }
    }

    coord[2] += 1;

    if (acc.probeValue(coord, data)) {
        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }
    }

    coord[0] -= 1;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }

        if (data.mYPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]) + data.mYDist;
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mYPrim);
        }
    }


    coord[1] += 1;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }
    }

    coord[2] -= 1;

    if (acc.probeValue(coord, data)) {

        if (data.mXPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]) + data.mXDist;
            point[1] = double(coord[1]);
            point[2] = double(coord[2]);

            points.push_back(point);
            primitives.push_back(data.mXPrim);
        }

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }
    }

    coord[0] += 1;

    if (acc.probeValue(coord, data)) {

        if (data.mZPrim != util::INVALID_IDX) {
            point[0] = double(coord[0]);
            point[1] = double(coord[1]);
            point[2] = double(coord[2]) + data.mZDist;

            points.push_back(point);
            primitives.push_back(data.mZPrim);
        }
    }
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MESH_TO_VOLUME_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
