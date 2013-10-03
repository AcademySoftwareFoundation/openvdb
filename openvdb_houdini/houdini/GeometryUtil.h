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
/// @file GeometryUtil.h
/// @author FX R&D Simulation team
/// @brief Utility methods and tools for geometry processing

#ifndef OPENVDB_HOUDINI_GEOMETRY_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_HOUDINI_GEOMETRY_UTIL_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/util/Util.h>

#include <GA/GA_SplittableRange.h>
#include <GU/GU_Detail.h>
#include <boost/shared_ptr.hpp>
#include "Utils.h"


namespace openvdb_houdini {


/// @brief  Add geometry to the given detail to indicate the extents
///         of a frustum transform.
OPENVDB_HOUDINI_API
void drawFrustum(GU_Detail&, const openvdb::math::Transform&,
    const UT_Vector3* boxColor, const UT_Vector3* tickColor, bool shaded);


////////////////////////////////////////


/// @brief  Utility method to check if a point is referenced by primitives
///         from a certain primitive group.
OPENVDB_HOUDINI_API
bool pointInPrimGroup(
    GA_Offset ptnOffset, GU_Detail& geo, const GA_PrimitiveGroup& group);


////////////////////////////////////////


/// @brief Converts geometry to quads and triangles.
///
/// @return A pointer to a new GU_Detail object if the geometry was
///         converted or subdivided else a null pointer is returned.
OPENVDB_HOUDINI_API
boost::shared_ptr<GU_Detail>
validateGeometry(const GU_Detail& geometry, std::string& warning, Interrupter*);


////////////////////////////////////////


/// @brief  TBB body object for threaded world to voxel space
///         transformation and copy of points.
class OPENVDB_HOUDINI_API TransformOp
{
public:
    TransformOp(GU_Detail const * const gdp,
        const openvdb::math::Transform& transform,
        std::vector<openvdb::Vec3s>& pointList);

    void operator()(const GA_SplittableRange &r) const;

private:
    GU_Detail const * const mGdp;
    const openvdb::math::Transform& mTransform;
    std::vector<openvdb::Vec3s>* const mPointList;
};


////////////////////////////////////////


/// @brief  TBB body object for threaded primitive copy.
/// @note   Produces a primitive-vertex index list.
class OPENVDB_HOUDINI_API PrimCpyOp
{
public:
    PrimCpyOp(GU_Detail const * const gdp, std::vector<openvdb::Vec4I>& primList);
    void operator()(const GA_SplittableRange &r) const;

private:
    GU_Detail const * const mGdp;
    std::vector<openvdb::Vec4I>* const mPrimList;
};


////////////////////////////////////////


/// @brief  TBB body object for threaded vertex normal generation.
/// @note   Averages face normals from all similarly oriented primitives,
///         that share the same vertex-point, to maintain sharp features.
class OPENVDB_HOUDINI_API VertexNormalOp
{
public:
    VertexNormalOp(GU_Detail&, const GA_PrimitiveGroup *interiorPrims = NULL, float angle = 0.7);
    void operator()(const GA_SplittableRange&) const;
private:
    const GU_Detail& mDetail;
    const GA_PrimitiveGroup *mInteriorPrims;
    GA_RWHandleV3 mNormalHandle;
    const float mAngle;

    bool isInteriorPrim(GA_Offset primOffset) const
    {
        return mInteriorPrims && mInteriorPrims->containsIndex(
            mDetail.primitiveIndex(primOffset));
    }
};


////////////////////////////////////////


/// @brief  TBB body object for threaded sharp feature construction.
class OPENVDB_HOUDINI_API SharpenFeaturesOp
{
public:
    typedef openvdb::tools::MeshToVoxelEdgeData EdgeData;

    SharpenFeaturesOp(GU_Detail& meshGeo, const GU_Detail& refGeo, EdgeData& edgeData,
        const openvdb::math::Transform& xform, const GA_PrimitiveGroup *surfacePrims = NULL,
        const openvdb::BoolTree * mask = NULL);

    void operator()(const GA_SplittableRange&) const;

private:
    GU_Detail& mMeshGeo;
    const GU_Detail& mRefGeo;
    EdgeData& mEdgeData;
    const openvdb::math::Transform& mXForm;
    const GA_PrimitiveGroup *mSurfacePrims;
    const openvdb::BoolTree * mMaskTree;
};


////////////////////////////////////////


/// @brief  TBB body object for threaded sharp feature construction.
template <typename IndexTreeType, typename BoolTreeType>
class GenAdaptivityMaskOp
{
public:
    typedef openvdb::tree::LeafManager<BoolTreeType> BoolLeafManager;

    GenAdaptivityMaskOp(const GU_Detail& refGeo,
        const IndexTreeType& indexTree, BoolLeafManager& leafs, float edgetolerance = 0.0);

    void run(bool threaded = true);

    void operator()(const tbb::blocked_range<size_t> &range) const;

private:
    const GU_Detail& mRefGeo;
    const IndexTreeType& mIndexTree;
    BoolLeafManager& mLeafs;
    float mEdgeTolerance;
};


template <typename IndexTreeType, typename BoolTreeType>
GenAdaptivityMaskOp<IndexTreeType, BoolTreeType>::GenAdaptivityMaskOp(const GU_Detail& refGeo,
    const IndexTreeType& indexTree, BoolLeafManager& leafs, float edgetolerance)
    : mRefGeo(refGeo)
    , mIndexTree(indexTree)
    , mLeafs(leafs)
    , mEdgeTolerance(edgetolerance)
{
    mEdgeTolerance = std::max(0.0f, mEdgeTolerance);
    mEdgeTolerance = std::min(1.0f, mEdgeTolerance);
}


template <typename IndexTreeType, typename BoolTreeType>
void
GenAdaptivityMaskOp<IndexTreeType, BoolTreeType>::run(bool threaded)
{
    if (threaded) {
        tbb::parallel_for(mLeafs.getRange(), *this);
    } else {
        (*this)(mLeafs.getRange());
    }
}


template <typename IndexTreeType, typename BoolTreeType>
void
GenAdaptivityMaskOp<IndexTreeType, BoolTreeType>::operator()(const tbb::blocked_range<size_t> &range) const
{
    typedef typename openvdb::tree::ValueAccessor<const IndexTreeType> IndexAccessorType;
    IndexAccessorType idxAcc(mIndexTree);

    UT_Vector3 tmpN, normal;
    GA_Offset primOffset;
    int tmpIdx;

    openvdb::Coord ijk, nijk;
    typename BoolTreeType::LeafNodeType::ValueOnIter iter;

    for (size_t n = range.begin(); n < range.end(); ++n) {
        iter = mLeafs.leaf(n).beginValueOn();
        for (; iter; ++iter) {
            ijk = iter.getCoord();
            
            bool edgeVoxel = false;

            int idx = idxAcc.getValue(ijk);

            primOffset = mRefGeo.primitiveOffset(idx);
            normal = mRefGeo.getGEOPrimitive(primOffset)->computeNormal();

            for (size_t i = 0; i < 18; ++i) {
                nijk = ijk + openvdb::util::COORD_OFFSETS[i];
                if (idxAcc.probeValue(nijk, tmpIdx) && tmpIdx != idx) {
                    primOffset = mRefGeo.primitiveOffset(tmpIdx);
                    tmpN = mRefGeo.getGEOPrimitive(primOffset)->computeNormal();

                    if (normal.dot(tmpN) < mEdgeTolerance) {
                        edgeVoxel = true;
                        break;
                    }
                }
            }

            if (!edgeVoxel) iter.setValueOff();
        }
    }
}



} // namespace openvdb_houdini




////////////////////////////////////////


#if (UT_VERSION_INT < 0x0c0500F5) // Prior to 12.5.245

// Symbols in namespace GU_Convert_H12_5 were added to GU_ConvertParms.h in 12.5.245

namespace GU_Convert_H12_5 {

/// Simple helper class for tracking a range of new primitives and points
class GU_ConvertMarker
{
public:
    GU_ConvertMarker(const GA_Detail &geo)
        : myGeo(geo)
        , myPrimBegin(primOff())
        , myPtBegin(ptOff())
    {
    }

    GA_Range getPrimitives() const
    {
        return GA_Range(myGeo.getPrimitiveMap(), myPrimBegin, primOff());
    }
    GA_Range getPoints() const
    {
        return GA_Range(myGeo.getPointMap(), myPtBegin, ptOff());
    }

    GA_Offset primitiveBegin() const { return myPrimBegin; }
    GA_Offset pointBegin() const { return myPtBegin; }

    GA_Size numPrimitives() const { return primOff() - myPrimBegin; }
    GA_Size numPoints() const { return ptOff() - myPtBegin; }

private:
    GA_Offset primOff() const { return myGeo.getPrimitiveMap().lastOffset() + 1; }
    GA_Offset ptOff() const { return myGeo.getPointMap().lastOffset() + 1; }

private:
    const GA_Detail& myGeo;
    GA_Offset myPrimBegin;
    GA_Offset myPtBegin;
};


OPENVDB_HOUDINI_API
void
GUconvertCopySingleVertexPrimAttribsAndGroups(
    GU_ConvertParms &parms,
    const GA_Detail &src,
    GA_Offset src_primoff,
    GA_Detail &dst,
    const GA_Range &dst_prims,
    const GA_Range &dst_points);

} // namespace GU_Convert_H12_5

using GU_Convert_H12_5::GU_ConvertMarker;
using GU_Convert_H12_5::GUconvertCopySingleVertexPrimAttribsAndGroups;

#endif // Prior to 12.5.245


////////////////////////////////////////


#endif // OPENVDB_HOUDINI_GEOMETRY_UTIL_HAS_BEEN_INCLUDED


// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
