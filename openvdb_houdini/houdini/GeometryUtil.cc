///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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
/// @file GeometryUtil.cc
/// @author FX R&D Simulation team
/// @brief Utility methods and tools for geometry processing

#include "GeometryUtil.h"
#include "Utils.h"
#include <houdini_utils/geometry.h> // for createBox()
#include <openvdb/tools/VolumeToMesh.h>

#include <UT/UT_ScopedPtr.h>
#include <UT/UT_String.h>
#include <UT/UT_BoundingBox.h>
#include <UT/UT_Version.h>
#include <GU/GU_PrimPoly.h>
#include <GU/GU_ConvertParms.h>
#include <GA/GA_ElementWrangler.h>
#include <GA/GA_PageIterator.h>
#include <GA/GA_SplittableRange.h>
#include <GA/GA_Types.h>
#include <OBJ/OBJ_Camera.h>


namespace openvdb_houdini {


void
drawFrustum(
    GU_Detail& geo, const openvdb::math::Transform& transform,
    const UT_Vector3* boxColor, const UT_Vector3* tickColor,
    bool shaded, bool drawTicks)
{
    if (transform.mapType() != openvdb::math::NonlinearFrustumMap::mapType()) {
        return;
    }

    const openvdb::math::NonlinearFrustumMap& frustum =
        *transform.map<openvdb::math::NonlinearFrustumMap>();
    const openvdb::BBoxd bbox = frustum.getBBox();

    UT_Vector3 corners[8];

    struct Local{
        static inline void floatVecFromDoubles(UT_Vector3& v, double x, double y, double z) {
            v[0] = static_cast<float>(x);
            v[1] = static_cast<float>(y);
            v[2] = static_cast<float>(z);
        }
    };

    openvdb::Vec3d wp = frustum.applyMap(bbox.min());
    Local::floatVecFromDoubles(corners[0], wp[0], wp[1], wp[2]);

    wp[0] = bbox.min()[0];
    wp[1] = bbox.min()[1];
    wp[2] = bbox.max()[2];
    wp = frustum.applyMap(wp);
    Local::floatVecFromDoubles(corners[1], wp[0], wp[1], wp[2]);

    wp[0] = bbox.max()[0];
    wp[1] = bbox.min()[1];
    wp[2] = bbox.max()[2];
    wp = frustum.applyMap(wp);
    Local::floatVecFromDoubles(corners[2], wp[0], wp[1], wp[2]);

    wp[0] = bbox.max()[0];
    wp[1] = bbox.min()[1];
    wp[2] = bbox.min()[2];
    wp = frustum.applyMap(wp);
    Local::floatVecFromDoubles(corners[3], wp[0], wp[1], wp[2]);

    wp[0] = bbox.min()[0];
    wp[1] = bbox.max()[1];
    wp[2] = bbox.min()[2];
    wp = frustum.applyMap(wp);
    Local::floatVecFromDoubles(corners[4], wp[0], wp[1], wp[2]);

    wp[0] = bbox.min()[0];
    wp[1] = bbox.max()[1];
    wp[2] = bbox.max()[2];
    wp = frustum.applyMap(wp);
    Local::floatVecFromDoubles(corners[5], wp[0], wp[1], wp[2]);

    wp = frustum.applyMap(bbox.max());
    Local::floatVecFromDoubles(corners[6], wp[0], wp[1], wp[2]);

    wp[0] = bbox.max()[0];
    wp[1] = bbox.max()[1];
    wp[2] = bbox.min()[2];
    wp = frustum.applyMap(wp);
    Local::floatVecFromDoubles(corners[7], wp[0], wp[1], wp[2]);

    float alpha = shaded ? 0.3f : 1.0f;

    houdini_utils::createBox(geo, corners, boxColor, shaded, alpha);

    // Add voxel ticks
    if (drawTicks) {
        GA_RWHandleV3 cd;
        int count = 0;
        double length = 4, maxLength = (bbox.max()[1] - bbox.min()[1]);
        size_t total_count = 0;

        if (tickColor) {
#if (UT_VERSION_INT >= 0x0e0000b4) // 14.0.180 or later
            cd.bind(geo.addDiffuseAttribute(GA_ATTRIB_POINT));
#else
            cd.bind(geo.addDiffuseAttribute(GA_ATTRIB_POINT).getAttribute());
#endif
        }

        for (double z = bbox.min()[2] + 1, Z = bbox.max()[2]; z < Z; ++z) {

            GA_Offset v0 = geo.appendPointOffset();
            GA_Offset v1 = geo.appendPointOffset();

            if (tickColor) {
                cd.set(v0, *tickColor);
                cd.set(v1, *tickColor);
            }

            length = 4;
            ++count;
            if (count == 5) {
                length = 8;
                count = 0;
            }

            length = std::min(length, maxLength);

            wp[0] = bbox.max()[0];
            wp[1] = bbox.max()[1]-length;
            wp[2] = z;
            wp = frustum.applyMap(wp);
            geo.setPos3(v0, wp[0], wp[1], wp[2]);

            wp[0] = bbox.max()[0];
            wp[1] = bbox.max()[1];
            wp[2] = z;
            wp = frustum.applyMap(wp);
            geo.setPos3(v1, wp[0], wp[1], wp[2]);


            GEO_PrimPoly& prim = *GU_PrimPoly::build(&geo, 0, GU_POLY_OPEN, 0);
            prim.appendVertex(v0);
            prim.appendVertex(v1);

            if (++total_count > 999) break;
        }

        count = 0;
        total_count = 0;
        maxLength = (bbox.max()[2] - bbox.min()[2]);
        for (double x = bbox.min()[0] + 1, X = bbox.max()[0]; x < X; ++x) {

            GA_Offset v0 = geo.appendPointOffset();
            GA_Offset v1 = geo.appendPointOffset();

            if (tickColor) {
                cd.set(v0, *tickColor);
                cd.set(v1, *tickColor);
            }

            length = 1;
            ++count;
            if (count == 5) {
                length = 2;
                count = 0;
            }

            length = std::min(length, maxLength);

            wp[0] = x;
            wp[1] = bbox.max()[1];
            wp[2] = bbox.max()[2]-length;
            wp = frustum.applyMap(wp);
            geo.setPos3(v0, wp[0], wp[1], wp[2]);

            wp[0] = x;
            wp[1] = bbox.max()[1];
            wp[2] = bbox.max()[2];
            wp = frustum.applyMap(wp);
            geo.setPos3(v1, wp[0], wp[1], wp[2]);


            GEO_PrimPoly& prim = *GU_PrimPoly::build(&geo, 0, GU_POLY_OPEN, 0);
            prim.appendVertex(v0);
            prim.appendVertex(v1);

            if (++total_count > 999) break;
        }
    }
}


////////////////////////////////////////


openvdb::math::Transform::Ptr
frustumTransformFromCamera(
    OP_Node& node, OP_Context& context, OBJ_Camera& cam,
    float offset, float nearPlaneDist, float farPlaneDist,
    float voxelDepthSize, int voxelCountX)
{
#if (UT_VERSION_INT >= 0x0c050000) // 12.5.0 or later
    cam.addInterestOnCameraParms(&node);
#else
    cam.addInterestsOnIFDParms(&node);
#endif

    const fpreal time = context.getTime();

    // Eval camera parms
    const fpreal camAspect = cam.ASPECT(time);
    const fpreal camFocal = cam.FOCAL(time);
    const fpreal camAperture = cam.APERTURE(time);
    const fpreal camXRes = cam.RESX(time);
    const fpreal camYRes = cam.RESY(time);

    nearPlaneDist += offset;
    farPlaneDist += offset;

    const fpreal depth = farPlaneDist - nearPlaneDist;
    const fpreal zoom = camAperture / camFocal;
    const fpreal aspectRatio = camYRes / (camXRes * camAspect);

    openvdb::Vec2d nearPlaneSize;
    nearPlaneSize.x() = nearPlaneDist * zoom;
    nearPlaneSize.y() = nearPlaneSize.x() * aspectRatio;

    openvdb::Vec2d farPlaneSize;
    farPlaneSize.x() = farPlaneDist * zoom;
    farPlaneSize.y() = farPlaneSize.x() * aspectRatio;


    // Create the linear map
    openvdb::math::Mat4d xform(openvdb::math::Mat4d::identity());
    xform.setToTranslation(openvdb::Vec3d(0, 0, -(nearPlaneDist - offset)));

    /// this will be used to scale the frust to the correct size, and orient the
    /// into the frustum as the negative z-direction
    xform.preScale(openvdb::Vec3d(nearPlaneSize.x(), nearPlaneSize.x(), -nearPlaneSize.x()));

    openvdb::math::Mat4d camxform(openvdb::math::Mat4d::identity());
    {
        UT_Matrix4 M;
        OBJ_Node *meobj = node.getCreator()->castToOBJNode();
        if (meobj) {
            node.addExtraInput(meobj, OP_INTEREST_DATA);
            if (!cam.getRelativeTransform(*meobj, M, context))
                node.addTransformError(cam, "relative");
        } else {
            if (!((OP_Node *)&cam)->getWorldTransform(M, context))
                node.addTransformError(cam, "world");
        }

        for (unsigned i = 0; i < 4; ++i) {
            for (unsigned j = 0; j < 4; ++j) {
                camxform(i,j) = M(i,j);
            }
        }
    }

    openvdb::math::MapBase::Ptr linearMap(openvdb::math::simplify(
        openvdb::math::AffineMap(xform * camxform).getAffineMap()));


    // Create the non linear map
    const int voxelCountY = int(std::ceil(float(voxelCountX) * aspectRatio));
    const int voxelCountZ = int(std::ceil(depth / voxelDepthSize));

    // the frustum will be the image of the coordinate in this bounding box
    openvdb::BBoxd bbox(openvdb::Vec3d(0, 0, 0),
        openvdb::Vec3d(voxelCountX, voxelCountY, voxelCountZ));
    // define the taper
    const fpreal taper = nearPlaneSize.x() / farPlaneSize.x();

    // note that the depth is scaled on the nearPlaneSize.
    // the linearMap will uniformly scale the frustum to the correct size
    // and rotate to align with the camera
    return openvdb::math::Transform::Ptr(new openvdb::math::Transform(
        openvdb::math::MapBase::Ptr(new openvdb::math::NonlinearFrustumMap(
            bbox, taper, depth/nearPlaneSize.x(), linearMap))));
}


////////////////////////////////////////


bool
pointInPrimGroup(GA_Offset ptnOffset, GU_Detail& geo, const GA_PrimitiveGroup& group)
{
    bool surfacePrim = false;

    GA_Offset primOffset, vtxOffset = geo.pointVertex(ptnOffset);

    while (GAisValid(vtxOffset)) {

        primOffset = geo.vertexPrimitive(vtxOffset);

        if (group.containsIndex(geo.primitiveIndex(primOffset))) {
            surfacePrim = true;
            break;
        }

        vtxOffset = geo.vertexToNextVertex(vtxOffset);
    }

    return surfacePrim;
}


////////////////////////////////////////


boost::shared_ptr<GU_Detail>
validateGeometry(const GU_Detail& geometry, std::string& warning, Interrupter* boss)
{
    const GU_Detail* geo = &geometry;
    boost::shared_ptr<GU_Detail> geoPtr;

    const GEO_Primitive *prim;
    bool needconvert = false, needdivide = false, needclean = false;

    GA_FOR_ALL_PRIMITIVES(geo, prim)
    {
        if (boss && boss->wasInterrupted()) return geoPtr;

        if (prim->getPrimitiveId() & GEO_PrimTypeCompat::GEOPRIMPOLY) {

            const GEO_PrimPoly *poly = static_cast<const GEO_PrimPoly*>(prim);

            if (poly->getVertexCount() > 4) needdivide = true;
            if (poly->getVertexCount() < 3) needclean = true;

        } else {
            needconvert = true;
            // Some conversions will break division requirements,
            // like polysoup -> polygon.
            needdivide = true;
            break;
        }
    }

    if (needconvert || needdivide || needclean) {
        geoPtr.reset(new GU_Detail());

        geoPtr->duplicate(*geo);
        geo = geoPtr.get();
    }

    if (boss && boss->wasInterrupted()) return geoPtr;

    if (needconvert) {
        GU_ConvertParms parms;
#if (UT_VERSION_INT < 0x0d0000b1) // before 13.0.177
        parms.fromType = GEO_PrimTypeCompat::GEOPRIMALL;
        parms.toType = GEO_PrimTypeCompat::GEOPRIMPOLY;
#else
        parms.setFromType(GEO_PrimTypeCompat::GEOPRIMALL);
        parms.setToType(GEO_PrimTypeCompat::GEOPRIMPOLY);
        // We don't want interior tetrahedron faces as they will just
        // distract us and fill up the vdb.
        parms.setSharedFaces(false);
#endif
        geoPtr->convert(parms);
    }

    if (boss && boss->wasInterrupted()) return geoPtr;

    if (needdivide) {
        geoPtr->convex(4);
    }

    if (needclean || needdivide || needconvert) {
        // Final pass to delete anything illegal.
        // There could be little fligs left over that
        // we don't want confusing the mesher.
        GEO_Primitive           *delprim;
        GA_PrimitiveGroup       *delgrp = 0;

        GA_FOR_ALL_PRIMITIVES(geoPtr.get(), delprim) {

            if (boss && boss->wasInterrupted()) return geoPtr;

            bool kill = false;
            if (delprim->getPrimitiveId() & GEO_PrimTypeCompat::GEOPRIMPOLY) {

                GEO_PrimPoly *poly = static_cast<GEO_PrimPoly*>(delprim);

                if (poly->getVertexCount() > 4) kill = true;
                if (poly->getVertexCount() < 3) kill = true;

            } else {
                kill = true;
            }

            if (kill) {
                if (!delgrp) {
                    delgrp = geoPtr->newPrimitiveGroup("__delete_group__", 1);
                }
                delgrp->add(delprim);
            }
        }

        if (delgrp) {
            geoPtr->deletePrimitives(*delgrp, 1);
            geoPtr->destroyPrimitiveGroup(delgrp);
            delgrp = 0;
        }
    }

    if (geoPtr && needconvert) {
        warning = "Geometry has been converted to quads and triangles.";
    }

    return geoPtr;
}


////////////////////////////////////////


TransformOp::TransformOp(GU_Detail const * const gdp,
    const openvdb::math::Transform& transform,
    std::vector<openvdb::Vec3s>& pointList)
    : mGdp(gdp)
    , mTransform(transform)
    , mPointList(&pointList)
{
}


void
TransformOp::operator()(const GA_SplittableRange &r) const
{
    GA_Offset start, end;
    UT_Vector3 pos;
    openvdb::Vec3d ipos;

    // Iterate over pages in the range
    for (GA_PageIterator pit = r.beginPages(); !pit.atEnd(); ++pit) {

        // Iterate over block
        for (GA_Iterator it(pit.begin()); it.blockAdvance(start, end); ) {
            // Element
            for (GA_Offset i = start; i < end; ++i) {
                pos = mGdp->getPos3(i);
                ipos = mTransform.worldToIndex(openvdb::Vec3d(pos.x(), pos.y(), pos.z()));

                (*mPointList)[mGdp->pointIndex(i)] = openvdb::Vec3s(ipos);
            }
        }
    }
}


////////////////////////////////////////


PrimCpyOp::PrimCpyOp(GU_Detail const * const gdp, std::vector<openvdb::Vec4I>& primList)
    : mGdp(gdp)
    , mPrimList(&primList)
{
}


void
PrimCpyOp::operator()(const GA_SplittableRange &r) const
{
    openvdb::Vec4I prim;
    GA_Offset start, end;
    GA_Size vtxn;
    int vtx;

    // Iterate over pages in the range
    for (GA_PageIterator pit = r.beginPages(); !pit.atEnd(); ++pit) {

        // Iterate over the elements in the page.
        for (GA_Iterator it(pit.begin()); it.blockAdvance(start, end); ) {
            for (GA_Offset i = start; i < end; ++i) {

                const GA_Primitive* primRef =  mGdp->getPrimitiveList().get(i);
                vtxn = primRef->getVertexCount();

                if (primRef->getTypeId() == GEO_PRIMPOLY && (3 == vtxn || 4 == vtxn)) {

                    GA_Primitive::const_iterator vit;
                    for (vtx = 0, primRef->beginVertex(vit); !vit.atEnd(); ++vit, ++vtx) {
                        prim[vtx] = static_cast<openvdb::Vec4I::ValueType>(
                            mGdp->pointIndex(vit.getPointOffset()));
                    }

                    if (vtxn != 4) prim[3] = openvdb::util::INVALID_IDX;

                    (*mPrimList)[mGdp->primitiveIndex(i)] = prim;
                } else {
                    throw std::runtime_error(
                        "Invalid geometry; only quads and triangles are supported.");
                }
            }
        }
    }
}


////////////////////////////////////////


VertexNormalOp::VertexNormalOp(GU_Detail& detail,
    const GA_PrimitiveGroup *interiorPrims, float angle)
    : mDetail(detail)
    , mInteriorPrims(interiorPrims)
    , mAngle(angle)
{
    GA_RWAttributeRef attributeRef =
        detail.findFloatTuple(GA_ATTRIB_VERTEX, "N", 3);

    if (!attributeRef.isValid()) {
        attributeRef = detail.addFloatTuple(
            GA_ATTRIB_VERTEX, "N", 3, GA_Defaults(0));
    }

    mNormalHandle = attributeRef.getAttribute();
}


void
VertexNormalOp::operator()(const GA_SplittableRange& range) const
{
    GA_Offset start, end, vtxOffset, primOffset;
    GA_Primitive::const_iterator it;
    UT_Vector3 primN, avgN, tmpN;
    bool interiorPrim = false;
    const GA_Primitive * primRef = NULL;

    for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
        for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
            for (GA_Offset i = start; i < end; ++i) {

                primRef = mDetail.getPrimitiveList().get(i);

                primN = mDetail.getGEOPrimitive(i)->computeNormal();
                interiorPrim = isInteriorPrim(i);

                for (primRef->beginVertex(it); !it.atEnd(); ++it) {

                    avgN = primN;
                    vtxOffset = mDetail.pointVertex(it.getPointOffset());

                    while (GAisValid(vtxOffset)) {

                        primOffset = mDetail.vertexPrimitive(vtxOffset);
                        if (interiorPrim == isInteriorPrim(primOffset)) {
                            tmpN = mDetail.getGEOPrimitive(primOffset)->computeNormal();
                            if (tmpN.dot(primN) > mAngle) avgN += tmpN;
                        }
                        vtxOffset = mDetail.vertexToNextVertex(vtxOffset);
                    }

                    avgN.normalize();
                    mNormalHandle.set(*it, avgN);

                } // prim vtx iteration.
            }
        }
    }
}


////////////////////////////////////////


SharpenFeaturesOp::SharpenFeaturesOp(
    GU_Detail& meshGeo, const GU_Detail& refGeo, EdgeData& edgeData,
    const openvdb::math::Transform& xform,
    const GA_PrimitiveGroup * surfacePrims,
    const openvdb::BoolTree * mask)
    : mMeshGeo(meshGeo)
    , mRefGeo(refGeo)
    , mEdgeData(edgeData)
    , mXForm(xform)
    , mSurfacePrims(surfacePrims)
    , mMaskTree(mask)
{
}

void
SharpenFeaturesOp::operator()(const GA_SplittableRange& range) const
{
    openvdb::tools::MeshToVoxelEdgeData::Accessor acc = mEdgeData.getAccessor();

    typedef openvdb::tree::ValueAccessor<const openvdb::BoolTree> BoolAccessor;
    boost::scoped_ptr<BoolAccessor> maskAcc;

    if (mMaskTree) {
        maskAcc.reset(new BoolAccessor(*mMaskTree));
    }

    GA_Offset start, end, ptnOffset, primOffset;

    UT_Vector3 tmpN, tmpP, avgP;
    UT_BoundingBoxD cell;

    openvdb::Vec3d pos, normal;
    openvdb::Coord ijk;

    std::vector<openvdb::Vec3d> points(12), normals(12);
    std::vector<openvdb::Index32> primitives(12);

    for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
        for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
            for (ptnOffset = start; ptnOffset < end; ++ptnOffset) {

                // Check if this point is referenced by a surface primitive.
                if (mSurfacePrims && !pointInPrimGroup(ptnOffset, mMeshGeo, *mSurfacePrims))
                    continue;

                tmpP = mMeshGeo.getPos3(ptnOffset);
                pos[0] = tmpP.x();
                pos[1] = tmpP.y();
                pos[2] = tmpP.z();

                pos = mXForm.worldToIndex(pos);

                ijk[0] = int(std::floor(pos[0]));
                ijk[1] = int(std::floor(pos[1]));
                ijk[2] = int(std::floor(pos[2]));


                if (maskAcc && !maskAcc->isValueOn(ijk)) continue;

                points.clear();
                normals.clear();
                primitives.clear();

                // get voxel-edge intersections
                mEdgeData.getEdgeData(acc, ijk, points, primitives);

                avgP.assign(0.0, 0.0, 0.0);

                // get normal list
                for (size_t n = 0, N = points.size(); n < N; ++n) {

                    avgP.x() = static_cast<float>(avgP.x() + points[n].x());
                    avgP.y() = static_cast<float>(avgP.y() + points[n].y());
                    avgP.z() = static_cast<float>(avgP.z() + points[n].z());

                    primOffset = mRefGeo.primitiveOffset(primitives[n]);

                    tmpN = mRefGeo.getGEOPrimitive(primOffset)->computeNormal();

                    normal[0] = tmpN.x();
                    normal[1] = tmpN.y();
                    normal[2] = tmpN.z();

                    normals.push_back(normal);
                }

                // Calculate feature point position
                if (points.size() > 1) {

                    pos = openvdb::tools::findFeaturePoint(points, normals);

                    // Constrain points to stay inside their initial
                    // coordinate cell.
                    cell.setBounds(double(ijk[0]), double(ijk[1]), double(ijk[2]),
                        double(ijk[0]+1), double(ijk[1]+1), double(ijk[2]+1));

                    cell.expandBounds(0.3, 0.3, 0.3);

                    if (!cell.isInside(pos[0], pos[1], pos[2])) {

                        UT_Vector3 org(
                            static_cast<float>(pos[0]),
                            static_cast<float>(pos[1]),
                            static_cast<float>(pos[2]));

                        avgP *= 1.f / float(points.size());
                        UT_Vector3 dir = avgP - org;
                        dir.normalize();

                        double distance;

                        if(cell.intersectRay(org, dir, 1E17F, &distance) > 0) {
                            tmpP = org + dir * distance;

                            pos[0] = tmpP.x();
                            pos[1] = tmpP.y();
                            pos[2] = tmpP.z();
                        }
                    }

                    pos = mXForm.indexToWorld(pos);

                    tmpP.x() = static_cast<float>(pos[0]);
                    tmpP.y() = static_cast<float>(pos[1]);
                    tmpP.z() = static_cast<float>(pos[2]);

                    mMeshGeo.setPos3(ptnOffset, tmpP);
                }
             }
        }
    }
}


} // namespace openvdb_houdini


////////////////////////////////////////


#if (UT_VERSION_INT < 0x0c0500F5) // Prior to 12.5.245

// Symbols in namespace GU_Convert_H12_5 were added to GU_ConvertParms.h in 12.5.245

namespace GU_Convert_H12_5 {

// Implementation which uses un-cached wranglers for H12.1, in H12.5 these
// wranglers are cached across all primitives with GU_ConvertParms itself.
void
GUconvertCopySingleVertexPrimAttribsAndGroups(
    GU_ConvertParms &/*parms*/,
    const GA_Detail &src,
    GA_Offset src_primoff,
    GA_Detail &dst,
    const GA_Range &dst_prims,
    const GA_Range &dst_points)
{
    UT_ScopedPtr<GA_ElementWranglerCache> cache;
#if 0
    if (parms.preserveGroups)
        cache.reset(new GA_ElementWranglerCache(dst, src,
            GA_AttributeFilter::selectGroup()));
    else
#endif
        cache.reset(new GA_ElementWranglerCache(dst, src,
            GA_PointWrangler::EXCLUDE_P));

    const GA_Primitive& src_prim = *(src.getPrimitiveList().get(src_primoff));
    GA_Offset           src_vtxoff = src_prim.getVertexOffset(0);
    GA_Offset           src_ptoff = src_prim.getPointOffset(0);
    GA_ElementWranglerCache&    wranglers = *cache;
    GA_PrimitiveWrangler&       prim_wrangler = wranglers.getPrimitive();
    GA_VertexWrangler&          vtx_wrangler = wranglers.getVertex();
    GA_PointWrangler&           pt_wrangler = wranglers.getPoint();
    GA_PrimitiveWrangler*       prim_group_wrangler = NULL;
    GA_PointWrangler*           pt_group_wrangler = NULL;

#if 1
    bool have_vtx_attribs = true;
    bool have_pt_attribs = true;
#else
    // This is only optimization available in H12.5
    bool have_vtx_attribs = (vtx_wrangler.getNumAttributes() > 0);
    bool have_pt_attribs = (pt_wrangler.getNumAttributes() > 0);
#endif

#if 0
    if (parms.preserveGroups)
    {
        prim_group_wrangler = &parms.getGroupWranglers(dst,&src).getPrimitive();
        if (prim_group_wrangler->getNumAttributes() <= 0)
            prim_group_wrangler = NULL;
        pt_group_wrangler = &parms.getGroupWranglers(dst,&src).getPoint();
        if (pt_group_wrangler->getNumAttributes() <= 0)
            pt_group_wrangler = NULL;
    }
#endif

    UT_ASSERT(src_prim.getVertexCount() == 1);
    for (GA_Iterator i(dst_prims); !i.atEnd(); ++i)
    {
        if (prim_group_wrangler)
            prim_group_wrangler->copyAttributeValues(*i, src_primoff);
        if (have_pt_attribs)
            prim_wrangler.copyAttributeValues(*i, src_primoff);
        if (have_vtx_attribs)
        {
            GA_Primitive &dst_prim = *(dst.getPrimitiveList().get(*i));
            GA_Size nvtx = dst_prim.getVertexCount();
            for (GA_Size j = 0; j < nvtx; ++j)
            {
                vtx_wrangler.copyAttributeValues(
                    dst_prim.getVertexOffset(j),
                    src_vtxoff);
            }
        }
    }

    for (GA_Iterator i(dst_points); !i.atEnd(); ++i)
    {
        if (pt_group_wrangler)
            pt_group_wrangler->copyAttributeValues(*i, src_ptoff);
        if (have_pt_attribs)
            pt_wrangler.copyAttributeValues(*i, src_ptoff);
    }
}

} // namespace GU_Convert_H12_5

#endif // Prior to 12.5.245

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
