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
/// @file SOP_OpenVDB_Ray.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Performs geometry projections using level set ray intersections or closest point queries.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>

#include <openvdb/tools/VolumeToSpheres.h>
#include <openvdb/tools/RayIntersector.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_PageIterator.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>
#include <GU/GU_PrimSphere.h>

#include <boost/algorithm/string/join.hpp>

#include <string>
#include <vector>


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_Ray: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Ray(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Ray() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
};


////////////////////////////////////////

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to surface.")
        .setChoiceList(&hutil::PrimGroupMenu));

    { // Method
        const char* items[] = {
            "rayintersection", "Ray Intersection",
            "closestpoint",   "Closest Surface Point",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "method", "Method")
            .setDefault(PRMzeroDefaults)
            .setHelpText("Projection method")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }

    parms.add(hutil::ParmFactory(PRM_FLT_J, "isovalue", "Isovalue")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0)
        .setHelpText("The crossing point of the VDB values that is considered "
            "the surface. The zero default value works for signed distance "
            "fields while fog volumes require a larger positive value, 0.5 is "
            "a good initial guess."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "dotrans", "Transform")
        .setDefault(PRMoneDefaults)
        .setHelpText("Move the intersected geometry."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "scale", "Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_UI, 0, PRM_RANGE_UI, 1));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "putdist", "Store Distances")
        .setHelpText("Creates a point attribute for the distance to the "
            "collision surface or the closest surface point."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "lookfar", "Intersect Farthest Surface")
        .setHelpText("The farthest intersection point is used instead of the closest."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "reverserays", "Reverse Rays")
        .setHelpText("Make rays fire in the direction opposite to the normals."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "bias", "Bias")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_UI, 0, PRM_RANGE_UI, 1)
        .setHelpText("Offsets the starting position of the rays."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "creategroup", "Create Ray Hit Group")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Point group of all successful intersections"));

    parms.add(hutil::ParmFactory(PRM_STRING, "hitgrp", "Ray Hit Group")
        .setDefault("rayHitGroup")
        .setHelpText("Point group name"));


    //////////

    hvdb::OpenVDBOpFactory("OpenVDB Ray", SOP_OpenVDB_Ray::factory, parms, *table)
        .addInput("points")
        .addInput("level set grids");
}


OP_Node*
SOP_OpenVDB_Ray::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Ray(net, name, op);
}


SOP_OpenVDB_Ray::SOP_OpenVDB_Ray(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


bool
SOP_OpenVDB_Ray::updateParmsFlags()
{
    bool changed = false;

    bool rayintersection = evalInt("method", 0, 0) == 0;

    changed |= enableParm("isovalue", !rayintersection);

    changed |= enableParm("lookfar", rayintersection);
    changed |= enableParm("reverserays", rayintersection);
    changed |= enableParm("creategroup", rayintersection);
    changed |= enableParm("bias", rayintersection);

    changed |= enableParm("scale", bool(evalInt("dotrans", 0, 0)));

    bool creategroup = evalInt("creategroup", 0, 0);
    changed |= enableParm("hitgrp", creategroup && rayintersection);

    return changed;
}


////////////////////////////////////////

template<typename GridType>
class IntersectPoints
{
public:
    IntersectPoints(
        const GU_Detail& gdp,
        const UT_Vector3Array& pointNormals,
        const GridType& grid,
        UT_Vector3Array& positions,
        UT_FloatArray& distances,
        std::vector<char>& intersections,
        bool keepMaxDist = false,
        bool reverseRays = false,
        double scale = 1.0,
        double bias = 0.0)
    : mGdp(gdp)
    , mPointNormals(pointNormals)
    , mIntersector(grid)
    , mPositions(positions)
    , mDistances(distances)
    , mIntersections(intersections)
    , mKeepMaxDist(keepMaxDist)
    , mReverseRays(reverseRays)
    , mScale(scale)
    , mBias(bias)
    {
    }

    void operator()(const GA_SplittableRange &range) const
    {
        GA_Offset start, end;
        GA_Index pointIndex;
        typedef openvdb::math::Ray<double> RayT;
        openvdb::Vec3d eye, dir, intersection;

        const bool doScaling = !openvdb::math::isApproxEqual(mScale, 1.0);
        const bool offsetRay = !openvdb::math::isApproxEqual(mBias, 0.0);

        GA_ROPageHandleV3 points(mGdp.getP());

        // Iterate over blocks
        for (GA_Iterator it(range); it.blockAdvance(start, end); ) {

            points.setPage(start);

            // Point Offsets
            for (GA_Offset pointOffset = start; pointOffset < end; ++pointOffset) {

                const UT_Vector3& pos = points.value(pointOffset);

                eye[0] = double(pos.x());
                eye[1] = double(pos.y());
                eye[2] = double(pos.z());

                pointIndex = mGdp.pointIndex(pointOffset);

                const UT_Vector3& normal = mPointNormals(pointIndex);

                dir[0] = double(normal.x());
                dir[1] = double(normal.y());
                dir[2] = double(normal.z());

                if (mReverseRays) dir = -dir;

                RayT ray((offsetRay ? (eye + dir * mBias) : eye), dir);

                if (!mIntersector.intersectsWS(ray, intersection)) {

                    if (!mIntersections[pointIndex]) mPositions(pointIndex) = pos;
                    continue;
                }

                float distance = float((intersection - eye).length());

                if ((!mKeepMaxDist && mDistances(pointIndex) > distance) ||
                    (mKeepMaxDist && mDistances(pointIndex) < distance)) {

                    mDistances(pointIndex) = distance;

                    UT_Vector3& position = mPositions(pointIndex);


                    if (doScaling) intersection = eye + dir * mScale * double(distance);

                    position.x() = float(intersection[0]);
                    position.y() = float(intersection[1]);
                    position.z() = float(intersection[2]);
                }

                mIntersections[pointIndex] = 1;
            }
        }

    }

private:
    const GU_Detail& mGdp;
    const UT_Vector3Array& mPointNormals;
    openvdb::tools::LevelSetRayIntersector<GridType> mIntersector;
    UT_Vector3Array& mPositions;
    UT_FloatArray& mDistances;
    std::vector<char>& mIntersections;
    const bool mKeepMaxDist, mReverseRays;
    const double mScale, mBias;
};


template<typename GridT, typename InterrupterT>
inline void
closestPoints(const GridT& grid, float isovalue, const GU_Detail& gdp,
    UT_FloatArray& distances, UT_Vector3Array* positions, InterrupterT& boss)
{

    std::vector<openvdb::Vec3R> tmpPoints(distances.entries());

    GA_ROHandleV3 points(gdp.getP());

    for (size_t n = 0, N = tmpPoints.size(); n < N; ++n) {
        const UT_Vector3 pos = points.get(gdp.pointOffset(n));
        tmpPoints[n][0] = pos.x();
        tmpPoints[n][1] = pos.y();
        tmpPoints[n][2] = pos.z();
    }


    std::vector<float> tmpDistances;

    const bool transformPoints = (positions != NULL);

    openvdb::tools::ClosestSurfacePoint<GridT> closestPoint;
    closestPoint.initialize(grid, isovalue, &boss);

    if (transformPoints) closestPoint.searchAndReplace(tmpPoints, tmpDistances);
    else closestPoint.search(tmpPoints, tmpDistances);

    for (size_t n = 0, N = tmpDistances.size(); n < N; ++n) {


        if (tmpDistances[n] < distances(n)) {
            distances(n) = tmpDistances[n];
            if (transformPoints) {
                UT_Vector3& pos = (*positions)(n);

                pos.x() = tmpPoints[n].x();
                pos.y() = tmpPoints[n].y();
                pos.z() = tmpPoints[n].z();
            }
        }
    }
}


class ScalePositions
{
public:
    ScalePositions(
        const GU_Detail& gdp,
        UT_Vector3Array& positions,
        UT_FloatArray& distances,
        double scale = 1.0)
    : mGdp(gdp)
    , mPositions(positions)
    , mDistances(distances)
    , mScale(scale)
    {
    }

    void operator()(const GA_SplittableRange &range) const
    {
        GA_Offset start, end;
        GA_Index pointIndex;
        UT_Vector3 dir;

        GA_ROPageHandleV3 points(mGdp.getP());

        // Iterate over blocks
        for (GA_Iterator it(range); it.blockAdvance(start, end); ) {

            points.setPage(start);

            // Point Offsets
            for (GA_Offset pointOffset = start; pointOffset < end; ++pointOffset) {

                pointIndex = mGdp.pointIndex(pointOffset);

                const UT_Vector3& point = points.value(pointOffset);
                UT_Vector3& pos = mPositions(pointIndex);

                dir = pos - point;
                dir.normalize();

                pos = point + dir * mDistances(pointIndex) * mScale;
            }
        }

    }

private:
    const GU_Detail& mGdp;
    UT_Vector3Array& mPositions;
    UT_FloatArray& mDistances;
    const double mScale;
};

////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Ray::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        duplicateSource(0, context);

        hvdb::Interrupter boss("OpenVDB Ray");

        const GU_Detail* vdbGeo = inputGeo(1);
        if(vdbGeo == NULL) return error();

        // Get the group of grids to surface.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group =
            matchGroup(const_cast<GU_Detail&>(*vdbGeo), groupStr.toStdString());
        hvdb::VdbPrimCIterator vdbIt(vdbGeo, group);

        if (!vdbIt) {
            addWarning(SOP_MESSAGE, "No VDB grids found.");
            return error();
        }

        // Eval attributes
        const bool keepMaxDist = bool(evalInt("lookfar", 0, time));
        const bool reverseRays = bool(evalInt("reverserays", 0, time));
        const bool rayIntersection = evalInt("method", 0, time) == 0;
        const double scale = double(evalFloat("scale", 0, time));
        const double bias = double(evalFloat("bias", 0, time));
        const float isovalue = evalFloat("isovalue", 0, time);

        UT_Vector3Array pointNormals;

        GA_ROAttributeRef attributeRef = gdp->findPointAttribute("N");
        if (attributeRef.isValid()) {
#if (UT_VERSION_INT >= 0x0d0000c0)  // 13.0.192 or later
            gdp->getAttributeAsArray(
                attributeRef.getAttribute(), gdp->getPointRange(), pointNormals);
#else
            gdp->getPointAttributeAsArray(
                attributeRef.getAttribute(), gdp->getPointRange(), pointNormals);
#endif
        } else {
            gdp->normal(pointNormals, /*use_internaln=*/false);
        }


        const size_t numPoints = gdp->getNumPoints();

        UT_Vector3Array positions(numPoints);

        std::vector<char> intersections(numPoints);

        const double limit = std::numeric_limits<double>::max();
        UT_FloatArray distances;
        distances.appendMultiple((keepMaxDist && rayIntersection) ? -limit : limit, numPoints);



        std::vector<std::string> skippedGrids;

        for (; vdbIt; ++vdbIt) {
            if (boss.wasInterrupted()) break;

            std::vector<openvdb::Vec4s> spheres;

            if (vdbIt->getGrid().getGridClass() == openvdb::GRID_LEVEL_SET &&
                vdbIt->getGrid().type() == openvdb::FloatGrid::gridType()) {

                openvdb::FloatGrid::ConstPtr gridPtr =
                    openvdb::gridConstPtrCast<openvdb::FloatGrid>(vdbIt->getGridPtr());

                if (rayIntersection) {
                    IntersectPoints<openvdb::FloatGrid> op(
                        *gdp, pointNormals, *gridPtr, positions, distances,
                        intersections, keepMaxDist, reverseRays, scale, bias);
                    UTparallelFor(GA_SplittableRange(gdp->getPointRange()), op);
                } else {
                    closestPoints(*gridPtr, isovalue, *gdp, distances, &positions, boss);
                }

            } else {
                skippedGrids.push_back(vdbIt.getPrimitiveNameOrIndex().toStdString());
                continue;
            }
        }


        if (bool(evalInt("dotrans", 0, time))) { // update point positions

            if (!rayIntersection && !openvdb::math::isApproxEqual(scale, 1.0)) {
                UTparallelFor(GA_SplittableRange(gdp->getPointRange()),
                    ScalePositions(*gdp, positions, distances, scale));
            }

            gdp->setPos3FromArray(gdp->getPointRange(), positions);
        }

        if (bool(evalInt("putdist", 0, time))) { // add distance attribute

            GA_RWAttributeRef aRef = gdp->findPointAttribute("dist");
            if (!aRef.isValid()) {
                aRef = gdp->addIntTuple(GA_ATTRIB_POINT, "dist", 1, GA_Defaults(0));
            }
#if (UT_VERSION_INT >= 0x0d0000c0)  // 13.0.192 or later
            gdp->setAttributeFromArray(aRef.getAttribute(), gdp->getPointRange(), distances);
#else
            gdp->setPointAttributeFromArray(aRef.getAttribute(), gdp->getPointRange(), distances);
#endif
        }

        if (rayIntersection && bool(evalInt("creategroup", 0, time))) { // group intersecting points

            groupStr = "";
            evalString(groupStr, "hitgrp", 0, time);

            if(groupStr.length() > 0) {
                GA_PointGroup *pointGroup = gdp->findPointGroup(groupStr);
                if (!pointGroup) pointGroup = gdp->newPointGroup(groupStr);

                for (size_t n = 0; n < numPoints; ++n) {
                    if (intersections[n]) pointGroup->addIndex(n);
                }
            }
        }

        if (!skippedGrids.empty()) {
            std::string s = "Only level set grids are supported, the following "
                "were skipped: '" + boost::algorithm::join(skippedGrids, ", ") + "'.";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        if (boss.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "Process was interrupted");
        }

        boss.end();
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
