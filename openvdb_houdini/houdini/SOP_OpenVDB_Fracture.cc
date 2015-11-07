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
/// @file SOP_OpenVDB_Fracture.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Level set fracturing

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetFracture.h>

#include <openvdb/util/Util.h>

#include <GA/GA_ElementGroupTable.h>
#include <GA/GA_PageHandle.h>
#include <GA/GA_PageIterator.h>
#include <GA/GA_AttributeInstanceMatrix.h>
#include <GEO/GEO_PrimClassifier.h>
#include <GEO/GEO_PointClassifier.h>
#include <GU/GU_ConvertParms.h>
#include <UT/UT_Quaternion.h>
#include <UT/UT_ScopedPtr.h>
#include <UT/UT_ValArray.h>
#include <UT/UT_Version.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/constants/constants.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <vector>
#include <list>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_Fracture: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Fracture(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Fracture() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

    template <class GridType>
    void process(
        std::list<openvdb::GridBase::Ptr>& grids,
        const GU_Detail* cutterGeo,
        const GU_Detail* pointGeo,
        hvdb::Interrupter&,
        const fpreal time);
};

////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    //////////
    // Input options

    parms.add(hutil::ParmFactory(PRM_STRING, "inputgroup", "Group")
        .setHelpText("Select a subset of the input OpenVDB grids to fracture.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));


    //////////
    // Fracture options
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "separatecutters", "Separate Cutters by Connectivity")
      .setHelpText("Enable if multiple cutter objects are provided. This option is only available "
        "without instance points."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "cutteroverlap", "Allow Cutter Overlap")
        .setHelpText("Allow consecutive cutter instances to fracture previously "
            "generated fragments."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "centercutter", "Center Cutter Geometry")
#ifndef SESI_OPENVDB
        .setDefault(PRMoneDefaults)
#else
        .setDefault(PRMzeroDefaults)
#endif
        .setHelpText("Pre-center cutter geometry about the origin before instancing."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "randomizerotation", "Randomize Cutter Rotation")
        .setHelpText("Apply a random rotation to each instance point. This option is only "
            "available when instance points are provided."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "seed", "Random Seed")
        .setDefault(PRMoneDefaults)
        .setHelpText("Seed for the random rotation."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "segmentfragments",
        "Split Input Fragments into Primitives")
        .setHelpText(
            "Split grids with disjoint fragments into multiple grids, one per fragment. "
            "In a chain of fracture nodes this operation is typically only applied"
            " to the last node. "));

    parms.add(hutil::ParmFactory(PRM_STRING, "fragmentgroup", "Fragment Group")
        .setHelpText("Specify a group name in order to associate all fragments generated "
            "by this fracture. The residual fragments of the input grids are excluded "
            "from this group."));

    {
        static const char *visnames[] = {
            "none", "None",
            "all",  "Pieces",
            "new",  "New Fragments",
            0,      0
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "visualizepieces", "Visualization")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, visnames)
            .setHelpText("Randomize output primitive colors."));
    }

    //////////

    hvdb::OpenVDBOpFactory("OpenVDB Fracture", SOP_OpenVDB_Fracture::factory, parms, *table)
        .addInput("OpenVDB grids to fracture\n"
            "(Required to have matching transforms and narrow band widths)")
        .addInput("Cutter objects (geometry).")
        .addOptionalInput("Optional points to instance the cutter object onto\n"
            "(The cutter object is used in place if no points are provided.)");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Fracture::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Fracture(net, name, op);
}


SOP_OpenVDB_Fracture::SOP_OpenVDB_Fracture(OP_Network* net,
    const char* name, OP_Operator* op): hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Fracture::updateParmsFlags()
{
    bool changed = false;

    const bool instancePointsExist = (nInputs() == 3);
    const bool multipleCutters = bool(evalInt("separatecutters", 0, 0));
    const bool randomizeRotation = bool(evalInt("randomizerotation", 0, 0));

    changed |= enableParm("separatecutters", !instancePointsExist);
    changed |= enableParm("centercutter", instancePointsExist);
    changed |= enableParm("randomizerotation", instancePointsExist);
    changed |= enableParm("seed", randomizeRotation);
    changed |= enableParm("cutteroverlap", instancePointsExist || multipleCutters);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Fracture::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();
        duplicateSourceStealable(0, context);

        hvdb::Interrupter boss("Converting geometry to volume");

        //////////
        // Validate inputs

        const GU_Detail* cutterGeo = inputGeo(1);
        if (!cutterGeo || !cutterGeo->getNumPrimitives()) {
            // All good, nothing to worry about with no cutting objects!
            return error();
        }


        std::string warningStr;
        boost::shared_ptr<GU_Detail> geoPtr =
            hvdb::validateGeometry(*cutterGeo, warningStr, &boss);

        if (geoPtr) {
            cutterGeo = geoPtr.get();
            if (!warningStr.empty()) addWarning(SOP_MESSAGE, warningStr.c_str());
        }

        const GU_Detail* pointGeo = inputGeo(2);

        const GA_PrimitiveGroup *group = NULL;
        {
            UT_String str;
            evalString(str, "inputgroup", 0, time);
            group = matchGroup(*gdp, str.toStdString());
        }

        std::list<openvdb::GridBase::Ptr> grids;
        std::vector<GU_PrimVDB*> origvdbs;

        std::vector<std::string> nonLevelSetList, nonLinearList;

        for (hvdb::VdbPrimIterator vdbIter(gdp, group); vdbIter; ++vdbIter) {

            if (boss.wasInterrupted()) break;

            const openvdb::GridClass gridClass = vdbIter->getGrid().getGridClass();
            if (gridClass != openvdb::GRID_LEVEL_SET) {
                nonLevelSetList.push_back(vdbIter.getPrimitiveNameOrIndex().toStdString());
                continue;
            }

            if (!vdbIter->getGrid().transform().isLinear()) {
                nonLinearList.push_back(vdbIter.getPrimitiveNameOrIndex().toStdString());
                continue;
            }

            GU_PrimVDB* vdb = vdbIter.getPrimitive();

            vdb->makeGridUnique();

            grids.push_back(vdb->getGrid().copyGrid());
            grids.back()->setName(vdb->getGridName());

            grids.back()->insertMeta("houdiniorigoffset",
                openvdb::Int64Metadata( vdb->getMapOffset() ) );

            origvdbs.push_back(vdb);
        }

        if (!nonLevelSetList.empty()) {
            std::string s = "The following non level set grids were skipped: '" +
                boost::algorithm::join(nonLevelSetList, ", ") + "'.";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        if (!nonLinearList.empty()) {
            std::string s = "The following grids were skipped: '" +
                boost::algorithm::join(nonLinearList, ", ") +
                "' because they don't have a linear/affine transform.";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        if (!grids.empty() && !boss.wasInterrupted()) {

            if (grids.front()->isType<openvdb::FloatGrid>()) {
                process<openvdb::FloatGrid>(grids, cutterGeo, pointGeo, boss, time);
            } else if (grids.front()->isType<openvdb::DoubleGrid>()) {
                process<openvdb::DoubleGrid>(grids, cutterGeo, pointGeo, boss, time);
            } else {
                addError(SOP_MESSAGE, "Unsupported grid type");
            }

            for (std::vector<GU_PrimVDB*>::iterator it = origvdbs.begin();
                it != origvdbs.end(); ++it)
            {
                gdp->destroyPrimitive(**it, /*andPoints=*/true);
            }
        } else {
             addWarning(SOP_MESSAGE, "No VDB grids to fracture");
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


////////////////////////////////////////


template <class GridType>
void
SOP_OpenVDB_Fracture::process(
    std::list<openvdb::GridBase::Ptr>& grids,
    const GU_Detail* cutterGeo,
    const GU_Detail* pointGeo,
    hvdb::Interrupter& boss,
    const fpreal time)
{
    GA_PrimitiveGroup* group = NULL;

    // Evaluate the UI parameters.

    {
        UT_String newGropStr;
        evalString(newGropStr, "fragmentgroup", 0, time);
        if(newGropStr.length() > 0) {
            group = gdp->findPrimitiveGroup(newGropStr);
            if (!group) group = gdp->newPrimitiveGroup(newGropStr);
        }
    }

    const bool randomizeRotation = bool(evalInt("randomizerotation", 0, time));
    const bool cutterOverlap = bool(evalInt("cutteroverlap", 0, time));
    const int visualization = evalInt("visualizepieces", 0, time);
    const bool segmentFragments = bool(evalInt("segmentfragments", 0, time));

    typedef typename GridType::ValueType ValueType;

    typename GridType::Ptr firstGrid = openvdb::gridPtrCast<GridType>(grids.front());
    if (!firstGrid) {
        addError(SOP_MESSAGE, "Unsupported grid type.");
        return;
    }

    // Get the first grid's transform and background value.
    openvdb::math::Transform::Ptr transform = firstGrid->transformPtr();
    const ValueType backgroundValue = firstGrid->background();

    std::vector<openvdb::Vec3s> instancePoints;
    std::vector<openvdb::math::Quats> instanceRotations;

#if (UT_VERSION_INT >= 0x0d000035) // 13.0.53 or later
    if (pointGeo != NULL) {
        instancePoints.resize(pointGeo->getNumPoints());

        GA_Range range(pointGeo->getPointRange());
        GA_AttributeInstanceMatrix instanceMatrix;
        instanceMatrix.initialize(pointGeo->pointAttribs(), "N", "v");
        // Ignore any scaling until levelset fracture supports it.
        instanceMatrix.resetScales();

        // If we're randomizing or there are *any* valid transformation
        // attributes, we need to create an instance matrix.
        if (randomizeRotation || instanceMatrix.hasAnyAttribs()) {
            instanceRotations.resize(instancePoints.size());
            typedef boost::mt19937 RandGen;
            RandGen rng(RandGen::result_type(evalInt("seed", 0, time)));
            boost::uniform_01<RandGen, float> uniform01(rng);
            const float two_pi = 2.0f * boost::math::constants::pi<float>();
            UT_DMatrix4 xform;
            UT_Vector3 trans;
            UT_DMatrix3 rotmat;
            UT_QuaternionD quat;

            for (GA_Iterator it(range); !it.atEnd(); it.advance()) {
                UT_Vector3 pos = pointGeo->getPos3(*it);

                if (randomizeRotation) {
                    // Generate uniform random rotations by picking random
                    // points in the unit cube and forming the unit quaternion.

                    const float u  = uniform01();
                    const float c1 = std::sqrt(1-u);
                    const float c2 = std::sqrt(u);
                    const float s1 = two_pi * uniform01();
                    const float s2 = two_pi * uniform01();

                    UT_Quaternion  orient(c1 * std::sin(s1), c1 * std::cos(s1),
                                          c2 * std::sin(s2), c2 * std::cos(s2));
                    instanceMatrix.getMatrix(xform, pos, orient, *it);
                }
                else {
                    instanceMatrix.getMatrix(xform, pos, *it);
                }
                GA_Index i = pointGeo->pointIndex(*it);
                xform.getTranslates(trans);
                xform.extractRotate(rotmat);
                quat.updateFromRotationMatrix(rotmat);
                instancePoints[i] = openvdb::Vec3s(trans.x(), trans.y(), trans.z());
                instanceRotations[i].init(
                    static_cast<float>(quat.x()),
                    static_cast<float>(quat.y()),
                    static_cast<float>(quat.z()),
                    static_cast<float>(quat.w()));
            }
        }
        else
        {
            // No randomization or valid instance attributes, just use P.
            for (GA_Iterator it(range); !it.atEnd(); it.advance()) {
                UT_Vector3 pos = pointGeo->getPos3(*it);
                instancePoints[pointGeo->pointIndex(*it)] =
                    openvdb::Vec3s(pos.x(), pos.y(), pos.z());
            }
        }
    }
#else // before 13.0.53
    if (pointGeo != NULL) {
        instancePoints.resize(pointGeo->getNumPoints());

        GA_Range range(pointGeo->getPointRange());
        GA_Offset start, end;

        // Copy world space instance points.
        GA_ROPageHandleV3 attribP(pointGeo->getP());
        for (GA_Iterator it(range); it.blockAdvance(start, end); ) {
            attribP.setPage(start);
            for (GA_Offset i = start; i < end; ++i) {
                UT_Vector3 pos = attribP.get(i);
                instancePoints[pointGeo->pointIndex(i)] =
                    openvdb::Vec3s(pos.x(), pos.y(), pos.z());
            }
        }

        // Add instance offset if found
        GA_ROPageHandleV3 attribTrans(pointGeo, GA_ATTRIB_POINT, "trans");
        if (attribTrans.isValid()) {
            for (GA_Iterator it(range); it.blockAdvance(start, end); ) {
                attribTrans.setPage(start);
                for (GA_Offset i = start; i < end; ++i) {
                    UT_Vector3 trans = attribTrans.get(i);
                    instancePoints[pointGeo->pointIndex(i)] +=
                        openvdb::Vec3s(trans.x(), trans.y(), trans.z());
                }
            }
        }

        if (randomizeRotation) {

            instanceRotations.resize(instancePoints.size());

            // Generate uniform random rotations by picking random points
            // in the unit cube and forming the unit quaternion.
            typedef boost::mt19937 RandGen;
            RandGen rng(RandGen::result_type(evalInt("seed", 0, time)));
            boost::uniform_01<RandGen, float> uniform01(rng);
            const float two_pi = 2.0 * boost::math::constants::pi<float>();
            for (size_t n = 0, N = instanceRotations.size(); n < N; ++n) {

                const float u  = uniform01();
                const float c1 = std::sqrt(1-u);
                const float c2 = std::sqrt(u);
                const float s1 = two_pi * uniform01();
                const float s2 = two_pi * uniform01();

                instanceRotations[n][0] = c1 * std::sin(s1);
                instanceRotations[n][1] = c1 * std::cos(s1);
                instanceRotations[n][2] = c2 * std::sin(s2);
                instanceRotations[n][3] = c2 * std::cos(s2);
            }
        } else {

            GA_ROAttributeRef refN = pointGeo->findNormalAttribute(GA_ATTRIB_POINT);
            if (!refN.isValid())
                refN = pointGeo->findVelocityAttribute(GA_ATTRIB_POINT);
            GA_ROHandleV3 attrN(refN.getAttribute());
            GA_ROHandleV3 attrUp(pointGeo, GA_ATTRIB_POINT, "up");
            GA_ROHandleQ attrRot(pointGeo, GA_ATTRIB_POINT, "rot");
            GA_ROHandleQ attrOrient(pointGeo, GA_ATTRIB_POINT, "orient");

            if (attrN.isValid() || attrUp.isValid() ||
                attrRot.isValid() || attrOrient.isValid()) {

                instanceRotations.resize(instancePoints.size());
                for (size_t i = 0, n = instanceRotations.size(); i < n; ++i) {

                    GA_Offset ptoff = pointGeo->pointOffset(i);
                    UT_Matrix4D mat4;
                    UT_QuaternionD quat;
                    UT_Vector3 normal(0.0, 0.0, 0.0);
                    UT_Vector3 up(0.0, 0.0, 0.0);
                    UT_Quaternion rot;
                    UT_Quaternion orient;

                    if (attrN.isValid())
                        normal = attrN.get(ptoff);
                    if (attrUp.isValid())
                        up = attrUp.get(ptoff);
                    if (attrRot.isValid())
                        rot = attrRot.get(ptoff);
                    if (attrOrient.isValid())
                        orient = attrOrient.get(ptoff);

                    mat4.instance(UT_Vector3(0.0, 0.0, 0.0),
                        normal,
                        /*s*/1.0, /*s3*/NULL,
                        attrUp.isValid() ? &up : NULL,
                        attrRot.isValid() ? &rot : NULL,
                        /*trans*/NULL,
                        attrOrient.isValid() ? &orient : NULL);

                    quat.updateFromRotationMatrix(UT_Matrix3(mat4));
                    instanceRotations[i].init(quat.x(), quat.y(), quat.z(), quat.w());
                }
            }
        }
    }
#endif
    if (boss.wasInterrupted()) return;

    std::list<typename GridType::Ptr> residuals;

    {
        std::list<openvdb::GridBase::Ptr>::iterator it = grids.begin();

        std::vector<std::string> badTransformList, badBackgroundList, badTypeList;

        for (; it != grids.end(); ++it) {
            typename GridType::Ptr residual = openvdb::gridPtrCast<GridType>(*it);

            if (residual) {

                if (residual->transform() != *transform) {
                    badTransformList.push_back(residual->getName());
                    continue;
                }

                if (!openvdb::math::isApproxEqual(residual->background(), backgroundValue)) {
                    badBackgroundList.push_back(residual->getName());
                    continue;
                }

                residuals.push_back(residual);

            } else {
                badTypeList.push_back(residual->getName());
                continue;
            }

        }
        grids.clear();


        if (!badTransformList.empty()) {
            std::string s = "The following grids were skipped: '" +
                boost::algorithm::join(badTransformList, ", ") +
                "' because they don't match the transform of the first grid.";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        if (!badBackgroundList.empty()) {
            std::string s = "The following grids were skipped: '" +
                boost::algorithm::join(badBackgroundList, ", ") +
                "' because they don't match the background value of the first grid.";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        if (!badTypeList.empty()) {
            std::string s = "The following grids were skipped: '" +
                boost::algorithm::join(badTypeList, ", ") +
                "' because they don't have the same data type as the first grid.";
            addWarning(SOP_MESSAGE, s.c_str());
        }
    }

    // Setup fracture tool
    openvdb::tools::LevelSetFracture<GridType, hvdb::Interrupter> lsFracture(&boss);

    const bool separatecutters = (pointGeo == NULL) && bool(evalInt("separatecutters", 0, time));

    std::vector<openvdb::Vec3s> pointList;

    {
        pointList.resize(cutterGeo->getNumPoints());
        openvdb::math::Transform::Ptr xform = transform->copy();

        if (!instancePoints.empty() && !separatecutters && bool(evalInt("centercutter", 0, time))) {
            UT_BoundingBox pointBBox;
            cutterGeo->getPointBBox(&pointBBox);
            UT_Vector3 center = pointBBox.center();
            xform->postTranslate(openvdb::Vec3s(center.x(), center.y(), center.z()));
        }

        UTparallelFor(GA_SplittableRange(cutterGeo->getPointRange()),
                hvdb::TransformOp(cutterGeo, *xform, pointList));
    }

    // Check for multiple cutter objects
#if (UT_VERSION_INT >= 0x0e000061) // 14.0.97 or later
    GEO_PrimClassifier primClassifier;
    if (separatecutters) {
        primClassifier.classifyBySharedPoints(*cutterGeo);
    }
#else
    GEO_PointClassifier pointClassifier;
    GEO_PrimClassifier primClassifier;
    if (separatecutters) {
        pointClassifier.classifyPoints(*cutterGeo);
        primClassifier.classifyBySharedPoints(*cutterGeo, pointClassifier);
    }
#endif

    const int cutterObjects = separatecutters ? primClassifier.getNumClass() : 1;
    const float bandWidth = float(backgroundValue / transform->voxelSize()[0]);

    if (cutterObjects > 1) {
        GA_Offset start, end;
        GA_Primitive::const_iterator vtxIt;
        GA_SplittableRange range(cutterGeo->getPrimitiveRange());

        for (int classId = 0; classId < cutterObjects; ++classId) {

            if (boss.wasInterrupted()) break;

            size_t numPrims = 0;
            for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
                for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
                    for (GA_Offset i = start; i < end; ++i) {
                        if (primClassifier.getClass(cutterGeo->primitiveIndex(i)) == classId) {
                            ++numPrims;
                        }
                    }
                }
            }

            typename GridType::Ptr cutterGrid;

            if (numPrims == 0) continue;

            {
                std::vector<openvdb::Vec4I> primList;
                primList.reserve(numPrims);

                openvdb::Vec4I prim;
                typedef openvdb::Vec4I::ValueType Vec4IValueType;
                unsigned int vtx;
                GA_Size vtxn;

                for (GA_PageIterator pageIt = range.beginPages(); !pageIt.atEnd(); ++pageIt) {
                    for (GA_Iterator blockIt(pageIt.begin()); blockIt.blockAdvance(start, end); ) {
                        for (GA_Offset i = start; i < end; ++i) {
                            if (primClassifier.getClass(cutterGeo->primitiveIndex(i)) == classId) {
                                const GA_Primitive* primRef = cutterGeo->getPrimitiveList().get(i);
                                vtxn = primRef->getVertexCount();

                                if (primRef->getTypeId() == GEO_PRIMPOLY &&
                                    (3 == vtxn || 4 == vtxn))
                                {
                                    GA_Primitive::const_iterator it;
                                    for (vtx = 0, primRef->beginVertex(it);
                                        !it.atEnd(); ++it, ++vtx)
                                    {
                                        prim[vtx] = static_cast<Vec4IValueType>(
                                            cutterGeo->pointIndex(it.getPointOffset()));
                                    }

                                    if (vtxn != 4) prim[3] = openvdb::util::INVALID_IDX;

                                    primList.push_back(prim);
                                }
                            }
                        }
                    }
                }

                openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> mesh(pointList, primList);

                cutterGrid = openvdb::tools::meshToVolume<GridType>(boss, mesh, *transform, bandWidth, bandWidth);
            }

            if (!cutterGrid || cutterGrid->activeVoxelCount() == 0) continue;

            lsFracture.fracture(residuals, *cutterGrid, segmentFragments,
                NULL, NULL, cutterOverlap);
        }
    } else {

        // Convert cutter object mesh to level-set
        typename GridType::Ptr cutterGrid;

        {
            std::vector<openvdb::Vec4I> primList;
            primList.resize(cutterGeo->getNumPrimitives());

            UTparallelFor(GA_SplittableRange(cutterGeo->getPrimitiveRange()),
                hvdb::PrimCpyOp(cutterGeo, primList));


            openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec4I> mesh(pointList, primList);

            cutterGrid = openvdb::tools::meshToVolume<GridType>(boss, mesh, *transform, bandWidth, bandWidth);
        }

        if (!cutterGrid || cutterGrid->activeVoxelCount() == 0 || boss.wasInterrupted()) return;


        lsFracture.fracture(residuals, *cutterGrid, segmentFragments, &instancePoints,
            &instanceRotations, cutterOverlap);

    }


    if (boss.wasInterrupted()) return;

    typename std::list<typename GridType::Ptr>::iterator it;

    // Primitive Color
    GA_RWHandleV3 color;
    if (visualization) {
        GA_RWAttributeRef attrRef = gdp->findDiffuseAttribute(GA_ATTRIB_PRIMITIVE);
        if (!attrRef.isValid()) attrRef = gdp->addDiffuseAttribute(GA_ATTRIB_PRIMITIVE);
        color.bind(attrRef.getAttribute());
    }

    UT_IntArray piececount;
    UT_IntArray totalpiececount;

    piececount.entries(gdp->getNumPrimitiveOffsets());
    totalpiececount.entries(gdp->getNumPrimitiveOffsets());

    boost::mt19937 rng(1);
    boost::uniform_real<float> range(0.3f, 0.8f);
    boost::variate_generator<boost::mt19937, boost::uniform_real<float> > randNr(rng, range);

    GU_ConvertParms parms;
    parms.preserveGroups = true;

    // Export residual fragments
    exint coloridx = 0;
    GA_RWHandleS name_h(gdp, GA_ATTRIB_PRIMITIVE, "name");

    // We have to do a pre-pass over all pieces to compute the total
    // number of pieces from each original object.  This way
    // we can tell if we need to do renaming or not.
    for (it = residuals.begin(); it != residuals.end(); ++it) {

        GA_Offset origvdboff = GA_INVALID_OFFSET;

        typename GridType::Ptr grid = *it;
        openvdb::Int64Metadata::Ptr offmeta =
            grid->template getMetadata<openvdb::Int64Metadata>("houdiniorigoffset");
        if (offmeta) {
            origvdboff = (GA_Offset)offmeta->value();
        }
        if (origvdboff != GA_INVALID_OFFSET) {
            totalpiececount(origvdboff)++;
        }
    }
    for (it = lsFracture.fragments().begin(); it != lsFracture.fragments().end(); ++it) {

        GA_Offset origvdboff = GA_INVALID_OFFSET;

        typename GridType::Ptr grid = *it;
        openvdb::Int64Metadata::Ptr offmeta =
            grid->template getMetadata<openvdb::Int64Metadata>("houdiniorigoffset");
        if (offmeta) {
            origvdboff = (GA_Offset)offmeta->value();
        }
        if (origvdboff != GA_INVALID_OFFSET) {
            totalpiececount(origvdboff)++;
        }
    }

    for (it = residuals.begin(); it != residuals.end(); ++it) {
        if (boss.wasInterrupted()) break;

        typename GridType::Ptr grid = *it;
        GA_Offset origvdboff = GA_INVALID_OFFSET;

        openvdb::Int64Metadata::Ptr offmeta =
            grid->template getMetadata<openvdb::Int64Metadata>("houdiniorigoffset");
        if (offmeta) {
            origvdboff = (GA_Offset)offmeta->value();
            grid->removeMeta("houdiniorigoffset");
        }

        std::string gridname = grid->getName();
        UT_String name;

        name.harden(gridname.c_str());

        // Suffix our name.
        if (name.isstring() && origvdboff != GA_INVALID_OFFSET
            && totalpiececount(origvdboff) > 1)
        {
            UT_WorkBuffer buf;
            buf.sprintf("%s_%d", (const char *) name, piececount(origvdboff));
            piececount(origvdboff)++;
            name.harden(buf.buffer());
        }

        GU_PrimVDB* vdb = hvdb::createVdbPrimitive(*gdp, grid, (const char *) name);

        if (origvdboff != GA_INVALID_OFFSET)
        {
            GU_PrimVDB* origvdb =
                dynamic_cast<GU_PrimVDB*>(gdp->getGEOPrimitive(origvdboff));

            if (origvdb)
            {
                GA_Offset newvdbpt;

                newvdbpt = vdb->getPointOffset(0);
                GUconvertCopySingleVertexPrimAttribsAndGroups(
                    parms,
                    *origvdb->getParent(),
                    origvdb->getMapOffset(),
                    *gdp,
                    GA_Range(gdp->getPrimitiveMap(), vdb->getMapOffset(), vdb->getMapOffset()+1),
                    GA_Range(gdp->getPointMap(), newvdbpt, newvdbpt+1));
            }
        }

        if (visualization == 1 && color.isValid())
        {
            float r, g, b;
            UT_Color::getUniqueColor(coloridx, &r, &g, &b);
            color.set(vdb->getMapOffset(), UT_Vector3(r, g, b));
        }
        coloridx++;

        if (name.isstring() && name_h.isValid())
        {
            name_h.set(vdb->getMapOffset(), (const char *) name);
        }
    }

    if (boss.wasInterrupted()) return;

    // Export new fragments
    for (it = lsFracture.fragments().begin(); it != lsFracture.fragments().end(); ++it) {

        if (boss.wasInterrupted()) break;

        typename GridType::Ptr grid = *it;

        GA_Offset origvdboff = GA_INVALID_OFFSET;

        openvdb::Int64Metadata::Ptr offmeta =
            grid->template getMetadata<openvdb::Int64Metadata>("houdiniorigoffset");
        if (offmeta) {
            origvdboff = (GA_Offset)offmeta->value();
            grid->removeMeta("houdiniorigoffset");
        }

        std::string gridname = grid->getName();
        UT_String name;

        name.harden(gridname.c_str());

        // Suffix our name.
        if (name.isstring() && origvdboff != GA_INVALID_OFFSET)
        {
            UT_WorkBuffer buf;
            buf.sprintf("%s_%d", (const char *) name, piececount(origvdboff));
            piececount(origvdboff)++;
            name.harden(buf.buffer());
        }
        GU_PrimVDB* vdb = hvdb::createVdbPrimitive(*gdp, grid, (const char *) name);
        if (origvdboff != GA_INVALID_OFFSET)
        {
            GU_PrimVDB* origvdb =
                dynamic_cast<GU_PrimVDB*>(gdp->getGEOPrimitive(origvdboff));

            if (origvdb)
            {
                GA_Offset newvdbpt;

                newvdbpt = vdb->getPointOffset(0);
                GUconvertCopySingleVertexPrimAttribsAndGroups(
                    parms,
                    *origvdb->getParent(),
                    origvdb->getMapOffset(),
                    *gdp,
                    GA_Range(gdp->getPrimitiveMap(), vdb->getMapOffset(), vdb->getMapOffset()+1),
                    GA_Range(gdp->getPointMap(), newvdbpt, newvdbpt+1));
            }
        }
        if (name.isstring() && name_h.isValid())
        {
            name_h.set(vdb->getMapOffset(), (const char *) name);
        }

        if (group) group->add(vdb);

        if (visualization && color.isValid())
        {
            float r, g, b;
            UT_Color::getUniqueColor(coloridx++, &r, &g, &b);
            color.set(vdb->getMapOffset(), UT_Vector3(r, g, b));
        }
        coloridx++;
    }
}

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
