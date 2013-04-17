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
/// @file SOP_OpenVDB_To_Polygons.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief OpenVDB level set to polygon conversion

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/AttributeTransferUtil.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/GeometryUtil.h>

#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/math/Operators.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_PageIterator.h>
#include <GU/GU_Detail.h>
#include <GU/GU_Surfacer.h>
#include <GU/GU_PolyReduce.h>
#include <GU/GU_PrimPoly.h>
#include <PRM/PRM_Parm.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_floating_point.hpp>

#include <string>
#include <list>
#include <vector>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_To_Polygons: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_To_Polygons(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_To_Polygons() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual unsigned disableParms();

    template <class GridType>
    void referenceMeshing(
        std::list<openvdb::GridBase::ConstPtr>& grids,
        openvdb::tools::VolumeToMesh& mesher,
        const GU_Detail* refGeo,
        hvdb::Interrupter& boss,
        const fpreal time);
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to surface.")
        .setChoiceList(&hutil::PrimGroupMenu));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "adaptivity", "Adaptivity")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setHelpText("Adaptivity threshold. (If a reference surface is "
            "provided this will only control the adaptivity in regions "
            "that are not interior to the reference surface.)"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "isovalue", "Iso Value")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "computenormals", "Compute Vertex Normals"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "keepvdbname", "Preserve VDB Name"));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR,"sep1", ""));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "internaladaptivity", "Internal Adaptivity")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setHelpText("Adaptivity threshold for regions that are "
            "interior to the reference surface."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "transferattributes", "Transfer Surface Attributes"));
    //parms.add(hutil::ParmFactory(PRM_TOGGLE, "reconstruct", "Reconstruct Sharp Features"));

    parms.add(hutil::ParmFactory(PRM_STRING, "surfacegroup", "Surface Group")
        .setDefault("surface_polygons")
        .setHelpText("Group all polygons that are coincident with the reference "
            "surface. (Attributes from the reference surface such as uv "
            "coordinates, normals etc. can be transfered to these polygons.)"));

    parms.add(hutil::ParmFactory(PRM_STRING, "interiorgroup", "Interior Group")
        .setDefault("interior_polygons")
        .setHelpText("Group all polygons that are interior to reference surface. "
            "(These might need projected UV coordinates or a different material.)"));

    parms.add(hutil::ParmFactory(PRM_STRING, "seamlinegroup", "Seam Line Group")
        .setDefault("seam_polygons")
        .setHelpText("Group all polygons that are in proximity to the seam lines. "
            "(Used to drive secondary elements such as debris and dust.)"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB To Polygons", SOP_OpenVDB_To_Polygons::factory, parms, *table)
        .addInput("OpenVDB grids to surface")
        .addOptionalInput("Optional reference surface. (When surfacing fractured fragments, "
            "the unfractured level set or polygonal model can be used to eliminate seams.");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_To_Polygons::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_To_Polygons(net, name, op);
}


SOP_OpenVDB_To_Polygons::SOP_OpenVDB_To_Polygons(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


// Disable UI Parms.
unsigned
SOP_OpenVDB_To_Polygons::disableParms()
{
    unsigned changed = 0;

    int refexists = (nInputs() == 2);
    changed += enableParm("internaladaptivity", refexists);
    changed += enableParm("surfacegroup", refexists);
    changed += enableParm("interiorgroup", refexists);
    changed += enableParm("seamlinegroup", refexists);
    changed += enableParm("transferattributes", refexists);
    //changed += enableParm("reconstruct", refexists);

    return changed;
}


////////////////////////////////////////


void
copyMesh(
    GU_Detail& detail,
    openvdb::tools::VolumeToMesh& mesher,
    hvdb::Interrupter& boss,
    const char* gridName = NULL,
    GA_PrimitiveGroup* surfaceGroup = NULL,
    GA_PrimitiveGroup* interiorGroup = NULL,
    GA_PrimitiveGroup* seamGroup = NULL)
{
    const openvdb::tools::PointList& points = mesher.pointList();
    openvdb::tools::PolygonPoolList& polygonPoolList = mesher.polygonPoolList();

    const GA_Offset lastIdx(detail.getNumPoints());

    for (size_t n = 0, N = mesher.pointListSize(); n < N; ++n) {
        GA_Offset ptoff = detail.appendPointOffset();
        detail.setPos3(ptoff, points[n].x(), points[n].y(), points[n].z());
    }

    if (boss.wasInterrupted()) return;

    const char exteriorFlag = char(openvdb::tools::POLYFLAG_EXTERIOR);
    const char seamLineFlag = char(openvdb::tools::POLYFLAG_FRACTURE_SEAM);


    const GA_Index firstPrim = detail.primitives().entries();

    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {

        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        if (boss.wasInterrupted()) break;

        // Copy quads
        for (size_t i = 0, I = polygons.numQuads(); i < I; ++i) {

            openvdb::Vec4I& quad = polygons.quad(i);
            GEO_PrimPoly& prim = *GU_PrimPoly::build(&detail, 4, GU_POLY_CLOSED, 0);

            for (int v = 0; v < 4; ++v) {
                prim(v).setPointOffset(lastIdx + quad[v]);
            }

            const bool surfacePrim = polygons.quadFlags(i) & exteriorFlag;
            if (surfaceGroup && surfacePrim) surfaceGroup->add(&prim);
            else if (interiorGroup && !surfacePrim) interiorGroup->add(&prim);

            if (seamGroup && (polygons.quadFlags(i) & seamLineFlag)) {
                seamGroup->add(&prim);
            }
        }

        if (boss.wasInterrupted()) break;

        // Copy triangles (if adaptive mesh.)
        for (size_t i = 0, I = polygons.numTriangles(); i < I; ++i) {

            openvdb::Vec3I& triangle = polygons.triangle(i);
            GEO_PrimPoly& prim = *GU_PrimPoly::build(&detail, 3, GU_POLY_CLOSED, 0);

            for (int v = 0; v < 3; ++v) {
                prim(v).setPointOffset(lastIdx + triangle[v]);
            }

            const bool surfacePrim = (polygons.triangleFlags(i) & exteriorFlag);
            if (surfaceGroup && surfacePrim) surfaceGroup->add(&prim);
            else if (interiorGroup && !surfacePrim) interiorGroup->add(&prim);

            if (seamGroup && (polygons.triangleFlags(i) & seamLineFlag)) {
                seamGroup->add(&prim);
            }
        }
    }

    // Keep VDB grid name
    const GA_Index lastPrim = detail.primitives().entries();
    if (gridName != NULL && firstPrim != lastPrim) {

        GA_RWAttributeRef aRef = detail.findPrimitiveAttribute("name");
        if (!aRef.isValid()) aRef = detail.addStringTuple(GA_ATTRIB_PRIMITIVE, "name", 1);

        GA_Attribute * nameAttr = aRef.getAttribute();
        if (nameAttr) {
            const GA_AIFSharedStringTuple * stringAIF = nameAttr->getAIFSharedStringTuple();
            if (stringAIF) {
                GA_Range range(detail.getPrimitiveMap(),
                    detail.primitiveOffset(firstPrim),  detail.primitiveOffset(lastPrim));
                stringAIF->setString(nameAttr, range, gridName, 0);
            }
        }
    }
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_To_Polygons::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        gdp->clearAndDestroy();

        hvdb::Interrupter boss("Surfacing VDB primitives");

        const GU_Detail* vdbGeo = inputGeo(0);
        if(vdbGeo == NULL) return error();

        // Get the group of grids to surface.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group =
            matchGroup(const_cast<GU_Detail&>(*vdbGeo), groupStr.toStdString());
        hvdb::VdbPrimCIterator vdbIt(vdbGeo, group);

        if (!vdbIt) {
            addWarning(SOP_MESSAGE, "No VDB primitives found.");
            return error();
        }

        // Eval attributes
        const double adaptivity = double(evalFloat("adaptivity", 0, time));
        const double iso = double(evalFloat("isovalue", 0, time));
        const bool computeNormals = evalInt("computenormals", 0, time);
        const bool keepVdbName = evalInt("keepvdbname", 0, time);

        // Setup level set mesher
        openvdb::tools::VolumeToMesh mesher(iso, adaptivity);

        // Check reference input
        const GU_Detail* refGeo = inputGeo(1);
        if (refGeo) {

            // Collect all level set grids.
            std::list<openvdb::GridBase::ConstPtr> grids;
            std::vector<std::string> nonLevelSetList, nonLinearList;
            for (; vdbIt; ++vdbIt) {
                if (boss.wasInterrupted()) break;

                const openvdb::GridClass gridClass = vdbIt->getGrid().getGridClass();
                if (gridClass != openvdb::GRID_LEVEL_SET) {
                    nonLevelSetList.push_back(vdbIt.getPrimitiveNameOrIndex().toStdString());
                    continue;
                }

                if (!vdbIt->getGrid().transform().isLinear()) {
                    nonLinearList.push_back(vdbIt.getPrimitiveNameOrIndex().toStdString());
                    continue;
                }

                grids.push_back(vdbIt->getConstGridPtr());
            }

            if (!nonLevelSetList.empty()) {
                std::string s = "Reference meshing is only supported for "
                    "Level Set grids, the following grids were skipped: '" +
                    boost::algorithm::join(nonLevelSetList, ", ") + "'.";
                addWarning(SOP_MESSAGE, s.c_str());
            }

            if (!nonLinearList.empty()) {
                std::string s = "The following grids were skipped: '" +
                    boost::algorithm::join(nonLinearList, ", ") +
                    "' because they don't have a linear/affine transform.";
                addWarning(SOP_MESSAGE, s.c_str());
            }

            // Mesh using a reference surface
            if (!grids.empty() && !boss.wasInterrupted()) {

                if (grids.front()->isType<openvdb::FloatGrid>()) {
                    referenceMeshing<openvdb::FloatGrid>(grids, mesher, refGeo, boss, time);
                } else if (grids.front()->isType<openvdb::DoubleGrid>()) {
                    referenceMeshing<openvdb::DoubleGrid>(grids, mesher, refGeo, boss, time);
                } else {
                    addError(SOP_MESSAGE, "Unsupported grid type.");
                }
            }

        } else {

            // Mesh each VDB primitive independently
            for (; vdbIt; ++vdbIt) {

                if (boss.wasInterrupted()) break;
                GEOvdbProcessTypedGridScalar(*vdbIt.getPrimitive(), mesher);
                copyMesh(*gdp, mesher, boss,
                    keepVdbName ? vdbIt.getPrimitive()->getGridName() : NULL);
            }

            if (!boss.wasInterrupted() && computeNormals) {
                UTparallelFor(GA_SplittableRange(gdp->getPrimitiveRange()),
                    hvdb::VertexNormalOp(*gdp));
            }
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


template<class GridType>
void
SOP_OpenVDB_To_Polygons::referenceMeshing(
    std::list<openvdb::GridBase::ConstPtr>& grids,
    openvdb::tools::VolumeToMesh& mesher,
    const GU_Detail* refGeo,
    hvdb::Interrupter& boss,
    const fpreal time)
{
    if (refGeo == NULL) return;
    const bool computeNormals = evalInt("computenormals", 0, time);
    const bool transferAttributes = evalInt("transferattributes", 0, time);
    const bool keepVdbName = evalInt("keepvdbname", 0, time);
    //const bool reconstruct = evalInt("reconstruct", 0, time);

    typedef typename GridType::TreeType TreeType;
    typedef typename GridType::ValueType ValueType;

    // Get the first grid's transform and background value.
    openvdb::math::Transform::Ptr transform = grids.front()->transform().copy();

    typename GridType::ConstPtr firstGrid = openvdb::gridConstPtrCast<GridType>(grids.front());

    if (!firstGrid) {
        addError(SOP_MESSAGE, "Unsupported grid type.");
        return;
    }

    const ValueType backgroundValue = firstGrid->background();
    const openvdb::GridClass gridClass = firstGrid->getGridClass();

    typename GridType::ConstPtr refGrid;
    typedef typename openvdb::tools::MeshToVolume<GridType>::IndexGridT IndexGridT;
    typename IndexGridT::Ptr indexGrid;

    // Check for reference VDB
    {
        const GA_PrimitiveGroup *refGroup =
            matchGroup(const_cast<GU_Detail&>(*refGeo), "");
        hvdb::VdbPrimCIterator refIt(refGeo, refGroup);
        if (refIt) {
            const openvdb::GridClass refClass = refIt->getGrid().getGridClass();
            if (refIt && refClass == openvdb::GRID_LEVEL_SET) {
                refGrid = openvdb::gridConstPtrCast<GridType>(refIt->getGridPtr());
            }
        }
    }
 
    // Check for reference mesh
    boost::shared_ptr<GU_Detail> geoPtr;
    if (!refGrid) {
        std::string warningStr;
        geoPtr = hvdb::validateGeometry(*refGeo, warningStr, &boss);

        if (geoPtr) {
            refGeo = geoPtr.get();
            if (!warningStr.empty()) addWarning(SOP_MESSAGE, warningStr.c_str());
        }

        std::vector<openvdb::Vec3s> pointList;
        std::vector<openvdb::Vec4I> primList;

        pointList.resize(refGeo->getNumPoints());
        primList.resize(refGeo->getNumPrimitives());

        UTparallelFor(GA_SplittableRange(refGeo->getPointRange()),
            hvdb::TransformOp(refGeo, *transform, pointList));

        UTparallelFor(GA_SplittableRange(refGeo->getPrimitiveRange()),
            hvdb::PrimCpyOp(refGeo, primList));

        if (boss.wasInterrupted()) return;

        openvdb::tools::MeshToVolume<GridType, hvdb::Interrupter>
            converter(transform, openvdb::tools::GENERATE_PRIM_INDEX_GRID, &boss);

        if (gridClass == openvdb::GRID_LEVEL_SET) {
            converter.convertToLevelSet(pointList, primList);
        } else {
            const ValueType bandWidth = backgroundValue / transform->voxelSize()[0];
            converter.convertToLevelSet(pointList, primList, bandWidth, bandWidth);
        }

        refGrid = converter.distGridPtr();
        indexGrid = converter.indexGridPtr();
    }

    if (boss.wasInterrupted()) return;

    const double iadaptivity = double(evalFloat("internaladaptivity", 0, time));
    mesher.setRefGrid(refGrid, iadaptivity);


    std::list<openvdb::GridBase::ConstPtr>::iterator it = grids.begin();
    std::vector<std::string> badTransformList, badBackgroundList, badTypeList;

    GA_PrimitiveGroup *surfaceGroup = NULL, *interiorGroup = NULL, *seamGroup = NULL;

    {
        UT_String newGropStr;
        evalString(newGropStr, "surfacegroup", 0, time);
        if(newGropStr.length() > 0) {
            surfaceGroup = gdp->findPrimitiveGroup(newGropStr);
            if (!surfaceGroup) surfaceGroup = gdp->newPrimitiveGroup(newGropStr);
        }

        evalString(newGropStr, "interiorgroup", 0, time);
        if(newGropStr.length() > 0) {
            interiorGroup = gdp->findPrimitiveGroup(newGropStr);
            if (!interiorGroup) interiorGroup = gdp->newPrimitiveGroup(newGropStr);
        }

        evalString(newGropStr, "seamlinegroup", 0, time);
        if(newGropStr.length() > 0) {
            seamGroup = gdp->findPrimitiveGroup(newGropStr);
            if (!seamGroup) seamGroup = gdp->newPrimitiveGroup(newGropStr);
        }
    }

    for (it = grids.begin(); it != grids.end(); ++it) {

        if (boss.wasInterrupted()) break;

        typename GridType::ConstPtr grid = openvdb::gridConstPtrCast<GridType>(*it);

        if (!grid) {
            badTypeList.push_back(grid->getName());
            continue;
        }

        if (grid->transform() != *transform) {
            badTransformList.push_back(grid->getName());
            continue;
        }

        if (!openvdb::math::isApproxEqual(grid->background(), backgroundValue)) {
            badBackgroundList.push_back(grid->getName());
            continue;
        }

        mesher(*grid);

        copyMesh(*gdp, mesher, boss, keepVdbName ? grid->getName().c_str() : NULL,
            surfaceGroup, interiorGroup, seamGroup);
    }

    grids.clear();

    // Reconstruct sharp fetures
    /*if (!boss.wasInterrupted() && reconstruct) {
        UTparallelFor(GA_SplittableRange(gdp->getPointRange()),
            SharpFeatureOp<IndexGridT>(*refGeo, *gdp, *indexGrid, surfaceGroup));
    }*/


    // Compute vertex normals
    if (!boss.wasInterrupted() && computeNormals) {

        UTparallelFor(GA_SplittableRange(gdp->getPrimitiveRange()),
            hvdb::VertexNormalOp(*gdp, interiorGroup));

        if (!interiorGroup) {
            addWarning(SOP_MESSAGE, "More accurate vertex normals can be generated "
                "if the interior polygon group is enabled.");
        }
    }

    // Transfer Primitive Attributes
    if (!boss.wasInterrupted() && transferAttributes && refGeo && indexGrid) {
        hvdb::transferPrimitiveAttributes(*refGeo, *gdp, *indexGrid, boss, surfaceGroup);
    }

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

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
