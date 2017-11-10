///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2017 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_To_Spheres.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Fills a volume with adaptively sized overlapping or nonoverlapping spheres.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GeometryUtil.h>
#include <openvdb_houdini/Utils.h>

#include <openvdb/tools/VolumeToSpheres.h>

#include <GU/GU_ConvertParms.h>
#include <GU/GU_Detail.h>
#include <GU/GU_PrimSphere.h>
#include <PRM/PRM_Parm.h>
#include <GA/GA_PageIterator.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_Version.h>

#include <boost/algorithm/string/join.hpp>

#include <string>
#include <vector>
#include <limits> // std::numeric_limits


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_To_Spheres: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_To_Spheres(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_To_Spheres() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i) const override { return (i > 0); }

    void checkActivePart(float time);

protected:
    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDBs to be processed.")
        .setDocumentation(
            "A subset of the input VDB grids to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "isovalue", "Isovalue")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0)
        .setTooltip(
            "The crossing point of the VDB values that is considered the surface\n\n"
            "Zero works for signed distance fields, while fog volumes"
            " require a larger positive value (0.5 is a good initial guess)."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldunits", "Use World Space Units")
        .setDocumentation(
            "If enabled, specify sphere sizes in world units, otherwise in voxels."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "minradius", "Min Radius in Voxels")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 2.0)
        .setTooltip("Determines the smallest sphere size, voxel units.")
        .setDocumentation(nullptr));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "minradiusworld", "Min Radius")
        .setDefault(0.1)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 2.0)
        .setTooltip("Determines the smallest sphere size, world units.")
        .setDocumentation("The radius of the smallest sphere allowed"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "maxradius", "Max Radius in Voxels")
        .setDefault(std::numeric_limits<float>::max())
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 100.0)
        .setTooltip("Determines the largest sphere size, voxel units.")
        .setDocumentation(nullptr));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "maxradiusworld", "Max Radius")
        .setDefault(std::numeric_limits<float>::max())
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 100.0)
        .setTooltip("Determines the largest sphere size, world units.")
        .setDocumentation("The radius of the largest sphere allowed"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "spheres", "Max Spheres")
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 100)
        .setDefault(50)
        .setTooltip("No more than this number of spheres are generated (but possibly fewer)."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "scatter", "Scatter Points")
        .setRange(PRM_RANGE_RESTRICTED, 1000, PRM_RANGE_UI, 50000)
        .setDefault(10000)
        .setTooltip(
            "How many interior points to consider for the sphere placement\n\n"
            "Increasing this count increases the chances of finding optimal sphere sizes."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "overlapping", "Overlapping")
#ifndef SESI_OPENVDB
        .setDefault(PRMzeroDefaults)
#else
        .setDefault(PRMoneDefaults)
#endif
        .setTooltip("If enabled, allow spheres to overlap/intersect."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "preserve", "Preserve Attributes and Groups")
#ifndef SESI_OPENVDB
        .setDefault(PRMzeroDefaults)
#else
        .setDefault(PRMoneDefaults)
#endif
        .setTooltip("If enabled, copy attributes and groups from the input."));

    // The "doid" parameter name comes from the standard in POPs
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "doid", "Add ID Attribute")
#ifndef SESI_OPENVDB
        .setDefault(PRMoneDefaults)
#else
        .setDefault(PRMzeroDefaults)
#endif
        .setTooltip("If enabled, add an id point attribute that denotes the source VDB.")
        .setDocumentation(
            "If enabled, add an `id` point attribute that denotes the source VDB"
            " for each sphere."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "dopscale", "Add PScale Attribute")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, add a pscale point attribute to each sphere.")
        .setDocumentation(
            "If enabled, add a `pscale` point attribute that indicates"
            " the radius of each sphere."));

    //////////

    hvdb::OpenVDBOpFactory("OpenVDB To Spheres", SOP_OpenVDB_To_Spheres::factory, parms, *table)
        .addInput("VDBs to convert")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Fill a VDB volume with adaptively-sized spheres.\"\"\"\n\
\n\
@overview\n\
\n\
This node is useful for generating proxy geometry for RBD simulations,\n\
since approximating nonconvex geometry with sphere compounds\n\
drastically improves the simulation time.\n\
This can be used, for example, on the output of an\n\
[OpenVDB Fracture node|Node:sop/DW_OpenVDBFracture].\n\
\n\
Another use is to produce the initial density volume for cloud modeling.\n\
\n\
@related\n\
- [OpenVDB Fracture|Node:sop/DW_OpenVDBFracture]\n\
- [Node:sop/cloud]\n\
- [Node:sop/vdbtospheres]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


OP_Node*
SOP_OpenVDB_To_Spheres::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_To_Spheres(net, name, op);
}


SOP_OpenVDB_To_Spheres::SOP_OpenVDB_To_Spheres(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


bool
SOP_OpenVDB_To_Spheres::updateParmsFlags()
{
    bool changed = false;

    const bool worldUnits = bool(evalInt("worldunits", 0, 0));

    changed |= setVisibleState("minradius", !worldUnits);
    changed |= setVisibleState("maxradius", !worldUnits);
    changed |= setVisibleState("minradiusworld", worldUnits);
    changed |= setVisibleState("maxradiusworld", worldUnits);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_To_Spheres::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        gdp->clearAndDestroy();

        hvdb::Interrupter boss("Filling VDBs with spheres");

        const GU_Detail* vdbGeo = inputGeo(0);
        if (vdbGeo == nullptr) return error();

        // Get the group of grids to surface.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*vdbGeo, groupStr.toStdString());
        hvdb::VdbPrimCIterator vdbIt(vdbGeo, group);

        if (!vdbIt) {
            addWarning(SOP_MESSAGE, "No VDB grids found.");
            return error();
        }

        // Eval attributes
        const float
            isovalue = static_cast<float>(evalFloat("isovalue", 0, time)),
            minradiusVoxel = static_cast<float>(evalFloat("minradius", 0, time)),
            maxradiusVoxel = static_cast<float>(evalFloat("maxradius", 0, time)),
            minradiusWorld = static_cast<float>(evalFloat("minradiusworld", 0, time)),
            maxradiusWorld = static_cast<float>(evalFloat("maxradiusworld", 0, time));

        const bool worldUnits = evalInt("worldunits", 0, time);

        const int sphereCount = static_cast<int>(evalInt("spheres", 0, time));
        const bool overlapping = evalInt("overlapping", 0, time);
        const int scatter = static_cast<int>(evalInt("scatter", 0, time));
        const bool preserve = evalInt("preserve", 0, time);

        const bool addID = evalInt("doid", 0, time) != 0;
        GA_RWHandleI idAttr;
        if (addID) {
            GA_RWAttributeRef aRef = gdp->findPointAttribute("id");
            if (!aRef.isValid()) {
                aRef = gdp->addIntTuple(GA_ATTRIB_POINT, "id", 1, GA_Defaults(0));
            }
            idAttr = aRef.getAttribute();
            if(!idAttr.isValid()) {
                addWarning(SOP_MESSAGE, "Failed to create the point ID attribute.");
                return error();
            }
        }

        const bool addPScale = evalInt("dopscale", 0, time) != 0;
        GA_RWHandleF pscaleAttr;
        if (addPScale) {
            GA_RWAttributeRef aRef = gdp->findFloatTuple(GA_ATTRIB_POINT, GEO_STD_ATTRIB_PSCALE);
            if (!aRef.isValid()) {
                aRef = gdp->addFloatTuple(
                    GA_ATTRIB_POINT, GEO_STD_ATTRIB_PSCALE, 1, GA_Defaults(0));
            }
            pscaleAttr = aRef.getAttribute();
            if(!pscaleAttr.isValid()) {
                addWarning(SOP_MESSAGE, "Failed to create the point pscale attribute.");
                return error();
            }
        }

        int idNumber = 1;

        GU_ConvertParms parms;
#if UT_VERSION_INT < 0x0d0000b1 // 13.0.177 or earlier
        parms.preserveGroups = true;
#else
        parms.setKeepGroups(true);
#endif

        std::vector<std::string> skippedGrids;

        for (; vdbIt; ++vdbIt) {
            if (boss.wasInterrupted()) break;

            float minradius = minradiusVoxel, maxradius = maxradiusVoxel;

            if (worldUnits) {
                const float voxelScale = float(1.0 / vdbIt->getGrid().voxelSize()[0]);
                minradius = minradiusWorld * voxelScale;
                maxradius = maxradiusWorld * voxelScale;
            }

            maxradius = std::max(maxradius, minradius + float(1e-5));

            std::vector<openvdb::Vec4s> spheres;

            if (vdbIt->getGrid().type() == openvdb::FloatGrid::gridType()) {

                openvdb::FloatGrid::ConstPtr gridPtr =
                    openvdb::gridConstPtrCast<openvdb::FloatGrid>(vdbIt->getGridPtr());

                openvdb::tools::fillWithSpheres(*gridPtr, spheres, sphereCount, overlapping,
                    minradius, maxradius, isovalue, scatter, &boss);


            } else if (vdbIt->getGrid().type() == openvdb::DoubleGrid::gridType()) {

                openvdb::DoubleGrid::ConstPtr gridPtr =
                    openvdb::gridConstPtrCast<openvdb::DoubleGrid>(vdbIt->getGridPtr());

                openvdb::tools::fillWithSpheres(*gridPtr, spheres, sphereCount, overlapping,
                    minradius, maxradius, isovalue, scatter, &boss);

            } else {
                skippedGrids.push_back(vdbIt.getPrimitiveNameOrIndex().toStdString());
                continue;
            }

#if (UT_VERSION_INT >= 0x0d050013) // 13.5.19 or later
            GA_Detail::OffsetMarker marker(*gdp);
#else
            GU_ConvertMarker marker(*gdp);
#endif

            // copy spheres to Houdini
            for (size_t n = 0, N = spheres.size(); n < N; ++n) {

                const openvdb::Vec4s& sphere = spheres[n];

                GA_Offset ptoff = gdp->appendPointOffset();

                gdp->setPos3(ptoff, sphere.x(), sphere.y(), sphere.z());

                if (addID) {
                    idAttr.set(ptoff, idNumber);
                }

                if (addPScale) {
                    pscaleAttr.set(ptoff, sphere[3]);
                }

                UT_Matrix4 mat = UT_Matrix4::getIdentityMatrix();
                mat.scale(sphere[3],sphere[3],sphere[3]);

                #if (UT_VERSION_INT >= 0x0c050000)  // 12.5.0 or later
                GU_PrimSphereParms sphereParms(gdp, ptoff);
                sphereParms.xform = mat;
                GU_PrimSphere::build(sphereParms);
                #else
                GU_PrimSphereParms sphereParms(gdp, gdp->getGEOPoint(ptoff));
                sphereParms.xform = mat;
                GU_PrimSphere::build(sphereParms);
                #endif
            }

            if (preserve) {
                GUconvertCopySingleVertexPrimAttribsAndGroups(
                    parms, *vdbGeo, vdbIt.getOffset(),
#if (UT_VERSION_INT >= 0x0d050013) // 13.5.19 or later
                    *gdp, marker.primitiveRange(), marker.pointRange());
#else
                    *gdp, marker.getPrimitives(), marker.getPoints());
#endif
            }
            ++idNumber;
        }

        if (!skippedGrids.empty()) {
            std::string s = "Only scalar (float/double) grids are supported, the following "
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

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
