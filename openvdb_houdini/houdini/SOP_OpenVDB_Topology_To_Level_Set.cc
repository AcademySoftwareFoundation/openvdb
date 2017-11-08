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
/// @file SOP_OpenVDB_Topology_To_Level_Set.cc
///
/// @author FX R&D OpenVDB team

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GU_VDBPointTools.h>

#include <openvdb/tools/TopologyToLevelSet.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/points/PointDataGrid.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_Handle.h>
#include <GA/GA_Types.h>
#include <GA/GA_Iterator.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>

#include <stdexcept>
#include <string>


namespace cvdb = openvdb;
namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Topology_To_Level_Set: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Topology_To_Level_Set(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Topology_To_Level_Set() override {}

    bool updateParmsFlags() override;

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    OP_ERROR cookMySop(OP_Context&) override;
};


////////////////////////////////////////


namespace {

struct Converter
{
    float bandWidthWorld;
    int bandWidthVoxels, closingWidth, dilation, smoothingSteps, outputName;
    bool worldSpaceUnits;
    std::string customName;

    Converter(GU_Detail& geo, hvdb::Interrupter& boss)
        : bandWidthWorld(0)
        , bandWidthVoxels(3)
        , closingWidth(1)
        , dilation(0)
        , smoothingSteps(0)
        , outputName(0)
        , worldSpaceUnits(false)
        , customName("vdb")
        , mGeoPt(&geo)
        , mBossPt(&boss)
    {
    }

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        int bandWidth = bandWidthVoxels;
        if (worldSpaceUnits) {
            bandWidth = int(openvdb::math::Round(bandWidthWorld / grid.transform().voxelSize()[0]));
        }

        openvdb::FloatGrid::Ptr sdfGrid = openvdb::tools::topologyToLevelSet(
           grid, bandWidth, closingWidth, dilation, smoothingSteps, mBossPt);

        std::string name = grid.getName();
        if (outputName == 1) name += customName;
        else if (outputName == 2) name = customName;

        hvdb::createVdbPrimitive(*mGeoPt, sdfGrid, name.c_str());
    }

private:
    GU_Detail         * const mGeoPt;
    hvdb::Interrupter * const mBossPt;
}; // struct Converter

} // unnamed namespace


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenu)
        .setTooltip("Specify a subset of the input VDBs to be processed.")
        .setDocumentation(
            "A subset of the input VDB grids to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    {
        char const * const items[] = {
            "keep",     "Keep Original Name",
            "append",   "Add Suffix",
            "replace",  "Custom Name",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "outputName", "Output Name")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("Output VDB naming scheme")
            .setDocumentation(
                "Give the output VDB the same name as the input VDB,"
                " or add a suffix to the input name, or use a custom name."));

        parms.add(hutil::ParmFactory(PRM_STRING, "customName", "Custom Name")
            .setTooltip("The suffix or custom name to be used"));
    }

    /// Narrow-band width {
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldSpaceUnits", "Use World Space for Band")
        .setDocumentation(
            "If enabled, specify the width of the narrow band in world units,"
            " otherwise specify it in voxels.  Voxel units work with all scales of geometry."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "bandWidth", "Half-Band in Voxels")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setTooltip(
            "Specify the half width of the narrow band in voxels."
            " Three voxels is optimal for many level set operations."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "bandWidthWS", "Half-Band in World")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 10)
        .setTooltip("Specify the half width of the narrow band in world units."));

    /// }

    parms.add(hutil::ParmFactory(PRM_INT_J, "dilation", "Voxel Dilation")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setTooltip("Expand the filled voxel region by the specified number of voxels."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "closingwidth", "Closing Width")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setTooltip(
            "First expand the filled voxel region, then shrink it by the specified "
            "number of voxels. This causes holes and valleys to be filled."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "smoothingsteps", "Smoothing Steps")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setTooltip("Number of smoothing iterations"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Topology To Level Set",
        SOP_OpenVDB_Topology_To_Level_Set::factory, parms, *table)
        .addAlias("OpenVDB From Mask")
        .addInput("VDB Grids")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Create a level set VDB based on the active voxels of another VDB.\"\"\"\n\
\n\
@overview\n\
\n\
This node creates a narrow-band level set VDB that conforms to the active voxels\n\
of the input VDB.  This forms a shell or wrapper that can be used\n\
to conservatively enclose the input volume.\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Topology_To_Level_Set::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Topology_To_Level_Set(net, name, op);
}


SOP_OpenVDB_Topology_To_Level_Set::SOP_OpenVDB_Topology_To_Level_Set(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Topology_To_Level_Set::updateParmsFlags()
{
    bool changed = false;
    const fpreal time = 0;
    const bool wsUnits = bool(evalInt("worldSpaceUnits", 0, time));

    changed |= enableParm("bandWidth", !wsUnits);
    changed |= enableParm("bandWidthWS", wsUnits);
    changed |= enableParm("bandWidth", !wsUnits);
    changed |= enableParm("bandWidthWS", wsUnits);

    changed |= setVisibleState("bandWidth", !wsUnits);
    changed |= setVisibleState("bandWidthWS", wsUnits);

    const bool useCustomName = evalInt("outputName", 0, time) != 0;
    changed |= enableParm("customName", useCustomName);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Topology_To_Level_Set::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();
        gdp->clearAndDestroy();

        const GU_Detail* inputGeoPt = inputGeo(0);
        if (inputGeoPt == nullptr) return error();

        hvdb::Interrupter boss;

        // Get UI settings

        UT_String customName, groupStr;
        evalString(customName, "customName", 0, time);
        evalString(groupStr, "group", 0, time);

        Converter converter(*gdp, boss);
        converter.worldSpaceUnits = evalInt("worldSpaceUnits", 0, time) != 0;
        converter.bandWidthWorld = float(evalFloat("bandWidthWS", 0, time));
        converter.bandWidthVoxels = static_cast<int>(evalInt("bandWidth", 0, time));
        converter.closingWidth = static_cast<int>(evalInt("closingwidth", 0, time));
        converter.dilation = static_cast<int>(evalInt("dilation", 0, time));
        converter.smoothingSteps = static_cast<int>(evalInt("smoothingsteps", 0, time));
        converter.outputName = static_cast<int>(evalInt("outputName", 0, time));
        converter.customName = customName.toStdString();

        // Process VDB primitives

        const GA_PrimitiveGroup* group = matchGroup(*inputGeoPt, groupStr.toStdString());

        hvdb::VdbPrimCIterator vdbIt(inputGeoPt, group);

        if (!vdbIt) {
            addWarning(SOP_MESSAGE, "No VDB grids to process.");
            return error();
        }

        for (; vdbIt; ++vdbIt) {

            if (boss.wasInterrupted()) break;

            const GU_PrimVDB *vdb = *vdbIt;

            if (!GEOvdbProcessTypedGridTopology(*vdb, converter)) {
                // Handle grid types that are not natively supported by Houdini.
                if (vdb->getGrid().isType<cvdb::tools::PointIndexGrid>()) {
                    cvdb::tools::PointIndexGrid::ConstPtr grid =
                        cvdb::gridConstPtrCast<cvdb::tools::PointIndexGrid>(vdb->getGridPtr());
                    converter(*grid);
                } else if (vdb->getGrid().isType<cvdb::points::PointDataGrid>()) {
                    cvdb::points::PointDataGrid::ConstPtr grid =
                        cvdb::gridConstPtrCast<cvdb::points::PointDataGrid>(vdb->getGridPtr());
                    converter(*grid);
                } else if (vdb->getGrid().isType<cvdb::MaskGrid>()) {
                    cvdb::MaskGrid::ConstPtr grid =
                        cvdb::gridConstPtrCast<cvdb::MaskGrid>(vdb->getGridPtr());
                    converter(*grid);
                }
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
