///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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

#include <UT/UT_Interrupt.h>
#include <GA/GA_Handle.h>
#include <GA/GA_Types.h>
#include <GA/GA_Iterator.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

class SOP_OpenVDB_Topology_To_Level_Set: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Topology_To_Level_Set(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Topology_To_Level_Set() {}

    virtual bool updateParmsFlags();

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
};


////////////////////////////////////////


namespace
{

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
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input grids.")
        .setChoiceList(&hutil::PrimGroupMenu));

    {
        const char* items[] = {
            "keep",     "Keep Incoming VDB Names",
            "append",   "Custom Append",
            "replace",   "Custom Replace",
            NULL
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "outputName", "Output Name")
            .setDefault(PRMzeroDefaults)
            .setHelpText("Rename output grid(s)")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));

        parms.add(hutil::ParmFactory(PRM_STRING, "customName", "Custom Name")
            .setHelpText("Used to rename the input grids"));
    }

    /// Narrow-band width {
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldSpaceUnits", "Use World Space for Band"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "bandWidth", "Half-Band in Voxels")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText("Specify the half width of the narrow band. "
            "(3 voxel units is optimal for level set operations.)"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "bandWidthWS", "Half-Band in World")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 10)
        .setHelpText("Specify the half width of the narrow band."));

    /// }

    parms.add(hutil::ParmFactory(PRM_INT_J, "dilation", "Voxel Dilation")
              .setDefault(PRMzeroDefaults)
              .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
              .setHelpText("Expands the filled voxel region by the specified "
                  "number of voxels."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "closingwidth", "Closing Width")
              .setDefault(PRMoneDefaults)
              .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
              .setHelpText("First expand the filled voxel region, then shrink it "
                  "by the specified number of voxels. This causes holes "
                  "and valleys to be filled."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "smoothingsteps", "Smoothing Steps")
              .setDefault(PRMzeroDefaults)
              .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
              .setHelpText("Number of smoothing interations"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Topology To Level Set",
        SOP_OpenVDB_Topology_To_Level_Set::factory, parms, *table)
        .addAlias("OpenVDB From Mask")
        .addInput("VDB Grids");
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
        if(inputGeoPt == NULL) return error();

        hvdb::Interrupter boss;

        // Get UI settings

        UT_String customName, groupStr;
        evalString(customName, "customName", 0, time);
        evalString(groupStr, "group", 0, time);

        Converter converter(*gdp, boss);
        converter.worldSpaceUnits = evalInt("worldSpaceUnits", 0, time) != 0;
        converter.bandWidthWorld = float(evalFloat("bandWidthWS", 0, time));
        converter.bandWidthVoxels = evalInt("bandWidth", 0, time);
        converter.closingWidth = evalInt("closingwidth", 0, time);
        converter.dilation = evalInt("dilation", 0, time);
        converter.smoothingSteps = evalInt("smoothingsteps", 0, time);
        converter.outputName = evalInt("outputName", 0, time);
        converter.customName = customName.toStdString();

        // Process VDB primitives

        const GA_PrimitiveGroup* group = matchGroup(const_cast<GU_Detail&>(*inputGeoPt), groupStr.toStdString());

        hvdb::VdbPrimCIterator vdbIt(inputGeoPt, group);

        if (!vdbIt) {
            addWarning(SOP_MESSAGE, "No VDB grids to process.");
            return error();
        }

        for (; vdbIt; ++vdbIt) {

            if (boss.wasInterrupted()) break;

            const GU_PrimVDB *vdb = *vdbIt;

            if (!GEOvdbProcessTypedGridTopology(*vdb, converter)) { // all hdk supported grid types

                // check grid types that are not natively supported by Houdini

                if (vdb->getGrid().type() == openvdb::tools::PointIndexGrid::gridType()) { // point index grid
                    openvdb::tools::PointIndexGrid::ConstPtr grid =
                        openvdb::gridConstPtrCast<openvdb::tools::PointIndexGrid>(vdb->getGridPtr());
                    converter(*grid);

                } else if (vdb->getGrid().type() == openvdb::MaskGrid::gridType()) { // mask grid
                    openvdb::MaskGrid::ConstPtr grid =
                        openvdb::gridConstPtrCast<openvdb::MaskGrid>(vdb->getGridPtr());
                    converter(*grid);
                }
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
