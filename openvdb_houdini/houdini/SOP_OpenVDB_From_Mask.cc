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
/// @file SOP_OpenVDB_From_Mask.cc
///
/// @author FX R&D OpenVDB team


#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GU_VDBPointTools.h>

#include <openvdb/tools/PointMaskGrid.h>
#include <openvdb/tools/TopologyToLevelSet.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_Handle.h>
#include <GA/GA_Types.h>
#include <GA/GA_Iterator.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

class SOP_OpenVDB_From_Mask: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_From_Mask(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_From_Mask() {}

    virtual bool updateParmsFlags();

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int convertUnits();

protected:
    virtual OP_ERROR cookMySop(OP_Context&);

private:
    float mVoxelSize;
};


////////////////////////////////////////


namespace
{

// Callback to convert from voxel to world space units
int
convertUnitsCB(void* data, int /*idx*/, float /*time*/, const PRM_Template*)
{
   SOP_OpenVDB_From_Mask* sop = static_cast<SOP_OpenVDB_From_Mask*>(data);
   if (sop == NULL) return 0;
   return sop->convertUnits();
}


struct TopologyConverter
{
    int bandWidthInVoxels, closingWidth, dilation, smoothingSteps; // public settings

    TopologyConverter(GU_Detail& geo, hvdb::Interrupter& boss)
        : bandWidthInVoxels(3)
        , closingWidth(1)
        , dilation(0)
        , smoothingSteps(0)
        , mGeoPt(&geo)
        , mBossPt(&boss)
    {
    }

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        openvdb::FloatGrid::Ptr sdfGrid = openvdb::tools::topologyToLevelSet(
           grid, bandWidthInVoxels, closingWidth, dilation, smoothingSteps, mBossPt);

        hvdb::createVdbPrimitive(*mGeoPt, sdfGrid, grid.getName().c_str());
    }

private:
    GU_Detail         * const mGeoPt;
    hvdb::Interrupter * const mBossPt;
}; // struct TopologyConverter


} // unnamed namespace



void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input grids.")
        .setChoiceList(&hutil::PrimGroupMenu));

    parms.add(hutil::ParmFactory(PRM_STRING, "gridname", "Distance VDB")
        .setDefault("surface")
        .setHelpText("Output grid name, ignored for VDB inputs."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 5)
        .setHelpText("Ignored for VDB inputs."));

    /// Narrow-band width {
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "worldSpaceUnits", "Use World Space for Band")
        .setCallbackFunc(&convertUnitsCB));

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
    hvdb::OpenVDBOpFactory("OpenVDB From Mask",
        SOP_OpenVDB_From_Mask::factory, parms, *table)
        .addInput("VDB Grids, Points and Packed Points");
}

////////////////////////////////////////


OP_Node*
SOP_OpenVDB_From_Mask::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_From_Mask(net, name, op);
}


SOP_OpenVDB_From_Mask::SOP_OpenVDB_From_Mask(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
    , mVoxelSize(0.1f)
{
}

////////////////////////////////////////

int
SOP_OpenVDB_From_Mask::convertUnits()
{
    const bool toWSUnits = static_cast<bool>(evalInt("worldSpaceUnits", 0, 0));
    float width;

    if (toWSUnits) {
        width = static_cast<float>(evalInt("bandWidth", 0, 0));
        setFloat("bandWidthWS", 0, 0, width * mVoxelSize);
        return 1;
    }

    width = static_cast<float>(evalFloat("bandWidthWS", 0, 0));
    int voxelWidth = std::max(static_cast<int>(width / mVoxelSize), 1);
    setInt("bandWidth", 0, 0, voxelWidth);

    return 1;
}


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_From_Mask::updateParmsFlags()
{
    bool changed = false;
    const fpreal time = 0; // No point using CHgetTime as that is unstable.


    // Conversion
    const bool wsUnits = bool(evalInt("worldSpaceUnits", 0, time));


    changed |= enableParm("bandWidth",
                          !wsUnits);

    changed |= enableParm("bandWidthWS",
                          wsUnits);

    changed |= enableParm("bandWidth", !wsUnits);
    changed |= enableParm("bandWidthWS", wsUnits);

    changed |= setVisibleState("bandWidth", !wsUnits);
    changed |= setVisibleState("bandWidthWS", wsUnits);


    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_From_Mask::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();
        gdp->clearAndDestroy();

        const GU_Detail* inputGeoPt = inputGeo(0);
        if(inputGeoPt == NULL) return error();

        // Get UI settings

        const float voxelSize = float(evalFloat("voxelsize", 0, time));
        if (voxelSize < 1e-5) {
            std::ostringstream ostr;
            ostr << "The voxel size ("<< mVoxelSize << ") is too small.";
            addError(SOP_MESSAGE, ostr.str().c_str());
            return error();
        }

        mVoxelSize = voxelSize; // stash for world to index conversion.

        const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

        int bandWidthInVoxels = 3;
        if (evalInt("worldSpaceUnits", 0, time) != 0) {
            bandWidthInVoxels = int(openvdb::math::Round(evalFloat("bandWidthWS", 0, time) / voxelSize));
        } else {
            bandWidthInVoxels = evalInt("bandWidth", 0, time);
        }

        const int dilation = evalInt("dilation", 0, time);
        const int closingWidth = evalInt("closingwidth", 0, time);
        const int smoothingSteps = evalInt("smoothingsteps", 0, time);

        UT_String gridName;
        evalString(gridName, "gridname", 0, time);

        hvdb::Interrupter boss;


        // Process VDB primitives

        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(const_cast<GU_Detail&>(*inputGeoPt), groupStr.toStdString());

        hvdb::VdbPrimCIterator vdbIt(inputGeoPt, group);


        if (vdbIt) {

            TopologyConverter converter(*gdp, boss);

            converter.bandWidthInVoxels = bandWidthInVoxels;
            converter.closingWidth = closingWidth;
            converter.dilation = dilation;
            converter.smoothingSteps = smoothingSteps;

            for (; vdbIt; ++vdbIt) {

                if (boss.wasInterrupted()) break;

                const GU_PrimVDB *vdb = *vdbIt;

                if (!GEOvdbProcessTypedGridTopology(*vdb, converter)) { // all hdk supported grid types

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

        } else {

            openvdb::MaskGrid::Ptr maskGrid = GUvdbCreatePointMaskGrid(*transform, *inputGeoPt);

            openvdb::FloatGrid::Ptr sdfGrid = openvdb::tools::topologyToLevelSet(
                *maskGrid, bandWidthInVoxels, closingWidth, dilation, smoothingSteps, &boss);

            hvdb::createVdbPrimitive(*gdp, sdfGrid, gridName.buffer());
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
