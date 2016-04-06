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
/// @file SOP_OpenVDB_Topology_Mask.cc
///
/// @author FX R&D OpenVDB team


#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/GU_VDBPointTools.h>

#include <openvdb/tools/PointMaskGrid.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_Handle.h>
#include <GA/GA_Types.h>
#include <GA/GA_Iterator.h>
#include <GU/GU_Detail.h>
#include <PRM/PRM_Parm.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

class SOP_OpenVDB_Topology_Mask: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Topology_Mask(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Topology_Mask() {}

    virtual bool updateParmsFlags();

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
};


////////////////////////////////////////


namespace
{

struct TopologyConverter
{
    TopologyConverter(GU_Detail& geo) : mGeoPt(&geo) { }

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        openvdb::MaskGrid::TreeType::Ptr maskTree(new openvdb::MaskGrid::TreeType(grid.tree(), false, openvdb::TopologyCopy()));

        openvdb::MaskGrid::Ptr maskGrid = openvdb::MaskGrid::create(maskTree);
        maskGrid->setTransform(grid.transform().copy());

        hvdb::createVdbPrimitive(*mGeoPt, maskGrid, grid.getName().c_str());
    }

    GU_Detail * const mGeoPt;
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

    parms.add(hutil::ParmFactory(PRM_STRING, "gridname", "Mask Name")
        .setDefault("topology")
        .setHelpText("Output grid name used when masking points."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 5)
        .setHelpText("Output grid voxel size used when masking points."));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Topology Mask",
        SOP_OpenVDB_Topology_Mask::factory, parms, *table)
        .addInput("VDB Grids, Points and Packed Points");
}

////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Topology_Mask::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Topology_Mask(net, name, op);
}


SOP_OpenVDB_Topology_Mask::SOP_OpenVDB_Topology_Mask(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Topology_Mask::updateParmsFlags()
{
    bool changed = false;
    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Topology_Mask::cookMySop(OP_Context& context)
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
            ostr << "The voxel size ("<< voxelSize << ") is too small.";
            addError(SOP_MESSAGE, ostr.str().c_str());
            return error();
        }

        const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

        UT_String gridName;
        evalString(gridName, "gridname", 0, time);

        hvdb::Interrupter boss;


        // Process VDB primitives

        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(const_cast<GU_Detail&>(*inputGeoPt), groupStr.toStdString());

        hvdb::VdbPrimCIterator vdbIt(inputGeoPt, group);


        if (vdbIt) {

            TopologyConverter converter(*gdp);

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
            hvdb::createVdbPrimitive(*gdp, maskGrid, gridName.buffer());
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
