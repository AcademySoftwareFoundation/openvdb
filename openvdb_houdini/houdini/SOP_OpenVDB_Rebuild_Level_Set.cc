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
/// @file SOP_OpenVDB_Rebuild_Level_Set.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Level set rebuild and resample.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <UT/UT_Interrupt.h>
#include <boost/algorithm/string/join.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_Rebuild_Level_Set: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Rebuild_Level_Set(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Rebuild_Level_Set() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
};


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenu)
        .setHelpText(
            "Specify a subset of the input VDB grids to be processed\n"
            "(scalar, floating-point grids only)"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "isovalue", "Isovalue")
        .setRange(PRM_RANGE_UI, -1, PRM_RANGE_UI, 1)
        .setHelpText("The isovalue that defines the implicit surface"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "halfbandwidth", "Half-Band Width")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText(
            "Half the width of the narrow band, in voxel units\n"
            "(3 is optimal for level set operations)"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "fillinterior", "Fill Interior")
        .setHelpText(
            "Densify the interior of a level set by computing\n"
            "signed distances for all interior voxels."));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Rebuild Level Set",
        SOP_OpenVDB_Rebuild_Level_Set::factory, parms, *table)
        .addInput("VDB grids to process");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Rebuild_Level_Set::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Rebuild_Level_Set(net, name, op);
}


SOP_OpenVDB_Rebuild_Level_Set::SOP_OpenVDB_Rebuild_Level_Set(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Enable/disable or show/hide parameters in the UI.
bool
SOP_OpenVDB_Rebuild_Level_Set::updateParmsFlags()
{
    bool changed = false;

    // Not sure if this is a desired feature.
    changed |= setVisibleState("fillinterior", false);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Rebuild_Level_Set::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        // This does a deep copy of native Houdini primitives
        // but only a shallow copy of VDB grids.
        duplicateSource(0, context);

        // Get the group of grids to process.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = this->matchGroup(*gdp, groupStr.toStdString());

        // Get other UI parameters.
        const float exBandWidth = evalFloat("halfbandwidth", 0, time);

        float inBandWidth = bool(evalInt("fillinterior", 0, time)) ?
            std::numeric_limits<float>::max() : exBandWidth;

        const float iso = evalFloat("isovalue", 0, time);

        hvdb::Interrupter boss("Rebuilding Level Set Grids");

        std::vector<std::string> skippedGrids;

        // Process each VDB primitive that belongs to the selected group.
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {

            if (boss.wasInterrupted()) break;

            GU_PrimVDB* vdbPrim = *it;

            // Process floating point grids.
            if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {
                openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
                vdbPrim->setGrid(*openvdb::tools::levelSetRebuild(
                    grid, iso, exBandWidth, inBandWidth, /*xform=*/NULL, &boss));
            } else if (vdbPrim->getStorageType() == UT_VDB_DOUBLE) {
                openvdb::DoubleGrid& grid = UTvdbGridCast<openvdb::DoubleGrid>(vdbPrim->getGrid());
                vdbPrim->setGrid(*openvdb::tools::levelSetRebuild(
                    grid, iso, exBandWidth, inBandWidth, /*xform=*/NULL, &boss));
            } else {
                skippedGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
            }
        }

        if (!skippedGrids.empty()) {
            std::string s = "The following non-floating-point grids were skipped: " +
                boost::algorithm::join(skippedGrids, ", ") + ".";
            addWarning(SOP_MESSAGE, s.c_str());
        }

        if (boss.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "Process was interrupted.");
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
