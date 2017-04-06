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
/// @file SOP_OpenVDB_Prune.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief SOP to prune tree branches from OpenVDB grids

#include <houdini_utils/ParmFactory.h>
#include <openvdb/tools/Prune.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <UT/UT_Interrupt.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Prune: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Prune(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Prune() {}

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
        .setHelpText("Specify a subset of the input grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    {
        const char* items[] = {
            "value",    "Value",
            "inactive", "Inactive",
            "levelset", "Level Set",
            NULL
        };
        parms.add(hutil::ParmFactory(PRM_STRING, "mode", "Mode")
            .setDefault("value")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setHelpText(
                "Value:\n"
                "    Collapse regions in which all voxels have the same\n"
                "    value and active state into tiles with those values\n"
                "    and active states.\n"
                "Inactive:\n"
                "    Collapse regions in which all voxels are inactive\n"
                "    into inactive background tiles.\n"
                "Level Set:\n"
                "    Collapse regions in which all voxels are inactive\n"
                "    into inactive tiles with either the inside or\n"
                "    the outside background value, depending on\n"
                "    the signs of the voxel values.\n"));
    }

    parms.add(hutil::ParmFactory(PRM_FLT_J, "tolerance", "Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1)
        .setHelpText(
            "Voxel values are considered equal if they differ\n"
            "by less than the specified threshold."));

    hvdb::OpenVDBOpFactory("OpenVDB Prune", SOP_OpenVDB_Prune::factory, parms, *table)
        .addInput("Grids to process");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Prune::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Prune(net, name, op);
}


SOP_OpenVDB_Prune::SOP_OpenVDB_Prune(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Enable/disable or show/hide parameters in the UI.
bool
SOP_OpenVDB_Prune::updateParmsFlags()
{
    bool changed = false;

    UT_String modeStr;
    evalString(modeStr, "mode", 0, 0);

    changed |= enableParm("tolerance", modeStr == "value");

    return changed;
}


////////////////////////////////////////


namespace {
struct PruneOp {
    PruneOp(const std::string m, fpreal tol = 0.0): mode(m), tolerance(tol) {}

    template<typename GridT>
    void operator()(GridT& grid) const
    {
        typedef typename GridT::ValueType ValueT;

        if (mode == "value") {
            openvdb::tools::prune(grid.tree(), ValueT(openvdb::zeroVal<ValueT>() + tolerance));
        } else if (mode == "inactive") {
            openvdb::tools::pruneInactive(grid.tree());
        } else if (mode == "levelset") {
            openvdb::tools::pruneLevelSet(grid.tree());
        }
    }

    std::string mode;
    fpreal tolerance;
};
}


OP_ERROR
SOP_OpenVDB_Prune::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        // This does a deep copy of native Houdini primitives
        // but only a shallow copy of OpenVDB grids.
        duplicateSourceStealable(0, context);

        // Get the group of grids to process.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group =
            this->matchGroup(*gdp, groupStr.toStdString());

        // Get other UI parameters.
        UT_String modeStr;
        evalString(modeStr, "mode", 0, time);
        const fpreal tolerance = evalFloat("tolerance", 0, time);

        // Construct a functor to process grids of arbitrary type.
        const PruneOp pruneOp(modeStr.toStdString(), tolerance);

        UT_AutoInterrupt progress("Pruning OpenVDB grids");

        // Process each VDB primitive that belongs to the selected group.
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *it;
            GEOvdbProcessTypedGridTopology(*vdbPrim, pruneOp);
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
