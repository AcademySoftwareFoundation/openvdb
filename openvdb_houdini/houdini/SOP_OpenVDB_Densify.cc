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
/// @file SOP_OpenVDB_Densify.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief SOP to replace active tiles with active voxels in OpenVDB grids

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <UT/UT_Interrupt.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Densify: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Densify(OP_Network*, const char* name, OP_Operator*);

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    OP_ERROR cookMySop(OP_Context&) override;
};


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDBs to be densified.")
        .setDocumentation(
            "A subset of the input VDBs to be densified"
            " (see [specifying volumes|/model/volumes#group])"));

    hvdb::OpenVDBOpFactory("OpenVDB Densify", SOP_OpenVDB_Densify::factory, parms, *table)
        .addInput("VDBs to densify")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Densify sparse VDB volumes.\"\"\"\n\
\n\
@overview\n\
\n\
This node replaces active\n\
[tiles|http://www.openvdb.org/documentation/doxygen/overview.html#secSparsity]\n\
in VDB [trees|http://www.openvdb.org/documentation/doxygen/overview.html#secTree]\n\
with dense, leaf-level voxels.\n\
This is useful for subsequent processing with nodes like [Node:sop/volumevop]\n\
that operate only on leaf voxels.\n\
\n\
WARNING:\n\
    Densifying a sparse VDB can significantly increase its memory footprint.\n\
\n\
@related\n\
- [OpenVDB Fill|Node:sop/DW_OpenVDBFill]\n\
- [OpenVDB Prune|Node:sop/DW_OpenVDBPrune]\n\
- [Node:sop/vdbactivate]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Densify::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Densify(net, name, op);
}


SOP_OpenVDB_Densify::SOP_OpenVDB_Densify(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {
struct DensifyOp {
    DensifyOp() {}

    template<typename GridT>
    void operator()(GridT& grid) const
    {
        grid.tree().voxelizeActiveTiles(/*threaded=*/true);
    }
};
}


OP_ERROR
SOP_OpenVDB_Densify::cookMySop(OP_Context& context)
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
        const GA_PrimitiveGroup* group = this->matchGroup(*gdp, groupStr.toStdString());

        // Construct a functor to process grids of arbitrary type.
        const DensifyOp densifyOp;

        UT_AutoInterrupt progress("Densifying VDBs");

        // Process each VDB primitive that belongs to the selected group.
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *it;
            GEOvdbProcessTypedGridTopology(*vdbPrim, densifyOp);
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
