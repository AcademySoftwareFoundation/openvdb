///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
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

/// @file SOP_OpenVDB_Points_Delete.cc
///
/// @author Francisco Gochez, Dan Bailey
///
/// @brief Delete points that are members of specific groups

#include <openvdb/openvdb.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointDelete.h>

#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/PointUtils.h>
#include <openvdb_houdini/Utils.h>
#include <houdini_utils/geometry.h>
#include <houdini_utils/ParmFactory.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#if UT_MAJOR_VERSION_INT >= 16
#define VDB_COMPILABLE_SOP 1
#else
#define VDB_COMPILABLE_SOP 0
#endif


using namespace openvdb;
using namespace openvdb::points;
using namespace openvdb::math;

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_Points_Delete: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Points_Delete(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Points_Delete() override = default;

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

#if VDB_COMPILABLE_SOP
    class Cache: public SOP_VDBCacheOptions { OP_ERROR cookVDBSop(OP_Context&) override; };
#else
protected:
    OP_ERROR cookVDBSop(OP_Context&) override;
#endif

protected:
    bool updateParmsFlags() override;
}; // class SOP_OpenVDB_Points_Delete


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    openvdb::initialize();

    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input point data grids to delete from.")
        .setChoiceList(&hutil::PrimGroupMenu));

    parms.add(hutil::ParmFactory(PRM_STRING, "vdbpointsgroups", "VDB Points Groups")
        .setHelpText("Specify VDB points groups to delete.")
        .setChoiceList(&hvdb::VDBPointsGroupMenuInput1));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invert", "Invert")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Invert point deletion so that points not belonging to any of the "
            "groups will be deleted."));
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "dropgroups", "Drop Points Groups")
        .setDefault(PRMoneDefaults)
        .setHelpText("Drop the VDB points groups that were used for deletion. This option is "
            "ignored if \"invert\" is enabled."));

    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory("OpenVDB Points Delete",
        SOP_OpenVDB_Points_Delete::factory, parms, *table)
        .addInput("VDB Points")
#if VDB_COMPILABLE_SOP
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_OpenVDB_Points_Delete::Cache; })
#endif
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Delete points that are members of specific groups.\"\"\"\n\
\n\
@overview\n\
\n\
The OpenVDB Points Delete SOP allows deletion of points that are members\n\
of a supplied group(s).\n\
An invert toggle may be enabled to allow deleting points that are not\n\
members of the supplied group(s).\n\
\n\
@related\n\
- [OpenVDB Points Convert|Node:sop/DW_OpenVDBPointsConvert]\n\
- [OpenVDB Points Group|Node:sop/DW_OpenVDBPointsGroup]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}

bool
SOP_OpenVDB_Points_Delete::updateParmsFlags()
{
    const bool invert = evalInt("invert", 0, 0) != 0;

    return enableParm("dropgroups", !invert);
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Points_Delete::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Points_Delete(net, name, op);
}


SOP_OpenVDB_Points_Delete::SOP_OpenVDB_Points_Delete(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


OP_ERROR
VDB_NODE_OR_CACHE(VDB_COMPILABLE_SOP, SOP_OpenVDB_Points_Delete)::cookVDBSop(OP_Context& context)
{
    try {
#if !VDB_COMPILABLE_SOP
        hutil::ScopedInputLock lock(*this, context);
        lock.markInputUnlocked(0);
        if (duplicateSourceStealable(0, context) >= UT_ERROR_ABORT) return error();
#endif

        const std::string groups = evalStdString("vdbpointsgroups", context.getTime());

        // early exit if the VDB points group field is empty
        if (groups.empty()) return error();

        UT_AutoInterrupt progress("Processing points group deletion");

        const bool invert = evalInt("invert", 0, context.getTime());
        const bool drop = evalInt("dropgroups", 0, context.getTime());

        // select Houdini primitive groups we wish to use
        hvdb::VdbPrimIterator vdbIt(gdp,
            matchGroup(*gdp, evalStdString("group", context.getTime())));

        for (; vdbIt; ++vdbIt) {
            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }
            GU_PrimVDB* vdbPrim = *vdbIt;

            PointDataGrid::ConstPtr inputGrid =
                    openvdb::gridConstPtrCast<PointDataGrid>(vdbPrim->getConstGridPtr());

            // early exit if the grid is of the wrong type
            if (!inputGrid) continue;

            // early exit if the tree is empty
            auto leafIter = inputGrid->tree().cbeginLeaf();
            if (!leafIter) continue;

            // extract names of all selected VDB groups

            std::vector<std::string> pointGroups;

            // the "exclude groups" parameter to parseNames is not used in this context,
            // so we disregard it by storing it in a temporary variable

            std::vector<std::string> tmp;

            AttributeSet::Descriptor::parseNames(pointGroups, tmp, groups);

            // determine in any of the requested groups are actually present in the tree

            const AttributeSet::Descriptor& descriptor = leafIter->attributeSet().descriptor();
            const bool hasPointsToDrop = std::any_of(pointGroups.begin(), pointGroups.end(),
                [&descriptor](const std::string& group) { return descriptor.hasGroup(group); });

            if (!hasPointsToDrop) continue;

            // deep copy the VDB tree if it is not already unique
            vdbPrim->makeGridUnique();

            PointDataGrid& outputGrid = UTvdbGridCast<PointDataGrid>(vdbPrim->getGrid());
            deleteFromGroups(outputGrid.tree(), pointGroups, invert, drop);
        }

    } catch (const std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
