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
/// @file SOP_OpenVDB_Vector_Split.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Split vector grids into component scalar grids.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <UT/UT_Interrupt.h>
#include <set>
#include <sstream>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Vector_Split: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Vector_Split(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Vector_Split() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

protected:
    OP_ERROR cookMySop(OP_Context&) override;
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    // Input vector grid group name
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip(
            "Specify a subset of the input VDB grids to be split.\n"
            "Vector-valued grids will be split into component scalar grids;\n"
            "all other grids will be unchanged.")
        .setDocumentation(
            "A subset of the input VDBs to be split"
            " (see [specifying volumes|/model/volumes#group])\n\n"
            "Vector-valued VDBs are split into component scalar VDBs;"
            " VDBs of other types are passed through unchanged."));

    // Toggle to keep/remove source grids
    parms.add(
        hutil::ParmFactory(PRM_TOGGLE, "remove_sources", "Remove Source VDBs")
        .setDefault(PRMoneDefaults)
        .setTooltip("Remove vector grids that have been split.")
        .setDocumentation("If enabled, delete vector grids that have been split."));

    // Toggle to copy inactive values in addition to active values
    parms.add(
        hutil::ParmFactory(PRM_TOGGLE, "copyinactive", "Copy Inactive Values")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "If enabled, split the values of both active and inactive voxels.\n"
            "If disabled, split the values of active voxels only."));

#ifndef SESI_OPENVDB
    // Verbosity toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose")
        .setDocumentation("If enabled, print debugging information to the terminal."));
#endif

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Vector Split",
        SOP_OpenVDB_Vector_Split::factory, parms, *table)
        .addInput("Vector VDBs to split into scalar VDBs")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Split a vector VDB primitive into three scalar VDB primitives.\"\"\"\n\
\n\
@overview\n\
\n\
This node will create three new scalar primitives named `<<input>>.x`,\n\
`<<input>>.y`, and `<<input>>.z`.\n\
\n\
TIP:\n\
    To reverse the split (i.e., to merge three scalar VDBs into a vector VDB),\n\
    use the [OpenVDB Vector Merge node|Node:sop/DW_OpenVDBVectorMerge]\n\
    and set the groups to `@name=*.x`, `@name=*.y`, and `@name=*.z`.\n\
\n\
@related\n\
- [OpenVDB Vector Merge|Node:sop/DW_OpenVDBVectorMerge]\n\
- [Node:sop/vdbvectorsplit]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


OP_Node*
SOP_OpenVDB_Vector_Split::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Vector_Split(net, name, op);
}


SOP_OpenVDB_Vector_Split::SOP_OpenVDB_Vector_Split(OP_Network* net,
    const char* name, OP_Operator* op):
    SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {

class VectorGridSplitter
{
private:
    const GEO_PrimVDB& mInVdb;
    hvdb::GridPtr mXGrid, mYGrid, mZGrid;
    bool mCopyInactiveValues;

public:
    VectorGridSplitter(const GEO_PrimVDB& _vdb, bool _inactive):
        mInVdb(_vdb), mCopyInactiveValues(_inactive) {}

    const hvdb::GridPtr& getXGrid() { return mXGrid; }
    const hvdb::GridPtr& getYGrid() { return mYGrid; }
    const hvdb::GridPtr& getZGrid() { return mZGrid; }

    template<typename VecGridT>
    void operator()(const VecGridT& vecGrid)
    {
        const std::string gridName = mInVdb.getGridName();

        using VecT = typename VecGridT::ValueType;
        using ScalarTreeT = typename VecGridT::TreeType::template
            ValueConverter<typename VecT::value_type>::Type;
        using ScalarGridT = typename openvdb::Grid<ScalarTreeT>;
        using ScalarGridPtr = typename ScalarGridT::Ptr;

        const VecT bkgd = vecGrid.background();

        // Construct the output scalar grids, with background values taken from
        // the components of the input vector grid's background value.
        ScalarGridPtr
            xGrid = ScalarGridT::create(bkgd.x()),
            yGrid = ScalarGridT::create(bkgd.y()),
            zGrid = ScalarGridT::create(bkgd.z());
        mXGrid = xGrid; mYGrid = yGrid; mZGrid = zGrid;

        // The output scalar grids share the input vector grid's transform.
        if (openvdb::math::Transform::Ptr xform = vecGrid.transform().copy()) {
            xGrid->setTransform(xform);
            yGrid->setTransform(xform);
            zGrid->setTransform(xform);
        }

        // Use accessors for fast sequential voxel access.
        typename ScalarGridT::Accessor
            xAccessor = xGrid->getAccessor(),
            yAccessor = yGrid->getAccessor(),
            zAccessor = zGrid->getAccessor();

        // For each tile or voxel value in the input vector tree,
        // set a corresponding value in each of the output scalar trees.
        openvdb::CoordBBox bbox;
        if (mCopyInactiveValues) {
            for (typename VecGridT::ValueAllCIter it = vecGrid.cbeginValueAll(); it; ++it) {
                if (!it.getBoundingBox(bbox)) continue;

                const VecT& val = it.getValue();
                const bool active = it.isValueOn();

                if (it.isTileValue()) {
                    xGrid->fill(bbox, val.x(), active);
                    yGrid->fill(bbox, val.y(), active);
                    zGrid->fill(bbox, val.z(), active);
                } else { // it.isVoxelValue()
                    xAccessor.setValue(bbox.min(), val.x());
                    yAccessor.setValue(bbox.min(), val.y());
                    zAccessor.setValue(bbox.min(), val.z());
                    if (!active) {
                        xAccessor.setValueOff(bbox.min());
                        yAccessor.setValueOff(bbox.min());
                        zAccessor.setValueOff(bbox.min());
                    }
                }
            }
        } else {
            for (typename VecGridT::ValueOnCIter it = vecGrid.cbeginValueOn(); it; ++it) {
                if (!it.getBoundingBox(bbox)) continue;

                const VecT& val = it.getValue();

                if (it.isTileValue()) {
                    xGrid->fill(bbox, val.x());
                    yGrid->fill(bbox, val.y());
                    zGrid->fill(bbox, val.z());
                } else { // it.isVoxelValue()
                    xAccessor.setValueOn(bbox.min(), val.x());
                    yAccessor.setValueOn(bbox.min(), val.y());
                    zAccessor.setValueOn(bbox.min(), val.z());
                }
            }
        }
    }
}; // class VectorGridSplitter

} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Vector_Split::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        duplicateSource(0, context);

        const bool copyInactiveValues = evalInt("copyinactive", 0, time);
        const bool removeSourceGrids = evalInt("remove_sources", 0, time);
#ifndef SESI_OPENVDB
        const bool verbose = evalInt("verbose", 0, time);
#else
        const bool verbose = false;
#endif

        UT_AutoInterrupt progress("Splitting VDB grids");

        using PrimVDBSet = std::set<GEO_PrimVDB*>;
        PrimVDBSet primsToRemove;

        // Get the group of grids to split.
        const GA_PrimitiveGroup* splitGroup = nullptr;
        {
            UT_String groupStr;
            evalString(groupStr, "group", 0, time);
            splitGroup = matchGroup(*gdp, groupStr.toStdString());
        }

        // Iterate over VDB primitives in the selected group.
        for (hvdb::VdbPrimIterator it(gdp, splitGroup); it; ++it) {
            if (progress.wasInterrupted()) return error();

            GU_PrimVDB* vdb = *it;

            const std::string gridName = vdb->getGridName();

            VectorGridSplitter op(*vdb, copyInactiveValues);
            bool ok = GEOvdbProcessTypedGridVec3(*vdb, op);

            if (!ok) {
                if (verbose && !gridName.empty()) {
                    addWarning(SOP_MESSAGE, (gridName + " is not a vector grid").c_str());
                }
                continue;
            }

            // Add the new scalar grids to the detail, copying attributes and
            // group membership from the input vector grid.
            const std::string
                xGridName = gridName.empty() ? "x" : gridName + ".x",
                yGridName = gridName.empty() ? "y" : gridName + ".y",
                zGridName = gridName.empty() ? "z" : gridName + ".z";
            GU_PrimVDB::buildFromGrid(*gdp, op.getXGrid(), vdb, xGridName.c_str());
            GU_PrimVDB::buildFromGrid(*gdp, op.getYGrid(), vdb, yGridName.c_str());
            GU_PrimVDB::buildFromGrid(*gdp, op.getZGrid(), vdb, zGridName.c_str());

            if (verbose) {
                std::ostringstream ostr;
                ostr << "Split ";
                if (!gridName.empty()) ostr << gridName << " ";
                ostr << "into " << xGridName << ", " << yGridName << " and " << zGridName;
                addMessage(SOP_MESSAGE, ostr.str().c_str());
            }

            primsToRemove.insert(vdb);
        }
        if (removeSourceGrids) {
            // Remove vector grids that were split.
            for (PrimVDBSet::iterator i = primsToRemove.begin(), e = primsToRemove.end();
                i != e; ++i)
            {
                gdp->destroyPrimitive(*(*i), /*andPoints=*/true);
            }
        }
        primsToRemove.clear();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
