// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file SOP_OpenVDB_Merge.cc
///
/// @authors Dan Bailey
///
/// @brief Merge OpenVDB grids.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/PointUtils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/Merge.h>
#include <openvdb/util/CpuTimer.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_Version.h>
#include <stdexcept>
#include <string>
#include <vector>


using namespace openvdb;

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Merge: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Merge(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Merge() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    class Cache: public SOP_VDBCacheOptions { OP_ERROR cookVDBSop(OP_Context&) override; };
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setTooltip("Specify a subset of the input VDBs to be modified.")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setDocumentation(
            "A subset of the input VDBs to be modified"
            " (see [specifying volumes|/model/volumes#group])"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("VDB Merge", SOP_OpenVDB_Merge::factory, parms, *table)
        .setMaxInputs()
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_OpenVDB_Merge::Cache; })
        .setDocumentation("");
}


OP_Node*
SOP_OpenVDB_Merge::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Merge(net, name, op);
}


SOP_OpenVDB_Merge::SOP_OpenVDB_Merge(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


OP_ERROR
SOP_OpenVDB_Merge::Cache::cookVDBSop(OP_Context& context)
{
    try {
        const fpreal time = context.getTime();

        const std::string groupName = evalStdString("group", time);

        UT_AutoInterrupt progress("Merging VDBs");

        // gridsToSteal are stolen
        // gridsToCopy are deep copied

        std::vector<FloatGrid::Ptr> gridsToSteal;
        std::vector<FloatGrid::ConstPtr> gridsToCopy;

        std::set<GU_PrimVDB*> vdbsToRemove;

        openvdb::util::CpuTimer timer("Collate In-Place Grids");

        hvdb::VdbPrimIterator vdbIt(gdp, matchGroup(*gdp, groupName));
        for (; vdbIt; ++vdbIt) {
            GU_PrimVDB* vdbPrim = *vdbIt;
            // only float VDBs supported for now
            if (vdbPrim->getStorageType() != UT_VDB_FLOAT)  continue;
            vdbsToRemove.insert(vdbPrim);
            auto gridPtr = openvdb::gridPtrCast<FloatGrid>(vdbPrim->getGridPtr());
            {
                // if tree is not unique, mark to be copied, otherwise mark to be stolen
                openvdb::TreeBase::Ptr localTreePtr = gridPtr->baseTreePtr();
                if (localTreePtr.use_count() <= 2) {
                    gridsToSteal.push_back(gridPtr);
                } else {
                    auto constGridPtr = openvdb::gridConstPtrCast<FloatGrid>(vdbPrim->getConstGridPtr());
                    gridsToCopy.push_back(constGridPtr);
                }
            }
        }

        timer.restart("Collate Other Grids");

        for (int i = 1; i < nInputs(); i++) {
            // Get the group of grids to process.
            const GU_Detail* pointsGeo = inputGeo(i);
            hvdb::VdbPrimCIterator vdbIt(pointsGeo, matchGroup(*pointsGeo, groupName));
            for (; vdbIt; ++vdbIt) {
                const GU_PrimVDB* constVdbPrim = *vdbIt;
                // only float VDBs supported for now
                if (constVdbPrim->getStorageType() != UT_VDB_FLOAT)  continue;
                // these grids cannot be stolen
                auto gridPtr = openvdb::gridConstPtrCast<FloatGrid>(constVdbPrim->getConstGridPtr());
                gridsToCopy.push_back(gridPtr);
            }
        }

        timer.stop();

        // merge the grids

        FloatGrid::Ptr result = tools::mergeSum<FloatGrid>(gridsToSteal, gridsToCopy);

        timer.start("Create Density VDB Prim");

        if (result) {
            hvdb::createVdbPrimitive(*gdp, result, "density");
        }

        timer.restart("Remove Old VDB Prims from First Input");

        // Remove primitives that were copied from input 0.
        for (GU_PrimVDB* vdb: vdbsToRemove) {
            if (vdb)    gdp->destroyPrimitive(*vdb, /*andPoints=*/true);
        }

        timer.stop();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}
