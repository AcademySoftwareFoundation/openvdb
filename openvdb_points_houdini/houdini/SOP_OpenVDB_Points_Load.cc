///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
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
/// @file SOP_OpenVDB_Points_Load.cc
///
/// @author Dan Bailey
///
/// @brief Explicitly loads OpenVDB points that are delay-loaded.


#include <openvdb_points/openvdb.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointLoad.h>

#include <openvdb/tools/LevelSetUtil.h> // sdfInteriorMask

#include "SOP_NodeVDBPoints.h"

#include <openvdb_houdini/Utils.h>
#include <houdini_utils/geometry.h>
#include <houdini_utils/ParmFactory.h>

using namespace openvdb;
using namespace openvdb::tools;
using namespace openvdb::math;

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


class SOP_OpenVDB_Points_Load: public hvdb::SOP_NodeVDBPoints
{
public:
    SOP_OpenVDB_Points_Load(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Points_Load() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned input) const { return (input == 1); }

protected:
    virtual bool updateParmsFlags();
    virtual OP_ERROR cookMySop(OP_Context&);

private:
    hvdb::Interrupter mBoss;
}; // class SOP_OpenVDB_Points_Load



////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    points::initialize();

    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Source Group")
        .setHelpText("Specify a subset of the input VDB grids to be loaded.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "usemask", "")
        .setHelpText(
            "If disabled, use the bounding box of the reference geometry\n"
            "as the loading region.")
        .setDefault(PRMzeroDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    parms.add(hutil::ParmFactory(PRM_STRING, "mask", "Mask VDB")
        .setHelpText("Specify a VDB grid whose active voxels are to be used as a loading mask.")
        .setChoiceList(&hutil::PrimGroupMenuInput2));

    //////////
    // Register this operator.

    hvdb::OpenVDBOpFactory("OpenVDB Points Load",
        SOP_OpenVDB_Points_Load::factory, parms, *table)
        .addInput("VDB Points")
        .addOptionalInput("Mask VDB or Bounding Geometry");
}


bool
SOP_OpenVDB_Points_Load::updateParmsFlags()
{
    bool changed = false;
    changed |= enableParm("mask", evalInt("usemask", 0, 0) != 0);
    return changed;
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Points_Load::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Points_Load(net, name, op);
}


SOP_OpenVDB_Points_Load::SOP_OpenVDB_Points_Load(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDBPoints(net, name, op)
{
}


namespace {

struct MaskOp
{
    typedef openvdb::BoolGrid MaskType;

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        maskGrid = MaskType::create(false);
        maskGrid->setTransform(grid.transform().copy());

        if (openvdb::GRID_LEVEL_SET == grid.getGridClass()) {
            BoolGrid::Ptr boolGrid = openvdb::tools::sdfInteriorMask(grid);
            maskGrid->tree().topologyUnion(boolGrid->tree());
        }
        else {
            maskGrid->tree().topologyUnion(grid.tree());
        }
    }

    MaskType::Ptr maskGrid;
};

} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Points_Load::cookMySop(OP_Context& context)
{
    typedef openvdb::tools::PointDataGrid PointDataGrid;

    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        const GU_Detail* maskGeo = inputGeo(1);

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSourceStealable(0, context);

        UT_AutoInterrupt progress("Processing Points Load");

        const bool useMask = evalInt("usemask", 0, time);

        // Get the group of grids to surface.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        hvdb::VdbPrimIterator vdbIt(gdp, group);

        // Handle no vdbs
        if (!vdbIt) {
            return error();
        }

        for (; vdbIt; ++vdbIt) {

            if (progress.wasInterrupted()) {
                throw std::runtime_error("processing was interrupted");
            }

            GU_PrimVDB* vdbPrim = *vdbIt;
            openvdb::GridBase::Ptr inGrid = vdbPrim->getGridPtr();

            if (!inGrid->isType<PointDataGrid>()) continue;
            PointDataGrid::Ptr pointDataGrid = openvdb::gridPtrCast<PointDataGrid>(inGrid);

            openvdb::tools::PointDataTree::LeafCIter leafIter = pointDataGrid->tree().cbeginLeaf();
            if (!leafIter) continue;

            // early exit if no delayed leaves (avoids redundant deep copy)

            bool hasDelayedLeaves = false;

            for (; leafIter; ++leafIter)
            {
                if (leafIter->buffer().isOutOfCore()) {
                    hasDelayedLeaves = true;
                    break;
                }
            }

            if (!hasDelayedLeaves)  continue;

            // deep copy the VDB tree if it is not already unique
            vdbPrim->makeGridUnique();

            PointDataGrid::Ptr outputGrid = openvdb::gridPtrCast<PointDataGrid>(vdbPrim->getGridPtr());

            if (!outputGrid) {
                addError(SOP_MESSAGE, "Failed to duplicate VDB Points");
                return error();
            }

            // load all points if mask is not provided as an input
            if (!maskGeo) {
                openvdb::tools::loadPoints(*outputGrid);
                return error();
            }

            if (useMask) {
                UT_String maskStr;
                evalString(maskStr, "mask", 0, time);
#if (UT_MAJOR_VERSION_INT >= 15)
                const GA_PrimitiveGroup* maskGroup = parsePrimitiveGroups(
                    maskStr.buffer(), GroupCreator(maskGeo));
#else
                const GA_PrimitiveGroup* maskGroup = parsePrimitiveGroups(
                    maskStr.buffer(), const_cast<GU_Detail*>(maskGeo));
#endif
                hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
                if (!maskIt) {
                    addWarning(SOP_MESSAGE, "Mask VDB not found.");
                    openvdb::tools::loadPoints(*outputGrid);
                    return error();
                }
                // load points based on topology, explicitly include grid types not
                // handled in GEOvdbProcessTypedGridTopology()
                MaskOp op;
                MaskOp::MaskType::Ptr maskGrid;
                const openvdb::GridBase& grid(maskIt->getConstGrid());
                if (grid.isType<MaskOp::MaskType>()) {
                    UT_VDBUtils::callTypedGrid<MaskOp::MaskType>(grid, op);
                    maskGrid = op.maskGrid;
                }
                else if (grid.isType<PointDataGrid>()) {
                    UT_VDBUtils::callTypedGrid<PointDataGrid>(grid, op);
                    maskGrid = op.maskGrid;
                }
                else {
                    GEOvdbProcessTypedGridTopology(**maskIt, op);
                    maskGrid = op.maskGrid;
                }
                openvdb::tools::loadPoints(*outputGrid, *maskGrid);
            }
            else {
                // implicitly compute mask from bounding box of input geo
                openvdb::BBoxd bbox;
                UT_BoundingBox box;
                maskGeo->computeQuickBounds(box);

                bbox.min()[0] = box.xmin();
                bbox.min()[1] = box.ymin();
                bbox.min()[2] = box.zmin();
                bbox.max()[0] = box.xmax();
                bbox.max()[1] = box.ymax();
                bbox.max()[2] = box.zmax();
                openvdb::tools::loadPoints(*outputGrid, bbox);
            }
        }

        return error();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}


////////////////////////////////////////

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
