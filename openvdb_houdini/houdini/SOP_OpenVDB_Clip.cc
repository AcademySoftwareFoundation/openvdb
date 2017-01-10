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

/// @file SOP_OpenVDB_Clip.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Clip grids

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/Clip.h> // for tools::clip()
#include <openvdb/tools/LevelSetUtil.h> // for tools::sdfInteriorMask()
#include <openvdb/points/PointDataGrid.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Clip: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Clip(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Clip() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned input) const override { return (input == 1); }

protected:
    bool updateParmsFlags() override;
    OP_ERROR cookMySop(OP_Context&) override;
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Source Group")
        .setHelpText("Specify a subset of the input VDB grids to be clipped.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "usemask", "")
        .setHelpText(
            "If disabled, use the bounding box of the reference geometry\n"
            "as the clipping region.")
        .setDefault(PRMzeroDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    parms.add(hutil::ParmFactory(PRM_STRING, "mask", "Mask VDB")
        .setHelpText("Specify a VDB grid whose active voxels are to be used as a clipping mask.")
        .setChoiceList(&hutil::PrimGroupMenuInput2));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "inside", "Keep Inside")
        .setHelpText(
            "If enabled, keep voxels that lie inside the clipping region.\n"
            "If disabled, keep voxels that lie outside the clipping region.")
        .setDefault(PRMoneDefaults));

    hvdb::OpenVDBOpFactory("OpenVDB Clip", SOP_OpenVDB_Clip::factory, parms, *table)
        .addInput("VDBs")
        .addInput("Mask VDB or bounding geometry");
}


bool
SOP_OpenVDB_Clip::updateParmsFlags()
{
    bool changed = false;
    changed |= enableParm("mask", evalInt("usemask", 0, 0) != 0);
    return changed;
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Clip::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Clip(net, name, op);
}


SOP_OpenVDB_Clip::SOP_OpenVDB_Clip(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {

struct LevelSetMaskOp
{
    template<typename GridType>
    void operator()(const GridType& grid)
    {
        outputGrid = openvdb::tools::sdfInteriorMask(grid);
    }

    hvdb::GridPtr outputGrid;
};


struct BBoxClipOp
{
    BBoxClipOp(const openvdb::BBoxd& bbox_, bool inside_ = true):
        bbox(bbox_), inside(inside_)
    {}

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        outputGrid = openvdb::tools::clip(grid, bbox, inside);
    }

    openvdb::BBoxd bbox;
    hvdb::GridPtr outputGrid;
    bool inside = true;
};


template<typename GridType>
struct MaskClipDispatchOp
{
    MaskClipDispatchOp(const GridType& grid_, bool inside_ = true):
        grid(&grid_), inside(inside_)
    {}

    template<typename MaskGridType>
    void operator()(const MaskGridType& mask)
    {
        outputGrid.reset();
        if (grid) outputGrid = openvdb::tools::clip(*grid, mask, inside);
    }

    const GridType* grid;
    hvdb::GridPtr outputGrid;
    bool inside = true;
};


struct MaskClipOp
{
    MaskClipOp(hvdb::GridCPtr mask_, bool inside_ = true):
        mask(mask_), inside(inside_)
    {}

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        outputGrid.reset();
        if (mask) {
            // Dispatch on the mask grid type, now that the source grid type is resolved.
            MaskClipDispatchOp<GridType> op(grid, inside);
            UTvdbProcessTypedGridTopology(UTvdbGetGridType(*mask), *mask, op);
            outputGrid = op.outputGrid;
        }
    }

    hvdb::GridCPtr mask;
    hvdb::GridPtr outputGrid;
    bool inside = true;
};

} // unnamed namespace


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Clip::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);

        const GU_Detail* maskGeo = inputGeo(1);

        const bool
            useMask = evalInt("usemask", 0, time),
            inside = evalInt("inside", 0, time);

        openvdb::BBoxd bbox;
        hvdb::GridCPtr maskGrid;
        if (maskGeo) {
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
                if (maskIt) {
                    if (maskIt->getConstGrid().getGridClass() == openvdb::GRID_LEVEL_SET) {
                        // If the mask grid is a level set, extract an interior mask from it.
                        LevelSetMaskOp op;
                        GEOvdbProcessTypedGridScalar(**maskIt, op);
                        maskGrid = op.outputGrid;
                    } else {
                        maskGrid = maskIt->getConstGridPtr();
                    }
                }
                if (!maskGrid) {
                    addError(SOP_MESSAGE, "mask VDB not found");
                    return error();
                }
            } else {
                UT_BoundingBox box;
                maskGeo->computeQuickBounds(box);

                bbox.min()[0] = box.xmin();
                bbox.min()[1] = box.ymin();
                bbox.min()[2] = box.zmin();
                bbox.max()[0] = box.xmax();
                bbox.max()[1] = box.ymax();
                bbox.max()[2] = box.zmax();
            }
        }

        // Get the group of grids to process.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        int numLevelSets = 0;
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (it->getConstGrid().getGridClass() == openvdb::GRID_LEVEL_SET) {
                ++numLevelSets;
            }

            hvdb::GridPtr outGrid;
            if (maskGrid) {
                MaskClipOp op(maskGrid, inside);
                if (GEOvdbProcessTypedGridTopology(**it, op)) { // all Houdini-supported grid types
                    outGrid = op.outputGrid;
                } else if (it->getConstGrid().isType<openvdb::points::PointDataGrid>()) {
                    addWarning(SOP_MESSAGE,
                        "only bounding box clipping is currently supported for point data grids");
                }
            } else {
                BBoxClipOp op(bbox, inside);
                if (GEOvdbProcessTypedGridTopology(**it, op)) { // all Houdini-supported grid types
                    outGrid = op.outputGrid;
                } else if (it->getConstGrid().isType<openvdb::points::PointDataGrid>()) {
                    if (inside) {
                        outGrid = it->getConstGrid().deepCopyGrid();
                        outGrid->clipGrid(bbox);
                    } else {
                        addWarning(SOP_MESSAGE,
                            "only Keep Inside mode is currently supported for point data grids");
                    }
                }
            }

            // Replace the original VDB primitive with a new primitive that contains
            // the output grid and has the same attributes and group membership.
            hvdb::replaceVdbPrimitive(*gdp, outGrid, **it, true);
        }

        if (numLevelSets > 0) {
            if (numLevelSets == 1) {
                addWarning(SOP_MESSAGE, "a level set grid was clipped;"
                    " the resulting grid might not be a valid level set");
            } else {
                addWarning(SOP_MESSAGE, "some level sets were clipped;"
                    " the resulting grids might not be valid level sets");
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
