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
/// @file SOP_OpenVDB_Analysis.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Compute gradient fields and other differential properties from VDB volumes

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/Mask.h> // for tools::interiorMask()
#include <openvdb/tools/GridTransformer.h>

#include <UT/UT_Interrupt.h>
#if (UT_VERSION_INT >= 0x0c050157) // 12.5.343 or later
#include <GEO/GEO_PrimVDB.h> // for GEOvdbProcessTypedGridScalar(), etc.
#endif

#include <sstream>
#include <stdexcept>
#include <string>

namespace cvdb = openvdb;
namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Analysis: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Analysis(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Analysis() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i) const override { return (i == 1); }

    static const char* sOpName[];

    enum OpId {
        OP_GRADIENT   = 0,
        OP_CURVATURE  = 1,
        OP_LAPLACIAN  = 2,
        OP_CPT        = 3,
        OP_DIVERGENCE = 4,
        OP_CURL       = 5,
        OP_MAGNITUDE  = 6,
        OP_NORMALIZE  = 7
    };

protected:
    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;
};


////////////////////////////////////////


const char* SOP_OpenVDB_Analysis::sOpName[] = {
    "gradient",
    "curvature",
    "laplacian",
    "closest point transform",
    "divergence",
    "curl",
    "magnitude",
    "normalize"
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    // Group pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDB grids to be processed.")
        .setDocumentation(
            "A subset of VDBs to analyze (see [specifying volumes|/model/volumes#group])"));

    // Operator
    {
        char const * const items[] = {
            "gradient",     "Gradient       (Scalar->Vector)",
            "curvature",    "Curvature     (Scalar->Scalar)",
            "laplacian",    "Laplacian      (Scalar->Scalar)",
            "closestpoint", "Closest Point (Scalar->Vector)",
            "divergence",   "Divergence    (Vector->Scalar)",
            "curl",         "Curl             (Vector->Vector)",
            "length",       "Length         (Vector->Scalar)",
            "normalize",    "Normalize     (Vector->Vector)",
            nullptr
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "operator", "Operator")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDocumentation("\
What to compute\n\
\n\
The labels on the items in the menu indicate what datatype\n\
the incoming VDB volume must be and the datatype of the output volume.\n\
\n\
Gradient (scalar -> vector):\n\
    The gradient of a scalar field\n\
\n\
Curvature (scalar -> scalar):\n\
    The mean curvature of a scalar field\n\
\n\
Laplacian (scalar -> scalar):\n\
    The Laplacian of a scalar field\n\
\n\
Closest Point (scalar -> vector):\n\
    The location, at each voxel, of the closest point on a surface\n\
    defined by the incoming signed distance field\n\
\n\
    You can use the resulting field with the\n\
    [OpenVDB Advect Points node|Node:sop/DW_OpenVDBAdvectPoints]\n\
    to stick points to the surface.\n\
\n\
Divergence (vector -> scalar):\n\
    The divergence of a vector field\n\
\n\
Curl (vector -> vector):\n\
    The curl of a vector field\n\
\n\
Magnitude (vector -> scalar):\n\
    The length of the vectors in a vector field\n\
\n\
Normalize (vector -> vector):\n\
    The vectors in a vector field divided by their lengths\n"));
    }

    parms.add(hutil::ParmFactory(PRM_STRING, "maskname", "Mask VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("VDB (from the second input) used to define the iteration space")
        .setDocumentation(
            "A VDB from the second input used to define the iteration space"
            " (see [specifying volumes|/model/volumes#group])\n\n"
            "The selected __Operator__ will be applied only where the mask VDB has"
            " [active|http://www.openvdb.org/documentation/doxygen/overview.html#subsecInactive]"
            " voxels or, if the mask VDB is a level set, only in the interior of the level set."));

    { // Output name
        char const * const items[] = {
            "keep",     "Keep Incoming VDB Names",
            "append",   "Append Operation Name",
            "custom",   "Custom Name",
            nullptr
        };
        parms.add(hutil::ParmFactory(PRM_ORD, "outputName", "Output Name")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("Rename output grid(s)")
            .setDocumentation(
                "How to name the generated VDB volumes\n\n"
                "If you choose __Keep Incoming VDB Names__, the generated fields"
                " will replace the input fields."));
    }

    parms.add(hutil::ParmFactory(PRM_STRING, "customName", "Custom Name")
        .setTooltip("Rename all output grids with this custom name")
        .setDocumentation("If this is not blank, the output VDB will use this name."));


    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "threaded", "Multithreaded"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Analysis", SOP_OpenVDB_Analysis::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("VDBs to Analyze")
        .addOptionalInput("Optional VDB mask input")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Compute an analytic property of a VDB volume, such as gradient or curvature.\"\"\"\n\
\n\
@overview\n\
\n\
This node computes certain properties from the values of VDB volumes,\n\
and generates new VDB volumes where the voxel values are the computed results.\n\
Using the __Output Name__ parameter you can choose whether the generated\n\
volumes replace the original volumes.\n\
\n\
@related\n\
\n\
- [OpenVDB Advect Points|Node:sop/DW_OpenVDBAdvectPoints]\n\
- [Node:sop/volumeanalysis]\n\
- [Node:sop/vdbanalysis]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Analysis::factory(OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Analysis(net, name, op);
}


SOP_OpenVDB_Analysis::SOP_OpenVDB_Analysis(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {

template<template<typename GridT, typename MaskType, typename InterruptT> class ToolT>
struct ToolOp
{
    ToolOp(bool t, hvdb::Interrupter& boss, const cvdb::BoolGrid *mask = nullptr)
        : mMaskGrid(mask)
        , mThreaded(t)
        , mBoss(boss)
    {
    }

    template<typename GridType>
    void operator()(const GridType& inGrid)
    {
        if (mMaskGrid) {

            // match transform
            cvdb::BoolGrid regionMask;
            regionMask.setTransform(inGrid.transform().copy());
            openvdb::tools::resampleToMatch<openvdb::tools::PointSampler>(
                *mMaskGrid, regionMask, mBoss);

            ToolT<GridType, cvdb::BoolGrid, hvdb::Interrupter> tool(inGrid, regionMask, &mBoss);
            mOutGrid = tool.process(mThreaded);

        } else {
            ToolT<GridType, cvdb::BoolGrid/*dummy*/, hvdb::Interrupter> tool(inGrid, &mBoss);
            mOutGrid = tool.process(mThreaded);
        }
    }

    const cvdb::BoolGrid    *mMaskGrid;
    hvdb::GridPtr           mOutGrid;
    bool                    mThreaded;
    hvdb::Interrupter&      mBoss;
};


struct MaskOp
{
    template<typename GridType>
    void operator()(const GridType& grid) { mMaskGrid = cvdb::tools::interiorMask(grid); }

    cvdb::BoolGrid::Ptr mMaskGrid;
};


} // unnamed namespace


////////////////////////////////////////


// Enable or disable parameters in the UI.
bool
SOP_OpenVDB_Analysis::updateParmsFlags()
{
    bool changed = false;

    bool useCustomName = evalInt("outputName", 0, 0) == 2;

    changed |= enableParm("customName", useCustomName);
#ifndef SESI_OPENVDB
    changed |= setVisibleState("customName", useCustomName);
#endif

    const bool hasMask = (2 == nInputs());
    changed |= enableParm("maskname", hasMask);

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Analysis::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        const fpreal time = context.getTime();

        // This does a shallow copy of VDB-grids and deep copy of native Houdini primitives.
        duplicateSource(0, context);

        // Get the group of grids to be transformed.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group = matchGroup(*gdp, groupStr.toStdString());

        const int whichOp = static_cast<int>(evalInt("operator", 0, time));
        if (whichOp < 0 || whichOp > 7) {
            std::ostringstream ostr;
            ostr << "expected 0 <= operator <= 7, got " << whichOp;
            throw std::runtime_error(ostr.str().c_str());
        }

        const bool threaded = true;

        hvdb::Interrupter boss(
            (std::string("Computing ") + sOpName[whichOp] + " of VDB grids").c_str());


        // Check mask input
        const GU_Detail* maskGeo = inputGeo(1);
        cvdb::BoolGrid::Ptr maskGrid;

        if (maskGeo) {

            UT_String maskStr;
            evalString(maskStr, "maskname", 0, time);

#if (UT_MAJOR_VERSION_INT >= 15)
            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(maskStr.buffer(), GroupCreator(maskGeo));
#else
            const GA_PrimitiveGroup * maskGroup =
                parsePrimitiveGroups(maskStr.buffer(), const_cast<GU_Detail*>(maskGeo));
#endif
            hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
            if (maskIt) {
                MaskOp op;
                UTvdbProcessTypedGridTopology(maskIt->getStorageType(), maskIt->getGrid(), op);
                maskGrid = op.mMaskGrid;
            }

            if (!maskGrid) addWarning(SOP_MESSAGE, "Mask VDB not found.");
        }


        // For each VDB primitive (with a non-null grid pointer) in the given group...
        std::string operationName;
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (boss.wasInterrupted()) throw std::runtime_error("was interrupted");

            GU_PrimVDB* vdb = *it;

            hvdb::GridPtr outGrid;
            bool ok = true;
            switch (whichOp)
            {
                case OP_GRADIENT: // gradient of scalar field
                {
                    ToolOp<cvdb::tools::Gradient> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridScalar(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_gradient";
                    break;
                }
                case OP_CURVATURE: // mean curvature of scalar field
                {
                    ToolOp<cvdb::tools::MeanCurvature> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridScalar(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_curvature";
                    break;
                }
                case OP_LAPLACIAN: // Laplacian of scalar field
                {
                    ToolOp<cvdb::tools::Laplacian> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridScalar(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_laplacian";
                    break;
                }
                case OP_CPT: // closest point transform of scalar level set
                {
                    ToolOp<cvdb::tools::Cpt> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridScalar(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_cpt";
                    break;
                }
                case OP_DIVERGENCE: // divergence of vector field
                {
                    ToolOp<cvdb::tools::Divergence> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridVec3(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_divergence";
                    break;
                }
                case OP_CURL: // curl (rotation) of vector field
                {
                    ToolOp<cvdb::tools::Curl> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridVec3(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_curl";
                    break;
                }
                case OP_MAGNITUDE: // magnitude of vector field
                {
                    ToolOp<cvdb::tools::Magnitude> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridVec3(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_magnitude";
                    break;
                }
                case OP_NORMALIZE: // normalize vector field
                {
                    ToolOp<cvdb::tools::Normalize> op(threaded, boss, maskGrid.get());
                    ok = GEOvdbProcessTypedGridVec3(*vdb, op, /*makeUnique=*/false);
                    if (ok) outGrid = op.mOutGrid;
                    operationName = "_normalize";
                    break;
                }
            }

            if (!ok) {
                UT_String inGridName = it.getPrimitiveNameOrIndex();
                std::ostringstream ss;
                ss << "Can't compute " << sOpName[whichOp] << " from grid";
                if (inGridName.isstring()) ss << " " << inGridName;
                ss << " of type " << vdb->getGrid().valueType();
                addWarning(SOP_MESSAGE, ss.str().c_str());
            }

            // Rename grid
            std::string gridName = vdb->getGridName();
            const auto renaming = evalInt("outputName", 0, time);
            if (renaming == 1) {
                if (operationName.size() > 0) gridName += operationName;
            } else if (renaming == 2) {
                UT_String customName;
                evalString(customName, "customName", 0, time);
                if (customName.length() > 0) gridName = customName.toStdString();
            }

            // Replace the original VDB primitive with a new primitive that contains
            // the output grid and has the same attributes and group membership.
            hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdb, true, gridName.c_str());
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
