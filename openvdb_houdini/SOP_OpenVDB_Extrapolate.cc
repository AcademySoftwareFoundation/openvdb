// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
// @file SOP_OpenVDB_Extrapolate.cc
//
// @author FX R&D OpenVDB team
//
// @brief Extrapolate SDF or attributes off a level set surface

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/FastSweeping.h>
#include <stdexcept>
#include <string>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Extrapolate: public hvdb::SOP_NodeVDB
{
public:

    SOP_OpenVDB_Extrapolate(OP_Network*, const char* name, OP_Operator*);

    ~SOP_OpenVDB_Extrapolate() override {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned input) const override { return (input == 1); }

    class Cache: public SOP_VDBCacheOptions
    {
    protected:
        OP_ERROR cookVDBSop(OP_Context&) override;
    private:
        template<typename GridT>
        bool process(hvdb::GridCPtr maskGrid, 
            hvdb::GU_PrimVDB* lsPrim, 
            fpreal time);
    }; // class Cache

protected:
    bool updateParmsFlags() override;
};


////////////////////////////////////////


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Source Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDB level sets.")
        .setDocumentation(
            "A subset of the input VDB level sets to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    // Modes
    parms.add(hutil::ParmFactory(PRM_STRING, "mode", "Operation")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "dilate",   "Dilate SDF",
            "mask",     "Extrapolate SDF Into Mask",
            "convert",  "Convert Scalar VDB Into SDF" ///< @todo move to Convert SOP
        })
        .setDefault("dilate")
        .setDocumentation(
            "The operation to perform\n\n"
            "Dilate SDF:\n"
            "    Extrapolate a narrow-band signed distance field.\n"
            "Extrapolate SDF Into Mask:\n"
            "    Extrapolate a signed distance field into the active voxels of a mask.\n"
            "Convert Scalar VDB Into SDF:\n"
            "    Compute a signed distance field for an isosurface\n"
            "    of a given scalar field. Active input voxels with\n"
            "    scalar values above the given isoValue will have\n"
            "    NEGATIVE distance values on output, i.e.\n"
            "    they are assumed to be INSIDE the uso-surface."));

    parms.add(hutil::ParmFactory(PRM_STRING, "mask", "Mask VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("Specify a VDB volume whose active voxels are to be used as a mask.")
        .setDocumentation(
            "A VDB volume whose active voxels are to be used as a mask"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "tiles", "Ignore Active Tiles")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Ignore active tiles in scalar field and mask VDBs.")
        .setDocumentation(
            "Ignore active tiles in scalar field and mask VDBs.\n\n"
            "This option should normally be disabled, but note that active tiles\n"
            "(sparsely represented regions of constant value) will in that case\n"
            "be densified, which could significantly increase memory usage.\n\n"
            "Proper signed distance fields don't have active tiles.\n"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "dilate", "Dilation")
        .setDefault(3)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setTooltip("The number of voxels by which to dilate the level set narrow band"));

    // Dilation Pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "nn", "Dilation Pattern")
         .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "NN6",  "Faces",
            "NN18", "Faces and Edges",
            "NN26", "Faces, Edges and Vertices"
         })
         .setDefault("NN6")
         .setTooltip("Select the neighborhood pattern for the dilation operation.")
         .setDocumentation(
             "The neighborhood pattern for the dilation operation\n\n"
             "__Faces__ is fastest.  __Faces, Edges and Vertices__ is slowest\n"
             "but can produce the best results for large dilations.\n"
             "__Faces and Edges__ is intermediate in speed and quality.\n"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "isovalue", "Isovalue")
         .setDefault(0.5)
         .setRange(PRM_RANGE_UI, -3, PRM_RANGE_UI, 3)
         .setTooltip("Isovalue for which the SDF is computed"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "sweeps", "Iterations")
         .setDefault(1)
         .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 5)
         .setTooltip(
            "The desired number of iterations of the Fast Sweeping algorithm"
            " (one is often enough)"));

    hvdb::OpenVDBOpFactory("OpenVDB Extrapolate", SOP_OpenVDB_Extrapolate::factory, parms, *table)
        .addInput("Level Set VDB")
        .addOptionalInput("Mask VDB")
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_OpenVDB_Extrapolate::Cache; })
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Extrapolate a VDB signed distance field.\"\"\"\n\
\n\
@overview\n\
\n\
This node extrapolates signed distance fields stored as VDB volumes.\n\
Optionally, extrapolation can be masked with another VDB, so that\n\
new distances are computed only where the mask is active.\n\
\n\
@related\n\
- [OpenVDB Convert|Node:sop/DW_OpenVDBConvert]\n\
- [OpenVDB Rebuild Level Set|Node:sop/DW_OpenVDBRebuildLevelSet]\n\
- [Node:sop/isooffset]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


bool
SOP_OpenVDB_Extrapolate::updateParmsFlags()
{
    UT_String str;
    bool changed = false;
    evalString(str, "mode", 0, 0);
    changed |= enableParm("mask", str == "mask");
    changed |= enableParm("dilate", str == "dilate");
    changed |= enableParm("nn", str == "dilate");
    changed |= enableParm("isovalue", str == "convert");
    changed |= enableParm("tiles", str != "dilate");
    return changed;
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Extrapolate::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Extrapolate(net, name, op);
}


SOP_OpenVDB_Extrapolate::SOP_OpenVDB_Extrapolate(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


namespace {

template <typename GridT>
struct FastSweepMaskOp
{
    FastSweepMaskOp(typename GridT::ConstPtr inGrid, bool ignoreTiles, int iter)
        : inGrid(inGrid), ignoreActiveTiles(ignoreTiles), iter(iter) {}

    template<typename MaskGridType>
    void operator()(const MaskGridType& mask)
    {
        outGrid = openvdb::tools::maskSdf(*inGrid, mask, ignoreActiveTiles, iter);
    }

    typename GridT::ConstPtr inGrid;
    const bool ignoreActiveTiles;
    const int iter;
    typename GridT::Ptr outGrid;
};

} // unnamed namespace


////////////////////////////////////////


template<typename GridT>
bool
SOP_OpenVDB_Extrapolate::Cache::process(
    hvdb::GridCPtr maskGrid,
    hvdb::GU_PrimVDB* lsPrim,
    fpreal time)
{
    typename GridT::Ptr outGrid; 
    typename GridT::ConstPtr inGrid = openvdb::gridConstPtrCast<GridT>(lsPrim->getConstGridPtr());

    UT_String mode;
    evalString(mode, "mode", 0, time);
    const int nSweeps = static_cast<int>(evalInt("sweeps", 0, time));

    using namespace openvdb::tools;
    if (mode == "mask") {
        FastSweepMaskOp<GridT> op(inGrid, evalInt("tiles", 0, time), nSweeps);
        UTvdbProcessTypedGridTopology(UTvdbGetGridType(*maskGrid), *maskGrid, op);
        outGrid = op.outGrid;
    } else if (mode == "dilate") {
        UT_String str;
        evalString(str, "nn", 0, time);
        const NearestNeighbors nn =
            (str == "NN18") ? NN_FACE_EDGE : ((str == "NN26") ? NN_FACE_EDGE_VERTEX : NN_FACE);
        outGrid = dilateSdf(*inGrid, static_cast<int>(evalInt("dilate", 0, time)), nn, nSweeps);
    } else if (mode == "convert") {
        // TODO: Confirm that convert is meant to be used for fogToSdf. Previously, the API call
        // is maskSDF(*outGrid, evalFloat("isovalue", 0, time), evalInt("tiles", 0, time), nSweeps);
        outGrid = fogToSdf(*inGrid, evalFloat("isovalue", 0, time), nSweeps);
        lsPrim->setVisualization(GEO_VOLUMEVIS_ISO, lsPrim->getVisIso(), lsPrim->getVisDensity());
    }

    // Replace the original VDB primitive with a new primitive that contains
    // the output grid and has the same attributes and group membership.
    hvdb::replaceVdbPrimitive(*gdp, outGrid, *lsPrim, true);

    return true;
}


OP_ERROR
SOP_OpenVDB_Extrapolate::Cache::cookVDBSop(OP_Context& context)
{
    try {
        const fpreal time = context.getTime();

        const GU_Detail* maskGeo = inputGeo(1);

        hvdb::GridCPtr maskGrid;
        if (evalStdString("mode", time) == "mask") {// selected to use a mask
            if (maskGeo) {// second input exists
                const GA_PrimitiveGroup* maskGroup = parsePrimitiveGroups(
                    evalStdString("mask", time).c_str(), GroupCreator(maskGeo));

                hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);

                if (maskIt) maskGrid = maskIt->getConstGridPtr();// only use the first grid

                if (!maskGrid) {
                    addError(SOP_MESSAGE, "mask VDB not found");
                    return error();
                }
            } else {
              addError(SOP_MESSAGE, "Please provide a mask VDB to the second input");
              return error();
            }
        }

        // Get the group of level sets to process.
        const GA_PrimitiveGroup* group = matchGroup(*gdp, evalStdString("group", time));
        for (hvdb::VdbPrimIterator it(gdp, group); it; ++it) {
            if (!this->process<openvdb::FloatGrid >(maskGrid, *it, time) &&
                !this->process<openvdb::DoubleGrid>(maskGrid, *it, time) ) {
                std::string s = it.getPrimitiveNameOrIndex().toStdString();
                s = "VDB primitive " + s + " was skipped because it is not a floating-point Grid.";
                addWarning(SOP_MESSAGE, s.c_str());
                continue;
            }
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}
