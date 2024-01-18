// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
// @file SOP_OpenVDB_Extrapolate.cc
//
// @author FX R&D OpenVDB team
//
// @brief Extrapolate SDF or attributes off a level set surface

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/Interpolation.h>
#include <stdexcept>
#include <string>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


namespace {

struct FastSweepingParms {
    FastSweepingParms() :
        mTime(0.0f),
        mFSGroup(nullptr),
        mExtGroup(nullptr),
        mMode(""),
        mNeedExt(false),
        mIgnoreTiles(false),
        mConvertOrRenormalize(false),
        mNSweeps(1),
        mPattern(""),
        mDilate(1),
        mFSPrimName(""),
        mExtPrimName(""),
        mExtFieldProcessed(false),
        mSweepingDomain(openvdb::tools::FastSweepingDomain::SWEEP_ALL),
        mNewFSGrid(nullptr),
        mNewExtGrid(nullptr)
    { }

    fpreal mTime;
    const GA_PrimitiveGroup* mFSGroup;
    const GA_PrimitiveGroup* mExtGroup;
    UT_String mMode;
    bool mNeedExt;
    bool mIgnoreTiles;
    bool mConvertOrRenormalize;
    int mNSweeps;
    UT_String mPattern;
    int mDilate;
    std::string mFSPrimName;
    std::string mExtPrimName;
    bool mExtFieldProcessed;
    openvdb::tools::FastSweepingDomain mSweepingDomain;

    // updated fast sweeping grid placeholder
    hvdb::Grid::Ptr mNewFSGrid;

    // updated extension grid placeholder
    hvdb::Grid::Ptr mNewExtGrid;
};


/// @brief Helper class to be used with GEOvdbApply and calling
///        openvdb::tools::maskSdf. The mask VDB is allowed to be
///        of any grid type.
template <typename GridT>
struct FastSweepingMaskOp
{
    FastSweepingMaskOp(const FastSweepingParms& parms, typename GridT::ConstPtr inGrid)
        : mParms(parms), mInGrid(inGrid), mOutGrid(nullptr) {}

    template<typename MaskGridType>
    void operator()(const MaskGridType& mask)
    {
        mOutGrid = openvdb::tools::maskSdf(*mInGrid, mask, mParms.mIgnoreTiles, mParms.mNSweeps);
    }

    const FastSweepingParms& mParms;
    typename GridT::ConstPtr mInGrid;
    hvdb::Grid::Ptr mOutGrid;
}; // FastSweepingMaskOp


/// @brief Sampler functor for the Dirichlet boundary condition on the
///        surface of the field to be extended.
template<typename ExtGridT>
struct DirichletSamplerOp
{
    using ExtValueT = typename ExtGridT::ValueType;
    using SamplerT = openvdb::tools::GridSampler<ExtGridT, openvdb::tools::BoxSampler>;

    DirichletSamplerOp(typename ExtGridT::ConstPtr functorGrid, SamplerT sampler)
        : mFunctorGrid (functorGrid),
          mSampler(sampler)
    {}

    ExtValueT operator()(const openvdb::Vec3d& xyz) const
    {
        return static_cast<ExtValueT>(mSampler.isSample(xyz));
    }

    typename ExtGridT::ConstPtr mFunctorGrid;
    SamplerT mSampler;
}; // DirichletSamplerOp

} // unnamed namespace


////////////////////////////////////////


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
        OP_ERROR evalFastSweepingParms(OP_Context&, FastSweepingParms&);
    private:
        /// @brief Helper function to process the chosen mode of operation.
        /// @details The template parameter resolves the type of the scalar
        ///          Fast Sweeping grid, i.e. UT_VDB_FLOAT or UT_VDB_DOUBLE.
        ///          If the chosen operation needs an extension grid, then
        ///          it will go through the extension primitives as defined by
        ///          the extension group and update them.
        template<typename FSGridT>
        bool processHelper(
            FastSweepingParms& parms,
            GU_PrimVDB* lsPrim,
            typename FSGridT::ValueType fsIsoValue = typename FSGridT::ValueType(0),
            const GU_PrimVDB* maskPrim = nullptr);

        /// @brief Process the Fast Sweeping operation.
        /// @details Calls the appropriate Fast Sweeping function from
        ///          tools/FastSweeping.h. It will update parms.mNewFSGrid and
        ///          parms.mNewExtGrid.
        template<typename FSGridT, typename ExtGridT = FSGridT>
        bool process(
            FastSweepingParms& parms,
            GU_PrimVDB* lsPrim,
            typename FSGridT::ValueType fsIsoValue = typename FSGridT::ValueType(0),
            const GU_PrimVDB* maskPrim = nullptr,
            GU_PrimVDB* exPrim = nullptr,
            const typename ExtGridT::ValueType& background = typename ExtGridT::ValueType(0));
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

    // Level set/Fog grid
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Source Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the input VDB scalar grid(s).")
        .setDocumentation(
            "A subset of the input VDB scalar grid(s) to be processed\n"
            "in an operation involving Fast Sweeping.\n"
            "(see [specifying volumes|/model/volumes#group])."));

    // Extension fields
    parms.add(hutil::ParmFactory(PRM_STRING, "extfields", "Extension Group")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setTooltip("Specify a subset of the VDB grid(s) to be extended off\n"
            "the surface defined by the scalar grid as specified by the __Source Group__.")
        .setDocumentation("Arbitrary VDB fields picked up by this group\n"
            "will be extended off an iso-surface of a scalar VDB (fog/level set)\n"
            "as specified by the __Source Group__. The mode enables this\n"
            "parameter is __Extend Field(s) Off Fog VDB__ or __Extend Field(s) Off SDF__."));

    // Mask grid
    parms.add(hutil::ParmFactory(PRM_STRING, "mask", "Mask VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("Specify a VDB volume whose active voxels are to be used as a mask.")
        .setDocumentation(
            "A VDB volume whose active voxels are to be used as a mask\n"
            "(see [specifying volumes|/model/volumes#group]).\n"
            "The mode that enables the use of this parameter is\n"
            "__Expand SDF Into Mask SDF__."));

    // Sweep
    parms.add(hutil::ParmFactory(PRM_HEADING, "sweep", "General Sweep")
         .setDocumentation(
             "These parameters control the Fast Sweeping operation."));

    // Modes
    parms.add(hutil::ParmFactory(PRM_STRING, "mode", "Operation")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "dilate",      "Expand SDF Narrowband",
            "mask",        "Expand SDF Into Mask SDF",
            "convert",     "Convert Fog VDB To SDF", ///< @todo move to Convert SOP
            "renormalize", "Renormalize SDF", // by solving the Eikonal equation
            "fogext",      "Extend Field(s) Off Fog VDB",
            "sdfext",      "Extend Field(s) Off SDF",
        })
        .setDefault("dilate")
        .setTooltip("The mode __Expand SDF Narrowband__, __Expand SDF Into Mask SDF__,\n"
            "__Convert Fog VDB To SDF__, and __Renormalize SDF__ will modify\n"
            " the scalar grid(s) specified by the __Source Group__ parameter. The mode\n"
            "__Extend Field(s) Off Fog VDB__ and __Extend Field(s) Off SDF__ will modify\n"
            "the grid(s) specified by the __Extension Group__ parameter and possibly\n"
            " the scalar grid specified by the __Source Group__ if the toggle\n"
            "__Convert Fog To SDF or Renormalize SDF__ is checked.")
        .setDocumentation(
            "The operation to perform\n\n"
            "__Expand SDF Narrowband__:\n"
            "    Dilates the narrowband of an existing signed distance field by a specified\n"
            "    number of voxels.\n"
            "__Expand SDF Into Mask SDF__:\n"
            "    Expand/extrapolate an existing signed distance field into\n"
            "    a mask.\n"
            "__Convert Fog VDB To SDF__:\n"
            "    Converts a scalar Fog volume into a signed distance\n"
            "    grid. Active input voxels with scalar values above\n"
            "    the given isoValue will have NEGATIVE distance\n"
            "    values on output, i.e. they are assumed to be INSIDE\n"
            "    the iso-surface.\n"
            "__Renormalize SDF__:\n"
            "    Given an existing approximate SDF it solves the Eikonal\n"
            "    equation for all its active voxels. Active input voxels\n"
            "    with a signed distance value above the given isoValue\n"
            "    will have POSITIVE distance values on output, i.e. they are\n"
            "    assumed to be OUTSIDE the iso-surface.\n"
            "__Extend Field(s) Off Fog VDB__:\n"
            "     Computes the extension of several attributes off a Fog volume.\n"
            "     The attributes are defined by VDB grids that will be sampled\n"
            "     on the iso-surface of a Fog volume (defined by the __Source Group__).\n"
            "     The attributes are defined by the __Extension Group__ parameter.\n"
            "     This mode only uses the first Fog grid\n"
            "     specified by the __Source Group__ parameter.\n"
            "__Extend Field(s) Off SDF__:\n"
            "     Computes the extension of several attributes off a signed distance field.\n"
            "     The attributes are defined by VDB grids that will be sampled\n"
            "     on the iso-surface of an SDF (defined by the __Source Group__).\n"
            "     The attributes are defined by the __Extension Group__ parameter.\n"
            "     This mode only uses the first SDF grid\n"
            "     specified by the __Source Group__ parameter."));

    // Sweeping Domain Direction
    parms.add(hutil::ParmFactory(PRM_STRING, "sweepdomain", "Domain Direction")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "alldirection", "All Directions",
            "greaterthanisovalue", "Greater Than Isovalue",
            "lessthanisovalue", "Less Than Isovalue",
        })
        .setDefault("alldirection")
        .setTooltip("Pick __Greater Than Isovalue__ or __Less Than Isovalue__\n"
                     "if you want to update voxels corresponding to a signed\n"
                     "distance function/fog value that is greater than or less than\n"
                     "an isovalue, respectively. This option only works for\n"
                     "__Extend Field(s) Off Fog VDB__ and __Extend Field(s) Off SDF__\n"
                     "option.")
        .setDocumentation(
            "The options for sweeping domain direction are:\n"
            "__All Directions__\n"
            "    Perform an update for the extension field(s) in all directions.\n"
            "__Greater Than Isovalue__\n"
            "    Perform an update for the extension field(s) for voxels corresponding.\n"
            "    to a signed distance function/fog that is greater than a given isovalue.\n"
            "__Less Than Isovalue__\n"
            "    Perform an update for the extension field for voxels corresponding.\n"
            "    to a signed distance function/fog that is less than a given isovalue."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "convertorrenormalize", "Convert Fog To SDF or Renormalize SDF")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Only works when __Extend Field(s) of Scalar VDB__ is chosen.\n"
            "If checked, it will either convert a Fog Grid to an SDF or it will renormalize an SDF.")
        .setDocumentation(
            "Use this option if one wants to convert the Fog grid specified by the __Source Group__\n"
            "to be an SDF or to renormalize an SDF."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "sweeps", "Iterations")
         .setDefault(1)
         .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 5)
         .setTooltip(
            "The desired number of iterations of the Fast Sweeping algorithm\n"
            "(one is often enough).")
         .setDocumentation(
            "The number of iterations of the Fast Sweeping algorithm\n"
            "(one is often enough)."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "sdfisovalue", "Sdf Isovalue")
         .setDefault(0.0)
         .setRange(PRM_RANGE_UI, -3, PRM_RANGE_UI, 3)
         .setTooltip("Use this to define an implicit surface from the SDF \n"
             "specified by the __Source Group__. To be used with __Renormalize SDF__.")
         .setDocumentation("Isovalue that defines an implicit surface of an SDF."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "fogisovalue", "Fog Isovalue")
         .setDefault(0.5)
         .setRange(PRM_RANGE_UI, -3, PRM_RANGE_UI, 3)
         .setTooltip("Use this to define an implicit surface from the Fog volume \n"
             "specified by the __Source Group__. To be used with __Convert Fog VDB To SDF__.")
         .setDocumentation("Isovalue that defines an implicit surface of a Fog volume."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "ignoretiles", "Ignore Active Tiles")
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
        .setTooltip("The number of voxels by which to dilate the level set narrow band.\n"
            "Works with __Expand SDF Narrowband__ mode of operation")
        .setDocumentation(
            "Specifies the number of voxels around an SDF narrow-band to be dilated."));

    // Dilation Pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "pattern", "Dilation Pattern")
         .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "NN6",  "Faces",
            "NN18", "Faces and Edges",
            "NN26", "Faces, Edges and Vertices"
         })
         .setDefault("NN6")
         .setTooltip("Select the neighborhood pattern for the dilation operation.")
         .setDocumentation(
             "The neighborhood pattern for the dilation operation\n\n"
             "__Faces__ is fastest. __Faces, Edges and Vertices__ is slowest\n"
             "but can produce the best results for large dilations.\n"
             "__Faces and Edges__ is intermediate in speed and quality.\n"));

    hvdb::OpenVDBOpFactory("VDB Extrapolate", SOP_OpenVDB_Extrapolate::factory, parms, *table)
        .addInput("Source VDB(s)")
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
    UT_String mode, tmpStr;
    bool changed = false;
    evalString(mode, "mode", 0, 0);
    changed |= enableParm("extfields", (mode == "fogext" || mode == "sdfext"));
    changed |= enableParm("mask", mode == "mask");
    changed |= enableParm("dilate", mode == "dilate");
    changed |= enableParm("pattern", mode == "dilate");
    changed |= enableParm("fogisovalue", (mode == "convert" || mode == "fogext")); // not mask & not dilate, but fog
    changed |= enableParm("sdfisovalue", (mode == "renormalize" || mode == "sdfext")); // not mask & not dilate, but sdf
    changed |= enableParm("ignoretiles", mode == "mask");
    changed |= enableParm("convertorrenormalize", (mode == "fogext" || mode == "sdfext"));
    changed |= enableParm("sweepdomain", (mode == "fogext" || mode == "sdfext" || mode == "dilate"));

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

template<typename FSGridT>
bool
SOP_OpenVDB_Extrapolate::Cache::processHelper(
    FastSweepingParms& parms,
    GU_PrimVDB* lsPrim,
    typename FSGridT::ValueType fsIsoValue,
    const GU_PrimVDB* maskPrim)
{
    using namespace openvdb;
    using namespace openvdb::tools;

    if (parms.mNeedExt) {
        // Get the extension primitive
        std::string tmpStr = evalStdString("extfields", parms.mTime);
        parms.mExtGroup = matchGroup(*gdp, tmpStr);
        hvdb::VdbPrimIterator extPrim(gdp, parms.mExtGroup);

        // If we want to extend a field defined by a group but we cannot find a VDB primitive
        // in that group, then throw
        if (!extPrim) {
            std::string msg = "Cannot find the correct VDB primitive named " + tmpStr + ".";
            throw std::runtime_error(msg);
        }

        // Go through the extension fields specified by the extension group
        // and extend each one according to the chosen mode
        for (; extPrim; ++extPrim) {
            // Reset the new extension grid placeholder
            parms.mNewExtGrid.reset();
            extPrim->makeGridUnique();

            openvdb::GridBase::Ptr extGridBase = extPrim->getGridPtr();
            UT_VDBType extType = UTvdbGetGridType(*extGridBase);
            parms.mExtPrimName = extPrim.getPrimitiveNameOrIndex().toStdString();

            // Skip and add message if we are trying to extend the scalar Fast Sweeping Grid.
            if (parms.mExtPrimName == parms.mFSPrimName) {
                std::string msg = "Skipping extending VDB primitive " + parms.mExtPrimName + " off the scalar " +
                    "grid " + parms.mFSPrimName + " because they are the same grid";
                addMessage(SOP_MESSAGE, msg.c_str());
                continue;
            }

            // Add warning if extension grid does not have the same transform as Fast Sweeping grid.
            if (extGridBase->transform() != (lsPrim->getGridPtr())->transform()) {
                std::string msg = "Skipping extending Extension grid " + parms.mExtPrimName + " because it does "
                                  "not have the same transform as Fast Sweeping grid " + parms.mFSPrimName;
                addWarning(SOP_MESSAGE, msg.c_str());
                continue;
            }

            // Call process with the correct template.
            switch (extType) {
                case UT_VDB_FLOAT:
                {
                    openvdb::FloatGrid::Ptr extGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(extGridBase);
                    if (extGrid) {
                        float extBg = static_cast<float>(extGrid->background());
                        process<FSGridT, openvdb::FloatGrid>(parms, lsPrim, fsIsoValue, nullptr /*=maskPrim*/, *extPrim, extBg);
                    } else {
                        std::string msg = "Skipping extending VDB primitive " + parms.mExtPrimName + " because of cast failure.";
                        addWarning(SOP_MESSAGE, msg.c_str());
                    }
                    break;
                } // UT_VDB_FLOAT
                case UT_VDB_DOUBLE:
                {
                    openvdb::DoubleGrid::Ptr extGrid = openvdb::gridPtrCast<openvdb::DoubleGrid>(extGridBase);
                    if (extGrid) {
                        double extBg = static_cast<double>(extGrid->background());
                        process<FSGridT, openvdb::DoubleGrid>(parms, lsPrim, fsIsoValue, nullptr /*=maskPrim*/, *extPrim, extBg);
                    } else {
                        std::string msg = "Skipping extending VDB primitive " + parms.mExtPrimName + " because of cast failure.";
                        addWarning(SOP_MESSAGE, msg.c_str());
                    }
                    break;
                } // UT_VDB_DOUBLE
                case UT_VDB_INT32:
                {
                    openvdb::Int32Grid::Ptr extGrid = openvdb::gridPtrCast<openvdb::Int32Grid>(extGridBase);
                    if (extGrid) {
                        int extBg = static_cast<int>(extGrid->background());
                        process<FSGridT, openvdb::Int32Grid>(parms, lsPrim, fsIsoValue, nullptr /*=maskPrim*/, *extPrim, extBg);
                    } else {
                        std::string msg = "Skipping extending VDB primitive " + parms.mExtPrimName + " because of cast failure.";
                        addWarning(SOP_MESSAGE, msg.c_str());
                    }
                    break;
                } // UT_VDB_INT32
                case UT_VDB_VEC3F:
                {
                    openvdb::Vec3SGrid::Ptr extGrid = openvdb::gridPtrCast<openvdb::Vec3SGrid>(extGridBase);
                    if (extGrid) {
                        openvdb::Vec3f extBg = static_cast<openvdb::Vec3f>(extGrid->background());
                        process<FSGridT, openvdb::Vec3SGrid>(parms, lsPrim, fsIsoValue, nullptr /*=maskPrim*/, *extPrim, extBg);
                    } else {
                        std::string msg = "Skipping extending VDB primitive " + parms.mExtPrimName + " because of cast failure.";
                        addWarning(SOP_MESSAGE, msg.c_str());
                    }
                    break;
                } // UT_VDB_VEC3F
                case UT_VDB_VEC3D:
                {
                    openvdb::Vec3DGrid::Ptr extGrid = openvdb::gridPtrCast<openvdb::Vec3DGrid>(extGridBase);
                    if (extGrid) {
                        openvdb::Vec3d extBg = static_cast<openvdb::Vec3d>(extGrid->background());
                        process<FSGridT, openvdb::Vec3DGrid>(parms, lsPrim, fsIsoValue, nullptr /*=maskPrim*/, *extPrim, extBg);
                    } else {
                        std::string msg = "Skipping extending VDB primitive " + parms.mExtPrimName + " because of cast failure.";
                        addWarning(SOP_MESSAGE, msg.c_str());
                    }
                    break;
                } // UT_VDB_VEC3D
                case UT_VDB_VEC3I:
                {
                    openvdb::Vec3IGrid::Ptr extGrid = openvdb::gridPtrCast<openvdb::Vec3IGrid>(extGridBase);
                    if (extGrid) {
                        openvdb::Vec3i extBg = static_cast<openvdb::Vec3i>(extGrid->background());
                        process<FSGridT, openvdb::Vec3IGrid>(parms, lsPrim, fsIsoValue, nullptr /*=maskPrim*/, *extPrim, extBg);
                    } else {
                        std::string msg = "Skipping extending VDB primitive " + parms.mExtPrimName + " because of cast failure.";
                        addWarning(SOP_MESSAGE, msg.c_str());
                    }
                    break;
                } // UT_VDB_VEC3I
                default:
                {
                    addWarning(SOP_MESSAGE, "Unsupported type of VDB grid chosen for extension");
                    break;
                }
            } // switch

            // Update the Extension grid
            if (parms.mNewExtGrid) extPrim->setGrid(*parms.mNewExtGrid);
        } // vdbprimiterator over extgroup
    } else {
        process<FSGridT>(parms, lsPrim, fsIsoValue, maskPrim);
    }
    return true;
}


////////////////////////////////////////


template<typename FSGridT, typename ExtGridT>
bool
SOP_OpenVDB_Extrapolate::Cache::process(
    FastSweepingParms& parms,
    GU_PrimVDB* lsPrim,
    typename FSGridT::ValueType fsIsoValue,
    const GU_PrimVDB* maskPrim,
    GU_PrimVDB* exPrim,
    const typename ExtGridT::ValueType& background)
{
    using namespace openvdb::tools;

    using SamplerT = openvdb::tools::GridSampler<ExtGridT, openvdb::tools::BoxSampler>;

    typename FSGridT::Ptr fsGrid = openvdb::gridPtrCast<FSGridT>(lsPrim->getGridPtr());

    if (parms.mNeedExt) {
        typename ExtGridT::ConstPtr extGrid = openvdb::gridConstPtrCast<ExtGridT>(exPrim->getConstGridPtr());
        if (!extGrid) {
            auto grid = exPrim->getConstGridPtr();
            std::string msg = "Extension grid (" + grid->getName() + ") cannot be converted " +
                              "to the explicit type specified.";
            throw std::runtime_error(msg);
        }
        SamplerT sampler(*extGrid);
        DirichletSamplerOp<ExtGridT> op(extGrid, sampler);

        if (parms.mMode == "fogext" || parms.mMode == "sdfext") {
            if (!parms.mConvertOrRenormalize) {
                // there are 4 cases:
                if (parms.mMode == "fogext" && (fsGrid->getGridClass() != openvdb::GRID_LEVEL_SET)) {
                    std::string msg = "Extending " + extGrid->getName() + " grid using " + parms.mFSPrimName + " Fog grid.";
                    addMessage(SOP_MESSAGE, msg.c_str());
                    parms.mNewExtGrid = fogToExt(*fsGrid, op, background, fsIsoValue, parms.mNSweeps, parms.mSweepingDomain, extGrid);
                }
                else if (parms.mMode == "fogext" && (fsGrid->getGridClass() == openvdb::GRID_LEVEL_SET)) {
                    std::string msg = "VDB primitive " + parms.mFSPrimName + " is a level set.\n"
                        "You may want to use __Extend Field(s) Off SDF__.";
                    addWarning(SOP_MESSAGE, msg.c_str());
                    return false;
                } else if (parms.mMode == "sdfext" && (fsGrid->getGridClass() == openvdb::GRID_LEVEL_SET)) {
                    std::string msg = "Extending " + extGrid->getName() + " grid using " + parms.mFSPrimName + " SDF grid.";
                    addMessage(SOP_MESSAGE, msg.c_str());
                    parms.mNewExtGrid = sdfToExt(*fsGrid, op, background, fsIsoValue, parms.mNSweeps, parms.mSweepingDomain, extGrid);
                } else {
                    std::string msg = "VDB primitive " + parms.mFSPrimName + " is not a level set.\n"
                        "You may want to use __Extend Field(s) Off Fog VDB__.";
                    addWarning(SOP_MESSAGE, msg.c_str());
                    return false;
                }

                // Update the Extension grid
                if (parms.mNewExtGrid) {
                    parms.mNewExtGrid->insertMeta(*extGrid);
                    parms.mNewExtGrid->setTransform(fsGrid->transform().copy());
                }
            } else {
                std::pair<hvdb::Grid::Ptr, hvdb::Grid::Ptr> outPair;
                // there are 4 cases:
                if (parms.mMode == "fogext" && (fsGrid->getGridClass() != openvdb::GRID_LEVEL_SET)) {
                    std::string msg = "Extending " + extGrid->getName() + " grid using " + parms.mFSPrimName + " Fog grid.";
                    addMessage(SOP_MESSAGE, msg.c_str());
                    outPair = fogToSdfAndExt(*fsGrid, op, background, fsIsoValue, parms.mNSweeps, parms.mSweepingDomain, extGrid);
                }
                else if (parms.mMode == "fogext" && (fsGrid->getGridClass() == openvdb::GRID_LEVEL_SET)) {
                    std::string msg = "VDB primitive " + parms.mFSPrimName + " is a level set.\n"
                        "You may want to use __Extend Field(s) Off SDF__.";
                    addWarning(SOP_MESSAGE, msg.c_str());
                    return false;
                } else if (parms.mMode == "sdfext" && (fsGrid->getGridClass() == openvdb::GRID_LEVEL_SET)) {
                    std::string msg = "Extending " + extGrid->getName() + " grid using " + parms.mFSPrimName + " SDF grid.";
                    addMessage(SOP_MESSAGE, msg.c_str());
                    outPair = sdfToSdfAndExt(*fsGrid, op, background, fsIsoValue, parms.mNSweeps, parms.mSweepingDomain, extGrid);
                } else {
                    std::string msg = "VDB primitive " + parms.mFSPrimName + " is not a level set.\n"
                        "You may want to use __Extend Field(s) Off Fog VDB__.";
                    addWarning(SOP_MESSAGE, msg.c_str());
                    return false;
                }

                // Update both the Fast Sweeping grid and the Extension grid
                if (outPair.first && outPair.second) {
                    outPair.first->setTransform(fsGrid->transform().copy());
                    outPair.first->setGridClass(openvdb::GRID_LEVEL_SET);
                    outPair.second->insertMeta(*extGrid);
                    outPair.second->setTransform(fsGrid->transform().copy());
                    parms.mNewExtGrid = outPair.second;
                    parms.mNewFSGrid = outPair.first;
                }
            }
        }
    } else {
        if (parms.mMode == "dilate") {
            if (fsGrid->getGridClass() != openvdb::GRID_LEVEL_SET) {
                std::string msg = "VDB primitive " + parms.mFSPrimName + " was skipped in dilation because it is not a level set.";
                addMessage(SOP_MESSAGE, msg.c_str());
                return false;
            }

            // no-operation if dilation is < 1
            if (parms.mDilate < 1) {
                std::string msg = "Expand SDF narrow-band with dilate value < 1 results in no-op.";
                addMessage(SOP_MESSAGE, msg.c_str());
                return false;
            }

            const NearestNeighbors nn =
                (parms.mPattern == "NN18") ? NN_FACE_EDGE : ((parms.mPattern == "NN26") ? NN_FACE_EDGE_VERTEX : NN_FACE);
            parms.mNewFSGrid = dilateSdf(*fsGrid, parms.mDilate, nn, parms.mNSweeps, parms.mSweepingDomain);
        } else if (parms.mMode == "convert") {
            if (fsGrid->getGridClass() == openvdb::GRID_LEVEL_SET) {
                std::string msg = "VDB primitive " + parms.mFSPrimName + " was not converted to SDF because it is already a level set.";
                addMessage(SOP_MESSAGE, msg.c_str());
                return false;
            }

            parms.mNewFSGrid = fogToSdf(*fsGrid, fsIsoValue, parms.mNSweeps);
            lsPrim->setVisualization(GEO_VOLUMEVIS_ISO, lsPrim->getVisIso(), lsPrim->getVisDensity());
        } else if (parms.mMode == "renormalize") {
            if (fsGrid->getGridClass() != openvdb::GRID_LEVEL_SET) {
                std::string msg = "VDB primitive " + parms.mFSPrimName + " was not renormalized because it is not a level set.\n"
                    "You may want to convert the FOG VDB into a level set before calling this mode.";
                addMessage(SOP_MESSAGE, msg.c_str());
                return false;
            }

            parms.mNewFSGrid = sdfToSdf(*fsGrid, fsIsoValue, parms.mNSweeps);
        } else if (parms.mMode == "mask") {
            if (fsGrid->getGridClass() != openvdb::GRID_LEVEL_SET) {
                std::string msg = "VDB primitive " + parms.mFSPrimName + " was skipped in mask-operation because it is not a level set.\n"
                    "You may want to convert the FOG VDB into a level set before calling this mode.";
                addMessage(SOP_MESSAGE, msg.c_str());
                return false;
            }

            // Add warning if extension grid does not have the same transform as Fast Sweeping grid.
            if (fsGrid->transform() != (maskPrim->getGridPtr())->transform()) {
                std::string msg = "Mask grid does not have the same transform as Fast Sweeping grid " + parms.mFSPrimName;
                addWarning(SOP_MESSAGE, msg.c_str());
                return false;
            }

            FastSweepingMaskOp<FSGridT> op(parms, fsGrid);
            // calling openvdb::tools::maskSdf with mask grid.
            hvdb::GEOvdbApply<hvdb::AllGridTypes>(*maskPrim, op);
            parms.mNewFSGrid = op.mOutGrid;
        }

        // Update the fast sweeping grid
        if (parms.mNewFSGrid) {
            parms.mNewFSGrid->setGridClass(openvdb::GRID_LEVEL_SET);
        }
    } // !parms.mNeedExt

    return true;
}


OP_ERROR
SOP_OpenVDB_Extrapolate::Cache::evalFastSweepingParms(OP_Context& context, FastSweepingParms& parms)
{
    const fpreal time = context.getTime();
    parms.mTime = context.getTime();

    // Get the group of level sets to process
    parms.mFSGroup = matchGroup(*gdp, evalStdString("group", time));

    evalString(parms.mMode, "mode", 0, time);

    parms.mNeedExt = (parms.mMode == "fogext" || parms.mMode == "sdfext");
    parms.mNSweeps = static_cast<int>(evalInt("sweeps", 0, time));
    parms.mIgnoreTiles = static_cast<bool>(evalInt("ignoretiles", 0, time));
    parms.mConvertOrRenormalize = static_cast<bool>(evalInt("convertorrenormalize", 0, time));

    // For dilate
    evalString(parms.mPattern, "pattern", 0, time);
    parms.mDilate = static_cast<int>(evalInt("dilate", 0, time));

    UT_String sweepDomain;
    evalString(sweepDomain, "sweepdomain", 0, time);
    if (sweepDomain == "alldirection")
        parms.mSweepingDomain = openvdb::tools::FastSweepingDomain::SWEEP_ALL;
    else if (sweepDomain == "greaterthanisovalue")
        parms.mSweepingDomain = openvdb::tools::FastSweepingDomain::SWEEP_GREATER_THAN_ISOVALUE;
    else if (sweepDomain == "lessthanisovalue")
        parms.mSweepingDomain = openvdb::tools::FastSweepingDomain::SWEEP_LESS_THAN_ISOVALUE;

    return error();
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Extrapolate::Cache::cookVDBSop(OP_Context& context)
{
    try {
        // Evaluate UI parameters
        FastSweepingParms parms;
        if (evalFastSweepingParms(context, parms) >= UT_ERROR_ABORT) return error();

        // Get the mask primitive if the mode is mask
        const fpreal time = context.getTime();
        const GU_Detail* maskGeo = inputGeo(1);
        const GU_PrimVDB* maskPrim = nullptr;
        hvdb::GridCPtr maskGrid = nullptr;
        if (parms.mMode == "mask") {// selected to use a mask
            if (maskGeo) {// second input exists
                const GA_PrimitiveGroup* maskGroup = parsePrimitiveGroups(
                    evalStdString("mask", time).c_str(), GroupCreator(maskGeo));

                hvdb::VdbPrimCIterator maskIt(maskGeo, maskGroup);
                maskPrim = *maskIt;
                if (!maskPrim) {
                     addError(SOP_MESSAGE, "Mask Geometry not found.\n"
                         "Please provide a mask VDB as a second input.");
                     return error();
                }
                if (maskIt) maskGrid = maskIt->getConstGridPtr();// only use the first grid

                if (++maskIt) {
                    addMessage(SOP_MESSAGE, "Multiple Mask grids were found.\n"
                       "Using the first one for reference.");
                }
            } else {
              addError(SOP_MESSAGE, "Mask Geometry not found.\n"
                  "Please provide a mask VDB as a second input");
              return error();
            }
        }

        UT_AutoInterrupt progress("Performing Fast Sweeping");

        // The behavior of this SOP depends on whether there is a extension field to be extended.
        // When there is no field to be extended, go through the VDB primitives as specified by the
        // Source Group and update the scalar VDB primitives. If there is a field to be extended,
        // only use the first scalar group in the Source Group as the underlying Fast Sweeping grid.
        // Note that only UT_VDB_FLOAT and UT_VDB_DOUBLE are supported as scalar (floating-point)
        // Fast Sweeping grids.
        for (hvdb::VdbPrimIterator it(gdp, parms.mFSGroup); it; ++it) {
            // Reset the new fast sweeping grid placeholder
            parms.mNewFSGrid.reset();
            it->makeGridUnique();

            if (progress.wasInterrupted()) {
                throw std::runtime_error("Processing was interrupted");
            }
            hvdb::Grid& inGrid = it->getGrid();
            UT_VDBType inType = UTvdbGetGridType(inGrid);
            parms.mFSPrimName = it.getPrimitiveNameOrIndex().toStdString();

            switch (inType) {
                case UT_VDB_FLOAT:
                {
                    float isoValue = (parms.mMode == "convert" || parms.mMode == "fogext") ?
                                         static_cast<float>(evalFloat("fogisovalue", 0, time)) :
                                         (parms.mMode == "renormalize" || parms.mMode == "sdfext") ?
                                         static_cast<float>(evalFloat("sdfisovalue", 0, time)) : 0.f;
                    processHelper<openvdb::FloatGrid>(parms, *it /*lsPrim*/, isoValue, maskPrim);
                    parms.mExtFieldProcessed = true;
                    break;
                }
                case UT_VDB_DOUBLE:
                {
                    double isoValue = (parms.mMode == "convert" || parms.mMode == "fogext") ?
                                        static_cast<double>(evalFloat("fogisovalue", 0, time)) :
                                        (parms.mMode == "renormalize" || parms.mMode == "sdfext") ?
                                        static_cast<double>(evalFloat("sdfisovalue", 0, time)) : 0.0;
                    processHelper<openvdb::DoubleGrid>(parms, *it /*lsPrim*/, isoValue, maskPrim);
                    parms.mExtFieldProcessed = true;
                    break;
                }
                default:
                {
                    std::string msg = "VDB primitive " + parms.mFSPrimName + " was skipped to be treated as a source group because it is not a floating-point grid.";
                    addMessage(SOP_MESSAGE, msg.c_str());
                    break;
                }
            }

            // Update the fast sweeping grid
            if (parms.mNewFSGrid) it->setGrid(*parms.mNewFSGrid);

            // If we need extension, we only process the first grid
            if (parms.mNeedExt && parms.mExtFieldProcessed) {
                break;
            }
        }
    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}
