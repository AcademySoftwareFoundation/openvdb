// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file SOP_OpenVDB_Visualize.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Visualize VDB grids and their tree topology

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/GeometryUtil.h> // hvdb::drawFrustum
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/NodeVisitor.h> // visitNodesDepthFirst

#include <GA/GA_Handle.h>
#include <GA/GA_Types.h>
#include <GEO/GEO_PolyCounts.h>
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <PRM/PRM_Parm.h>
#include <UT/UT_Ramp.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_VectorTypes.h> // for UT_Vector3i

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


namespace {

template <typename T>
struct IsGridTypeIntegral
    : std::conditional_t<   std::is_integral<T>::value
                         || std::is_same<T,openvdb::PointIndex32>::value
                         || std::is_same<T,openvdb::PointIndex64>::value
                         || std::is_same<T,openvdb::PointDataIndex32>::value
                         || std::is_same<T,openvdb::PointDataIndex64>::value
                         , std::true_type
                         , std::false_type>
{
};

template <typename T>
struct IsGridTypeArithmetic
    : std::conditional_t<   IsGridTypeIntegral<T>::value
                         || std::is_floating_point<T>::value
                         , std::true_type
                         , std::false_type>
{
};

} // anonymous namespace

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


enum RenderStyle { STYLE_NONE = 0, STYLE_POINTS, STYLE_WIRE_BOX, STYLE_SOLID_BOX };
enum SliceStyle { NO_SLICE = 0, SLICE_VOXEL_FLATTEN, SLICE_VOXEL, SLICE_LEAF };
enum SlicePlane { SLICE_XY = 0, SLICE_YZ, SLICE_ZX };


class SOP_OpenVDB_Visualize: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Visualize(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Visualize() override = default;

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i) const override { return (i == 1); }

    int storeOffsets();

    static int storeOffsetsCB(void* data, int /*idx*/, float /*time*/, const PRM_Template*)
    {
        if (auto* sop = static_cast<SOP_OpenVDB_Visualize*>(data)) {
            return sop->storeOffsets();
        }
        return 0;
    }

    static UT_Vector3 colorLevel(int level) { return sColors[std::max(3-level,0)]; }
    static const UT_Vector3& colorSign(bool negative) { return sColors[negative ? 5 : 4]; }

    class Cache: public SOP_VDBCacheOptions
    {
    public:
        double offset() const { return mOffset; }
        double offsetWS() const { return mOffsetWS; }

    protected:
        OP_ERROR cookVDBSop(OP_Context&) override;

    private:
        // cache offsets for use when switching between index-space and world-space
        double mOffset = 0.0;
        double mOffsetWS = 0.0;
    };

protected:
    bool updateParmsFlags() override;
    void resolveObsoleteParms(PRM_ParmList*) override;

private:
    static const UT_Vector3 sColors[];
};


// Same color scheme as the VDB TOG paper.
const UT_Vector3 SOP_OpenVDB_Visualize::sColors[] = {
    UT_Vector3(0.045f, 0.045f, 0.045f),         // 0. Root
    UT_Vector3(0.0432f, 0.33f, 0.0411023f),     // 1. First internal node level
    UT_Vector3(0.871f, 0.394f, 0.01916f),       // 2. Intermediate internal node levels
    UT_Vector3(0.00608299f, 0.279541f, 0.625f), // 3. Leaf level
    UT_Vector3(0.523f, 0.0325175f, 0.0325175f), // 4. Value >= ZeroVal (for voxels or tiles)
    UT_Vector3(0.92f, 0.92f, 0.92f)             // 5. Value < ZeroVal (for voxels or tiles)
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
        .setTooltip("Specify a subset of the input VDBs to be processed.")
        .setDocumentation(
            "The VDBs to be visualized (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "addcolor", "Color")
        .setDefault(PRMoneDefaults)
        .setTooltip("Specify whether to draw in color.")
        .setDocumentation(
            "Specify whether to generate geometry with the `Cd` color attribute."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "previewfrustum", "Frustum")
        .setTooltip(
            "Specify whether to draw the camera frustums\n"
            "of VDBs with frustum transforms.")
        .setDocumentation(
            "For VDBs with [frustum transforms|https://academysoftwarefoundation.github.io/openvdb/"
            "transformsAndMaps.html#sFrustumTransforms],"
            " generate geometry representing the frustum bounding box."));

    char const * const boxItems[] = {
        "wirebox",  "Wireframe Boxes",
        "box",      "Solid Boxes",
        nullptr
    };
    char const * const pointAndBoxItems[] = {
        "points",   "Points",
        "wirebox",  "Wireframe Boxes",
        "box",      "Solid Boxes",
        nullptr
    };

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "drawleafnodes", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));
    parms.add(hutil::ParmFactory(PRM_STRING, "leafstyle", "Leaf Nodes")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, pointAndBoxItems)
        .setDefault("wirebox")
        .setDocumentation(
            "Specify whether to render the leaf nodes of VDB trees"
            " as wireframe boxes, as solid boxes, or as a single point"
            " in the middle of each node.\n\n"
            "If __Color__ is enabled, leaf nodes will be blue."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "drawinternalnodes", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));
    parms.add(hutil::ParmFactory(PRM_STRING, "internalstyle", "Internal Nodes")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, boxItems)
        .setDefault("wirebox")
        .setDocumentation(
            "Specify whether to render the internal nodes of VDB trees"
            " as wireframe boxes or as solid boxes.\n\n"
            "If __Color__ is enabled, the lowest-level internal nodes will be green"
            " and higher-level internal nodes will be orange."));

    // Active tiles
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "drawtiles", "")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    parms.add(hutil::ParmFactory(PRM_STRING, "tilestyle", "Active Tiles")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, pointAndBoxItems)
        .setDefault("wirebox")
        .setDocumentation(
            "Specify whether to render the active tiles of VDB trees"
            " as wireframe boxes, as solid boxes, or as a single point"
            " in the middle of each tile.\n\n"
            "If __Color__ is enabled, negative-valued tiles will be white"
            " and nonnegative tiles will be red."));

    // Active voxels
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "drawvoxels", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    parms.add(hutil::ParmFactory(PRM_ORD, "voxelstyle", "Active Voxels")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, pointAndBoxItems)
        .setDefault("points")
        .setDocumentation(
            "Specify whether to render the active voxels of VDB trees"
            " as wireframe boxes, as solid boxes, or as a single point"
            " in the middle of each voxel.\n\n"
            "If __Color__ is enabled, negative-valued voxels will be white"
            " and nonnegative voxels will be red.\n\n"
            "WARNING:\n"
            "    Rendering active voxels as boxes can generate large amounts of geometry."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "ignorestaggered", "Ignore Staggered Vectors")
        .setTooltip("Draw staggered vectors as if they were collocated."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "addindexcoord", "Points with Index Coordinates")
        .setTooltip("Add a voxel/tile index coordinate attribute to points.")
        .setDocumentation(
            "For voxels, tiles, and leaf nodes rendered as points, add an attribute to"
            " the points that gives the coordinates of the points in the VDB's [index space|"
            "https://academysoftwarefoundation.github.io/openvdb/overview.html#secSpaceAndTrans]."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "addvalue", "Points with Values")
        .setTooltip("Add a voxel/tile value attribute to points.")
        .setDocumentation(
            "For voxels and tiles rendered as points, add an attribute to the points"
            " that gives the voxel and tile values."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "usegridname", "Name Point Attributes After VDBs")
        .setTooltip(
            "If enabled, use the VDB name as the attribute name when\n"
            "displaying points with values.\n"
            "If disabled or if a VDB has no name, use either \"vdb_int\",\n"
            "\"vdb_float\" or \"vdb_vec3f\" as the attribute name.")
        .setDocumentation(
            "If enabled, name the attribute added by __Points with Values__ after"
            " the VDB primitive.  If disabled or if a VDB has no name, name the point"
            " attribute according to its type: `vdb_int`, `vdb_float`, `vdb_vec3f`, etc."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "slice", "")
        .setDefault(PRMzeroDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    {   // Slice by leaf or voxel
        char const * const items[] = {
            "voxelflatten",  "By Voxel (Flatten)",
            "voxel",  "By Voxel",
            "leaf",  "By Leaf",
            nullptr
        };
        parms.add(hutil::ParmFactory(PRM_STRING, "slicemethod", "Slice")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault("voxelflatten")
            .setDocumentation(
                "Slice by voxel renders only geometry that intersects with the slice "
                "(and optionally flattens this geometry to the slice plane). "
                "Slice by leaf can render all voxels in the leaf that intersects with the slice. "));
    }

    {   // Slice plane
        char const * const items[] = {
            "xy",  "XY Plane",
            "yz",  "YZ Plane",
            "zx",  "ZX Plane",
            nullptr
        };
        parms.add(hutil::ParmFactory(PRM_STRING, "plane", "Plane")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault("xy")
            .setDocumentation(
                "What axis to extract from the VDB. This is local to the VDB's space. "
                "Rotated or transformed VDBs will have a corresponding rotated or transformed plane."));
    }

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "useworldspace", "Use World Space for Offset")
        .setCallbackFunc(&SOP_OpenVDB_Visualize::storeOffsetsCB)
        .setDefault(PRMzeroDefaults)
        .setDocumentation(
            "Sets whether offsets are specified in Houdini units. "
            "When this option is off, the offset is relative to the VDB in the range -1..1. "
            "After first cook of this SOP, the offset of the first VDB will be converted."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "relativeoffset", "Relative Offset")
        .setRange(PRM_RANGE_UI, -1.0, PRM_RANGE_UI, 1.0)
        .setDocumentation(
            "Where the plane should be positioned inside the VDB. "
            "This is a relative coordinate with -1..1 being the total range, "
            "so 0 means the center of the VDB."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "offset", "Offset")
        .setRange(PRM_RANGE_FREE, -10.0, PRM_RANGE_FREE, 10.0)
        .setDocumentation(
            "Where the plane should be positioned inside the VDB in Houdini units."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "visualize", "")
        .setDefault(PRMzeroDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    // infra-red values with controls disabled by default

    PRM_SpareData visRampSpare(PRM_SpareArgs()
        << PRM_SpareToken("rampshowcontrolsdefault", "0")
        << PRM_SpareToken("rampcolordefault",
            "1pos ( 0 ) 1c ( 0.2 0 1 ) 1interp ( linear ) "
            "2pos ( 0.25 ) 2c ( 0 0.85 1 ) 2interp ( linear ) "
            "3pos ( 0.5 ) 3c ( 0 1 0.1 ) 3interp ( linear ) "
            "4pos ( 0.75 ) 4c ( 0.95 1 0 ) 4interp ( linear ) "
            "5pos ( 1 ) 5c ( 1 0 0 ) 5interp ( linear )"));

    parms.add(hutil::ParmFactory(PRM_MULTITYPE_RAMP_RGB, "visramp", "Visualization Ramp")
        .setDefault(PRMfiveDefaults)
        .setSpareData(&visRampSpare)
        .setHelpText("Color ramp to visualize the attribute values."));

    std::vector<fpreal> visDefaults{0, 1};
    parms.add(hutil::ParmFactory(PRM_FLT_E, "visrange", "Visualization Range")
        .setVectorSize(2)
        .setDefault(visDefaults)
        .setDocumentation(
            "The VDB values that correspond to the beginning and end of the visualization ramp."));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.beginSwitcher("tabMenu");
    obsoleteParms.addFolder("Tree Topology");
    obsoleteParms.endSwitcher();
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING,"treeTopology", "Tree Topology"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING,"renderOptions", "Render Options"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "extractMesh", "Extract Mesh"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "computeNormals", "Compute Point Normals"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "reverse", "Reverse Faces"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "optionsHeading", "Options"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "Other", "Other"));
    {
        char const * const items[] = {
            "none",     "Disabled",
            "opevdb",   "OpenVDB Mesher", // note the misspelling
            "houdini",  "Houdini Surfacer",
            nullptr
        };
        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "meshing", "Meshing")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    {
        char const * const items[] = {
            "none",     "Disabled",
            "leaf",     "Leaf Nodes and Active Tiles",
            "nonconst", "Leaf and Internal Nodes",
            nullptr
        };
        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "nodes", "Tree Nodes")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    {
        char const * const items[] = {
            "none",     "Disabled",
            "points",   "Points",
            "pvalue",   "Points with Values",
            "wirebox",  "Wireframe Box",
            "box",      "Solid Box",
            nullptr
        };
        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "tiles", "Active Constant Tiles")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "voxels", "Active Voxels")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "previewFrustum", "Frustum"));
    obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "leafmode", "Leaf Nodes")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, boxItems));
    obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "internalmode", "Internal Nodes")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, boxItems));
    obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "tilemode", "Active Tiles")
        .setDefault(PRMoneDefaults)
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, pointAndBoxItems));
    obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "voxelmode", "Active Voxels")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, pointAndBoxItems));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "previewroi", ""));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "surfacing", "Surfacing"));
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "drawsurface", ""));
    {   // Meshing scheme
        char const * const items[] = {
            "openvdb",  "OpenVDB Mesher",
            "houdini",  "Houdini Surfacer",
            nullptr
        };
        obsoleteParms.add(hutil::ParmFactory(PRM_ORD, "mesher", "Mesher")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));
    }
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "adaptivity", "Adaptivity"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT_J, "isoValue", "Isovalue"));
    obsoleteParms.add(hutil::ParmFactory(PRM_RGB_J, "surfaceColor", "Surface Color")
        .setVectorSize(3));

    // Register this operator.
    hvdb::OpenVDBOpFactory("VDB Visualize Tree",
        SOP_OpenVDB_Visualize::factory, parms, *table)
#ifndef SESI_OPENVDB
        .setInternalName("DW_OpenVDBVisualize")
#endif
        .setObsoleteParms(obsoleteParms)
        .addInput("Input with VDBs to visualize")
        .setVerb(SOP_NodeVerb::COOK_GENERATOR, []() { return new SOP_OpenVDB_Visualize::Cache; })
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Generate geometry to visualize the internal\n\
[tree structure|https://academysoftwarefoundation.github.io/openvdb/overview.html#secTree]\n\
of a VDB volume.\"\"\"\n\
\n\
@overview\n\
\n\
This node can be a useful troubleshooting tool.\n\
Among other things, it allows one to evaluate the\n\
[sparseness|https://academysoftwarefoundation.github.io/openvdb/overview.html#secSparsity]\n\
of VDB volumes as well as to examine their extents and the values of individual voxels.\n\
\n\
@related\n\
- [Node:sop/vdbvisualizetree]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Visualize::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Visualize(net, name, op);
}


SOP_OpenVDB_Visualize::SOP_OpenVDB_Visualize(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


void
SOP_OpenVDB_Visualize::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    const fpreal time = 0.0;

    // The "nodes", "tiles" and "voxels" menus all included "Disabled" options,
    // which are now handled with toggles.  The old scheme also had two conflicting
    // ways to enable display of tiles, which the following attempts to reconcile.

    const UT_String pointStr("points"), boxStr("box"), wireStr("wirebox");

    PRM_Parm* parm = obsoleteParms->getParmPtr("nodes");
    if (parm && !parm->isFactoryDefault()) {
        // 0: disabled, 1: leaf nodes and active tiles, 2: leaf and internal nodes
        const int mode = obsoleteParms->evalInt("nodes", 0, time);
        // Enable leaf nodes if the old mode was not "Disabled".
        setInt("drawleafnodes", 0, time, mode > 0);
        // Set the leaf style to wire box, but only if leaf nodes are displayed.
        if (mode > 0) setString(wireStr, CH_STRING_LITERAL, "leafstyle", 0, time);
        // Enable internal nodes if the old mode was "Leaf and Internal Nodes".
        setInt("drawinternalnodes", 0, time, mode == 2);
        // Set the internal node style to wire box, but only if internal nodes are displayed.
        if (mode == 2) {
            setString(wireStr, CH_STRING_LITERAL, "internalstyle", 0, time);
        }
        // Disable tiles if the old mode was not "Leaf Nodes and Active Tiles".
        setInt("drawtiles", 0, time, mode == 1);
        if (mode == 1) {
            // Display tiles as wire boxes if the old mode was "Leaf Nodes and Active Tiles".
            // (This setting took precedence over the tile mode, below.)
            setString(wireStr, CH_STRING_LITERAL, "tilestyle", 0, time);
        }
    }

    parm = obsoleteParms->getParmPtr("tiles");
    if (parm && !parm->isFactoryDefault()) {
        // 0: disabled, 1: points, 2: points with values, 3: wire boxes, 4: solid boxes
        const int mode = obsoleteParms->evalInt("tiles", 0, time);
        if (mode > 0) setInt("drawtiles", 0, time, true);
        switch (mode) {
            case 1:
                setString(pointStr, CH_STRING_LITERAL, "tilestyle", 0, time);
                setInt("addvalue", 0, time, false);
                break;
            case 2:
                setString(pointStr, CH_STRING_LITERAL, "tilestyle", 0, time);
                setInt("addvalue", 0, time, true);
                break;
            case 3: setString(wireStr, CH_STRING_LITERAL, "tilestyle", 0, time); break;
            case 4: setString(boxStr, CH_STRING_LITERAL, "tilestyle", 0, time); break;
        }
    }

    parm = obsoleteParms->getParmPtr("voxels");
    if (parm && !parm->isFactoryDefault()) {
        // 0: disabled, 1: points, 2: points with values, 3: wire boxes, 4: solid boxes
        const int mode = obsoleteParms->evalInt("voxels", 0, time);
        setInt("drawvoxels", 0, time, mode > 0);
        switch (mode) {
            case 1:
                setString(pointStr, CH_STRING_LITERAL, "voxelstyle", 0, time);
                setInt("addvalue", 0, time, false);
                break;
            case 2:
                setString(pointStr, CH_STRING_LITERAL, "voxelstyle", 0, time);
                setInt("addvalue", 0, time, true);
                break;
            case 3: setString(wireStr, CH_STRING_LITERAL, "voxelstyle", 0, time); break;
            case 4: setString(boxStr, CH_STRING_LITERAL, "voxelstyle", 0, time); break;
        }
    }

    for (const auto* name: {"leaf", "internal"}) {
        const auto oldName = std::string(name) + "mode";
        const auto newName = std::string(name) + "style";
        parm = obsoleteParms->getParmPtr(oldName.c_str());
        if (parm && !parm->isFactoryDefault()) {
            const int mode = obsoleteParms->evalInt(oldName.c_str(), 0, time);
            setString(mode == 0 ? wireStr : boxStr, CH_STRING_LITERAL, newName.c_str(), 0, time);
        }
    }

    for (const auto* name: {"tile", "voxel"}) {
        const auto oldName = std::string(name) + "mode";
        const auto newName = std::string(name) + "style";
        parm = obsoleteParms->getParmPtr(oldName.c_str());
        if (parm && !parm->isFactoryDefault()) {
            switch (obsoleteParms->evalInt(oldName.c_str(), 0, time)) {
                case 0: setString(pointStr, CH_STRING_LITERAL, newName.c_str(), 0, time); break;
                case 1: setString(wireStr, CH_STRING_LITERAL, newName.c_str(), 0, time); break;
                case 2: setString(boxStr, CH_STRING_LITERAL, newName.c_str(), 0, time); break;
            }
        }
    }

    resolveRenamedParm(*obsoleteParms, "previewFrustum", "previewfrustum");

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


// Update UI parm display
bool
SOP_OpenVDB_Visualize::updateParmsFlags()
{
    bool changed = false;

    const fpreal time = 0.0;

    const std::string
        leafMode = evalStdString("leafstyle", time),
        tileMode = evalStdString("tilestyle", time),
        voxelMode = evalStdString("voxelstyle", time);
    const bool
        drawLeafNodes = bool(evalInt("drawleafnodes", 0, time)),
        drawVoxels = bool(evalInt("drawvoxels", 0, time)),
        drawTiles = bool(evalInt("drawtiles", 0, time)),
        drawPoints = (drawTiles && tileMode == "points") || (drawVoxels && voxelMode == "points");

    changed |= enableParm("leafstyle", drawLeafNodes);
    changed |= enableParm("internalstyle", bool(evalInt("drawinternalnodes", 0, time)));
    changed |= enableParm("tilestyle", drawTiles);
    changed |= enableParm("voxelstyle", drawVoxels);
    changed |= enableParm("ignorestaggered", drawVoxels);
    changed |= enableParm("addvalue", drawPoints);
    changed |= enableParm("addindexcoord", drawPoints || (drawLeafNodes && leafMode == "points"));
    changed |= enableParm("usegridname", drawPoints);

    const bool volumeSlice = bool(evalInt("slice", 0, time));

    changed |= enableParm("slicemethod", volumeSlice);
    changed |= enableParm("plane", volumeSlice);
    changed |= enableParm("useworldspace", volumeSlice);

    const bool worldSpaceUnits = bool(evalInt("useworldspace", 0, time));

    changed |= enableParm("relativeoffset", volumeSlice);
    changed |= setVisibleState("relativeoffset", !worldSpaceUnits);
    changed |= enableParm("offset", volumeSlice);
    changed |= setVisibleState("offset", worldSpaceUnits);

    const bool addcolor = bool(evalInt("addcolor", 0, time));
    const bool visualize = bool(evalInt("visualize", 0, time));

    changed |= enableParm("visualize", addcolor && (drawTiles || drawVoxels));
    changed |= enableParm("visramp", addcolor && visualize && (drawTiles || drawVoxels));
    changed |= enableParm("visrange", addcolor && visualize && (drawTiles || drawVoxels));

    return changed;
}


// Callback to convert from voxel to world-space units
int
SOP_OpenVDB_Visualize::storeOffsets()
{
    const fpreal time = CHgetEvalTime();

    // Attempt to extract the offsets from our cache.
    if (const auto* cache = dynamic_cast<SOP_OpenVDB_Visualize::Cache*>(myNodeVerbCache)) {

        if (bool(evalInt("useworldspace", 0, time))) {
            setFloat("offset", 0, time, cache->offsetWS());
        } else {
            setFloat("relativeoffset", 0, time, cache->offset());
        }
    }

    return 1;
}


////////////////////////////////////////


template<typename OpType>
inline RenderStyle
evalRenderStyle(OpType& op, const char* toggleName, const char* modeName, fpreal time)
{
    RenderStyle style = STYLE_NONE;
    if (op.evalInt(toggleName, 0, time)) {
        const std::string mode = op.evalStdString(modeName, time);
        if (mode == "points") {
            style = STYLE_POINTS;
        } else if (mode == "wirebox") {
            style = STYLE_WIRE_BOX;
        } else if (mode == "box") {
            style = STYLE_SOLID_BOX;
        }
    }
    return style;
}


////////////////////////////////////////


struct TreeParms
{
    RenderStyle internalStyle = STYLE_NONE;
    RenderStyle tileStyle = STYLE_NONE;
    RenderStyle leafStyle = STYLE_NONE;
    RenderStyle voxelStyle = STYLE_NONE;
    bool addColor = true;
    bool ignoreStaggeredVectors = false;
    bool addValue = false;
    bool addIndexCoord = false;
    bool useGridName = false;
    SliceStyle sliceStyle = NO_SLICE;
    SlicePlane slicePlane = SLICE_XY;
    bool useWorldSpace = false;
    double sliceOffset = 0;
    bool visualize = false;
    UT_Ramp colorRamp;
    double colorMin = 0.0f;
    double colorRange = 1.0f;
    double* cachedOffset = nullptr;
    double* cachedOffsetWS = nullptr;
};


class TreeVisualizer
{
public:
    TreeVisualizer(GU_Detail&, const TreeParms&, openvdb::util::NullInterrupter* = nullptr);

    // create all the point attributes
    template<typename GridType>
    void createPointAttributes(const GridType&);

    // compute the index-space value corresponding to the desired slice
    template<typename GridType>
    openvdb::Int32 computeIndexSlice(const GridType&);

    // allocate the point, polygon and vertex offset arrays
    template<typename GridType>
    void allocateOffsetArrays(const GridType&);

    // render nodes, tiles or voxels
    template<typename GridType>
    void render(bool node, const GridType& grid, openvdb::Int32 sliceIndex,
        const RenderStyle& style1, const RenderStyle& style2);

    template<typename GridType>
    void operator()(const GridType&);

private:
    // count the number of points, polygons and vertices per node
    template <typename TreeT>
    struct CountOp;

    // turn per-node counts into cumulative offsets
    struct ComputeOffsetsOp;

    // render points and write point attributes
    template <typename TreeT>
    struct RenderPointsOp;

    // render vertices for wireframe or solid boxes
    template <typename TreeT>
    struct RenderVerticesOp;

    // render wireframe or solid boxes sequentially
    template <typename TreeT>
    struct RenderGeometrySingleThreadedOp;

    TreeParms mParms;
    std::vector<std::unique_ptr<size_t[]>> mPointOffsets;
    std::vector<std::unique_ptr<size_t[]>> mVertexOffsets;
    std::vector<std::unique_ptr<size_t[]>> mPolygonOffsets;
    GU_Detail* mGeo;
    openvdb::util::NullInterrupter* mInterrupter;
    const openvdb::math::Transform* mXform;
    GA_RWHandleF  mFloatHandle;
    GA_RWHandleI  mInt32Handle;
    GA_RWHandleV3 mVec3fHandle;
    GA_RWHandleV3 mCdHandle;
    GA_RWHandleT<UT_Vector3i> mIndexCoordHandle;
};


////////////////////////////////////////


// return true if slice index intersects with the coord bbox with the given slice plane
inline
bool isCoordBBoxValid(const openvdb::CoordBBox& bbox, const SlicePlane& slicePlane, openvdb::Int32 sliceIndex)
{
    openvdb::Int32 start(0);
    openvdb::Int32 end(0);

    if (slicePlane == SLICE_XY) {
        start = bbox.min().z();
        end = bbox.max().z();
    } else if (slicePlane == SLICE_YZ) {
        start = bbox.min().x();
        end = bbox.max().x();
    } else if (slicePlane == SLICE_ZX) {
        start = bbox.min().y();
        end = bbox.max().y();
    }

    return (sliceIndex >= start) && (sliceIndex <= end);
}


// return true if slice index intersects with the node with the given slice plane
template <typename NodeT>
bool isNodeValid(const NodeT& node, const SlicePlane& slicePlane, openvdb::Int32 sliceIndex)
{
    return isCoordBBoxValid(node.getNodeBoundingBox(), slicePlane, sliceIndex);
}


// return true if slice index intersects with the iterator with the given slice plane
// note that for performance reasons, this method does not do leaf slice intersection testing
template <typename IterT>
bool isIterValid(const IterT& iter, const SlicePlane& slicePlane, openvdb::Int32 sliceIndex)
{
    const auto& parent = iter.parent();

    if (parent.getLevel() == 0) {
        // voxels - direct coord comparison
        if (slicePlane == SLICE_XY) {
            return iter.getCoord().z() == sliceIndex;
        } else if (slicePlane == SLICE_YZ) {
            return iter.getCoord().x() == sliceIndex;
        } else if (slicePlane == SLICE_ZX) {
            return iter.getCoord().y() == sliceIndex;
        }
    } else {
        // tiles - bbox comparison

        // compute tile bounding box
        const openvdb::Coord origin = iter.getCoord();
        const openvdb::Index dim = parent.getChildDim();
        const openvdb::CoordBBox bbox = openvdb::CoordBBox::createCube(origin, dim);

        return isCoordBBoxValid(bbox, slicePlane, sliceIndex);
    }

    return false;
}


TreeVisualizer::TreeVisualizer(GU_Detail& geo, const TreeParms& parms,
    openvdb::util::NullInterrupter* interrupter)
    : mParms(parms)
    , mGeo(&geo)
    , mInterrupter(interrupter)
    , mXform(nullptr)
{
}


template <typename TreeT>
struct TreeVisualizer::CountOp
{
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    CountOp(bool node, openvdb::Int32 sliceIndex, bool staggered, TreeVisualizer& parent)
        : mNode(node)
        , mSliceIndex(sliceIndex)
        , mStaggered(staggered)
        , mParent(parent) { }

    template<typename NodeT>
    void countNodes(const NodeT& node, size_t idx, const RenderStyle& style) const
    {
        const bool valid = mParent.mParms.sliceStyle == NO_SLICE ||
            isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex);

        if (valid && style == STYLE_SOLID_BOX) {
            if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                mParent.mPointOffsets[NodeT::LEVEL][idx] = 4;
                mParent.mVertexOffsets[NodeT::LEVEL][idx] = 4;
                mParent.mPolygonOffsets[NodeT::LEVEL][idx] = 1;
            } else {
                mParent.mPointOffsets[NodeT::LEVEL][idx] = 8;
                mParent.mVertexOffsets[NodeT::LEVEL][idx] = 4*6; // each polygon has 4 vertices
                mParent.mPolygonOffsets[NodeT::LEVEL][idx] = 6;
            }
        } else if (valid && style == STYLE_WIRE_BOX) {
            if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                mParent.mPointOffsets[NodeT::LEVEL][idx] = 4;
                mParent.mVertexOffsets[NodeT::LEVEL][idx] = 5;
                mParent.mPolygonOffsets[NodeT::LEVEL][idx] = 1;
            } else {
                mParent.mPointOffsets[NodeT::LEVEL][idx] = 8;
                mParent.mVertexOffsets[NodeT::LEVEL][idx] = 16;
                mParent.mPolygonOffsets[NodeT::LEVEL][idx] = 1;
            }
        } else if (valid && style == STYLE_POINTS) {
            mParent.mPointOffsets[NodeT::LEVEL][idx] = 1;
            mParent.mVertexOffsets[NodeT::LEVEL][idx] = 0;
            mParent.mPolygonOffsets[NodeT::LEVEL][idx] = 0;
        } else {
            mParent.mPointOffsets[NodeT::LEVEL][idx] = 0;
            mParent.mVertexOffsets[NodeT::LEVEL][idx] = 0;
            mParent.mPolygonOffsets[NodeT::LEVEL][idx] = 0;
        }
    }

    void countNodes(const RootT& root, size_t idx, const RenderStyle& style) const
    {
        mParent.mPointOffsets[RootT::LEVEL][idx] = 0;
        mParent.mVertexOffsets[RootT::LEVEL][idx] = 0;
        mParent.mPolygonOffsets[RootT::LEVEL][idx] = 0;
    }

    template<typename NodeT>
    size_t activeChildrenByIter(const NodeT& node) const
    {
        // iterate over each active child and count valid nodes
        size_t onValues = 0;
        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            if (mParent.mParms.sliceStyle == NO_SLICE ||
                isIterValid(iter, mParent.mParms.slicePlane, mSliceIndex)) {
                onValues++;
            }
        }
        return onValues;
    }

    size_t activeChildren(const RootT& root) const
    {
        // no fast option for counting node children for root nodes
        return activeChildrenByIter(root);
    }

    template<typename NodeT>
    size_t activeChildren(const NodeT& node) const
    {
        // when not slicing, count active children using the node value mask for performance
        if (mParent.mParms.sliceStyle == NO_SLICE) {
            return node.getValueMask().countOn();
        } else if (isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex)) {
            // this is an optimization - if a valid leaf and using slice by leaf,
            // counting active children is faster than using a value iterator
            const bool leaf = NodeT::LEVEL == 0;
            if (leaf && mParent.mParms.sliceStyle == SLICE_LEAF) {
                return node.getValueMask().countOn();
            }
            // otherwise count by iterator
            return activeChildrenByIter(node);
        }
        return size_t(0);
    }

    template<typename NodeT>
    void countVoxels(const NodeT& node, size_t idx, const RenderStyle& style) const
    {
        size_t pointsPerValue = 0;
        size_t verticesPerValue = 0;
        size_t polygonsPerValue = 0;

        if (style == STYLE_SOLID_BOX) {
            if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                pointsPerValue = 4;
                verticesPerValue = 4;
                polygonsPerValue = 1;
            } else {
                pointsPerValue = 8;
                verticesPerValue = 4*6; // each polygon has 4 vertices
                polygonsPerValue = 6;
            }
        } else if (style == STYLE_WIRE_BOX) {
            if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                pointsPerValue = 4;
                verticesPerValue = 5;
                polygonsPerValue = 1;
            } else {
                pointsPerValue = 8;
                verticesPerValue = 16;
                polygonsPerValue = 1;
            }
        } else if (style == STYLE_POINTS) {
            // in staggered mode, visualize three points - one component per face
            pointsPerValue = (mStaggered && NodeT::LEVEL == 0) ? 3 : 1;
        }

        size_t onValues = activeChildren(node);

        mParent.mPointOffsets[NodeT::LEVEL][idx] = pointsPerValue * onValues;
        mParent.mVertexOffsets[NodeT::LEVEL][idx] = verticesPerValue * onValues;
        mParent.mPolygonOffsets[NodeT::LEVEL][idx] = polygonsPerValue * onValues;
    }

    bool operator()(const RootT& root, size_t idx) const
    {
        if (mNode)  countNodes(root, idx, STYLE_NONE);
        else        countVoxels(root, idx, mParent.mParms.tileStyle);
        return true;
    }

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t idx) const
    {
        if (mNode)  countNodes(node, idx, mParent.mParms.internalStyle);
        else        countVoxels(node, idx, mParent.mParms.tileStyle);
        return true;
    }

    bool operator()(const LeafT& leaf, size_t idx) const
    {
        if (mNode)  countNodes(leaf, idx, mParent.mParms.leafStyle);
        else        countVoxels(leaf, idx, mParent.mParms.voxelStyle);
        return false;
    }

    const bool mNode;
    const openvdb::Int32 mSliceIndex;
    const bool mStaggered;
    TreeVisualizer& mParent;
}; // struct TreeVisualizer::CountOp


struct TreeVisualizer::ComputeOffsetsOp
{
    explicit ComputeOffsetsOp(TreeVisualizer& parent)
        : mParent(parent)
        , mNodeIndices(4) { }

    template<typename NodeT>
    void operator()(NodeT& node, size_t)
    {
        size_t idx = mNodeIndices[NodeT::LEVEL]++;

        // points
        size_t count = mParent.mPointOffsets[NodeT::LEVEL][idx];
        mParent.mPointOffsets[NodeT::LEVEL][idx] = mPointOffset;
        mPointOffset += count;
        // vertices
        count = mParent.mVertexOffsets[NodeT::LEVEL][idx];
        mParent.mVertexOffsets[NodeT::LEVEL][idx] = mVertexOffset;
        mVertexOffset += count;
        // polygons
        count = mParent.mPolygonOffsets[NodeT::LEVEL][idx];
        mParent.mPolygonOffsets[NodeT::LEVEL][idx] = mPolygonOffset;
        mPolygonOffset += count;
    }

    size_t pointOffset() const { return mPointOffset; }
    size_t vertexOffset() const { return mVertexOffset; }
    size_t polygonOffset() const { return mPolygonOffset; }

    TreeVisualizer& mParent;
    size_t mPointOffset{0};
    size_t mVertexOffset{0};
    size_t mPolygonOffset{0};
    std::vector<size_t> mNodeIndices;
}; // struct TreeVisualizer::ComputeOffsetsOp


template <typename TreeT>
struct TreeVisualizer::RenderPointsOp
{
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    RenderPointsOp(bool node, GA_Offset offset, openvdb::Int32 sliceIndex, bool staggered, TreeVisualizer& parent)
        : mNode(node)
        , mOffset(offset)
        , mSliceIndex(sliceIndex)
        , mStaggered(staggered)
        , mParent(parent) { }

    openvdb::Vec3d bboxToVec3d(const openvdb::CoordBBox& bbox) const
    {
        return openvdb::Vec3d(  0.5*(bbox.min().x()+bbox.max().x()),
                                0.5*(bbox.min().y()+bbox.max().y()),
                                0.5*(bbox.min().z()+bbox.max().z()));

    }

    UT_Vector3i vec3dToUTV3i(const openvdb::Vec3d& pos) const
    {
        openvdb::Coord idxPos = openvdb::Coord::floor(pos);
        return UT_Vector3i(idxPos[0], idxPos[1], idxPos[2]);
    }

    UT_Vector3 indexToWorldUTV3(const openvdb::Vec3d& pos) const
    {
        const openvdb::Vec3d posWS = mParent.mXform->indexToWorld(pos);
        return UT_Vector3(float(posWS.x()), float(posWS.y()), float(posWS.z()));
    }

    void setPos(size_t pointIndex, const openvdb::Vec3d& pos) const
    {
        mParent.mGeo->setPos3(pointIndex, indexToWorldUTV3(pos));
    }

    void setStaggeredPos(size_t pointIndex, const openvdb::Vec3d& pos) const
    {
        openvdb::Vec3d pos0 = pos - openvdb::Vec3d(0.5, 0, 0);
        openvdb::Vec3d pos1 = pos - openvdb::Vec3d(0, 0.5, 0);
        openvdb::Vec3d pos2 = pos - openvdb::Vec3d(0, 0, 0.5);

        mParent.mGeo->setPos3(pointIndex,   indexToWorldUTV3(pos0));
        mParent.mGeo->setPos3(pointIndex+1, indexToWorldUTV3(pos1));
        mParent.mGeo->setPos3(pointIndex+2, indexToWorldUTV3(pos2));
    }

    void setBoxPos(size_t pointIndex, const openvdb::CoordBBox& bbox) const
    {
        const openvdb::Vec3d min(bbox.min().x()-0.5, bbox.min().y()-0.5, bbox.min().z()-0.5);
        const openvdb::Vec3d max(bbox.max().x()+0.5, bbox.max().y()+0.5, bbox.max().z()+0.5);

        // set the box corners

        mParent.mGeo->setPos3(pointIndex, indexToWorldUTV3(openvdb::Vec3d(min.x(), min.y(), min.z())));
        mParent.mGeo->setPos3(pointIndex+1, indexToWorldUTV3(openvdb::Vec3d(min.x(), min.y(), max.z())));
        mParent.mGeo->setPos3(pointIndex+2, indexToWorldUTV3(openvdb::Vec3d(max.x(), min.y(), max.z())));
        mParent.mGeo->setPos3(pointIndex+3, indexToWorldUTV3(openvdb::Vec3d(max.x(), min.y(), min.z())));
        mParent.mGeo->setPos3(pointIndex+4, indexToWorldUTV3(openvdb::Vec3d(min.x(), max.y(), min.z())));
        mParent.mGeo->setPos3(pointIndex+5, indexToWorldUTV3(openvdb::Vec3d(min.x(), max.y(), max.z())));
        mParent.mGeo->setPos3(pointIndex+6, indexToWorldUTV3(openvdb::Vec3d(max.x(), max.y(), max.z())));
        mParent.mGeo->setPos3(pointIndex+7, indexToWorldUTV3(openvdb::Vec3d(max.x(), max.y(), min.z())));
    }

    void setPlanePos(size_t pointIndex, const openvdb::CoordBBox& bbox) const
    {
        const openvdb::Vec3d min(bbox.min().x()-0.5, bbox.min().y()-0.5, bbox.min().z()-0.5);
        const openvdb::Vec3d max(bbox.max().x()+0.5, bbox.max().y()+0.5, bbox.max().z()+0.5);

        // set the plane corners

        if (mParent.mParms.slicePlane == SLICE_XY) {
            mParent.mGeo->setPos3(pointIndex, indexToWorldUTV3(openvdb::Vec3d(min.x(), min.y(), mSliceIndex)));
            mParent.mGeo->setPos3(pointIndex+1, indexToWorldUTV3(openvdb::Vec3d(min.x(), max.y(), mSliceIndex)));
            mParent.mGeo->setPos3(pointIndex+2, indexToWorldUTV3(openvdb::Vec3d(max.x(), max.y(), mSliceIndex)));
            mParent.mGeo->setPos3(pointIndex+3, indexToWorldUTV3(openvdb::Vec3d(max.x(), min.y(), mSliceIndex)));
        } else if (mParent.mParms.slicePlane == SLICE_YZ) {
            mParent.mGeo->setPos3(pointIndex, indexToWorldUTV3(openvdb::Vec3d(mSliceIndex, min.y(), min.z())));
            mParent.mGeo->setPos3(pointIndex+1, indexToWorldUTV3(openvdb::Vec3d(mSliceIndex, min.y(), max.z())));
            mParent.mGeo->setPos3(pointIndex+2, indexToWorldUTV3(openvdb::Vec3d(mSliceIndex, max.y(), max.z())));
            mParent.mGeo->setPos3(pointIndex+3, indexToWorldUTV3(openvdb::Vec3d(mSliceIndex, max.y(), min.z())));
        } else if (mParent.mParms.slicePlane == SLICE_ZX) {
            mParent.mGeo->setPos3(pointIndex, indexToWorldUTV3(openvdb::Vec3d(min.x(), mSliceIndex, min.z())));
            mParent.mGeo->setPos3(pointIndex+1, indexToWorldUTV3(openvdb::Vec3d(max.x(), mSliceIndex, min.z())));
            mParent.mGeo->setPos3(pointIndex+2, indexToWorldUTV3(openvdb::Vec3d(max.x(), mSliceIndex, max.z())));
            mParent.mGeo->setPos3(pointIndex+3, indexToWorldUTV3(openvdb::Vec3d(min.x(), mSliceIndex, max.z())));
        }
    }

    void setIndex(size_t pointIndex, const openvdb::Vec3d& pos) const
    {
        if (!mParent.mIndexCoordHandle.isValid())   return;
        // Attach the (integer) index coordinates of the voxel at the given pos.
        mParent.mIndexCoordHandle.set(pointIndex, vec3dToUTV3i(pos));
    }

    void setStaggeredIndex(size_t pointIndex, const openvdb::Vec3d& pos) const
    {
        if (!mParent.mIndexCoordHandle.isValid())   return;

        // Attach the (integer) index coordinates of the voxel at the given pos.
        openvdb::Vec3d pos0 = pos - openvdb::Vec3d(0.5, 0, 0);
        openvdb::Vec3d pos1 = pos - openvdb::Vec3d(0, 0.5, 0);
        openvdb::Vec3d pos2 = pos - openvdb::Vec3d(0, 0, 0.5);

        mParent.mIndexCoordHandle.set(pointIndex,   vec3dToUTV3i(pos0));
        mParent.mIndexCoordHandle.set(pointIndex+1, vec3dToUTV3i(pos1));
        mParent.mIndexCoordHandle.set(pointIndex+2, vec3dToUTV3i(pos2));
    }

    template<typename ValueT>
    typename std::enable_if<IsGridTypeIntegral<ValueT>::value>::type
    setValue(GA_Offset offset, ValueT s) const
    {
        if (!mParent.mInt32Handle.isValid())    return;
        mParent.mInt32Handle.set(offset, int(s));
    }

    template<typename ValueT>
    typename std::enable_if<std::is_floating_point<ValueT>::value>::type
    setValue(GA_Offset offset, ValueT s) const
    {
        if (!mParent.mFloatHandle.isValid())    return;
        mParent.mFloatHandle.set(offset, float(s));
    }

    template<typename ValueT>
    typename std::enable_if<!IsGridTypeArithmetic<ValueT>::value>::type
    setValue(GA_Offset offset, ValueT v) const
    {
        if (!mParent.mVec3fHandle.isValid())    return;
        mParent.mVec3fHandle.set(offset, UT_Vector3(float(v[0]), float(v[1]), float(v[2])));
    }

    template<typename ValueT>
    typename std::enable_if<IsGridTypeArithmetic<ValueT>::value>::type
    setStaggeredValue(GA_Offset offset, ValueT v) const { }

    template<typename ValueT>
    typename std::enable_if<!IsGridTypeArithmetic<ValueT>::value>::type
    setStaggeredValue(GA_Offset offset, ValueT v) const
    {
        if (!mParent.mVec3fHandle.isValid())    return;
        mParent.mVec3fHandle.set(offset, UT_Vector3(float(v[0]), 0.0, 0.0));
        mParent.mVec3fHandle.set(offset+1, UT_Vector3(0.0, float(v[1]), 0.0));
        mParent.mVec3fHandle.set(offset+2, UT_Vector3(0.0, 0.0, float(v[2])));
    }

    template<typename ValueT>
    void setColorBySign(size_t idx, const ValueT& value, size_t count = 1) const
    {
        const bool negative = openvdb::math::isNegative(value);
        const auto color = SOP_OpenVDB_Visualize::colorSign(negative);
        for (size_t i = 0; i < count; i++) {
            mParent.mCdHandle.set(idx+i, color);
        }
    }

    template<typename ValueT>
    void setColorByRamp(size_t idx, const ValueT& value, size_t count = 1) const
    {
        const double min = mParent.mParms.colorMin;
        const double range = mParent.mParms.colorRange;
        const double remap = (value - min) * range;

        float values[4];
        mParent.mParms.colorRamp.getColor(remap, values);
        UT_Vector3 color(values[0], values[1], values[2]);
        for (size_t i = 0; i < count; i++) {
            mParent.mCdHandle.set(idx+i, color);
        }
    }

    void setColorByLevel(size_t idx, const openvdb::Index level, size_t count = 1) const
    {
        if (mParent.mCdHandle.isValid()) {
            const auto color = SOP_OpenVDB_Visualize::colorLevel(level);
            for (size_t i = 0; i < count; i++) {
                mParent.mCdHandle.set(idx+i, color);
            }
        }
    }

    void setColorByComponent(size_t idx) const
    {
        if (mParent.mCdHandle.isValid()) {
            mParent.mCdHandle.set(idx+0, /*red*/UT_Vector3(1.0, 0.0, 0.0));
            mParent.mCdHandle.set(idx+1, /*green*/UT_Vector3(0.0, 1.0, 0.0));
            mParent.mCdHandle.set(idx+2, /*blue*/UT_Vector3(0.0, 0.0, 1.0));
        }
    }

    template<typename ValueT>
    typename std::enable_if<std::is_floating_point<ValueT>::value>::type
    setColorByValue(size_t idx, const ValueT& value, size_t count = 1) const
    {
        if (mParent.mCdHandle.isValid()) {
            if (mParent.mParms.visualize) {
                setColorByRamp(idx, value, count);
            } else {
                setColorBySign(idx, value, count);
            }
        }
    }

    template<typename ValueT>
    typename std::enable_if<!std::is_floating_point<ValueT>::value>::type
    setColorByValue(size_t idx, const ValueT& value, size_t count = 1) const
    {
        if (mParent.mCdHandle.isValid()) {
            setColorBySign(idx, value, count);
        }
    }

    template<typename NodeT>
    void renderNodes(const NodeT& node, size_t idx, const RenderStyle& style) const
    {
        if (!style)     return;
        if (mParent.mParms.sliceStyle == NO_SLICE ||
            isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex)) {
            const openvdb::CoordBBox bbox = node.getNodeBoundingBox();
            const size_t pointIndex = mOffset + mParent.mPointOffsets[NodeT::LEVEL][idx];

            if (style == STYLE_POINTS) {
                openvdb::Vec3d pos = bboxToVec3d(bbox);
                setPos(pointIndex, pos);
                setIndex(pointIndex, pos);
                setColorByLevel(pointIndex, node.getLevel(), 1);
            } else if (style == STYLE_WIRE_BOX || style == STYLE_SOLID_BOX) {
                if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                    setPlanePos(pointIndex, bbox);
                    setColorByLevel(pointIndex, node.getLevel(), 4);
                } else {
                    setBoxPos(pointIndex, bbox);
                    setColorByLevel(pointIndex, node.getLevel(), 8);
                }
            }
        }
    }

    template<typename NodeT>
    void renderVoxels(const NodeT& node, size_t idx, const RenderStyle& style) const
    {
        if (!style)  return;

        bool allValid = mParent.mParms.sliceStyle == NO_SLICE;

        if (allValid ||
            isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex)) {

            size_t pointIndex = mOffset + mParent.mPointOffsets[NodeT::LEVEL][idx];

            // if node is a leaf node, optionally render all voxels in the leaf
            if (NodeT::LEVEL == 0) {
                allValid = allValid || mParent.mParms.sliceStyle == SLICE_LEAF;
            }

            for (auto iter = node.cbeginValueOn(); iter; ++iter) {

                if (allValid || isIterValid(iter, mParent.mParms.slicePlane, mSliceIndex)) {

                    openvdb::CoordBBox bbox;
                    bbox.expand(iter.getCoord(), node.getChildDim());

                    if (style == STYLE_POINTS) {
                        openvdb::Vec3d pos = bboxToVec3d(bbox);
                        if (mStaggered && NodeT::LEVEL == 0) {
                            setStaggeredPos(pointIndex, pos);
                            setStaggeredIndex(pointIndex, pos);
                            setStaggeredValue(pointIndex, iter.getValue());
                            setColorByComponent(pointIndex);
                            pointIndex += 3;
                        } else {
                            setPos(pointIndex, pos);
                            setIndex(pointIndex, pos);
                            setColorByValue(pointIndex, iter.getValue());
                            setValue(pointIndex, iter.getValue());
                            pointIndex++;
                        }
                    } else if (style == STYLE_WIRE_BOX || style == STYLE_SOLID_BOX) {
                        if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                            setPlanePos(pointIndex, bbox);
                            setColorByValue(pointIndex, iter.getValue(), 4);
                            pointIndex += 4;
                        } else {
                            setBoxPos(pointIndex, bbox);
                            setColorByValue(pointIndex, iter.getValue(), 8);
                            pointIndex += 8;
                        }
                    }
                }
            }
        }
    }

    bool operator()(const RootT& root, size_t idx) const
    {
        // don't render root nodes
        if (!mNode)     renderVoxels(root, idx, mParent.mParms.tileStyle);
        return true;
    }

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t idx) const
    {
        if (mNode)      renderNodes(node, idx, mParent.mParms.internalStyle);
        else            renderVoxels(node, idx, mParent.mParms.tileStyle);
        return true;
    }

    bool operator()(const LeafT& leaf, size_t idx) const
    {
        if (mNode)      renderNodes(leaf, idx, mParent.mParms.leafStyle);
        else            renderVoxels(leaf, idx, mParent.mParms.voxelStyle);
        return false;
    }

    const bool mNode;
    const GA_Offset mOffset;
    const openvdb::Int32 mSliceIndex;
    const bool mStaggered;
    TreeVisualizer& mParent;
}; // struct TreeVisualizer::RenderPointsOp


template <typename TreeT>
struct TreeVisualizer::RenderVerticesOp
{
    using ArrayT = std::vector<std::unique_ptr<size_t[]>>;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    RenderVerticesOp(bool node, std::unique_ptr<int[]>& vertices,
        openvdb::Int32 sliceIndex, TreeVisualizer& parent)
        : mNode(node)
        , mVertices(vertices)
        , mSliceIndex(sliceIndex)
        , mParent(parent) { }

    void renderWirePlane(int& pointIndex, size_t& vertexIndex) const
    {
        // 4 edges as one line
        mVertices[vertexIndex] = pointIndex;
        mVertices[vertexIndex+1] = pointIndex+1;
        mVertices[vertexIndex+2] = pointIndex+2;
        mVertices[vertexIndex+3] = pointIndex+3;
        mVertices[vertexIndex+4] = pointIndex;

        vertexIndex += 5;
        pointIndex += 4;
    }

    void renderWireBox(int& pointIndex, size_t& vertexIndex) const
    {
        // 12 edges as one line
        mVertices[vertexIndex] = pointIndex+0;
        mVertices[vertexIndex+1] = pointIndex+1;
        mVertices[vertexIndex+2] = pointIndex+2;
        mVertices[vertexIndex+3] = pointIndex+3;
        mVertices[vertexIndex+4] = pointIndex+0;
        mVertices[vertexIndex+5] = pointIndex+4;
        mVertices[vertexIndex+6] = pointIndex+5;
        mVertices[vertexIndex+7] = pointIndex+6;
        mVertices[vertexIndex+8] = pointIndex+7;
        mVertices[vertexIndex+9] = pointIndex+4;
        mVertices[vertexIndex+10] = pointIndex+5;
        mVertices[vertexIndex+11] = pointIndex+1;
        mVertices[vertexIndex+12] = pointIndex+2;
        mVertices[vertexIndex+13] = pointIndex+6;
        mVertices[vertexIndex+14] = pointIndex+7;
        mVertices[vertexIndex+15] = pointIndex+3;

        vertexIndex += 16;
        pointIndex += 8;
    }

    void renderSolidPlane(int& pointIndex, size_t& vertexIndex) const
    {
        // 1 polygon with 4 edges
        mVertices[vertexIndex] = pointIndex;
        mVertices[vertexIndex+1] = pointIndex+1;
        mVertices[vertexIndex+2] = pointIndex+2;
        mVertices[vertexIndex+3] = pointIndex+3;

        vertexIndex += 4;
        pointIndex += 4;
    }

    void renderSolidBox(int& pointIndex, size_t& vertexIndex) const
    {
        // 6 polygons with 4 edges each
        mVertices[vertexIndex] = pointIndex;
        mVertices[vertexIndex+1] = pointIndex+1;
        mVertices[vertexIndex+2] = pointIndex+2;
        mVertices[vertexIndex+3] = pointIndex+3;

        mVertices[vertexIndex+4] = pointIndex+7;
        mVertices[vertexIndex+5] = pointIndex+6;
        mVertices[vertexIndex+6] = pointIndex+5;
        mVertices[vertexIndex+7] = pointIndex+4;

        mVertices[vertexIndex+8] = pointIndex+4;
        mVertices[vertexIndex+9] = pointIndex+5;
        mVertices[vertexIndex+10] = pointIndex+1;
        mVertices[vertexIndex+11] = pointIndex;

        mVertices[vertexIndex+12] = pointIndex+6;
        mVertices[vertexIndex+13] = pointIndex+7;
        mVertices[vertexIndex+14] = pointIndex+3;
        mVertices[vertexIndex+15] = pointIndex+2;

        mVertices[vertexIndex+16] = pointIndex;
        mVertices[vertexIndex+17] = pointIndex+3;
        mVertices[vertexIndex+18] = pointIndex+7;
        mVertices[vertexIndex+19] = pointIndex+4;

        mVertices[vertexIndex+20] = pointIndex+1;
        mVertices[vertexIndex+21] = pointIndex+5;
        mVertices[vertexIndex+22] = pointIndex+6;
        mVertices[vertexIndex+23] = pointIndex+2;

        vertexIndex += 24;
        pointIndex += 8;
    }

    template<typename NodeT>
    void renderNodes(const NodeT& node, size_t idx, const RenderStyle& style) const
    {
        if (mParent.mParms.sliceStyle == NO_SLICE ||
            isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex)) {

            int pointIndex = static_cast<int>(mParent.mPointOffsets[NodeT::LEVEL][idx]);
            size_t vertexIndex = mParent.mVertexOffsets[NodeT::LEVEL][idx];

            if (style == STYLE_WIRE_BOX) {
                if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                    renderWirePlane(pointIndex, vertexIndex);
                } else {
                    renderWireBox(pointIndex, vertexIndex);
                }
            } else if (style == STYLE_SOLID_BOX) {
                if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                    renderSolidPlane(pointIndex, vertexIndex);
                } else {
                    renderSolidBox(pointIndex, vertexIndex);
                }
            }
        }
    }

    template<typename NodeT>
    void renderVoxels(const NodeT& node, size_t idx, const RenderStyle& style) const
    {
        bool allValid = mParent.mParms.sliceStyle == NO_SLICE;

        if (allValid || isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex)) {

            // if node is a leaf node, optionally render all voxels in the leaf
            if (NodeT::LEVEL == 0) {
                allValid = allValid || mParent.mParms.sliceStyle == SLICE_LEAF;
            }

            int pointIndex = static_cast<int>(mParent.mPointOffsets[NodeT::LEVEL][idx]);
            size_t vertexIndex = mParent.mVertexOffsets[NodeT::LEVEL][idx];

            for (auto iter = node.cbeginValueOn(); iter; ++iter) {
                if (allValid || isIterValid(iter, mParent.mParms.slicePlane, mSliceIndex)) {
                    if (style == STYLE_WIRE_BOX) {
                        if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                            renderWirePlane(pointIndex, vertexIndex);
                        } else {
                            renderWireBox(pointIndex, vertexIndex);
                        }
                    } else if (style == STYLE_SOLID_BOX) {
                        if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                            renderSolidPlane(pointIndex, vertexIndex);
                        } else {
                            renderSolidBox(pointIndex, vertexIndex);
                        }
                    }
                }
            }
        }
    }

    bool operator()(const RootT& root, size_t idx) const
    {
        // don't render root nodes
        if (!mNode)     renderVoxels(root, idx, mParent.mParms.tileStyle);
        return true;
    }

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t idx) const
    {
        if (mNode)      renderNodes(node, idx, mParent.mParms.internalStyle);
        else            renderVoxels(node, idx, mParent.mParms.tileStyle);
        return true;
    }

    bool operator()(const LeafT& leaf, size_t idx) const
    {
        if (mNode)      renderNodes(leaf, idx, mParent.mParms.leafStyle);
        else            renderVoxels(leaf, idx, mParent.mParms.voxelStyle);
        return false;
    }

    const bool mNode;
    std::unique_ptr<int[]>& mVertices;
    const openvdb::Int32 mSliceIndex;
    TreeVisualizer& mParent;
}; // struct TreeVisualizer::RenderVerticesOp


template <typename TreeT>
struct TreeVisualizer::RenderGeometrySingleThreadedOp
{
    using ArrayT = std::vector<std::unique_ptr<size_t[]>>;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    RenderGeometrySingleThreadedOp(bool node, GA_Offset offset, std::unique_ptr<int[]>& vertices,
        openvdb::Int32 sliceIndex, TreeVisualizer& parent)
        : mNode(node)
        , mOffset(offset)
        , mVertices(vertices.get())
        , mSliceIndex(sliceIndex)
        , mParent(parent) { }

    void renderWire(int count)
    {
        GU_PrimPoly* poly = GU_PrimPoly::build(mParent.mGeo, 0, GU_POLY_OPEN);
        for (int i = 0; i < count; i++) {
            poly->appendVertex(mOffset + *mVertices++);
        }
    }

    void renderSolid(int count)
    {
        for (int i = 0; i < count; i++) {
            GU_PrimPoly* poly = GU_PrimPoly::build(mParent.mGeo, 0);
            for (int j = 0; j < 4; j++) {
                poly->appendVertex(mOffset + *mVertices++);
            }
            poly->close();
        }
    }

    template <typename NodeT>
    void renderNodes(const NodeT& node, const RenderStyle& style)
    {
        if (mParent.mParms.sliceStyle == NO_SLICE ||
            isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex)) {
            if (style == STYLE_WIRE_BOX) {
                if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                    renderWire(5);
                } else {
                    renderWire(16);
                }
            } else if (style == STYLE_SOLID_BOX) {
                if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                    renderSolid(1);
                } else {
                    renderSolid(6);
                }
            }
        }
    }

    template <typename NodeT>
    void renderVoxels(const NodeT& node, const RenderStyle& style)
    {
        bool allValid = mParent.mParms.sliceStyle == NO_SLICE;

        if (allValid ||
            isNodeValid(node, mParent.mParms.slicePlane, mSliceIndex)) {

            // if node is a leaf node, optionally render all voxels in the leaf
            if (NodeT::LEVEL == 0) {
                allValid = allValid || mParent.mParms.sliceStyle == SLICE_LEAF;
            }

            for (auto iter = node.cbeginValueOn(); iter; ++iter) {
                if (allValid || isIterValid(iter, mParent.mParms.slicePlane, mSliceIndex)) {
                    if (style == STYLE_WIRE_BOX) {
                        if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                            renderWire(5);
                        } else {
                            renderWire(16);
                        }
                    } else if (style == STYLE_SOLID_BOX) {
                        if (mParent.mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                            renderSolid(1);
                        } else {
                            renderSolid(6);
                        }
                    }
                }
            }
        }
    }

    void operator()(const RootT& root, size_t)
    {
        // don't render root nodes
        if (!mNode)     renderVoxels(root, mParent.mParms.tileStyle);
    }

    template<typename NodeT>
    void operator()(const NodeT& node, size_t)
    {
        if (mNode)      renderNodes(node, mParent.mParms.internalStyle);
        else            renderVoxels(node, mParent.mParms.tileStyle);
    }

    void operator()(const LeafT& leaf, size_t)
    {
        if (mNode)      renderNodes(leaf, mParent.mParms.leafStyle);
        else            renderVoxels(leaf, mParent.mParms.voxelStyle);
    }

    const bool mNode;
    const GA_Offset mOffset;
    int* mVertices;
    const openvdb::Int32 mSliceIndex;
    TreeVisualizer& mParent;
}; // struct TreeVisualizer::RenderGeometrySingleThreadedOp


template <typename GridT>
void
TreeVisualizer::createPointAttributes(const GridT& grid)
{
    if (mParms.addColor) {
        mCdHandle.bind(mGeo->findDiffuseAttribute(GA_ATTRIB_POINT));
        if (!mCdHandle.isValid()) {
            mCdHandle.bind(mGeo->addDiffuseAttribute(GA_ATTRIB_POINT));
        }
    }

    if (mParms.addIndexCoord &&
        ((mParms.tileStyle == STYLE_POINTS)
        || (mParms.voxelStyle == STYLE_POINTS)
        || (mParms.leafStyle == STYLE_POINTS)))
    {
        const UT_String attrName = "vdb_ijk";

        GA_RWAttributeRef attribHandle = mGeo->findIntTuple(GA_ATTRIB_POINT, attrName, 3);
        if (!attribHandle.isValid()) {
            attribHandle = mGeo->addIntTuple(GA_ATTRIB_POINT, attrName, 3, GA_Defaults(0));
        }

        mIndexCoordHandle = attribHandle.getAttribute();

        UT_String varName = attrName;
        varName.toUpper();
        mGeo->addVariableName(attrName, varName);
    }

    if (mParms.addValue &&
        ((mParms.tileStyle == STYLE_POINTS) || (mParms.voxelStyle == STYLE_POINTS)))
    {
        const std::string valueType = grid.valueType();

        UT_String attrName;
        if (mParms.useGridName) {
            attrName = grid.getName();
            attrName.forceValidVariableName();
        }

        if (valueType == openvdb::typeNameAsString<float>() ||
            valueType == openvdb::typeNameAsString<double>())
        {
            if (!attrName.isstring()) attrName = "vdb_float";
            UT_String varName = attrName;
            varName.toUpper();

            GA_RWAttributeRef attribHandle =
                mGeo->findFloatTuple(GA_ATTRIB_POINT, attrName, 1);

            if (!attribHandle.isValid()) {
                attribHandle = mGeo->addFloatTuple(
                    GA_ATTRIB_POINT, attrName, 1, GA_Defaults(0));
            }

            mFloatHandle = attribHandle.getAttribute();
            mGeo->addVariableName(attrName, varName);

        } else if (valueType == openvdb::typeNameAsString<int32_t>() ||
            valueType == openvdb::typeNameAsString<int64_t>() ||
            valueType == openvdb::typeNameAsString<bool>())
        {
            if (!attrName.isstring()) attrName = "vdb_int";
            UT_String varName = attrName;
            varName.toUpper();

            GA_RWAttributeRef attribHandle =
                mGeo->findIntTuple(GA_ATTRIB_POINT, attrName, 1);

            if (!attribHandle.isValid()) {
                attribHandle = mGeo->addIntTuple(
                    GA_ATTRIB_POINT, attrName, 1, GA_Defaults(0));
            }

            mInt32Handle = attribHandle.getAttribute();
            mGeo->addVariableName(attrName, varName);

        } else if (valueType == openvdb::typeNameAsString<openvdb::Vec3s>() ||
            valueType == openvdb::typeNameAsString<openvdb::Vec3d>())
        {
            if (!attrName.isstring()) attrName = "vdb_vec3f";
            UT_String varName = attrName;
            varName.toUpper();

            GA_RWAttributeRef attribHandle =
                mGeo->findFloatTuple(GA_ATTRIB_POINT, attrName, 3);

            if (!attribHandle.isValid()) {
                attribHandle = mGeo->addFloatTuple(
                    GA_ATTRIB_POINT, attrName, 3, GA_Defaults(0));
            }

            mVec3fHandle = attribHandle.getAttribute();
            mGeo->addVariableName(attrName, varName);

        } else {
            throw std::runtime_error(
                "value attributes are not supported for values of type " + valueType);
        }
    }
}

template<typename GridType>
openvdb::Int32 TreeVisualizer::computeIndexSlice(const GridType& grid)
{
    openvdb::Int32 slice(std::numeric_limits<openvdb::Int32>::max());

    if (mParms.sliceStyle == NO_SLICE)  return slice;

    // set axis index

    int axis = -1;
    if (mParms.slicePlane == SLICE_XY) {
        axis = 2; // z
    } else if (mParms.slicePlane == SLICE_YZ) {
        axis = 0; // x
    } else if (mParms.slicePlane == SLICE_ZX) {
        axis = 1; // y
    }

    // compute min and max index space bounds

    openvdb::CoordBBox activeVoxelBbox;
    if (!grid.tree().evalActiveVoxelBoundingBox(activeVoxelBbox)) {
        return slice;
    }

    const openvdb::Int32 min = activeVoxelBbox.min()[axis];
    const openvdb::Int32 max = activeVoxelBbox.max()[axis];

    if (mParms.useWorldSpace) {
        // convert world space offset to index space

        openvdb::Vec3d posWS(0);
        posWS(axis) = mParms.sliceOffset;
        const openvdb::Vec3d pos = grid.worldToIndex(posWS);
        double offsetIndexSpace = pos(axis);

        // now round to nearest index-space integer

        slice = openvdb::Int32(openvdb::math::Round(offsetIndexSpace));

        // store the offsets in the node cache so that they can be seamlessly converted if desired

        if (mParms.cachedOffsetWS) {
            *mParms.cachedOffsetWS = slice;
            mParms.cachedOffsetWS = nullptr; // reset pointer so only the offset from the first grid is stored
        }
        if (mParms.cachedOffset) {
            // remap index space value from (min,max) to (-1,1)

            const double gridOffset = (offsetIndexSpace - min + 0.5) / (max - min + 1);
            const double sliceOffset = (gridOffset * 2.0) - 1.0;

            *mParms.cachedOffset = sliceOffset;
            mParms.cachedOffset = nullptr; // reset pointer so only the offset from the first grid is stored
        }
    } else {

        // remap offset from (-1,1) => (min, max)

        const double gridOffset = (mParms.sliceOffset + 1.0) * 0.5;;
        const double offsetIndexSpace = (gridOffset * (max - min + 1)) + min - 0.5;

        // now round to nearest index-space integer

        slice = openvdb::Int32(openvdb::math::Round(offsetIndexSpace));

        // store the offsets in the node cache so that they can be seamlessly converted if desired

        if (mParms.cachedOffset) {
            *mParms.cachedOffset = slice;
            mParms.cachedOffset = nullptr; // reset pointer so only the offset from the first grid is stored
        }
        if (mParms.cachedOffsetWS) {
            // convert index space offset to world space

            openvdb::Vec3d pos(0);
            pos(axis) = offsetIndexSpace;
            const openvdb::Vec3d posWS = grid.indexToWorld(pos);

            *mParms.cachedOffsetWS = posWS(axis);
            mParms.cachedOffsetWS = nullptr; // reset pointer so only the offset from the first grid is stored
        }
    }

    return slice;
}

template<typename GridType>
void
TreeVisualizer::allocateOffsetArrays(const GridType& grid)
{
    // allocate offsets per node arrays - no zero value initialization

    const auto nodeCounts = grid.tree().nodeCount();

    for (const auto& count : nodeCounts) {
        mPointOffsets.emplace_back(new size_t[count]);
        mVertexOffsets.emplace_back(new size_t[count]);
        mPolygonOffsets.emplace_back(new size_t[count]);
    }
}

template<typename GridType>
void
TreeVisualizer::render(bool node, const GridType& grid, openvdb::Int32 sliceIndex,
    const RenderStyle& style1, const RenderStyle& style2)
{
    using TreeType = typename GridType::TreeType;

    const bool staggered = !mParms.ignoreStaggeredVectors &&
        (grid.getGridClass() == openvdb::GRID_STAGGERED);

    // define the number of points and polygons to generate per node

    CountOp<TreeType> countOp(node, sliceIndex, staggered, *this);
    openvdb::tree::DynamicNodeManager<const TreeType> nodeManager(grid.constTree());
    nodeManager.foreachTopDown(countOp, /*threaded=*/true);

    // make points per node cumulative - note that to preserve existing behavior,
    // the depth-first node visitor is used to ensure the same point order

    ComputeOffsetsOp offsetsOp(*this);
    openvdb::tools::visitNodesDepthFirst(grid.tree(), offsetsOp);

    GA_Size pointCount = offsetsOp.pointOffset();
    GA_Size vertexCount = offsetsOp.vertexOffset();
    GA_Size polygonCount = offsetsOp.polygonOffset();

    if (pointCount > GA_Size(0)) {

        // allocate points

        GA_Offset pointOffset = mGeo->appendPointBlock(pointCount);

        // use nested parallelism to render the points and geometry concurrently
        // and to harden the point attributes

        tbb::task_group task_group;
        task_group.run([&]{
            // auto harden attributes for threading
            using HardenPtr = std::unique_ptr<GA_AutoHardenForThreading>;
            HardenPtr hardenP;
            HardenPtr hardenIndexCoord;
            HardenPtr hardenCd;
            HardenPtr hardenInt32;
            HardenPtr hardenFloat;
            HardenPtr hardenVec3f;

            auto getAutoHarden = [](GA_Attribute* attribute, bool harden = true)
            {
                if (harden && attribute) {
                    return std::make_unique<GA_AutoHardenForThreading>(*attribute);
                }
                return HardenPtr();
            };

            tbb::task_group task_group2;
            task_group2.run([&]{ hardenP = getAutoHarden(mGeo->getP()); });
            task_group2.run([&]{ hardenIndexCoord = getAutoHarden(mIndexCoordHandle.getAttribute()); });
            task_group2.run([&]{ hardenCd = getAutoHarden(mCdHandle.getAttribute()); });
            task_group2.run([&]{ hardenInt32 = getAutoHarden(mInt32Handle.getAttribute(), !node); });
            task_group2.run([&]{ hardenFloat = getAutoHarden(mFloatHandle.getAttribute(), !node); });
            task_group2.run([&]{ hardenVec3f = getAutoHarden(mVec3fHandle.getAttribute(), !node); });
            task_group2.wait();

            RenderPointsOp<TreeType> renderPointsOp(node, pointOffset, sliceIndex, staggered, *this);
            nodeManager.foreachTopDown(renderPointsOp, /*threaded=*/true);

        });
        task_group.run([&]{
            // no zero value initialization
            std::unique_ptr<int[]> vertices(new int[vertexCount]);

            // note that vertex indices have range [0 - pointCount) so no need to increment by pointOffset
            RenderVerticesOp<TreeType> renderVerticesOp(node, vertices, sliceIndex, *this);
            nodeManager.foreachTopDown(renderVerticesOp, /*threaded=*/true);

            if ((style1 == STYLE_WIRE_BOX && style2 != STYLE_SOLID_BOX) ||
                (style2 == STYLE_WIRE_BOX && style1 != STYLE_SOLID_BOX)) {
                if (mParms.sliceStyle == SLICE_VOXEL_FLATTEN) {
                    GEO_PolyCounts sizelist;
                    sizelist.append(5, polygonCount);
                    GU_PrimPoly::buildBlock(mGeo, pointOffset, pointCount, sizelist, vertices.get(), /*closed=*/false);
                } else {
                    GEO_PolyCounts sizelist;
                    sizelist.append(16, polygonCount);
                    GU_PrimPoly::buildBlock(mGeo, pointOffset, pointCount, sizelist, vertices.get(), /*closed=*/false);
                }
            } else if ((style1 == STYLE_SOLID_BOX && style2 != STYLE_WIRE_BOX) ||
                        (style2 == STYLE_SOLID_BOX && style1 != STYLE_WIRE_BOX)) {
                GEO_PolyCounts sizelist;
                sizelist.append(4, polygonCount);
                GU_PrimPoly::buildBlock(mGeo, pointOffset, pointCount, sizelist, vertices.get(), /*closed=*/true);
            } else {
                // when rendering a mixture of solid and wire boxes, generate geometry single-threaded
                RenderGeometrySingleThreadedOp<TreeType> renderGeometryOp(node, pointOffset, vertices, sliceIndex, *this);
                openvdb::tools::visitNodesDepthFirst(grid.tree(), renderGeometryOp);
            }
        });
        task_group.wait();
    }
}


template<typename GridType>
void
TreeVisualizer::operator()(const GridType& grid)
{
    using TreeType = typename GridType::TreeType;

    bool renderNodes = mParms.internalStyle || mParms.leafStyle;
    bool renderVoxels = mParms.tileStyle || mParms.voxelStyle;

    if (!renderNodes && !renderVoxels)      return;

    mXform = &grid.transform();

    createPointAttributes(grid);

    openvdb::Int32 indexSlice = computeIndexSlice(grid);

    allocateOffsetArrays(grid);

    if (renderNodes) {
        render(/*node=*/true, grid, indexSlice, mParms.internalStyle, mParms.leafStyle);
    }

    if (mInterrupter && mInterrupter->wasInterrupted(-1))   return;

    if (renderVoxels) {
        render(/*node=*/false, grid, indexSlice, mParms.tileStyle, mParms.voxelStyle);
    }
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Visualize::Cache::cookVDBSop(OP_Context& context)
{
    try {
        const fpreal time = context.getTime();

        hvdb::HoudiniInterrupter boss("Visualizing VDBs");

        const GU_Detail* refGdp = inputGeo(0);
        if (refGdp == nullptr) return error();

        // Get the group of grids to visualize.
        const GA_PrimitiveGroup* group = matchGroup(*refGdp, evalStdString("group", time));

        // Evaluate the UI parameters.
        TreeParms treeParms;
        treeParms.internalStyle =
            evalRenderStyle(*this, "drawinternalnodes", "internalstyle", time);
        treeParms.tileStyle = evalRenderStyle(*this, "drawtiles", "tilestyle", time);
        treeParms.leafStyle = evalRenderStyle(*this, "drawleafnodes", "leafstyle", time);
        treeParms.voxelStyle = evalRenderStyle(*this, "drawvoxels", "voxelstyle", time);
        treeParms.addColor = bool(evalInt("addcolor", 0, time));
        treeParms.addValue = bool(evalInt("addvalue", 0, time));
        treeParms.addIndexCoord = bool(evalInt("addindexcoord", 0, time));
        treeParms.useGridName = bool(evalInt("usegridname", 0, time));
        treeParms.ignoreStaggeredVectors = bool(evalInt("ignorestaggered", 0, time));

        if (bool(evalInt("slice", 0, time))) {
            const std::string method = evalStdString("slicemethod", time);
            const std::string plane = evalStdString("plane", time);
            if (method == "voxelflatten") {
                treeParms.sliceStyle = SLICE_VOXEL_FLATTEN;
            } else if (method == "voxel") {
                treeParms.sliceStyle = SLICE_VOXEL;
            } else if (method == "leaf") {
                treeParms.sliceStyle = SLICE_LEAF;
            }
            if (plane == "xy")      treeParms.slicePlane = SLICE_XY;
            else if (plane == "yz") treeParms.slicePlane = SLICE_YZ;
            else if (plane == "zx") treeParms.slicePlane = SLICE_ZX;
            treeParms.useWorldSpace = bool(evalInt("useworldspace", 0, time));
            if (treeParms.useWorldSpace) {
                treeParms.sliceOffset = evalFloat("offset", 0, time);
            } else {
                treeParms.sliceOffset = evalFloat("relativeoffset", 0, time);
            }
            // cache the offsets
            treeParms.cachedOffset = &mOffset;
            treeParms.cachedOffsetWS = &mOffsetWS;
        }

        treeParms.visualize = bool(evalInt("visualize", 0, time));
        if (treeParms.visualize) {
            // copy data to the TreeParms UT_Ramp object
            auto &&sopparms = cookparms()->parms<SOP_NodeParmsOptions>();
            const UT_OptionEntry *rampentry = sopparms.getOptionEntry("visramp");
            if (rampentry) {
                UT_StringHolder value;
                if (rampentry->importOption(value)) {
                    UT_IStream is(value.c_str(), value.length(), UT_ISTREAM_ASCII);
                    treeParms.colorRamp.load(is);
                }
            }
            treeParms.colorMin = evalFloat("visrange", 0, time);
            double colorMax = evalFloat("visrange", 1, time);
            treeParms.colorRange = 1.0 / (colorMax - treeParms.colorMin);
        }

        const bool drawTree = (treeParms.internalStyle || treeParms.tileStyle
            || treeParms.leafStyle || treeParms.voxelStyle);
        const bool showFrustum = bool(evalInt("previewfrustum", 0, time));

        if (!boss.wasInterrupted()
            && (drawTree || showFrustum))
        {
            // for each VDB primitive...
            for (hvdb::VdbPrimCIterator it(refGdp, group); it; ++it) {

                if (boss.wasInterrupted()) break;

                const GU_PrimVDB *vdb = *it;

                // draw tree topology
                if (drawTree) {
                    TreeVisualizer draw(*gdp, treeParms, &boss.interrupter());
                    hvdb::GEOvdbApply<hvdb::AllGridTypes>(*vdb, draw);
                }

                if (showFrustum) {
                    UT_Vector3 box_color(0.6f, 0.6f, 0.6f);
                    UT_Vector3 tick_color(0.0f, 0.0f, 0.0f);
                    hvdb::drawFrustum(*gdp, vdb->getGrid().transform(),
                        &box_color, &tick_color, /*shaded*/true);
                }
            }
        }

        if (boss.wasInterrupted()) {
            addWarning(SOP_MESSAGE, "Process was interrupted");
        }

        // create and set a gl_lit detail attribute to zero to disable gl lighting

        gdp->setDetailAttributeI("gl_lit", 0);

        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}
