// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file SOP_OpenVDB_Activate.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Activate VDBs according to various rules

// OpenVDB and Houdini use different relative directories, but SESI_OPENVDB
// is not yet defined at this point.
#if 1
  #include <houdini_utils/ParmFactory.h>
  #include <openvdb_houdini/Utils.h>
  #include <openvdb_houdini/SOP_NodeVDB.h>
#else
  #include "Utils.h"
  #include "ParmFactory.h"
  #include "SOP_NodeVDB.h"
#endif

#include <UT/UT_Version.h>
#include <GEO/GEO_PrimVolume.h>
#include <GU/GU_PrimVDB.h>
#include <GU/GU_ConvexHull3D.h>
#include <OP/OP_Node.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Parm.h>

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/Activate.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/Prune.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

using namespace openvdb_houdini;

enum REGIONTYPE_NAMES
{
    REGIONTYPE_POSITION,
    REGIONTYPE_VOXEL,
    REGIONTYPE_EXPAND,
    REGIONTYPE_REFERENCE,
    REGIONTYPE_DEACTIVATE,
    REGIONTYPE_FILL
};

enum OPERATION_NAMES
{
    OPERATION_UNION,
    OPERATION_INTERSECT,
    OPERATION_SUBTRACT,
    OPERATION_COPY
};

class SOP_VDBActivate : public hvdb::SOP_NodeVDB
{
public:
    const char  *inputLabel(unsigned idx) const override;
    int          isRefInput(unsigned i) const override;

    bool         updateParmsFlags() override;

    static OP_Node       *factory(OP_Network*, const char *, OP_Operator*);

    class Cache: public SOP_VDBCacheOptions
    {
        OP_ERROR cookVDBSop(OP_Context&) override;

        openvdb::CoordBBox    getIndexSpaceBounds(OP_Context &context,
                                                  const GEO_PrimVDB &vdb);
        UT_BoundingBox        getWorldBBox(fpreal t);

    protected:
        REGIONTYPE_NAMES REGIONTYPE(double t) { return (REGIONTYPE_NAMES) evalInt("regiontype", 0, t); }

        OPERATION_NAMES OPERATION(fpreal t) { return (OPERATION_NAMES) evalInt("operation", 0, t); }


        UT_Vector3D CENTER(fpreal t)
            { return UT_Vector3D(evalFloat("center", 0, t),
                                 evalFloat("center", 1, t),
                                 evalFloat("center", 2, t)); }
        UT_Vector3D SIZE(fpreal t)
            { return UT_Vector3D(evalFloat("size", 0, t),
                                 evalFloat("size", 1, t),
                                 evalFloat("size", 2, t)); }

        openvdb::Coord MINPOS(fpreal t)
            { return openvdb::Coord(evalVec3i("min", t)); }
        openvdb::Coord MAXPOS(fpreal t)
            { return openvdb::Coord(evalVec3i("max", t)); }
    };
protected:
             SOP_VDBActivate(OP_Network *net, const char *name, OP_Operator *entry);
    ~SOP_VDBActivate() override {}

    REGIONTYPE_NAMES REGIONTYPE(double t) { return (REGIONTYPE_NAMES) evalInt("regiontype", 0, t); }

    OPERATION_NAMES OPERATION(fpreal t) { return (OPERATION_NAMES) evalInt("operation", 0, t); }
};

void
#ifdef SESI_OPENVDB
new_SOP_VDBActivate(OP_OperatorTable *table)
#else
newSopOperator(OP_OperatorTable *table)
#endif
{
    if (table == NULL) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Source Group")
              .setHelpText("Specify a subset of the input VDB grids to be processed.")
              .setChoiceList(&hutil::PrimGroupMenuInput1)
              .setTooltip("The vdb primitives to change the active region in.")
              .setDocumentation("The vdb primitives to change the active region in."));

    // Match OPERATION
    const char* operations[] = {
        "union",        "Union",
        "intersect",    "Intersect",
        "subtract",     "A - B",
        "copy",         "Copy",
        NULL
    };
    parms.add(hutil::ParmFactory(PRM_ORD, "operation", "Operation")
              .setDefault(PRMzeroDefaults)
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, operations)
              .setTooltip("The vdb's current region is combined with the specified region in one of several ways.")
              .setDocumentation(
R"(The vdb's current region is combined with the specified region
in one of several ways.

Union:
    All voxels that lie in the specified region will
    be activated.  Other voxels will retain their original
    activation states.

Intersect:
    Any voxel not in the specified region will be deactivated
    and set to the background value.

A - B:
    Any voxel that is in the specified region will be deactivated
    and set to the background value.

Copy:
    If a voxel is outside the specified region, it is set
    to inactive and the background value.  If it is inside,
    it is marked as active.)"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "setvalue", "Write Value")
                .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
                .setDefault(PRMoneDefaults));
    parms.add(hutil::ParmFactory(PRM_FLT, "value", "Value")
                .setDefault(PRMoneDefaults)
                .setTooltip("In the Union and Copy modes, when voxels are marked active they can also be initialized to a constant value.")
                .setDocumentation(
R"(In the Union and Copy modes, when voxels are marked active they can also be
initialized to a constant value.  This will be done to all voxels that are
made active by the specification - including those that were already
active.

Thus, the Voxel Coordinats option will have the effect of setting a cube
area to a constant value.)"));

    // Match REGIONTYPE
    parms.beginExclusiveSwitcher("regiontype", "Region Type");

    parms.addFolder("Position");
/*
    This defines a cube in SOP space.  Any voxel that touches this
    cube will be part of the selected region.
*/
    parms.add(hutil::ParmFactory(PRM_XYZ, "center", "Center")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults)
              .setTooltip("This defines a cube in SOP space.")
              .setDocumentation(
R"(This defines a cube in SOP space.  Any voxel that touches this cube will be part of the selected region.)"));

    parms.add(hutil::ParmFactory(PRM_XYZ, "size", "Size")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults)
              .setTooltip("This defines a cube in SOP space.")
              .setDocumentation(
R"(This defines a cube in SOP space.  Any voxel that touches this cube will be part of the selected region.)"));

    parms.addFolder("Voxel");
/*
    Defines minimum and maximum values of a box in voxel-coordinates.  This is
    an inclusive range, so includes the maximum voxel.
*/
    parms.add(hutil::ParmFactory(PRM_XYZ, "min", "Min")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults)
              .setTooltip("Defines minimum and maximum values of a box in voxel-coordinates.")
              .setDocumentation(
R"(Defines minimum values of a box in voxel-coordinates.  This is
an inclusive range, so includes the maximum voxel.)"));
    parms.add(hutil::ParmFactory(PRM_XYZ, "max", "Max")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults)
              .setTooltip("Defines minimum and maximum values of a box in voxel-coordinates.")
              .setDocumentation(
R"(Defines maximum values of a box in voxel-coordinates.  This is
an inclusive range, so includes the maximum voxel.)"));

    parms.addFolder("Expand");
/*
    Expand the active area by at least the specified number of voxels.  Does not
    support operation or setting of values.
*/
    parms.add(hutil::ParmFactory(PRM_INT, "expand", "Expand Voxels")
                .setDefault(PRMoneDefaults)
              .setRange(PRM_RANGE_UI, -5, PRM_RANGE_UI, 5)
              .setTooltip("Expand the active area by at least the specified number of voxels.")
                .setDocumentation(
R"(Expand the active area by at least the specified number of voxels.  Does not support
operation or setting of values.)"));

/*
    Expand the active area by at least the specified distance. Does not support
    operation or setting of values.
*/
    parms.add(hutil::ParmFactory(PRM_FLT, "expanddist", "Expand Distance")
              .setDefault(PRMzeroDefaults)
              .setRange(PRM_RANGE_UI, 0.0f, PRM_RANGE_UI, 2.0f)
              .setTooltip("Expand the active area by at least the specified distance.")
              .setDocumentation(
                                R"(Expand the active area by at least the specified distance. Does not support operation or setting of values.)"));

/*
    Specifies which nearby voxels are considered neighbors for expansion.
*/
    parms.add(hutil::ParmFactory(PRM_STRING, "expansionpattern", "Expansion Pattern")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "face", "Plus",
            "faceedge", "Diamond",
            "faceedgevertex", "Box"
        })
        .setDefault("face")
        .setTooltip("Set pattern used to identify neighbor voxels for expansion.")
        .setDocumentation(
R"(Specifies which nearby voxels are considered neighbors for expansion.)"));

    parms.addFolder("Reference");
/*
    Uses the second input to determine the selected region.
*/
    parms.add(hutil::ParmFactory(PRM_STRING, "boundgroup", "Bound Group")
              .setChoiceList(&hutil::PrimGroupMenuInput2)
              .setTooltip("Which primitives of the second input contribute to the bounding box computation.")
              .setDocumentation(
R"(Which primitives of the second input contribute to the bounding box
computation.)"));
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "usevdb", "Activate Using VDBs")
                .setDefault(PRMzeroDefaults)
                .setTooltip("If turned on, only VDBs are used for activation.")
                .setDocumentation(
R"(If turned on, only VDBs are used for activation.  They will activate
wherever they themselves are already active.  This can be used to
transfer the active region from one VDB to another, even if they are
not aligned.

If turned off, the bounding box of the chosen primitives are used
instead and activated as if they were specified as World Positions.)"));

#if UT_VERSION_INT >= 0x13050000        // 19.5 or later
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "usehull", "Activate Using Convex Hull")
                .setDefault(PRMzeroDefaults)
                .setTooltip("If turned on, activate with convex hull of points.")
                .setDocumentation(
R"(If turned on, only convex hull fo points are used for activation.)"));
    parms.add(hutil::ParmFactory(PRM_STRING, "boundptgroup", "Convex Hull Group")
              .setChoiceList(&SOP_Node::pointGroupMenu)
              .setSpareData(SOP_Node::getGroupSelectButton(GA_GROUP_POINT,
                    nullptr, 1, &SOP_Node::theSecondInput))
              .setTooltip("Points of the second input contribute to the convex hull computation.")
              .setDocumentation(
R"(Which points of the second input contribute to the convex hull
computation.)"));
    parms.add(hutil::ParmFactory(PRM_FLT, "voxeloffset", "Voxel Offset")
              .setDefault(PRMzeroDefaults)
              .setRange(PRM_RANGE_UI, -10.0f, PRM_RANGE_UI, 10.0f)
              .setTooltip("Expand the convex hull by this number of voxels.")
              .setDocumentation(
R"(Expand the convex hull by the specified number of voxels.)"));
    parms.add(hutil::ParmFactory(PRM_FLT, "worldoffset", "World Offset")
              .setDefault(PRMzeroDefaults)
              .setRange(PRM_RANGE_UI, -1.0f, PRM_RANGE_UI, 1.0f)
              .setTooltip("Expand the convex hull by this distance.")
              .setDocumentation(
R"(Expand the convex hull by the specified distance.)"));
#endif



    parms.addFolder("Deactivate");
/*
    Any voxels that have the background value will be deactivated.  This
    is useful for cleaning up the result of an operation that may
    have speculatively activated a large band of voxels, but may
    not have placed non-background values in all of them.

    For example, you may have a VDB Activate before a Volume VOP
    with Expand turned on to ensure you have room to displace the volume.
    Then when you are done, you can use one with Deactivate to free
    up the voxels you didn't need to use.
*/

    // Deactivation tolerance slider
    parms.add(hutil::ParmFactory(PRM_FLT, "bgtolerance", "Deactivate Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1)
        .setTooltip(
            "Deactivate active output voxels whose values\n"
            "equal the output VDB's background value.\n"
            "Voxel values are considered equal if they differ\n"
            "by less than the specified tolerance.")
        .setDocumentation(
            "When deactivation of background voxels is enabled,"
            " voxel values are considered equal to the background"
            " if they differ by less than this tolerance."));

    parms.addFolder("Fill SDF");
/*
    Any voxels that are inside the SDF will be marked active.  If they
    were previously inactive, they will be set to the negative-background
    values.   Tiles will remain sparse in this process.
*/

    parms.endSwitcher();

    // Prune toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "Prune Tolerance")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setTooltip("This tolerance is used to detect constant regions and collapse them.")
        .setDocumentation(
R"(After building the VDB grid there may be undetected constant tiles.
This tolerance is used to detect constant regions and collapse them.
Such areas that are within the background value will also be marked
inactive.)"));

    // Pruning tolerance slider
    parms.add(hutil::ParmFactory(
        PRM_FLT_J, "tolerance", "Prune Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1)
        .setTooltip("This tolerance is used to detect constant regions and collapse them.")
        .setDocumentation(
R"(After building the VDB grid there may be undetected constant tiles.
This tolerance is used to detect constant regions and collapse them.
Such areas that are within the background value will also be marked
inactive.)"));

    hvdb::OpenVDBOpFactory("VDB Activate",
        SOP_VDBActivate::factory, parms, *table)
        .addInput("VDBs to Activate")
        .addOptionalInput("Bounds to Activate")
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_VDBActivate::Cache; })
        .setDocumentation(
R"(#icon: COMMON/openvdb)"
R"(#tags: vdb

= OpenVDB Activate =

"""Activates voxel regions of a VDB for further processing."""

[Include:volume_types]

Many volume operations, such as Volume Mix and Volume VOP, only process
active voxels in the sparse volume.  This can be a problem if you know a
certain area in space will evaluate to a non-zero value, but it is
inactive in your original volume.

The VDB Activate SOP provides tools for manipulating this active region.
It can also fill the newly added regions to a constant value, useful
for interactively determining what is changing.

TIP: To see the current active region, you can use the VDB Visualize SOP
     and set it to Tree Nodes, Disabled; Active Constant Tiles, Wireframe Box;
     and Active Voxels, Wireframe Box.

@related

- [Node:sop/vdb]
- [Node:sop/vdbactivatesdf]
- [Node:sop/volumevop]
- [Node:sop/volumemix]
)");
}

bool
SOP_VDBActivate::updateParmsFlags()
{
    bool has_bounds = (nInputs() > 1);
    REGIONTYPE_NAMES regiontype = REGIONTYPE(0.0f);
    OPERATION_NAMES operation = OPERATION(0.0f);

    bool        regionusesvalue = (regiontype != REGIONTYPE_EXPAND) && (regiontype != REGIONTYPE_DEACTIVATE);
    bool        operationusesvalue = (operation == OPERATION_UNION) || (operation == OPERATION_COPY);

    if (regiontype == REGIONTYPE_FILL)
        regionusesvalue = false;

    // Disable the region type switcher
    int changed = 0;
    changed += enableParm("boundgroup", has_bounds);
    changed += enableParm("usevdb", has_bounds);
#if UT_VERSION_INT >= 0x13050000        // 19.5 or later
    bool usevdb = evalInt("usevdb", 0, 0.0f);
    bool usehull = evalInt("usehull", 0, 0.0f);
    changed += enableParm("usehull", has_bounds && !usevdb);
    changed += enableParm("boundptgroup", has_bounds && !usevdb && usehull);
    changed += enableParm("worldoffset", has_bounds && !usevdb && usehull);
    changed += enableParm("voxeloffset", has_bounds && !usevdb && usehull);
#endif

    changed += enableParm("operation", regionusesvalue);
    // Only union supports writing values.
    changed += enableParm("setvalue", regionusesvalue && operationusesvalue);
    changed += enableParm("value", regionusesvalue && operationusesvalue && evalInt("setvalue", 0, 0.0));
    changed += enableParm("tolerance", (evalInt("prune", 0, 0.0f) != 0));
    return changed > 0;
}

SOP_VDBActivate::SOP_VDBActivate(OP_Network *net,
                                 const char *name,
                                 OP_Operator *entry)
    : SOP_NodeVDB(net, name, entry)
{}

OP_Node *
SOP_VDBActivate::factory(OP_Network *net,
                               const char *name,
                               OP_Operator *entry)
{
    return new SOP_VDBActivate(net, name, entry);
}


UT_BoundingBox
SOP_VDBActivate::Cache::getWorldBBox(fpreal t)
{
    UT_Vector3D center = CENTER(t);
    UT_Vector3D size = SIZE(t);
    return UT_BoundingBox(center - 0.5*size, center + 0.5*size);
}

// Get a bounding box around the world space bbox in index space
static
openvdb::CoordBBox
sopSopToIndexBBox(UT_BoundingBoxD sop_bbox, const GEO_PrimVDB &vdb)
{
    UT_Vector3D corners[8];
    sop_bbox.getBBoxPoints(corners);

    openvdb::CoordBBox index_bbox;
    for (int i=0; i<8; i++)
    {
        int x, y, z;
        vdb.posToIndex(corners[i], x, y, z);
        openvdb::Coord coord(x,y,z);
        if (i == 0)
            index_bbox = openvdb::CoordBBox(coord, coord);
        else
            index_bbox.expand(openvdb::Coord(x, y, z));
    }
    return index_bbox;
}


template <typename GridType>
void
sopDoPrune(GridType &grid, bool doprune, double tolerance)
{
    typedef typename GridType::ValueType ValueT;

    // No matter what, axe inactive voxels.
    openvdb::tools::pruneInactive(grid.tree());
    // Optionally prune live tiles
    if (doprune) {
        OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
        const auto value = openvdb::zeroVal<ValueT>() + tolerance;
        OPENVDB_NO_TYPE_CONVERSION_WARNING_END
        grid.tree().prune(static_cast<ValueT>(value));
    }
}


template <typename GridType>
static void
sopDeactivate(GridType &grid, double tolerance)
{
    using ValueT = typename GridType::ValueType;

    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
    const auto value = openvdb::zeroVal<ValueT>() + tolerance;
    OPENVDB_NO_TYPE_CONVERSION_WARNING_END
    openvdb::tools::deactivate(grid.tree(), grid.background(), static_cast<ValueT>(value));
}

template <typename TreeT>
class sop_FillSDFOp
{
public:
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    sop_FillSDFOp(ValueT bg)
    : myNegBackground(-bg)
    {
    }

    // Process roots
    bool operator()(RootT&, size_t) const
    { return true; }

    // Process internal nodes.
    template <typename NodeT>
    bool operator()(NodeT &node, size_t idx) const
    {
        for (auto iter = node.beginChildAll(); iter; ++iter)
        {
            typename NodeT::ChildNodeType    *child;
            ValueT               value;
            if (iter.getItem(iter.pos(), child, value))
            {
                // Dense tile, recurse.
            }
            else
            {
                UT_ASSERT(!child);

                // Constant tile
                // If value is outside, we are unchanged.
                if (value >= 0)
                    continue;

                // Inside, activate!
                node.addTile(iter.pos(), myNegBackground, /*active=*/true);
            }
        }

        // Proceed to sub tiles...
        return true;
    }

    bool operator()(LeafT &node, size_t idx) const
    {
        ValueT         dist;

        for (openvdb::Index offset = 0; offset < LeafT::NUM_VALUES; offset++)
        {
            if (!node.probeValue(offset, dist))
            {
                // Inactive voxel, if inside set to background
                if (dist < 0)
                {
                    node.setValueOn(offset, myNegBackground);
                }
            }
        }
        return true;
    }

private:
    ValueT      myNegBackground;
};


template <typename GridType>
static void
sopFillSDF(GridType &grid, int dummy)
{
    using TreeT = typename GridType::TreeType;

    sop_FillSDFOp<TreeT> gridop(grid.background());
    openvdb::tree::DynamicNodeManager<TreeT> nodes(grid.tree());
    nodes.foreachTopDown(gridop);
}

#if UT_VERSION_INT >= 0x13050000        // 19.5 or later
template <typename TreeT>
class sop_ConvexHullOp
{
public:
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename TreeT::ValueType;

    sop_ConvexHullOp(const GU_ConvexHullHalfPlanesF &hull,
                    GEO_PrimVDB::ActivateOperation op,
                    ValueT value,
                    ValueT bg)
    : myHull(hull)
    , myOp(op)
    , myValue(value)
    , myBackground(bg)
    {
    }

    // Process roots
    bool operator()(RootT&, size_t) const
    { return true; }

    // Process internal nodes.
    template <typename NodeT>
    bool operator()(NodeT &node, size_t idx) const
    {
        using ChildT = typename NodeT::ChildNodeType;

        // Check if entire node is rejected
        auto coordbbox = node.getNodeBoundingBox();
        auto bbox = UTvdbConvert(coordbbox);
        if (myHull.contains(bbox))
        {
            // Fully inside.
            // Intersection: Do nothing.
            // Subtract: Deactivate
            // Union: Activate
            // Copy: Activate
            switch (myOp)
            {
                case GEO_PrimVDB::ACTIVATE_INTERSECT:
                    break;
                case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                    node = NodeT(node.origin(), myBackground, /*active=*/false);
                    break;
                case GEO_PrimVDB::ACTIVATE_UNION:
                case GEO_PrimVDB::ACTIVATE_COPY:
                    node = NodeT(node.origin(), myValue, /*active=*/true);
                    break;
            }
            return false;
        }

        if (myHull.excludes(bbox))
        {
            // Fully outside.
            // Intersection: Deactivate & outside
            // Copy: Deactivate & outside
            // Difference: Do nothing
            // Union: Do nothing
            switch (myOp)
            {
                case GEO_PrimVDB::ACTIVATE_COPY:
                case GEO_PrimVDB::ACTIVATE_INTERSECT:
                    node = NodeT(node.origin(), myBackground, /*active=*/false);
                    break;
                case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                case GEO_PrimVDB::ACTIVATE_UNION:
                    break;
            }
            return false;
        }

        for (auto iter = node.beginChildAll(); iter; ++iter)
        {
            typename NodeT::ChildNodeType    *child;
            ValueT               value;

            iter.getItem(iter.pos(), child, value);

            if (!child) // Tile
            {
                // We may have nothing to do with this tile if our operation
                // cannot change it.
                bool    isactive = node.isValueMaskOn(iter.pos());
                switch (myOp)
                {
                    case GEO_PrimVDB::ACTIVATE_INTERSECT:
                    case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                        // Inactive tiles cannot change.
                        if (!isactive)
                            continue;
                        break;
                    case GEO_PrimVDB::ACTIVATE_UNION:
                        // Active tiles can still change as we have
                        // to set the value.
                        if (isactive && value == myValue)
                            continue;
                        break;
                    case GEO_PrimVDB::ACTIVATE_COPY:
                        // All tiles may change under copy.
                        break;
                }
            }

            // Check for culling of the entire child, whether it is a tile
            // or a dense node.
            const openvdb::Coord xyz(node.offsetToGlobalCoord(iter.pos()));
            openvdb::CoordBBox vdbtilebbox(xyz, xyz.offsetBy(ChildT::DIM-1));
            auto tilebbox = UTvdbConvert(vdbtilebbox);

            if (myHull.contains(tilebbox))
            {
                // Fully inside.
                // Intersection: Do nothing.
                // Subtract: Deactivate
                // Union: Activate
                // Copy: Activate
                switch (myOp)
                {
                    case GEO_PrimVDB::ACTIVATE_INTERSECT:
                        break;
                    case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                        node.addTile(iter.pos(), myBackground, /*active*/false);
                        break;
                    case GEO_PrimVDB::ACTIVATE_UNION:
                    case GEO_PrimVDB::ACTIVATE_COPY:
                        node.addTile(iter.pos(), myValue, /*active*/true);
                        break;
                }
                continue;
            }

            if (myHull.excludes(tilebbox))
            {
                // Fully outside.
                // Intersection: Deactivate & outside
                // Copy: Deactivate & outside
                // Difference: Do nothing
                // Union: Do nothing
                switch (myOp)
                {
                    case GEO_PrimVDB::ACTIVATE_COPY:
                    case GEO_PrimVDB::ACTIVATE_INTERSECT:
                        node.addTile(iter.pos(), myBackground, /*active*/false);
                        break;
                    case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                    case GEO_PrimVDB::ACTIVATE_UNION:
                        break;
                }
                continue;
            }

            // We've hit none of the early exit, so we want to process
            // this tile.  But if it is a constant tile, we have to
            // densify it first.
            if (!child)
            {
                // We are in the grey-zone so have to densify.
                auto *newchild = new ChildT(xyz, value, node.isValueMaskOn(iter.pos()));
                node.addChild(newchild);
            }
        }

        // Proceed to sub tiles...
        return true;
    }

    template <GEO_PrimVDB::ActivateOperation Op>
    void doLeaf(const UT_Array<int> &activeplanes, LeafT &node, const openvdb::CoordBBox &coordbbox) const
    {
        openvdb::Index offset = 0;
        for (int x = coordbbox.min().x(); x <= coordbbox.max().x(); x++)
            for (int y = coordbbox.min().y(); y <= coordbbox.max().y(); y++)
                for (int z = coordbbox.min().z(); z <= coordbbox.max().z(); z++)
                {
                    float dist = myHull.distance(activeplanes, UT_Vector3(x, y, z), 0.001);

                    if (dist < 0)
                    {
                        // Inside.
                        switch (Op)
                        {
                            case GEO_PrimVDB::ACTIVATE_INTERSECT:
                                break;
                            case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                                node.setValueOff(offset, myBackground);
                                break;
                            case GEO_PrimVDB::ACTIVATE_COPY:
                            case GEO_PrimVDB::ACTIVATE_UNION:
                                node.setValueOn(offset, myValue);
                                break;
                        }
                    }
                    else
                    {
                        // Outside.
                        switch (Op)
                        {
                            case GEO_PrimVDB::ACTIVATE_UNION:
                            case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                                break;
                            case GEO_PrimVDB::ACTIVATE_COPY:
                            case GEO_PrimVDB::ACTIVATE_INTERSECT:
                                node.setValueOff(offset, myBackground);
                                break;
                        }
                    }
                    offset++;
                }
    }


    bool operator()(LeafT &node, size_t idx) const
    {
        auto coordbbox = node.getNodeBoundingBox();
        auto bbox = UTvdbConvert(coordbbox);
        if (myHull.contains(bbox))
        {
            // Fully inside.
            // Union: Activate & Inside
            // Copy: Activate & Inside
            // Intersection: Do nothing.
            // Difference: Deactivate & Outside.
            switch (myOp)
            {
                case GEO_PrimVDB::ACTIVATE_INTERSECT:
                    break;
                case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                    node.fill(myBackground, /*active=*/false);
                    break;
                case GEO_PrimVDB::ACTIVATE_UNION:
                case GEO_PrimVDB::ACTIVATE_COPY:
                    node.fill(myValue, /*active=*/true);
                    break;
            }
            return false;
        }

        if (myHull.excludes(bbox))
        {
            // Fully outside.
            // Intersection: Deactivate & outside
            // Copy: Deactivate & outside
            // Difference: Do nothing
            // Union: Do nothing
            switch (myOp)
            {
                case GEO_PrimVDB::ACTIVATE_COPY:
                case GEO_PrimVDB::ACTIVATE_INTERSECT:
                    node.fill(myBackground, /*active=*/false);
                    break;
                case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                case GEO_PrimVDB::ACTIVATE_UNION:
                    break;
            }
            return false;
        }

        UT_SmallArray<int>      activeplanes;
        bool inside = myHull.findActivePlanes(
                activeplanes,
                UT_Vector3( 0.5 * (coordbbox.max().x() + coordbbox.min().x()),
                            0.5 * (coordbbox.max().y() + coordbbox.min().y()),
                            0.5 * (coordbbox.max().z() + coordbbox.min().z()) ),
                sqrt(3) * (LeafT::DIM / 2) * 1.01);

        // Usually if  the active planes aren't set we should have
        // culled with the first earlier bbox tests, but in case things
        // fall in the cracks...
        if (!activeplanes.entries())
        {
            switch (myOp)
            {
                case GEO_PrimVDB::ACTIVATE_INTERSECT:
                    if (!inside)
                        node.fill(myBackground, /*active=*/false);
                    break;
                case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                    if (inside)
                        node.fill(myBackground, /*active=*/false);
                    break;
                case GEO_PrimVDB::ACTIVATE_UNION:
                    if (inside)
                        node.fill(myValue, /*active=*/true);
                    break;
                case GEO_PrimVDB::ACTIVATE_COPY:
                    node.fill(inside ? myValue : myBackground, /*active=*/inside);
                    break;
            }
            return false;
        }

        switch (myOp)
        {
            case GEO_PrimVDB::ACTIVATE_COPY:
                doLeaf<GEO_PrimVDB::ACTIVATE_COPY>(activeplanes, node, coordbbox);
                break;
            case GEO_PrimVDB::ACTIVATE_UNION:
                doLeaf<GEO_PrimVDB::ACTIVATE_UNION>(activeplanes, node, coordbbox);
                break;
            case GEO_PrimVDB::ACTIVATE_INTERSECT:
                doLeaf<GEO_PrimVDB::ACTIVATE_INTERSECT>(activeplanes, node, coordbbox);
                break;
            case GEO_PrimVDB::ACTIVATE_SUBTRACT:
                doLeaf<GEO_PrimVDB::ACTIVATE_SUBTRACT>(activeplanes, node, coordbbox);
                break;
        }

        return true;
    }

private:
    GEO_PrimVDB::ActivateOperation myOp;
    const GU_ConvexHullHalfPlanesF &myHull;
    ValueT      myValue, myBackground;
};

template <typename GridType>
static void
sopConvexHull(GridType& grid, const GU_ConvexHullHalfPlanesF &hull,
        GEO_PrimVDB::ActivateOperation op,
        bool setvalue, fpreal floatvalue,
        const UT_BoundingBoxF &hullbbox)
{
    using ValueT = typename GridType::ValueType;
    using TreeT = typename GridType::TreeType;
    using RootT = typename TreeT::RootNodeType;

    if (!setvalue && op == GEO_PrimVDB::ACTIVATE_COPY)
    {
        sopConvexHull(grid, hull, GEO_PrimVDB::ACTIVATE_INTERSECT,
                setvalue, floatvalue, hullbbox);
        sopConvexHull(grid, hull, GEO_PrimVDB::ACTIVATE_UNION,
                setvalue, floatvalue, hullbbox);
        return;
    }

    if (!setvalue && op == GEO_PrimVDB::ACTIVATE_UNION)
    {
        openvdb::MaskGrid       mask(false);
        sopConvexHull(mask, hull, op, true, 1, hullbbox);
        grid.topologyUnion(mask);
        return;
    }

    // We need to touch all the top-level root nodes that might be hit.
    if (op == GEO_PrimVDB::ACTIVATE_COPY || op == GEO_PrimVDB::ACTIVATE_UNION)
    {
        const int shift = RootT::ChildNodeType::TOTAL;

        for (int x = (int(SYSfloor(hullbbox.xmin())) >> shift);
                 x <= (int(SYSceil(hullbbox.xmax())) >> shift);
                 x++)
        {
            for (int y = (int(SYSfloor(hullbbox.ymin())) >> shift);
                     y <= (int(SYSceil(hullbbox.ymax())) >> shift);
                     y++)
            {
                for (int z = (int(SYSfloor(hullbbox.zmin())) >> shift);
                         z <= (int(SYSceil(hullbbox.zmax())) >> shift);
                         z++)
                {
                    openvdb::Coord      idx(x << shift, y << shift, z << shift);
                    if (grid.tree().root().getValueDepth(idx) < 0)
                    {
                        // This root node is implicit background so
                        // need to densify.
                        // We are in the grey-zone so have to densify.
                        auto *child = new typename RootT::ChildNodeType(idx, grid.background(), false);
                        grid.tree().root().addChild(child);
                    }

                }
            }
        }
    }


    // This ugly construction avoids compiler warnings when,
    // for example, initializing an openvdb::Vec3i with a double.
    ValueT      value = ValueT(openvdb::zeroVal<ValueT>() + floatvalue);

    sop_ConvexHullOp<TreeT> gridop(hull, op, value, grid.background());
    openvdb::tree::DynamicNodeManager<TreeT> nodes(grid.tree());
    nodes.foreachTopDown(gridop);
}
#endif


template <typename GridType>
static void
sopDilateVoxels(GridType& grid, exint count, openvdb::tools::NearestNeighbors nn)
{
    openvdb::tools::dilateActiveValues(grid.tree(), static_cast<int>(count), nn);
}

template <typename GridType>
static void
sopErodeVoxels(GridType& grid, exint count, openvdb::tools::NearestNeighbors nn)
{
    openvdb::tools::erodeActiveValues(grid.tree(), static_cast<int>(count), nn);
    if (grid.getGridClass() == openvdb::GRID_LEVEL_SET) {
        openvdb::tools::pruneLevelSet(grid.tree());
    }
    else {
        openvdb::tools::pruneInactive(grid.tree());
    }
}

// Based on mode the parameters imply, get an index space bounds for this vdb
openvdb::CoordBBox
SOP_VDBActivate::Cache::getIndexSpaceBounds(OP_Context &context,
                                     const GEO_PrimVDB &vdb)
{
    fpreal t = context.getTime();

    using namespace openvdb;
    CoordBBox index_bbox;
    // Get the bbox
    switch(REGIONTYPE(t))
    {
        case REGIONTYPE_POSITION: // world
            index_bbox = sopSopToIndexBBox(getWorldBBox(t), vdb);
            break;
        case REGIONTYPE_VOXEL: // index
            index_bbox = CoordBBox(MINPOS(t), MAXPOS(t));
            break;
        default:
            UT_ASSERT("Invalid region type" == nullptr);
            break;
    }
    return index_bbox;
}

GEO_PrimVDB::ActivateOperation
sopXlateOperation(OPERATION_NAMES operation)
{
    switch (operation)
    {
        case OPERATION_UNION:
            return GEO_PrimVDB::ACTIVATE_UNION;
        case OPERATION_INTERSECT:
            return GEO_PrimVDB::ACTIVATE_INTERSECT;
        case OPERATION_SUBTRACT:
            return GEO_PrimVDB::ACTIVATE_SUBTRACT;
        case OPERATION_COPY:
            return GEO_PrimVDB::ACTIVATE_COPY;
    }

    UT_ASSERT("Unhandled operation" == nullptr);
    return GEO_PrimVDB::ACTIVATE_UNION;
}

OP_ERROR
SOP_VDBActivate::Cache::cookVDBSop(OP_Context &context)
{
    using namespace openvdb;
    using namespace openvdb::math;
    using namespace openvdb::tools;

    try
    {
        fpreal t = context.getTime();
        UT_Interrupt    *boss = UTgetInterrupt();


        // Get the group
        UT_String group_name;
        evalString(group_name, "group", 0, t);
        const GA_PrimitiveGroup* group = 0;
        if (group_name.isstring())
        {
            bool        success;
            group = gop.parseOrderedPrimitiveDetached((const char *) group_name, gdp, false, success);
        }

        // A group was specified but not found
        if (!group && group_name.isstring())
        {
            addError(SOP_ERR_BADGROUP, group_name);
            return error();
        }

        UT_AutoInterrupt progress("Activating VDB grids");

        // For each primitive in the group, go through the primitives in the
        // second input's group and
        GEO_Primitive *prim;
        GA_FOR_ALL_GROUP_PRIMITIVES(gdp, group, prim)
        {
            if (!(prim->getPrimitiveId() & GEO_PrimTypeCompat::GEOPRIMVDB))
                continue;

            GEO_PrimVDB *vdb = UTverify_cast<GEO_PrimVDB *>(prim);
            vdb->makeGridUnique();

            // Apply the operation for all VDB primitives on input 2
            const GU_Detail *bounds_src = inputGeo(1, context);

            switch (REGIONTYPE(t))
            {
                case REGIONTYPE_REFERENCE:                      // Second input!
                {
                    if (bounds_src)
                    {
#if UT_VERSION_INT >= 0x13050000        // 19.5 or later
                        UT_StringHolder      boundgroupname, boundptgroupname;
#else
                        UT_String            boundgroupname;
#endif

                        evalString(boundgroupname, "boundgroup", 0, t);
                        const GA_PrimitiveGroup *boundgroup = 0;

                        bool                usevdb = evalInt("usevdb", 0, t);
#if UT_VERSION_INT >= 0x13050000        // 19.5 or later
                        bool                usehull = evalInt("usehull", 0, t);
#else
                        bool                usehull = false;
#endif

                        if ((usevdb || !usehull) && boundgroupname.isstring())
                        {
                            bool        success;
                            boundgroup = gop.parseOrderedPrimitiveDetached((const char *) boundgroupname, bounds_src, true, success);
                            if (!success)
                                addWarning(SOP_ERR_BADGROUP, boundgroupname);
                        }

#if UT_VERSION_INT >= 0x13050000        // 19.5 or later
                        evalString(boundptgroupname, "boundptgroup", 0, t);
                        const GA_PointGroup *boundptgroup = 0;
                        if ((!usevdb && usehull) && boundptgroupname.isstring())
                        {
                            bool        success;
                            boundptgroup = gop.parsePointDetached((const char *) boundptgroupname, bounds_src, true, success);
                            if (!success)
                                addWarning(SOP_ERR_BADGROUP, boundptgroupname);
                        }
#endif

                        if (evalInt("usevdb", 0, t))
                        {
                            bool                foundvdb = false;
                            const GEO_Primitive *input_prim;
                            GA_FOR_ALL_GROUP_PRIMITIVES(bounds_src, boundgroup, input_prim)
                            {
                                if (!(input_prim->getPrimitiveId() & GEO_PrimTypeCompat::GEOPRIMVDB))
                                    break;
                                const GEO_PrimVDB *input_vdb =
                                    UTverify_cast<const GEO_PrimVDB *>(input_prim);

                                vdb->activateByVDB(input_vdb,
                                            sopXlateOperation(OPERATION(t)),
                                            evalInt("setvalue", 0, t),
                                            evalFloat("value", 0, t));

                                foundvdb = true;

                            }

                            if (!foundvdb)
                            {
                                addWarning(SOP_MESSAGE, "No VDB primitives found in second input");
                            }
                        }
#if UT_VERSION_INT >= 0x13050000        // 19.5 or later
                        else if (evalInt("usehull", 0, t))
                        {
                            UT_Vector3DArray    poses;
                            UT_BoundingBoxF     hullbbox;
                            float               voxelsize = vdb->getVoxelSize().maxComponent();

                            GEO_PrimVolumeXform index_xform = vdb->getIndexSpaceTransform();
                            GA_Offset           ptoff;
                            hullbbox.initBounds();
                            GA_FOR_ALL_GROUP_PTOFF(bounds_src, boundptgroup, ptoff)
                            {
                                UT_Vector3D     pos = bounds_src->getPos3D(ptoff);
                                pos = index_xform.toVoxelSpace(pos) - 0.5;
                                poses.append(pos);
                                hullbbox.enlargeBounds(
                                    pos.x(),
                                    pos.y(),
                                    pos.z());
                            }
                            GU_ConvexHullHalfPlanesF    hull(poses);
                            hull.applyOffset(evalFloat("worldoffset", 0, t) / voxelsize + evalFloat("voxeloffset", 0, t));

                            UTvdbCallAllTopology(vdb->getStorageType(),
                                             sopConvexHull,
                                             vdb->getGrid(),
                                             hull,
                                             sopXlateOperation(OPERATION(t)),
                                             evalInt("setvalue", 0, t),
                                             evalFloat("value", 0, t),
                                             hullbbox);
                        }
#endif
                        else
                        {
                            // Activate by bounding box.
                            UT_BoundingBox              bbox;
                            bounds_src->getBBox(&bbox, boundgroup);
                            vdb->activateIndexBBox(sopSopToIndexBBox(bbox, *vdb), sopXlateOperation(OPERATION(t)), evalInt("setvalue", 0, t), evalFloat("value", 0, t));
                        }
                    }
                    else
                    {
                        addError(SOP_MESSAGE, "Not enough inputs.");
                    }
                    break;
                }

                case REGIONTYPE_POSITION:               // World space
                case REGIONTYPE_VOXEL:          // Coord Space
                {
                    vdb->activateIndexBBox(getIndexSpaceBounds(context, *vdb), sopXlateOperation(OPERATION(t)), evalInt("setvalue", 0, t), evalFloat("value", 0, t));
                    break;
                }

                case REGIONTYPE_EXPAND:         // Dilate
                {
                    exint dilatevoxels = evalInt("expand", 0, t);
                    exint dilatedist = static_cast<exint>(
                            SYSceil(sqrt(3.0)
                                    * SYSsafediv(
                                            evalFloat("expanddist", 0, t),
                                            vdb->getVoxelDiameter())));

                    NearestNeighbors nn = NN_FACE;
                    const auto str = evalStdString("expansionpattern", t);
                    if (str == "faceedge")
                        nn = NN_FACE_EDGE;
                    else if (str == "faceedgevertex")
                        nn = NN_FACE_EDGE_VERTEX;

                    exint maxdilate = SYSmax(dilatevoxels, dilatedist);
                    if (maxdilate > 0)
                    {
                        if (boss->opInterrupt())
                            break;

                        UTvdbCallAllTopology(vdb->getStorageType(),
                                         sopDilateVoxels,
                                         vdb->getGrid(), maxdilate, nn);
                    }

                    exint mindilate = SYSmin(dilatevoxels, dilatedist);
                    if (mindilate < 0)
                    {
                        if (boss->opInterrupt())
                            break;
                        UTvdbCallAllTopology(vdb->getStorageType(),
                                         sopErodeVoxels,
                                         vdb->getGrid(), -mindilate, nn);
                    }
                    if (mindilate < 0 && maxdilate > 0)
                    {
                        addWarning(SOP_MESSAGE, "Conflicting signs in Voxel/Worldspace dilation request.  Applying both, which may not be expected.");
                    }
                    break;
                }

                case REGIONTYPE_DEACTIVATE:             // Deactivate
                {
                    if (boss->opInterrupt())
                        break;
                    UTvdbCallAllTopology(vdb->getStorageType(),
                                     sopDeactivate,
                                     vdb->getGrid(), evalFloat("bgtolerance", 0, t));
                    break;
                }

                case REGIONTYPE_FILL:           // Fill interior of SDF.
                {
                    if (boss->opInterrupt())
                        break;

                    UTvdbCallRealType(vdb->getStorageType(),
                                    sopFillSDF,
                                    vdb->getGrid(), 1);

                    break;
                }

            }
            UTvdbCallAllTopology(vdb->getStorageType(),
                            sopDoPrune,
                            vdb->getGrid(),
                            evalInt("prune", 0, t),
                            evalFloat("tolerance", 0, t));
        }
    }
    catch (std::exception& e)
    {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

const char *
SOP_VDBActivate::inputLabel(unsigned index) const
{
    switch (index)
    {
        case 0:
            return "VDBs to activate";
        case 1:
            return "Region to activate";
    }
    return NULL;
}

int
SOP_VDBActivate::isRefInput(unsigned i) const
{
    switch (i)
    {
        case 0:
            return false;
        case 1:
            return true;
        default:
            return true;
    }
}
