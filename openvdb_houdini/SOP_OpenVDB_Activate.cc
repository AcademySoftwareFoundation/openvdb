// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file SOP_OpenVDB_Activate.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Activate VDBs according to various rules

#if 1
  #include <houdini_utils/ParmFactory.h>
  #include <openvdb_houdini/Utils.h>
  #include <openvdb_houdini/SOP_NodeVDB.h>
#else
  #include "Utils.h"
  #include "ParmFactory.h"
  #include "SOP_NodeVDB.h"
#endif

#include <GU/GU_PrimVDB.h>
#include <OP/OP_Node.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Parm.h>

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
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

    static OP_Node       *myConstructor(OP_Network*, const char *, OP_Operator*);

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
            { return openvdb::Coord(evalInt("min", 0, t),
                                    evalInt("min", 1, t),
                                    evalInt("min", 2, t)); }
        openvdb::Coord MAXPOS(fpreal t)
            { return openvdb::Coord(evalInt("max", 0, t),
                                    evalInt("max", 1, t),
                                    evalInt("max", 2, t)); }

        int         EXPAND(double t)
        { return evalInt("expand", 0, t); }
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
              .setChoiceList(&hutil::PrimGroupMenuInput1));

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
              .setChoiceListItems(PRM_CHOICELIST_SINGLE, operations));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "setvalue", "Write Value")
                .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
                .setDefault(PRMoneDefaults));
    parms.add(hutil::ParmFactory(PRM_FLT, "value", "Value")
                .setDefault(PRMoneDefaults));

    // Match REGIONTYPE
    std::vector<PRM_Default> items;
    items.push_back(PRM_Default(2, "Position"));
    items.push_back(PRM_Default(2, "Voxel"));
    items.push_back(PRM_Default(1, "Expand"));
    items.push_back(PRM_Default(2, "Reference"));
    items.push_back(PRM_Default(0, "Deactivate"));
    items.push_back(PRM_Default(0, "Fill SDF"));
    parms.add(hutil::ParmFactory(PRM_SWITCHER_EXCLUSIVE, "regiontype", "Region Type")
              .setDefault(items)
              .setVectorSize(items.size()));

    parms.add(hutil::ParmFactory(PRM_XYZ, "center", "Center")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults));
    parms.add(hutil::ParmFactory(PRM_XYZ, "size", "Size")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults));

    parms.add(hutil::ParmFactory(PRM_XYZ, "min", "Min")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults));
    parms.add(hutil::ParmFactory(PRM_XYZ, "max", "Max")
              .setVectorSize(3)
              .setDefault(PRMzeroDefaults));

    parms.add(hutil::ParmFactory(PRM_INT, "expand", "Voxels to Expand")
                .setDefault(PRMoneDefaults)
                .setRange(PRM_RANGE_FREE, -5, PRM_RANGE_FREE, 5));

    parms.add(hutil::ParmFactory(PRM_STRING, "boundgroup", "Bound Group")
              .setChoiceList(&hutil::PrimGroupMenuInput2));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "usevdb", "Activate Using VDBs")
                .setDefault(PRMzeroDefaults));

    // Prune toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "prune", "Prune Tolerance")
        .setDefault(PRMoneDefaults)
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN));

    // Pruning tolerance slider
    parms.add(hutil::ParmFactory(
        PRM_FLT_J, "tolerance", "Prune Tolerance")
        .setDefault(PRMzeroDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1));

    hvdb::OpenVDBOpFactory("VDB Activate",
        SOP_VDBActivate::myConstructor, parms, *table)
        .addInput("VDBs to Activate")
        .addOptionalInput("Bounds to Activate")
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_VDBActivate::Cache; })
        ;
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
SOP_VDBActivate::myConstructor(OP_Network *net,
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
    if (doprune)
        grid.tree().prune(ValueT(openvdb::zeroVal<ValueT>() + tolerance));
}

// The result of the union of active regions goes into grid_a
template <typename GridType>
static void
sopDeactivate(GridType &grid, int dummy)
{
    typename GridType::Accessor         access = grid.getAccessor();
    typedef typename GridType::ValueType ValueT;

    ValueT              background = grid.background();
    ValueT              value;
    UT_Interrupt        *boss = UTgetInterrupt();

    for (typename GridType::ValueOnCIter
         iter = grid.cbeginValueOn(); iter; ++iter)
    {
        if (boss->opInterrupt())
            break;
        openvdb::CoordBBox bbox = iter.getBoundingBox();
        for (int k=bbox.min().z(); k<=bbox.max().z(); k++)
        {
            for (int j=bbox.min().y(); j<=bbox.max().y(); j++)
            {
                for (int i=bbox.min().x(); i<=bbox.max().x(); i++)
                {
                    openvdb::Coord coord(i, j, k);

                    // If it is on...
                    if (access.probeValue(coord, value))
                    {
                        if (value == background)
                        {
                            access.setValueOff(coord);
                        }
                    }
                }
            }
        }
    }
}

template <typename GridType>
static void
sopFillSDF(GridType &grid, int dummy)
{
    typename GridType::Accessor         access = grid.getAccessor();
    typedef typename GridType::ValueType ValueT;

    ValueT              value;
    UT_Interrupt        *boss = UTgetInterrupt();
    ValueT              background = grid.background();

    for (typename GridType::ValueOffCIter
         iter = grid.cbeginValueOff(); iter; ++iter)
    {
        if (boss->opInterrupt())
            break;

        openvdb::CoordBBox bbox = iter.getBoundingBox();

        // Assuming the SDF is at all well-formed, any crossing
        // of sign must have a crossing of inactive->active.
        openvdb::Coord coord(bbox.min().x(), bbox.min().y(), bbox.min().z());

        // We do not care about the active state as it is hopefully inactive
        access.probeValue(coord, value);

        if (value < 0)
        {
            // Fill the region to negative background.
            grid.fill(bbox, -background, /*active=*/true);
        }
    }
}

template <typename GridType>
static void
sopDilateVoxels(GridType& grid, int count)
{
    openvdb::tools::dilateVoxels(grid.tree(), count);
}

template <typename GridType>
static void
sopErodeVoxels(GridType& grid, int count)
{
    openvdb::tools::erodeVoxels(grid.tree(), count);
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
            UT_ASSERT(!"Invalid region type");
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

    UT_ASSERT(!"Unhandled operation");
    return GEO_PrimVDB::ACTIVATE_UNION;
}

OP_ERROR
SOP_VDBActivate::Cache::cookVDBSop(OP_Context &context)
{
    using namespace openvdb;
    using namespace openvdb::math;

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
                break;

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
                        UT_String       boundgroupname;

                        evalString(boundgroupname, "boundgroup", 0, t);
                        const GA_PrimitiveGroup *boundgroup = 0;

                        if (boundgroupname.isstring())
                        {
                            bool        success;
                            boundgroup = gop.parseOrderedPrimitiveDetached((const char *) boundgroupname, bounds_src, true, success);
                            if (!success)
                                addWarning(SOP_ERR_BADGROUP, boundgroupname);
                        }

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
                    int         dilation = evalInt("expand", 0, t);

                    if (dilation > 0)
                    {
                        if (boss->opInterrupt())
                            break;
                        UTvdbCallAllTopology(vdb->getStorageType(),
                                         sopDilateVoxels,
                                         vdb->getGrid(), dilation);
                    }

                    if (dilation < 0)
                    {
                        if (boss->opInterrupt())
                            break;
                        UTvdbCallAllTopology(vdb->getStorageType(),
                                         sopErodeVoxels,
                                         vdb->getGrid(), -dilation);
                    }
                    break;
                }

                case REGIONTYPE_DEACTIVATE:             // Deactivate
                {
                    if (boss->opInterrupt())
                        break;
                    UTvdbCallAllTopology(vdb->getStorageType(),
                                     sopDeactivate,
                                     vdb->getGrid(), 1);
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
