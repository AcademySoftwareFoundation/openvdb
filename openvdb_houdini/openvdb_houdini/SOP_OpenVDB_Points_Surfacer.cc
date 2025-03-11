// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @author Nick Avramoussis
///
/// @file SOP_OpenVDB_Points_Surfacer.cpp
///
/// @brief Surface VDB Points into a VDB Level Set using a variety of methods

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/PointUtils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/points/IndexFilter.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointStatistics.h>
#include <openvdb/points/PointRasterizeSDF.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/util/NullInterrupter.h>

#include <CH/CH_Manager.h>
#include <PRM/PRM_Parm.h>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;

using SupportedGridT =
    openvdb::TypeList<bool, int32_t, int64_t, float, double,
        openvdb::Vec3f, openvdb::Vec3d, openvdb::Vec3i>;

class SOP_VDBPointsSurfacer: public openvdb_houdini::SOP_NodeVDB
{
public:
    SOP_VDBPointsSurfacer(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_VDBPointsSurfacer() {}
    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);
protected:
    virtual OP_ERROR cookVDBSop(OP_Context&);
    virtual bool updateParmsFlags();
};


////////////////////////////////////////

namespace
{

enum SurfaceType
{
    Spheres,
    AveragePositions
};

}

////////////////////////////////////////

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    // INPUT PARMS

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setChoiceList(&hutil::PrimGroupMenu)
        .setTooltip("Specify a subset of the input point VDBs to surface.")
        .setDocumentation(
            "A subset of the input VDB Points primitives to be processed"));

    // SURFACE PARMS
    parms.add(hutil::ParmFactory(PRM_STRING, "surfacevdbname", "Output Surface VDB")
        .setDefault("surface")
        .setTooltip("The name of the surface VDB to be created."));

    parms.add(hutil::ParmFactory(PRM_STRING, "referencegroup", "Reference VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip(
            "Give the output VDB the same orientation and voxel size as the selected VDB."));

    parms.add(hutil::ParmFactory(PRM_STRING, "vdbpointsgroups", "VDB Points Groups")
        .setChoiceList(&hvdb::VDBPointsGroupMenuInput1)
        .setDefault("")
        .setHelpText("Specify VDB Points Groups to use. (Default is all groups)"));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "keep", "Keep VDB Points")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, VDB point grids will not be removed from the geometry stream."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1e-5, PRM_RANGE_UI, 5)
        .setTooltip("Uniform voxel edge length in world units.  "
            "Decrease the voxel size to increase the volume resolution."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "halfbandvoxels", "Half-Band Voxels")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setTooltip("Half the width of the narrow band in voxel units.  "
            "The default value 3 is recommended for level set volumes."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "rebuildlevelset", "Rebuild Level Set")
        .setDefault(PRMoneDefaults)
        .setTooltip("Rebuild the level set after running the surfacing algorithm"));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR,"sepOutput", ""));

    parms.add(hutil::ParmFactory(PRM_STRING, "radiusattribute", "Particle Radius Attribute")
        .setDefault("pscale")
        .setTooltip("The point attribute representing the particle radius,"
                    " if the attribute does not exist, a uniform value of 1 is assumed."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "particleradius", "Particle Radius Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0)
        .setTooltip("A multiplier on the radius of the particles to be surfaced,"
                    " if no radius attribute is supplied this becomes the particle radius."));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR,"sepRadius", ""));

    parms.add(hutil::ParmFactory(PRM_ORD, "mode", "Mode")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
              "spherical",          "Spherical",
              "averagepositions",   "Average Positions"
            })
        .setDefault(PRMoneDefaults)
        .setDocumentation("The method used to create a surface from the points.\n\n"
                    "*Spherical* - stamps spheres into a signed distance field."
                    " This is very fast and gives a good approximation of the surface suitable for simulation or for use with further post-processing.\n\n"
                    "*Average Positions* - uses a weighted-average method to create smooth surfaces from the points."
                    " This is good for slow moving and viscous fluids as it gives fast smooth results but can smooth out droplets and fine-details."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "useworldspaceinfluence", "Use World Space Influence Radius")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, specify the influence radius explicitly in world space units, "
                    "otherwise is specified as a scale on the average (scaled by above) particle radius."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "influencescale", "Influence Radius Scale")
        .setDefault(PRMtwoDefaults)
        .setRange(PRM_RANGE_UI, 1.0, PRM_RANGE_UI, 4.0)
        .setTooltip("The distance at which particles interact is this value multiplied by the final average particle radius."
                    "Suggested values are around 2-4. "
                    "Values much larger than this can be very inefficient and give undesirable results."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "influenceradius", "Influence Radius")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 1.0)
        .setTooltip("The absolute world space value for the distance at which particles interact."
                    "Suggested values are of around 2-4x the average particle radius."
                    "Values much larger than this can be very inefficient and give undesirable results."));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR,"sepInfluence", ""));

    // transfer attributes

    parms.add(hutil::ParmFactory(PRM_HEADING, "transferheading", "Attribute Transfer"));

    hutil::ParmList attrParms;
    attrParms.add(hutil::ParmFactory(PRM_STRING, "name#", "Name")
        .setHelpText("Attribute name"));

    parms.add(hutil::ParmFactory(PRM_MULTITYPE_LIST, "numattr", "Number of Attributes")
        .setHelpText("The number of attributes to transfer.")
        .setMultiparms(attrParms)
        .setDefault(PRMzeroDefaults));

    hvdb::OpenVDBOpFactory("VDB Points Surfacer",
        SOP_VDBPointsSurfacer::factory, parms, *table)
        .addInput("VDB Points to surface")
        .addOptionalInput("Optional VDB grid that defines the output transform. "
            "The half-band width is matched if the input grid is a level set.")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Converts a points VDB to a levelset surface.\"\"\"\n\
\n\
@overview\n\
\n\
This node converts a points VDB to a levelset surface. It has a collection of modes\n\
to allow various methods of performing this conversion. Points can stamp a simple spherical\n\
footprint or use some more advanced methods to average (and smooth) their footprint.\n\
\n\
:tip:\n\
Convert points to a points VDB using a [OpenVDB Points Convert|Node:sop/DW_OpenVDBPointsConvert] node.\n\
");
}


bool
SOP_VDBPointsSurfacer::updateParmsFlags()
{
    bool changed = false;
    const fpreal t = CHgetEvalTime();
    const SurfaceType mode = static_cast<SurfaceType>(evalInt("mode", 0, t));

    const bool averagePositions = mode == SurfaceType::AveragePositions;
    const bool hasRefInput = this->nInputs() == 2;
    const bool requiresInfluence = averagePositions;
    const bool absoluteInfluence = static_cast<bool>(evalInt("useworldspaceinfluence", 0, t));

    changed |= enableParm("voxelsize", !hasRefInput);
    changed |= enableParm("referencegroup", hasRefInput);
    changed |= setVisibleState("useworldspaceinfluence", requiresInfluence);
    changed |= setVisibleState("sepInfluence", requiresInfluence);
    changed |= enableParm("influencescale", requiresInfluence && !absoluteInfluence);
    changed |= setVisibleState("influencescale", requiresInfluence && !absoluteInfluence);
    changed |= enableParm("influenceradius", requiresInfluence && absoluteInfluence);
    changed |= setVisibleState("influenceradius", requiresInfluence && absoluteInfluence);
    return changed;
}

////////////////////////////////////////


OP_Node*
SOP_VDBPointsSurfacer::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_VDBPointsSurfacer(net, name, op);
}


SOP_VDBPointsSurfacer::SOP_VDBPointsSurfacer(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
{}

////////////////////////////////////////

template <typename FilterT, typename ...Args>
inline openvdb::GridPtrVec raster(const Args&... args)
{
    return openvdb::points::rasterizeSpheres<
                openvdb::points::PointDataGrid,
                SupportedGridT,
                openvdb::FloatGrid,
                FilterT,
                hvdb::Interrupter>
                    (args...);
}

template <typename FilterT, typename ...Args>
inline openvdb::GridPtrVec rasterP(const Args&... args)
{
    return openvdb::points::rasterizeSpheres<
                openvdb::points::PointDataGrid,
                SupportedGridT,
                float,
                openvdb::FloatGrid,
                FilterT,
                hvdb::Interrupter>
                    (args...);
}

template <typename FilterT, typename ...Args>
inline openvdb::GridPtrVec rasterZb(const Args&... args)
{
    return openvdb::points::rasterizeSmoothSpheres<
                openvdb::points::PointDataGrid,
                SupportedGridT,
                openvdb::FloatGrid,
                FilterT,
                hvdb::Interrupter>
                    (args...);
}

template <typename FilterT, typename ...Args>
inline openvdb::GridPtrVec rasterZbP(const Args&... args)
{
    return openvdb::points::rasterizeSmoothSpheres<
                openvdb::points::PointDataGrid,
                SupportedGridT,
                float,
                openvdb::FloatGrid,
                FilterT,
                hvdb::Interrupter>
                    (args...);
}

OP_ERROR
SOP_VDBPointsSurfacer::cookVDBSop(OP_Context& context)
{
    using namespace openvdb;
    using namespace openvdb::points;

    try {
        hutil::ScopedInputLock lock(*this, context);
        OPENVDB_NO_DEPRECATION_WARNING_BEGIN
        if (duplicateSourceStealable(0, context) >= UT_ERROR_ABORT) return error();
        OPENVDB_NO_DEPRECATION_WARNING_END

        hvdb::Interrupter boss("VDB Point Surfacer");

        const fpreal time = context.getTime();

        math::Transform::Ptr sdfTransform;
        const GU_Detail* refGeo = inputGeo(1);
        if (refGeo) {
            // Get the first grid in the group's transform
            const GA_PrimitiveGroup *refGroup = matchGroup(*refGeo, evalStdString("referencegroup", time));

            hvdb::VdbPrimCIterator gridIter(refGeo, refGroup);

            if (gridIter) {
                sdfTransform = (*gridIter)->getGrid().transform().copy();
            } else {
                addError(SOP_MESSAGE, "Could not find a reference grid");
                return error();
            }
        }
        else {
            auto voxelSize = evalFloat("voxelsize", 0, time);
            sdfTransform = math::Transform::createLinearTransform(voxelSize);
        }

        if (!sdfTransform->isLinear()) throw std::runtime_error("Surfacing only supports uniform voxels");

        const std::string surfaceName = evalStdString("surfacevdbname", time);
        const Real halfBand = Real(evalInt("halfbandvoxels", 0, time));
        const bool keepPoints = evalInt("keep", 0, time) == 1;
        const std::string groupStr = evalStdString("group", time);
        const GA_PrimitiveGroup *group = matchGroup(*gdp, groupStr);
        const SurfaceType mode = static_cast<SurfaceType>(evalInt("mode", 0, time));
        const bool absoluteInfluence = static_cast<bool>(evalInt("useworldspaceinfluence", 0, time));
        const Real influenceRadius = Real(evalFloat("influenceradius", 0, time));
        const Real influenceScale = Real(evalFloat("influencescale", 0, time));
        const std::string radiusAttributeName = evalStdString("radiusattribute", time);
        const Real radiusScale = Real(evalFloat("particleradius", 0, time));
        const bool rebuildLevelSet = static_cast<bool>(evalInt("rebuildlevelset", 0, time));

        // check to see if we have a point data grid
        bool hasPoints = false;
        // prims to remove if keepPoints is false
        UT_Array<GEO_Primitive*> primsToDelete;

        openvdb_houdini::VdbPrimIterator vdbIt(gdp, group);

        // surface all point data grids
        for (; vdbIt; ++vdbIt) {

            GU_PrimVDB* vdbPrim = *vdbIt;

            // only process if grid is a PointDataGrid with leaves
            if (!gridConstPtrCast<PointDataGrid>(vdbPrim->getConstGridPtr())) continue;
            if (!keepPoints) primsToDelete.append(*vdbIt);
            hasPoints = true;

            vdbPrim->makeGridUnique();

            PointDataGrid::Ptr points = gridPtrCast<PointDataGrid>(vdbPrim->getGridPtr());
            const auto iter = points->constTree().cbeginLeaf();

            if (!iter) continue;
            if (boss.wasInterrupted()) break;

            const points::AttributeSet::Descriptor&
                descriptor = iter->attributeSet().descriptor();
            const bool hasPscale(iter->hasAttribute(radiusAttributeName));
            if (hasPscale && descriptor.valueType(descriptor.find(radiusAttributeName)) !=
                std::string("float")) {
                throw std::runtime_error("Wrong attribute type for attribute " + radiusAttributeName + ", expected float");
            }

            const std::string groupStr(evalStdString("vdbpointsgroups", time));
            std::vector<std::string> include, exclude;
            points::AttributeSet::Descriptor::parseNames(include, exclude, groupStr);

            // determine attributes to transfer

            const int numAttrs = int(evalInt("numattr", 0, time));
            std::vector<std::string> transferAttributes;
            transferAttributes.reserve(numAttrs);

            for(int i = 1; i < numAttrs + 1; i++) {
                UT_String attrName;
                evalStringInst("name#", &i, attrName, 0, time);
                const std::string attrNameStr = attrName.toStdString();

                // warn if attribute is missing

                if ((!attrNameStr.empty()) && descriptor.find(attrNameStr) !=
                    points::AttributeSet::INVALID_POS) {
                    transferAttributes.emplace_back(attrNameStr);
                }
                else {
                    std::string warning = "Attribute " + attrNameStr +
                        " not available for transfer to volume";
                    addWarning(SOP_MESSAGE, warning.c_str());
                }
            }

            GridPtrVec grids;

            if (mode == SurfaceType::Spheres) {
                if (exclude.empty() && include.empty()) {
                    NullFilter filter;
                    if (hasPscale) grids = rasterP<NullFilter>(*points, radiusAttributeName, transferAttributes, radiusScale, halfBand, sdfTransform, filter, &boss);
                    else           grids = raster<NullFilter>(*points, radiusScale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                }
                else if (exclude.empty() && include.size() == 1) {
                    GroupFilter filter(include.front(), iter->attributeSet());
                    if (hasPscale) grids = rasterP<GroupFilter>(*points, radiusAttributeName, transferAttributes, radiusScale, halfBand, sdfTransform, filter, &boss);
                    else           grids = raster<GroupFilter>(*points, radiusScale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                }
                else {
                    MultiGroupFilter filter(include, exclude, iter->attributeSet());
                    if (hasPscale) grids = rasterP<MultiGroupFilter>(*points, radiusAttributeName, transferAttributes, radiusScale, halfBand, sdfTransform, filter, &boss);
                    else           grids = raster<MultiGroupFilter>(*points, radiusScale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                }
            }
            else { //mode == SurfaceType::AveragePositions

                double scale;
                if (absoluteInfluence) {
                    scale = influenceRadius;
                }
                else {
                    scale = influenceScale * radiusScale;
                    if (hasPscale) {
                        double avg;
                        if (openvdb::points::evalAverage<float>(points->tree(), radiusAttributeName, avg)) scale *= avg;
                    }
                }

                if (exclude.empty() && include.empty()) {
                    NullFilter filter;
                    if (hasPscale) grids = rasterZbP<NullFilter>(*points, radiusAttributeName, radiusScale, scale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                    else           grids = rasterZb<NullFilter>(*points, radiusScale, scale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                }
                else if (exclude.empty() && include.size() == 1) {
                    GroupFilter filter(include.front(), iter->attributeSet());
                    if (hasPscale) grids = rasterZbP<GroupFilter>(*points, radiusAttributeName, radiusScale, scale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                    else           grids = rasterZb<GroupFilter>(*points, radiusScale, scale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                }
                else {
                    MultiGroupFilter filter(include, exclude, iter->attributeSet());
                    if (hasPscale) grids = rasterZbP<MultiGroupFilter>(*points, radiusAttributeName, radiusScale, scale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                    else           grids = rasterZb<MultiGroupFilter>(*points, radiusScale, scale, transferAttributes, halfBand, sdfTransform, filter, &boss);
                }
            }

            FloatGrid::Ptr outputGrid;
            if (!grids.empty()) {
                outputGrid = StaticPtrCast<FloatGrid>(grids.front());
                std::rotate(grids.begin(), grids.begin()+1, grids.end());
                grids.pop_back();
            }

            if (outputGrid) {
                if (rebuildLevelSet) {
                    outputGrid =
                        tools::levelSetRebuild(*outputGrid, 0, float(halfBand), float(halfBand));
                }
                outputGrid->setName(surfaceName);
                hvdb::createVdbPrimitive(*gdp, outputGrid);
            }

            for (const auto& grid : grids) {
                if (!grid) continue;
                hvdb::createVdbPrimitive(*gdp, grid);
            }
        }

        // if no point data grids found throw warning
        if (!hasPoints) {
            addWarning(SOP_MESSAGE, "No VDB Points primitives found.");
            return error();
        }

        if (!primsToDelete.isEmpty()) {
            gdp->deletePrimitives(primsToDelete, true);
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}
