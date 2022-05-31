// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file SOP_OpenVDB_Rasterize_Frustum.cc
///
/// @author Dan Bailey
///
/// @brief  Rasterize points into density and attribute grids.

#include <houdini_utils/ParmFactory.h>

#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb_houdini/PointUtils.h>
#include <openvdb_houdini/Utils.h>

#include <openvdb/tools/Mask.h>
#include <openvdb/points/PointRasterizeFrustum.h>

#include <GU/GU_Detail.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Parm.h>

#include <hboost/algorithm/string/classification.hpp> // is_any_of
#include <hboost/algorithm/string/join.hpp>
#include <hboost/algorithm/string/split.hpp>

#include <algorithm> // std::sort
#include <memory>
#include <set>
#include <string>
#include <vector>


namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////


// Local Utility Methods

namespace {

struct MaskOp
{
    using MaskGridType = openvdb::MaskGrid;

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        // TODO: interiorMask should be made to support MaskGrids
        auto boolGrid = openvdb::tools::interiorMask(grid);

        mMaskGrid.reset(new MaskGridType(*boolGrid));
        mMaskGrid->setTransform(grid.constTransform().copy());
        mMaskGrid->setGridClass(openvdb::GRID_UNKNOWN);
    }

    MaskGridType::Ptr mMaskGrid;
};

inline openvdb::MaskGrid::Ptr
getMaskGridVDB(const GU_Detail * geoPt, const GA_PrimitiveGroup *group = nullptr)
{
    if (geoPt) {
        hvdb::VdbPrimCIterator maskIt(geoPt, group);
        if (maskIt) {
            MaskOp op;
            if (UTvdbProcessTypedGridTopology(maskIt->getStorageType(),
                maskIt->getGrid(), op)) {
                return op.mMaskGrid;
            }
            else if (UTvdbProcessTypedGridPoint(maskIt->getStorageType(),
                maskIt->getGrid(), op)) {
                return op.mMaskGrid;
            }
        }
    }

    return {};
}

inline std::unique_ptr<openvdb::BBoxd>
getMaskGeoBBox(const GU_Detail * geoPt)
{
    if (geoPt) {
        UT_BoundingBox box;
        geoPt->computeQuickBounds(box);

        std::unique_ptr<openvdb::BBoxd> bbox(new openvdb::BBoxd());
        bbox->min()[0] = box.xmin();
        bbox->min()[1] = box.ymin();
        bbox->min()[2] = box.zmin();
        bbox->max()[0] = box.xmax();
        bbox->max()[1] = box.ymax();
        bbox->max()[2] = box.zmax();

        return bbox;
    }

    return {};
}


inline bool
isScalarType(const std::string& type, const openvdb::Index stride)
{
    return  stride == 1 &&
            (   type == "bool" ||
                type == "mask" ||
                type == "half" ||
                type == "float" ||
                type == "double" ||
                type == "uint8" ||
                type == "int16" ||
                type == "uint16" ||
                type == "int32" ||
                type == "uint32" ||
                type == "int64");
}


inline bool
isVectorType(const std::string& type, const openvdb::Index stride)
{
    if (stride == 1 &&
        (   type == "vec3s" ||
            type == "vec3d" ||
            type == "vec3i"))   return true;
    if (stride == 3 &&
        (   type == "float" ||
            type == "double" ||
            type == "int32"))   return true;
    return false;
}


/// Populates the @a scalarAttribNames and @a vectorAttribNames lists.
inline void
getAttributeNames(
    const std::string& attributeNames,
    const GU_Detail& geo,
    std::vector<std::string>& scalarAttribNames,
    std::vector<std::string>& vectorAttribNames,
    bool createVelocityAttribute,
    UT_ErrorManager* log = nullptr)
{
    if (attributeNames.empty() && !createVelocityAttribute) {
        return;
    }

    std::vector<std::string> allNames;
    hboost::algorithm::split(allNames, attributeNames, hboost::is_any_of(", "));

    std::set<std::string> uniqueNames(allNames.begin(), allNames.end());

    if (createVelocityAttribute) {
        uniqueNames.insert("v");
    }

    // remove position (if it exists)

    uniqueNames.erase("P");

    std::vector<std::string> skipped;

    // Houdini attributes

    for (const std::string& name: uniqueNames) {
        GA_ROAttributeRef floatAttr = geo.findFloatTuple(GA_ATTRIB_POINT, name.c_str(), 1, 1);
        GA_ROAttributeRef floatVecAttr = geo.findFloatTuple(GA_ATTRIB_POINT, name.c_str(), 3, 3);
        GA_ROAttributeRef intAttr = geo.findIntTuple(GA_ATTRIB_POINT, name.c_str(), 1, 1);
        GA_ROAttributeRef intVecAttr = geo.findIntTuple(GA_ATTRIB_POINT, name.c_str(), 3, 3);
        if (floatAttr.isValid()) {
            scalarAttribNames.push_back(name);
        } else if (intAttr.isValid()) {
            scalarAttribNames.push_back(name);
        } else if (floatVecAttr.isValid()) {
            vectorAttribNames.push_back(name);
        } else if (intVecAttr.isValid()) {
            vectorAttribNames.push_back(name);
        } else {
            // only mark as skipped if attribute exists but is the wrong type
            if (geo.pointAttribs().find(GA_SCOPE_PUBLIC, name)) {
                skipped.push_back(name);
            }
        }
    }

    // add warnings for any incompatible Houdini attributes

    if (!skipped.empty() && log) {
        log->addWarning(SOP_OPTYPE_NAME, SOP_MESSAGE, ("Unable to rasterize Houdini attribute(s): " +
            hboost::algorithm::join(skipped, ", ")).c_str());
        log->addWarning(SOP_OPTYPE_NAME, SOP_MESSAGE, "Only supporting Houdini attributes "
            "of scalar or vector type with integer or floating-point values.");
    }

    // VDB Points attributes

    for (hvdb::VdbPrimCIterator vdbIt(&geo); vdbIt; ++vdbIt) {
        const GU_PrimVDB* vdbPrim = *vdbIt;
        if (vdbPrim->getStorageType() != UT_VDB_POINTDATA)  continue;

        const auto& points = static_cast<const openvdb::points::PointDataGrid&>(vdbPrim->getConstGrid());
        auto leaf = points.constTree().cbeginLeaf();
        if (!leaf)  continue;
        auto descriptor = leaf->attributeSet().descriptor();

        for (auto it : descriptor.map()) {
            if (std::find(uniqueNames.begin(), uniqueNames.end(),
                it.first) == uniqueNames.end()) {
                continue;
            }
            const auto* attributeArray = leaf->attributeSet().getConst(it.first);
            if (!attributeArray)    continue;

            const openvdb::Index stride = attributeArray->stride();
            const openvdb::Name& valueType = descriptor.valueType(it.second);
            if (isScalarType(valueType, stride)) {
                scalarAttribNames.push_back(it.first);
            } else if (isVectorType(valueType, stride)) {
                vectorAttribNames.push_back(it.first);
            }
        }
    }

    // remove duplicates

    std::sort(scalarAttribNames.begin(), scalarAttribNames.end());
    scalarAttribNames.erase(std::unique(scalarAttribNames.begin(), scalarAttribNames.end()), scalarAttribNames.end());

    std::sort(vectorAttribNames.begin(), vectorAttribNames.end());
    vectorAttribNames.erase(std::unique(vectorAttribNames.begin(), vectorAttribNames.end()), vectorAttribNames.end());
}


/// Returns a null pointer if geoPt is null or if no reference vdb is found.
inline openvdb::math::Transform::Ptr
getReferenceTransform(const GU_Detail* geoPt, const GA_PrimitiveGroup* group = nullptr,
    UT_ErrorManager* log = nullptr)
{
    if (geoPt) {
        hvdb::VdbPrimCIterator vdbIt(geoPt, group);
        if (vdbIt) {
            return (*vdbIt)->getGrid().transform().copy();
        } else if (log) {
            log->addWarning(SOP_OPTYPE_NAME, SOP_MESSAGE, "Could not find a reference VDB grid");
        }
    }

    return {};
}


////////////////////////////////////////


inline int
lookupAttrInput(const PRM_SpareData* spare)
{
    if (!spare) return 0;
    const char* istring = spare->getValue("sop_input");
    return istring ? atoi(istring) : 0;
}


inline void
populateMeshMenu(void* data, PRM_Name* choicenames, int themenusize,
    const PRM_SpareData* spare, const PRM_Parm*)
{
    choicenames[0].setToken(0);
    choicenames[0].setLabel(0);

    SOP_Node* sop = CAST_SOPNODE(static_cast<OP_Node*>(data));
    if (sop == nullptr) return;

    size_t count = 0;

    const int inputIndex = lookupAttrInput(spare);
    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

    if (gdp) {
        GA_AttributeDict::iterator iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);
        size_t maxSize(themenusize - 1);

        std::vector<std::string> scalarNames, vectorNames;
        scalarNames.reserve(gdp->pointAttribs().entries(GA_SCOPE_PUBLIC));
        vectorNames.reserve(scalarNames.capacity());

        // add Houdini attributes

        for (; !iter.atEnd(); ++iter) {
            GA_Attribute const * const attrib = iter.attrib();

            if (attrib->getStorageClass() == GA_STORECLASS_FLOAT ||
                attrib->getStorageClass() == GA_STORECLASS_INT) {

                const int tupleSize = attrib->getTupleSize();

                const UT_StringHolder& attribName = attrib->getName();
                // ignore position
                if (attribName.buffer() == std::string("P"))   continue;
                if (tupleSize == 1) scalarNames.push_back(attribName.buffer());
                else if (tupleSize == 3) vectorNames.push_back(attribName.buffer());
            }
        }

        // add VDB Points attributes

        for (hvdb::VdbPrimCIterator vdbIt(gdp); vdbIt; ++vdbIt) {
            const GU_PrimVDB* vdbPrim = *vdbIt;
            if (vdbPrim->getStorageType() != UT_VDB_POINTDATA)  continue;

            const auto& points = static_cast<const openvdb::points::PointDataGrid&>(vdbPrim->getConstGrid());
            auto leaf = points.constTree().cbeginLeaf();
            if (!leaf)  continue;
            const auto& descriptor = leaf->attributeSet().descriptor();

            for (const auto& it : descriptor.map()) {
                // ignore position
                if (it.first == "P")   continue;
                const auto* attributeArray = leaf->attributeSet().getConst(it.second);
                if (!attributeArray)    continue;
                const openvdb::Index stride = attributeArray->stride();
                const openvdb::Name& valueType = descriptor.valueType(it.second);
                if (isScalarType(valueType, stride)) {
                    scalarNames.push_back(it.first);
                } else if (isVectorType(valueType, stride)) {
                    vectorNames.push_back(it.first);
                }
            }
        }

        std::sort(scalarNames.begin(), scalarNames.end());
        scalarNames.erase(std::unique(scalarNames.begin(), scalarNames.end()), scalarNames.end());

        for (size_t n = 0, N = scalarNames.size(); n < N && count < maxSize; ++n) {
            const char * str = scalarNames[n].c_str();
            choicenames[count].setToken(str);
            choicenames[count++].setLabel(str);
        }

        if (!scalarNames.empty() && !vectorNames.empty() && count < maxSize) {
            choicenames[count].setToken(PRM_Name::mySeparator);
            choicenames[count++].setLabel(PRM_Name::mySeparator);
        }

        std::sort(vectorNames.begin(), vectorNames.end());
        vectorNames.erase(std::unique(vectorNames.begin(), vectorNames.end()), vectorNames.end());

        for (size_t n = 0, N = vectorNames.size(); n < N && count < maxSize; ++n) {
            choicenames[count].setToken(vectorNames[n].c_str());
            choicenames[count++].setLabel(vectorNames[n].c_str());
        }
    }

    // Terminate the list.
    choicenames[count].setToken(0);
    choicenames[count].setLabel(0);
}

inline void
populateVelocityMenu(void* data, PRM_Name* choicenames, int themenusize,
    const PRM_SpareData* spare, const PRM_Parm*)
{
    choicenames[0].setToken(0);
    choicenames[0].setLabel(0);

    SOP_Node* sop = CAST_SOPNODE(static_cast<OP_Node*>(data));
    if (sop == nullptr) return;

    size_t count = 0;

    const int inputIndex = lookupAttrInput(spare);
    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

    if (gdp) {
        GA_AttributeDict::iterator iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);
        size_t maxSize(themenusize - 1);

        std::vector<std::string> velocityNames;

        // add Houdini attributes

        for (; !iter.atEnd(); ++iter) {
            GA_Attribute const * const attrib = iter.attrib();


            if (attrib->getStorageClass() == GA_STORECLASS_FLOAT) {

                const int tupleSize = attrib->getTupleSize();

                const UT_StringHolder& attribName = attrib->getName();
                // ignore position
                if (attribName.buffer() == std::string("P"))   continue;
                if (tupleSize == 3) velocityNames.push_back(attribName.buffer());
            }
        }

        // add VDB Points attributes

        for (hvdb::VdbPrimCIterator vdbIt(gdp); vdbIt; ++vdbIt) {
            const GU_PrimVDB* vdbPrim = *vdbIt;
            if (vdbPrim->getStorageType() != UT_VDB_POINTDATA)  continue;

            const auto& points = static_cast<const openvdb::points::PointDataGrid&>(vdbPrim->getConstGrid());
            auto leaf = points.constTree().cbeginLeaf();
            if (!leaf)  continue;
            const auto& descriptor = leaf->attributeSet().descriptor();

            for (const auto& it : descriptor.map()) {
                // ignore position
                if (it.first == "P")   continue;
                // velocity can only be Vec3s or Vec3d
                const auto& valueType = descriptor.valueType(it.second);
                if (valueType == "vec3s" || valueType == "vec3d") {
                    velocityNames.push_back(it.first);
                }
            }
        }

        std::sort(velocityNames.begin(), velocityNames.end());

        for (size_t n = 0, N = velocityNames.size(); n < N && count < maxSize; ++n) {
            choicenames[count].setToken(velocityNames[n].c_str());
            choicenames[count++].setLabel(velocityNames[n].c_str());
        }
    }

    // Terminate the list.
    choicenames[count].setToken(0);
    choicenames[count].setLabel(0);
}

inline void
populateRadiusMenu(void* data, PRM_Name* choicenames, int themenusize,
    const PRM_SpareData* spare, const PRM_Parm*)
{
    choicenames[0].setToken(0);
    choicenames[0].setLabel(0);

    SOP_Node* sop = CAST_SOPNODE(static_cast<OP_Node*>(data));
    if (sop == nullptr) return;

    size_t count = 0;

    const int inputIndex = lookupAttrInput(spare);
    const GU_Detail* gdp = sop->getInputLastGeo(inputIndex, CHgetEvalTime());

    if (gdp) {
        GA_AttributeDict::iterator iter = gdp->pointAttribs().begin(GA_SCOPE_PUBLIC);
        size_t maxSize(themenusize - 1);

        std::vector<std::string> radiusNames;

        // add Houdini attributes

        for (; !iter.atEnd(); ++iter) {
            GA_Attribute const * const attrib = iter.attrib();


            if (attrib->getStorageClass() == GA_STORECLASS_FLOAT) {

                const int tupleSize = attrib->getTupleSize();

                const UT_StringHolder& attribName = attrib->getName();
                // ignore position
                if (attribName.buffer() == std::string("P"))   continue;
                if (tupleSize == 1) radiusNames.push_back(attribName.buffer());
            }
        }

        // add VDB Points attributes

        for (hvdb::VdbPrimCIterator vdbIt(gdp); vdbIt; ++vdbIt) {
            const GU_PrimVDB* vdbPrim = *vdbIt;
            if (vdbPrim->getStorageType() != UT_VDB_POINTDATA)  continue;

            const auto& points = static_cast<const openvdb::points::PointDataGrid&>(vdbPrim->getConstGrid());
            auto leaf = points.constTree().cbeginLeaf();
            if (!leaf)  continue;
            const auto& descriptor = leaf->attributeSet().descriptor();

            for (const auto& it : descriptor.map()) {
                // ignore position
                if (it.first == "P")   continue;
                // velocity can only be Vec3s or Vec3d
                const auto& valueType = descriptor.valueType(it.second);
                if (valueType == "float") {
                    radiusNames.push_back(it.first);
                }
            }
        }

        std::sort(radiusNames.begin(), radiusNames.end());

        for (size_t n = 0, N = radiusNames.size(); n < N && count < maxSize; ++n) {
            choicenames[count].setToken(radiusNames[n].c_str());
            choicenames[count++].setLabel(radiusNames[n].c_str());
        }
    }

    // Terminate the list.
    choicenames[count].setToken(0);
    choicenames[count].setLabel(0);
}

struct GridsToRasterize
{
    using GridType = openvdb::points::PointDataGrid;
    using TreeType = GridType::TreeType;
    using ConstPtr = GridType::ConstPtr;

    struct Grid
    {
    private:
        static bool isTreeOutOfCore(const TreeType& tree)
        {
            using LeafManagerT = openvdb::tree::LeafManager<const TreeType>;
            using LeafRangeT = typename LeafManagerT::LeafRange;
            LeafManagerT leafManager(tree);
            return tbb::parallel_reduce(leafManager.leafRange(), true,
                [] (const LeafRangeT& range, bool result) -> bool {
                    for (const auto& leaf : range) {
                        if (!leaf.buffer().isOutOfCore()) return false;
                    }
                    return result;
                },
                [] (bool n, bool m) -> bool { return n || m; });
        }

    public:
        explicit Grid(const ConstPtr& grid)
            : mGrid(grid)
            , mOutOfCore(isTreeOutOfCore(grid->constTree())) { }

        inline bool isOutOfCore() const { return mOutOfCore; }

        template <typename RasterizerT>
        void addToRasterizer(RasterizerT& rasterizer)
        {
            if (mOutOfCore) {
                auto newGrid = mGrid->deepCopy();
                rasterizer.addPoints(newGrid, /*stream=*/true);
            }
            else {
                rasterizer.addPoints(mGrid);
            }
        }

    private:
        ConstPtr mGrid;
        const bool mOutOfCore;
    }; // struct Grid

    void push_back(ConstPtr& grid) { mGrids.emplace_back(grid); }
    size_t size() const { return mGrids.size(); }
    bool empty() const { return mGrids.empty(); }

    // return true if any of the grids is out-of-core
    bool streaming() const
    {
        for (auto& grid : mGrids) {
            if (grid.isOutOfCore()) return true;
        }
        return false;
    }

    template <typename RasterizerT>
    void addGridToRasterizer(RasterizerT& rasterizer, size_t index)
    {
        rasterizer.clear();
        mGrids[index].addToRasterizer(rasterizer);
    }

    template <typename RasterizerT>
    void addGridsToRasterizer(RasterizerT& rasterizer)
    {
        rasterizer.clear();
        for (auto& grid : mGrids) {
            grid.addToRasterizer(rasterizer);
        }
    }

private:
    std::vector<Grid> mGrids;
}; // struct GridsToRasterize

} // unnamed namespace


////////////////////////////////////////

// SOP Implementation

struct SOP_OpenVDB_Rasterize_Frustum: public hvdb::SOP_NodeVDB
{
    SOP_OpenVDB_Rasterize_Frustum(OP_Network*, const char* name, OP_Operator*);
    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    int isRefInput(unsigned i) const override { return i > 0; }

protected:
    OP_ERROR cookVDBSop(OP_Context&) override;
    bool updateParmsFlags() override;
}; // struct SOP_OpenVDB_Rasterize_Frustum


////////////////////////////////////////

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify input primitive groups to rasterize.")
        .setChoiceList(&hutil::PrimGroupMenu));

    parms.add(hutil::ParmFactory(PRM_STRING, "vdbpointsgroups", "VDB Points Groups")
        .setHelpText("Specify VDB Points groups to rasterize.")
        .setChoiceList(&hvdb::VDBPointsGroupMenuInput1));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "mergevdbpoints", "Merge VDB Points")
        .setDefault(PRMoneDefaults)
        .setTooltip(
            "Merge VDB Points grids during rasterization to output a single VDB volume."));

    parms.add(hutil::ParmFactory(PRM_STRING, "transformvdb", "Transform VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput2)
        .setTooltip("VDB grid that defines the output transform."));

    parms.add(hutil::ParmFactory(PRM_STRING, "maskvdb", "Mask VDB")
        .setChoiceList(&hutil::PrimGroupMenuInput3)
        .setTooltip("VDB grid whose active topology defines what region to rasterize into.")
        .setDocumentation("VDB grid whose active topology defines what region to rasterize into. "
            "Clipping is performed based on volume samples, not point positions."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "invertmask", "Invert Mask")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Toggle to rasterize in the region outside the mask."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "voxelsize", "Voxel Size")
        .setDefault(PRMpointOneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 5)
        .setTooltip("Uniform voxel edge length in world units.  "
            "Decrease the voxel size to increase the volume resolution."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "cliptofrustum", "Clip to Frustum")
        .setDefault(PRMoneDefaults)
        .setTooltip(
            "When using a frustum transform, only rasterize data inside the frustum region."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "optimizeformemory", "Optimize for Memory")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "Keep the memory footprint as small as possible at the cost of slower performance."));

    parms.beginSwitcher("tabMenu");
    parms.addFolder("Volume");

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "createdensity", "Create Density Volume")
        .setDefault(PRMoneDefaults)
        .setTooltip("Toggle to enable or disable the density volume generation. "
            "Attribute volumes are still constructed as usual."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "densityscale", "Density Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setTooltip("Scale the density attribute by this amount."
            " If the density attribute is missing, use a value of one as the reference.")
        .setDocumentation("Scale the `density` attribute by this amount."
            " If the `density` attribute is missing, use a value of one as the reference."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "contributionthreshold", "Contribution Threshold")
        .setDefault(0.0001)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 0.1)
        .setTooltip("Only rasterize attribute contribution when within this threshold."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "createmask", "Create Mask")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, output a rasterization mask."));

    parms.add(hutil::ParmFactory(PRM_STRING, "attributes", "Attributes")
        .setChoiceList(new PRM_ChoiceList(PRM_CHOICELIST_TOGGLE, populateMeshMenu))
        .setTooltip("Specify a list of point attributes to be rasterized."));

    { // density point compositing (also supports average)
        char const * const items[] = {
            "add",      "Accumulated",
            "max",      "Maximum",
            "average",  "Weighted Average",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "densitymode", "Density Mode")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("How to blend point densities in the density volume")
            .setDocumentation(
                "How to blend point densities in the density volume\n\n"
                "Accumulated:\n"
                "    Density contributions from each particle are simply summed up.\n"
                "Maximum:\n"
                "    Choose the maximum density at each voxel.\n"
                "Weighted Average:\n"
                "    Compute the weighted-average of the densities at each voxel.\n"));
    }

    { // scalar attribute compositing (supports add, max and average)
        char const * const items[] = {
            "add",      "Accumulated",
            "max",      "Maximum",
            "average",  "Weighted Average",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "scalarmode", "Scalar Mode")
            .setDefault(PRMtwoDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("How to blend scalar point attribute values (except density)")
            .setDocumentation(
                "How to blend scalar point attribute values (except density)\n\n"
                "Accumulated:\n"
                "    Contributions from each particle are simply summed up.\n"
                "Maximum:\n"
                "    Choose the maximum value at each voxel.\n"
                "Weighted Average:\n"
                "    Compute the weighted-average of the value at each voxel.\n"));
    }

    { // vector attribute compositing (supports add and average)
        char const * const items[] = {
            "add",      "Accumulated",
            "average",  "Weighted Average",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "vectormode", "Vector Mode")
            .setDefault(PRMoneDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("How to blend vector point attribute values")
            .setDocumentation(
                "How to blend vector point attribute values\n\n"
                "Accumulated:\n"
                "    Contributions from each particle are simply summed up.\n"
                "Weighted Average:\n"
                "    Compute the weighted-average of the value at each voxel.\n"));
    }

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "scalebyvoxelvolume", "Scale Contribution by Voxel Volume")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "Scale each voxel rasterization contribution by the volume of the voxel."));

    parms.addFolder("Radius");

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "enableradius", "Enable")
        .setDefault(PRMoneDefaults)
        .setTooltip("Toggle to enable or disable the use of particle radius."));

    parms.add(hutil::ParmFactory(PRM_STRING, "radiusattribute", "Radius Attribute")
        .setDefault("pscale")
        .setChoiceList(new PRM_ChoiceList(PRM_CHOICELIST_TOGGLE, populateRadiusMenu))
        .setTooltip("Radius attribute (defaults to \"pscale\")."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "radiusscale", "Radius Scale")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setTooltip("Scale the radius by this amount."
            " If radius attribute is missing, use a value of one as the reference."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "accuratefrustumradius", "Accurate Frustum Radius")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, use a highly accurate algorithm to rasterize points with radius "
            "into a frustum grid at the cost of much slower performance."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "accuratespheremotionblur", "Accurate Sphere Motion Blur")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, pack spheres more tightly along the motion-blurred rasterization "
            "path to hide visible sphere artefacts at the cost of slower performance."));

    parms.addFolder("Motion Blur");

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "bakemotionblur", "Bake Motion Blur")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Bake the motion blur of the point when applying geometry and/or "
            "camera deformation."));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "shutter", "Shutter")
        .setDefault(0.5)
        .setRange(PRM_RANGE_UI, 0, PRM_RANGE_UI, 1)
        .setTooltip("The portion of the frame interval that the camera shutter is open when "
            "baking motion blur. [0,1]"));

    parms.add(hutil::ParmFactory(PRM_FLT_J, "shutteroffset", "Shutter Offset")
        .setDefault(1)
        .setRange(PRM_RANGE_UI, -1, PRM_RANGE_UI, 1)
        .setTooltip("Controls where the motion blur occurs relative to the current frame. "
            "A value of -1 will blur from the previous frame to the current frame. A value "
            "of 0 will blur from halfway to the previous frame to halfway to the next frame. "
            "A value of 1 will blur from the current frame to the next frame. "));

    parms.add(hutil::ParmFactory(PRM_FLT, "framespersecond", "Frames / Second")
        .setDefault(1, "$FPS")
        .setTooltip("Frames-per-second to use when computing motion blur."));

    parms.add(hutil::ParmFactory(PRM_INT_J, "motionsamples", "Motion Samples")
        .setDefault(2)
        .setRange(PRM_RANGE_RESTRICTED, 2, PRM_RANGE_UI, 10)
        .setTooltip("How many motion samples to use when computing motion blur.")
        .setDocumentation("How many motion samples to use when computing motion blur. "
            "The default is two, which means a sample at the beginning and end of the shutter "
            "interval. Increase this value to compute more accurate camera motion blur through "
            "sampling the reference transform at a higher frequency. It can also be useful in "
            "eliminating any warping effect when applying geometry motion blur in a "
            "frustum grid. Note that motion sampling will automatically be disabled if the "
            "camera is static and frustum rasterization is not in use."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "geometrymotionblur", "Geometry Motion Blur")
        .setDefault(PRMoneDefaults)
        .setTooltip("Bake geometry motion blur as computed using point velocity."));

    parms.add(hutil::ParmFactory(PRM_STRING, "velocityattribute", "Velocity Attribute")
        .setDefault("v")
        .setChoiceList(new PRM_ChoiceList(PRM_CHOICELIST_TOGGLE, populateVelocityMenu))
        .setTooltip("Velocity attribute to apply geometry motion blur (defaults to \"v\")."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "cameramotionblur", "Camera Motion Blur")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Bake camera motion blur as computed using the motion derived from the "
            "reference transform."));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "allowcameratransforminterpolation", "Allow Interpolation of Camera Motion")
        .setDefault(PRMoneDefaults)
        .setTooltip("If enabled, camera transform interpolation is performed when camera motion "
            "is detected not to be continuous. In certain cases, it may be desirable to disable "
            "interpolation when using complex camera motion that is known to be continuous."));

    parms.endSwitcher();

    /////

    hvdb::OpenVDBOpFactory("VDB Rasterize Frustum",
        SOP_OpenVDB_Rasterize_Frustum::factory, parms, *table)
        .addInput("Particles to rasterize")
#ifndef SESI_OPENVDB
        .setInternalName("DW_OpenVDBRasterizeParticles")
#endif
        .addOptionalInput("Optional VDB grid that defines the output transform.")
        .addOptionalInput("Optional VDB or bounding box mask.")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Rasterize points into density and attribute volumes.\"\"\"\n\
\n\
@overview\n\
\n\
This node rasterizes point attributes into density, mask, float, vector and integer grids.\n\
It supports using particle radius and can efficiently choose between point, ray or sphere\n\
rasterization algorithms for each point in a data set based on the input values. \n\
\n\
Clipping can be performed by frustum, bounding box or mask and is done on the volume samples,\n\
not the positions of the points.\n\
\n\
This node accepts Houdini points and VDB points. When using Houdini points, there will be a\n\
small performance cost as the points and required attributes are implicitly converted into\n\
VDB Points for processing.\n\
\n\
Geometry and camera motion blur can be baked into the resulting grids. Geometry motion blur is\n\
computed using a velocity point attribute, camera motion blur is computed using an animated\n\
transform provided using the reference VDB supplied in the second node input. \n\
\n\
Rasterizing can be performed into cartesian and frustum grids (otherwise referred to as tapered\n\
grids). For frustum grids, approximations are used by default to accelerate the rasterization.\n\
This can be disabled to achieve a more accurate result.\n\
\n\
This node supports streaming of VDB points and attributes if one or more delayed-load VDB is\n\
provided as an input.\n\
\n\
@related\n\
- [OpenVDB Rasterize Points|Node:sop/DW_OpenVDBRasterizePoints]\n\
- [Node:sop/cloud]\n\
\n");
}

bool
SOP_OpenVDB_Rasterize_Frustum::updateParmsFlags()
{
    bool changed = false;

    const bool refexists = getInput(1) != nullptr;
    changed |= enableParm("voxelsize", !refexists);
    changed |= enableParm("transformvdb", refexists);
    changed |= enableParm("cliptofrustum", refexists);

    const bool maskexists = getInput(2) != nullptr;
    changed |= enableParm("maskvdb", maskexists);
    changed |= enableParm("invertmask", maskexists);

    const std::string attributes = evalStdString("attributes", 0);
    const bool createDensity = bool(evalInt("createdensity", 0, 0));

    const bool pointsDensity = createDensity;
    changed |= enableParm("densityscale", pointsDensity);
    changed |= enableParm("contributionthreshold", pointsDensity);

    const bool enableMotionBlur = bool(evalInt("bakemotionblur", 0, 0));

    const bool useRadius = bool(evalInt("enableradius", 0, 0));
    changed |= enableParm("radiusattribute", useRadius);
    changed |= enableParm("radiusscale", useRadius);
    changed |= enableParm("accuratefrustumradius", useRadius);
    changed |= enableParm("accuratespheremotionblur", useRadius && enableMotionBlur);

    changed |= enableParm("densitymode", createDensity);

    changed |= enableParm("scalarmode", !attributes.empty());
    changed |= enableParm("vectormode", !attributes.empty());

    changed |= enableParm("shutter", enableMotionBlur);
    changed |= enableParm("shutteroffset", enableMotionBlur);
    changed |= enableParm("framespersecond", enableMotionBlur);
    changed |= enableParm("motionsamples", enableMotionBlur);
    changed |= enableParm("geometrymotionblur", enableMotionBlur);
    changed |= enableParm("cameramotionblur", enableMotionBlur && refexists);

    const bool geometryMotionBlur = enableMotionBlur && bool(evalInt("geometrymotionblur", 0, 0));
    changed |= enableParm("velocityattribute", geometryMotionBlur);

    const bool cameraMotionBlur = enableMotionBlur && bool(evalInt("cameramotionblur", 0, 0));
    changed |= enableParm("allowcameratransforminterpolation", cameraMotionBlur);

    return changed;
}

OP_Node*
SOP_OpenVDB_Rasterize_Frustum::factory(OP_Network* net, const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Rasterize_Frustum(net, name, op);
}


SOP_OpenVDB_Rasterize_Frustum::SOP_OpenVDB_Rasterize_Frustum(OP_Network* net,
    const char* name, OP_Operator* op)
    : hvdb::SOP_NodeVDB(net, name, op)
{
}


OP_ERROR
SOP_OpenVDB_Rasterize_Frustum::cookVDBSop(OP_Context& context)
{
    try {
        OP_AutoLockInputs inputs(this);
        if (inputs.lock(context) >= UT_ERROR_ABORT)
            return error();
        gdp->clearAndDestroy();

        UT_ErrorManager* log = UTgetErrorManager();

        // Get UI parameters

        const fpreal time = context.getTime();

        float voxelSize = float(evalFloat("voxelsize", 0, time));
        const bool clipToFrustum = evalInt("cliptofrustum", 0, time) == 1;
        const bool invertMask = evalInt("invertmask", 0, time) == 1;

        const GU_Detail* pointGeo = inputGeo(0, context);
        const GA_PrimitiveGroup* group = matchGroup(*pointGeo, evalStdString("group", time));

        const GU_Detail* refGeo = inputGeo(1, context);
        const GA_PrimitiveGroup* refGroup = nullptr;

        if (refGeo) {
            refGroup = parsePrimitiveGroups(
                evalStdString("transformvdb", time).c_str(), GroupCreator(refGeo));
        }

        const GU_Detail* maskGeo = inputGeo(2, context);

        std::unique_ptr<openvdb::BBoxd> maskBBox;
        openvdb::MaskGrid::Ptr maskGrid;
        if (maskGeo) {
            bool expectingVDBMask = false;
            const auto groupStr = evalStdString("maskvdb", time);
            const GA_PrimitiveGroup* maskGroup =
                parsePrimitiveGroups(groupStr.c_str(), GroupCreator(maskGeo));
            if (!groupStr.empty()) {
                expectingVDBMask = true;
            }
            maskGrid = getMaskGridVDB(maskGeo, maskGroup);
            if (!maskGrid) {
                if (expectingVDBMask) {
                    addWarning(SOP_MESSAGE, "VDB mask not found.");
                } else {
                    maskBBox = getMaskGeoBBox(maskGeo);
                }
            }
        }

        // Get selected point attribute names

        std::vector<std::string> scalarAttribNames;
        std::vector<std::string> vectorAttribNames;

        getAttributeNames(evalStdString("attributes", time), *pointGeo,
            scalarAttribNames, vectorAttribNames, /*createVelocityAttribute=*/false, log);

        const bool createDensity = 0 != evalInt("createdensity", 0, time);

        const bool createMask = 0 != evalInt("createmask", 0, time);

        if (createDensity || createMask || !scalarAttribNames.empty() || !vectorAttribNames.empty())
        {
            hvdb::HoudiniInterrupter boss("Rasterizing points");

            // Set rasterization settings

            openvdb::math::Transform::Ptr xform = getReferenceTransform(refGeo, refGroup, log);
            if (xform) {
                voxelSize = float(xform->voxelSize().x());
            } else {
                voxelSize = static_cast<float>(evalFloat("voxelsize", 0, time));
                xform = openvdb::math::Transform::createLinearTransform(voxelSize);
            }

            assert(xform);

            std::vector<openvdb::GridBase::Ptr> outputGrids;

            std::vector<GA_Offset> vdbPrimOffsets;
            for (hvdb::VdbPrimCIterator vdbIt(pointGeo, group); vdbIt; ++vdbIt) {
                const GU_PrimVDB* vdbPrim = *vdbIt;

                // mark point offset as a point referencing a VDB
                vdbPrimOffsets.push_back(vdbPrim->getPointOffset(0));
            }

            // Rasterize VDB Points attributes

            const bool bakeMotionBlur = 0 != evalInt("bakemotionblur", 0, time);
            const std::string densityAttribute = "density";
            const std::string velocityAttribute = evalStdString("velocityattribute", time);
            const std::string radiusAttribute = evalStdString("radiusattribute", time);

            openvdb::points::FrustumRasterizerSettings settings(*xform);
            // settings.threaded = false;
            settings.threshold = static_cast<float>(evalFloat("contributionthreshold", 0, time));
            settings.useRadius = 0 != evalInt("enableradius", 0, time);
            settings.radiusScale = static_cast<float>(evalFloat("radiusscale", 0, time));
            settings.accurateFrustumRadius = 0 != evalInt("accuratefrustumradius", 0, time);
            settings.accurateSphereMotionBlur = 0 != evalInt("accuratespheremotionblur", 0, time);
            settings.scaleByVoxelVolume = 0 != evalInt("scalebyvoxelvolume", 0, time);
            settings.velocityAttribute = velocityAttribute;
            settings.radiusAttribute = radiusAttribute;
            settings.velocityMotionBlur = bakeMotionBlur && 0 != evalInt("geometrymotionblur", 0, time);
            settings.framesPerSecond = static_cast<float>(evalFloat("framespersecond", 0, time));
            settings.motionSamples = std::max(2, static_cast<int>(evalInt("motionsamples", 0, time)));

            openvdb::points::FrustumRasterizerMask mask(*xform,
                maskGrid ? maskGrid.get() : nullptr,
                maskBBox ? *maskBBox : openvdb::BBoxd(), clipToFrustum, invertMask);

            const bool mergeVDBPoints = 0 != evalInt("mergevdbpoints", 0, time);
            const bool reduceMemory = 0 == evalInt("optimizeformemory", 0, time);

            GridsToRasterize pointGrids;

            for (hvdb::VdbPrimCIterator vdbIt(pointGeo, group); vdbIt; ++vdbIt) {

                const GU_PrimVDB* vdbPrim = *vdbIt;

                // only process if grid is a PointDataGrid
                auto gridPtr = openvdb::gridConstPtrCast<openvdb::points::PointDataGrid>(vdbPrim->getConstGridPtr());
                if(!gridPtr) continue;
                pointGrids.push_back(gridPtr);
            }

            // Convert all Houdini points that don't reference a VDB into a new VDB

            if (pointGeo->getNumPoints() > vdbPrimOffsets.size()) {
                // compute auto voxel-size based on point distribution
                openvdb::math::Mat4d matrix(openvdb::math::Mat4d::identity());
                voxelSize = hvdb::computeVoxelSizeFromHoudini(*pointGeo, /*pointsPerVoxel=*/8,
                    matrix, /*rounding=*/5, boss);
                matrix.preScale(openvdb::Vec3d(voxelSize) / openvdb::math::getScale(matrix));
                auto pointsTransform = openvdb::math::Transform::createLinearTransform(matrix);

                // convert Houdini points to VDB Points
                openvdb_houdini::AttributeInfoMap attributes;
                attributes[densityAttribute] = {0, false};
                if (!velocityAttribute.empty()) {
                    attributes[velocityAttribute] = {0, false};
                }
                if (!radiusAttribute.empty()) {
                    attributes[radiusAttribute] = {0, false};
                }
                for (const auto& name : scalarAttribNames) {
                    attributes[name] = {0, false};
                }
                for (const auto& name : vectorAttribNames) {
                    attributes[name] = {0, false};
                }

                openvdb::points::PointDataGrid::Ptr houdiniPointsAsGridNonConst = hvdb::convertHoudiniToPointDataGrid(
                    *pointGeo, /*compression=*/1, attributes, *pointsTransform);
                openvdb::points::PointDataGrid::ConstPtr houdiniPointsAsGrid = openvdb::ConstPtrCast<
                    const openvdb::points::PointDataGrid>(houdiniPointsAsGridNonConst);
                pointGrids.push_back(houdiniPointsAsGrid);
            }

            if (!pointGrids.empty()) {
                const std::string groups = evalStdString("vdbpointsgroups", time);
                // Get and parse the vdb points groups
                openvdb::points::RasterGroups rasterGroups;
                openvdb::points::AttributeSet::Descriptor::parseNames(
                    rasterGroups.includeNames, rasterGroups.excludeNames, groups);

                const float shutter = static_cast<float>(evalFloat("shutter", 0, time));
                const float shutterOffset = static_cast<float>(evalFloat("shutteroffset", 0, time));

                // translate shutter parameters into shutter start and end
                const float shutterStart = ((shutterOffset-1.0f) * shutter) / 2.0f;
                const float shutterEnd = ((shutterOffset+1.0f) * shutter)  / 2.0f;

                auto& camera = settings.camera;

                // set shutter start and end if motion blur enabled
                if (bakeMotionBlur) {
                    camera.setShutter(shutterStart, shutterEnd);
                }

                // set camera transform deduced from reference transform
                const bool cameraMotionBlur = bakeMotionBlur && getInput(1) != nullptr &&
                    0 != evalInt("cameramotionblur", 0, time);
                if (cameraMotionBlur) {
                    // explicitly unlock reference geo input to be able to re-evaluate with a different context
                    inputs.unlockInput(1);

                    OP_Context refContext = context;

                    const int samples = settings.motionSamples;
                    const float frame = static_cast<float>(context.getFloatFrame());
                    const float shutterIncrement = (shutterEnd - shutterStart) /
                        (static_cast<float>(samples) - 1);

                    bool continuousSampling = true;

                    const bool enableTransformInterpolation =
                        0 != evalInt("allowcameratransforminterpolation", 0, time);

                    if (enableTransformInterpolation)
                    {
                        // test if transforms at small frame intervals from the current frame
                        // are equal and if not, enable interpolation of transformed positions
                        refContext.setFrame(frame - 0.1);
                        if (inputs.lockInput(1, refContext) >= UT_ERROR_ABORT)
                            return error();
                        auto prevTransform = getReferenceTransform(inputGeo(1));

                        if (*prevTransform == *xform) {
                            continuousSampling = false;
                        } else {
                            refContext.setFrame(frame + 0.1);
                            if (inputs.lockInput(1, refContext) >= UT_ERROR_ABORT)
                                return error();
                            auto nextTransform = getReferenceTransform(inputGeo(1));
                            if (*nextTransform == *xform)   continuousSampling = false;
                        }
                    }

                    const float tolerance = 1e-3f;

                    // drop default transform
                    camera.clear();

                    for (int i = 0; i < samples; i++) {
                        // Sample the transform and apply the matrix
                        const float frameSample = i == samples - 1 ? frame + shutterEnd :
                            frame + shutterStart + shutterIncrement * static_cast<float>(i);
                        float adjustedFrameSample(frameSample);
                        if (continuousSampling) refContext.setFrame(frameSample);
                        else {
                            if (!openvdb::math::isApproxEqual(frameSample, frame, tolerance)) {
                                if (frameSample < frame) {
                                    adjustedFrameSample =
                                        static_cast<float>(openvdb::math::Floor(adjustedFrameSample-tolerance));
                                } else {
                                    adjustedFrameSample =
                                        static_cast<float>(openvdb::math::Ceil(adjustedFrameSample+tolerance));
                                }
                            }
                            refContext.setFrame(adjustedFrameSample);
                        }
                        if (inputs.lockInput(1, refContext) >= UT_ERROR_ABORT)
                            return error();
                        auto transform = getReferenceTransform(inputGeo(1));
                        if (!transform) {
                            throw std::runtime_error{"Cannot extract camera transform from VDB in reference input."};
                        }
                        if (continuousSampling) camera.appendTransform(*transform);
                        else {
                            float transformWeight = 1.0f;
                            if (!openvdb::math::isApproxZero(adjustedFrameSample, tolerance)) {
                                transformWeight = (frameSample - frame) / (adjustedFrameSample - frame);
                            }
                            camera.appendTransform(*transform, transformWeight);
                        }
                        inputs.unlockInput(1);
                    }

                    // Reduce the number of transforms stored to one if static
                    camera.simplify();
                }

                const exint densityCompositing = evalInt("densitymode", 0, time);
                const exint scalarCompositing = evalInt("scalarmode", 0, time);
                const exint vectorCompositing = evalInt("vectormode", 0, time);

                const openvdb::points::RasterMode accumulateMode(openvdb::points::RasterMode::ACCUMULATE);
                const openvdb::points::RasterMode maximumMode(openvdb::points::RasterMode::MAXIMUM);
                const openvdb::points::RasterMode averageMode(openvdb::points::RasterMode::AVERAGE);

                openvdb::points::RasterMode          densityMode = accumulateMode;
                if (densityCompositing == 1)        densityMode = maximumMode;
                else if (densityCompositing == 2)   densityMode = averageMode;

                openvdb::points::RasterMode          scalarMode = accumulateMode;
                if (scalarCompositing == 1)         scalarMode = maximumMode;
                else if (scalarCompositing == 2)    scalarMode = averageMode;

                openvdb::points::RasterMode          vectorMode = accumulateMode;
                if (vectorCompositing == 1)         vectorMode = averageMode;

                const float densityScale = static_cast<float>(evalFloat("densityscale", 0, time));
                const float scale = 1.0f;

                openvdb::points::FrustumRasterizer<
                    openvdb::points::PointDataGrid> rasterizer(settings, mask, &boss);

                size_t iterations = pointGrids.size();

                if (mergeVDBPoints) {
                    pointGrids.addGridsToRasterizer(rasterizer);
                    iterations = 1;
                }

                bool streaming = pointGrids.streaming();

                for (size_t i = 0; i < iterations; i++) {

                    if (!mergeVDBPoints) {
                        pointGrids.addGridToRasterizer(rasterizer, i);
                    }

                    openvdb::GridBase::Ptr velocity;

                    // rasterize velocity as the first attribute (but retain ordering)
                    // otherwise the velocity attribute can be discarded through streaming when
                    // applying geometry motion blur

                    if (std::find(vectorAttribNames.begin(), vectorAttribNames.end(), velocityAttribute) !=
                        vectorAttribNames.end()) {
                        velocity = rasterizer.rasterizeAttribute(velocityAttribute, vectorMode, reduceMemory, scale, rasterGroups);

                        // need to deep-copy input grids again if caches are being discarded
                        if (streaming && reduceMemory) {
                            if (mergeVDBPoints)     pointGrids.addGridsToRasterizer(rasterizer);
                            else                    pointGrids.addGridToRasterizer(rasterizer, i);
                        }
                    }

                    // rasterize density

                    if (createDensity) {
                        auto density = rasterizer.rasterizeDensity(densityAttribute, densityMode, reduceMemory, densityScale, rasterGroups);
                        outputGrids.push_back(density);

                        // need to deep-copy input grids again if caches are being discarded
                        if (streaming && reduceMemory) {
                            if (mergeVDBPoints)     pointGrids.addGridsToRasterizer(rasterizer);
                            else                    pointGrids.addGridToRasterizer(rasterizer, i);
                        }
                    }

                    // rasterize mask

                    if (createMask) {
                        auto mask = rasterizer.rasterizeMask<openvdb::BoolGrid>(reduceMemory, rasterGroups);
                        outputGrids.push_back(mask);
                    }

                    // rasterize scalar attributes

                    for (const auto& name : scalarAttribNames) {
                        auto scalar = rasterizer.rasterizeAttribute(name, scalarMode, reduceMemory, scale, rasterGroups);
                        outputGrids.push_back(scalar);

                        // need to deep-copy input grids again if caches are being discarded
                        if (streaming && reduceMemory) {
                            if (mergeVDBPoints)     pointGrids.addGridsToRasterizer(rasterizer);
                            else                    pointGrids.addGridToRasterizer(rasterizer, i);
                        }
                    }

                    // rasterize vector attributes

                    for (const auto& name : vectorAttribNames) {
                        // velocity attribute is rasterized first to ensure streaming doesn't discard it
                        if (name == velocityAttribute) {
                            if (velocity) {
                                outputGrids.push_back(velocity);
                            }
                        }
                        else {
                            auto vector = rasterizer.rasterizeAttribute(name, vectorMode, reduceMemory, scale, rasterGroups);
                            outputGrids.push_back(vector);

                            // need to deep-copy input grids again if caches are being discarded
                            if (streaming && reduceMemory) {
                                if (mergeVDBPoints)     pointGrids.addGridsToRasterizer(rasterizer);
                                else                    pointGrids.addGridToRasterizer(rasterizer, i);
                            }
                        }
                    }
                }
            }

            // Export VDB grids

            for (size_t n = 0, N = outputGrids.size(); n < N && !boss.wasInterrupted(); ++n) {
                hvdb::createVdbPrimitive(*gdp, outputGrids[n]);
            }

        } else {
            addWarning(SOP_MESSAGE, "No output selected");
        }

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}
