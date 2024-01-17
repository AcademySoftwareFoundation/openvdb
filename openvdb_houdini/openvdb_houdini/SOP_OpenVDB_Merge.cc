// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file SOP_OpenVDB_Merge.cc
///
/// @authors Dan Bailey
///
/// @brief Merge OpenVDB grids.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/PointUtils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <GEO/GEO_PrimVDB.h>

#include <openvdb/points/PointDataGrid.h> // points::PointDataGrid
#include <openvdb/tools/GridTransformer.h> // tools::replaceToMatch()
#include <openvdb/tools/LevelSetRebuild.h> // tools::doLevelSetRebuild()
#include <openvdb/tools/Merge.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_Version.h>
#include <UT/UT_ConcurrentVector.h>
#include <stdexcept>
#include <string>
#include <vector>


using namespace openvdb;

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Merge final : public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Merge(OP_Network*, const char* name, OP_Operator*);
    ~SOP_OpenVDB_Merge() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    class Cache: public SOP_VDBCacheOptions
    {
    public:
        fpreal getTime() const { return mTime; }
    protected:
        OP_ERROR cookVDBSop(OP_Context&) override;
    private:
        fpreal mTime = 0.0;
    };
};


void
newSopOperator(OP_OperatorTable* table)
{
    if (table == nullptr) return;

    hutil::ParmList parms;

    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setTooltip("Specify a subset of the input VDBs to be modified.")
        .setChoiceList(&hutil::PrimGroupMenuInput1)
        .setDocumentation(
            "A subset of the input VDBs to be modified"
            " (see [specifying volumes|/model/volumes#group])"));

    parms.add(hutil::ParmFactory(PRM_STRING, "collation", "Collation")
        .setDefault("name")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "name",             "Name",
            "primitive_number", "Primitive Number",
            "all",              "All"
        })
        .setTooltip(
            "Criteria under which groups of grids are merged.")
        .setDocumentation(
            "The criteria under which groups of grids are merged. In addition to these collation"
            " options, only grids with the same class (fog volume, level set, etc) and"
            " value type (float, int, etc) are merged.\n\n"
            "Name:\n"
            "   Collate VDBs with the same grid name.\n\n"
            "Primitive Number:\n"
            "   Collate first primitives from each input, then second primitives from each input, etc.\n\n"
            "None:\n"
            "   Collate all VDBs (provided they have the same class and value type).\n\n"));

    // // Menu of resampling options
    // parms.add(hutil::ParmFactory(PRM_STRING, "resample", "Resample VDBs")
    //     .setDefault("first")
    //     .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
    //         "first",        "To Match First VDB"
    //     })
    //     .setTypeExtended(PRM_TYPE_JOIN_PAIR)
    //     .setDocumentation("Specify which grid to use as reference when resampling."));

    // Menu of resampling interpolation order options
    parms.add(hutil::ParmFactory(PRM_ORD, "resampleinterp", "Interpolation")
        .setDefault(PRMoneDefaults)
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "point",     "Nearest",
            "linear",    "Linear",
            "quadratic", "Quadratic"
        })
        .setTooltip(
            "Specify the type of interpolation to be used when\n"
            "resampling one VDB to match the other's transform.")
        .setDocumentation(
            "The type of interpolation to be used when resampling one VDB"
            " to match the other's transform\n\n"
            "Nearest neighbor interpolation is fast but can introduce noticeable"
            " sampling artifacts.  Quadratic interpolation is slow but high-quality."
            " Linear interpolation is intermediate in speed and quality."));

    parms.beginSwitcher("Group1");

    parms.addFolder("Merge Operation");

    parms.add(hutil::ParmFactory(PRM_STRING, "op_fog", "Fog VDBs")
        .setDefault("add")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "none",             "None            ",
            "add",              "Add"
        })
        .setTooltip("Merge operation for Fog VDBs.")
        .setDocumentation(
            "Merge operation for Fog VDBs.\n\n"
            "None:\n"
            "   Leaves input fog VDBs unchanged.\n\n"
            "Add:\n"
            "   Generate the sum of all fog VDBs within the same collation.\n\n"));

    parms.add(hutil::ParmFactory(PRM_STRING, "op_sdf", "SDF VDBs")
        .setDefault("sdfunion")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, {
            "none",             "None            ",
            "sdfunion",         "SDF Union",
            "sdfintersect",     "SDF Intersect"
        })
        .setTooltip("Merge operation for SDF VDBs.")
        .setDocumentation(
            "Merge operation for SDF VDBs.\n\n"
            "None:\n"
            "    Leaves input SDF VDBs unchanged.\n\n"
            "SDF Union:\n"
            "    Generate the union of all signed distance fields within the same collation.\n\n"
            "SDF Intersection:\n"
            "    Generate the intersection of all signed distance fields within the same collation.\n\n"));

    parms.endSwitcher();

    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "resample", "Resample VDBs"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("VDB Merge", SOP_OpenVDB_Merge::factory, parms, *table)
#ifndef SESI_OPENVDB
        .setInternalName("DW_OpenVDBMerge")
#endif
        .setObsoleteParms(obsoleteParms)
        .addInput("VDBs To Merge")
        .setMaxInputs()
        .setVerb(SOP_NodeVerb::COOK_INPLACE, []() { return new SOP_OpenVDB_Merge::Cache; })
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Merge VDB SDF grids.\"\"\"\n\
\n\
@overview\n\
\n\
Merges VDB SDF grids.\n\
\n\
Unlike the VDB Combine SOP, it provides a multi-input so can merge more than two grids without needing an \
additional merge SOP. \n\
Float and double SDF VDBs and fog volume VDBs can be merged, all other VDB grids are left unchanged. \n\
\n\
Grids with different transforms will be resampled to match the first VDB in the collation group. \n\
\n\
@related\n\
\n\
- [OpenVDB Combine|Node:sop/DW_OpenVDBCombine]\n\
- [Node:sop/merge]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


OP_Node*
SOP_OpenVDB_Merge::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Merge(net, name, op);
}


SOP_OpenVDB_Merge::SOP_OpenVDB_Merge(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


namespace {


using PrimitiveNumberMap = std::unordered_map<GEO_PrimVDB::UniqueId, int>;


// The merge key stores the grid class and value type and optionally the grid name
// depending on the requested collation and is an abstraction used to test whether
// different grids should be merged together or not
struct MergeKey
{
    enum Collation
    {
        NameClassType = 0,
        NumberClassType,
        ClassType
    };

    explicit MergeKey(  const GU_PrimVDB& vdbPrim,
                        const Collation collation,
                        const PrimitiveNumberMap& primitiveNumbers)
        : name(collation == NameClassType ? vdbPrim.getGridName() : "")
        , number(collation == NumberClassType ? primitiveNumbers.at(vdbPrim.getUniqueId()) : -1)
        , valueType(vdbPrim.getStorageType())
        , gridClass(vdbPrim.getGrid().getGridClass()) { }

    inline bool operator==(const MergeKey& rhs) const
    {
        return (name == rhs.name &&
                valueType == rhs.valueType &&
                gridClass == rhs.gridClass &&
                number == rhs.number);
    }

    inline bool operator!=(const MergeKey& rhs) const { return !this->operator==(rhs); }

    inline bool operator<(const MergeKey& rhs) const
    {
        if (number < rhs.number)                                                return true;
        if (number == rhs.number) {
            if (name < rhs.name)                                                return true;
            if (name == rhs.name) {
                if (valueType < rhs.valueType)                                  return true;
                if (valueType == rhs.valueType && gridClass < rhs.gridClass)    return true;
            }
        }
        return false;
    }

    std::string name = "";
    int number = 0;
    UT_VDBType valueType = UT_VDB_INVALID;
    openvdb::GridClass gridClass = GRID_UNKNOWN;
}; // struct MergeKey


// This class reads all grids from the vdbPrims and constVdbPrims arrays and merges those
// that have the same merge key using the requested merge mode for each grid type/class.
// It chooses whether to deep-copy or steal grids and will resample if needed adding all
// resulting grids to the output. It is thread-safe and is designed to be called from
// multiple threads in parallel so as to be able to concurrently merge different input
// grids using different merge operations at once.
struct MergeOp
{
    enum Resample
    {
        Nearest = 0,
        Linear,
        Quadratic
    };

    struct OutputGrid
    {
        explicit OutputGrid(GridBase::Ptr _grid, GEO_PrimVDB* _primitive = nullptr,
            const GEO_PrimVDB* _referencePrimitive = nullptr)
            : grid(_grid)
            , primitive(_primitive)
            , referencePrimitive(_referencePrimitive) { }

        GridBase::Ptr grid;
        GEO_PrimVDB* primitive = nullptr;
        const GEO_PrimVDB* referencePrimitive = nullptr;
    };

    using OutputGrids = std::deque<OutputGrid>;

    using StringRemapType = std::unordered_map<std::string, std::string>;

    SOP_OpenVDB_Merge::Cache* self;
    openvdb::util::NullInterrupter& interrupt;
    StringRemapType opRemap;
    std::deque<GU_PrimVDB*> vdbPrims;
    std::deque<const GU_PrimVDB*> constVdbPrims;
    UT_ConcurrentVector<GU_PrimVDB*> vdbPrimsToRemove;
    PrimitiveNumberMap primNumbers;

    explicit MergeOp(openvdb::util::NullInterrupter& _interrupt): self(nullptr), interrupt(_interrupt) { }

    inline std::string getOp(const MergeKey& key) const
    {
        std::string op;

        if (key.gridClass == openvdb::GRID_LEVEL_SET) {
            if (key.valueType == UT_VDB_FLOAT || key.valueType == UT_VDB_DOUBLE) {
                op = opRemap.at("op_sdf");
            }
        } else if (key.gridClass == openvdb::GRID_FOG_VOLUME) {
            if (key.valueType == UT_VDB_FLOAT || key.valueType == UT_VDB_DOUBLE) {
                op = opRemap.at("op_fog");
            }
        }

        if (op == "none")   op = "";

        return op;
    }

    template<typename GridT>
    typename GridT::Ptr resampleToMatch(const GridT& src, const GridT& ref, int order)
    {
        using ValueT = typename GridT::ValueType;
        const ValueT ZERO = openvdb::zeroVal<ValueT>();

        const openvdb::math::Transform& refXform = ref.constTransform();

        typename GridT::Ptr dest;
        if (src.getGridClass() == openvdb::GRID_LEVEL_SET) {
            // For level set grids, use the level set rebuild tool to both resample the
            // source grid to match the reference grid and to rebuild the resulting level set.
            const bool refIsLevelSet = ref.getGridClass() == openvdb::GRID_LEVEL_SET;
            OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
            const ValueT halfWidth = refIsLevelSet
                ? ValueT(ZERO + ref.background() * (1.0 / ref.voxelSize()[0]))
                : ValueT(src.background() * (1.0 / src.voxelSize()[0]));
            OPENVDB_NO_TYPE_CONVERSION_WARNING_END

            if (!openvdb::math::isFinite(halfWidth)) {
                std::stringstream msg;
                msg << "Resample to match: Illegal narrow band width = " << halfWidth
                    << ", caused by grid '" << src.getName() << "' with background "
                    << ref.background();
                throw std::invalid_argument(msg.str());
            }

            try {
                dest = openvdb::tools::doLevelSetRebuild(src, /*iso=*/ZERO,
                    /*exWidth=*/halfWidth, /*inWidth=*/halfWidth, &refXform, &interrupt);
            } catch (openvdb::TypeError&) {
                self->addWarning(SOP_MESSAGE, ("skipped rebuild of level set grid "
                    + src.getName() + " of type " + src.type()).c_str());
                dest.reset();
            }
        }
        if (!dest && src.constTransform() != refXform) {
            // For non-level set grids or if level set rebuild failed due to an unsupported
            // grid type, use the grid transformer tool to resample the source grid to match
            // the reference grid.
            dest = src.copyWithNewTree();
            dest->setTransform(refXform.copy());
            switch (order) {
                case 0: tools::resampleToMatch<tools::PointSampler>(src, *dest, interrupt); break;
                case 1: tools::resampleToMatch<tools::BoxSampler>(src, *dest, interrupt); break;
                case 2: tools::resampleToMatch<tools::QuadraticSampler>(src, *dest, interrupt); break;
            }
        }
        return dest;
    }

    template <typename GridT>
    OutputGrids mergeTypedNoop( const MergeKey& mergeKey,
                                const MergeKey::Collation& collationKey)
    {
        using TreeT = typename GridT::TreeType;

        // this method steals or deep-copies grids from the multi-input
        // stealing can only be performed for the first input in the multi-input,
        // provided that the tree is unique, all other inputs are read-only

        OutputGrids result;

        std::deque<tools::TreeToMerge<TreeT>> trees;

        tbb::task_group tasks;

        auto hasUniqueTree = [&](GU_PrimVDB* vdbPrim)
        {
            return vdbPrim->getConstGridPtr()->isTreeUnique();
        };

        auto stealTree = [&](auto& gridBase, GU_PrimVDB* vdbPrim = nullptr)
        {
            result.emplace_back(gridBase, vdbPrim);
        };

        auto copyTree = [&](auto& gridBase, GU_PrimVDB* vdbPrim = nullptr,
            const GU_PrimVDB* constVdbPrim = nullptr)
        {
            auto grid = GridBase::grid<GridT>(gridBase);
            // insert an empty shared pointer and asynchronously replace with a deep copy
            result.emplace_back(GridBase::Ptr(), vdbPrim, constVdbPrim);
            OutputGrid& output = result.back();
            tasks.run(
                [&, grid] {
                    output.grid = grid->deepCopy();
                }
            );
        };

        for (GU_PrimVDB* vdbPrim : vdbPrims) {
            MergeKey key(*vdbPrim, collationKey, primNumbers);
            if (key != mergeKey)    continue;

            if (hasUniqueTree(vdbPrim)) {
                GridBase::Ptr gridBase = vdbPrim->getGridPtr();
                stealTree(gridBase, vdbPrim);
            } else {
                GridBase::ConstPtr gridBase = vdbPrim->getConstGridPtr();
                copyTree(gridBase, vdbPrim);
            }
        }

        for (const GU_PrimVDB* constVdbPrim : constVdbPrims) {
            MergeKey key(*constVdbPrim, collationKey, primNumbers);
            if (key != mergeKey)    continue;

            GridBase::ConstPtr gridBase = constVdbPrim->getConstGridPtr();
            copyTree(gridBase, nullptr, constVdbPrim);
        }

        if (interrupt.wasInterrupted())
        {
            tasks.cancel();
            return OutputGrids();
        }

        // resampling and deep copying of trees is done in parallel asynchronously,
        // now synchronize to ensure all these tasks have completed
        tasks.wait();

        if (interrupt.wasInterrupted())     return OutputGrids();

        return result;
    }

    template <typename GridT>
    OutputGrids mergeTyped( const MergeKey& mergeKey,
                            const MergeKey::Collation& collationKey)
    {
        using TreeT = typename GridT::TreeType;

        // this method steals or deep-copies grids from the multi-input and resamples if necessary
        // stealing can only be performed for the first input in the multi-input,
        // provided that the tree is unique, all other inputs are read-only
        // const GU_PrimVDBs cannot be stolen

        OutputGrids result;

        const std::string op = this->getOp(mergeKey);

        const int samplingOrder = static_cast<int>(
            self->evalInt("resampleinterp", 0, self->getTime()));

        GridBase::Ptr reference;

        std::deque<tools::TreeToMerge<TreeT>> trees;

        tbb::task_group tasks;

        auto hasUniqueTree = [&](GU_PrimVDB* vdbPrim)
        {
            return vdbPrim->getConstGridPtr()->isTreeUnique();
        };

        auto stealTree = [&](auto& gridBase, GU_PrimVDB* vdbPrim = nullptr)
        {
            result.emplace_back(gridBase, vdbPrim);
            if (!reference)  reference = gridBase;
        };

        auto copyTree = [&](auto& gridBase, GU_PrimVDB* vdbPrim = nullptr, const GU_PrimVDB* constVdbPrim = nullptr)
        {
            auto grid = GridBase::grid<GridT>(gridBase);
            if (!reference)  reference = grid->copyWithNewTree();
            // insert a reference and asynchronously replace with a deep copy
            result.emplace_back(reference, vdbPrim, constVdbPrim);
            OutputGrid& output = result.back();
            tasks.run(
                [&, grid] {
                    output.grid = grid->deepCopy();
                }
            );
        };

        auto addConstTree = [&](auto& gridBase, GU_PrimVDB* vdbPrim = nullptr)
        {
            auto grid = GridBase::grid<GridT>(gridBase);
            if (grid->constTransform() == reference->constTransform()) {
                trees.emplace_back(grid->tree(), openvdb::DeepCopy());
            } else {
                // insert an empty tree and asynchronously replace with a resampled tree
                trees.emplace_back(typename TreeT::Ptr(new TreeT), openvdb::Steal());
                tools::TreeToMerge<TreeT>& treeToMerge = trees.back();
                tasks.run(
                    [&, grid] {
                        auto refGrid = GridBase::grid<GridT>(reference);
                        auto newGrid = this->resampleToMatch(*grid, *refGrid, samplingOrder);
                        treeToMerge.reset(newGrid->treePtr(), openvdb::Steal());
                    }
                );
            }
            if (vdbPrim)    vdbPrimsToRemove.push_back(vdbPrim);
        };

        for (GU_PrimVDB* vdbPrim : vdbPrims) {
            MergeKey key(*vdbPrim, collationKey, primNumbers);
            if (key != mergeKey)    continue;

            if (hasUniqueTree(vdbPrim)) {
                GridBase::Ptr gridBase = vdbPrim->getGridPtr();
                if ((!reference) || op.empty()) stealTree(gridBase, vdbPrim);
                else                            addConstTree(gridBase, vdbPrim);
            } else {
                GridBase::ConstPtr gridBase = vdbPrim->getConstGridPtr();
                if ((!reference) || op.empty()) copyTree(gridBase, vdbPrim);
                else                            addConstTree(gridBase, vdbPrim);
            }
        }
        // const GU_PrimVDBs cannot be stolen
        for (const GU_PrimVDB* constVdbPrim : constVdbPrims) {
            MergeKey key(*constVdbPrim, collationKey, primNumbers);
            if (key != mergeKey)    continue;

            GridBase::ConstPtr gridBase = constVdbPrim->getConstGridPtr();
            if ((!reference) || op.empty())     copyTree(gridBase, nullptr, constVdbPrim);
            else                                addConstTree(gridBase);
        }

        if (interrupt.wasInterrupted())
        {
            tasks.cancel();
            return OutputGrids();
        }

        // resampling and deep copying of trees is done in parallel asynchronously,
        // now synchronize to ensure all these tasks have completed
        tasks.wait();

        if (interrupt.wasInterrupted())     return OutputGrids();

        // perform merge

        if (result.size() == 1 && !trees.empty()) {
            auto grid = GridBase::grid<GridT>(result.front().grid);
            tree::DynamicNodeManager<TreeT> nodeManager(grid->tree());
            if (op == "sdfunion") {
                nodeManager.foreachTopDown(tools::CsgUnionOp<TreeT>(trees));
            } else if (op == "sdfintersect") {
                nodeManager.foreachTopDown(tools::CsgIntersectionOp<TreeT>(trees));
            } else if (op == "add") {
                nodeManager.foreachTopDown(tools::SumMergeOp<TreeT>(trees));
            }
        }

        return result;
    }

    OutputGrids merge(  const MergeKey& key,
                        const MergeKey::Collation& collationKey)
    {
        // only float and double grids can be merged currently

        if (key.valueType == UT_VDB_FLOAT) {
            return this->mergeTyped<FloatGrid>(key, collationKey);
        } else if (key.valueType == UT_VDB_DOUBLE) {
            return this->mergeTyped<DoubleGrid>(key, collationKey);
        } else if (key.valueType == UT_VDB_INT32) {
            return this->mergeTypedNoop<Int32Grid>(key, collationKey);
        } else if (key.valueType == UT_VDB_INT64) {
            return this->mergeTypedNoop<Int64Grid>(key, collationKey);
        } else if (key.valueType == UT_VDB_BOOL) {
            return this->mergeTypedNoop<BoolGrid>(key, collationKey);
        } else if (key.valueType == UT_VDB_VEC3F) {
            return this->mergeTypedNoop<Vec3SGrid>(key, collationKey);
        } else if (key.valueType == UT_VDB_VEC3D) {
            return this->mergeTypedNoop<Vec3DGrid>(key, collationKey);
        } else if (key.valueType == UT_VDB_VEC3I) {
            return this->mergeTypedNoop<Vec3IGrid>(key, collationKey);
        } else if (key.valueType == UT_VDB_POINTDATA) {
            return this->mergeTypedNoop<points::PointDataGrid>(key, collationKey);
        }

        return OutputGrids();
    }

}; // struct MergeOp

} // namespace


OP_ERROR
SOP_OpenVDB_Merge::Cache::cookVDBSop(OP_Context& context)
{
    try {
        mTime = context.getTime();

        const std::string groupName = evalStdString("group", mTime);

        MergeKey::Collation collationKey = MergeKey::ClassType;
        const std::string collation = evalStdString("collation", mTime);
        if (collation == "name")                    collationKey = MergeKey::NameClassType;
        else if (collation == "primitive_number")   collationKey = MergeKey::NumberClassType;

        hvdb::HoudiniInterrupter boss("Merging VDBs");

        MergeOp mergeOp(boss.interrupter());
        mergeOp.self = this;
        mergeOp.opRemap["op_sdf"] = evalStdString("op_sdf", mTime);
        mergeOp.opRemap["op_fog"] = evalStdString("op_fog", mTime);

        // extract non-const VDB primitives from first input

        hvdb::VdbPrimIterator vdbIt(gdp, matchGroup(*gdp, groupName));
        int number = 0;
        for (; vdbIt; ++vdbIt) {
            GU_PrimVDB* vdbPrim = *vdbIt;
            mergeOp.vdbPrims.push_back(vdbPrim);
            // store primitive index for each primitive keyed by the unique id
            mergeOp.primNumbers[vdbPrim->getUniqueId()] = number++;
        }

        // extract const VDB primitives from second or more input

        for (int i = 1; i < nInputs(); i++) {
            const GU_Detail* pointsGeo = inputGeo(i);
            number = 0;
            hvdb::VdbPrimCIterator vdbIt(pointsGeo, matchGroup(*pointsGeo, groupName));
            for (; vdbIt; ++vdbIt) {
                const GU_PrimVDB* constVdbPrim = *vdbIt;
                mergeOp.constVdbPrims.push_back(constVdbPrim);
                // store primitive index for each primitive keyed by the unique id
                mergeOp.primNumbers[constVdbPrim->getUniqueId()] = number++;
            }
        }

        // extract all merge keys

        std::set<MergeKey> uniqueKeys;

        for (GU_PrimVDB* vdbPrim : mergeOp.vdbPrims) {
            MergeKey key(*vdbPrim, collationKey, mergeOp.primNumbers);
            uniqueKeys.insert(key);
        }
        for (const GU_PrimVDB* constVdbPrim : mergeOp.constVdbPrims) {
            MergeKey key(*constVdbPrim, collationKey, mergeOp.primNumbers);
            uniqueKeys.insert(key);
        }

        std::vector<MergeKey> keys(uniqueKeys.begin(), uniqueKeys.end());

        // iterate over each merge key in parallel and perform merge operations

        std::vector<MergeOp::OutputGrids> outputGridsArray;
        outputGridsArray.resize(keys.size());

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, keys.size()),
            [&](tbb::blocked_range<size_t>& range)
            {
                for (size_t i = range.begin(); i < range.end(); i++) {
                    outputGridsArray[i] = mergeOp.merge(keys[i], collationKey);
                }
            }
        );

        // replace primitives from first input, create primitives from second or more input

        for (MergeOp::OutputGrids& outputGrids : outputGridsArray) {
            for (MergeOp::OutputGrid& outputGrid : outputGrids) {
                GridBase::Ptr grid = outputGrid.grid;
                if (!grid)  continue;
                GEO_PrimVDB* primitive = outputGrid.primitive;
                if (primitive)  hvdb::replaceVdbPrimitive(*gdp, grid, *primitive);
                else {
                    const GEO_PrimVDB* referencePrimitive = outputGrid.referencePrimitive;
                    GU_PrimVDB::buildFromGrid(*gdp, grid,
                        /*copyAttrsFrom=*/bool(referencePrimitive) ? referencePrimitive : nullptr,
                        /*gridName=*/bool(referencePrimitive) ? referencePrimitive->getGridName() : nullptr);
                }
            }
        }

        // remove old primitives that have now been merged into another

        for (GU_PrimVDB* primitive : mergeOp.vdbPrimsToRemove) {
            if (primitive)  gdp->destroyPrimitive(*primitive, /*andPoints=*/true);
        }

        if (boss.wasInterrupted()) addWarning(SOP_MESSAGE, "processing was interrupted");
        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}
