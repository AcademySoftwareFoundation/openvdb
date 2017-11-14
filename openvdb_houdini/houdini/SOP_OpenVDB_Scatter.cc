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
/// @file SOP_OpenVDB_Scatter.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Scatter points on a VDB grid, either by fixed count or by
/// global or local point density.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/points/PointDelete.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Scatter: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Scatter(OP_Network* net, const char* name, OP_Operator* op);
    ~SOP_OpenVDB_Scatter() override {}

    static OP_Node* factory(OP_Network*, const char*, OP_Operator*);

protected:
    OP_ERROR cookMySop(OP_Context&) override;
    bool updateParmsFlags() override;
    void resolveObsoleteParms(PRM_ParmList*) override;
};


////////////////////////////////////////


// Build UI and register this operator.
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
            "A subset of the input VDBs to be processed"
            " (see [specifying volumes|/model/volumes#group])"));

    // Export VDBs
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "keep", "Keep Input VDBs")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, the output will contain the input VDB grids."));

    // Enable VDB Points
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "vdbpoints", "Scatter VDB Points")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Generate VDB Points instead of Houdini Points."));

    // VDB points grid name
    {
        char const * const items[] = {
            "keep",     "Keep Original Name",
            "append",   "Add Suffix",
            "replace",  "Custom Name",
            nullptr
        };

        parms.add(hutil::ParmFactory(PRM_ORD, "outputname", "Output Name")
            .setDefault(PRMzeroDefaults)
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setTooltip("Output VDB naming scheme")
            .setDocumentation(
                "Give the output VDB Points the same name as the input VDB,"
                " or add a suffix to the input name, or use a custom name."));

        parms.add(hutil::ParmFactory(PRM_STRING, "customname", "Custom Name")
            .setDefault("points")
            .setTooltip("The suffix or custom name to be used"));
    }

    // Group scattered points
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "dogroup", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setDefault(PRMzeroDefaults));

    // Scatter group name
    parms.add(hutil::ParmFactory(PRM_STRING, "sgroup", "Scatter Group")
        .setDefault(0, "scatter")
        .setTooltip("If enabled, add scattered points to the group with the given name."));

    // Random seed
    parms.add(hutil::ParmFactory(PRM_INT_J, "seed", "Random Seed")
        .setDefault(PRMzeroDefaults)
        .setTooltip("Specify the random number seed."));

    // Spread
    parms.add(hutil::ParmFactory(PRM_FLT_J, "spread", "Spread")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)
        .setTooltip(
            "How far each point may be displaced from the center of its voxel or tile\n\n"
            "A value of zero means that the point is placed exactly at the center."
            " A value of one means that the point can be placed randomly anywhere"
            " inside the voxel or tile."));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep1", ""));

    // Mode for point scattering
    char const * const items[] = {
        "count",            "Point Total",
        "density",          "Point Density",
        "pointspervoxel",   "Points Per Voxel",
        nullptr
    };
    parms.add(hutil::ParmFactory(PRM_ORD, "pointmode", "Mode")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
        .setTooltip(
            "How to determine the number of points to scatter\n\n"
            "Point Total:\n"
            "    Specify a fixed, total point count.\n"
            "Point Density:\n"
            "    Specify the number of points per unit volume.\n"
            "Points Per Voxel:\n"
            "    Specify the number of points per voxel."));

    // Point count
    parms.add(hutil::ParmFactory(PRM_INT_J, "count", "Count")
        .setDefault(5000)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10000)
        .setTooltip("Specify the total number of points to scatter."));

    // Point density
    parms.add(hutil::ParmFactory(PRM_FLT_J, "density", "Density")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 1000)
        .setTooltip("The number of points per unit volume (when __Mode__ is Point Density)"));

    // Toggle to use voxel value as local point density multiplier
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "multiply", "Scale Density by Voxel Values")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "If enabled, use voxel values as local multipliers for the point density."
            " Has no impact if interior scattering is enabled."));

    // Points per voxel
    parms.add(hutil::ParmFactory(PRM_FLT_J , "ppv", "Count")
        .setDefault(8)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10)
        .setTooltip("Specify the number of points per voxel."));

    // Toggle to scatter inside level sets
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "interior", "Scatter Points Inside Level Sets")
        .setDefault(PRMzeroDefaults)
        .setTooltip(
            "If enabled, scatter points in the interior region of a level set."
            " Otherwise, scatter points only in the narrow band."));

    // scatter to the iso surface for interior and vdb points scattering
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "cliptoisosurface", "Clip To Isosurface")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, removes scattered points outside the zero iso surface."
                    " Only available when scattering VDB Points."));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep2", ""));

    // Verbose output toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose")
        .setDefault(PRMzeroDefaults)
        .setTooltip("If enabled, print the sequence of operations to the terminal."));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_ORD| PRM_TYPE_JOIN_NEXT, "pointMode", "Point"));

    // Register the SOP.
    hvdb::OpenVDBOpFactory("OpenVDB Scatter", SOP_OpenVDB_Scatter::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("VDB on which points will be scattered")
        .setDocumentation("\
#icon: COMMON/openvdb\n\
#tags: vdb\n\
\n\
\"\"\"Scatter Houdini or VDB points on a VDB volume.\"\"\"\n\
\n\
@overview\n\
\n\
This node scatters points randomly on or inside a VDB volume. It can produce a set\n\
of Houdini points or a VDB Points grid for every provided VDB. Each VDB Points grid\n\
created will copy the transform and topology of each source VDB.\n\
The number of points generated can be specified either by fixed count\n\
or by global or local point density.\n\
\n\
For level set VDBs, points can be scattered either throughout the interior\n\
of the volume or only in the\n\
[narrow band|http://www.openvdb.org/documentation/doxygen/overview.html#secGrid]\n\
region surrounding the zero crossing.\n\
\n\
@related\n\
- [Node:sop/scatter]\n\
\n\
@examples\n\
\n\
See [openvdb.org|http://www.openvdb.org/download/] for source code\n\
and usage examples.\n");
}


void
SOP_OpenVDB_Scatter::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    this->resolveRenamedParm(*obsoleteParms, "pointMode", "pointmode");

    // Delegate to the base class.
    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


bool
SOP_OpenVDB_Scatter::updateParmsFlags()
{
    bool changed = false;
    const auto vdbpoints = evalInt("vdbpoints", /*idx=*/0, /*time=*/0);
    const auto pmode = evalInt("pointmode", /*idx=*/0, /*time=*/0);
    const auto interior = evalInt("interior", /*idx=*/0, /*time=*/0);

    changed |= setVisibleState("count",      (0 == pmode));
    changed |= setVisibleState("density",    (1 == pmode));
    changed |= setVisibleState("multiply",   (1 == pmode));
    changed |= setVisibleState("ppv",        (2 == pmode));
    changed |= setVisibleState("name",       (1 == vdbpoints));
    changed |= setVisibleState("outputname", (1 == vdbpoints));
    changed |= setVisibleState("customname", (1 == vdbpoints));
    changed |= setVisibleState("cliptoisosurface", interior == 1 && vdbpoints == 1);

    const bool useCustomName = evalInt("outputname", 0, 0) != 0;
    changed |= enableParm("customname", useCustomName);

    const auto dogroup = evalInt("dogroup", 0, 0);
    changed |= enableParm("sgroup", 1 == dogroup);

    return changed;
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Scatter::factory(OP_Network* net, const char* name, OP_Operator *op)
{
    return new SOP_OpenVDB_Scatter(net, name, op);
}


SOP_OpenVDB_Scatter::SOP_OpenVDB_Scatter(OP_Network* net, const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////


// Simple wrapper class required by openvdb::tools::UniformPointScatter and
// NonUniformPointScatter
class PointAccessor
{
public:
    PointAccessor(GEO_Detail* gdp) : mGdp(gdp)
    {
    }
    void add(const openvdb::Vec3R &pos)
    {
        GA_Offset ptoff = mGdp->appendPointOffset();
        mGdp->setPos3(ptoff, pos.x(), pos.y(), pos.z());
    }
protected:
    GEO_Detail*    mGdp;
};


struct BaseScatter
{
    using PositionArray =
        openvdb::points::TypedAttributeArray<openvdb::Vec3f, openvdb::points::NullCodec>;

    BaseScatter(const unsigned int seed,
                const float spread,
                hvdb::Interrupter* interrupter)
        : mPoints()
        , mSeed(seed)
        , mSpread(spread)
        , mInterrupter(interrupter) {}
    virtual ~BaseScatter() {}

    /// @brief Print information about the scattered points
    /// @parm name  A name to insert into the printed info
    /// @parm os    The output stream
    virtual void print(const std::string &name, std::ostream& os = std::cout) const
    {
        if (!mPoints) return;
        const openvdb::Index64 points = openvdb::points::pointCount(mPoints->tree());
        const openvdb::Index64 voxels = mPoints->activeVoxelCount();
        os << points << " points into " << voxels << " active voxels in \""
           << name << "\" corresponding to " << (double(points) / double(voxels))
           << " points per voxel." << std::endl;
    }

    inline openvdb::points::PointDataGrid::Ptr points()
    {
        assert(mPoints);
        return mPoints;
    }

protected:
    openvdb::points::PointDataGrid::Ptr mPoints;
    const unsigned int mSeed;
    const float mSpread;
    hvdb::Interrupter* mInterrupter;
}; // BaseScatter


struct VDBUniformScatter : public BaseScatter
{
    VDBUniformScatter(const openvdb::Index64 count,
                      const unsigned int seed,
                      const float spread,
                      hvdb::Interrupter* interrupter)
        : BaseScatter(seed, spread, interrupter)
        , mCount(count)
    {}

    template <typename GridT>
    inline void operator()(const GridT& grid)
    {
        using namespace openvdb::points;
        using PointDataGridT =
            openvdb::Grid<typename TreeConverter<typename GridT::TreeType>::Type>;
        mPoints = uniformPointScatter<GridT, std::mt19937, PositionArray, PointDataGridT,
                hvdb::Interrupter>(grid, mCount, mSeed, mSpread, mInterrupter);
    }

    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Uniformly scattered ";
        BaseScatter::print(name, os);
    }

    const openvdb::Index64 mCount;
}; // VDBUniformScatter


struct VDBDenseUniformScatter : public BaseScatter
{
    VDBDenseUniformScatter(const float pointsPerVoxel,
                           const unsigned int seed,
                           const float spread,
                           hvdb::Interrupter* interrupter)
        : BaseScatter(seed, spread, interrupter)
        , mPointsPerVoxel(pointsPerVoxel)
        {}

    template <typename GridT>
    inline void operator()(const GridT& grid)
    {
        using namespace openvdb::points;
        using PointDataGridT =
            openvdb::Grid<typename TreeConverter<typename GridT::TreeType>::Type>;
        mPoints = denseUniformPointScatter<GridT, std::mt19937, PositionArray, PointDataGridT,
                hvdb::Interrupter>(grid, mPointsPerVoxel, mSeed, mSpread, mInterrupter);
    }

    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Dense uniformly scattered ";
        BaseScatter::print(name, os);
    }

    const float mPointsPerVoxel;
}; // VDBDenseUniformScatter


struct VDBNonUniformScatter : public BaseScatter
{
    VDBNonUniformScatter(const float pointsPerVoxel,
                      const unsigned int seed,
                      const float spread,
                      hvdb::Interrupter* interrupter)
        : BaseScatter(seed, spread, interrupter)
        , mPointsPerVoxel(pointsPerVoxel)
    {}

    template <typename GridT>
    inline void operator()(const GridT& grid)
    {
        using namespace openvdb::points;
        using PointDataGridT =
            openvdb::Grid<typename TreeConverter<typename GridT::TreeType>::Type>;
        mPoints = nonUniformPointScatter<GridT, std::mt19937, PositionArray, PointDataGridT,
                hvdb::Interrupter>(grid, mPointsPerVoxel, mSeed, mSpread, mInterrupter);
    }

    void print(const std::string &name, std::ostream& os = std::cout) const
    {
        os << "Non-uniformly scattered ";
        BaseScatter::print(name, os);
    }

    const float mPointsPerVoxel;
}; // VDBNonUniformScatter


template <typename SurfaceGridT>
struct MarkPointsOutsideIso
{
    using GroupIndex = openvdb::points::AttributeSet::Descriptor::GroupIndex;
    using LeafManagerT = openvdb::tree::LeafManager<openvdb::points::PointDataTree>;
    using PositionHandleT =
        openvdb::points::AttributeHandle<openvdb::Vec3f, openvdb::points::NullCodec>;
    using SurfaceValueT = typename SurfaceGridT::ValueType;

    MarkPointsOutsideIso(const SurfaceGridT& grid,
                         const GroupIndex& deadIndex)
        : mGrid(grid)
        , mDeadIndex(deadIndex) {}

    void operator()(const LeafManagerT::LeafRange& range) const {
        openvdb::math::BoxStencil<const SurfaceGridT> stencil(mGrid);
        for (auto leaf = range.begin(); leaf; ++leaf)  {

            PositionHandleT::Ptr positionHandle =
                PositionHandleT::create(leaf->constAttributeArray(0));
            openvdb::points::GroupWriteHandle deadHandle =
                leaf->groupWriteHandle(mDeadIndex);

            for (auto voxel = leaf->cbeginValueOn(); voxel; ++voxel) {

                const openvdb::Coord& ijk = voxel.getCoord();
                const openvdb::Vec3d vec = ijk.asVec3d();

                for (auto iter = leaf->beginIndexVoxel(ijk); iter; ++iter) {
                    const openvdb::Index index = *iter;
                    const openvdb::Vec3d pos = openvdb::Vec3d(positionHandle->get(index)) + vec;

                    stencil.moveTo(pos);
                    if (stencil.interpolation(pos) > openvdb::zeroVal<SurfaceValueT>()) {
                        deadHandle.set(index, true);
                    }
                }
            }
        }
    }

private:
    const SurfaceGridT& mGrid;
    const GroupIndex& mDeadIndex;
}; // MarkPointsOutsideIso


template<typename OpType>
inline bool
process(const UT_VDBType type, const openvdb::GridBase& grid, OpType& op, const std::string* name)
{
    bool success(false);
    success = UTvdbProcessTypedGridTopology(type, grid, op);
    if (!success) {
#if UT_VERSION_INT >= 0x10000258 // 16.0.600 or later
        success = UTvdbProcessTypedGridPoint(type, grid, op);
#endif
    }
    if (name) op.print(*name);
    return success;
}


// Method to extract the interior mask before scattering points.
openvdb::GridBase::ConstPtr
extractInteriorMask(const openvdb::GridBase::ConstPtr grid, const float offset)
{
    if (grid->isType<openvdb::FloatGrid>()) {
        using MaskT = openvdb::Grid<openvdb::FloatTree::ValueConverter<bool>::Type>;
        const openvdb::FloatGrid& typedGrid = static_cast<const openvdb::FloatGrid&>(*grid);
        MaskT::Ptr mask = openvdb::tools::sdfInteriorMask(typedGrid, offset);
        return mask;

    } else if (grid->isType<openvdb::DoubleGrid>()) {
        using MaskT = openvdb::Grid<openvdb::DoubleTree::ValueConverter<bool>::Type>;
        const openvdb::DoubleGrid& typedGrid = static_cast<const openvdb::DoubleGrid&>(*grid);
        MaskT::Ptr mask = openvdb::tools::sdfInteriorMask(typedGrid, offset);
        return mask;
    }
    return openvdb::GridBase::ConstPtr();
}


// Remove VDB Points scattered outside of a level set
inline void
cullVDBPoints(openvdb::points::PointDataTree& tree,
              const openvdb::GridBase::ConstPtr grid)
{
    const auto leaf = tree.cbeginLeaf();
    if (leaf) {
        using GroupIndex = openvdb::points::AttributeSet::Descriptor::GroupIndex;
        openvdb::points::appendGroup(tree, "dead");
        const GroupIndex idx = leaf->attributeSet().groupIndex("dead");

        openvdb::tree::LeafManager<openvdb::points::PointDataTree>
            leafManager(tree);

        if (grid->isType<openvdb::FloatGrid>()) {
            const openvdb::FloatGrid& typedGrid =
                static_cast<const openvdb::FloatGrid&>(*grid);
            MarkPointsOutsideIso<openvdb::FloatGrid> mark(typedGrid, idx);
            tbb::parallel_for(leafManager.leafRange(), mark);
        }
        else if (grid->isType<openvdb::DoubleGrid>()) {
            const openvdb::DoubleGrid& typedGrid =
                static_cast<const openvdb::DoubleGrid&>(*grid);
            MarkPointsOutsideIso<openvdb::DoubleGrid> mark(typedGrid, idx);
            tbb::parallel_for(leafManager.leafRange(), mark);
        }
        openvdb::points::deleteFromGroup(tree, "dead");
    }
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Scatter::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        const GU_Detail* vdbgeo;
        if (1 == evalInt("keep", 0, time)) {
            // This does a deep copy of native Houdini primitives
            // but only a shallow copy of OpenVDB grids.
            duplicateSourceStealable(0, context);
            vdbgeo = gdp;
        }
        else {
            vdbgeo = inputGeo(0);
            gdp->clearAndDestroy();
        }

        const int seed = static_cast<int>(evalInt("seed", 0, time));
        const auto spread = static_cast<float>(evalFloat("spread", 0, time));
        const bool verbose = evalInt("verbose", 0, time) != 0;
        const openvdb::Index64 pointCount = evalInt("count", 0, time);
        const float ptsPerVox = static_cast<float>(evalFloat("ppv", 0, time));
        const bool interior = evalInt("interior", 0, time) != 0;
        const float density = static_cast<float>(evalFloat("density", 0, time));
        const bool multiplyDensity = evalInt("multiply", 0, time) != 0;
        const int outputName = static_cast<int>(evalInt("outputname", 0, time));

        // Get the group of grids to process.
        UT_String tmp;
        evalString(tmp, "group", 0, time);
        const GA_PrimitiveGroup* group = this->matchGroup(*vdbgeo, tmp.toStdString());

        evalString(tmp, "customname", 0, time);
        const std::string customName = tmp.toStdString();

        hvdb::Interrupter boss("Scattering points on VDBs");

        // Choose a fast random generator with a long period. Drawback here for
        // mt11213b is that it requires 352*sizeof(uint32) bytes.
        using RandGen = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19,
            0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>; // mt11213b
        RandGen mtRand(seed);

        const auto pmode = evalInt("pointmode", 0, time);
        const bool vdbPoints = evalInt("vdbpoints", 0, time) == 1;
        const bool clipPoints = vdbPoints && bool(evalInt("cliptoisosurface", 0, time));

        std::vector<std::string> emptyGrids;
        std::vector<openvdb::points::PointDataGrid::Ptr> pointGrids;
        PointAccessor pointAccessor(gdp);

        const GA_Offset firstOffset = gdp->getNumPointOffsets();

        // Process each VDB primitive (with a non-null grid pointer)
        // that belongs to the selected group.
        for (hvdb::VdbPrimCIterator primIter(vdbgeo, group); primIter; ++primIter) {

            // Retrieve a read-only grid pointer.
            UT_VDBType gridType = primIter->getStorageType();
            openvdb::GridBase::ConstPtr grid = primIter->getConstGridPtr();
            const std::string gridName = primIter.getPrimitiveName().toStdString();

            if (grid->empty()) {
                emptyGrids.push_back(gridName);
                continue;
            }

            const std::string* const name = verbose ? &gridName : nullptr;
            const openvdb::GridClass gridClass = grid->getGridClass();
            const bool isSignedDistance = (gridClass == openvdb::GRID_LEVEL_SET);
            bool performCull = false;

            if (interior && isSignedDistance) {
                float iso = 0.0f;
                if (clipPoints) {
                    const openvdb::Vec3d voxelSize = grid->voxelSize();
                    const double maxVoxelSize =
                        openvdb::math::Max(voxelSize.x(), voxelSize.y(), voxelSize.z());
                    iso = static_cast<float>(maxVoxelSize / 2.0);
                    performCull = true;
                }

                grid = extractInteriorMask(grid, iso);
                gridType = UT_VDB_BOOL;
                if (!grid) continue;
            }

            std::string vdbName;
            if (vdbPoints) {
                if (outputName == 0) vdbName = gridName;
                else if (outputName == 1) vdbName = gridName + customName;
                else vdbName = customName;
            }

            if (pmode == 0) { // fixed point count
                if (vdbPoints) { // vdb points
                    VDBUniformScatter scatter(pointCount, seed, spread, &boss);
                    if (process(gridType, *grid, scatter, name))  {
                        openvdb::points::PointDataGrid::Ptr points = scatter.points();
                        if (performCull) {
                            cullVDBPoints(points->tree(), primIter->getConstGridPtr());
                        }
                        points->setName(vdbName);
                        pointGrids.push_back(points);
                    }
                }
                else { // houdini points
                    openvdb::tools::UniformPointScatter<PointAccessor, RandGen, hvdb::Interrupter>
                        scatter(pointAccessor, pointCount, mtRand, spread, &boss);
                    process(gridType, *grid, scatter, name);
                }

            } else if (pmode == 1) { // points per unit volume
                if (multiplyDensity && !isSignedDistance) { // local density
                    if (vdbPoints) { // vdb points
                        const openvdb::Vec3d dim = openvdb::Vec3f(grid->transform().voxelSize());
                        VDBNonUniformScatter scatter(
                            static_cast<float>(density * dim.product()), seed, spread, &boss);
                        if (!UTvdbProcessTypedGridScalar(gridType, *grid, scatter)) {
                            throw std::runtime_error
                                ("Only scalar grids support voxel scaling of density");
                        }
                        openvdb::points::PointDataGrid::Ptr points = scatter.points();
                        points->setName(vdbName);
                        pointGrids.push_back(points);
                        if (verbose) scatter.print(gridName);
                    }
                    else { // houdini points
                        openvdb::tools::NonUniformPointScatter<
                            PointAccessor,RandGen,hvdb::Interrupter> scatter(
                                pointAccessor, density, mtRand, spread, &boss);

                        if (!UTvdbProcessTypedGridScalar(gridType, *grid, scatter)) {
                            throw std::runtime_error
                                ("Only scalar grids support voxel scaling of density");
                        }
                        if (verbose) scatter.print(gridName);
                    }
                } else { // global density
                    if (vdbPoints) { // vdb points
                        const openvdb::Vec3f dim = openvdb::Vec3f(grid->transform().voxelSize());
                        const openvdb::Index64 totalPointCount =
                            openvdb::Index64(density * dim.product()) * grid->activeVoxelCount();
                        VDBUniformScatter scatter(totalPointCount, seed, spread, &boss);
                        if (process(gridType, *grid, scatter, name))  {
                            openvdb::points::PointDataGrid::Ptr points = scatter.points();
                            if (performCull) {
                                cullVDBPoints(points->tree(), primIter->getConstGridPtr());
                            }
                            points->setName(vdbName);
                            pointGrids.push_back(points);
                        }
                    }
                    else { // houdini points
                        openvdb::tools::UniformPointScatter<
                            PointAccessor, RandGen, hvdb::Interrupter> scatter(
                                pointAccessor, density, mtRand, spread, &boss);
                        process(gridType, *grid, scatter, name);
                    }
                }
            } else if (pmode == 2) { // points per voxel
                if (vdbPoints) { // vdb points
                    VDBDenseUniformScatter scatter(ptsPerVox, seed, spread, &boss);
                    if (process(gridType, *grid, scatter, name))  {
                        openvdb::points::PointDataGrid::Ptr points = scatter.points();
                        if (performCull) {
                            cullVDBPoints(points->tree(), primIter->getConstGridPtr());
                        }
                        points->setName(vdbName);
                        pointGrids.push_back(scatter.points());
                    }
                }
                else { // houdini points
                    openvdb::tools::DenseUniformPointScatter<
                        PointAccessor, RandGen, hvdb::Interrupter> scatter(
                            pointAccessor, ptsPerVox, mtRand, spread, &boss);
                    process(gridType, *grid, scatter, name);
                }
            }

        } // for each grid

        if (!emptyGrids.empty()) {
            std::string s = "The following grids were empty: "
                + boost::algorithm::join(emptyGrids, ", ");
            addWarning(SOP_MESSAGE, s.c_str());
        }

        // add points to a group if requested
        if (1 == evalInt("dogroup", 0, time)) {
            UT_String scatterStr;
            evalString(scatterStr, "sgroup", 0, time);
            GA_PointGroup* ptgroup = gdp->newPointGroup(scatterStr);

            // add the scattered points to this group

            const GA_Offset lastOffset = gdp->getNumPointOffsets();
            ptgroup->addRange(GA_Range(gdp->getPointMap(), firstOffset, lastOffset));

            const std::string groupName(scatterStr.toStdString());
            for (auto& pointGrid : pointGrids) {
                openvdb::points::appendGroup(pointGrid->tree(), groupName);
                openvdb::points::setGroup(pointGrid->tree(), groupName);
            }
        }

        for (auto& pointGrid : pointGrids) {
            hvdb::createVdbPrimitive(*gdp, pointGrid, pointGrid->getName().c_str());
        }
    }
    catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }
    return error();
}

// Copyright (c) 2012-2017 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
