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

#include <openvdb/tools/PointScatter.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string/join.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


class SOP_OpenVDB_Scatter: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Scatter(OP_Network* net, const char* name, OP_Operator* op);
    virtual ~SOP_OpenVDB_Scatter() {}

    static OP_Node* factory(OP_Network*, const char*, OP_Operator*);

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
    virtual void resolveObsoleteParms(PRM_ParmList*);
};


////////////////////////////////////////


// Build UI and register this operator.
void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Group pattern
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("Specify a subset of the input VDB grids to be processed.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    // Export VDBs
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "keep", "Keep input VDB grids")
        .setHelpText("The output will contain the input VDB grids.")
        .setDefault(PRMzeroDefaults));

    // Group scattered points
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "dogroup", "")
        .setTypeExtended(PRM_TYPE_TOGGLE_JOIN)
        .setHelpText("Add scattered points to the group with the given name.")
        .setDefault(PRMzeroDefaults));

    // Scatter group name
    parms.add(hutil::ParmFactory(PRM_STRING, "sgroup", "Scatter Group")
        .setHelpText("Name of the group to which to add scattered points")
        .setDefault(0, "scatter"));

    // Random seed
    parms.add(hutil::ParmFactory(PRM_INT_J, "seed", "Random Seed")
        .setDefault(PRMzeroDefaults));

    // Spread           
    parms.add(hutil::ParmFactory(PRM_FLT_J, "spread", "Spread")
        .setDefault(PRMoneDefaults)
        .setHelpText("Defines how far each point may be displaced from the center "
              "of its voxel or tile. A value of zero means that the point is "
              "placed exactly at the center. A value of one means that the "
              "point can be placed randomly anywhere inside the voxel or tile.")
        .setRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0));

    parms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep1", ""));

    // Mode for point scattering
    const char* items[] = {
        "count",            "count =",
        "density",          "density =",
        "pointspervoxel",   "points per voxel =",
        NULL
    };
    parms.add(hutil::ParmFactory(PRM_ORD, "pointmode", "Mode")
        .setHelpText(
            "Specify how many points to scatter.\n"
            "Point Total: specify a fixed, total point count\n"
            "Point Density: specify the number of points per unit volume\n"
            "Points Per Voxel: specify the number of points per voxel")
        .setChoiceListItems(PRM_CHOICELIST_SINGLE, items));

    // Point count
    parms.add(hutil::ParmFactory(PRM_INT_J, "count", "Count")
        .setDefault(5000)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10000));

    // Point density
    parms.add(hutil::ParmFactory(PRM_FLT_J, "density", "Density")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10));

    // Toggle to use voxel value as local point density multiplier
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "multiply", "Scale density by voxel values")
        .setHelpText("Use voxel values as local multipliers for the point density.")
        .setDefault(PRMzeroDefaults) /* off by default */);

    // Points per voxel
    parms.add(hutil::ParmFactory(PRM_FLT_J , "ppv", "Count")
         .setDefault(8)
         .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 10));

    // Toggle to scatter inside level sets
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "interior", "Scatter points inside level set grids")
        .setHelpText("Toggle to scatter points in the interior region of a level set. "
            "(Instead of the narrow band region used by default.)")
        .setDefault(PRMzeroDefaults) /* off by default */);


    parms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep2", ""));

    // Verbose output toggle
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "verbose", "Verbose")
        .setDefault(PRMzeroDefaults));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_ORD| PRM_TYPE_JOIN_NEXT, "pointMode", "Point"));

    // Register the SOP.
    hvdb::OpenVDBOpFactory("OpenVDB Scatter", SOP_OpenVDB_Scatter::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("VDB on which points will be scattered");
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
    const int pmode = evalInt("pointmode", /*idx=*/0, /*time=*/0);

    changed |= setVisibleState("count",    (0 == pmode));
    changed |= setVisibleState("density",  (1 == pmode));
    changed |= setVisibleState("multiply", (1 == pmode));
    changed |= setVisibleState("ppv",      (2 == pmode));

    const int dogroup = evalInt("dogroup", 0, 0);
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


// Method to extract the interior mask before scattering points.
template<typename OpType>
bool
processLSInterior(UT_VDBType gridType, const openvdb::GridBase& gridRef, OpType& op)
{
    if (gridType == UT_VDB_FLOAT) {
        const openvdb::FloatGrid* grid = static_cast<const openvdb::FloatGrid*>(&gridRef);
        if (grid == NULL) return false;

        typename openvdb::Grid<typename openvdb::FloatTree::template ValueConverter<bool>::Type>::Ptr maskGrid;
        maskGrid = openvdb::tools::sdfInteriorMask(*grid);
        op(*maskGrid);

        return true;

    } else if (gridType == UT_VDB_DOUBLE) {
        const openvdb::DoubleGrid* grid = static_cast<const openvdb::DoubleGrid*>(&gridRef);
        if (grid == NULL) return false;

        typename openvdb::Grid<typename openvdb::DoubleTree::template ValueConverter<bool>::Type>::Ptr maskGrid;
        maskGrid = openvdb::tools::sdfInteriorMask(*grid);
        op(*maskGrid);

        return true;
    }
    return false;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Scatter::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        const fpreal time = context.getTime();

        const GU_Detail* vdbgeo = inputGeo(0);

        gdp->clearAndDestroy();

        const int seed = evalInt("seed", /*idx=*/0, time);
        const double spread = evalFloat("spread", 0, time);
        const bool verbose   = evalInt("verbose", /*idx=*/0, time) != 0;
        const openvdb::Index64 pointCount = evalInt("count", 0, time);
        const float density  = static_cast<float>(evalFloat("density", 0, time));
        const float ptsPerVox = static_cast<float>(evalFloat("ppv", 0, time));
        const bool interior  = evalInt("interior", /*idx=*/0, time) != 0;


        // Get the group of grids to process.
        UT_String groupStr;
        evalString(groupStr, "group", 0, time);
        const GA_PrimitiveGroup* group
            = this->matchGroup(const_cast<GU_Detail&>(*vdbgeo), groupStr.toStdString());

        hvdb::Interrupter boss("OpenVDB Scatter");
        PointAccessor points(gdp);

        // Choose a fast random generator with a long period. Drawback here for
        // mt11213b is that it requires 352*sizeof(uint32) bytes.
        typedef boost::mt11213b RandGen;
        RandGen mtRand(seed);

        const int pmode = evalInt("pointmode", 0, time);

        std::vector<std::string> emptyGrids;

        // Process each VDB primitive (with a non-null grid pointer)
        // that belongs to the selected group.
        for (hvdb::VdbPrimCIterator primIter(vdbgeo, group); primIter; ++primIter) {

            // Retrieve a read-only grid pointer.
            hvdb::GridCRef grid = primIter->getGrid();
            UT_VDBType gridType = primIter->getStorageType();
            const openvdb::GridClass gridClass = grid.getGridClass();
            const bool isSignedDistance = (gridClass == openvdb::GRID_LEVEL_SET);
            const std::string gridName = primIter.getPrimitiveName().toStdString();

            if (grid.empty()) {
                emptyGrids.push_back(gridName);
                continue;
            }

            if (pmode == 0) { // fixed point count

                openvdb::tools::UniformPointScatter<PointAccessor, RandGen, hvdb::Interrupter>
                    scatter(points, pointCount, mtRand, spread, &boss);

                if (interior && isSignedDistance) {
                    processLSInterior(gridType, grid, scatter);
                } else {
                    UTvdbProcessTypedGridScalar(gridType, grid, scatter);
                }

                if (verbose) scatter.print(gridName);

            } else if (pmode == 1) { // points per unit volume

                if (evalInt("multiply", 0, time) != 0) { // local density
                    openvdb::tools::NonUniformPointScatter<PointAccessor,RandGen,hvdb::Interrupter>
                        scatter(points, density, mtRand, spread, &boss);


                    if (interior && isSignedDistance) {
                        processLSInterior(gridType, grid, scatter);
                    } else {

                        if (!UTvdbProcessTypedGridScalar(gridType, grid, scatter)) {
                            throw std::runtime_error
                                ("Only scalar grids support voxel scaling of density");
                        }
                    }

                    if (verbose) scatter.print(gridName);

                } else { // global density
                    openvdb::tools::UniformPointScatter<PointAccessor, RandGen, hvdb::Interrupter>
                        scatter(points, density, mtRand, spread, &boss);

                    if (interior && isSignedDistance) {
                        processLSInterior(gridType, grid, scatter);
                    } else {
                        UTvdbProcessTypedGridTopology(gridType, grid, scatter);
                    }

                    if (verbose) scatter.print(gridName);
                }

            } else if (pmode == 2) { // points per voxel

                openvdb::tools::DenseUniformPointScatter<PointAccessor, RandGen, hvdb::Interrupter>
                    scatter(points, ptsPerVox, mtRand, spread, &boss);

                if (interior && isSignedDistance) {
                    processLSInterior(gridType, grid, scatter);
                } else {
                    UTvdbProcessTypedGridTopology(gridType, grid, scatter);
                }

                if (verbose) scatter.print(gridName);
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

            // add ALL the points to this group
            ptgroup->addRange(gdp->getPointRange());
        }

        // add the VDBs to the output if requested
        if (1 == evalInt("keep", 0, time)) {
            gdp->mergePrimitives(*vdbgeo, vdbgeo->getPrimitiveRange());
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
