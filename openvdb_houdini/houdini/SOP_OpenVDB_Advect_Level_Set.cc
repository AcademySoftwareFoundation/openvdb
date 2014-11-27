///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2014 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Advect_Level_Set.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Level set advection SOP

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/LevelSetAdvect.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_PageIterator.h>
#include <GU/GU_PrimPoly.h>

#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/join.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////

// Utilities

namespace {

struct AdvectionParms {
    AdvectionParms()
        : mLSGroup(NULL)
        , mAdvectSpatial(openvdb::math::UNKNOWN_BIAS)
        , mRenormSpatial(openvdb::math::UNKNOWN_BIAS)
        , mAdvectTemporal(openvdb::math::UNKNOWN_TIS)
        , mRenormTemporal(openvdb::math::UNKNOWN_TIS)
        , mNormCount(1)
        , mTimeStep(0.0)
        , mStaggered(false)
    {
    }

    const GA_PrimitiveGroup *mLSGroup;
    hvdb::Grid::ConstPtr mVelocityGrid;
    openvdb::math::BiasedGradientScheme mAdvectSpatial, mRenormSpatial;
    openvdb::math::TemporalIntegrationScheme mAdvectTemporal, mRenormTemporal;
    int mNormCount;
    float mTimeStep;
    bool mStaggered;
};


template <class VelocityGridT>
class AdvectOp
{
public:
    AdvectOp(AdvectionParms& parms, const VelocityGridT& velGrid, hvdb::Interrupter& boss)
        : mParms(parms)
        , mVelGrid(velGrid)
        , mBoss(boss)
    {
    }

    template<typename GridT, typename SamplerT>
    void process(GridT& grid)
    {
        typedef openvdb::tools::DiscreteField<VelocityGridT, SamplerT> FieldT;
        FieldT field(mVelGrid);

        openvdb::tools::LevelSetAdvection<GridT, FieldT, hvdb::Interrupter>
            advection(grid, field, &mBoss);

        advection.setSpatialScheme(mParms.mAdvectSpatial);
        advection.setTemporalScheme(mParms.mAdvectTemporal);
        advection.setTrackerSpatialScheme(mParms.mRenormSpatial);
        advection.setTrackerTemporalScheme(mParms.mRenormTemporal);
        advection.setNormCount(mParms.mNormCount);

        if (mBoss.wasInterrupted()) return;

        advection.advect(0, mParms.mTimeStep);
    }


    template<typename GridT>
    void operator()(GridT& grid)
    {
        if (mBoss.wasInterrupted()) return;

        if (mParms.mStaggered) process<GridT, openvdb::tools::StaggeredBoxSampler>(grid);
        else process<GridT, openvdb::tools::BoxSampler>(grid);
    }

private:
    AdvectionParms& mParms;
    const VelocityGridT& mVelGrid;
    hvdb::Interrupter& mBoss;
};

} // namespace


////////////////////////////////////////

// SOP Declaration

class SOP_OpenVDB_Advect_Level_Set: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Advect_Level_Set(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Advect_Level_Set() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

    OP_ERROR evalAdvectionParms(OP_Context&, AdvectionParms&);

    template <class VelocityGridT>
    bool processGrids(AdvectionParms&, hvdb::Interrupter&);
};

////////////////////////////////////////

// Build UI and register this operator

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    using namespace openvdb::math;

    hutil::ParmList parms;

    // Level set grid
    parms.add(hutil::ParmFactory(PRM_STRING, "lsGroup", "Group")
        .setHelpText("Level set grid(s) to advect.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    // Velocity grid
    parms.add(hutil::ParmFactory(PRM_STRING, "velGroup", "Velocity")
        .setHelpText("Velocity grid")
        .setChoiceList(&hutil::PrimGroupMenuInput2));

    parms.add(hutil::ParmFactory(PRM_HEADING, "advectionHeading", "Advection"));

    // Advect: timestep
    parms.add(hutil::ParmFactory(PRM_FLT, "timestep", "Time Step")
        .setDefault(1, "1.0/$FPS"));

    // Advect: spatial menu
    {
        std::vector<std::string> items;
        items.push_back(biasedGradientSchemeToString(FIRST_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(FIRST_BIAS));

        items.push_back(biasedGradientSchemeToString(HJWENO5_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(HJWENO5_BIAS));


        parms.add(hutil::ParmFactory(PRM_STRING, "advectSpatial", "Spatial Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(biasedGradientSchemeToString(HJWENO5_BIAS).c_str()))
            .setHelpText("Set the spatial finite difference scheme."));
    }

    // Advect: temporal menu
    {
        std::vector<std::string> items;
        for (int i = 0; i < NUM_TEMPORAL_SCHEMES; ++i) {
            TemporalIntegrationScheme it = TemporalIntegrationScheme(i);
            items.push_back(temporalIntegrationSchemeToString(it)); // token
            items.push_back(temporalIntegrationSchemeToMenuName(it)); // label
        }

        parms.add(hutil::ParmFactory(PRM_STRING, "advectTemporal", "Temporal Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(temporalIntegrationSchemeToString(TVD_RK2).c_str()))
            .setHelpText("Set the temporal integration scheme."));
    }

    parms.add(hutil::ParmFactory(PRM_HEADING, "renormHeading", "Renormalization"));

    parms.add(hutil::ParmFactory(PRM_INT_J, "normSteps", "Steps")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText("The number of normalizations performed after each CFL iteration."));

    // Renorm: spatial menu
    {
        std::vector<std::string> items;
        items.push_back(biasedGradientSchemeToString(FIRST_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(FIRST_BIAS));

        items.push_back(biasedGradientSchemeToString(HJWENO5_BIAS));
        items.push_back(biasedGradientSchemeToMenuName(HJWENO5_BIAS));

        parms.add(hutil::ParmFactory(PRM_STRING, "renormSpatial", "Spatial Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(biasedGradientSchemeToString(HJWENO5_BIAS).c_str()))
            .setHelpText("Set the spatial finite difference scheme."));
    }

    // Renorm: temporal menu
    {
        std::vector<std::string> items;
        for (int i = 0; i < NUM_TEMPORAL_SCHEMES; ++i) {
            TemporalIntegrationScheme it = TemporalIntegrationScheme(i);
            items.push_back(temporalIntegrationSchemeToString(it)); // token
            items.push_back(temporalIntegrationSchemeToMenuName(it)); // label
        }

        parms.add(hutil::ParmFactory(PRM_STRING, "renormTemporal", "Temporal Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(items[0].c_str()))
            .setHelpText("Set the temporal integration scheme."));
    }

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT, "beginTime", "Begin time"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT, "endTime", "Time step"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Advect Level Set",
        SOP_OpenVDB_Advect_Level_Set::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("SDF VDBs to Advect")
        .addInput("Velocity VDB");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Advect_Level_Set::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Advect_Level_Set(net, name, op);
}


SOP_OpenVDB_Advect_Level_Set::SOP_OpenVDB_Advect_Level_Set(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////

// Enable/disable or show/hide parameters in the UI.

bool
SOP_OpenVDB_Advect_Level_Set::updateParmsFlags()
{
    return false;
}



////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Advect_Level_Set::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();
        duplicateSourceStealable(0, context);

        // Evaluate UI parameters
        AdvectionParms parms;
        if (evalAdvectionParms(context, parms) >= UT_ERROR_ABORT) return error();

        hvdb::Interrupter boss("Advecting level set");

        processGrids<openvdb::Vec3SGrid>(parms, boss);

        if (boss.wasInterrupted()) addWarning(SOP_MESSAGE, "Process was interrupted");
        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Advect_Level_Set::evalAdvectionParms(OP_Context& context, AdvectionParms& parms)
{
    fpreal now = context.getTime();
    UT_String str;

    evalString(str, "lsGroup", 0, now);
    parms.mLSGroup = matchGroup(*gdp, str.toStdString());

    parms.mTimeStep = static_cast<float>(evalFloat("timestep", 0, now));

    evalString(str, "advectSpatial", 0, now);

    parms.mAdvectSpatial =
        openvdb::math::stringToBiasedGradientScheme(str.toStdString());

    if (parms.mAdvectSpatial == openvdb::math::UNKNOWN_BIAS) {
        addError(SOP_MESSAGE, "Advect: Unknown biased gradient");
        return UT_ERROR_ABORT;
    }

    evalString(str, "renormSpatial", 0, now);

    parms.mRenormSpatial =
        openvdb::math::stringToBiasedGradientScheme(str.toStdString());

    if (parms.mRenormSpatial == openvdb::math::UNKNOWN_BIAS) {
        addError(SOP_MESSAGE, "Renorm: Unknown biased gradient");
        return UT_ERROR_ABORT;
    }

    evalString(str, "advectTemporal", 0, now);
    parms.mAdvectTemporal =
        openvdb::math::stringToTemporalIntegrationScheme(str.toStdString());

    if (parms.mAdvectTemporal == openvdb::math::UNKNOWN_TIS) {
        addError(SOP_MESSAGE, "Advect: Unknown temporal integration");
        return UT_ERROR_ABORT;
    }

    evalString(str, "renormTemporal", 0, now);
    parms.mRenormTemporal =
        openvdb::math::stringToTemporalIntegrationScheme(str.toStdString());

    if (parms.mRenormTemporal == openvdb::math::UNKNOWN_TIS) {
        addError(SOP_MESSAGE, "Renorm: Unknown temporal integration");
        return UT_ERROR_ABORT;
    }

    parms.mNormCount = evalInt("normSteps", 0, now);

    const GU_Detail* velGeo = inputGeo(1);

    if (!velGeo) {
        addError(SOP_MESSAGE, "Missing velocity grid input");
        return UT_ERROR_ABORT;
    }

    evalString(str, "velGroup", 0, now);
    const GA_PrimitiveGroup *velGroup =
        matchGroup(const_cast<GU_Detail&>(*velGeo), str.toStdString());

    hvdb::VdbPrimCIterator it(velGeo, velGroup);
    if (it) {
        if (it->getStorageType() != UT_VDB_VEC3F) {
            addError(SOP_MESSAGE, "Unrecognized velocity grid type.");
            return UT_ERROR_ABORT;
        }

        parms.mVelocityGrid = it->getConstGridPtr();
    }

    if (!parms.mVelocityGrid) {
        addError(SOP_MESSAGE, "Missing velocity grid");
        return UT_ERROR_ABORT;
    }

    parms.mStaggered = parms.mVelocityGrid->getGridClass() == openvdb::GRID_STAGGERED;

    return error();
}


////////////////////////////////////////


template <class VelocityGridT>
bool
SOP_OpenVDB_Advect_Level_Set::processGrids(AdvectionParms& parms, hvdb::Interrupter& boss)
{
    typename VelocityGridT::ConstPtr velGrid =
        hvdb::Grid::constGrid<VelocityGridT>(parms.mVelocityGrid);
    if (!velGrid) return false;

    AdvectOp<VelocityGridT> op(parms, *velGrid, boss);

    std::vector<std::string> skippedGrids, nonLevelSetGrids, narrowBands;

    for (hvdb::VdbPrimIterator it(gdp, parms.mLSGroup); it; ++it) {

        if (boss.wasInterrupted()) break;

        GU_PrimVDB* vdbPrim = *it;

        const openvdb::GridClass gridClass = vdbPrim->getGrid().getGridClass();
        if (gridClass != openvdb::GRID_LEVEL_SET) {
            nonLevelSetGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
            continue;
        }

        if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {
            vdbPrim->makeGridUnique();
            openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
            if ( grid.background() < float(openvdb::LEVEL_SET_HALF_WIDTH * grid.voxelSize()[0]) ) {
                narrowBands.push_back(it.getPrimitiveNameOrIndex().toStdString());
            }
            op(grid);
#if 0
        } else if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {
            vdbPrim->makeGridUnique();
            openvdb::DoubleGrid& grid =
                UTvdbGridCast<openvdb::DoubleGrid>(vdbPrim->getGrid());
            if (grid.background() < float(openvdb::LEVEL_SET_HALF_WIDTH * grid.voxelSize()[0])) {
                narrowBands.push_back(it.getPrimitiveNameOrIndex().toStdString());
            }
            op(grid);
#endif
        } else {
            skippedGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
        }
    }

    if (!skippedGrids.empty()) {
        std::string s = "The following non-floating-point grids were skipped: "
            + boost::algorithm::join(skippedGrids, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    if (!nonLevelSetGrids.empty()) {
        std::string s = "The following non-level-set grids were skipped: "
            + boost::algorithm::join(nonLevelSetGrids, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    if (!narrowBands.empty()) {
        std::string s = "The following grids have a narrow band width that is"
            " less than 3 voxel units: " + boost::algorithm::join(narrowBands, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    return true;
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
