///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2016 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Advect.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief Level set advection SOP

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>

#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeAdvect.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_PageIterator.h>
#include <GU/GU_PrimPoly.h>
#include <CH/CH_Manager.h>
#include <PRM/PRM_Parm.h>

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
        : mGroup(NULL)
        , mAdvectSpatial(openvdb::math::UNKNOWN_BIAS)
        , mRenormSpatial(openvdb::math::UNKNOWN_BIAS)
        , mAdvectTemporal(openvdb::math::UNKNOWN_TIS)
        , mRenormTemporal(openvdb::math::UNKNOWN_TIS)
        , mIntegrator(openvdb::tools::Scheme::SEMI)
        , mLimiter(openvdb::tools::Scheme::NO_LIMITER)
        , mNormCount(1)
        , mSubSteps(1)
        , mTimeStep(0.0)
        , mStaggered(false)
        , mRespectClass(true)
    {
    }

    const GA_PrimitiveGroup *                   mGroup;
    hvdb::Grid::ConstPtr                        mVelocityGrid;
    openvdb::math::BiasedGradientScheme         mAdvectSpatial, mRenormSpatial;
    openvdb::math::TemporalIntegrationScheme    mAdvectTemporal, mRenormTemporal;
    openvdb::tools::Scheme::SemiLagrangian      mIntegrator;
    openvdb::tools::Scheme::Limiter             mLimiter;
    int                                         mNormCount, mSubSteps;
    float                                       mTimeStep;
    bool                                        mStaggered, mRespectClass;
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

class SOP_OpenVDB_Advect: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Advect(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Advect() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();
    virtual void resolveObsoleteParms(PRM_ParmList*);

    OP_ERROR evalAdvectionParms(OP_Context&, AdvectionParms&);

    template <typename VelocityGridT, bool StaggeredVelocity>
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
    parms.add(hutil::ParmFactory(PRM_STRING, "group", "Group")
        .setHelpText("VDB grid(s) to advect.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    // Velocity grid
    parms.add(hutil::ParmFactory(PRM_STRING, "velGroup", "Velocity")
        .setHelpText("Velocity grid")
        .setChoiceList(&hutil::PrimGroupMenuInput2));

    parms.add(hutil::ParmFactory(PRM_TOGGLE, "respectclass", "Respect Grid Class")
        .setDefault(PRMoneDefaults)
        .setHelpText("If disabled, advect level sets using general "
            "advection scheme."));

    // Advect: timestep
    parms.add(hutil::ParmFactory(PRM_FLT, "timestep", "Time Step")
        .setDefault(1, "1.0/$FPS"));

    parms.add(hutil::ParmFactory(PRM_HEADING, "general", "General Advection"));


    // SubSteps
    parms.add(hutil::ParmFactory(PRM_INT_J, "substeps", "Sub-steps")
         .setDefault(1)
         .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
         .setHelpText("The number of substeps per integration step. The only "
                      "reason to increase it above its default value of one "
                      "is to reduce the memory-footprint from dilations "
                      "- likely at the cost of more smoothing!"));


    // Advection Scheme
    {
        std::vector<std::string> items;
        items.push_back("semi");
        items.push_back("Semi-Lagrangian");
        items.push_back("mid");
        items.push_back("Mid-Point");
        items.push_back("rk3");
        items.push_back("3rd order Runge-Kutta");
        items.push_back("rk4");
        items.push_back("4th order Runge-Kutta");
        items.push_back("mac");
        items.push_back("MacCormack");
        items.push_back("bfecc");
        items.push_back("BFECC");

        parms.add(hutil::ParmFactory(PRM_STRING, "advection", "Advection Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(0, ::strdup("semi"))
            .setHelpText("Set the numerical advection scheme."));
    }

    // Limiter Scheme
    {
        std::vector<std::string> items;
        items.push_back("none");
        items.push_back("No limiter");
        items.push_back("clamp");
        items.push_back("Clamp to extrema");
        items.push_back("revert");
        items.push_back("Revert to 1st order");

        parms.add(hutil::ParmFactory(PRM_STRING, "limiter", "Limiter Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(2, ::strdup("revert"))
            .setHelpText("Set the limiter scheme use to stabalize the 2nd "
                         "order schemes MacCormack and BFECC."));
    }





    parms.add(hutil::ParmFactory(PRM_HEADING, "advectionHeading", "Level Set Advection"));

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

    parms.add(hutil::ParmFactory(PRM_INT_J, "normSteps", "Renormalization Steps")
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

        parms.add(hutil::ParmFactory(PRM_STRING, "renormSpatial", "Spatial Renormalization")
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

        parms.add(hutil::ParmFactory(PRM_STRING, "renormTemporal", "Temporal Renormalization")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(1, ::strdup(items[0].c_str()))
            .setHelpText("Set the temporal integration scheme."));
    }

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT, "beginTime", "Begin time"));
    obsoleteParms.add(hutil::ParmFactory(PRM_FLT, "endTime", "Time step"));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "lsGroup", "Group"));
    obsoleteParms.add(hutil::ParmFactory(PRM_STRING, "densityGroup", "Group"));
    obsoleteParms.add(hutil::ParmFactory(PRM_HEADING, "renormHeading", ""));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Advect",
        SOP_OpenVDB_Advect::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addAlias("OpenVDB Advect Level Set")
        .addAlias("OpenVDB Advect Density")
        .addInput("VDBs to Advect")
        .addInput("Velocity VDB");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Advect::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Advect(net, name, op);
}


SOP_OpenVDB_Advect::SOP_OpenVDB_Advect(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}

////////////////////////////////////////


void
SOP_OpenVDB_Advect::resolveObsoleteParms(PRM_ParmList* obsoleteParms)
{
    if (!obsoleteParms) return;

    const fpreal time = CHgetEvalTime();
    UT_String groupStr;

    PRM_Parm* parm = obsoleteParms->getParmPtr("lsGroup");

    if (parm && !parm->isFactoryDefault()) {
        obsoleteParms->evalString(groupStr, "lsGroup", 0, time);
        if (groupStr.length() > 0) {
            setString(groupStr, CH_STRING_LITERAL, "group", 0, time);
        }
    }

    parm = obsoleteParms->getParmPtr("densityGroup");

    if (parm && !parm->isFactoryDefault()) {
        obsoleteParms->evalString(groupStr, "densityGroup", 0, time);
        if (groupStr.length() > 0) {
            setString(groupStr, CH_STRING_LITERAL, "group", 0, time);
        }
    }

    hvdb::SOP_NodeVDB::resolveObsoleteParms(obsoleteParms);
}


////////////////////////////////////////


bool
SOP_OpenVDB_Advect::updateParmsFlags()
{
    bool changed = false;

    const bool respectClass = bool(evalInt("respectclass", 0, 0));

    changed |= enableParm("advectSpatial", respectClass);
    changed |= enableParm("advectTemporal", respectClass);
    changed |= enableParm("normSteps", respectClass);
    changed |= enableParm("renormSpatial", respectClass);
    changed |= enableParm("renormTemporal", respectClass);

    return changed;
}



////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Advect::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();
        duplicateSourceStealable(0, context);

        // Evaluate UI parameters
        AdvectionParms parms;
        if (evalAdvectionParms(context, parms) >= UT_ERROR_ABORT) return error();

        hvdb::Interrupter boss("Advecting level set");

        if (parms.mStaggered) {
            processGrids<openvdb::Vec3SGrid, true>(parms, boss);
        } else {
            processGrids<openvdb::Vec3SGrid, false>(parms, boss);
        }

        if (boss.wasInterrupted()) addWarning(SOP_MESSAGE, "Process was interrupted");
        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Advect::evalAdvectionParms(OP_Context& context, AdvectionParms& parms)
{
    fpreal now = context.getTime();
    UT_String str;

    evalString(str, "group", 0, now);
    parms.mGroup = matchGroup(*gdp, str.toStdString());

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
    parms.mRespectClass = bool(evalInt("respectclass", 0, now));

    // General advection options

    parms.mSubSteps = static_cast<int>(evalInt("substeps", 0, now));

    evalString(str, "advection", 0, now);
    if ( str == "semi" ) {
        parms.mIntegrator = openvdb::tools::Scheme::SEMI;
    } else if ( str == "mid" ) {
        parms.mIntegrator = openvdb::tools::Scheme::MID;
    } else if ( str == "rk3" ) {
        parms.mIntegrator = openvdb::tools::Scheme::RK3;
    } else if ( str == "rk4" ) {
        parms.mIntegrator = openvdb::tools::Scheme::RK4;
    } else if ( str == "mac" ) {
        parms.mIntegrator = openvdb::tools::Scheme::MAC;
    } else if ( str == "bfecc" ) {
        parms.mIntegrator = openvdb::tools::Scheme::BFECC;
    } else {
        addError(SOP_MESSAGE, "Invalid advection scheme");
        return UT_ERROR_ABORT;
    }

    evalString(str, "limiter", 0, now);
    if ( str == "none" ) {
        parms.mLimiter = openvdb::tools::Scheme::NO_LIMITER;
        if (parms.mIntegrator == openvdb::tools::Scheme::MAC) {
            addWarning(SOP_MESSAGE, "MacCormack is unstable without a limiter");
        }
    } else if ( str == "clamp" ) {
        parms.mLimiter = openvdb::tools::Scheme::CLAMP;
    } else if ( str == "revert" ) {
        parms.mLimiter = openvdb::tools::Scheme::REVERT;
    } else {
        addError(SOP_MESSAGE, "Invalid limiter scheme");
        return UT_ERROR_ABORT;
    }

    return error();
}


////////////////////////////////////////


template <typename VelocityGridT, bool StaggeredVelocity>
bool
SOP_OpenVDB_Advect::processGrids(AdvectionParms& parms, hvdb::Interrupter& boss)
{
    typedef openvdb::tools::VolumeAdvection<VelocityGridT, StaggeredVelocity, hvdb::Interrupter> VolumeAdvection;
    typedef typename VelocityGridT::ConstPtr  VelocityGridCPtr;

    VelocityGridCPtr velGrid = hvdb::Grid::constGrid<VelocityGridT>(parms.mVelocityGrid);
    if (!velGrid) return false;

    AdvectOp<VelocityGridT> advectLevelSet(parms, *velGrid, boss);

    VolumeAdvection advectVolume(*velGrid, &boss);
    advectVolume.setIntegrator(parms.mIntegrator);
    advectVolume.setLimiter(parms.mLimiter);
    advectVolume.setSubSteps(parms.mSubSteps);


    std::vector<std::string> skippedGrids, doubleGrids;

    for (hvdb::VdbPrimIterator it(gdp, parms.mGroup); it; ++it) {

        if (boss.wasInterrupted()) break;

        GU_PrimVDB* vdbPrim = *it;

        if (parms.mRespectClass && vdbPrim->getGrid().getGridClass() == openvdb::GRID_LEVEL_SET) {

            if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {
                vdbPrim->makeGridUnique();
                openvdb::FloatGrid& grid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
                advectLevelSet(grid);

            } /*else if (vdbPrim->getStorageType() == UT_VDB_DOUBLE) {
                vdbPrim->makeGridUnique();
                openvdb::DoubleGrid& grid = UTvdbGridCast<openvdb::DoubleGrid>(vdbPrim->getGrid());
                advectLevelSet(grid);

            }*/ else {
                skippedGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
            }


        } else {

            if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {

                const openvdb::FloatGrid& inGrid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getConstGrid());

                openvdb::FloatGrid::Ptr outGrid = advectVolume.template advect<openvdb::FloatGrid,
                    openvdb::tools::Sampler<1, false> >(inGrid, parms.mTimeStep);

                hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdbPrim);

            } else if (vdbPrim->getStorageType() == UT_VDB_DOUBLE) {

                const openvdb::DoubleGrid& inGrid = UTvdbGridCast<openvdb::DoubleGrid>(vdbPrim->getConstGrid());

                openvdb::DoubleGrid::Ptr outGrid = advectVolume.template advect<openvdb::DoubleGrid,
                    openvdb::tools::Sampler<1, false> >(inGrid, parms.mTimeStep);

                hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdbPrim);

            } else if (vdbPrim->getStorageType() == UT_VDB_VEC3F) {

                const openvdb::Vec3SGrid& inGrid = UTvdbGridCast<openvdb::Vec3SGrid>(vdbPrim->getConstGrid());

                openvdb::Vec3SGrid::Ptr outGrid;

                outGrid = advectVolume.template advect<openvdb::Vec3SGrid,
                        openvdb::tools::Sampler<1, false> >(inGrid, parms.mTimeStep);

                hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdbPrim);

            } else {
                skippedGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
            }
        }


    }

    if (!skippedGrids.empty()) {
        std::string s = "The following non-floating-point grids were skipped: "
            + boost::algorithm::join(skippedGrids, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    return true;
}

// Copyright (c) 2012-2016 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
