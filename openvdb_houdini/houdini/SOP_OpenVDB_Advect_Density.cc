///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
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
/// @file SOP_OpenVDB_Advect_Density.cc
///
/// @author Ken Museth
///
/// @brief Density and veclocity (i.e. non-level-set) advection SOP
///
/// @todo Add optional mask (allready supported by tools::DensityAdvect)

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/DensityAdvect.h>

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
        : mGroup(NULL)
        , mVelocityGrid()
        , mCountRK(1)
        , mOrderRK(1)
        , mSampleOrder(1)
        , mTimeStep(0.0)
        , mStaggered(false)
    {
    }

    const GA_PrimitiveGroup *mGroup;
    hvdb::Grid::ConstPtr mVelocityGrid;
    size_t mCountRK, mOrderRK, mSampleOrder;
    float mTimeStep;
    bool mStaggered;
};

} // namespace


////////////////////////////////////////

// SOP Declaration

class SOP_OpenVDB_Advect_Density: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDB_Advect_Density(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDB_Advect_Density() {}

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

    OP_ERROR evalAdvectionParms(OP_Context&, AdvectionParms&);

    template <bool StaggeredVelocity, size_t SampleOrder>
    bool processGrids(const AdvectionParms&, hvdb::Interrupter&);
};

////////////////////////////////////////

// Build UI and register this operator

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    using namespace openvdb::math;

    hutil::ParmList parms;

    // Density grid
    parms.add(hutil::ParmFactory(PRM_STRING, "densityGroup", "Group")
        .setHelpText("VDB grid(s) to advect.")
        .setChoiceList(&hutil::PrimGroupMenuInput1));

    // Velocity grid
    parms.add(hutil::ParmFactory(PRM_STRING, "velGroup", "Velocity")
        .setHelpText("Velocity grid")
        .setChoiceList(&hutil::PrimGroupMenuInput2));

    parms.add(hutil::ParmFactory(PRM_HEADING, "advectionHeading", "Advection"));

    // Timestep
    parms.add(hutil::ParmFactory(PRM_FLT, "timestep", "Time Step")
        .setDefault(1, "1.0/$FPS"));
              
    // Density interpolation order
    parms.add(hutil::ParmFactory(PRM_INT_J, "OrderOfSampling", "Sampling Order")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_RESTRICTED, 2)
        .setHelpText("The order of the interpolation scheme to resample density"));

    // Integration order
    parms.add(hutil::ParmFactory(PRM_INT_J, "OrderOfRK", "Runge-Kutta Order")
        .setDefault(PRMthreeDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_RESTRICTED, 4)
        .setHelpText("The order of the Runge-Kutta integration scheme"
                     "employed during the back-tracing in the semi-lagrgangian"
                     "integration."));

    // Integration steps
    parms.add(hutil::ParmFactory(PRM_INT_J, "StepsOfRK", "Runge-Kutta Steps")
        .setDefault(PRMoneDefaults)
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText("The number of backwards Runge-Kutta sub-steps to be performed"
                     "per time integration step."));
              
    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Advect Density",
        SOP_OpenVDB_Advect_Density::factory, parms, *table)
        .addInput("VDBs to Advect")
        .addInput("Velocity VDB");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDB_Advect_Density::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDB_Advect_Density(net, name, op);
}


SOP_OpenVDB_Advect_Density::SOP_OpenVDB_Advect_Density(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////

// Enable/disable or show/hide parameters in the UI.

bool
SOP_OpenVDB_Advect_Density::updateParmsFlags()
{
    return false;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDB_Advect_Density::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);
        gdp->clearAndDestroy();
        duplicateSource(0, context);

        // Evaluate UI parameters
        AdvectionParms parms;
        if (evalAdvectionParms(context, parms) >= UT_ERROR_ABORT) return error();

        hvdb::Interrupter boss("Advecting VDBs");

        if (parms.mStaggered) {
            if (parms.mSampleOrder == 0) {
                this->processGrids<true, 0>(parms, boss);
            } else if (parms.mSampleOrder == 1) {
                this->processGrids<true, 1>(parms, boss);
            } else if (parms.mSampleOrder == 2) {
                this->processGrids<true, 2>(parms, boss);
            } else {
                addError(SOP_MESSAGE, "Unsupported order of staggered velocity sampling");
                return UT_ERROR_ABORT;
            }
        } else {
            if (parms.mSampleOrder == 0) {
                this->processGrids<false, 0>(parms, boss);
            } else if (parms.mSampleOrder == 1) {
                this->processGrids<false, 1>(parms, boss);
            } else if (parms.mSampleOrder == 2) {
                this->processGrids<false, 2>(parms, boss);
            } else {
                addError(SOP_MESSAGE, "Unsupported order of density sampling");
                return UT_ERROR_ABORT;
            }
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
SOP_OpenVDB_Advect_Density::evalAdvectionParms(OP_Context& context, AdvectionParms& parms)
{
    fpreal now = context.getTime();
    UT_String str;

    evalString(str, "densityGroup", 0, now);
    parms.mGroup = matchGroup(*gdp, str.toStdString());

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

    parms.mTimeStep = static_cast<float>(evalFloat("timestep", 0, now));

    parms.mCountRK = evalInt("StepsOfRK", 0, now);

    parms.mOrderRK = evalInt("OrderOfRK", 0, now);

    parms.mSampleOrder = evalInt("OrderOfSampling", 0, now);

    parms.mStaggered = parms.mVelocityGrid->getGridClass() == openvdb::GRID_STAGGERED;

    return error();
}


////////////////////////////////////////


template <bool StaggeredVelocity, size_t SampleOrder>
bool
SOP_OpenVDB_Advect_Density::processGrids(const AdvectionParms& parms, hvdb::Interrupter& boss)
{
    typedef openvdb::Vec3SGrid VelGridT;
    typedef openvdb::tools::DensityAdvection<VelGridT, StaggeredVelocity, hvdb::Interrupter> AdvT;
    typename VelGridT::ConstPtr velGrid = hvdb::Grid::constGrid<VelGridT>(parms.mVelocityGrid);
    if (!velGrid) return false;

    AdvT adv(*velGrid, &boss);
    adv.setIntegrationCount(parms.mCountRK);
    adv.setIntegrationOrder(parms.mOrderRK);

    std::vector<std::string> skippedGrids, levelSetGrids;

    for (hvdb::VdbPrimIterator it(gdp, parms.mGroup); it; ++it) {

        if (boss.wasInterrupted()) break;

        GU_PrimVDB* vdbPrim = *it;

        const openvdb::GridClass gridClass = vdbPrim->getGrid().getGridClass();
        if (gridClass == openvdb::GRID_LEVEL_SET) {
            levelSetGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
        }

        if (vdbPrim->getStorageType() == UT_VDB_FLOAT) {
            const openvdb::FloatGrid& inGrid = UTvdbGridCast<openvdb::FloatGrid>(vdbPrim->getGrid());
            const int d = adv.getMaxDistance(inGrid, parms.mTimeStep);
            if (d > 20) {
                std::ostringstream tmp;
                tmp << "Dilation by " << d << " voxels could be slow!"
                    << " Consider lowing time-step or integration count!";
                addWarning(SOP_MESSAGE, tmp.str().c_str());
            }
            typename openvdb::FloatGrid::Ptr outGrid = adv.template advect<openvdb::FloatGrid,
                openvdb::tools::Sampler<SampleOrder, false> >(inGrid, parms.mTimeStep);
            hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdbPrim);
        } else if (vdbPrim->getStorageType() == UT_VDB_VEC3F) {
            vdbPrim->makeGridUnique();
            const openvdb::Vec3SGrid& inGrid = UTvdbGridCast<openvdb::Vec3SGrid>(vdbPrim->getGrid());
            const int d = adv.getMaxDistance(inGrid, parms.mTimeStep);
            if (d > 20) {
                std::ostringstream tmp;
                tmp << "Dilation by " << d << " voxels could be slow!"
                    << " Consider lowing time-step or integration count!";
                addWarning(SOP_MESSAGE, tmp.str().c_str());
            }
            typename openvdb::Vec3SGrid::Ptr outGrid;
            if (inGrid.getGridClass() == openvdb::GRID_STAGGERED) {
                outGrid = adv.template advect<openvdb::Vec3SGrid,
                    openvdb::tools::Sampler<SampleOrder, true> >(inGrid, parms.mTimeStep);
            } else {
                outGrid = adv.template advect<openvdb::Vec3SGrid,
                    openvdb::tools::Sampler<SampleOrder, false> >(inGrid, parms.mTimeStep);
            }
            hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdbPrim);
        } else {
            skippedGrids.push_back(it.getPrimitiveNameOrIndex().toStdString());
        }
    }

    if (!skippedGrids.empty()) {
        std::string s = "The following non-floating-point grids were skipped: "
            + boost::algorithm::join(skippedGrids, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    if (!levelSetGrids.empty()) {
        std::string s = "Consider using the LevelSetAdvect SOP for the following grid: "
            + boost::algorithm::join(levelSetGrids, ", ");
        addWarning(SOP_MESSAGE, s.c_str());
    }

    return true;
}

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
