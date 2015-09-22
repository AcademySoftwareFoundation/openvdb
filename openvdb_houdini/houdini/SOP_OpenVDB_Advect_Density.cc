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
/// @todo 1) Fix a potential bug in MacCormack
///       2) Add optional mask (allready supported by tools::DensityAdvect)
///       3) Improve performance of tools::DensityAdvect
///
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
        , mIntegrator( openvdb::tools::Scheme::SEMI )
        , mTimeStep(0.0)
        , mStaggered(false)
    {
    }
    const GA_PrimitiveGroup *mGroup;
    hvdb::Grid::ConstPtr mVelocityGrid;
    openvdb::tools::Scheme::SemiLagrangian mIntegrator;
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

    template <bool StaggeredVelocity>
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

    // Advection Scheme
    {
        std::vector<std::string> items;
        items.push_back("semi");
        items.push_back("Semi-Lagrangian");
        items.push_back("mid");
        items.push_back("Mid-Point");
        items.push_back("rk3");
        items.push_back("3rd Order Runge-Kutta");
        items.push_back("rk4");
        items.push_back("4th Order Runge-Kutta");
        items.push_back("mac");
        items.push_back("MacCormack");
        items.push_back("bfecc");
        items.push_back("BFECC");

        parms.add(hutil::ParmFactory(PRM_STRING, "scheme", "Advection Scheme")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(0, ::strdup("semi"))
            .setHelpText("Set the numerical advection scheme."));
    }
      
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
            this->processGrids<true >(parms, boss);
        } else {
            this->processGrids<false>(parms, boss);
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
    
    evalString(str, "scheme", 0, now);
    if ( str == "semi" )
        parms.mIntegrator = openvdb::tools::Scheme::SEMI;
    else if ( str == "mid" )
        parms.mIntegrator = openvdb::tools::Scheme::MID;
    else if ( str == "rk3" )
        parms.mIntegrator = openvdb::tools::Scheme::RK3;
    else if ( str == "rk4" )
        parms.mIntegrator = openvdb::tools::Scheme::RK4;
    else if ( str == "mac" )
        parms.mIntegrator = openvdb::tools::Scheme::MAC;
    else if ( str == "bfecc" )
        parms.mIntegrator = openvdb::tools::Scheme::BFECC;
    else {
        addError(SOP_MESSAGE, "Invalid scheme");
        return UT_ERROR_ABORT;
    }

    parms.mStaggered = parms.mVelocityGrid->getGridClass() == openvdb::GRID_STAGGERED;

    return error();
}


////////////////////////////////////////


template <bool StaggeredVelocity>
bool
SOP_OpenVDB_Advect_Density::processGrids(const AdvectionParms& parms, hvdb::Interrupter& boss)
{
    typedef openvdb::Vec3SGrid VelGridT;
    typedef openvdb::tools::DensityAdvection<VelGridT, StaggeredVelocity, hvdb::Interrupter> AdvT;
    typename VelGridT::ConstPtr velGrid = hvdb::Grid::constGrid<VelGridT>(parms.mVelocityGrid);
    if (!velGrid) return false;

    AdvT adv(*velGrid, &boss);
    adv.setIntegrator(parms.mIntegrator);

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
                    << " Consider reducing the time-step!";
                addWarning(SOP_MESSAGE, tmp.str().c_str());
            }
            typename openvdb::FloatGrid::Ptr outGrid = adv.template advect<openvdb::FloatGrid,
                openvdb::tools::Sampler<1, false> >(inGrid, parms.mTimeStep);
            hvdb::replaceVdbPrimitive(*gdp, outGrid, *vdbPrim);
        } else if (vdbPrim->getStorageType() == UT_VDB_VEC3F) {
            vdbPrim->makeGridUnique();
            const openvdb::Vec3SGrid& inGrid = UTvdbGridCast<openvdb::Vec3SGrid>(vdbPrim->getGrid());
            const int d = adv.getMaxDistance(inGrid, parms.mTimeStep);
            if (d > 20) {
                std::ostringstream tmp;
                tmp << "Dilation by " << d << " voxels could be slow!"
                    << " Consider reducing the time-step!";
                addWarning(SOP_MESSAGE, tmp.str().c_str());
            }
            typename openvdb::Vec3SGrid::Ptr outGrid;
            if (inGrid.getGridClass() == openvdb::GRID_STAGGERED) {
                outGrid = adv.template advect<openvdb::Vec3SGrid,
                    openvdb::tools::Sampler<1, true> >(inGrid, parms.mTimeStep);
            } else {
                outGrid = adv.template advect<openvdb::Vec3SGrid,
                    openvdb::tools::Sampler<1, false> >(inGrid, parms.mTimeStep);
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
