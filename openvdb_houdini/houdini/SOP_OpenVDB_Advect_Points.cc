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
/// @file SOP_OpenVDB_Advect_Points.cc
///
/// @author FX R&D OpenVDB team
///
/// @brief SOP to perform advection of points through a velocity field.

#include <houdini_utils/ParmFactory.h>
#include <openvdb_houdini/Utils.h>
#include <openvdb_houdini/SOP_NodeVDB.h>
#include <openvdb/tools/PointAdvect.h>

#include <UT/UT_Interrupt.h>
#include <GA/GA_PageIterator.h>
#include <GU/GU_PrimPoly.h>

#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>

namespace hvdb = openvdb_houdini;
namespace hutil = houdini_utils;


////////////////////////////////////////

// Utilities

namespace {

// Add new items to the *end* of this list, and update NUM_PROPAGATION_TYPES.
enum PropagationType {
    PROPAGATION_TYPE_UNKNOWN = -1,
    PROPAGATION_TYPE_ADVECTION = 0,
    PROPAGATION_TYPE_PROJECTION,
    PROPAGATION_TYPE_CONSTRAINED_ADVECTION
};

enum { NUM_PROPAGATION_TYPES = PROPAGATION_TYPE_CONSTRAINED_ADVECTION + 1 };

std::string
propagationTypeToString(PropagationType pt)
{
    std::string ret;
    switch (pt) {
        case PROPAGATION_TYPE_ADVECTION:             ret = "advection";  break;
        case PROPAGATION_TYPE_PROJECTION:            ret = "projection"; break;
        case PROPAGATION_TYPE_CONSTRAINED_ADVECTION: ret = "cadvection"; break;
        case PROPAGATION_TYPE_UNKNOWN:               ret = "unknown";    break;
    }
    return ret;
}

std::string
propagationTypeToMenuName(PropagationType pt)
{
    std::string ret;
    switch (pt) {
        case PROPAGATION_TYPE_ADVECTION:             ret = "Advection";              break;
        case PROPAGATION_TYPE_PROJECTION:            ret = "Projection";             break;
        case PROPAGATION_TYPE_CONSTRAINED_ADVECTION: ret = "Constrained Advection";  break;
        case PROPAGATION_TYPE_UNKNOWN:               ret = "Unknown";                break;
    }
    return ret;
}

PropagationType
stringToPropagationType(const std::string& s)
{
    PropagationType ret = PROPAGATION_TYPE_UNKNOWN;

    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);

    if (str == propagationTypeToString(PROPAGATION_TYPE_ADVECTION)) {
        ret = PROPAGATION_TYPE_ADVECTION;
    } else if (str == propagationTypeToString(PROPAGATION_TYPE_PROJECTION)) {
        ret = PROPAGATION_TYPE_PROJECTION;
    } else if (str == propagationTypeToString(PROPAGATION_TYPE_CONSTRAINED_ADVECTION)) {
        ret = PROPAGATION_TYPE_CONSTRAINED_ADVECTION;
    }

    return ret;
}


// Add new items to the *end* of this list, and update NUM_INTEGRATION_TYPES.
enum IntegrationType {
    INTEGRATION_TYPE_UNKNOWN = -1,
    INTEGRATION_TYPE_FWD_EULER = 0,
    INTEGRATION_TYPE_RK_2ND,
    INTEGRATION_TYPE_RK_3RD,
    INTEGRATION_TYPE_RK_4TH
};

enum { NUM_INTEGRATION_TYPES = INTEGRATION_TYPE_RK_4TH + 1 };

std::string
integrationTypeToString(IntegrationType it)
{
    std::string ret;
    switch (it) {
        case INTEGRATION_TYPE_FWD_EULER:    ret = "fwd euler";  break;
        case INTEGRATION_TYPE_RK_2ND:       ret = "2nd rk";     break;
        case INTEGRATION_TYPE_RK_3RD:       ret = "3rd rk";     break;
        case INTEGRATION_TYPE_RK_4TH:       ret = "4th rk";     break;
        case INTEGRATION_TYPE_UNKNOWN:      ret = "unknown";    break;
    }
    return ret;
}

std::string
integrationTypeToMenuName(IntegrationType it)
{
    std::string ret;
    switch (it) {
        case INTEGRATION_TYPE_FWD_EULER:    ret = "Forward Euler";            break;
        case INTEGRATION_TYPE_RK_2ND:       ret = "Second-Order Runge-Kutta"; break;
        case INTEGRATION_TYPE_RK_3RD:       ret = "Third-Order Runge-Kutta";  break;
        case INTEGRATION_TYPE_RK_4TH:       ret = "Fourth-Order Runge-Kutta"; break;
        case INTEGRATION_TYPE_UNKNOWN:      ret = "Unknown";                  break;
    }
    return ret;
}

IntegrationType
stringToIntegrationType(const std::string& s)
{
    IntegrationType ret = INTEGRATION_TYPE_UNKNOWN;

    std::string str = s;
    boost::trim(str);
    boost::to_lower(str);

    if (str == integrationTypeToString(INTEGRATION_TYPE_FWD_EULER)) {
        ret = INTEGRATION_TYPE_FWD_EULER;
    } else if (str == integrationTypeToString(INTEGRATION_TYPE_RK_2ND)) {
        ret = INTEGRATION_TYPE_RK_2ND;
    } else if (str == integrationTypeToString(INTEGRATION_TYPE_RK_3RD)) {
        ret = INTEGRATION_TYPE_RK_3RD;
    } else if (str == integrationTypeToString(INTEGRATION_TYPE_RK_4TH)) {
        ret = INTEGRATION_TYPE_RK_4TH;
    }

    return ret;
}


struct AdvectionParms
{
    AdvectionParms(GU_Detail *outputGeo)
        : mOutputGeo(outputGeo)
        , mPointGeo(NULL)
        , mPointGroup(NULL)
        , mVelPrim(NULL)
        , mCptPrim(NULL)
        , mPropagationType(PROPAGATION_TYPE_ADVECTION)
        , mIntegrationType(INTEGRATION_TYPE_FWD_EULER)
        , mTimeStep(1.0)
        , mIterations(1)
        , mSteps(1)
        , mStaggered(false)
        , mStreamlines(false)
    {
    }

    GU_Detail* mOutputGeo;
    const GU_Detail* mPointGeo;
    const GA_PointGroup* mPointGroup;
    const GU_PrimVDB *mVelPrim;
    const GU_PrimVDB *mCptPrim;
    PropagationType mPropagationType;
    IntegrationType mIntegrationType;
    float mTimeStep;
    int mIterations, mSteps;
    bool mStaggered, mStreamlines;
};


/// @brief Creates a new line segment for each point in @c ptnGeo
/// @note The lines will only have one node.
void
createNewLines(GU_Detail& geo, const GA_PointGroup* group)
{
    GA_SplittableRange ptnRange(geo.getPointRange(group));
    GA_Offset start, end, pt;

    for (GA_PageIterator pIt = ptnRange.beginPages(); !pIt.atEnd(); ++pIt) {
        for (GA_Iterator it(pIt.begin()); it.blockAdvance(start, end); ) {
            for (GA_Offset i = start; i < end; ++i) {

                pt = geo.appendPointOffset();
                geo.setPos3(pt, geo.getPos3(i));

                GU_PrimPoly& prim = *GU_PrimPoly::build(&geo, 0, GU_POLY_OPEN, 0);
                prim.appendVertex(pt);
            }
        }
    }
}


/// @brief Append a new node to each line.
/// @note The numbers of lines and points have to match.
void
appendLineNodes(GU_Detail& geo, GA_Size firstline, const GU_Detail& ptnGeo)
{
    GA_SplittableRange ptnRange(ptnGeo.getPointRange());
    GA_Offset start, end, pt;

    GA_Size n = firstline, N = geo.getNumPrimitives();


    for (GA_PageIterator pIt = ptnRange.beginPages(); !pIt.atEnd(); ++pIt) {
        for (GA_Iterator it(pIt.begin()); it.blockAdvance(start, end); ) {
            for (GA_Offset i = start; i < end; ++i) {

                pt = geo.appendPointOffset();
                geo.setPos3(pt, ptnGeo.getPos3(i));

                GA_Offset offset = geo.primitiveOffset(n);
                GU_PrimPoly& prim = *static_cast<GU_PrimPoly*>(
                    geo.getPrimitiveList().get(offset));

                prim.appendVertex(pt);

                if (++n == N) break;
            }
            if (n == N) break;
        }
        if (n == N) break;
    }
}


// Threaded closest point projection
template<typename GridType>
class ProjectionOp
{
    typedef openvdb::tools::ClosestPointProjector<GridType> ProjectorType;
    typedef typename GridType::ValueType VectorType;
    typedef typename VectorType::ValueType ElementType;

public:

    ProjectionOp(const GridType& cptGrid, int cptIterations, GU_Detail& geo,
        hvdb::Interrupter& boss)
        : mProjector(cptGrid, cptIterations)
        , mGeo(geo)
        , mBoss(boss)
    {
    }

    void operator()(const GA_SplittableRange &range) const
    {
        GA_Offset start, end;
        UT_Vector3 p;
        VectorType w;

        for (GA_PageIterator pIt = range.beginPages(); !pIt.atEnd(); ++pIt) {

            if (mBoss.wasInterrupted()) return;

            for (GA_Iterator it(pIt.begin()); it.blockAdvance(start, end); ) {

                if (mBoss.wasInterrupted()) return;

                for (GA_Offset i = start; i < end; ++i) {

                    p = mGeo.getPos3(i);
                    w[0] = ElementType(p[0]);
                    w[1] = ElementType(p[1]);
                    w[2] = ElementType(p[2]);

                    mProjector.projectToConstraintSurface(w);

                    p[0] = UT_Vector3::value_type(w[0]);
                    p[1] = UT_Vector3::value_type(w[1]);
                    p[2] = UT_Vector3::value_type(w[2]);
                    mGeo.setPos3(i, p);
                }
            }
        }
    }

private:
    ProjectorType mProjector;
    GU_Detail& mGeo;
    hvdb::Interrupter& mBoss;
};


class Projection
{
public:
    Projection(AdvectionParms& parms, hvdb::Interrupter& boss)
        : mParms(parms)
        , mBoss(boss)
    {
    }

    template<typename GridType>
    void operator()(const GridType& grid)
    {
        if (mBoss.wasInterrupted()) return;

        ProjectionOp<GridType> op(grid, mParms.mIterations, *mParms.mOutputGeo, mBoss);
        UTparallelFor(GA_SplittableRange(mParms.mOutputGeo->getPointRange(mParms.mPointGroup)), op);
    }

private:
    AdvectionParms& mParms;
    hvdb::Interrupter& mBoss;
};


// Threaded point advection
template<typename GridType, int IntegrationOrder, bool StaggeredVelocity, bool Constrained = false>
class AdvectionOp
{
    typedef openvdb::tools::VelocityIntegrator<GridType, StaggeredVelocity> IntegrationType;
    typedef openvdb::tools::ClosestPointProjector<GridType> ProjectorType; // Used for constrained advection

    typedef typename GridType::ValueType VectorType;
    typedef typename VectorType::ValueType ElementType;

public:

    AdvectionOp(const GridType& velocityGrid, GU_Detail& geo, hvdb::Interrupter& boss,
        float timeStep, GA_ROHandleF traillen, int steps)
        : mVelocityGrid(velocityGrid)
        , mCptGrid(NULL)
        , mGeo(geo)
        , mBoss(boss)
        , mTimeStep(timeStep)
        , mTrailLen(traillen)
        , mSteps(steps)
        , mCptIterations(0)
    {
    }

    AdvectionOp(const GridType& velocityGrid, const GridType& cptGrid, GU_Detail& geo,
        hvdb::Interrupter& boss, float timeStep, int steps, int cptIterations)
        : mVelocityGrid(velocityGrid)
        , mCptGrid(&cptGrid)
        , mGeo(geo)
        , mBoss(boss)
        , mTimeStep(timeStep)
        , mSteps(steps)
        , mCptIterations(cptIterations)
    {
    }

    void operator()(const GA_SplittableRange &range) const
    {
        GA_Offset start, end;
        UT_Vector3 p;
        VectorType w;

        IntegrationType integrator(mVelocityGrid);

        // Constrained-advection compiled out if Constrained == false
        boost::scoped_ptr<ProjectorType> projector(NULL);
        if (Constrained && mCptGrid != NULL) {
            projector.reset(new ProjectorType(*mCptGrid, mCptIterations));
        }

        for (GA_PageIterator pIt = range.beginPages(); !pIt.atEnd(); ++pIt) {

            if (mBoss.wasInterrupted()) return;

            for (GA_Iterator it(pIt.begin()); it.blockAdvance(start, end); ) {

                if (mBoss.wasInterrupted()) return;

                for (GA_Offset i = start; i < end; ++i) {

                    p = mGeo.getPos3(i);
                    w[0] = ElementType(p[0]);
                    w[1] = ElementType(p[1]);
                    w[2] = ElementType(p[2]);

                    float timestep = mTimeStep;
                    if (mTrailLen.isValid()) {
                        timestep *= mTrailLen.get(i);
                    }

                    for (int n = 0; n < mSteps; ++n) {
                        integrator.template rungeKutta<IntegrationOrder, VectorType>(timestep, w);

                        if (Constrained) projector->projectToConstraintSurface(w);
                    }

                    p[0] = UT_Vector3::value_type(w[0]);
                    p[1] = UT_Vector3::value_type(w[1]);
                    p[2] = UT_Vector3::value_type(w[2]);
                    mGeo.setPos3(i, p);
                }
            }
        }
    }

private:
    const GridType& mVelocityGrid;
    const GridType* mCptGrid;
    GU_Detail& mGeo;
    hvdb::Interrupter& mBoss;
    float mTimeStep;
    GA_ROHandleF mTrailLen;
    const int mSteps, mCptIterations;
};


class Advection
{
public:
    Advection(AdvectionParms& parms, hvdb::Interrupter& boss)
        : mParms(parms)
        , mBoss(boss)
    {
    }

    template<typename GridType, int IntegrationOrder, bool StaggeredVelocity>
    void advection(const GridType& velocityGrid)
    {
        if (mBoss.wasInterrupted()) return;

        if (!mParms.mStreamlines) { // Advect points

            GA_ROHandleF traillen_h(mParms.mOutputGeo, GA_ATTRIB_POINT, "traillen");

            AdvectionOp<GridType, IntegrationOrder, StaggeredVelocity>
                op(velocityGrid, *mParms.mOutputGeo, mBoss, mParms.mTimeStep,
                    traillen_h, mParms.mSteps);

            UTparallelFor(
                GA_SplittableRange(mParms.mOutputGeo->getPointRange(mParms.mPointGroup)), op);

        } else { // Advect points and generate streamlines.

            GA_Index firstline = mParms.mOutputGeo->getNumPrimitives();

            GU_Detail geo;
            geo.mergePoints(*mParms.mOutputGeo, mParms.mPointGroup);

            createNewLines(*mParms.mOutputGeo, mParms.mPointGroup);

            for (int n = 0; n < mParms.mSteps; ++n) {

                if (mBoss.wasInterrupted()) return;

                GA_ROHandleF traillen_h(&geo, GA_ATTRIB_POINT, "traillen");

                AdvectionOp<GridType, IntegrationOrder, StaggeredVelocity>
                    op(velocityGrid, geo, mBoss, mParms.mTimeStep, traillen_h, 1);

                UTparallelFor(GA_SplittableRange(geo.getPointRange()), op);

                appendLineNodes(*mParms.mOutputGeo, firstline, geo);
            }
        }
    }

    template<typename GridType, int IntegrationOrder, bool StaggeredVelocity>
    void constrainedAdvection(const GridType& velocityGrid)
    {
        const GridType& cptGrid = static_cast<const GridType&>(mParms.mCptPrim->getGrid());
        typedef AdvectionOp<GridType, IntegrationOrder, StaggeredVelocity, /*Constrained*/true>
            AdvectionOp;

        if (mBoss.wasInterrupted()) return;

        if (!mParms.mStreamlines) { // Advect points
            AdvectionOp op(velocityGrid, cptGrid, *mParms.mOutputGeo, mBoss,
                mParms.mTimeStep, mParms.mSteps, mParms.mIterations);

            UTparallelFor(
                GA_SplittableRange(mParms.mOutputGeo->getPointRange(mParms.mPointGroup)), op);

        } else { // Advect points and generate streamlines.

            GA_Index firstline = mParms.mOutputGeo->getNumPrimitives();

            GU_Detail geo;
            geo.mergePoints(*mParms.mOutputGeo, mParms.mPointGroup);

            createNewLines(*mParms.mOutputGeo, mParms.mPointGroup);

            for (int n = 0; n < mParms.mSteps; ++n) {

                if (mBoss.wasInterrupted()) return;

                AdvectionOp op(velocityGrid, cptGrid, geo, mBoss,
                    mParms.mTimeStep, 1, mParms.mIterations);

                UTparallelFor(GA_SplittableRange(geo.getPointRange()), op);

                appendLineNodes(*mParms.mOutputGeo, firstline, geo);
            }
        }
    }

    // Resolves velocity representation and advection type
    template<typename GridType, int IntegrationOrder>
    void resolveAdvection(const GridType& velocityGrid)
    {
        if (mBoss.wasInterrupted()) return;

        if (mParms.mPropagationType == PROPAGATION_TYPE_ADVECTION) {
            if (!mParms.mStaggered) advection<GridType, IntegrationOrder, false>(velocityGrid);
            else advection<GridType, IntegrationOrder, true>(velocityGrid);
        } else if (mParms.mCptPrim != NULL) { // constrained
            if (!mParms.mStaggered) {
                constrainedAdvection<GridType, IntegrationOrder, false>(velocityGrid);
            } else {
                constrainedAdvection<GridType, IntegrationOrder, true>(velocityGrid);
            }
        }
    }

    template<typename GridType>
    void operator()(const GridType& velocityGrid)
    {
        if (mBoss.wasInterrupted()) return;

        // Resolve integration order
        switch (mParms.mIntegrationType) {
            case INTEGRATION_TYPE_FWD_EULER: resolveAdvection<GridType, 1>(velocityGrid); break;
            case INTEGRATION_TYPE_RK_2ND:    resolveAdvection<GridType, 2>(velocityGrid); break;
            case INTEGRATION_TYPE_RK_3RD:    resolveAdvection<GridType, 3>(velocityGrid); break;
            case INTEGRATION_TYPE_RK_4TH:    resolveAdvection<GridType, 4>(velocityGrid); break;
            case INTEGRATION_TYPE_UNKNOWN: break;
        }
    }

private:
    AdvectionParms& mParms;
    hvdb::Interrupter& mBoss;
};

} // namespace


////////////////////////////////////////

// SOP Declaration

class SOP_OpenVDBAdvectPoints: public hvdb::SOP_NodeVDB
{
public:
    SOP_OpenVDBAdvectPoints(OP_Network*, const char* name, OP_Operator*);
    virtual ~SOP_OpenVDBAdvectPoints() {};

    static OP_Node* factory(OP_Network*, const char* name, OP_Operator*);

    virtual int isRefInput(unsigned i ) const { return (i > 0); }

protected:
    virtual OP_ERROR cookMySop(OP_Context&);
    virtual bool updateParmsFlags();

    bool evalAdvectionParms(OP_Context&, AdvectionParms&);
};


////////////////////////////////////////

// Build UI and register this operator

void
newSopOperator(OP_OperatorTable* table)
{
    if (table == NULL) return;

    hutil::ParmList parms;

    // Points to process
    parms.add(hutil::ParmFactory(PRM_STRING, "ptnGroup", "Point Group")
        .setChoiceList(&SOP_Node::pointGroupMenu));

    // Velocity grid
    parms.add(hutil::ParmFactory(PRM_STRING, "velGroup", "Velocity VDB")
        .setHelpText("Velocity grid")
        .setChoiceList(&hutil::PrimGroupMenuInput2));

    // Closest point grid
    parms.add(hutil::ParmFactory(PRM_STRING, "cptGroup", "Closest-Point VDB")
        .setHelpText("Vector grid that in each voxel stores the closest point on a surface.")
        .setChoiceList(&hutil::PrimGroupMenuInput3));

    // Propagation scheme
    {
        std::vector<std::string> items;
        for (int i = 0; i < NUM_PROPAGATION_TYPES; ++i) {
            PropagationType pt = PropagationType(i);
            items.push_back(propagationTypeToString(pt)); // token
            items.push_back(propagationTypeToMenuName(pt)); // label
        }

        parms.add(hutil::ParmFactory(PRM_STRING, "propagation", "Operation")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(items[0])
            .setHelpText("Advection: Move the point along the velocity field.\n"
                "Projection: Move point to the nearest surface point.\n"
                "Projected advection: Advect, then project to the nearest surface point."));
    }

    // Integration scheme
    {
        std::vector<std::string> items;
        for (int i = 0; i < NUM_INTEGRATION_TYPES; ++i) {
            IntegrationType it = IntegrationType(i);
            items.push_back(integrationTypeToString(it)); // token
            items.push_back(integrationTypeToMenuName(it)); // label
        }

        parms.add(hutil::ParmFactory(PRM_STRING, "integration", "Integration")
            .setChoiceListItems(PRM_CHOICELIST_SINGLE, items)
            .setDefault(items[0])
            .setHelpText("Lower order means faster performance, "
                "but the points will not follow the velocity field closely."));
    }

    // Closest point iterations
    parms.add(hutil::ParmFactory(PRM_INT_J, "cptIterations", "Iterations")
        .setDefault(PRMzeroDefaults )
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10)
        .setHelpText("The interpolation step when sampling nearest points introduces\n"
            "error so that the result of a single sample may not lie exactly\n"
            "on the surface. Multiple iterations help minimize this error."));

    // Time step
    parms.add(hutil::ParmFactory(PRM_FLT, "timeStep", "Time Step")
        .setDefault(1, "1.0/$FPS")
        .setRange(PRM_RANGE_UI, 0, PRM_RANGE_UI, 10));

    // Steps
    parms.add(hutil::ParmFactory(PRM_INT_J, "steps", "Substeps")
        .setDefault(1)
        .setHelpText("Number of timesteps to take per frame.")
        .setRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 10));

    // Output streamlines
    parms.add(hutil::ParmFactory(PRM_TOGGLE, "outputStreamlines", "Output Streamlines")
        .setDefault(PRMzeroDefaults)
        .setHelpText("Output the particle path as line segments."));

    // Obsolete parameters
    hutil::ParmList obsoleteParms;
    obsoleteParms.add(hutil::ParmFactory(PRM_TOGGLE, "staggered", "Staggered Velocities"));
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep1", "Sep"));
    obsoleteParms.add(hutil::ParmFactory(PRM_SEPARATOR, "sep2", "Sep"));

    // Register this operator.
    hvdb::OpenVDBOpFactory("OpenVDB Advect Points",
        SOP_OpenVDBAdvectPoints::factory, parms, *table)
        .setObsoleteParms(obsoleteParms)
        .addInput("Points to Advect")
        .addOptionalInput("Velocity VDB")
        .addOptionalInput("Closest Point VDB");
}


////////////////////////////////////////


OP_Node*
SOP_OpenVDBAdvectPoints::factory(OP_Network* net,
    const char* name, OP_Operator* op)
{
    return new SOP_OpenVDBAdvectPoints(net, name, op);
}


SOP_OpenVDBAdvectPoints::SOP_OpenVDBAdvectPoints(OP_Network* net,
    const char* name, OP_Operator* op):
    hvdb::SOP_NodeVDB(net, name, op)
{
}


////////////////////////////////////////

// Enable/disable or show/hide parameters in the UI.

bool
SOP_OpenVDBAdvectPoints::updateParmsFlags()
{
    bool changed = false;

    UT_String str;
    evalString(str, "propagation", 0, 0);
    const PropagationType propagation = stringToPropagationType(str.toStdString());

    changed |= enableParm("cptIterations", propagation != PROPAGATION_TYPE_ADVECTION);
    changed |= enableParm("integration", propagation != PROPAGATION_TYPE_PROJECTION);
    changed |= enableParm("timeStep", propagation != PROPAGATION_TYPE_PROJECTION);
    changed |= enableParm("steps", propagation != PROPAGATION_TYPE_PROJECTION);
    changed |= enableParm("outputStreamlines", propagation != PROPAGATION_TYPE_PROJECTION);

    changed |= setVisibleState("cptIterations", getEnableState("cptIterations"));
    changed |= setVisibleState("integration", getEnableState("integration"));
    changed |= setVisibleState("timeStep", getEnableState("timeStep"));
    changed |= setVisibleState("steps", getEnableState("steps"));
    changed |= setVisibleState("outputStreamlines", getEnableState("outputStreamlines"));

    return changed;
}


////////////////////////////////////////


OP_ERROR
SOP_OpenVDBAdvectPoints::cookMySop(OP_Context& context)
{
    try {
        hutil::ScopedInputLock lock(*this, context);

        duplicateSource(0, context);

        // Evaluate UI parameters
        AdvectionParms parms(gdp);
        if (!evalAdvectionParms(context, parms)) return error();

        hvdb::Interrupter boss("Processing points");

        switch (parms.mPropagationType) {

            case PROPAGATION_TYPE_ADVECTION:
            case PROPAGATION_TYPE_CONSTRAINED_ADVECTION:
            {
                Advection advection(parms, boss);
                GEOvdbProcessTypedGridVec3(*parms.mVelPrim, advection);
                break;
            }
            case PROPAGATION_TYPE_PROJECTION:
            {
                Projection projection(parms, boss);
                GEOvdbProcessTypedGridVec3(*parms.mCptPrim, projection);
                break;
            }
            case PROPAGATION_TYPE_UNKNOWN: break;
        }

        if (boss.wasInterrupted()) addWarning(SOP_MESSAGE, "processing was interrupted");
        boss.end();

    } catch (std::exception& e) {
        addError(SOP_MESSAGE, e.what());
    }

    return error();
}


////////////////////////////////////////


bool
SOP_OpenVDBAdvectPoints::evalAdvectionParms(OP_Context& context, AdvectionParms& parms)
{
    fpreal now = context.getTime();
    UT_String str;

    parms.mPointGeo = inputGeo(0);

    if (!parms.mPointGeo) {
        addError(SOP_MESSAGE, "Missing point input");
        return false;
    }

    evalString(str, "ptnGroup", 0, now);
    parms.mPointGroup = parsePointGroups(str, gdp);

    if (!parms.mPointGroup && str.length() > 0) {
        addWarning(SOP_MESSAGE, "Point group not found");
        return false;
    }

    evalString(str, "propagation", 0, now);
    parms.mPropagationType = stringToPropagationType(str.toStdString());

    if (parms.mPropagationType == PROPAGATION_TYPE_UNKNOWN) {
        addError(SOP_MESSAGE, "Unknown propargation scheme");
        return false;
    }

    if (parms.mPropagationType == PROPAGATION_TYPE_ADVECTION ||
        parms.mPropagationType == PROPAGATION_TYPE_CONSTRAINED_ADVECTION) {

        const GU_Detail* velGeo = inputGeo(1);

        if (!velGeo) {
            addError(SOP_MESSAGE, "Missing velocity grid input");
            return false;
        }

        evalString(str, "velGroup", 0, now);
        const GA_PrimitiveGroup *velGroup =
            matchGroup(const_cast<GU_Detail&>(*velGeo), str.toStdString());

        hvdb::VdbPrimCIterator it(velGeo, velGroup);
        parms.mVelPrim = *it;

        if (!parms.mVelPrim) {
            addError(SOP_MESSAGE, "Missing velocity grid");
            return false;
        }

        // Check if the velocity grid uses a staggered representation.
        parms.mStaggered =
            parms.mVelPrim->getGrid().getGridClass() == openvdb::GRID_STAGGERED;

        parms.mTimeStep = static_cast<float>(evalFloat("timeStep", 0, now));
        parms.mSteps    = evalInt("steps", 0, now);
        // The underlying code will accumulate, so to make it substeps
        // we need to divide out.
        parms.mTimeStep /= (float) parms.mSteps;
        parms.mStreamlines  = bool(evalInt("outputStreamlines", 0, now));

        evalString(str, "integration", 0, now);
        parms.mIntegrationType = stringToIntegrationType(str.toStdString());

        if (parms.mIntegrationType == INTEGRATION_TYPE_UNKNOWN) {
            addError(SOP_MESSAGE, "Unknown integration scheme");
            return false;
        }

    }

    if (parms.mPropagationType == PROPAGATION_TYPE_PROJECTION ||
        parms.mPropagationType == PROPAGATION_TYPE_CONSTRAINED_ADVECTION) {

        const GU_Detail* cptGeo = inputGeo(2);

        if (!cptGeo) {
            addError(SOP_MESSAGE, "Missing closest point grid input");
            return false;
        }

        evalString(str, "cptGroup", 0, now);
        const GA_PrimitiveGroup *cptGroup =
            matchGroup(const_cast<GU_Detail&>(*cptGeo), str.toStdString());

        hvdb::VdbPrimCIterator it(cptGeo, cptGroup);
        parms.mCptPrim = *it;

        if (!parms.mCptPrim) {
            addError(SOP_MESSAGE, "Missing closest point grid");
            return false;
        }

        parms.mIterations = evalInt("cptIterations", 0, now);
    }

    return true;
}

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
