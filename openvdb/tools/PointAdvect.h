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
/// @author Ken Museth, D.J. Hill (openvdb port, added staggered grid support)
/// @file PointAdvect.h
///
/// @brief Class PointAdvect advects points (with position) in a static velocity field

#ifndef OPENVDB_TOOLS_POINT_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_ADVECT_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/math/Math.h>             // min
#include <openvdb/Types.h>                 // Vec3 types and version number
#include <openvdb/Grid.h>                  // grid
#include <openvdb/util/NullInterrupter.h>
#include "Interpolation.h"                 // sampling

#include <boost/static_assert.hpp>
#include <tbb/blocked_range.h>             // threading
#include <tbb/parallel_for.h>              // threading
#include <tbb/task.h>                      // for cancel


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// Class that holds a Vec3 grid, to be interpreted as the closest point to a constraint
/// surface.  Supports a method to allow a point to be projected onto the closest point
/// on the constraint surface.  Uses Caching.
template<typename CptGridT = Vec3fGrid>
class ClosestPointProjector
{
public:
    typedef CptGridT                            CptGridType;
    typedef typename CptGridType::ConstAccessor CptAccessor;
    typedef typename CptGridType::ValueType     CptValueType;

    ClosestPointProjector():
        mCptIterations(0)
    {
    }
    ClosestPointProjector(const CptGridType& cptGrid, int n):
        mCptGrid(&cptGrid),
        mCptAccessor(cptGrid.getAccessor()),
        mCptIterations(n)
    {
    }
    ClosestPointProjector(const ClosestPointProjector &other):
        mCptGrid(other.mCptGrid),
        mCptAccessor(mCptGrid->getAccessor()),
        mCptIterations(other.mCptIterations)
    {
    }
    void setConstraintIterations(unsigned int cptIterations) { mCptIterations = cptIterations; }
    unsigned int numIterations() { return mCptIterations; }

    // point constraint
    template <typename LocationType>
    inline void projectToConstraintSurface(LocationType& W) const
    {
        /// Entries in the CPT tree are the closest point to the constraint surface.
        /// The interpolation step in sample introduces error so that the result
        /// of a single sample may not lie exactly on the surface.  The iterations
        /// in the loop exist to minimize this error.
        CptValueType result(W[0], W[1],W[2]);
        for (unsigned int i = 0; i < mCptIterations; ++i) {
            const Vec3R location = mCptGrid->worldToIndex(Vec3R(result[0], result[1], result[2]));
            BoxSampler::sample<CptAccessor>(mCptAccessor, location, result);
        }
        W[0] = result[0];
        W[1] = result[1];
        W[2] = result[2];
    }

private:
    const CptGridType*  mCptGrid; // Closest-Point-Transform vector field
    CptAccessor         mCptAccessor;
    unsigned int        mCptIterations;
};// end of ClosestPointProjector class


/// Class to hold a Vec3 field interperated as a velocity field.
/// Primarily exists to provide a method(s) that integrate a passive
/// point forward in the velocity field for a single time-step (dt)
template<typename GridT = Vec3fGrid, bool StaggeredVelocity = false>
class VelocitySampler
{
public:
    typedef typename GridT::ConstAccessor   VelAccessor;
    typedef typename GridT::ValueType       VelValueType;

    VelocitySampler(const GridT& velGrid):
        mVelGrid(&velGrid),
        mVelAccessor(mVelGrid->getAccessor())
    {
    }
    VelocitySampler(const VelocitySampler& other):
        mVelGrid(other.mVelGrid),
        mVelAccessor(mVelGrid->getAccessor())
    {
    }
    ~VelocitySampler()
    {
    }
    /// Samples the velocity at position W onto result. Supports both
    /// staggered (i.e. MAC) and collocated velocity grids.
    template <typename LocationType>
    inline void sample(const LocationType& W, VelValueType& result) const
    {
        const Vec3R location = mVelGrid->worldToIndex(Vec3R(W[0], W[1], W[2]));
        /// Note this if-branch is optimized away at compile time
        if (StaggeredVelocity) {
            // the velocity Grid stores data in MAC-style staggered layout
            StaggeredBoxSampler::sample<VelAccessor>(mVelAccessor, location, result);
        } else {
            // the velocity Grid uses collocated data
            BoxSampler::sample<VelAccessor>(mVelAccessor, location, result);
        }
    }

private:
    // holding the Grids for the transforms
    const GridT* mVelGrid; // Velocity vector field
    VelAccessor mVelAccessor;
};// end of VelocitySampler class


/// @brief Performs runge-kutta time integration of variable order in
/// a static velocity field
template<typename GridT = Vec3fGrid, bool StaggeredVelocity = false>
class VelocityIntegrator
{
public:
    typedef typename GridT::ValueType  VecType;
    typedef typename VecType::ValueType ElementType;

    VelocityIntegrator(const GridT& velGrid):
        mVelField(velGrid)
    {
    }
    // variable order Runge-Kutta time integration for a single time step
    template<int Order, typename LocationType>
    void rungeKutta(const float dt, LocationType& loc) {
        VecType P(loc[0],loc[1],loc[2]), V0, V1, V2, V3;

        BOOST_STATIC_ASSERT((Order < 5) &&  (Order > -1));
        /// Note the if-braching below is optimized away at compile time
        if (Order == 0) {
            // do nothing
            return ;
        } else if (Order == 1) {
            mVelField.sample(P, V0);
            P =  dt*V0;

        } else if (Order == 2) {
            mVelField.sample(P, V0);
            mVelField.sample(P + ElementType(0.5) * ElementType(dt) * V0, V1);
            P = dt*V1;

        } else if (Order == 3) {
            mVelField.sample(P, V0);
            mVelField.sample(P+ElementType(0.5)*ElementType(dt)*V0, V1);
            mVelField.sample(P+dt*(ElementType(2.0)*V1-V0), V2);
            P = dt*(V0 + ElementType(4.0)*V1 + V2)*ElementType(1.0/6.0);

        } else if (Order == 4) {
            mVelField.sample(P, V0);
            mVelField.sample(P+ElementType(0.5)*ElementType(dt)*V0, V1);
            mVelField.sample(P+ElementType(0.5)*ElementType(dt)*V1, V2);
            mVelField.sample(P+     dt*V2, V3);
            P = dt*(V0 + ElementType(2.0)*(V1 + V2) + V3)*ElementType(1.0/6.0);

        }
        loc += LocationType(P[0], P[1], P[2]);

    }
private:
    VelocitySampler<GridT, StaggeredVelocity> mVelField;
};// end of VelocityIntegrator class


////////////////////////////////////////


/// Performs passive or constrained advection of points in a velocity field
/// represented by an OpenVDB grid and an optional closest-point-transform (CPT)
/// represented in another OpenVDB grid.  Note the CPT is assumed to be
/// in world coordinates and NOT index coordinates!
/// Supports both collocated velocity grids and staggered velocity grids
///
/// The @c PointListT template argument refers to any class with the following
/// interface (e.g., std::vector<openvdb::Vec3f>):
/// @code
/// class PointList {
///     ...
/// public:
///     typedef internal_vector3_type value_type; // must support [] component access
///     openvdb::Index size() const;              // number of points in list
///     value_type& operator[](int n);            // world space position of nth point
/// };
/// @endcode
///
/// @note All methods (except size) are assumed to be thread-safe and
/// the positions are returned as non-const references since the
/// advection method needs to modify them!
template<typename GridT = Vec3fGrid,
         typename PointListT = std::vector<typename GridT::ValueType>,
         bool StaggeredVelocity = false,
         typename InterrupterType = util::NullInterrupter>
class PointAdvect
{
public:
    typedef GridT                                        GridType;
    typedef PointListT                                   PointListType;
    typedef typename PointListT::value_type              LocationType;
    typedef VelocityIntegrator<GridT, StaggeredVelocity> VelocityFieldIntegrator;

    PointAdvect(const GridT& velGrid, InterrupterType* interrupter=NULL) :
        mVelGrid(&velGrid),
        mPoints(NULL),
        mIntegrationOrder(1),
        mThreaded(true),
        mInterrupter(interrupter)
    {
    }
    PointAdvect(const PointAdvect &other) :
        mVelGrid(other.mVelGrid),
        mPoints(other.mPoints),
        mDt(other.mDt),
        mAdvIterations(other.mAdvIterations),
        mIntegrationOrder(other.mIntegrationOrder),
        mThreaded(other.mThreaded),
        mInterrupter(other.mInterrupter)
    {
    }
    virtual ~PointAdvect()
    {
    }
    /// If the order of the integration is set to zero no advection is performed
    bool earlyOut() const { return (mIntegrationOrder==0);}
    /// get & set
    void setThreaded(bool threaded) { mThreaded = threaded; }
    bool getThreaded() { return mThreaded; }
    void setIntegrationOrder(unsigned int order) {mIntegrationOrder = order;}

    /// Constrained advection of a list of points over a time = dt * advIterations
    void advect(PointListT& points, float dt, unsigned int advIterations = 1)
    {
        if (this->earlyOut()) return; // nothing to do!
        mPoints        = &points;
        mDt            = dt;
        mAdvIterations = advIterations;

        if (mInterrupter) mInterrupter->start("Advecting points by OpenVDB velocity field: ");
        if (mThreaded) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, mPoints->size()), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, mPoints->size()));
        }
        if (mInterrupter) mInterrupter->end();
    }

    /// Never call this method directly - it is use by TBB and has to be public!
    void operator() (const tbb::blocked_range<size_t> &range) const
    {
        if (mInterrupter && mInterrupter->wasInterrupted()) {
            tbb::task::self().cancel_group_execution();
        }

        VelocityFieldIntegrator  velField(*mVelGrid);
        switch (mIntegrationOrder) {
        case 1:
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    // loop over number of time steps
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<1>(mDt, X0);
                    }
                }
            }
            break;
        case 2:
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    // loop over number of time steps
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<2>(mDt, X0);
                    }
                }
            }
            break;
        case 3:
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    // loop over number of time steps
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<3>(mDt, X0);
                    }
                }
            }
            break;
        case 4:
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    // loop over number of time steps
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<4>(mDt, X0);
                    }
                }
            }
            break;
        }
    }

private:
    // the velocity field
    const GridType*        mVelGrid;

    // vertex list of all the points
    PointListT*            mPoints;

    // time integration parameters
    float                  mDt;                // time step
    unsigned int           mAdvIterations;     // number of time steps
    unsigned int           mIntegrationOrder;

    // operational parameters
    bool                   mThreaded;
    InterrupterType*       mInterrupter;

};//end of PointAdvect class


template<typename GridT = Vec3fGrid,
         typename PointListT = std::vector<typename GridT::ValueType>,
         bool StaggeredVelocity = false,
         typename CptGridType = GridT,
         typename InterrupterType = util::NullInterrupter>
class ConstrainedPointAdvect
{
public:
    typedef GridT                                        GridType;
    typedef typename PointListT::value_type              LocationType;
    typedef VelocityIntegrator<GridT, StaggeredVelocity> VelocityIntegratorType;
    typedef ClosestPointProjector<CptGridType>           ClosestPointProjectorType;
    typedef PointListT PointListType;

    ConstrainedPointAdvect(const GridType& velGrid,
        const GridType& cptGrid, int cptn, InterrupterType* interrupter = NULL):
        mVelGrid(&velGrid),
        mCptGrid(&cptGrid),
        mCptIter(cptn),
        mInterrupter(interrupter)
    {
    }
    ConstrainedPointAdvect(const ConstrainedPointAdvect& other):
        mVelGrid(other.mVelGrid),
        mCptGrid(other.mCptGrid),
        mCptIter(other.mCptIter),
        mPoints(other.mPoints),
        mDt(other.mDt),
        mAdvIterations(other.mAdvIterations),
        mIntegrationOrder(other.mIntegrationOrder),
        mThreaded(other.mThreaded),
        mInterrupter(other.mInterrupter)
    {
    }
    virtual ~ConstrainedPointAdvect(){}

    void setConstraintIterations(unsigned int cptIter) {mCptIter = cptIter;}
    void setIntegrationOrder(unsigned int order) {mIntegrationOrder = order;}

    void setThreaded(bool threaded) { mThreaded = threaded; }
    bool getThreaded() { return mThreaded; }

    /// Constrained Advection a list of points over a time = dt * advIterations
    void advect(PointListT& points, float dt, unsigned int advIterations = 1)
    {
        mPoints = &points;
        mDt     = dt;

        if (mIntegrationOrder==0 && mCptIter == 0) {
            return; // nothing to do!
        }
        (mIntegrationOrder>0) ? mAdvIterations = advIterations : mAdvIterations = 1;

        if (mInterrupter) mInterrupter->start("Advecting points by OpenVDB velocity field: ");
        const size_t N = mPoints->size();

        if (mThreaded) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, N), *this);
        } else {
            (*this)(tbb::blocked_range<size_t>(0, N));
        }
        if (mInterrupter) mInterrupter->end();
    }


    /// Never call this method directly - it is use by TBB and has to be public!
    void operator() (const tbb::blocked_range<size_t> &range) const
    {
        if (mInterrupter && mInterrupter->wasInterrupted()) {
            tbb::task::self().cancel_group_execution();
        }

        VelocityIntegratorType velField(*mVelGrid);
        ClosestPointProjectorType cptField(*mCptGrid, mCptIter);
        switch (mIntegrationOrder) {
        case 0://pure CPT projection
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        cptField.projectToConstraintSurface(X0);
                    }
                }
            }
            break;
        case 1://1'th order advection and CPT projection
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<1>(mDt, X0);
                        cptField.projectToConstraintSurface(X0);
                    }
                }
            }
            break;
        case 2://2'nd order advection and CPT projection
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<2>(mDt, X0);
                        cptField.projectToConstraintSurface(X0);
                    }
                }
            }
            break;

        case 3://3'rd order advection and CPT projection
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<3>(mDt, X0);
                        cptField.projectToConstraintSurface(X0);
                    }
                }
            }
            break;
        case 4://4'th order advection and CPT projection
            {
                for (size_t n = range.begin(); n != range.end(); ++n) {
                    LocationType& X0 = (*mPoints)[n];
                    for (unsigned int i = 0; i < mAdvIterations; ++i) {
                        velField.template rungeKutta<4>(mDt, X0);
                        cptField.projectToConstraintSurface(X0);
                    }
                }
            }
            break;
        }
    }

private:
    const GridType*         mVelGrid;           // the velocity field
    const GridType*         mCptGrid;
    int                     mCptIter;
    PointListT*             mPoints;            // vertex list of all the points

    // time integration parameters
    float                   mDt;                // time step
    unsigned int            mAdvIterations;     // number of time steps
    unsigned int            mIntegrationOrder;  // order of Runge-Kutta integration
    // operational parameters
    bool                    mThreaded;
    InterrupterType*        mInterrupter;
};// end of ConstrainedPointAdvect class

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POINT_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
