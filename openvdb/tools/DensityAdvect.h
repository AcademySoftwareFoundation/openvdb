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
///////////////////////////////////////////////////////////////////////////
//
/// @author Ken Museth
///
/// @file DensityAdvect.h
///
/// @brief Hyperbolic advection of density (vs a level set interface)
///
/// @warning Currently assumes that the velocity and density grids are aligned.
///
/// @todo Allow for misaligned grids (as is the case for LevelSetAdvection).

#ifndef OPENVDB_TOOLS_DENSITY_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_DENSITY_ADVECT_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/NullInterrupter.h>
#include "Interpolation.h"// for Sampler
#include "VelocityFields.h" // for VelocityIntegrator
#include "Morphology.h"//for dilateVoxels
#include "Prune.h"// for prune
#include "Statistics.h" // for extrema

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Performs Semi-Lagrangian advection of an arbitrary grid in a
///        passive velocity field.
///
/// @warning The velocity field is currently assumed to be in the same
///          frame of reference as the density grid.
template<typename VelocityGridT = Vec3fGrid,
         bool StaggeredVelocity = false,
         typename InterrupterType = util::NullInterrupter>
class DensityAdvection
{
public:
    /// @brief Constructor
    ///
    /// @param velGrid     Velocity grid responsible for the (passive) density advection.
    /// @param interrupter Optional interrupter used to prematurely end computations.
    ///
    /// @note The velocity field is assumed to be constant for the duration of the
    ///       advection and in the same frame of reference as the density grid.
    DensityAdvection(const VelocityGridT& velGrid, InterrupterType* interrupter = NULL)
        : mVelGrid(velGrid)
        , mInterrupter(interrupter)
        , mCountRK(1)
        , mOrderRK(1)
        , mGrainSize(1)
    {
        math::Extrema e = extrema(velGrid.cbeginValueAll(), /*threading*/true);
        e.add(velGrid.background().length());
        mMaxVelocity = e.max();
    }

    virtual ~DensityAdvection()
    {
    }

    /// @brief Define the number of integration steps to be performed per
    /// advection step (the default is one).
    void setIntegrationCount(size_t iterations) { mCountRK = iterations; }

    /// @return the number of integration steps performed per advection step.
    size_t getIntegrationCount() const { return mCountRK; }

    /// @brief Define the order of the Runge-Kutta integration scheme
    /// employed during the back-tracing in the semi-lagrangian integration.
    ///
    /// @note Note the order of the velocity sampling is always one,
    ///       i.e. based on tri-linear interpolation! However the
    ///       order of the interpolation kernel used to interpolate
    ///       the density is defined by the DensitySampleT template type.
    void setIntegrationOrder(size_t orderRK) { mOrderRK = math::Min(orderRK, size_t(4)); }

    /// @return The order of the Runge-Kutta integration scheme employed
    ///         during the back-tracing in the semi-lagrangian integration.
    size_t getIntegrationOrder() const { return mOrderRK; }

     /// @return the grain-size used for multi-threading
    size_t  getGrainSize() const { return mGrainSize; }

    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(size_t grainsize) { mGrainSize = grainsize; }

    /// @return Returns a new density grid that is the result of passive advection
    ///         of all the active values the input density for the
    ///         time = dt * IntegrationCount.
    ///
    /// @param grid0   The input density grid to be advected (unmodified)
    /// @param dt      Time-step of the Runge-Kutta integrator.
    ///
    /// @note It is important to note that the input density @c grid1 is assumed
    ///       to be represented in the same frame of reference as the velocity field!
    ///
    /// @details This method will advect all of the active values in
    ///          the input density @a grid0. To achieve this a
    ///          deep-copy is dilated to account for the material
    ///          transport. So for large time-steps and/or fast
    ///          velocity fields this dilation can be slow. In such
    ///          cases consider using the alternate advect method below.
    ///
    /// @throw RuntimeError if @a grid0 is not aligned with the velocity grid.
    template<typename DensityGridT,
             typename DensitySamplerT>
    typename DensityGridT::Ptr advect(const DensityGridT& grid0, float dt)
    {
        if (grid0.transform() != mVelGrid.transform()) {
            OPENVDB_THROW(RuntimeError, "Density grid and velocity grid are misaligned! Consider "
                          "resampling either of the two grids into the index space of the other.");
        }
        typename DensityGridT::Ptr grid1 = grid0.deepCopy();
        dilateVoxels( grid1->tree(), static_cast<int>(math::RoundUp(mMaxVelocity * dt * mCountRK)));
        this->template process<DensityGridT, DensitySamplerT>(*grid1, grid0, dt);
        return grid1;
    }

    /// @return Returns a new density grid that is the result of
    ///         passive advection of the active values in @a grid0
    ///         that intersect the active values in @c mask. The time
    ///         of the output grid corresponds to dt * IntegrationCount.
    ///
    /// @param grid0     The input density grid to be advected (unmodified).
    /// @param mask      The mask of active values defining the active voxels
    ///                  in @c grid0 on which to perform advection. Only
    ///                  if a value is active in both grids will it be modified.
    /// @param dt        Time-step for a single Runge-Kutta integration step.
    ///
    /// @note It is important to note that the input density @c grid0 is assumed
    ///       to be represented in the same frame of reference as the velocity field!
    ///
    /// @throw RuntimeError if @a grid0 is not aligned with the velocity grid.
    template<typename DensityGridT,
             typename MaskGridT,
             typename DensitySamplerT>
    typename DensityGridT::Ptr advect(const DensityGridT& grid0, const MaskGridT& mask, float dt)
    {
        if (grid0.transform() != mVelGrid.transform()) {
            OPENVDB_THROW(RuntimeError, "Density grid and velocity grid are misaligned! Consider "
                          "resampling either of the two grids into the index space of the other.");
        }
        typename DensityGridT::Ptr grid1 = grid0.deepCopy();
        dilateVoxels( grid1->tree(), static_cast<int>(math::RoundUp(mMaxVelocity * dt * mCountRK)));
        grid1->topologyIntersection(mask);
        pruneInactive(grid1->tree(), mGrainSize>0, mGrainSize);
        this->template process<DensityGridT, DensitySamplerT>(*grid1, grid0, dt);
        grid1->topologyUnion(grid0);
        return grid1;
    }

private:
    // disallow copy construction and copy by assinment!
    DensityAdvection(const DensityAdvection&);// not implemented
    DensityAdvection& operator=(const DensityAdvection&);// not implemented

    template<typename DensityGridT, typename DensitySamplerT>
    void process(DensityGridT& grid1, const DensityGridT& grid0, float dt)
    {
        mDt = dt;
        grid1.tree().voxelizeActiveTiles();
        if (mOrderRK == 1) {
            Advect<DensityGridT, 1, DensitySamplerT> adv(grid0, *this);
            adv.run(grid1);
        } else if (mOrderRK == 2) {
            Advect<DensityGridT, 2, DensitySamplerT> adv(grid0, *this);
            adv.run(grid1);
        } else if (mOrderRK == 3) {
            Advect<DensityGridT, 3, DensitySamplerT> adv(grid0, *this);
            adv.run(grid1);
        } else if (mOrderRK == 4) {
            Advect<DensityGridT, 4, DensitySamplerT> adv(grid0, *this);
            adv.run(grid1);
        }
        pruneInactive(grid1.tree(), mGrainSize>0, mGrainSize);
    }

    // Private class that implements the semi-lagrangian integration
    template<typename DensityGridT, size_t OrderRK, typename SamplerT>
    struct Advect
    {
        typedef typename DensityGridT::TreeType      TreeT;
        typedef typename DensityGridT::ConstAccessor AccT;
        typedef typename TreeT::ValueType            ValueT;
        typedef typename tree::LeafManager<TreeT>    LeafManagerT;
        typedef typename LeafManagerT::LeafRange     LeafRangeT;
        typedef VelocityIntegrator<VelocityGridT, StaggeredVelocity> VelocityIntegratorT;

        Advect(const DensityGridT& grid, const DensityAdvection& parent)
            : mDensityAcc(grid.getAccessor())
            , mVelocityInt(parent.mVelGrid)
            , mParent(&parent)
        {
        }
        Advect(const Advect& other)
            : mDensityAcc(other.mDensityAcc.tree())
            , mVelocityInt(other.mVelocityInt)
            , mParent(other.mParent)
        {
        }
        void run(DensityGridT& grid)
        {
            if (mParent->mInterrupter) mParent->mInterrupter->start("Advecting density");

            const LeafManagerT manger(grid.tree());
            const LeafRangeT range = manger.leafRange(mParent->mGrainSize);

            if (mParent->mGrainSize > 0) {
                tbb::parallel_for(range, *this);
            } else {
                (*this)(range);
            }

            if (mParent->mInterrupter) mParent->mInterrupter->end();
        }
        // Called by tbb::parallel_for()
        void operator() (const LeafRangeT& range) const
        {
            if (mParent->mInterrupter && util::wasInterrupted(mParent->mInterrupter)) {
                tbb::task::self().cancel_group_execution();
                return;
            }
            const float dt = - mParent->mDt;//back-tracking
            const int n = static_cast<int>(mParent->mCountRK);
            const ValueT backg = mDensityAcc.tree().background();
            for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
                ValueT* phi = leafIter.buffer(0).data();
                typedef typename TreeT::LeafNodeType::ValueOnIter VoxelIterT;
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    Vec3s p = voxelIter.getCoord().asVec3s();
                    for (int i = 0; i < n; ++i) {
                        mVelocityInt.template rungeKutta<OrderRK, Vec3s>(dt, p);
                    }
                    const ValueT sample = SamplerT::sample(mDensityAcc, p);
                    const Index j = voxelIter.pos();
                    if (math::isApproxEqual(sample, backg, math::Delta<ValueT>::value())) {
                        phi[j] = backg;
                        leafIter->setValueOff(j);
                    } else {
                        phi[j] = sample;
                    }
                }//loop over active voxels
            }//loop over leaf nodes
        }
        // Public member data of the private Advect class
        AccT                    mDensityAcc;
        VelocityIntegratorT     mVelocityInt;
        const DensityAdvection* mParent;
    };// end of private Advect class

    // Protected member data of DensityAdvection
    const VelocityGridT& mVelGrid;
    double               mMaxVelocity;
    InterrupterType*     mInterrupter;
    float                mDt;// time step per RK integration step
    size_t               mCountRK;// number of RK integration steps
    size_t               mOrderRK;// order of the RK integrator
    size_t               mGrainSize;// for multi-threading (0 means no threading)
};//end of DensityAdvection class

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_DENSITY_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
