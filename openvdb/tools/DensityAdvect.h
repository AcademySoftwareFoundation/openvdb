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
///////////////////////////////////////////////////////////////////////////
//
/// @author Ken Museth
///
/// @file DensityAdvect.h
///
/// @brief Sparse hyperbolic advection of volumes, e.g. a density or
///        velocity (vs a level set interface).
///
/// @todo  MacCormack, BFECC and better estimate of padding by dilation.

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

    /// @brief Define the number of integration sub-steps to be performed per
    /// advection time-step (the default is one).
    void setIntegrationCount(size_t iterations) { mCountRK = iterations; }

    /// @return the number of integration sub-steps performed per
    /// advection time-step.
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
    ///         during the back-tracing in the semi-Lagrangian integration.
    size_t getIntegrationOrder() const { return mOrderRK; }

     /// @return the grain-size used for multi-threading
    size_t  getGrainSize() const { return mGrainSize; }

    /// @brief Set the grain-size used for multi-threading.
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(size_t grainsize) { mGrainSize = grainsize; }

    /// @brief Return the maximum magnitude of the velocity in the
    /// advection velocity field (defined during construction).
    double getMaxVelocity() const { return mMaxVelocity; }

    /// @return Returns the maximum distance in voxel units of @a inGrid
    /// that a particle might travel in the time-step @a dt when advected
    /// in the velocity field defined during construction.
    ///
    /// @details This method is useful when dilating sparse density
    /// grids to pad boundary regions. Excessive dilation can be
    /// computationally expensive so use this method to prevent
    /// or warn against run-away computation.
    ///
    /// @throw RuntimeError if @a inGrid does not have uniform voxels.
    template<typename DensityGridT>
    int getMaxDistance(const DensityGridT& inGrid, double dt) const
    {
        if (!inGrid.hasUniformVoxels()) {
            OPENVDB_THROW(RuntimeError, "Density grid does not have uniform voxels!");
        }
        const double d = mMaxVelocity*math::Abs(dt)/inGrid.voxelSize()[0];
        return static_cast<int>( math::RoundUp(d) );
    }

    /// @return Returns a new density grid that is the result of passive advection
    ///         of all the active values the input density for the
    ///         time = dt * IntegrationCount.
    ///
    /// @param inGrid  The input density grid to be advected (unmodified)
    /// @param dt      Time-step of the Runge-Kutta integrator.
    ///
    /// @note It is important to note that the input density @c grid1 is assumed
    ///       to be represented in the same frame of reference as the velocity field!
    ///
    /// @details This method will advect all of the active values in
    ///          the input density @a inGrid. To achieve this a
    ///          deep-copy is dilated to account for the material
    ///          transport. This dilation step can be slow for large
    ///          time steps @a dt or fast moving velocity fields.
    ///
    /// @throw RuntimeError if @a inGrid does not have uniform voxels.
    template<typename DensityGridT,
             typename DensitySamplerT>
    typename DensityGridT::Ptr advect(const DensityGridT& inGrid, double dt)
    {
        typename DensityGridT::Ptr outGrid = inGrid.deepCopy();
        dilateVoxels( outGrid->tree(), this->getMaxDistance(inGrid, dt) );
        this->template process<DensityGridT, DensitySamplerT>(*outGrid, inGrid, dt);
        return outGrid;
    }

    /// @return Returns a new density grid that is the result of
    ///         passive advection of the active values in @a inGrid
    ///         that intersect the active values in @c mask. The time
    ///         of the output grid corresponds to dt * IntegrationCount.
    ///
    /// @param inGrid    The input density grid to be advected (unmodified).
    /// @param mask      The mask of active values defining the active voxels
    ///                  in @c inGrid on which to perform advection. Only
    ///                  if a value is active in both grids will it be modified.
    /// @param dt        Time-step for a single Runge-Kutta integration step.
    ///
    /// @note It is important to note that the input density @a inGrid is assumed
    ///       to be represented in the same frame of reference as the
    ///       input @a mask grid!
    ///
    /// @details This method will advect all of the active values in
    ///          the input density @a inGrid that intersects with the
    ///          active values in @a mask. To achieve this a
    ///          deep-copy is dilated to account for the material
    ///          transport and finally cropped to the intersection
    ///          with @a mask. The dilation step can be slow for large
    ///          time steps @a dt or fast moving velocity fields.
    ///
    /// @throw RuntimeError if @a inGrid is not aligned with @a mask
    ///        or if its voxels are not uniform.
    template<typename DensityGridT,
             typename MaskGridT,
             typename DensitySamplerT>
    typename DensityGridT::Ptr advect(const DensityGridT& inGrid, const MaskGridT& mask, double dt)
    {
        if (inGrid.transform() != mask.transform()) {
            OPENVDB_THROW(RuntimeError, "Density grid and mask grid are misaligned! Consider "
                          "resampling either of the two grids into the index space of the other.");
        }
        typename DensityGridT::Ptr outGrid = inGrid.deepCopy();
        dilateVoxels( outGrid->tree(), this->getMaxDistance(inGrid, dt) );
        outGrid->topologyIntersection(mask);
        pruneInactive(outGrid->tree(), mGrainSize>0, mGrainSize);
        this->template process<DensityGridT, DensitySamplerT>(*outGrid, inGrid, dt);
        outGrid->topologyUnion(inGrid);
        return outGrid;
    }

private:
    // disallow copy construction and copy by assignment!
    DensityAdvection(const DensityAdvection&);// not implemented
    DensityAdvection& operator=(const DensityAdvection&);// not implemented

    template<typename DensityGridT, typename DensitySamplerT>
    void process(DensityGridT& outGrid, const DensityGridT& inGrid, double dt)
    {
        mDt = dt/mCountRK;
        outGrid.tree().voxelizeActiveTiles();
        if (mOrderRK == 1) {
            Advect<DensityGridT, 1, DensitySamplerT> adv(inGrid, *this);
            adv.run(outGrid);
        } else if (mOrderRK == 2) {
            Advect<DensityGridT, 2, DensitySamplerT> adv(inGrid, *this);
            adv.run(outGrid);
        } else if (mOrderRK == 3) {
            Advect<DensityGridT, 3, DensitySamplerT> adv(inGrid, *this);
            adv.run(outGrid);
        } else if (mOrderRK == 4) {
            Advect<DensityGridT, 4, DensitySamplerT> adv(inGrid, *this);
            adv.run(outGrid);
        }
        pruneInactive(outGrid.tree(), mGrainSize>0, mGrainSize);
    }

    // Private class that implements the semi-Lagrangian integration
    template<typename DensityGridT, size_t OrderRK, typename SamplerT>
    struct Advect
    {
        typedef typename DensityGridT::TreeType      TreeT;
        typedef typename DensityGridT::ConstAccessor AccT;
        typedef typename TreeT::ValueType            ValueT;
        typedef typename tree::LeafManager<TreeT>    LeafManagerT;
        typedef typename LeafManagerT::LeafRange     LeafRangeT;
        typedef VelocityIntegrator<VelocityGridT, StaggeredVelocity> VelocityIntegratorT;
        typedef typename VelocityIntegratorT::ElementType RealT;

        Advect(const DensityGridT& inGrid, const DensityAdvection& parent)
            : mTransform(&inGrid.transform())
            , mDensityAcc(inGrid.getAccessor())
            , mVelocityInt(parent.mVelGrid)
            , mParent(&parent)
        {
        }
        Advect(const Advect& other)
            : mTransform(other.mTransform)
            , mDensityAcc(other.mDensityAcc.tree())
            , mVelocityInt(other.mVelocityInt)
            , mParent(other.mParent)
        {
        }
        void run(DensityGridT& outGrid)
        {
            if (mParent->mInterrupter) mParent->mInterrupter->start("Advecting density");

            const LeafManagerT manger(outGrid.tree());
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
            typedef typename VelocityIntegratorT::ElementType RealT;
            const int n = static_cast<int>(mParent->mCountRK);
            const RealT dt = static_cast<RealT>(- mParent->mDt);//back-tracking
            const ValueT backg = mDensityAcc.tree().background();
            for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
                ValueT* phi = leafIter.buffer(0).data();
                typedef typename TreeT::LeafNodeType::ValueOnIter VoxelIterT;
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    Vec3d w = mTransform->indexToWorld(voxelIter.getCoord());
                    for (int i = 0; i < n; ++i) {
                        mVelocityInt.template rungeKutta<OrderRK, Vec3d>(dt, w);
                    }
                    const ValueT sample = SamplerT::sample(mDensityAcc, mTransform->worldToIndex(w));
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
        const math::Transform*  mTransform;
        AccT                    mDensityAcc;
        VelocityIntegratorT     mVelocityInt;
        const DensityAdvection* mParent;
    };// end of private Advect class

    // Protected member data of DensityAdvection
    const VelocityGridT& mVelGrid;
    double               mMaxVelocity;
    InterrupterType*     mInterrupter;
    double               mDt;// time step per RK integration step
    size_t               mCountRK;// number of RK integration sub-steps
    size_t               mOrderRK;// order of the RK integrator
    size_t               mGrainSize;// for multi-threading (0 means no threading)
};//end of DensityAdvection class

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_DENSITY_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
