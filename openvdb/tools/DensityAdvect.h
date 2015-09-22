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
/// @todo  Fix potential bug in MacCormack (missing flux-limitor!?)
///        Improve (i.e. limit) padding by topology dilation.

#ifndef OPENVDB_TOOLS_DENSITY_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_DENSITY_ADVECT_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
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

namespace Scheme { enum SemiLagrangian { SEMI, MID, RK3, RK4, MAC, BFECC }; }
    
/// @brief Performs advections of an arbitrary types of volumes in a
///        static velocity field. The advections is performed by means
///        of various derivatives of Semi-Lagrangian integration, i.e.
///        backwards tracking by of the hyperbolic characteristics.     
///
/// @details The supported integrations schemes:
///    =================================================================
///    |  Lable | Accuracy |  Integration Scheme    |  Interpolations  |
///    |        |Time/Space|                        | velocity/density |
///    =================================================================    
///    |  SEMI  |   1/1    | Semi-Lagrangian        |        1/1       | 
///    |  MID   |   2/1    | Mid-Point              |        2/1       |
///    |  RK3   |   3/1    | 3'rd Order Runge-Kutta |        3/1       |
///    |  RK4   |   4/1    | 4'th Order Runge-Kutta |        4/1       |
///    |  MAC   |   2/2    | MacCormack             |        2/2       |
///    |  BFECC |   2/2    | BFECC                  |        3/2       |           
///    =================================================================
template<typename VelocityGridT = Vec3fGrid,
         bool StaggeredVelocity = false,
         typename InterrupterType = util::NullInterrupter>
class DensityAdvection
{
public:
    
    /// @brief Constructor
    ///
    /// @param velGrid     Velocity grid responsible for the (passive) advection.
    /// @param interrupter Optional interrupter used to prematurely end computations.
    ///
    /// @note The velocity field is assumed to be constant for the duration of the
    ///       advection.
    DensityAdvection(const VelocityGridT& velGrid, InterrupterType* interrupter = NULL)
        : mVelGrid(velGrid)
        , mInterrupter(interrupter)
        , mIntegrator( Scheme::SEMI )
        , mGrainSize(1)
    {
        math::Extrema e = extrema(velGrid.cbeginValueAll(), /*threading*/true);
        e.add(velGrid.background().length());
        mMaxVelocity = e.max();
    }

    virtual ~DensityAdvection()
    {
    }

    /// @brief Set the integrator (see details in the table above)
    void setIntegrator(Scheme::SemiLagrangian integrator) { mIntegrator = integrator; }

    /// @brief Retrun the integrator (see details in the table above)
    Scheme::SemiLagrangian getIntegrator() const { return mIntegrator; }

     /// @return the grain-size used for multi-threading
    size_t getGrainSize() const { return mGrainSize; }

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
    /// @warning If the DensitySamplerT is of higher order the one
    ///          (i.e. tri-linear interpolation) instabilities are
    ///          known to occure. To suppress those monotonicity
    ///          constrains or flux-limiters need to be applies.
    ///
    /// @throw RuntimeError if @a inGrid does not have uniform voxels.
    template<typename DensityGridT,
             typename DensitySamplerT>//only C++11 allows for a default argument
    typename DensityGridT::Ptr advect(const DensityGridT& inGrid, double dt)
    {
        typename DensityGridT::Ptr outGrid = inGrid.deepCopy();
        dilateVoxels( outGrid->tree(), this->getMaxDistance(inGrid, dt) );
        this->template cook<DensityGridT, DensitySamplerT>(*outGrid, inGrid, dt);
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
    /// @warning If the DensitySamplerT is of higher order the one
    ///          (i.e. tri-linear interpolation) instabilities are
    ///          known to occure. To suppress those monotonicity
    ///          constrains or flux-limiters need to be applies.
    ///
    /// @throw RuntimeError if @a inGrid is not aligned with @a mask
    ///        or if its voxels are not uniform.
    template<typename DensityGridT,
             typename MaskGridT,
             typename DensitySamplerT>//only C++11 allows for a default argument
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
        this->template cook<DensityGridT, DensitySamplerT>(*outGrid, inGrid, dt);
        outGrid->topologyUnion(inGrid);
        return outGrid;
    }

private:
    // disallow copy construction and copy by assignment!
    DensityAdvection(const DensityAdvection&);// not implemented
    DensityAdvection& operator=(const DensityAdvection&);// not implemented

    void start(const char* str) const
    {
        if (mInterrupter) mInterrupter->start(str);
    }
    void stop() const
    {
        if (mInterrupter) mInterrupter->end();
    }      
    bool interrupt() const
    {
        if (mInterrupter && util::wasInterrupted(mInterrupter)) {
            tbb::task::self().cancel_group_execution();
            return true;
        }
        return false;
    }
    
    template<typename DensityGridT, typename DensitySamplerT>
    void cook(DensityGridT& outGrid, const DensityGridT& inGrid, double dt)
    {
        outGrid.tree().voxelizeActiveTiles();
        switch (mIntegrator) {
        case Scheme::SEMI: {
            Advect<DensityGridT, 1, DensitySamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::MID: {
            Advect<DensityGridT, 2, DensitySamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::RK3: {
            Advect<DensityGridT, 3, DensitySamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::RK4: {
            Advect<DensityGridT, 4, DensitySamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::BFECC: {
            Advect<DensityGridT, 1, DensitySamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        }
        case Scheme::MAC: {
            Advect<DensityGridT, 1, DensitySamplerT> adv(inGrid, *this);
            adv.cook(outGrid, dt);
            break;
        } 
        default:
            OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
        }
        pruneInactive(outGrid.tree(), mGrainSize>0, mGrainSize);
    }

    // Private class that implements the multi-threaded advection
    template<typename DensityGridT, size_t OrderRK, typename SamplerT> struct Advect;

    // Private member data of DensityAdvection
    const VelocityGridT&   mVelGrid;
    double                 mMaxVelocity;
    InterrupterType*       mInterrupter;
    Scheme::SemiLagrangian mIntegrator;
    size_t                 mGrainSize;
};//end of DensityAdvection class
    
// Private class that implements the multi-threaded advection
template<typename VelocityGridT, bool StaggeredVelocity, typename InterrupterType>
template<typename DensityGridT, size_t OrderRK, typename SamplerT>
struct DensityAdvection<VelocityGridT, StaggeredVelocity, InterrupterType>::Advect
{
    typedef typename DensityGridT::TreeType      TreeT;
    typedef typename TreeT::ValueType            ValueT;
    typedef typename tree::LeafManager<TreeT>    LeafManagerT;
    typedef typename LeafManagerT::LeafNodeType  LeafNodeT;
    typedef typename LeafManagerT::LeafRange     LeafRangeT;
    typedef VelocityIntegrator<VelocityGridT, StaggeredVelocity> VelocityIntegratorT;
    typedef typename VelocityIntegratorT::ElementType RealT;
    typedef typename TreeT::LeafNodeType::ValueOnIter VoxelIterT;
    
    
    Advect(const DensityGridT& inGrid, const DensityAdvection& parent)
        : mTask(0)
        , mInGrid(&inGrid)
        , mVelocityInt(parent.mVelGrid)
        , mParent(&parent)
    {
    }
    inline void cook(const LeafRangeT& range)
    {
        if (mParent->mGrainSize > 0) {
            tbb::parallel_for(range, *this);
        } else {
            (*this)(range);
        }
    }
    void operator()(const LeafRangeT& range) const
    {
        assert(mTask);
        mTask(const_cast<Advect*>(this), range);
    }
    void cook(DensityGridT& outGrid, double time_step)
    {
        mParent->start("Advecting density");
        const int auxBufferCount = (mParent->mIntegrator == Scheme::MAC ||
                                    mParent->mIntegrator == Scheme::BFECC) ? 1 : 0;
        LeafManagerT manager(outGrid.tree(), auxBufferCount);
        const LeafRangeT range = manager.leafRange(mParent->mGrainSize);
        
        const RealT dt = static_cast<RealT>(-time_step);//method of characteristics backtracks
        if (mParent->mIntegrator == Scheme::MAC) {
            mTask = boost::bind(&Advect::rk,  _1, _2, dt, 0, *mInGrid, false);//out[0]=forward 
            this->cook(range);
            mTask = boost::bind(&Advect::rk,  _1, _2,-dt, 1,  outGrid, false);//out[1]=backward
            this->cook(range);
            mTask = boost::bind(&Advect::mac, _1, _2);//out[0] = out[0] + (in[0] - out[1])/2
            this->cook(range);
        } else if (mParent->mIntegrator == Scheme::BFECC) {
            mTask = boost::bind(&Advect::rk, _1, _2, dt, 0, *mInGrid, false);//out[0]=forward
            this->cook(range);
            mTask = boost::bind(&Advect::rk, _1, _2,-dt, 1,  outGrid, false);//out[1]=backward
            this->cook(range);
            mTask = boost::bind(&Advect::bfecc, _1, _2);//out[0] = (3*in[0] - out[1])/2
            this->cook(range);
            mTask = boost::bind(&Advect::rk, _1, _2, dt, 1,  outGrid, true);//out[1]=forward
            this->cook(range);
            manager.swapLeafBuffer(1);// out[0] = out[1]
        } else {// SEMI, MID, RK3 and RK4
            mTask = boost::bind(&Advect::rk, _1, _2,  dt, 0, *mInGrid, true);//forward
            this->cook(range);
        }
        
        manager.removeAuxBuffers();
        
        mParent->stop();
    }
    // Last step of the MacCormack scheme
    void mac(const LeafRangeT& range) const
    {
        if (mParent->interrupt()) return;
        typename DensityGridT::ConstAccessor acc = mInGrid->getAccessor();
        const ValueT backg = mInGrid->background();
        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueT* out0 = leafIter.buffer( 0 ).data();// forward
            const ValueT* out1 = leafIter.buffer( 1 ).data();// backward
            const LeafNodeT* leaf = acc.probeConstLeaf( leafIter->origin() );
            // const ValueT* in0 = leaf != NULL ? leaf->buffer().data() : NULL;
            // for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
            //     const Index i = voxelIter.pos();
            //     const ValueT phi = in0 != NULL ? in0[i] : acc.getValue(voxelIter.getCoord());
            //     out0[i] += RealT(0.5) * ( phi - out1[i] );
            //     if (math::isApproxEqual(out0[i], backg, math::Delta<ValueT>::value())) {
            //         out0[i] = backg;
            //         leafIter->setValueOff(i);
            //     }
            // }//loop over active voxels
            if (leaf !=NULL) {
                const ValueT* in0 = leaf->buffer().data();
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] += RealT(0.5) * ( in0[i] - out1[i] );
                    if (math::isApproxEqual(out0[i], backg, math::Delta<ValueT>::value())) {
                        out0[i] = backg;
                        leafIter->setValueOff(i);
                    }
                }
            } else {
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] += RealT(0.5) * ( acc.getValue(voxelIter.getCoord()) - out1[i] );
                    if (math::isApproxEqual(out0[i], backg, math::Delta<ValueT>::value())) {
                        out0[i] = backg;
                        leafIter->setValueOff(i);
                    }
                }//loop over active voxels
            }
        }//loop over leaf nodes
    }
    // Intermediate step in the BFECC scheme
    void bfecc(const LeafRangeT& range) const
    {
        if (mParent->interrupt()) return;
        typename DensityGridT::ConstAccessor acc = mInGrid->getAccessor();
        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueT* out0 = leafIter.buffer( 0 ).data();// forward
            const ValueT* out1 = leafIter.buffer( 1 ).data();// backward
            const LeafNodeT* leaf = acc.probeConstLeaf(leafIter->origin());
            if (leaf !=NULL) {
                const ValueT* in0 = leaf->buffer().data();
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] = RealT(0.5)*( RealT(3)*in0[i] - out1[i] );
                }//loop over active voxels
            } else {
                for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                    const Index i = voxelIter.pos();
                    out0[i] = RealT(0.5)*( RealT(3)*acc.getValue(voxelIter.getCoord()) - out1[i] );
                }//loop over active voxels
            }
        }//loop over leaf nodes
    }
    // Semi-Lagrangian integration with Runge-Kutta of various orders (1->4)
    void rk(const LeafRangeT& range, RealT dt, size_t n, const DensityGridT& grid, bool trim) const
    {
        if (mParent->interrupt()) return;
        const math::Transform& xform = mInGrid->transform();
        typename DensityGridT::ConstAccessor acc = grid.getAccessor();
        const ValueT backg = mInGrid->background();
        for (typename LeafRangeT::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
            ValueT* phi = leafIter.buffer( n ).data();
            for (VoxelIterT voxelIter = leafIter->beginValueOn(); voxelIter; ++voxelIter) {
                ValueT& value = phi[voxelIter.pos()];
                Vec3d wPos = xform.indexToWorld(voxelIter.getCoord());
                mVelocityInt.template rungeKutta<OrderRK, Vec3d>(dt, wPos);
                value = SamplerT::sample(acc, xform.worldToIndex(wPos));
                if (trim && math::isApproxEqual(value, backg, math::Delta<ValueT>::value())) {
                    value = backg;
                    leafIter->setValueOff( voxelIter.pos() );
                }
            }//loop over active voxels
        }//loop over leaf nodes
    }
    // Public member data of the private Advect class
    typename boost::function<void (Advect*, const LeafRangeT&)> mTask;
    const DensityGridT*     mInGrid;
    VelocityIntegratorT     mVelocityInt;//very lightweight!
    const DensityAdvection* mParent;
};// end of private member class Advect    
    
} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_DENSITY_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
