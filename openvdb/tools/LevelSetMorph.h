///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
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
/// @file LevelSetMorph.h
///
/// @brief Shape morphology of level sets. Morphing from a source
/// narrow-band level sets to a target narrow-band level set.

#ifndef OPENVDB_TOOLS_LEVEL_SET_MORPH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_MORPH_HAS_BEEN_INCLUDED

#include "LevelSetTracker.h"
#include "Interpolation.h" // for BoxSampler, etc.
#include <openvdb/math/FiniteDifference.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Shape morphology of level sets. Morphing from a source
/// narrow-band level sets to a target narrow-band level set.
///
/// @details
/// The @c InterruptType template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = NULL)// called when computations begin
///   void end()                         // called when computations end
///   bool wasInterrupted(int percent=-1)// return true to break computation
/// };
/// @endcode
///
/// @note If no template argument is provided for this InterruptType,
/// the util::NullInterrupter is used, which implies that all interrupter
/// calls are no-ops (i.e., they incur no computational overhead).
template<typename GridT,
         typename InterruptT = util::NullInterrupter>
class LevelSetMorphing
{
public:
    typedef GridT                              GridType;
    typedef typename GridT::TreeType           TreeType;
    typedef LevelSetTracker<GridT, InterruptT> TrackerT;
    typedef typename TrackerT::LeafRange       LeafRange;
    typedef typename TrackerT::LeafType        LeafType;
    typedef typename TrackerT::BufferType      BufferType;
    typedef typename TrackerT::ValueType       ScalarType;

    /// Main constructor
    LevelSetMorphing(GridT& sourceGrid, const GridT& targetGrid, InterruptT* interrupt = NULL):
        mTracker(sourceGrid, interrupt), mTarget(&targetGrid),
        mSpatialScheme(math::HJWENO5_BIAS),
        mTemporalScheme(math::TVD_RK2) {}

    virtual ~LevelSetMorphing() {};

    /// Redefine the target level set
    void setTarget(const GridT& targetGrid) { mTarget = &targetGrid; }
    
    /// Return the spatial finite-difference scheme
    math::BiasedGradientScheme getSpatialScheme() const { return mSpatialScheme; }
    /// Set the spatial finite-difference scheme
    void setSpatialScheme(math::BiasedGradientScheme scheme) { mSpatialScheme = scheme; }

    /// Return the temporal integration scheme
    math::TemporalIntegrationScheme getTemporalScheme() const { return mTemporalScheme; }
    /// Set the temporal integration scheme
    void setTemporalScheme(math::TemporalIntegrationScheme scheme) { mTemporalScheme = scheme; }

    /// Return the spatial finite-difference scheme
    math::BiasedGradientScheme getTrackerSpatialScheme() const
    {
        return mTracker.getSpatialScheme();
    }
    /// Set the spatial finite-difference scheme
    void setTrackerSpatialScheme(math::BiasedGradientScheme scheme)
    {
        mTracker.setSpatialScheme(scheme);
    }
    /// Return the temporal integration scheme
    math::TemporalIntegrationScheme getTrackerTemporalScheme() const
    {
        return mTracker.getTemporalScheme();
    }
    /// Set the temporal integration scheme
    void setTrackerTemporalScheme(math::TemporalIntegrationScheme scheme)
    {
        mTracker.setTemporalScheme(scheme);
    }
    /// Return the number of normalizations performed per track or normalize call.
    int  getNormCount() const { return mTracker.getNormCount(); }
    /// Set the number of normalizations performed per track or normalize call.
    void setNormCount(int n) { mTracker.setNormCount(n); }

    /// Return the grain size used for multithreading
    int  getGrainSize() const { return mTracker.getGrainSize(); }
    /// @brief Set the grain size used for multithreading.
    /// @note A grain size of 0 or less disables multithreading!
    void setGrainSize(int grainsize) { mTracker.setGrainSize(grainsize); }

    /// @brief Advect the level set from its current time, @a time0, to its
    /// final time, @a time1. If @a time0 > @a time1, perform backward advection.
    ///
    /// @return the number of CFL iterations used to advect from @a time0 to @a time1
    size_t advect(ScalarType time0, ScalarType time1);

private:

    // This templated private class implements all the level set magic.
    template<typename MapT, math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    class LevelSetMorph
    {
    public:
        /// Main constructor
        LevelSetMorph(TrackerT& tracker, const GridT* target);
        /// Shallow copy constructor called by tbb::parallel_for() threads
        LevelSetMorph(const LevelSetMorph& other);
        /// Shallow copy constructor called by tbb::parallel_reduce() threads
        LevelSetMorph(LevelSetMorph& other, tbb::split);
        /// destructor
        virtual ~LevelSetMorph() {}
        /// Advect the level set from it's current time, time0, to it's final time, time1.
        /// @return number of CFL iterations
        size_t advect(ScalarType time0, ScalarType time1);
        /// Used internally by tbb::parallel_for()
        void operator()(const LeafRange& r) const
        {
            if (mTask) mTask(const_cast<LevelSetMorph*>(this), r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// Used internally by tbb::parallel_reduce()
        void operator()(const LeafRange& r)
        {
            if (mTask) mTask(this, r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// This is only called by tbb::parallel_reduce() threads
        void join(const LevelSetMorph& other) { mMaxAbsS = math::Max(mMaxAbsS, other.mMaxAbsS); }
    private:
        typedef typename boost::function<void (LevelSetMorph*, const LeafRange&)> FuncType;
        TrackerT*    mTracker;
        const GridT* mTarget;
        ScalarType   mMinAbsS, mMaxAbsS;
        const MapT*  mMap;
        FuncType     mTask;

        /// Enum to define the type of multithreading
        enum ThreadingMode { PARALLEL_FOR, PARALLEL_REDUCE }; // for internal use
        // method calling tbb
        void cook(ThreadingMode mode, size_t swapBuffer = 0);

        /// Sample field and return the CFT time step
        typename GridT::ValueType sampleSpeed(
            ScalarType time0, ScalarType time1, Index speedBuffer);
        void sampleXformedSpeed(const LeafRange& r, Index speedBuffer);
        void sampleAlignedSpeed(const LeafRange& r, Index speedBuffer);

        // Forward Euler advection steps: Phi(result) = Phi(0) - dt * Speed(speed)*|Grad[Phi(0)]|;
        void euler1(const LeafRange& r, ScalarType dt, Index resultBuffer, Index speedBuffer);

        // Convex combination of Phi and a forward Euler advection steps:
        // Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * Speed(speed)*|Grad[Phi(0)]|);
        void euler2(const LeafRange& r, ScalarType dt, ScalarType alpha,
                    Index phiBuffer, Index resultBuffer, Index speedBuffer);

    }; // end of private LevelSetMorph class

    template<math::BiasedGradientScheme SpatialScheme>
    size_t advect1(ScalarType time0, ScalarType time1);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    size_t advect2(ScalarType time0, ScalarType time1);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme,
             typename MapType>
    size_t advect3(ScalarType time0, ScalarType time1);

    TrackerT                        mTracker;
    const GridT*                    mTarget;
    math::BiasedGradientScheme      mSpatialScheme;
    math::TemporalIntegrationScheme mTemporalScheme;

    // disallow copy by assignment
    void operator=(const LevelSetMorphing& other) {}

};//end of LevelSetMorphing

template<typename GridT, typename InterruptT>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect(ScalarType time0, ScalarType time1)
{
    switch (mSpatialScheme) {
    case math::FIRST_BIAS:
        return this->advect1<math::FIRST_BIAS  >(time0, time1);
        //case math::SECOND_BIAS:
        //return this->advect1<math::SECOND_BIAS >(time0, time1);
        //case math::THIRD_BIAS:
        //return this->advect1<math::THIRD_BIAS  >(time0, time1);
        //case math::WENO5_BIAS:
        //return this->advect1<math::WENO5_BIAS  >(time0, time1);
    case math::HJWENO5_BIAS:
        return this->advect1<math::HJWENO5_BIAS>(time0, time1);
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect1(ScalarType time0, ScalarType time1)
{
    switch (mTemporalScheme) {
    case math::TVD_RK1:
        return this->advect2<SpatialScheme, math::TVD_RK1>(time0, time1);
    case math::TVD_RK2:
        return this->advect2<SpatialScheme, math::TVD_RK2>(time0, time1);
    case math::TVD_RK3:
        return this->advect2<SpatialScheme, math::TVD_RK3>(time0, time1);
    default:
        OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect2(ScalarType time0, ScalarType time1)
{
    const math::Transform& trans = mTracker.grid().transform();
    if (trans.mapType() == math::UniformScaleMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UniformScaleMap>(time0, time1);
    } else if (trans.mapType() == math::UniformScaleTranslateMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UniformScaleTranslateMap>(
            time0, time1);
    } else if (trans.mapType() == math::UnitaryMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UnitaryMap    >(time0, time1);
    } else if (trans.mapType() == math::TranslationMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::TranslationMap>(time0, time1);
    } else {
        OPENVDB_THROW(ValueError, "MapType not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MapT>
inline size_t
LevelSetMorphing<GridT, InterruptT>::advect3(ScalarType time0, ScalarType time1)
{
    LevelSetMorph<MapT, SpatialScheme, TemporalScheme> tmp(mTracker, mTarget);
    return tmp.advect(time0, time1);
}


///////////////////////////////////////////////////////////////////////


template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
LevelSetMorph(TrackerT& tracker, const GridT* target):
    mTracker(&tracker),
    mTarget(target),
    mMinAbsS(1e-6),
    mMap(tracker.grid().transform().template constMap<MapT>().get()),
    mTask(0)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
LevelSetMorph(const LevelSetMorph& other):
    mTracker(other.mTracker),
    mTarget(other.mTarget),
    mMinAbsS(other.mMinAbsS),
    mMaxAbsS(other.mMaxAbsS),
    mMap(other.mMap),
    mTask(other.mTask)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
LevelSetMorph(LevelSetMorph& other, tbb::split):
    mTracker(other.mTracker),
    mTarget(other.mTarget),
    mMinAbsS(other.mMinAbsS),
    mMaxAbsS(other.mMaxAbsS),
    mMap(other.mMap),
    mTask(other.mTask)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
advect(ScalarType time0, ScalarType time1)
{
    // Make sure we have enough temporal auxiliary buffers for the time
    // integration AS WELL AS an extra buffer with the speed function!
    static const Index auxBuffers = 1 + (TemporalScheme == math::TVD_RK3 ? 2 : 1);
    size_t countCFL = 0;
    while (time0 < time1 && mTracker->checkInterrupter()) {
        mTracker->leafs().rebuildAuxBuffers(auxBuffers);

        const ScalarType dt = this->sampleSpeed(time0, time1, auxBuffers);
        if ( math::isZero(dt) ) break;//V is essentially zero so terminate

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN //switch is resolved at compile-time
        switch(TemporalScheme) {
        case math::TVD_RK1:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * Speed(2) * |Grad[Phi(0)]|
            mTask = boost::bind(&LevelSetMorph::euler1, _1, _2, dt, /*result=*/1, /*speed*/2);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK2:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * Speed(2) * |Grad[Phi(0)]|
            mTask = boost::bind(&LevelSetMorph::euler1, _1, _2, dt, /*result=*/1, /*speed*/2);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * Speed(2) * |Grad[Phi(0)]|)
            mTask = boost::bind(&LevelSetMorph::euler2, _1, _2, dt, ScalarType(0.5),
                                /*phi=*/1, /*result=*/1, /*speed*/2);
            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK3:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * Speed(3) * |Grad[Phi(0)]|
            mTask = boost::bind(&LevelSetMorph::euler1, _1, _2, dt, /*result=*/1, /*speed*/3);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt/2
            // Phi_t2(2) = 3/4 * Phi_t0(1) + 1/4 * (Phi_t1(0) - dt * Speed(3) * |Grad[Phi(0)]|)
            mTask = boost::bind(&LevelSetMorph::euler2, _1, _2, dt, ScalarType(0.75),
                                /*phi=*/1, /*result=*/2, /*speed*/3);
            // Cook and swap buffer 0 and 2 such that Phi_t2(0) and Phi_t1(2)
            this->cook(PARALLEL_FOR, 2);

            // Convex combine explict Euler step: t3 = t0 + dt
            // Phi_t3(2) = 1/3 * Phi_t0(1) + 2/3 * (Phi_t2(0) - dt * Speed(3) * |Grad[Phi(0)]|)
            mTask = boost::bind(&LevelSetMorph::euler2, _1, _2, dt, ScalarType(1.0/3.0),
                                /*phi=*/1, /*result=*/2, /*speed*/3);
            // Cook and swap buffer 0 and 2 such that Phi_t3(0) and Phi_t2(2)
            this->cook(PARALLEL_FOR, 2);
            break;
        default:
            OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
        }//end of compile-time resolved switch
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END

        time0 += dt;
        ++countCFL;
        mTracker->leafs().removeAuxBuffers();

        // Track the narrow band
        mTracker->track();
    }//end wile-loop over time

    return countCFL;//number of CLF propagation steps
}

template<typename GridT, typename InterruptT>
template<typename MapT, math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline typename GridT::ValueType
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
sampleSpeed(ScalarType time0, ScalarType time1, Index speedBuffer)
{
    mMaxAbsS = mMinAbsS;
    const size_t leafCount = mTracker->leafs().leafCount();
    if (leafCount==0 || time0 >= time1) return ScalarType(0);

    if (mTarget->transform() == mTracker->grid().transform()) {
        mTask = boost::bind(&LevelSetMorph::sampleAlignedSpeed, _1, _2, speedBuffer);
    } else {
        mTask = boost::bind(&LevelSetMorph::sampleXformedSpeed, _1, _2, speedBuffer);
    }
    this->cook(PARALLEL_REDUCE);
    if (math::isApproxEqual(mMinAbsS, mMaxAbsS)) return ScalarType(0);//speed is essentially zero
    static const ScalarType CFL = (TemporalScheme == math::TVD_RK1 ? ScalarType(0.3) :
                                   TemporalScheme == math::TVD_RK2 ? ScalarType(0.9) :
                                   ScalarType(1.0))/math::Sqrt(ScalarType(3.0));
    const ScalarType dt = math::Abs(time1 - time0), dx = mTracker->voxelSize();
    return math::Min(dt, ScalarType(CFL*dx/mMaxAbsS));
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
sampleXformedSpeed(const LeafRange& range, Index speedBuffer)
{
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    typedef tools::GridSampler<typename GridT::ConstAccessor, tools::BoxSampler> SamplerT;
    const MapT& map = *mMap;
    mTracker->checkInterrupter();

    SamplerT sampler(mTarget->getAccessor(),  mTarget->transform());
    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        BufferType& speed = leafIter.buffer(speedBuffer);
        for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            ScalarType& s = const_cast<ScalarType&>(speed.getValue(voxelIter.pos()));
            s -= sampler.wsSample(map.applyMap(voxelIter.getCoord().asVec3d()));
            mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
        }
    }
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
sampleAlignedSpeed(const LeafRange& range, Index speedBuffer)
{
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    mTracker->checkInterrupter();

    typename GridT::ConstAccessor target = mTarget->getAccessor();
    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        BufferType& speed = leafIter.buffer(speedBuffer);
        for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            ScalarType& s = const_cast<ScalarType&>(speed.getValue(voxelIter.pos()));
            s -= target.getValue(voxelIter.getCoord());
            mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
        }
    }
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
cook(ThreadingMode mode, size_t swapBuffer)
{
    mTracker->startInterrupter("Morphing level set");

    const int grainSize   = mTracker->getGrainSize();
    const LeafRange range = mTracker->leafs().leafRange(grainSize);

    if (mTracker->getGrainSize()==0) {
        (*this)(range);
    } else if (mode == PARALLEL_FOR) {
        tbb::parallel_for(range, *this);
    } else if (mode == PARALLEL_REDUCE) {
        tbb::parallel_reduce(range, *this);
    } else {
        throw std::runtime_error("Undefined threading mode");
    }

    mTracker->leafs().swapLeafBuffer(swapBuffer, grainSize == 0);

    mTracker->endInterrupter();
}

// Forward Euler advection steps:
// Phi(result) = Phi(0) - dt * Phi(speed) * |Grad[Phi(0)]|
template<typename GridT, typename InterruptT>
template <typename MapT,
          math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
euler1(const LeafRange& range, ScalarType dt, Index resultBuffer, Index speedBuffer)
{
    typedef math::BIAS_SCHEME<SpatialScheme>                             Scheme;
    typedef typename Scheme::template ISStencil<GridType>::StencilType   Stencil;
    typedef typename LeafType::ValueOnCIter VoxelIterT;

    mTracker->checkInterrupter();
    const MapT& map = *mMap;
    Stencil stencil(mTracker->grid());

    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        BufferType& speed  = leafIter.buffer(speedBuffer);
        BufferType& result = leafIter.buffer(resultBuffer);
        for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const Index n = voxelIter.pos();
            stencil.moveTo(voxelIter);
            const ScalarType G = math::GradientNormSqrd<MapT,SpatialScheme>::result(map, stencil);
            result.setValue(n, *voxelIter - dt * speed.getValue(n) * G);
        }
    }
}

// Convex combination of Phi and a forward Euler advection steps:
// Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * Phi(speed) * |Grad[Phi(0)]|)
template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetMorphing<GridT, InterruptT>::
LevelSetMorph<MapT, SpatialScheme, TemporalScheme>::
euler2(const LeafRange& range, ScalarType dt, ScalarType alpha,
       Index phiBuffer, Index resultBuffer, Index speedBuffer)
{
    typedef math::BIAS_SCHEME<SpatialScheme>                             Scheme;
    typedef typename Scheme::template ISStencil<GridType>::StencilType   Stencil;
    typedef typename LeafType::ValueOnCIter VoxelIterT;

    mTracker->checkInterrupter();
    const MapT& map = *mMap;
    const ScalarType beta = ScalarType(1.0) - alpha;
    Stencil stencil(mTracker->grid());

    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        BufferType& speed  = leafIter.buffer(speedBuffer);
        BufferType& result = leafIter.buffer(resultBuffer);
        BufferType& phi    = leafIter.buffer(phiBuffer);
        for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const Index n = voxelIter.pos();
            stencil.moveTo(voxelIter);
            const ScalarType G = math::GradientNormSqrd<MapT,SpatialScheme>::result(map, stencil);
            result.setValue(n,
                alpha * phi.getValue(n) + beta * (*voxelIter - dt * speed.getValue(n) * G));
        }
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MORPH_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
