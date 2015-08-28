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
/// @file LevelSetAdvect.h
///
/// @brief Hyperbolic advection of narrow-band level sets

#ifndef OPENVDB_TOOLS_LEVEL_SET_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_ADVECT_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <openvdb/Platform.h>
#include "LevelSetTracker.h"
#include "VelocityFields.h" // for EnrightField
#include <openvdb/math/FiniteDifference.h>
#include <boost/math/constants/constants.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief  Hyperbolic advection of narrow-band level sets in an
/// external velocity field
///
/// The @c FieldType template argument below refers to any functor
/// with the following interface (see tools/VelocityFields.h
/// for examples):
///
/// @code
/// class VelocityField {
///   ...
/// public:
///   openvdb::VectorType operator() (const openvdb::Coord& xyz, ValueType time) const;
///   ...
/// };
/// @endcode
///
/// @note The functor method returns the velocity field at coordinate
/// position xyz of the advection grid, and for the specified
/// time. Note that since the velocity is returned in the local
/// coordinate space of the grid that is being advected, the functor
/// typically depends on the transformation of that grid. This design
/// is chosen for performance reasons.
///
/// The @c InterruptType template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = NULL)// called when computations begin
///   void end()                         // called when computations end
///   bool wasInterrupted(int percent=-1)// return true to break computation
///};
/// @endcode
///
/// @note If no template argument is provided for this InterruptType
/// the util::NullInterrupter is used which implies that all
/// interrupter calls are no-ops (i.e. incurs no computational overhead).
///

template<typename GridT,
         typename FieldT     = EnrightField<typename GridT::ValueType>,
         typename InterruptT = util::NullInterrupter>
class LevelSetAdvection
{
public:
    typedef GridT                              GridType;
    typedef LevelSetTracker<GridT, InterruptT> TrackerT;
    typedef typename TrackerT::LeafRange       LeafRange;
    typedef typename TrackerT::LeafType        LeafType;
    typedef typename TrackerT::BufferType      BufferType;
    typedef typename TrackerT::ValueType       ValueType;
    typedef typename FieldT::VectorType        VectorType;

    /// Main constructor
    LevelSetAdvection(GridT& grid, const FieldT& field, InterruptT* interrupt = NULL):
        mTracker(grid, interrupt), mField(field),
        mSpatialScheme(math::HJWENO5_BIAS),
        mTemporalScheme(math::TVD_RK2) {}

    virtual ~LevelSetAdvection() {}

    /// @return the spatial finite difference scheme
    math::BiasedGradientScheme getSpatialScheme() const { return mSpatialScheme; }
    /// @brief Set the spatial finite difference scheme
    void setSpatialScheme(math::BiasedGradientScheme scheme) { mSpatialScheme = scheme; }

    /// @return the temporal integration scheme
    math::TemporalIntegrationScheme getTemporalScheme() const { return mTemporalScheme; }
    /// @brief Set the spatial finite difference scheme
    void setTemporalScheme(math::TemporalIntegrationScheme scheme) { mTemporalScheme = scheme; }

    /// @return the spatial finite difference scheme
    math::BiasedGradientScheme getTrackerSpatialScheme() const { return mTracker.getSpatialScheme(); }
    /// @brief Set the spatial finite difference scheme
    void setTrackerSpatialScheme(math::BiasedGradientScheme scheme) { mTracker.setSpatialScheme(scheme); }

    /// @return the temporal integration scheme
    math::TemporalIntegrationScheme getTrackerTemporalScheme() const { return mTracker.getTemporalScheme(); }
    /// @brief Set the spatial finite difference scheme
    void setTrackerTemporalScheme(math::TemporalIntegrationScheme scheme) { mTracker.setTemporalScheme(scheme); }

    /// @return The number of normalizations performed per track or
    /// normalize call.
    int  getNormCount() const { return mTracker.getNormCount(); }
    /// @brief Set the number of normalizations performed per track or
    /// normalize call.
    void setNormCount(int n) { mTracker.setNormCount(n); }

    /// @return the grain-size used for multi-threading
    int  getGrainSize() const { return mTracker.getGrainSize(); }
    /// @brief Set the grain-size used for multi-threading.
    /// @note A grain size of 0 or less disables multi-threading!
    void setGrainSize(int grainsize) { mTracker.setGrainSize(grainsize); }

    /// Advect the level set from its current time, time0, to its
    /// final time, time1. If time0>time1 backward advection is performed.
    ///
    /// @return number of CFL iterations used to advect from time0 to time1
    size_t advect(ValueType time0, ValueType time1);

private:

    // disallow copy construction and copy by assinment!
    LevelSetAdvection(const LevelSetAdvection&);// not implemented
    LevelSetAdvection& operator=(const LevelSetAdvection&);// not implemented

    // This templated private struct implements all the level set magic.
    template<typename MapT, math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    struct Advect
    {
        /// Main constructor
        Advect(LevelSetAdvection& parent);
        /// Shallow copy constructor called by tbb::parallel_for() threads
        Advect(const Advect& other);
        /// Shallow copy constructor called by tbb::parallel_reduce() threads
        Advect(Advect& other, tbb::split);
        /// destructor
        virtual ~Advect() { if (mIsMaster) this->clearField(); }
        /// Advect the level set from its current time, time0, to its final time, time1.
        /// @return number of CFL iterations
        size_t advect(ValueType time0, ValueType time1);
        /// Used internally by tbb::parallel_for()
        void operator()(const LeafRange& r) const
        {
            if (mTask) mTask(const_cast<Advect*>(this), r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// Used internally by tbb::parallel_reduce()
        void operator()(const LeafRange& r)
        {
            if (mTask) mTask(this, r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// This is only called by tbb::parallel_reduce() threads
        void join(const Advect& other) { mMaxAbsV = math::Max(mMaxAbsV, other.mMaxAbsV); }
        /// Enum to define multi-threading type
        enum ThreadingMode { PARALLEL_FOR, PARALLEL_REDUCE }; // for internal use
        // method calling tbb
        void cook(ThreadingMode mode, size_t swapBuffer = 0);
        /// Sample field and return the CFT time step
        typename GridT::ValueType sampleField(ValueType time0, ValueType time1);
        void  clearField();
        void  sampleXformedField(const LeafRange& r, ValueType time0, ValueType time1);
        void  sampleAlignedField(const LeafRange& r, ValueType time0, ValueType time1);

        // Convex combination of Phi and a forward Euler advection steps:
        // Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * Speed(speed)*|Grad[Phi(0)]|);
        template <int Nominator, int Denominator>
        void euler(const LeafRange&, ValueType, Index, Index);
        inline void euler01(const LeafRange& r, ValueType t) {this->euler<0,1>(r, t, 0, 1);}
        inline void euler12(const LeafRange& r, ValueType t) {this->euler<1,2>(r, t, 1, 1);}
        inline void euler34(const LeafRange& r, ValueType t) {this->euler<3,4>(r, t, 1, 2);}
        inline void euler13(const LeafRange& r, ValueType t) {this->euler<1,3>(r, t, 1, 2);}

        LevelSetAdvection& mParent;
        VectorType**       mVec;
        const ValueType    mMinAbsV;
        ValueType          mMaxAbsV;
        const MapT*        mMap;
        typename boost::function<void (Advect*, const LeafRange&)> mTask;
        const bool         mIsMaster;
    }; // end of private Advect struct

    template<math::BiasedGradientScheme SpatialScheme>
    size_t advect1(ValueType time0, ValueType time1);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    size_t advect2(ValueType time0, ValueType time1);

    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme,
             typename MapType>
    size_t advect3(ValueType time0, ValueType time1);

    TrackerT                        mTracker;
    //each thread needs a deep copy of the field since it might contain a ValueAccessor
    const FieldT                    mField;
    math::BiasedGradientScheme      mSpatialScheme;
    math::TemporalIntegrationScheme mTemporalScheme;

};//end of LevelSetAdvection

template<typename GridT, typename FieldT, typename InterruptT>
inline size_t
LevelSetAdvection<GridT, FieldT, InterruptT>::advect(ValueType time0, ValueType time1)
{
    switch (mSpatialScheme) {
    case math::FIRST_BIAS:
        return this->advect1<math::FIRST_BIAS  >(time0, time1);
    case math::SECOND_BIAS:
        return this->advect1<math::SECOND_BIAS >(time0, time1);
    case math::THIRD_BIAS:
        return this->advect1<math::THIRD_BIAS  >(time0, time1);
    case math::WENO5_BIAS:
        return this->advect1<math::WENO5_BIAS  >(time0, time1);
    case math::HJWENO5_BIAS:
        return this->advect1<math::HJWENO5_BIAS>(time0, time1);
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
    return 0;
}

template<typename GridT, typename FieldT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme>
inline size_t
LevelSetAdvection<GridT, FieldT, InterruptT>::advect1(ValueType time0, ValueType time1)
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

template<typename GridT, typename FieldT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetAdvection<GridT, FieldT, InterruptT>::advect2(ValueType time0, ValueType time1)
{
    const math::Transform& trans = mTracker.grid().transform();
    if (trans.mapType() == math::UniformScaleMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UniformScaleMap>(time0, time1);
    } else if (trans.mapType() == math::UniformScaleTranslateMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UniformScaleTranslateMap>(time0, time1);
    } else if (trans.mapType() == math::UnitaryMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::UnitaryMap    >(time0, time1);
    } else if (trans.mapType() == math::TranslationMap::mapType()) {
        return this->advect3<SpatialScheme, TemporalScheme, math::TranslationMap>(time0, time1);
    } else {
        OPENVDB_THROW(ValueError, "MapType not supported!");
    }
    return 0;
}

template<typename GridT, typename FieldT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme,
         typename MapT>
inline size_t
LevelSetAdvection<GridT, FieldT, InterruptT>::advect3(ValueType time0, ValueType time1)
{
    Advect<MapT, SpatialScheme, TemporalScheme> tmp(*this);
    return tmp.advect(time0, time1);
}


///////////////////////////////////////////////////////////////////////


template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
Advect(LevelSetAdvection& parent):
    mParent(parent),
    mVec(NULL),
    mMinAbsV(ValueType(1e-6)),
    mMap(parent.mTracker.grid().transform().template constMap<MapT>().get()),
    mTask(0),
    mIsMaster(true)
{
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
Advect(const Advect& other):
    mParent(other.mParent),
    mVec(other.mVec),
    mMinAbsV(other.mMinAbsV),
    mMaxAbsV(other.mMaxAbsV),
    mMap(other.mMap),
    mTask(other.mTask),
    mIsMaster(false)
{
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
Advect(Advect& other, tbb::split):
    mParent(other.mParent),
    mVec(other.mVec),
    mMinAbsV(other.mMinAbsV),
    mMaxAbsV(other.mMaxAbsV),
    mMap(other.mMap),
    mTask(other.mTask),
    mIsMaster(false)
{
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
advect(ValueType time0, ValueType time1)
{
    size_t countCFL = 0;
    if ( math::isZero(time0 - time1) ) return countCFL;
    const bool isForward = time0 < time1;
    while ((isForward ? time0<time1 : time0>time1) && mParent.mTracker.checkInterrupter()) {
        /// Make sure we have enough temporal auxiliary buffers
        mParent.mTracker.leafs().rebuildAuxBuffers(TemporalScheme == math::TVD_RK3 ? 2 : 1);

        const ValueType dt = this->sampleField(time0, time1);
        if ( math::isZero(dt) ) break;//V is essentially zero so terminate

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN //switch is resolved at compile-time
        switch(TemporalScheme) {
        case math::TVD_RK1:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(0)
            mTask = boost::bind(&Advect::euler01, _1, _2, dt);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK2:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(0)
            mTask = boost::bind(&Advect::euler01, _1, _2, dt);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&Advect::euler12, _1, _2, dt);

            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK3:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(0)
            mTask = boost::bind(&Advect::euler01, _1, _2, dt);

            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt/2
            // Phi_t2(2) = 3/4 * Phi_t0(1) + 1/4 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&Advect::euler34, _1, _2, dt);

            // Cook and swap buffer 0 and 2 such that Phi_t2(0) and Phi_t1(2)
            this->cook(PARALLEL_FOR, 2);

            // Convex combine explict Euler step: t3 = t0 + dt
            // Phi_t3(2) = 1/3 * Phi_t0(1) + 2/3 * (Phi_t2(0) - dt * V.Grad_t2(0)
            mTask = boost::bind(&Advect::euler13, _1, _2, dt);

            // Cook and swap buffer 0 and 2 such that Phi_t3(0) and Phi_t2(2)
            this->cook(PARALLEL_FOR, 2);
            break;
        default:
            OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
        }//end of compile-time resolved switch
        OPENVDB_NO_UNREACHABLE_CODE_WARNING_END

        time0 += isForward ? dt : -dt;
        ++countCFL;
        mParent.mTracker.leafs().removeAuxBuffers();
        this->clearField();
        /// Track the narrow band
        mParent.mTracker.track();
    }//end wile-loop over time
    return countCFL;//number of CLF propagation steps
}

template<typename GridT, typename FieldT, typename InterruptT>
template<typename MapT, math::BiasedGradientScheme SpatialScheme,
	math::TemporalIntegrationScheme TemporalScheme>
inline typename GridT::ValueType
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
sampleField(ValueType time0, ValueType time1)
{
    mMaxAbsV = mMinAbsV;
    const size_t leafCount = mParent.mTracker.leafs().leafCount();
    if (leafCount==0) return ValueType(0.0);
    mVec = new VectorType*[leafCount];
    if (mParent.mField.transform() == mParent.mTracker.grid().transform()) {
        mTask = boost::bind(&Advect::sampleAlignedField, _1, _2, time0, time1);
    } else {
        mTask = boost::bind(&Advect::sampleXformedField, _1, _2, time0, time1);
    }
    this->cook(PARALLEL_REDUCE);
    if (math::isExactlyEqual(mMinAbsV, mMaxAbsV)) return ValueType(0.0);//V is essentially zero
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    const ValueType CFL = (TemporalScheme == math::TVD_RK1 ? ValueType(0.3) :
        TemporalScheme == math::TVD_RK2 ? ValueType(0.9) :
        ValueType(1.0))/math::Sqrt(ValueType(3.0));
    const ValueType dt = math::Abs(time1 - time0), dx = mParent.mTracker.voxelSize();
    return math::Min(dt, ValueType(CFL*dx/math::Sqrt(mMaxAbsV)));
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
sampleXformedField(const LeafRange& range, ValueType time0, ValueType time1)
{
    const bool isForward = time0 < time1;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    const MapT& map = *mMap;
    mParent.mTracker.checkInterrupter();
    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        VectorType* vec = new VectorType[leafIter->onVoxelCount()];
        mVec[leafIter.pos()] = vec;
        for (VoxelIterT iter = leafIter->cbeginValueOn(); iter; ++iter, ++vec) {
            const VectorType v = mParent.mField(map.applyMap(iter.getCoord().asVec3d()), time0);
            mMaxAbsV = math::Max(mMaxAbsV, ValueType(math::Pow2(v[0])+math::Pow2(v[1])+math::Pow2(v[2])));
            *vec = isForward ? v : -v;
        }
    }
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
sampleAlignedField(const LeafRange& range, ValueType time0, ValueType time1)
{
    const bool isForward = time0 < time1;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    mParent.mTracker.checkInterrupter();
    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        VectorType* vec = new VectorType[leafIter->onVoxelCount()];
        mVec[leafIter.pos()] = vec;
        for (VoxelIterT iter = leafIter->cbeginValueOn(); iter; ++iter, ++vec) {
            const VectorType v = mParent.mField(iter.getCoord(), time0);
            mMaxAbsV = math::Max(mMaxAbsV, ValueType(math::Pow2(v[0])+math::Pow2(v[1])+math::Pow2(v[2])));
            *vec = isForward ? v : -v;
        }
    }
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
clearField()
{
    if (mVec == NULL) return;
    for (size_t n=0, e=mParent.mTracker.leafs().leafCount(); n<e; ++n) delete [] mVec[n];
    delete [] mVec;
    mVec = NULL;
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
cook(ThreadingMode mode, size_t swapBuffer)
{
    mParent.mTracker.startInterrupter("Advecting level set");

    const int grainSize   = mParent.mTracker.getGrainSize();
    const LeafRange range = mParent.mTracker.leafs().leafRange(grainSize);

    if (grainSize == 0) {
        (*this)(range);
    } else if (mode == PARALLEL_FOR) {
        tbb::parallel_for(range, *this);
    } else if (mode == PARALLEL_REDUCE) {
        tbb::parallel_reduce(range, *this);
    } else {
        OPENVDB_THROW(ValueError,"Undefined threading mode");
    }

    mParent.mTracker.leafs().swapLeafBuffer(swapBuffer, grainSize == 0);

    mParent.mTracker.endInterrupter();
}

// Convex combination of Phi and a forward Euler advection steps:
// Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * V.Grad(0));
template<typename GridT, typename FieldT, typename InterruptT>
template<typename MapT, math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
template <int Nominator, int Denominator>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
Advect<MapT, SpatialScheme, TemporalScheme>::
euler(const LeafRange& range, ValueType dt, Index phiBuffer, Index resultBuffer)
{
    typedef math::BIAS_SCHEME<SpatialScheme>                             SchemeT;
    typedef typename SchemeT::template ISStencil<GridType>::StencilType  StencilT;
    typedef typename LeafType::ValueOnCIter                              VoxelIterT;
    typedef math::GradientBiased<MapT, SpatialScheme>                    GradT;

    static const ValueType Alpha = ValueType(Nominator)/ValueType(Denominator);
    static const ValueType Beta  = ValueType(1) - Alpha;

    mParent.mTracker.checkInterrupter();
    const MapT& map = *mMap;
    StencilT stencil(mParent.mTracker.grid());
    for (typename LeafRange::Iterator leafIter = range.begin(); leafIter; ++leafIter) {
        const VectorType* v = mVec[leafIter.pos()];
        const ValueType* phi = leafIter.buffer(phiBuffer).data();
        ValueType* result = leafIter.buffer(resultBuffer).data();
        for (VoxelIterT voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter, ++v) {
            const Index i = voxelIter.pos();
            stencil.moveTo(voxelIter);
            const ValueType a = stencil.getValue() - dt * v->dot(GradT::result(map, stencil,*v));
            result[i] = Nominator ? Alpha * phi[i] + Beta * a : a;
        }//loop over active voxels in the leaf of the mask
    }//loop over leafs of the level set
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
