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
/// @file LevelSetAdvect.h
///
/// @brief Hyperbolic advection of narrow-band level sets

#ifndef OPENVDB_TOOLS_LEVEL_SET_ADVECT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_ADVECT_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include "LevelSetTracker.h"
#include "Interpolation.h" // for BoxSampler, etc.
#include <openvdb/math/FiniteDifference.h>
#include <boost/math/constants/constants.hpp>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// Below are two simple wrapper classes for advection velocity fields
/// DiscreteField wraps a velocity grid and EnrightField is mostly
/// intended for debugging (it's an analytical divergence free and
/// periodic field). They both share the same API required by the
/// LevelSetAdvection class defined below. Thus, any class with this
/// API should work with LevelSetAdvection.

/// Note the Field wrapper classes below always assume the velocity
/// is represented in the world-frame of reference. For DiscreteField
/// this implies the input grid must contain velocities in world
/// coordinates.

/// @brief Thin wrapper class for a velocity grid
/// @note Consider replacing BoxSampler with StaggeredBoxSampler
template <typename VelGridT, typename Interpolator = BoxSampler>
class DiscreteField
{
public:
    typedef typename VelGridT::ValueType     VectorType;
    typedef typename VectorType::ValueType   ScalarType;

    DiscreteField(const VelGridT &vel): mAccessor(vel.tree()), mTransform(&vel.transform()) {}

    /// @return const reference to the transfrom between world and index space
    /// @note Use this method to determine if a client grid is
    /// aligned with the coordinate space of the velocity grid.
    const math::Transform& transform() const { return *mTransform; }

    /// @return the interpolated velocity at the world space position xyz
    inline VectorType operator() (const Vec3d& xyz, ScalarType) const
    {
        VectorType result = zeroVal<VectorType>();
        Interpolator::sample(mAccessor, mTransform->worldToIndex(xyz), result);
        return result;
    }

    /// @return the velocity at the coordinate space position ijk
    inline VectorType operator() (const Coord& ijk, ScalarType) const
    {
        return mAccessor.getValue(ijk);
    }

private:
    const typename VelGridT::ConstAccessor mAccessor;//Not thread-safe
    const math::Transform*                 mTransform;

}; // end of DiscreteField

/// @brief Analytical, divergence-free and periodic vecloity field
/// @note Primarily intended for debugging!
/// @warning This analytical velocity only produce meaningfull values
/// in the unitbox in world space. In other words make sure any level
/// set surface in fully enclodes in the axis aligned bounding box
/// spanning 0->1 in world units.
template <typename ScalarT = float>
class EnrightField
{
public:
    typedef ScalarT             ScalarType;
    typedef math::Vec3<ScalarT> VectorType;

    EnrightField() {}

    /// @return const reference to the identity transfrom between world and index space
    /// @note Use this method to determine if a client grid is
    /// aligned with the coordinate space of this velocity field
    math::Transform transform() const { return math::Transform(); }

    /// @return the velocity in world units, evaluated at the world
    /// position xyz and at the specified time
    inline VectorType operator() (const Vec3d& xyz, ScalarType time) const;

    /// @return the velocity at the coordinate space position ijk
    inline VectorType operator() (const Coord& ijk, ScalarType time) const
    {
        return (*this)(ijk.asVec3d(), time);
    }
}; // end of EnrightField

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
///   openvdb::VectorType operator() (const openvdb::Coord& xyz, ScalarType time) const;
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
    typedef typename TrackerT::RangeType       RangeType;
    typedef typename TrackerT::LeafType        LeafType;
    typedef typename TrackerT::BufferType      BufferType;
    typedef typename TrackerT::ValueType       ScalarType;
    typedef typename FieldT::VectorType        VectorType;

    /// Main constructor
    LevelSetAdvection(GridT& grid, const FieldT& field, InterruptT* interrupt = NULL):
        mTracker(grid, interrupt), mField(field),
        mSpatialScheme(math::HJWENO5_BIAS),
        mTemporalScheme(math::TVD_RK2) {}

    virtual ~LevelSetAdvection() {};

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
    /// @note A grainsize of 0 or less disables multi-threading!
    void setGrainSize(int grainsize) { mTracker.setGrainSize(grainsize); }

    /// Advect the level set from it's current time, time0, to it's
    /// final time, time1. If time0>time1 backward advection is performed.
    ///
    /// @return number of CFL iterations used to advect from time0 to time1
    size_t advect(ScalarType time0, ScalarType time1);

private:

    // This templated private class implements all the level set magic.
    template<typename MapT, math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    class LevelSetAdvect
    {
    public:
        /// Main constructor
        LevelSetAdvect(LevelSetAdvection& parent);
        /// Shallow copy constructor called by tbb::parallel_for() threads
        LevelSetAdvect(const LevelSetAdvect& other);
        /// Shallow copy constructor called by tbb::parallel_reduce() threads
        LevelSetAdvect(LevelSetAdvect& other, tbb::split);
        /// destructor
        virtual ~LevelSetAdvect() {if (mIsMaster) this->clearField();};
        /// Advect the level set from it's current time, time0, to it's final time, time1.
        /// @return number of CFL iterations
        size_t advect(ScalarType time0, ScalarType time1);
        /// Used internally by tbb::parallel_for()
        void operator()(const RangeType& r) const
        {
            if (mTask) mTask(const_cast<LevelSetAdvect*>(this), r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// Used internally by tbb::parallel_reduce()
        void operator()(const RangeType& r)
        {
            if (mTask) mTask(this, r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// This is only called by tbb::parallel_reduce() threads
        void join(const LevelSetAdvect& other) { mMaxAbsV = math::Max(mMaxAbsV, other.mMaxAbsV); }
    private:
        typedef typename boost::function<void (LevelSetAdvect*, const RangeType&)> FuncType;
        LevelSetAdvection& mParent;
        VectorType**       mVec;
        const ScalarType   mMinAbsV;
        ScalarType         mMaxAbsV;
        const MapT*        mMap;
        FuncType           mTask;
        const bool         mIsMaster;
        /// Enum to defeing the type of multi-threading
        enum ThreadingMode { PARALLEL_FOR, PARALLEL_REDUCE }; // for internal use
        // method calling tbb
        void cook(ThreadingMode mode, size_t swapBuffer = 0);
        /// Sample field and return the CFT time step
        typename GridT::ValueType sampleField(ScalarType time0, ScalarType time1);
        void  clearField();
        void  sampleXformedField(const RangeType& r, ScalarType time0, ScalarType time1);
        void  sampleAlignedField(const RangeType& r, ScalarType time0, ScalarType time1);
        // Forward Euler advection steps: Phi(result) = Phi(0) - dt * V.Grad(0);
        void euler1(const RangeType& r, ScalarType dt, Index resultBuffer);
        // Convex combination of Phi and a forward Euler advection steps:
        // Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * V.Grad(0));
        void euler2(const RangeType& r, ScalarType dt, ScalarType alpha, Index phiBuffer, Index resultBuffer);
    }; // end of private LevelSetAdvect class

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
    //each thread needs a deep copy of the field since it might contain a ValueAccessor
    const FieldT                    mField;
    math::BiasedGradientScheme      mSpatialScheme;
    math::TemporalIntegrationScheme mTemporalScheme;

    // disallow copy by assignment
    void operator=(const LevelSetAdvection& other) {}

};//end of LevelSetAdvection

template<typename GridT, typename FieldT, typename InterruptT>
inline size_t
LevelSetAdvection<GridT, FieldT, InterruptT>::advect(ScalarType time0, ScalarType time1)
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
LevelSetAdvection<GridT, FieldT, InterruptT>::advect1(ScalarType time0, ScalarType time1)
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
LevelSetAdvection<GridT, FieldT, InterruptT>::advect2(ScalarType time0, ScalarType time1)
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
LevelSetAdvection<GridT, FieldT, InterruptT>::advect3(ScalarType time0, ScalarType time1)
{
    LevelSetAdvect<MapT, SpatialScheme, TemporalScheme> tmp(*this);
    return tmp.advect(time0, time1);
}

///////////////////////////////////////////////////////////////////////

template <typename ScalarT>
inline math::Vec3<ScalarT>
EnrightField<ScalarT>::operator() (const Vec3d& xyz, ScalarType time) const
{
    const ScalarT pi = boost::math::constants::pi<ScalarT>();
    const ScalarT phase = pi / ScalarT(3.0);
    const ScalarT Px =  pi * ScalarT(xyz[0]), Py = pi * ScalarT(xyz[1]), Pz = pi * ScalarT(xyz[2]);
    const ScalarT tr =  cos(ScalarT(time) * phase);
    const ScalarT a  =  sin(ScalarT(2.0)*Py);
    const ScalarT b  = -sin(ScalarT(2.0)*Px);
    const ScalarT c  =  sin(ScalarT(2.0)*Pz);
    return math::Vec3<ScalarT>(
        tr * ( ScalarT(2) * math::Pow2(sin(Px)) * a * c ),
        tr * ( b * math::Pow2(sin(Py)) * c ),
        tr * ( b * a * math::Pow2(sin(Pz)) ));
}


///////////////////////////////////////////////////////////////////////


template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetAdvection<GridT, FieldT, InterruptT>::
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
LevelSetAdvect(LevelSetAdvection& parent):
    mParent(parent),
    mVec(NULL),
    mMinAbsV(ScalarType(1e-6)),
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
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
LevelSetAdvect(const LevelSetAdvect& other):
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
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
LevelSetAdvect(LevelSetAdvect& other, tbb::split):
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
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
advect(ScalarType time0, ScalarType time1)
{
    size_t countCFL = 0;
    if ( math::isZero(time0 - time1) ) return countCFL;
    const bool isForward = time0 < time1;
    while ((isForward ? time0<time1 : time0>time1) && mParent.mTracker.checkInterrupter()) {
        /// Make sure we have enough temporal auxiliary buffers
        mParent.mTracker.leafs().rebuildAuxBuffers(TemporalScheme == math::TVD_RK3 ? 2 : 1);

        const ScalarType dt = this->sampleField(time0, time1);
        if ( math::isZero(dt) ) break;//V is essentially zero so terminate

        OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN //switch is resolved at compile-time
        switch(TemporalScheme) {
        case math::TVD_RK1:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(0)
            mTask = boost::bind(&LevelSetAdvect::euler1, _1, _2, dt, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK2:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(0)
            mTask = boost::bind(&LevelSetAdvect::euler1, _1, _2, dt, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt
            // Phi_t2(1) = 1/2 * Phi_t0(1) + 1/2 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&LevelSetAdvect::euler2, _1, _2, dt, ScalarType(0.5), /*phi=*/1, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t2(0) and Phi_t1(1)
            this->cook(PARALLEL_FOR, 1);
            break;
        case math::TVD_RK3:
            // Perform one explicit Euler step: t1 = t0 + dt
            // Phi_t1(1) = Phi_t0(0) - dt * VdotG_t0(0)
            mTask = boost::bind(&LevelSetAdvect::euler1, _1, _2, dt, /*result=*/1);
            // Cook and swap buffer 0 and 1 such that Phi_t1(0) and Phi_t0(1)
            this->cook(PARALLEL_FOR, 1);

            // Convex combine explict Euler step: t2 = t0 + dt/2
            // Phi_t2(2) = 3/4 * Phi_t0(1) + 1/4 * (Phi_t1(0) - dt * V.Grad_t1(0))
            mTask = boost::bind(&LevelSetAdvect::euler2, _1, _2, dt, ScalarType(0.75), /*phi=*/1, /*result=*/2);
            // Cook and swap buffer 0 and 2 such that Phi_t2(0) and Phi_t1(2)
            this->cook(PARALLEL_FOR, 2);

            // Convex combine explict Euler step: t3 = t0 + dt
            // Phi_t3(2) = 1/3 * Phi_t0(1) + 2/3 * (Phi_t2(0) - dt * V.Grad_t2(0)
            mTask = boost::bind(&LevelSetAdvect::euler2, _1, _2, dt, ScalarType(1.0/3.0), /*phi=*/1, /*result=*/2);
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
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
sampleField(ScalarType time0, ScalarType time1)
{
    mMaxAbsV = mMinAbsV;
    const size_t leafCount = mParent.mTracker.leafs().leafCount();
    if (leafCount==0) return ScalarType(0.0);
    mVec = new VectorType*[leafCount];
    if (mParent.mField.transform() == mParent.mTracker.grid().transform()) {
        mTask = boost::bind(&LevelSetAdvect::sampleAlignedField, _1, _2, time0, time1);
    } else {
        mTask = boost::bind(&LevelSetAdvect::sampleXformedField, _1, _2, time0, time1);
    }
    this->cook(PARALLEL_REDUCE);
    if (math::isExactlyEqual(mMinAbsV, mMaxAbsV)) return ScalarType(0.0);//V is essentially zero
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    const ScalarType CFL = (TemporalScheme == math::TVD_RK1 ? ScalarType(0.3) :
        TemporalScheme == math::TVD_RK2 ? ScalarType(0.9) :
        ScalarType(1.0))/math::Sqrt(ScalarType(3.0));
    const ScalarType dt = math::Abs(time1 - time0), dx = mParent.mTracker.voxelSize();
    return math::Min(dt, ScalarType(CFL*dx/math::Sqrt(mMaxAbsV)));
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
sampleXformedField(const RangeType& range, ScalarType time0, ScalarType time1)
{
    const bool isForward = time0 < time1;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    const MapT& map = *mMap;
    mParent.mTracker.checkInterrupter();
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        const LeafType& leaf = mParent.mTracker.leafs().leaf(n);
        VectorType* vec = new VectorType[leaf.onVoxelCount()];
        int m = 0;
        for (VoxelIterT iter = leaf.cbeginValueOn(); iter; ++iter, ++m) {
            const VectorType V = mParent.mField(map.applyMap(iter.getCoord().asVec3d()), time0);
            mMaxAbsV = math::Max(mMaxAbsV, ScalarType(math::Pow2(V[0])+math::Pow2(V[1])+math::Pow2(V[2])));
            vec[m] = isForward ? V : -V;
        }
        mVec[n] = vec;
    }
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
sampleAlignedField(const RangeType& range, ScalarType time0, ScalarType time1)
{
    const bool isForward = time0 < time1;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    mParent.mTracker.checkInterrupter();
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        const LeafType& leaf = mParent.mTracker.leafs().leaf(n);
        VectorType* vec = new VectorType[leaf.onVoxelCount()];
        int m = 0;
        for (VoxelIterT iter = leaf.cbeginValueOn(); iter; ++iter, ++m) {
            const VectorType V = mParent.mField(iter.getCoord(), time0);
            mMaxAbsV = math::Max(mMaxAbsV, ScalarType(math::Pow2(V[0])+math::Pow2(V[1])+math::Pow2(V[2])));
            vec[m] = isForward ? V : -V;
        }
        mVec[n] = vec;
    }
}

template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
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
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
cook(ThreadingMode mode, size_t swapBuffer)
{
    mParent.mTracker.startInterrupter("Advecting level set");

    if (mParent.mTracker.getGrainSize()==0) {
        (*this)(mParent.mTracker.leafs().getRange());
    } else if (mode == PARALLEL_FOR) {
        tbb::parallel_for(mParent.mTracker.leafs().getRange(mParent.mTracker.getGrainSize()), *this);
    } else if (mode == PARALLEL_REDUCE) {
        tbb::parallel_reduce(mParent.mTracker.leafs().getRange(mParent.mTracker.getGrainSize()), *this);
    } else {
        throw std::runtime_error("Undefined threading mode");
    }

    mParent.mTracker.leafs().swapLeafBuffer(swapBuffer, mParent.mTracker.getGrainSize()==0);

    mParent.mTracker.endInterrupter();
}

// Forward Euler advection steps:
// Phi(result) = Phi(0) - dt * V.Grad(0);
template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
euler1(const RangeType& range, ScalarType dt, Index resultBuffer)
{
    typedef math::BIAS_SCHEME<SpatialScheme>                             Scheme;
    typedef typename Scheme::template ISStencil<GridType>::StencilType   Stencil;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    mParent.mTracker.checkInterrupter();
    const MapT& map = *mMap;
    typename TrackerT::LeafManagerType& leafs = mParent.mTracker.leafs();
    Stencil stencil(mParent.mTracker.grid());
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        BufferType& result = leafs.getBuffer(n, resultBuffer);
        const VectorType* vec = mVec[n];
        int m=0;
        for (VoxelIterT iter = leafs.leaf(n).cbeginValueOn(); iter; ++iter, ++m) {
            stencil.moveTo(iter);
            const VectorType V = vec[m], G = math::GradientBiased<MapT, SpatialScheme>::result(map, stencil, V);
            result.setValue(iter.pos(), *iter - dt * V.dot(G));
        }
    }
}

// Convex combination of Phi and a forward Euler advection steps:
// Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * V.Grad(0));
template<typename GridT, typename FieldT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetAdvection<GridT, FieldT, InterruptT>::
LevelSetAdvect<MapT, SpatialScheme, TemporalScheme>::
euler2(const RangeType& range, ScalarType dt, ScalarType alpha, Index phiBuffer, Index resultBuffer)
{
    typedef math::BIAS_SCHEME<SpatialScheme>                             Scheme;
    typedef typename Scheme::template ISStencil<GridType>::StencilType   Stencil;
    typedef typename LeafType::ValueOnCIter VoxelIterT;
    mParent.mTracker.checkInterrupter();
    const MapT& map = *mMap;
    typename TrackerT::LeafManagerType& leafs = mParent.mTracker.leafs();
    const ScalarType beta = ScalarType(1.0) - alpha;
    Stencil stencil(mParent.mTracker.grid());
    for (size_t n=range.begin(), e=range.end(); n != e; ++n) {
        const BufferType& phi = leafs.getBuffer(n, phiBuffer);
        BufferType& result = leafs.getBuffer(n, resultBuffer);
        const VectorType* vec = mVec[n];
        int m=0;
        for (VoxelIterT iter = leafs.leaf(n).cbeginValueOn(); iter; ++iter, ++m) {
            stencil.moveTo(iter);
            const VectorType V = vec[m], G = math::GradientBiased<MapT, SpatialScheme>::result(map, stencil, V);
            result.setValue(iter.pos(), alpha*phi[iter.pos()] + beta*(*iter - dt * V.dot(G)));
        }
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_ADVECT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2014 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
