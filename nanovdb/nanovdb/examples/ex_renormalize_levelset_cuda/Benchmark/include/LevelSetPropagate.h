// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth
///
/// @file tools/LevelSetPropagate.h
///
/// @brief Propagation of level sets in the normal direction. Unlike LevelSetAdvect
///        that advects a level set in a passive vector field this class advects
///        a level set in a scalar field representing the scalar speed function.

#ifndef OPENVDB_TOOLS_LEVEL_SET_PROPAGATE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_PROPAGATE_HAS_BEEN_INCLUDED

#include <LevelSetTrackerNew.h>
#include <openvdb/tools/Interpolation.h> // for BoxSampler, etc.
#include <openvdb/math/FiniteDifference.h>
#include <functional>
#include <limits>

// #define USE_SINGLE_TIME_STEP_PER_FRAME
// #define USE_MASKING_FOR_NEAR_ZERO_SPEED_VALUES

// Uncomment to enable all additional combination of numerical schemes.
// However note that this will significantly increase compilation times!
//#define OPENVDB_TOOLS_LEVEL_SET_PROPADATE_ADVANCED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Propagation of level sets scalar by means of speed functions, i.e. the
/// interface is advected in the local normal direction with a speed defined by
/// a scalar grid.
///
/// @details
/// The @c InterruptType template argument below refers to any class
/// with the following interface:
/// @code
/// class Interrupter {
///   ...
/// public:
///   void start(const char* name = nullptr) // called when computations begin
///   void end()                             // called when computations end
///   bool wasInterrupted(int percent=-1)    // return true to break computation
/// };
/// @endcode
///
/// @note If no template argument is provided for this InterruptType,
/// the util::NullInterrupter is used, which implies that all interrupter
/// calls are no-ops (i.e., they incur no computational overhead).
template<typename GridT,
         typename InterruptT = util::NullInterrupter>
class LevelSetPropagate
{
public:
    using GridType = GridT;
    using TreeType = typename GridT::TreeType;
    using TrackerT = LevelSetTrackerNew<GridT, InterruptT>;
    using LeafRange = typename TrackerT::LeafRange;
    using LeafType = typename TrackerT::LeafType;
    using BufferType = typename TrackerT::BufferType;
    using ValueType = typename TrackerT::ValueType;

    /// Main constructor
    LevelSetPropagate(GridT& sourceGrid, InterruptT* interrupt = nullptr)
        : mTracker(sourceGrid, interrupt)
        , mSpeed(nullptr)
        , mMask(nullptr)
        , mSpatialScheme(math::HJWENO5_BIAS)
        , mTemporalScheme(math::TVD_RK2)
        , mMinMask(0)
        , mDeltaMask(1)
        , mInvertMask(false)
    {
    }

    virtual ~LevelSetPropagate() {}

    /// Redefine the speed function
    void setSpeed(const GridT& speedGrid) { mSpeed = &speedGrid; }

    /// Define the alpha mask
    void setAlphaMask(const GridT& maskGrid) { mMask = &maskGrid; }

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

    /// @brief Return the minimum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    ValueType minMask() const { return mMinMask; }

    /// @brief Return the maximum value of the mask to be used for the
    /// derivation of a smooth alpha value.
    ValueType maxMask() const { return mDeltaMask + mMinMask; }

    /// @brief Define the range for the (optional) scalar mask.
    /// @param min Minimum value of the range.
    /// @param max Maximum value of the range.
    /// @details Mask values outside the range maps to alpha values of
    /// respectfully zero and one, and values inside the range maps
    /// smoothly to 0->1 (unless of course the mask is inverted).
    /// @throw ValueError if @a min is not smaller than @a max.
    void setMaskRange(ValueType min, ValueType max)
    {
        if (!(min < max)) OPENVDB_THROW(ValueError, "Invalid mask range (expects min < max)");
        mMinMask   = min;
        mDeltaMask = max - min;
    }

    /// @brief Return true if the mask is inverted, i.e. min->max in the
    /// original mask maps to 1->0 in the inverted alpha mask.
    bool isMaskInverted() const { return mInvertMask; }
    /// @brief Invert the optional mask, i.e. min->max in the original
    /// mask maps to 1->0 in the inverted alpha mask.
    void invertMask(bool invert=true) { mInvertMask = invert; }

    /// @brief Propagates the level set from its current time, @a time0, to its
    /// final time, @a time1. If @a time0 > @a time1, perform backward propagation.
    ///
    /// @return the number of CFL iterations used to propagate from @a time0 to @a time1
    size_t propagate(ValueType time0, ValueType time1);

private:

    // disallow copy construction and copy by assignment!
    LevelSetPropagate(const LevelSetPropagate&);// not implemented
    LevelSetPropagate& operator=(const LevelSetPropagate&);// not implemented

    // templated on the spatial scheme
    template<math::BiasedGradientScheme SpatialScheme>
    size_t propagate1(ValueType time0, ValueType time1);

    // templated on both the spatial and temoral schemes
    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    size_t propagate2(ValueType time0, ValueType time1);

    // templated on the map type as well as the spatial and temporal schemes
    template<math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme,
             typename MapType>
    size_t propagate3(ValueType time0, ValueType time1);

    TrackerT                        mTracker;
    const GridT                    *mSpeed, *mMask;
    math::BiasedGradientScheme      mSpatialScheme;
    math::TemporalIntegrationScheme mTemporalScheme;
    ValueType                       mMinMask, mDeltaMask;
    bool                            mInvertMask;

    // This templated private class implements all the level set magic.
    template<typename MapT, math::BiasedGradientScheme SpatialScheme,
             math::TemporalIntegrationScheme TemporalScheme>
    struct Propagate
    {
        /// Main constructor
        Propagate(LevelSetPropagate<GridT, InterruptT>& parent);
        /// Shallow copy constructor called by tbb::parallel_for() threads
        Propagate(const Propagate& other);
        /// Shallow copy constructor called by tbb::parallel_reduce() threads
        Propagate(Propagate& other, tbb::split);
        /// destructor
        virtual ~Propagate() {}
        /// Propagates the level set from its current time, time0, to its final time, time1.
        /// @return number of CFL iterations
        size_t propagate(ValueType time0, ValueType time1);
        /// Used internally by tbb::parallel_for()
        void operator()(const LeafRange& r) const
        {
            if (mTask) mTask(const_cast<Propagate*>(this), r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// Used internally by tbb::parallel_reduce()
        void operator()(const LeafRange& r)
        {
            if (mTask) mTask(this, r);
            else OPENVDB_THROW(ValueError, "task is undefined - don\'t call this method directly");
        }
        /// This is only called by tbb::parallel_reduce() threads
        void join(const Propagate& other) { mMaxAbsS = math::Max(mMaxAbsS, other.mMaxAbsS); }

        /// Enum to define the type of multithreading
        enum ThreadingMode { PARALLEL_FOR, PARALLEL_REDUCE }; // for internal use
        // method calling tbb
        void cook(ThreadingMode mode, size_t swapBuffer = 0);

        /// Sample field and return the CFT time step
        typename GridT::ValueType sampleSpeed(ValueType time0, ValueType time1, Index speedBuffer);
        void sampleXformedSpeed(const LeafRange& r, Index speedBuffer);
        void sampleAlignedSpeed(const LeafRange& r, Index speedBuffer);

        // Convex combination of Phi and a forward Euler propagateion steps:
        // Phi(result) = alpha * Phi(phi) + (1-alpha) * (Phi(0) - dt * Speed(speed)*|Grad[Phi(0)]|);
        template <int Nominator, int Denominator>
        void euler(const LeafRange&, ValueType, Index, Index, Index);
        inline void euler01(const LeafRange& r, ValueType t, Index s) {this->euler<0,1>(r,t,0,1,s);}
        inline void euler12(const LeafRange& r, ValueType t) {this->euler<1,2>(r, t, 1, 1, 2);}
        inline void euler34(const LeafRange& r, ValueType t) {this->euler<3,4>(r, t, 1, 2, 3);}
        inline void euler13(const LeafRange& r, ValueType t) {this->euler<1,3>(r, t, 1, 2, 3);}

        using FuncType = typename std::function<void (Propagate*, const LeafRange&)>;
        LevelSetPropagate* mParent;
        ValueType          mMinAbsS, mMaxAbsS;
        const MapT*        mMap;
        FuncType           mTask;
    }; // end of private Propagate struct

};//end of LevelSetPropagate

template<typename GridT, typename InterruptT>
inline size_t
LevelSetPropagate<GridT, InterruptT>::propagate(ValueType time0, ValueType time1)
{
    switch (mSpatialScheme) {
    case math::HJWENO5_BIAS:
        return this->propagate1<math::HJWENO5_BIAS>(time0, time1);
    case math::UNKNOWN_BIAS:
    default:
        OPENVDB_THROW(ValueError, "Spatial difference scheme not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme>
inline size_t
LevelSetPropagate<GridT, InterruptT>::propagate1(ValueType time0, ValueType time1)
{
    switch (mTemporalScheme) {
    case math::TVD_RK2:
        return this->propagate2<SpatialScheme, math::TVD_RK2>(time0, time1);
    case math::UNKNOWN_TIS:
    default:
        OPENVDB_THROW(ValueError, "Temporal integration scheme not supported!");
    }
    return 0;
}

template<typename GridT, typename InterruptT>
template<math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetPropagate<GridT, InterruptT>::propagate2(ValueType time0, ValueType time1)
{
    const math::Transform& trans = mTracker.grid().transform();
    if (trans.mapType() == math::UniformScaleMap::mapType()) {
        return this->propagate3<SpatialScheme, TemporalScheme, math::UniformScaleMap>(time0, time1);
    } else if (trans.mapType() == math::UniformScaleTranslateMap::mapType()) {
        return this->propagate3<SpatialScheme, TemporalScheme, math::UniformScaleTranslateMap>(
            time0, time1);
    } else if (trans.mapType() == math::UnitaryMap::mapType()) {
        return this->propagate3<SpatialScheme, TemporalScheme, math::UnitaryMap    >(time0, time1);
    } else if (trans.mapType() == math::TranslationMap::mapType()) {
        return this->propagate3<SpatialScheme, TemporalScheme, math::TranslationMap>(time0, time1);
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
LevelSetPropagate<GridT, InterruptT>::propagate3(ValueType time0, ValueType time1)
{
    Propagate<MapT, SpatialScheme, TemporalScheme> tmp(*this);
    return tmp.propagate(time0, time1);
}

///////////////////////////////////////////////////////////////////////

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
Propagate(LevelSetPropagate<GridT, InterruptT>& parent)
    : mParent(&parent)
    , mMinAbsS(ValueType(1e-6))
    , mMap(parent.mTracker.grid().transform().template constMap<MapT>().get())
    , mTask(0)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
Propagate(const Propagate& other)
    : mParent(other.mParent)
    , mMinAbsS(other.mMinAbsS)
    , mMaxAbsS(other.mMaxAbsS)
    , mMap(other.mMap)
    , mTask(other.mTask)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
Propagate(Propagate& other, tbb::split)
    : mParent(other.mParent)
    , mMinAbsS(other.mMinAbsS)
    , mMaxAbsS(other.mMaxAbsS)
    , mMap(other.mMap)
    , mTask(other.mTask)
{
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline size_t
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
propagate(ValueType time0, ValueType time1)
{
    openvdb::util::CpuTimer timer;
    namespace ph = std::placeholders;

    // Make sure we have enough temporal auxiliary buffers for the time
    // integration AS WELL AS an extra buffer with the speed function!
    static const Index auxBuffers = 1 + (TemporalScheme == math::TVD_RK3 ? 2 : 1);
    size_t countCFL = 0;
    while (time0 < time1 && mParent->mTracker.checkInterrupter()) {
        const ValueType dt = time1-time0;
        time0 += dt;
        ++countCFL;

        timer.start("[LevelSetPropagate::propagate()] Advecting level set; no tracking [NanoVDB]");
        if (Benchmark::getInstance().onCPU())
            Benchmark::propagateLevelSet<ExecutionPolicy::CPU>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, Benchmark::getInstance().mSpeed,
                Benchmark::getInstance().mBackground, dt, Benchmark::getInstance().mDx);
        else
            Benchmark::propagateLevelSet<ExecutionPolicy::CUDA>(Benchmark::getInstance().mHandle, Benchmark::getInstance().mVBMHandle, Benchmark::getInstance().mPhi, Benchmark::getInstance().mSpeed,
                Benchmark::getInstance().mBackground, dt, Benchmark::getInstance().mDx);
        timer.stop();

        // Track the narrow band
        mParent->mTracker.track();
    }//end wile-loop over time

    return countCFL;//number of CLF propagation steps
}// LevelSetPropagate::Propagate::propagate

template<typename GridT, typename InterruptT>
template<typename MapT, math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
inline typename GridT::ValueType
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
sampleSpeed(ValueType time0, ValueType time1, Index speedBuffer)
{
    namespace ph = std::placeholders;

    mMaxAbsS = mMinAbsS;
    const size_t leafCount = mParent->mTracker.leafs().leafCount();
    if (leafCount==0 || time0 >= time1) return ValueType(0);

    const math::Transform& xform  = mParent->mTracker.grid().transform();
    if (mParent->mSpeed == nullptr) {
        OPENVDB_THROW(openvdb::ValueError, "Propagate::sampleSpeed: setSpeed was not called\n");
    }
    if (mParent->mSpeed->transform() == xform &&
       (mParent->mMask == nullptr || mParent->mMask->transform() == xform)) {
        mTask = std::bind(&Propagate::sampleAlignedSpeed, ph::_1, ph::_2, speedBuffer);
    } else {
        throw std::runtime_error("Unexpected call to Propagate::sampleXformedSpeed");
        mTask = std::bind(&Propagate::sampleXformedSpeed, ph::_1, ph::_2, speedBuffer);
    }
    this->cook(PARALLEL_REDUCE);
    if (math::isApproxEqual(mMinAbsS, mMaxAbsS)) return ValueType(0);//speed is essentially zero
    static const ValueType CFL = (TemporalScheme == math::TVD_RK1 ? ValueType(0.3) :
                                  TemporalScheme == math::TVD_RK2 ? ValueType(0.9) :
                                  ValueType(1.0))/math::Sqrt(ValueType(3.0));
    const ValueType dt = math::Abs(time1 - time0), dx = mParent->mTracker.voxelSize();
    return math::Min(dt, ValueType(CFL*dx/mMaxAbsS));
}

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
sampleXformedSpeed(const LeafRange& range, Index speedBuffer)
{
    const MapT& map = *mMap;
    mParent->mTracker.checkInterrupter();
    if (mParent->mSpeed == nullptr) {
        OPENVDB_THROW(openvdb::ValueError, "Propagate::sampleXformedSpeed: setSpeed was not called\n");
    }
    auto speedAcc = mParent->mSpeed->getConstAccessor();
    using SamplerT = tools::GridSampler<typename GridT::ConstAccessor, tools::BoxSampler>;
    SamplerT speed(speedAcc, mParent->mSpeed->transform());

    if (mParent->mMask == nullptr) {// no alpha masking
        for (auto leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(speedBuffer).data();
            bool isZero = true;
            for (auto voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                ValueType& s = buffer[voxelIter.pos()];
                s = speed.wsSample(map.applyMap(voxelIter.getCoord().asVec3d()));
                if (!math::isApproxZero(s)) isZero = false;
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
            if (isZero) buffer[0] = std::numeric_limits<ValueType>::max();//tag first voxel
        }
    } else {// alpha masking
        const ValueType min = mParent->mMinMask, invNorm = 1.0f/(mParent->mDeltaMask);
        const bool invMask = mParent->isMaskInverted();
        auto maskAcc = mParent->mMask->getConstAccessor();
        SamplerT mask(maskAcc,  mParent->mMask->transform());
        for (auto leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(speedBuffer).data();
            bool isZero = true;
            for (auto voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                const Vec3R xyz = map.applyMap(voxelIter.getCoord().asVec3d());//world space
                const ValueType a = math::SmoothUnitStep((mask.wsSample(xyz)-min)*invNorm);
                ValueType& s = buffer[voxelIter.pos()];
                s = speed.wsSample(xyz) * (invMask ? 1 - a : a);
                if (!math::isApproxZero(s)) isZero = false;
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
            if (isZero) buffer[0] = std::numeric_limits<ValueType>::max();//tag first voxel
        }
    }
}// LevelSetPropagate::Propagate::sampleXformedSpeed

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
sampleAlignedSpeed(const LeafRange& range, Index speedBuffer)
{
    mParent->mTracker.checkInterrupter();
    auto speed = mParent->mSpeed->getConstAccessor();

    if (mParent->mMask == nullptr) {// no alpha masking
        for (auto leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(speedBuffer).data();
#ifdef USE_MASKING_FOR_NEAR_ZERO_SPEED_VALUES
            bool isZero = true;
#endif
            for (auto voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                ValueType& s = buffer[voxelIter.pos()];
                s = speed.getValue(voxelIter.getCoord());
#ifdef USE_MASKING_FOR_NEAR_ZERO_SPEED_VALUES
                if (!math::isApproxZero(s)) isZero = false;
#endif
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
#ifdef USE_MASKING_FOR_NEAR_ZERO_SPEED_VALUES
            if (isZero) buffer[0] = std::numeric_limits<ValueType>::max();//tag first voxel
#endif
        }
    } else {// alpha masking
        throw std::runtime_error("Unexpected use of alpha masking");
        const ValueType min = mParent->mMinMask, invNorm = 1.0f/(mParent->mDeltaMask);
        const bool invMask = mParent->isMaskInverted();
        auto mask = mParent->mMask->getConstAccessor();
        for (auto leafIter = range.begin(); leafIter; ++leafIter) {
            ValueType* buffer = leafIter.buffer(speedBuffer).data();
            bool isZero = true;
            for (auto voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
                const Coord ijk = voxelIter.getCoord();//index space
                const ValueType a = math::SmoothUnitStep((mask.getValue(ijk)-min)*invNorm);
                ValueType& s = buffer[voxelIter.pos()];
                s = speed.getValue(ijk) * (invMask ? 1 - a : a);
                if (!math::isApproxZero(s)) isZero = false;
                mMaxAbsS = math::Max(mMaxAbsS, math::Abs(s));
            }
            if (isZero) buffer[0] = std::numeric_limits<ValueType>::max();//tag first voxel
        }
    }
}// LevelSetPropagate::Propagate::sampleAlignedSpeed

template<typename GridT, typename InterruptT>
template <typename MapT, math::BiasedGradientScheme SpatialScheme,
          math::TemporalIntegrationScheme TemporalScheme>
inline void
LevelSetPropagate<GridT, InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
cook(ThreadingMode mode, size_t swapBuffer)
{
    mParent->mTracker.startInterrupter("Propagateing level set");

    const int grainSize   = mParent->mTracker.getGrainSize();
    const LeafRange range = mParent->mTracker.leafs().leafRange(grainSize);

    if (mParent->mTracker.getGrainSize()==0) {
        (*this)(range);
    } else if (mode == PARALLEL_FOR) {
        tbb::parallel_for(range, *this);
    } else if (mode == PARALLEL_REDUCE) {
        tbb::parallel_reduce(range, *this);
    } else {
        OPENVDB_THROW(ValueError, "expected threading mode " << int(PARALLEL_FOR)
            << " or " << int(PARALLEL_REDUCE) << ", got " << int(mode));
    }

    mParent->mTracker.leafs().swapLeafBuffer(swapBuffer, grainSize == 0);

    mParent->mTracker.endInterrupter();
}

template<typename GridT, typename InterruptT>
template<typename MapT, math::BiasedGradientScheme SpatialScheme,
         math::TemporalIntegrationScheme TemporalScheme>
template <int Numerator, int Denominator>
inline void
LevelSetPropagate<GridT,InterruptT>::
Propagate<MapT, SpatialScheme, TemporalScheme>::
euler(const LeafRange& range, ValueType dt,
      Index phiBuffer, Index resultBuffer, Index speedBuffer)
{
    using SchemeT = math::BIAS_SCHEME<SpatialScheme>;
    using StencilT = typename SchemeT::template ISStencil<GridType>::StencilType;
#ifdef USE_NANOVDB_IMPLEMENTATION_FOR_NORMGRAD
    using GradientT = math::ISGradientNormSqrd<SpatialScheme>;
    using NanoGradientT = typename nanovdb::math::WenoStencil<nanovdb::FloatGrid>; // For debugging purposes
#else
    using NumGrad = math::GradientNormSqrd<MapT, SpatialScheme>;
#endif

    static const ValueType Alpha = ValueType(Numerator)/ValueType(Denominator);
    static const ValueType Beta  = ValueType(1) - Alpha;

    mParent->mTracker.checkInterrupter();
    const MapT& map = *mMap;
    StencilT stencil(mParent->mTracker.grid());

#ifdef USE_NANOVDB_IMPLEMENTATION_FOR_NORMGRAD
    auto uniformMapPtr = dynamic_cast<const math::UniformScaleMap*>(mMap);
    ValueType invdxdx = ValueType(uniformMapPtr->getInvScaleSqr()[0]);
#endif

    for (auto leafIter = range.begin(); leafIter; ++leafIter) {
        const ValueType* speed = leafIter.buffer(speedBuffer).data();
#ifdef USE_MASKING_FOR_NEAR_ZERO_SPEED_VALUES
        if (math::isExactlyEqual(speed[0], std::numeric_limits<ValueType>::max())) continue;//check for zero speed
#else
        if (math::isExactlyEqual(speed[0], std::numeric_limits<ValueType>::max()))
            throw std::runtime_error("Unexpected sentinel value for speed encountered");
#endif
        const ValueType* phi = leafIter.buffer(phiBuffer).data();
        ValueType* result = leafIter.buffer(resultBuffer).data();
        for (auto voxelIter = leafIter->cbeginValueOn(); voxelIter; ++voxelIter) {
            const Index n = voxelIter.pos();
            if (math::isApproxZero(speed[n])) continue;
            stencil.moveTo(voxelIter);
#ifdef USE_NANOVDB_IMPLEMENTATION_FOR_NORMGRAD
            ValueType normSqGradPhi = GradientT::result(stencil);
            ValueType nanoStencil[NanoGradientT::SIZE];
            nanoStencil[nanovdb::math::WenoPt< 0, 0, 0>::idx] = stencil.template getValue< 0, 0, 0>();
            nanoStencil[nanovdb::math::WenoPt< 1, 0, 0>::idx] = stencil.template getValue< 1, 0, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 1, 0>::idx] = stencil.template getValue< 0, 1, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 0, 1>::idx] = stencil.template getValue< 0, 0, 1>();
            nanoStencil[nanovdb::math::WenoPt<-1, 0, 0>::idx] = stencil.template getValue<-1, 0, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0,-1, 0>::idx] = stencil.template getValue< 0,-1, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 0,-1>::idx] = stencil.template getValue< 0, 0,-1>();
            nanoStencil[nanovdb::math::WenoPt< 2, 0, 0>::idx] = stencil.template getValue< 2, 0, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 2, 0>::idx] = stencil.template getValue< 0, 2, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 0, 2>::idx] = stencil.template getValue< 0, 0, 2>();
            nanoStencil[nanovdb::math::WenoPt<-2, 0, 0>::idx] = stencil.template getValue<-2, 0, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0,-2, 0>::idx] = stencil.template getValue< 0,-2, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 0,-2>::idx] = stencil.template getValue< 0, 0,-2>();
            nanoStencil[nanovdb::math::WenoPt< 3, 0, 0>::idx] = stencil.template getValue< 3, 0, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 3, 0>::idx] = stencil.template getValue< 0, 3, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 0, 3>::idx] = stencil.template getValue< 0, 0, 3>();
            nanoStencil[nanovdb::math::WenoPt<-3, 0, 0>::idx] = stencil.template getValue<-3, 0, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0,-3, 0>::idx] = stencil.template getValue< 0,-3, 0>();
            nanoStencil[nanovdb::math::WenoPt< 0, 0,-3>::idx] = stencil.template getValue< 0, 0,-3>();
            const ValueType normSqGradPhiNano = NanoGradientT::normSqGrad(nanoStencil, 1.0, 1.0);
            if (fabs(normSqGradPhi-normSqGradPhiNano)>2e-3){
                std::cout << "difference = " << fabs(normSqGradPhi-normSqGradPhiNano) << std::endl;
                throw std::runtime_error("Inconsistency in WENO calculation");
            }
            normSqGradPhi = normSqGradPhiNano;
            const ValueType v = stencil.getValue() - dt * speed[n] * invdxdx * normSqGradPhiNano;
#else
            const ValueType v = stencil.getValue() - dt * speed[n] * NumGrad::result(map, stencil);
#endif
            result[n] = Numerator ? Alpha * phi[n] + Beta * v : v;
        }//loop over active voxels in the leaf of the mask
    }//loop over leafs of the level set
}// LevelSetPropagate::Propagate::euler

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_PROPAGATE_HAS_BEEN_INCLUDED
