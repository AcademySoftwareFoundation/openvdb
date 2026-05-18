// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_IMPL_HAS_BEEN_INCLUDED

/// when enabled, prints timings for each substep of the PCA algorithm
#ifdef OPENVDB_PROFILE_PCA
#include <openvdb/util/CpuTimer.h>
#endif

/// @brief  Experimental option to skip storing the self weight for the
///   weighted PCA when set to 0
/// @todo  Decide what's more correct and remove this. Self contributions aren't
///   guaranteed with maxSourcePointsPerVoxel/maxTargetPointsPerVoxel
#define OPENVDB_PCA_SELF_CONTRIBUTION 1

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace pca_internal {

#ifdef OPENVDB_PROFILE_PCA
using TimerT = util::CpuTimer;
#else
struct NoTimer {
inline void start(const char*) {}
inline void stop() {}
};
using TimerT = NoTimer;
#endif

struct PcaTimer final : public TimerT
{
    PcaTimer(util::NullInterrupter* const interrupt)
        : TimerT()
        , mInterrupt(interrupt) {}
    inline void start(const char* msg)
    {
        TimerT::start(msg);
        if (mInterrupt) mInterrupt->start(msg);
    }
    inline void stop()
    {
        TimerT::stop();
        if (mInterrupt) mInterrupt->end();
    }
    util::NullInterrupter* const mInterrupt;
};

using WeightSumT = double;
using WeightedPositionSumT = Vec3d;
using GroupIndexT = points::AttributeSet::Descriptor::GroupIndex;

struct AttrIndices
{
    size_t mPosSumIndex;
    size_t mWeightSumIndex;
    size_t mCovMatrixIndex;
    size_t mPWsIndex;
    GroupIndexT mEllipsesGroupIndex;
};

template <typename T, typename LeafNodeT>
inline T* initPcaArrayAttribute(LeafNodeT& leaf, const size_t idx, const bool fill = true)
{
    auto& array = leaf.attributeArray(idx);
    OPENVDB_ASSERT(array.valueType() == typeNameAsString<T>());
    OPENVDB_ASSERT(array.codecType() == std::string(NullCodec::name()));
    array.expand(fill); // @note  does nothing if array is already not uniform
    const char* data = array.constDataAsByteArray();
    return reinterpret_cast<T*>(const_cast<char*>(data));
}

/// @note The PCA transfer modules are broken into two separate steps, the
///   first which computes the weighted neighbourhoods of points and the second
///   which computes the covariance matrices. Both these steps perform an
///   identical loop over each points neighbourhood, with the second being
///   necessary to re-compute the exact positions and per position weights.
///   In theory, only a single loop is required; each point could instead
///   create and append a large list of the exact neighbourhood positions and
///   weights that impact it and use these to compute the covariance after the
///   first neighbourhood loop has completed (a true gather style approach).
///
///   The double loop technique was chosen to better handle the computation
///   of anisotropy for _all_ points and _all_ neighbourhoods (i.e. no limiting
///   via the max point per voxel (MPPV) options). It makes the PCA method
///   extremely memory efficient for _all_ MPPV and radius values. However
///   it might be worth falling back to a gather stlye approach when the MPPV
///   and radius values are relatively small. We should investigate this in the
///   future.
template <typename PointDataTreeT>
struct PcaTransfer
    : public VolumeTransfer<PointDataTreeT>
    , public InterruptableTransfer
{
    using BaseT = VolumeTransfer<PointDataTreeT>;
    using LeafNodeType = typename PointDataTreeT::LeafNodeType;
    // We know the codec is null as this works on the world space positions
    // that are temporarily computed for this algorithm
    using PositionHandleT = points::AttributeHandle<Vec3d, NullCodec>;

    PcaTransfer(const AttrIndices& indices,
                const PcaSettings& settings,
                const Real vs,
                tree::LeafManager<PointDataTreeT>& manager,
                util::NullInterrupter* interrupt)
        : BaseT(manager.tree())
        , InterruptableTransfer(interrupt)
        , mIndices(indices)
        , mSettings(settings)
        , mDxInv(1.0/vs)
        , mManager(manager)
        , mTargetPosition()
        , mSourcePosition() {
            OPENVDB_ASSERT(std::isfinite(mSettings.searchRadius));
            OPENVDB_ASSERT(std::isfinite(mDxInv));
        }

    PcaTransfer(const PcaTransfer& other)
        : BaseT(other)
        , InterruptableTransfer(other)
        , mIndices(other.mIndices)
        , mSettings(other.mSettings)
        , mDxInv(other.mDxInv)
        , mManager(other.mManager)
        , mTargetPosition()
        , mSourcePosition() {}

    /*
    // This is no longer used but kept for reference; each axis iterations of
    // derived PCA methods uses this technique to skip voxels entirely if a
    // point's position and search radius does not intersect it

    static bool VoxelIntersectsSphere(const Coord& ijk, const Vec3d& PosIS, const Real r2)
    {
        const Vec3d min = ijk.asVec3d() - 0.5;
        const Vec3d max = ijk.asVec3d() + 0.5;
        Real dmin = 0;
        for (int i = 0; i < 3; ++i) {
            if (PosIS[i] < min[i])      dmin += math::Pow2(PosIS[i] - min[i]);
            else if (PosIS[i] > max[i]) dmin += math::Pow2(PosIS[i] - max[i]);
        }
        return dmin <= r2;
    }
    */

    template <typename RealT>
    static constexpr size_t GetBatchSize()
    {
        // @note You can return 1 here to disable batched rasterization
        using NativeT = typename simd::SimdNativeT<RealT>::Type;
        return simd::SimdTraits<NativeT>::size;
    }

    float searchRadius() const { return mSettings.searchRadius; }
    size_t neighbourThreshold() const { return mSettings.neighbourThreshold; }
    size_t maxSourcePointsPerVoxel() const { return mSettings.maxSourcePointsPerVoxel; }
    size_t maxTargetPointsPerVoxel() const { return mSettings.maxTargetPointsPerVoxel; }

    Vec3i range(const Coord&, size_t) const { return this->range(); }
    Vec3i range() const { return Vec3i(math::Round(mSettings.searchRadius * mDxInv)); }

    inline LeafNodeType* initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        BaseT::initialize(origin, idx, bounds);
        auto& leaf = mManager.leaf(idx);
        mTargetPosition = std::make_unique<PositionHandleT>(leaf.constAttributeArray(mIndices.mPWsIndex));
        return &leaf;
    }

    inline bool startPointLeaf(const typename PointDataTreeT::LeafNodeType& leaf)
    {
        mSourcePosition = std::make_unique<PositionHandleT>(leaf.constAttributeArray(mIndices.mPWsIndex));
#if OPENVDB_PCA_SELF_CONTRIBUTION == 0
        mIsSameLeaf = this->mTargetPosition->array() == this->mSourcePosition->array();
#endif
        return true;
    }

    bool endPointLeaf(const typename PointDataTreeT::LeafNodeType&) { return true; }

protected:
    const AttrIndices& mIndices;
    const PcaSettings& mSettings;
    const Real mDxInv;
    const tree::LeafManager<PointDataTreeT>& mManager;
    std::unique_ptr<PositionHandleT> mTargetPosition;
    std::unique_ptr<PositionHandleT> mSourcePosition;
#if OPENVDB_PCA_SELF_CONTRIBUTION == 0
    bool mIsSameLeaf {false};
#endif
};

template <typename PointDataTreeT>
struct WeightPosSumsTransfer
    : public PcaTransfer<PointDataTreeT>
{
    using BaseT = PcaTransfer<PointDataTreeT>;

    static const Index DIM = PointDataTreeT::LeafNodeType::DIM;
    static const Index LOG2DIM = PointDataTreeT::LeafNodeType::LOG2DIM;

    WeightPosSumsTransfer(const AttrIndices& indices,
                          const PcaSettings& settings,
                          const Real vs,
                          tree::LeafManager<PointDataTreeT>& manager,
                          util::NullInterrupter* interrupt)
        : BaseT(indices, settings, vs, manager, interrupt)
        , mWeights()
        , mWeightedPositions()
        , mCounts() {}

    WeightPosSumsTransfer(const WeightPosSumsTransfer& other)
        : BaseT(other)
        , mWeights()
        , mWeightedPositions()
        , mCounts() {}

    static constexpr size_t GetBatchSize()
    {
        return BaseT::template GetBatchSize<WeightSumT>();
    }

    inline void initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        auto& leaf = (*BaseT::initialize(origin, idx, bounds));
        mWeights = initPcaArrayAttribute<WeightSumT>(leaf, this->mIndices.mWeightSumIndex);
        mWeightedPositions = initPcaArrayAttribute<WeightedPositionSumT>(leaf, this->mIndices.mPosSumIndex);
        // track neighbours
        mCounts.assign(this->mTargetPosition->size(), 0);
    }

    inline void rasterizePoints(const Coord& ijk,
                    const Index start,
                    const Index end,
                    const CoordBBox& bounds)
    {
        constexpr auto N2 = GetBatchSize();
        const Index step = std::max(Index(1), Index((end - start) / this->maxSourcePointsPerVoxel()));

        if constexpr(N2 == 1) {
            // Fallback to per point rasterization
            for (Index i = start; i < end; i += step) {
                this->rasterizePoint(ijk, i, bounds);
            }
        }
        else {
            // Batched/vectorized rasterization. Expect power of two for
            // batched size
            static_assert((N2 > 1) && !(N2 & (N2 - 1)));

            std::array<int64_t, N2> ids;
            Index offset = 0;
            for (Index i = start; i < end; i += step) {
                ids[offset++] = int64_t(i);
                if (offset == N2) {
                    this->rasterizeN2<N2>(ijk, ids, bounds);
                    offset = 0;
                }
            }

            if (offset == 0) return;
            else if (offset == 1) this->rasterizePoint(ijk, Index(ids[0]), bounds);
            else {
                for (; offset < N2; ++offset) ids[offset] = int64_t(-1);
                this->rasterizeN2<N2>(ijk, ids, bounds);
            }
        }
    }

    /// @brief  Single point rasterization
    inline void rasterizePoint(const Coord&,
                    const Index id,
                    const CoordBBox& bounds)
    {
        const Vec3d Pws = math::Vec3<WeightSumT>(this->mSourcePosition->get(id));
        const Vec3d Pis = Pws * this->mDxInv;
        this->stamp(Pws.x(), Pws.y(), Pws.z(), Pis.x(), Pis.y(), Pis.z(), true, bounds);
    }

    bool finalize(const Coord&, size_t idx)
    {
        // Add points to group with counts which are more than the neighbouring threshold
        auto& leaf = this->mManager.leaf(idx);

        {
            // @todo add API to get the array from the group handle. The handle
            //   calls loadData but not expand.
            auto& array = leaf.attributeArray(this->mIndices.mEllipsesGroupIndex.first);
            array.loadData(); // so we can call setUnsafe/getUnsafe
            array.expand();
        }

        points::GroupWriteHandle group(leaf.groupWriteHandle(this->mIndices.mEllipsesGroupIndex));

        const int32_t threshold = int32_t(this->neighbourThreshold());
        for (Index i = 0; i < this->mTargetPosition->size(); ++i) {
            // turn points OFF if they are ON and don't meet max neighbour requirements
            if ((threshold == 0 || (mCounts[i] < threshold)) && group.getUnsafe(i)) {
                group.setUnsafe(i, false);
            }
            if (mCounts[i] <= 0) continue;
            // Normalize weights
            OPENVDB_ASSERT(mWeights[i] >= 0.0f);
            mWeights[i] = 1.0 / mWeights[i];
            mWeightedPositions[i] *= mWeights[i];
        }
        return true;
    }

private:
    template <size_t Size>
    inline void rasterizeN2(const Coord&,
        const std::array<int64_t, GetBatchSize()>& points,
        const CoordBBox& bounds)
    {
        using namespace openvdb::util;

        using SimdT  = typename simd::SimdT<WeightSumT, Size>::Type;
        using SimdIT = typename simd::SimdT<int64_t, Size>::Type;
        using SimdMT = typename simd::SimdTraits<SimdT>::MaskT;

        OPENVDB_ASSERT(points[0] != -1);

        std::array<WeightSumT, 3*Size> cache;
        math::Vec3<WeightSumT> tmp;
        // convert AoS to SoA
        for (size_t i = 0; i < Size; ++i) {
            if (points[i] != -1) {
                tmp = math::Vec3<WeightSumT>(this->mSourcePosition->get(Index(points[i])));
            }
            cache[i+(Size*0)] = tmp[0];
            cache[i+(Size*1)] = tmp[1];
            cache[i+(Size*2)] = tmp[2];
        }

        const SimdT pwx = simd::load<Size>(cache.data() + (Size*0));
        const SimdT pwy = simd::load<Size>(cache.data() + (Size*1));
        const SimdT pwz = simd::load<Size>(cache.data() + (Size*2));
        const SimdT pix = pwx * SimdT(this->mDxInv);
        const SimdT piy = pwy * SimdT(this->mDxInv);
        const SimdT piz = pwz * SimdT(this->mDxInv);
        const SimdIT ids = simd::load<Size>(points.data());
        const SimdMT validMask = ids != SimdIT(-1);

        this->stamp(pwx, pwy, pwz, pix, piy, piz, validMask, bounds);
    }

    template<typename ScalarT, typename MaskT> /// Real or SimdT
    inline void stamp(const ScalarT& Pwx,
                    const ScalarT& Pwy,
                    const ScalarT& Pwz,
                    const ScalarT& Pix,
                    const ScalarT& Piy,
                    const ScalarT& Piz,
                    const MaskT& m,
                    const CoordBBox& intersection)
    {
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pwx)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pwy)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pwz)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pix)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Piy)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Piz)));

        const auto* const data = this->template buffer<0>();
        const auto& mask = *(this->template mask<0>());

        const ScalarT searchRadiusInv(1.0f / this->searchRadius());
        const ScalarT searchRadius2(math::Pow2(this->searchRadius()));
        const Real searchRadiusIS2 = math::Pow2(this->searchRadius() * this->mDxInv);

        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(searchRadiusInv)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(searchRadius2)));
        OPENVDB_ASSERT(std::isfinite(searchRadiusIS2));

        ScalarT tx, ty, tz;

        const Coord& a(intersection.min());
        const Coord& b(intersection.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            const ScalarT minx(c.x() - 0.5);
            const ScalarT maxx(c.x() + 0.5);
            const ScalarT dminx =
                (simd::select(Pix < minx, simd::pow2(Pix - minx),
                    (simd::select(Pix > maxx, simd::pow2(Pix - maxx), ScalarT(0)))));
            if (simd::horizontal_min(dminx) > searchRadiusIS2) continue;
            const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const ScalarT miny(c.y() - 0.5);
                const ScalarT maxy(c.y() + 0.5);
                const ScalarT dminxy = dminx +
                    (simd::select(Piy < miny, simd::pow2(Piy - miny),
                        (simd::select(Piy > maxy, simd::pow2(Piy - maxy), ScalarT(0)))));
                if (simd::horizontal_min(dminxy) > searchRadiusIS2) continue;
                const Index ij = i + ((c.y() & (DIM-1u)) << LOG2DIM);
                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    const Index offset = ij + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(offset)) continue; // next target voxel

                    const ScalarT minz(c.z() - 0.5);
                    const ScalarT maxz(c.z() + 0.5);
                    const ScalarT dminxyz = dminxy +
                        (simd::select(Piz < minz, simd::pow2(Piz - minz),
                            (simd::select(Piz > maxz, simd::pow2(Piz - maxz), ScalarT(0)))));
                    // Does this point's radius overlap the voxel c
                    if (simd::horizontal_min(dminxyz) > searchRadiusIS2) continue;

                    // src point overlaps voxel c
                    const Index targetEnd = data[offset];
                    const Index targetStart = (offset == 0) ? 0 : Index(data[offset - 1]);
                    const Index targetStep =
                        std::max(Index(1), Index((targetEnd - targetStart) / this->maxTargetPointsPerVoxel()));

                    /// @warning  stepping in this way does not guarantee
                    ///   we get a self contribution, could guarantee this
                    ///   by enabling the OPENVDB_PCA_SELF_CONTRIBUTION == 0
                    ///   check and adding it afterwards.
                    for (Index tgtid = targetStart; tgtid < targetEnd; tgtid += targetStep)
                    {
#if OPENVDB_PCA_SELF_CONTRIBUTION == 0
                        if (OPENVDB_UNLIKELY(this->mIsSameLeaf && tgtid == srcid)) continue;
#endif
                        const math::Vec3<WeightSumT> Ptgt(this->mTargetPosition->get(tgtid));
                        // compute length to Ptgt and store in tx
                        tx = Pwx - ScalarT(Ptgt[0]);
                        ty = Pwy - ScalarT(Ptgt[1]);
                        tz = Pwz - ScalarT(Ptgt[2]);
                        tx = simd::pow2(tx) + simd::pow2(ty) + simd::pow2(tz);

                        const auto pmask = ((tx <= searchRadius2) && m);
                        const int c = simd::horizontal_count(pmask);
                        if (c == 0) continue;

                        // some of src point (srcid) reaches target point (tgtid)
                        ScalarT weights = (ScalarT(1.0) - simd::pow3(simd::sqrt(tx) * searchRadiusInv));
                        weights = simd::select(pmask, weights, ScalarT(0.0));
                        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(weights)));
                        OPENVDB_ASSERT(simd::horizontal_min(weights) >= 0.0);
                        OPENVDB_ASSERT(simd::horizontal_max(weights) <= 1.0);

                        const WeightSumT weight = simd::horizontal_add(weights);
                        mWeights[tgtid] += weight;
                        auto& wp = mWeightedPositions[tgtid];
                        wp[0] += simd::horizontal_add(Pwx * weights); // @note: world space position is weighted
                        wp[1] += simd::horizontal_add(Pwy * weights); // @note: world space position is weighted
                        wp[2] += simd::horizontal_add(Pwz * weights); // @note: world space position is weighted
                        mCounts[tgtid] += c;
                    } //point idx
                }
            }
        } // outer loop
    }

private:
    WeightSumT* mWeights;
    WeightedPositionSumT* mWeightedPositions;
    std::vector<int32_t> mCounts;
};

template <typename PointDataTreeT>
struct CovarianceTransfer
    : public PcaTransfer<PointDataTreeT>
{
    using BaseT = PcaTransfer<PointDataTreeT>;

    static const Index DIM = PointDataTreeT::LeafNodeType::DIM;
    static const Index LOG2DIM = PointDataTreeT::LeafNodeType::LOG2DIM;

    CovarianceTransfer(const AttrIndices& indices,
                       const PcaSettings& settings,
                       const Real vs,
                       tree::LeafManager<PointDataTreeT>& manager,
                       util::NullInterrupter* interrupt)
        : BaseT(indices, settings, vs, manager, interrupt)
        , mInclusionGroupHandle()
        , mWeights()
        , mWeightedPositions()
        , mCovMats() {}

    CovarianceTransfer(const CovarianceTransfer& other)
        : BaseT(other)
        , mInclusionGroupHandle()
        , mWeights()
        , mWeightedPositions()
        , mCovMats() {}

    static constexpr size_t GetBatchSize()
    {
        return BaseT::template GetBatchSize<WeightSumT>();
    }

    inline void initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        auto& leaf = (*BaseT::initialize(origin, idx, bounds));
        mInclusionGroupHandle = std::make_unique<points::GroupHandle>(leaf.groupHandle(this->mIndices.mEllipsesGroupIndex));
        mWeights = initPcaArrayAttribute<WeightSumT>(leaf, this->mIndices.mWeightSumIndex);
        mWeightedPositions = initPcaArrayAttribute<WeightedPositionSumT>(leaf, this->mIndices.mPosSumIndex);
        mCovMats = initPcaArrayAttribute<math::Mat3s>(leaf, this->mIndices.mCovMatrixIndex);
    }

    inline void rasterizePoints(const Coord& ijk,
                    const Index start,
                    const Index end,
                    const CoordBBox& bounds)
    {
        constexpr auto N2 = GetBatchSize();
        const Index step = std::max(Index(1), Index((end - start) / this->maxSourcePointsPerVoxel()));

        if constexpr(N2 == 1) {
            // Fallback to per point rasterization
            for (Index i = start; i < end; i += step) {
                this->rasterizePoint(ijk, i, bounds);
            }
        }
        else {
            // Batched/vectorized rasterization. Expect power of two for
            // batched size
            static_assert((N2 > 1) && !(N2 & (N2 - 1)));

            std::array<int64_t, N2> ids;
            Index offset = 0;
            for (Index i = start; i < end; i += step) {
                ids[offset++] = int64_t(i);
                if (offset == N2) {
                    this->rasterizeN2<N2>(ijk, ids, bounds);
                    offset = 0;
                }
            }

            if (offset == 0) return;
            else if (offset == 1) this->rasterizePoint(ijk, Index(ids[0]), bounds);
            else {
                for (; offset < N2; ++offset) ids[offset] = int64_t(-1);
                this->rasterizeN2<N2>(ijk, ids, bounds);
            }
        }
    }

    /// @brief  Single point rasterization
    inline void rasterizePoint(const Coord&,
                    const Index id,
                    const CoordBBox& bounds)
    {
        const Vec3d Pws = math::Vec3<WeightSumT>(this->mSourcePosition->get(id));
        const Vec3d Pis = Pws * this->mDxInv;
        this->stamp(Pws.x(), Pws.y(), Pws.z(), Pis.x(), Pis.y(), Pis.z(), true, bounds);
    }

    bool finalize(const Coord&, size_t) { return true; }

private:
    template <size_t Size>
    inline void rasterizeN2(const Coord&,
        const std::array<int64_t, GetBatchSize()>& points,
        const CoordBBox& bounds)
    {
        using namespace openvdb::util;

        using SimdT  = typename simd::SimdT<WeightSumT, Size>::Type;
        using SimdIT = typename simd::SimdT<int64_t, Size>::Type;
        using SimdMT = typename simd::SimdTraits<SimdT>::MaskT;

        OPENVDB_ASSERT(points[0] != -1);

        std::array<WeightSumT, 3*Size> cache;
        math::Vec3<WeightSumT> tmp;
        // convert AoS to SoA
        for (size_t i = 0; i < Size; ++i) {
            if (points[i] != -1) {
                tmp = math::Vec3<WeightSumT>(this->mSourcePosition->get(Index(points[i])));
            }
            cache[i+(Size*0)] = tmp[0];
            cache[i+(Size*1)] = tmp[1];
            cache[i+(Size*2)] = tmp[2];
        }

        const SimdT pwx = simd::load<Size>(cache.data() + (Size*0));
        const SimdT pwy = simd::load<Size>(cache.data() + (Size*1));
        const SimdT pwz = simd::load<Size>(cache.data() + (Size*2));
        const SimdT pix = pwx * SimdT(this->mDxInv);
        const SimdT piy = pwy * SimdT(this->mDxInv);
        const SimdT piz = pwz * SimdT(this->mDxInv);
        const SimdIT ids = simd::load<Size>(points.data());
        const SimdMT validMask = ids != SimdIT(-1);

        this->stamp(pwx, pwy, pwz, pix, piy, piz, validMask, bounds);
    }

    template<typename ScalarT, typename MaskT> /// Real or SimdT
    inline void stamp(const ScalarT& Pwx,
                    const ScalarT& Pwy,
                    const ScalarT& Pwz,
                    const ScalarT& Pix,
                    const ScalarT& Piy,
                    const ScalarT& Piz,
                    const MaskT& m,
                    const CoordBBox& intersection)
    {
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pwx)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pwy)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pwz)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pix)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Piy)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Piz)));

        const auto* const data = this->template buffer<0>();
        const auto& mask = *(this->template mask<0>());

        const ScalarT searchRadiusInv(1.0f / this->searchRadius());
        const ScalarT searchRadius2(math::Pow2(this->searchRadius()));
        const Real searchRadiusIS2 = math::Pow2(this->searchRadius() * this->mDxInv);

        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(searchRadiusInv)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(searchRadius2)));
        OPENVDB_ASSERT(std::isfinite(searchRadiusIS2));

        ScalarT tx, ty, tz, wx, wy, wz;

        const Coord& a(intersection.min());
        const Coord& b(intersection.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            const ScalarT minx(c.x() - 0.5);
            const ScalarT maxx(c.x() + 0.5);
            const ScalarT dminx =
                (simd::select(Pix < minx, simd::pow2(Pix - minx),
                    (simd::select(Pix > maxx, simd::pow2(Pix - maxx), ScalarT(0)))));
            if (simd::horizontal_min(dminx) > searchRadiusIS2) continue;
            const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const ScalarT miny(c.y() - 0.5);
                const ScalarT maxy(c.y() + 0.5);
                const ScalarT dminxy = dminx +
                    (simd::select(Piy < miny, simd::pow2(Piy - miny),
                        (simd::select(Piy > maxy, simd::pow2(Piy - maxy), ScalarT(0)))));
                if (simd::horizontal_min(dminxy) > searchRadiusIS2) continue;
                const Index ij = i + ((c.y() & (DIM-1u)) << LOG2DIM);
                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    const Index offset = ij + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(offset)) continue; // next target voxel

                    const ScalarT minz(c.z() - 0.5);
                    const ScalarT maxz(c.z() + 0.5);
                    const ScalarT dminxyz = dminxy +
                        (simd::select(Piz < minz, simd::pow2(Piz - minz),
                            (simd::select(Piz > maxz, simd::pow2(Piz - maxz), ScalarT(0)))));
                    // Does this point's radius overlap the voxel c
                    if (simd::horizontal_min(dminxyz) > searchRadiusIS2) continue;

                    const Index targetEnd = data[offset];
                    const Index targetStart = (offset == 0) ? 0 : Index(data[offset - 1]);
                    const Index targetStep =
                        std::max(Index(1), Index((targetEnd - targetStart) / this->maxTargetPointsPerVoxel()));

                    for (Index tgtid = targetStart; tgtid < targetEnd; tgtid += targetStep) {
                        if (!mInclusionGroupHandle->get(tgtid)) continue;
#if OPENVDB_PCA_SELF_CONTRIBUTION == 0
                        if (OPENVDB_UNLIKELY(this->mIsSameLeaf && tgtid == srcid)) continue;
#endif
                        const math::Vec3<WeightSumT> Ptgt(this->mTargetPosition->get(tgtid));
                        tx = Pwx - ScalarT(Ptgt[0]);
                        ty = Pwy - ScalarT(Ptgt[1]);
                        tz = Pwz - ScalarT(Ptgt[2]);
                        tx = simd::pow2(tx) + simd::pow2(ty) + simd::pow2(tz);

                        const MaskT pmask = ((tx <= searchRadius2) && m);
                        if (!simd::horizontal_or(pmask)) continue;

                        // @note  I've observed some performance degradation if
                        //   we don't take copies of the buffers here (aliasing?)
                        const WeightSumT totalWeightInv = mWeights[tgtid];
                        const WeightedPositionSumT currWeightSum = mWeightedPositions[tgtid];

                        // re-compute weight
                        // @note  A gather style approach might be better,
                        //   where each point appends weights/positions into
                        //   a container. We lose some time having to re-
                        //   iterate, but this is far more memory efficient.
                        ScalarT weights = ScalarT(1.0) - simd::pow3(simd::sqrt(tx) * searchRadiusInv);
                        weights = simd::select(pmask, weights, ScalarT(0.0));
                        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(weights)));
                        OPENVDB_ASSERT(simd::horizontal_min(weights) >= 0.0);
                        OPENVDB_ASSERT(simd::horizontal_max(weights) <= 1.0);

                        // posMeanDiff
                        tx = Pwx - ScalarT(currWeightSum[0]);
                        ty = Pwy - ScalarT(currWeightSum[1]);
                        tz = Pwz - ScalarT(currWeightSum[2]);

                        // @note  Could extract the mult by totalWeightInv
                        //   and put it into a loop in finalize() - except
                        //   it would be mat3*float rather than vec*float,
                        //   which would probably better as maxppv increases
                        weights = weights * ScalarT(totalWeightInv);
                        wx = tx * weights;
                        wy = ty * weights;
                        wz = tz * weights;

                        float* const m = mCovMats[tgtid].asPointer();
                        /// @note: equal to:
                        // mat.setCol(0, mat.col(0) + (x * posMeanDiff[0]));
                        // mat.setCol(1, mat.col(1) + (x * posMeanDiff[1]));
                        // mat.setCol(2, mat.col(2) + (x * posMeanDiff[2]));
                        // @todo formalize precision of these methods
                        m[0] += float(simd::horizontal_add(wx * tx));
                        m[1] += float(simd::horizontal_add(wx * ty));
                        m[2] += float(simd::horizontal_add(wx * tz));
                        //
                        m[3] += float(simd::horizontal_add(wy * tx));
                        m[4] += float(simd::horizontal_add(wy * ty));
                        m[5] += float(simd::horizontal_add(wy * tz));
                        //
                        m[6] += float(simd::horizontal_add(wz * tx));
                        m[7] += float(simd::horizontal_add(wz * ty));
                        m[8] += float(simd::horizontal_add(wz * tz));
                    } //point idx
                }
            }
        } // outer sdf voxel
    }

private:
    points::GroupHandle::UniquePtr mInclusionGroupHandle;
    const WeightSumT* mWeights;
    const WeightedPositionSumT* mWeightedPositions;
    math::Mat3s* mCovMats;
    std::vector<int32_t> mCounts;
};

/// @brief Sort a vector into descending order and output a vector of the resulting order
/// @param vector Vector to sort
template <typename Scalar>
inline Vec3i
descendingOrder(math::Vec3<Scalar>& vector)
{
    Vec3i order(0,1,2);
    if (vector[0] < vector[1]) {
        std::swap(vector[0], vector[1]);
        std::swap(order[0], order[1]);
    }
    if (vector[1] < vector[2]) {
        std::swap(vector[1], vector[2]);
        std::swap(order[1], order[2]);
    }
    if (vector[0] < vector[1]) {
        std::swap(vector[0], vector[1]);
        std::swap(order[0], order[1]);
    }
    return order;
}

/// @brief Decomposes a symmetric matrix into its eigenvalues and a rotation matrix of eigenvectors.
///        Note that if mat is positive-definite, this will be equivalent to a singular value
///        decomposition where V = U.
/// @param mat Matrix to decompose
/// @param U rotation matrix.  The order of its columns (which will be eigenvectors) will match
///          the eigenvalues in sigma
/// @param sigma vector of eigenvalues
template <typename Scalar>
inline bool
decomposeSymmetricMatrix(const math::Mat3<Scalar>& mat,
                         math::Mat3<Scalar>& U,
                         math::Vec3<Scalar>& sigma)
{
    math::Mat3<Scalar> Q;
    const bool diagonalized = math::diagonalizeSymmetricMatrix(mat, Q, sigma);

    if (!diagonalized) return false;

    // need to sort eigenvalues and eigenvectors
    Vec3i order = descendingOrder(sigma);

    // we need to re-order the matrix ("Q") columns to match the new eigenvalue order
    // to obtain the correct "U" matrix
    U.setColumns(Q.col(order[0]), Q.col(order[1]), Q.col(order[2]));

    return true;
}

} // namespace pca_internal

/// @endcond

template <typename PointDataGridT>
inline void
pca(PointDataGridT& points,
    const PcaSettings& settings,
    const PcaAttributes& attrs)
{
    static_assert(IsSpecializationOf<PointDataGridT, Grid>::value);
    static constexpr size_t INVALID_IDX = std::numeric_limits<size_t>::max();

    using namespace pca_internal;

    using PointDataTreeT = typename PointDataGridT::TreeType;
    using LeafManagerT = tree::LeafManager<PointDataTreeT>;
    using LeafNodeT = typename PointDataTreeT::LeafNodeType;
    using Vec3T = PcaAttributes::StretchT;
    using Mat3T = PcaAttributes::RotationT;

    auto& tree = points.tree();
    const auto leaf = tree.cbeginLeaf();
    if (!leaf) return;

    // Small lambda to init any one of the necessary ellipsoid attributes
    // @note  Various algorithms here assume that we are responsible for creating
    //   these attributes and so can optimize accordingly. If this changes we'll
    //   need to also change those optimizations (e.g. NullCodec, loadData, etc)
    const auto initAttribute = [&](const std::string& name, const auto val)
    {
        using ValueT = std::decay_t<decltype(val)>;
        if (leaf->hasAttribute(name)) {
            OPENVDB_THROW(KeyError, "PCA attribute '"  << name << "' already exists!");
        }

        points::appendAttribute<ValueT>(tree, name, val);
        return leaf->attributeSet().find(name);
    };

    //

    const size_t pvsIdx = leaf->attributeSet().find("P");
    const auto& xform = points.constTransform();
    const double vs = xform.voxelSize()[0];
    LeafManagerT manager(tree);

    // Configure attributes names
    const auto& descriptor = leaf->attributeSet().descriptor();
    std::vector<std::string> temps {
        descriptor.uniqueName("_weightedpositionsums"),
        descriptor.uniqueName("_inv_weightssum")
    };

    // Separate rotation (Mat3) and stretch (Vec3f) are always required in the
    // below calculations. If we're outputting rotation as a quarternion we
    // must create a temporary covariance matrix attribute. Otherwise we reuse
    // output xform matrix attribute for the intermediate convariance
    // representation. Same applies for the stretch; if we're outputting a
    // combined transform, we must create temporary stretch storage.
    const std::string covAttribName = [&]() {
        if (attrs.xformOutput == PcaAttributes::XformOutput::STRETCH_AND_QUATERNION) {
            temps.emplace_back(descriptor.uniqueName("_covariance"));
            return temps.back();
        }
        return attrs.xform;
    }();

    const std::string stretchAttribName = [&]() {
        if (attrs.xformOutput == PcaAttributes::XformOutput::COMBINED_TRANSFORM)  {
            temps.emplace_back(descriptor.uniqueName("_stretch"));
            return temps.back();
        }
        return attrs.stretch;
    }();

    // 1) Create persisting attributes
    const size_t pwsIdx = initAttribute(attrs.positionWS, zeroVal<PcaAttributes::PosWsT>());
    const size_t rotIdx = initAttribute(covAttribName, zeroVal<Mat3T>());
    const size_t qutIdx =
        attrs.xformOutput == PcaAttributes::XformOutput::STRETCH_AND_QUATERNION ?
            initAttribute(attrs.xform, zeroVal<PcaAttributes::QuatT>()) :
            INVALID_IDX;
    const size_t strIdx = initAttribute(stretchAttribName, PcaAttributes::StretchT(settings.nonAnisotropicStretch));

    // 2) Create temporary attributes
    const size_t posSumIndex = initAttribute(temps[0], zeroVal<WeightedPositionSumT>());
    const size_t weightSumIndex = initAttribute(temps[1], zeroVal<WeightSumT>());

OPENVDB_NO_DEPRECATION_WARNING_BEGIN
    // DEPRECATED, TO REMOVE
    size_t oldRotAttr = INVALID_IDX;
    if (!attrs.rotation.empty()) {
        oldRotAttr = initAttribute(attrs.rotation, zeroVal<Mat3T>());
    }
OPENVDB_NO_DEPRECATION_WARNING_END

    // 3) Create ellipses group
    if (!leaf->attributeSet().descriptor().hasGroup(attrs.ellipses)) {
        points::appendGroup(tree, attrs.ellipses);
        // Include everything by default to start with
        points::setGroup(tree, attrs.ellipses, true);
    }

    // Re-acquire the updated descriptor and get the group idx
    const GroupIndexT ellipsesIdx =
        leaf->attributeSet().descriptor().groupIndex(attrs.ellipses);

    PcaTimer timer(settings.interrupter);

    // 3) Store the world space position on the PDG to speed up subsequent
    //    calculations.
    timer.start("Compute position world spaces");
    manager.foreach([&](LeafNodeT& leafnode, size_t)
    {
        using PvsT = Vec3f;
        using PwsT = PcaAttributes::PosWsT;

        points::AttributeHandle<PvsT> Pvs(leafnode.constAttributeArray(pvsIdx));
        PwsT* Pws = initPcaArrayAttribute<PwsT>(leafnode, pwsIdx, /*fill=*/false);

        for (auto voxel = leafnode.cbeginValueOn(); voxel; ++voxel) {
            const Coord voxelCoord = voxel.getCoord();
            const Vec3d coordVec = voxelCoord.asVec3d();
            for (auto iter = leafnode.beginIndexVoxel(voxelCoord); iter; ++iter) {
                Pws[*iter] = xform.indexToWorld(Pvs.get(*iter) + coordVec);
            }
        }
    });

    timer.stop();
    if (util::wasInterrupted(settings.interrupter)) return;

    AttrIndices indices;
    indices.mPosSumIndex = posSumIndex;
    indices.mWeightSumIndex = weightSumIndex;
    indices.mCovMatrixIndex = rotIdx;
    indices.mPWsIndex = pwsIdx;
    indices.mEllipsesGroupIndex = ellipsesIdx;

    // 4) Init temporary attributes and calculate:
    //        sum_j w_{i,j} * x_j / (sum_j w_j)
    //    And neighbour counts for each point.
    // simultaneously calculates the sum of weighted vector positions (sum w_{i,j} * x_i)
    // weighted against the inverse sum of weights (1.0 / sum w_{i,j}). Also counts number
    // of neighours each point has and updates the ellipses group based on minimum
    // neighbour threshold. Those points which are "included" but which lack sufficient
    // neighbours will be marked as "not included".
    timer.start("Compute position weights");
    {
        WeightPosSumsTransfer<PointDataTreeT>
            transfer(indices, settings, float(vs), manager, settings.interrupter);
        points::rasterize<PointDataGridT, decltype(transfer)> (points, transfer);
    }

    timer.stop();
    if (util::wasInterrupted(settings.interrupter)) return;

    // 5) Principal axes define the rotation matrix of the ellipsoid.
    //    Calculates covariance matrices given weighted sums of positions and
    //    sums of weights per-particle
    timer.start("Compute covariance matrices");
    {
        CovarianceTransfer<PointDataTreeT>
            transfer(indices, settings, float(vs), manager, settings.interrupter);
        points::rasterize<PointDataGridT, decltype(transfer)>(points, transfer);
    }

    timer.stop();
    if (util::wasInterrupted(settings.interrupter)) return;

    // 6) radii stretches are given by the scaled singular values. Decompose
    //    the covariance matrix into its principal axes and their lengths
    timer.start("Decompose covariance matrices");
    manager.foreach([&](LeafNodeT& leafnode, size_t)
    {
        AttributeWriteHandle<Vec3T, NullCodec> stretchHandle(leafnode.attributeArray(strIdx));
        AttributeWriteHandle<Mat3T, NullCodec> rotHandle(leafnode.attributeArray(rotIdx));

        // we don't use a group filter here since we need to set the rotation
        // matrix for excluded points - the handle handles the potential array
        // expansion
        GroupHandle ellipsesGroupHandle(leafnode.groupHandle(ellipsesIdx));

        for (Index idx = 0; idx < stretchHandle.size(); ++idx) {
            if (!ellipsesGroupHandle.get(idx)) {
                rotHandle.set(idx, Mat3T::identity());
                continue;
            }

            // get singular values of the covariance matrix
            Mat3T u;
            Vec3s sigma;
            decomposeSymmetricMatrix(rotHandle.get(idx), u, sigma);

            // fix sigma values, the principal lengths
            auto maxs = sigma[0] * settings.allowedAnisotropyRatio;
            sigma[1] = std::max(sigma[1], maxs);
            sigma[2] = std::max(sigma[2], maxs);

            // should only happen if all neighbours are coincident
            // @note  The specific tolerance here relates to the normalization
            //   of the stetch values in step (7) e.g. s*(1.0/cbrt(s.product())).
            //   math::Tolerance<float>::value is 1e-7f, but we have a bit more
            //   flexibility here, we can deal with smaller values, common for
            //   the case where a point only has one neighbour
            // @todo  have to manually construct the tolerance because
            //   math::Tolerance<Vec3T> resolves to 0.0. fix this in the math lib
            if (math::isApproxZero(sigma, Vec3T(1e-11f))) {
                sigma = Vec3T::ones();
            }

            stretchHandle.set(idx, sigma);
            rotHandle.set(idx, u);
        }
    });
    timer.stop();

    // 7) normalise the principal lengths such that the transformation they
    //    describe 'preserves volume' thus becoming the stretch of the ellipsoids

    // Calculates the average volume change that would occur if applying the
    // transformations defined by the calculate principal axes and their
    // lengths using the determinant of the stretch component of the
    // transformation (given by the sum of the diagonal values) and uses this
    // value to normalise these lengths such that they are relative to the
    // identity. This ensures that the transformation preserves volume.
    timer.start("Normalise the principal lengths");
    manager.foreach([&](LeafNodeT& leafnode, size_t)
    {
        points::GroupFilter filter(ellipsesIdx);
        filter.reset(leafnode);
        AttributeWriteHandle<Vec3T, NullCodec> stretchHandle(leafnode.attributeArray(strIdx));

        for (Index i = 0; i < stretchHandle.size(); ++i)
        {
            if (!filter.valid(&i)) continue;
            const Vec3T stretch = stretchHandle.get(i);
            OPENVDB_ASSERT(stretch != Vec3T::zero());
            const float stretchScale = 1.0f / std::cbrt(stretch.product());
            stretchHandle.set(i, stretchScale * stretch);
        }
    });

    // DEPRECATED, TO REMOVE
    if (oldRotAttr != INVALID_IDX) {
        // Copy xform to "rotation"
        manager.foreach([&](LeafNodeT& leafnode, size_t) {
            // pretty inefficient way to do this, but its deprecated behaviour
            AttributeWriteHandle<Mat3T, NullCodec> oldRotHandle(leafnode.attributeArray(oldRotAttr));
            Mat3T* R = initPcaArrayAttribute<Mat3T>(leafnode, rotIdx, /*fill=*/false);
            for (Index idx = 0; idx < oldRotHandle.size(); ++idx) {
                R[idx] = oldRotHandle.get(idx);
            }
        });
    }

    /// 8) Covert attributes to desired format
    /// @note  We could consolidate this method into the parallel operators above,
    ///   but in the future we may want to do more post-processing with these
    ///   attributes (such as monitor and manipulate volume preservation, etc)
    ///   which will require a separate step. For now, this step is relatively
    ///   cheap.
    timer.start("Coverting attributes");
    if (attrs.xformOutput == PcaAttributes::XformOutput::STRETCH_AND_QUATERNION)
    {
        manager.foreach([&](LeafNodeT& leafnode, size_t) {
            AttributeWriteHandle<Mat3T, NullCodec> rotHandle(leafnode.attributeArray(rotIdx));
            PcaAttributes::QuatT* Q = initPcaArrayAttribute<PcaAttributes::QuatT>(leafnode, qutIdx, /*fill=*/false);
            for (Index idx = 0; idx < rotHandle.size(); ++idx) {
                Mat3T rot = rotHandle.get(idx);
                // Tolerance for quaternion construction - can be pretty low as
                // we assume our mats to be fully unitary (det of +/-1) at this
                // point, but may have precision issues when checking it - also
                // the quat class will try to perform a (mat * mat^t) to check
                // unitary which is suseptible to small fp values causing
                // drifts, so construct with UnsafeConstruct{}
                constexpr float tolerance = 1e-3f;
                // If we're a pure reflection (partial or global, det of -1),
                // we need to convert to a rotation for quat support. Just flip
                // the sign of one of the columns (axis) to change the
                // "handedness" of the basis
                // @todo  Consider never outputing reflection/always preserve
                //   orientation? Handle this in the svd result?
                if (math::isApproxEqual(rot.det(), -1.0f, tolerance))
                {
                    // Choosing to apply to last column
                    rot(0,2) *= -1.0f;
                    rot(1,2) *= -1.0f;
                    rot(2,2) *= -1.0f;
                    // Quat construtor throws so guard with our own check
                    // @todo  Re-write quat class...
                    if (math::isApproxEqual(rot.det(), 1.0f, tolerance)) {
                        Q[idx] = PcaAttributes::QuatT(rot, PcaAttributes::QuatT::UnsafeConstruct{});
                        continue;
                    }
                    OPENVDB_ASSERT_MESSAGE(false,
                        "Unable to convert reflection to valid rotation matrix");
                    // shouldn't be possible but fall back to ident
                    Q[idx].setIdentity();
                }
                else {
                    Q[idx] = PcaAttributes::QuatT(rot, PcaAttributes::QuatT::UnsafeConstruct{});
                }
            }
            // remove matrix representation as we've covered it to a quaternion
            rotHandle.collapse();
        });
    }
    else if (attrs.xformOutput == PcaAttributes::XformOutput::COMBINED_TRANSFORM)
    {
        manager.foreach([&](LeafNodeT& leafnode, size_t) {
            AttributeWriteHandle<Vec3T, NullCodec> stretchHandle(leafnode.attributeArray(strIdx));
            Mat3T* R = initPcaArrayAttribute<Mat3T>(leafnode, rotIdx, /*fill=*/false);
            for (Index idx = 0; idx < stretchHandle.size(); ++idx) {
                R[idx] = R[idx].timesDiagonal(stretchHandle.get(idx));
            }
            // Remove individual scale components which have now been combined
            stretchHandle.collapse();
        });
    }

    timer.stop();
    if (util::wasInterrupted(settings.interrupter)) return;

    /// 9) do laplacian position smoothing here as we have the weights already
    ///   calculated (calculates the smoothed kernel origins as described in
    ///   the paper). averagePositions value biases the smoothed positions
    ///   towards the weighted mean positions. 1.0  will use the weighted means
    ///   while 0.0 will use the original world-space positions
    timer.start("Laplacian smooth positions");
    if (settings.averagePositions > 0.0f)
    {
        manager.foreach([&](LeafNodeT& leafnode, size_t) {
            AttributeWriteHandle<Vec3d, NullCodec> Pws(leafnode.attributeArray(pwsIdx));
            AttributeHandle<WeightedPositionSumT, NullCodec> weightedPosSumHandle(leafnode.constAttributeArray(posSumIndex));

            for (Index i = 0; i < Pws.size(); ++i)
            {
                // @note  Technically possible for the weights to be valid
                //   _and_ zero.
                if (math::isApproxZero(weightedPosSumHandle.get(i),
                    Vec3d(math::Tolerance<double>::value()))) {
                    continue;
                }
                const Vec3d smoothedPosition = (1.0f - settings.averagePositions) *
                    Pws.get(i) + settings.averagePositions * weightedPosSumHandle.get(i);
                Pws.set(i, smoothedPosition);
            }
        });
    }

    timer.stop();

    // Remove temporary attributes
    points::dropAttributes(tree, temps);
}

}
}
}

#endif // OPENVDB_POINTS_POINT_PRINCIPAL_COMPONENT_ANALYSIS_IMPL_HAS_BEEN_INCLUDED
