// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @author Nick Avramoussis
///
/// @file PointRasterizeTrilinearImpl.h
///

#ifndef OPENVDB_POINTS_RASTERIZE_TRILINEAR_IMPL_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_RASTERIZE_TRILINEAR_IMPL_HAS_BEEN_INCLUDED

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

/// @cond OPENVDB_DOCS_INTERNAL

namespace rasterize_trilinear_internal {

template <bool _Staggered,
    typename TreeT,
    typename PositionCodecT,
    typename SourceValueT,
    typename SourceCodecT,
    typename FilterT>
struct TArgs
{
    static constexpr bool Staggered = _Staggered;
    using TreeType = TreeT;
    using PositionCodecType = PositionCodecT;
    using SourceValueType = SourceValueT;
    using SourceCodecType = SourceCodecT;
    using FilterType = FilterT;
};

//// @note Kernel value evaluator
template <typename ScalarT>
static inline ScalarT value(const ScalarT x)
{
    const ScalarT abs_x = simd::abs(x);
    const ScalarT r0 = ScalarT(1.0) - abs_x;
    return simd::select(abs_x < ScalarT(1.0), r0, ScalarT(0.0));
}

template <typename DerivedT, typename TArgsT>
struct TrilinearTransfer :
    public VolumeTransfer<typename TArgsT::TreeType>,
    public FilteredTransfer<typename TArgsT::FilterType>
{
    using BaseT = VolumeTransfer<typename TArgsT::TreeType>;
    using FilterType = typename TArgsT::FilterType;
    using FilterTransferT = FilteredTransfer<FilterType>;
    using SourceValueT = typename TArgsT::SourceValueType;

    using PositionHandleT = points::AttributeHandle<Vec3f, typename TArgsT::PositionCodecType>;
    using SourceHandleT = points::AttributeHandle<SourceValueT, typename TArgsT::SourceCodecType>;

    using SourceElementT = typename ValueTraits<SourceValueT>::ElementType;
    static const size_t SourceSize = ValueTraits<SourceValueT>::Elements;

    // Destination tree expected to always be at floating point precision. Its
    // floating point type defines the precision of the kernel arithmetic
    using RealT = typename ValueTraits<typename TArgsT::TreeType::ValueType>::ElementType;
    static_assert(std::is_floating_point_v<RealT>);
    using NativeT = typename simd::NativeSimdOrScalar<RealT>::Type;

    // Per extent weight for staggered trilinear interpolation, only ever one
    // weight required for cell centered
    static constexpr size_t kNumWeights = TArgsT::TreeType::LeafNodeType::NUM_VALUES *
        (TArgsT::Staggered ? 3 : 1);

    TrilinearTransfer(const size_t pidx,
        const size_t sidx,
        const FilterType& filter,
        typename TArgsT::TreeType& tree)
        : BaseT(tree)
        , FilterTransferT(filter)
        , mPIdx(pidx)
        , mSIdx(sidx)
        , mPHandle()
        , mSHandle()
        , mWeights() {}

    TrilinearTransfer(const TrilinearTransfer& other)
        : BaseT(other)
        , FilterTransferT(other)
        , mPIdx(other.mPIdx)
        , mSIdx(other.mSIdx)
        , mPHandle()
        , mSHandle()
        , mWeights() {}

    inline static Int32 range() { return 1; }

    inline Int32 range(const Coord&, size_t) const { return this->range(); }

    inline void initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        this->BaseT::initialize(origin, idx, bounds);
        this->FilterTransferT::initialize(origin, idx, bounds);
        mWeights.fill(NativeT(0));
    }

    inline bool startPointLeaf(const PointDataTree::LeafNodeType& leaf)
    {
        this->FilterTransferT::startPointLeaf(leaf);
        mPHandle = std::make_unique<PositionHandleT>(leaf.constAttributeArray(mPIdx));
        mSHandle = std::make_unique<SourceHandleT>(leaf.constAttributeArray(mSIdx));
        return true;
    }

    /// @brief  Multi point rasterization
    /// @note   This is the only allowed entry point from the transfer schemes
    inline void rasterizePoints(const Coord& ijk,
                    const Index start,
                    const Index end,
                    const CoordBBox& bounds)
    {
        constexpr auto N2 = simd::SimdTraits<NativeT>::size;
        if constexpr(N2 == 1) {
            // Fallback to per point rasterization
            for (Index i = start; i < end; ++i) {
                if (!FilterTransferT::filter(i)) continue;
                this->rasterizePoint(ijk, i, bounds);
            }
        }
        else {
            // Batched/vectorized rasterization. Expect power of two for
            // batched size
            static_assert((N2 > 1) && !(N2 & (N2 - 1)));
            std::array<int64_t, N2> ids;
            Index offset = 0;
            for (Index i = start; i < end; ++i) {
                if (!FilterTransferT::filter(i)) continue;
                ids[offset++] = int64_t(i);
                if (offset == N2) {
                    this->rasterizeN2<N2>(ijk, ids, bounds);
                    offset = 0;
                }
            }
            if (offset == 0) return;
            else {
                for (; offset < N2; ++offset) ids[offset] = int64_t(-1);
                this->rasterizeN2<N2>(ijk, ids, bounds);
            }
        }
    }

    inline bool endPointLeaf(const PointDataTree::LeafNodeType&) { return true; }

    inline bool finalize(const Coord&, const size_t)
    {
        auto* const data = this->buffer();
        const auto& mask = *(this->mask());

        for (auto iter = mask.beginOn(); iter; ++iter) {
            const Index offset = iter.pos();
            auto& v = data[offset];

            if constexpr (TArgsT::Staggered) {
                const auto* w = &(this->mWeights[offset*3]);
                auto w0 = simd::horizontal_add(w[0]);
                if (!math::isZero(w0)) v[0] /= w0;
                auto w1 = simd::horizontal_add(w[1]);
                if (!math::isZero(w1)) v[1] /= w1;;
                auto w2 = simd::horizontal_add(w[2]);
                if (!math::isZero(w2)) v[2] /= w2;
            }
            else {
                const auto* w = &(this->mWeights[offset]);
                auto w0 = simd::horizontal_add(*w);
                if (math::isZero(w0)) continue;
                if constexpr (SourceSize == 1) v /= w0;
                else {
                    for (size_t j = 0; j < SourceSize; ++j) {
                        v.asPointer()[j] /= w0;
                    }
                }
            }
        }

        return true;
    }

private:
    /// @brief  Single point rasterization
    inline void rasterizePoint(const Coord& ijk,
                    const Index id,
                    const CoordBBox& bounds)
    {
        const math::Vec3<RealT> P(this->mPHandle->get(id));

        CoordBBox intersection = this->derived().intersection(P.x(), P.y(), P.z(), ijk);
        intersection.intersect(bounds);
        if (intersection.empty()) return;

        const auto S(this->mSHandle->get(id));
        if constexpr(SourceSize == 1) {
            this->derived().template stamp<RealT>(P.x(), P.y(), P.z(), &S, ijk, intersection);
        }
        else {
            this->derived().template stamp<RealT>(P.x(), P.y(), P.z(), S.asPointer(), ijk, intersection);
        }
    }

    template <size_t N2>
    inline void rasterizeN2(const Coord& ijk,
        const std::array<int64_t, N2>& points,
        const CoordBBox& bounds)
    {
        static_assert((N2 > 1) && !(N2 & (N2 - 1)));
        using SimdT  = typename simd::SimdT<RealT, N2>::Type;

        OPENVDB_ASSERT(points[0] != -1);

        std::array<RealT, std::max(size_t(3), SourceSize)*N2> cache;
        math::Vec3<RealT> tmp;
        // convert AoS to SoA
        for (size_t i = 0; i < N2; ++i) {
            if (points[i] != -1) {
                tmp = math::Vec3<RealT>(this->mPHandle->get(Index(points[i])));
            }
            else {
                // For positions, if the point is invalid, set it to a value
                // which corresponds to a voxel space position far enough away
                // from the kernel range such that all weights evaluate to 0.
                // Arbitrarily gone with the value of 10
                // @todo  For cell cnetered rasterization, we should offset in
                //   the direction of the last position to try and avoid zero
                //   crossings which icnrease the stencil lookup size.
                tmp.init(RealT(10.0), RealT(10.0), RealT(10.0));
            }
            cache[i+(N2*0)] = tmp[0];
            cache[i+(N2*1)] = tmp[1];
            cache[i+(N2*2)] = tmp[2];
        }

        const SimdT Px = simd::load<N2>(cache.data() + (N2*0));
        const SimdT Py = simd::load<N2>(cache.data() + (N2*1));
        const SimdT Pz = simd::load<N2>(cache.data() + (N2*2));

        CoordBBox intersection = this->derived().intersection(Px, Py, Pz, ijk);
        intersection.intersect(bounds);
        if (intersection.empty()) return;

#if defined(__GNUC__)  && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
        SourceValueT tmps;
        for (size_t i = 0; i < N2; ++i) {
            if (points[i] != -1) {
                tmps = this->mSHandle->get(Index(points[i]));
            }
            if constexpr (SourceSize == 1) {
                cache[i+(N2*0)] = RealT(tmps);
            }
            else {
                for (size_t j = 0; j < SourceSize; ++j) {
                    cache[i+(N2*j)] = RealT(tmps.asPointer()[j]);
                }
            }
        }
#if defined(__GNUC__)  && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

        SimdT S[TArgsT::Staggered ? 3 : SourceSize];
        for (size_t j = 0; j < SourceSize; ++j) {
            S[j] = simd::load<N2>(cache.data() + (N2*j));
        }
        if constexpr (SourceSize == 1 && TArgsT::Staggered) {
            S[1] = S[0];
            S[2] = S[0];
        }
        this->derived().stamp(Px, Py, Pz, S, ijk, intersection);
    }

    inline DerivedT& derived() {
        return *(static_cast<DerivedT*>(this));
    }

protected:
    const size_t mPIdx;
    const size_t mSIdx;
    typename PositionHandleT::UniquePtr mPHandle;
    typename SourceHandleT::UniquePtr mSHandle;
    std::array<NativeT, kNumWeights> mWeights;
};

template <typename TArgsT>
struct StaggeredTransfer :
    public TrilinearTransfer<StaggeredTransfer<TArgsT>, TArgsT>
{
    using BaseT = TrilinearTransfer<StaggeredTransfer<TArgsT>, TArgsT>;
    using RealT = typename BaseT::RealT;

    static_assert(VecTraits<typename TArgsT::TreeType::ValueType>::IsVec &&
        VecTraits<typename TArgsT::TreeType::ValueType>::Size == 3,
        "Target Tree must be a Vec3 tree for staggered rasterization");

    static const Index DIM = TArgsT::TreeType::LeafNodeType::DIM;
    static const Index LOG2DIM = TArgsT::TreeType::LeafNodeType::LOG2DIM;

    StaggeredTransfer(const size_t pidx,
        const size_t sidx,
        const typename TArgsT::FilterType& filter,
        typename TArgsT::TreeType& tree)
        : BaseT(pidx, sidx, filter, tree) {}

private:
    friend BaseT;

    template<typename ScalarT> /// RealT or SimdT
    inline CoordBBox intersection(
        const ScalarT&,
        const ScalarT&,
        const ScalarT&,
        const Coord& ijk)
    {
        // Stencil size is always 27
        return CoordBBox(ijk.offsetBy(-1), ijk.offsetBy(1));
    }

    template<typename ScalarT> /// RealT or SimdT
    inline void stamp(const ScalarT& Px,
                    const ScalarT& Py,
                    const ScalarT& Pz,
                    const ScalarT* S,
                    const Coord& ijk,
                    const CoordBBox& intersection)
    {
        // Some of the arithmetic in this function assumes these are the same,
        // otherwise we can end up writing scalars to all lanes of a simdt
        // (e.g. wp[0] += (Pwx * weights), where wp is a simdt and Pwx is
        // a scalar - this ends up filing all the lanes).
        static_assert(simd::IsSimdT<ScalarT>::value == simd::IsSimdT<typename BaseT::NativeT>::value);
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Px)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Py)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pz)));

        auto* const data = this->buffer();
        const auto& mask = *(this->mask());

        ScalarT cwx, cwy, cwz, cwxy; // center weights
        ScalarT mwx, mwxcy, mwycx, mwz; // mac weights
        ScalarT result;
        int32_t xoffset, xyoffset, xyzoffset; // voxel offsets

        const Coord& a(intersection.min());
        const Coord& b(intersection.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            xoffset = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            // @todo can probably simplify the double call to value() in
            //   some way, haven't spent too much time attempting too
            const ScalarT x(static_cast<RealT>(c.x()-ijk.x())); // distance from ijk to c
            cwx = value(Px - x); // center dist
            mwx = value(Px - (x - ScalarT(RealT(0.5)))); // mac dist

            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                xyoffset = xoffset + /*j*/((c.y() & (DIM-1u)) << LOG2DIM);
                const ScalarT y(static_cast<RealT>(c.y()-ijk.y()));
                cwy  = value(Py - y);
                cwxy = cwx * cwy;
                mwycx = cwx * value(Py - (y - ScalarT(RealT(0.5))));
                mwxcy = mwx * cwy;

                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    OPENVDB_ASSERT(intersection.isInside(c));
                    xyzoffset = xyoffset + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(xyzoffset)) continue;
                    auto& v = data[xyzoffset]; // Must be a Vec3
                    auto* w = &(this->mWeights[xyzoffset * 3]);

                    const ScalarT z(static_cast<RealT>(c.z()-ijk.z()));
                    cwz = value(Pz - z);
                    mwz = value(Pz - (z - ScalarT(RealT(0.5))));

                    // @todo  Could remove the last 3 reductions here with another
                    //   cached array/vector of weighted values
                    // x
                    result = mwxcy * cwz;
                    OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(result)));
                    OPENVDB_ASSERT(simd::horizontal_min(result) >= 0.0);
                    OPENVDB_ASSERT(simd::horizontal_max(result) <= 1.0);
                    w[0] += result;
                    v[0] += simd::horizontal_add(S[0] * result);

                    // y
                    result = mwycx * cwz;
                    OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(result)));
                    OPENVDB_ASSERT(simd::horizontal_min(result) >= 0.0);
                    OPENVDB_ASSERT(simd::horizontal_max(result) <= 1.0);
                    w[1] += result;
                    v[1] += simd::horizontal_add(S[1] * result);

                    // z
                    result = cwxy * mwz;
                    OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(result)));
                    OPENVDB_ASSERT(simd::horizontal_min(result) >= 0.0);
                    OPENVDB_ASSERT(simd::horizontal_max(result) <= 1.0);
                    w[2] += result;
                    v[2] += simd::horizontal_add(S[2] * result);
                }
            }
        }
    }
};

template <typename TArgsT>
struct CellCenteredTransfer :
    public TrilinearTransfer<CellCenteredTransfer<TArgsT>, TArgsT>
{
    using BaseT = TrilinearTransfer<CellCenteredTransfer<TArgsT>, TArgsT>;
    using RealT = typename BaseT::RealT;

    static const Index DIM = TArgsT::TreeType::LeafNodeType::DIM;
    static const Index LOG2DIM = TArgsT::TreeType::LeafNodeType::LOG2DIM;

    CellCenteredTransfer(const size_t pidx,
        const size_t sidx,
        const typename TArgsT::FilterType& filter,
        typename TArgsT::TreeType& tree)
        : BaseT(pidx, sidx, filter, tree) {}

private:
    friend BaseT;

    template<typename ScalarT> /// RealT or SimdT
    inline CoordBBox intersection(
        const ScalarT& Px,
        const ScalarT& Py,
        const ScalarT& Pz,
        const Coord& ijk)
    {
        // Build area of influence depending on point position
        // @note  We should only movemask once on a single mask but VCL doesn't
        //   expose a nice portable way to deal with that. We should introduce
        //   some kind of zero crossing API method in the simd namespace i.e:
        //     auto m = movemask(P < 0.0);
        //     m == 0x0 // all positive
        //     m == 0xF // all negative
        //     // else mixed
        CoordBBox intersectBox(ijk, ijk);
        if (simd::horizontal_or(Px <  ScalarT(0.0))) intersectBox.min().x() -= 1;
        if (simd::horizontal_or(Px >= ScalarT(0.0))) intersectBox.max().x() += 1;
        if (simd::horizontal_or(Py <  ScalarT(0.0))) intersectBox.min().y() -= 1;
        if (simd::horizontal_or(Py >= ScalarT(0.0))) intersectBox.max().y() += 1;
        if (simd::horizontal_or(Pz <  ScalarT(0.0))) intersectBox.min().z() -= 1;
        if (simd::horizontal_or(Pz >= ScalarT(0.0))) intersectBox.max().z() += 1;
        return intersectBox;
    }

    template<typename ScalarT> /// RealT or SimdT
    inline void stamp(const ScalarT& Px,
                    const ScalarT& Py,
                    const ScalarT& Pz,
                    const ScalarT* S,
                    const Coord& ijk,
                    const CoordBBox& intersection)
    {
        // Some of the arithmetic in this function assumes these are the same,
        // otherwise we can end up writing scalars to all lanes of a simdt
        // (e.g. wp[0] += (Pwx * weights), where wp is a simdt and Pwx is
        // a scalar - this ends up filing all the lanes).
        static_assert(simd::IsSimdT<ScalarT>::value == simd::IsSimdT<typename BaseT::NativeT>::value);
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Px)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Py)));
        OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(Pz)));

        auto* const data = this->buffer();
        const auto& mask = *(this->mask());

        ScalarT cwx, cwxy, weight; // center weights
        int32_t xoffset, xyoffset, xyzoffset; // voxel offsets

        const Coord& a(intersection.min());
        const Coord& b(intersection.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            // @todo can probably simplify the double call to value() in
            //   some way, haven't spent too much time attempting too
            const ScalarT x(static_cast<RealT>(c.x()-ijk.x())); // distance from ijk to c
            cwx = value(Px - x); // center dist
            xoffset = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult

            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const ScalarT y(static_cast<RealT>(c.y()-ijk.y()));
                cwxy = cwx * value(Py - y);
                xyoffset = xoffset + /*j*/((c.y() & (DIM-1u)) << LOG2DIM);

                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    OPENVDB_ASSERT(intersection.isInside(c));
                    xyzoffset = xyoffset + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(xyzoffset)) continue;
                    auto& v = data[xyzoffset];
                    auto& w = this->mWeights[xyzoffset];

                    const ScalarT z(static_cast<RealT>(c.z()-ijk.z()));
                    weight = cwxy * value(Pz - z);
                    OPENVDB_ASSERT(simd::horizontal_and(simd::is_finite(weight)));
                    OPENVDB_ASSERT(simd::horizontal_min(weight) >= 0.0);
                    OPENVDB_ASSERT(simd::horizontal_max(weight) <= 1.0);

                    if constexpr (BaseT::SourceSize == 1) {
                        v += simd::horizontal_add(S[0] * weight);
                    }
                    else {
                        for (size_t j = 0; j < BaseT::SourceSize; ++j) {
                            v.asPointer()[j] += simd::horizontal_add(S[j] * weight);
                        }
                    }
                    w += weight;
                }
            }
        }
    }
};

// @note  If building with MSVC we have to use auto to deduce the return type
//   due to a compiler bug. We can also use that for the public API - but
//   we explicitly define it in non-msvc builds to ensure the API remains
//   consistent
template <bool Staggered,
    typename ValueT,
    typename CodecT,
    typename PositionCodecT,
    typename FilterT,
    typename PointDataTreeT>
inline
#ifndef _MSC_VER
typename TrilinearTraits<ValueT, Staggered>::template TreeT<PointDataTreeT>::Ptr
#else
auto
#endif
rasterizeTrilinear(const PointDataTreeT& points,
           const size_t pidx,
           const size_t sidx,
           const FilterT& filter)
{
    using TraitsT = TrilinearTraits<ValueT, Staggered>;
    using TargetTreeT = typename TraitsT::template TreeT<PointDataTree>;
    using TArgsT = TArgs<Staggered, TargetTreeT, PositionCodecT, ValueT, CodecT, FilterT>;
    using TransferT = typename std::conditional<Staggered,
            StaggeredTransfer<TArgsT>,
            CellCenteredTransfer<TArgsT>
        >::type;

    typename TargetTreeT::Ptr tree = std::make_shared<TargetTreeT>();
    if constexpr (std::is_same_v<FilterT, NullFilter>) {
        tree->topologyUnion(points);
    }
    else {
        using MaskTreeT = typename PointDataTreeT::template ValueConverter<ValueMask>::Type;
        auto mask = convertPointsToMask<PointDataTreeT, MaskTreeT>(points, filter);
        tree->topologyUnion(*mask);
    }

    TransferT transfer(pidx, sidx, filter, *tree);
    tools::dilateActiveValues(*tree, transfer.range(),
        tools::NN_FACE_EDGE_VERTEX, tools::EXPAND_TILES);

    rasterize<PointDataTreeT, TransferT>(points, transfer);
    return tree;
}

} // namespace rasterize_trilinear_internal

/// @endcond

///////////////////////////////////////////////////

template <bool Staggered,
    typename ValueT,
    typename FilterT,
    typename PointDataTreeT>
inline auto
rasterizeTrilinear(const PointDataTreeT& points,
           const std::string& attribute,
           const FilterT& filter)
{
    using TraitsT = TrilinearTraits<ValueT, Staggered>;
    using TargetTreeT = typename TraitsT::template TreeT<PointDataTree>;

    const auto iter = points.cbeginLeaf();
    if (!iter) return std::make_shared<TargetTreeT>();

    const AttributeSet::Descriptor& descriptor = iter->attributeSet().descriptor();
    const size_t pidx = descriptor.find("P");
    const size_t sidx = descriptor.find(attribute);
    if (pidx == AttributeSet::INVALID_POS) {
        OPENVDB_THROW(RuntimeError, "Failed to find position attribute");
    }
    if (sidx == AttributeSet::INVALID_POS) {
        OPENVDB_THROW(RuntimeError, "Failed to find source attribute");
    }

    const NamePair& ptype = descriptor.type(pidx);
    const NamePair& stype = descriptor.type(sidx);
    if (ptype.second == NullCodec::name()) {
        if (stype.second == NullCodec::name()) {
            return rasterize_trilinear_internal::rasterizeTrilinear
                <Staggered, ValueT, NullCodec, NullCodec>
                    (points, pidx, sidx, filter);
        }
        else {
            return rasterize_trilinear_internal::rasterizeTrilinear
                <Staggered, ValueT, UnknownCodec, NullCodec>
                    (points, pidx, sidx, filter);
        }
    }
    else {
        if (stype.second == NullCodec::name()) {
            return rasterize_trilinear_internal::rasterizeTrilinear
                <Staggered, ValueT, NullCodec, UnknownCodec>
                    (points, pidx, sidx, filter);
        }
        else {
            return rasterize_trilinear_internal::rasterizeTrilinear
                <Staggered, ValueT, UnknownCodec, UnknownCodec>
                    (points, pidx, sidx, filter);
        }
    }
}


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif //OPENVDB_POINTS_RASTERIZE_TRILINEAR_IMPL_HAS_BEEN_INCLUDED
