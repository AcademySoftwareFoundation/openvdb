// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
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

template <typename TreeType,
          typename PositionCodecT,
          typename SourceValueT,
          typename SourceCodecT>
struct TrilinearTransfer : public VolumeTransfer<TreeType>
{
    using BaseT = VolumeTransfer<TreeType>;
    using WeightT = typename TreeType::ValueType;
    using PositionHandleT = points::AttributeHandle<Vec3f, PositionCodecT>;
    using SourceHandleT = points::AttributeHandle<SourceValueT, SourceCodecT>;

    // precision of kernel arithmetic - aliased to the floating point
    // element type of the source attribute.
    using SourceElementT = typename ValueTraits<SourceValueT>::ElementType;
    using RealT = SourceElementT;

    static_assert(std::is_floating_point<SourceElementT>::value,
        "Trilinear rasterization only supports floating point values.");

    static const Index NUM_VALUES = TreeType::LeafNodeType::NUM_VALUES;

    TrilinearTransfer(const size_t pidx,
        const size_t sidx, TreeType& tree)
        : BaseT(tree)
        , mPIdx(pidx)
        , mSIdx(sidx)
        , mPHandle()
        , mSHandle()
        , mWeights() {}

    TrilinearTransfer(const TrilinearTransfer& other)
        : BaseT(other)
        , mPIdx(other.mPIdx)
        , mSIdx(other.mSIdx)
        , mPHandle()
        , mSHandle()
        , mWeights() {}

    //// @note Kernel value evaluator
    static inline RealT value(const RealT x)
    {
        const RealT abs_x = std::fabs(x);
        if (abs_x < RealT(1.0)) return RealT(1.0) - abs_x;
        return RealT(0.0);
    }

    inline static Int32 range() { return 1; }
    inline Int32 range(const Coord&, size_t) const { return this->range(); }

    inline void initialize(const Coord& origin, const size_t idx, const CoordBBox& bounds)
    {
        this->BaseT::initialize(origin, idx, bounds);
        mWeights.fill(openvdb::zeroVal<WeightT>());
    }

    inline bool startPointLeaf(const PointDataTree::LeafNodeType& leaf)
    {
        mPHandle.reset(new PositionHandleT(leaf.constAttributeArray(mPIdx)));
        mSHandle.reset(new SourceHandleT(leaf.constAttributeArray(mSIdx)));
        return true;
    }

    inline bool endPointLeaf(const PointDataTree::LeafNodeType&) { return true; }

protected:
    const size_t mPIdx;
    const size_t mSIdx;
    typename PositionHandleT::UniquePtr mPHandle;
    typename SourceHandleT::UniquePtr mSHandle;
    std::array<WeightT, NUM_VALUES> mWeights;
};

template <typename TreeType,
          typename PositionCodecT,
          typename SourceValueT,
          typename SourceCodecT>
struct StaggeredTransfer :
    public TrilinearTransfer<TreeType, PositionCodecT, SourceValueT, SourceCodecT>
{
    using BaseT = TrilinearTransfer<TreeType, PositionCodecT, SourceValueT, SourceCodecT>;
    using RealT = typename BaseT::RealT;
    using BaseT::value;

    static_assert(VecTraits<typename TreeType::ValueType>::IsVec,
        "Target Tree must be a vector tree for staggered rasterization");

    static const Index DIM = TreeType::LeafNodeType::DIM;
    static const Index LOG2DIM = TreeType::LeafNodeType::LOG2DIM;

    StaggeredTransfer(const size_t pidx,
        const size_t sidx, TreeType& tree)
        : BaseT(pidx, sidx, tree) {}

    void rasterizePoint(const Coord& ijk,
                    const Index id,
                    const CoordBBox& bounds)
    {
        CoordBBox intersectBox(ijk.offsetBy(-1), ijk.offsetBy(1));
        intersectBox.intersect(bounds);
        if (intersectBox.empty()) return;

        auto* const data = this->buffer();
        const auto& mask = *(this->mask());

        const math::Vec3<RealT> P(this->mPHandle->get(id));
        const SourceValueT s(this->mSHandle->get(id));

        math::Vec3<RealT> centerw, macw;

        const Coord& a(intersectBox.min());
        const Coord& b(intersectBox.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            // @todo can probably simplify the double call to value() in some way
            const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            const RealT x = static_cast<RealT>(c.x()-ijk.x()); // distance from ijk to c
            centerw[0] = value(P.x() - x); // center dist
            macw.x() = value(P.x() - (x-RealT(0.5))); // mac dist

            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const Index ij = i + ((c.y() & (DIM-1u)) << LOG2DIM);
                const RealT y = static_cast<RealT>(c.y()-ijk.y());
                centerw[1] = value(P.y() - y);
                macw.y() = value(P.y() - (y-RealT(0.5)));

                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    assert(bounds.isInside(c));
                    const Index offset = ij + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(offset)) continue;
                    const RealT z = static_cast<RealT>(c.z()-ijk.z());
                    centerw[2] = value(P.z() - z);
                    macw.z() = value(P.z() - (z-RealT(0.5)));

                    const math::Vec3<RealT> r {
                        (macw[0] * centerw[1] * centerw[2]),
                        (macw[1] * centerw[0] * centerw[2]),
                        (macw[2] * centerw[0] * centerw[1])
                    };

                    data[offset] += s * r;
                    this->mWeights[offset] += r;
                }
            }
        }
    }

    inline bool finalize(const Coord&, const size_t)
    {
        auto* const data = this->buffer();
        const auto& mask = *(this->mask());

        for (auto iter = mask.beginOn(); iter; ++iter) {
            const Index offset = iter.pos();
            const auto& w = this->mWeights[offset];
            auto& v = data[offset];
            if (!math::isZero(w[0])) v[0] /= w[0];
            if (!math::isZero(w[1])) v[1] /= w[1];
            if (!math::isZero(w[2])) v[2] /= w[2];
        }

        return true;
    }
};

template <typename TreeType,
          typename PositionCodecT,
          typename SourceValueT,
          typename SourceCodecT>
struct CellCenteredTransfer :
    public TrilinearTransfer<TreeType, PositionCodecT, SourceValueT, SourceCodecT>
{
    using BaseT = TrilinearTransfer<TreeType, PositionCodecT, SourceValueT, SourceCodecT>;
    using RealT = typename BaseT::RealT;
    using BaseT::value;

    static const Index DIM = TreeType::LeafNodeType::DIM;
    static const Index LOG2DIM = TreeType::LeafNodeType::LOG2DIM;

    CellCenteredTransfer(const size_t pidx,
        const size_t sidx, TreeType& tree)
        : BaseT(pidx, sidx, tree) {}

    void rasterizePoint(const Coord& ijk,
                    const Index id,
                    const CoordBBox& bounds)
    {
        const Vec3f P(this->mPHandle->get(id));

        // build area of influence depending on point position
        CoordBBox intersectBox(ijk, ijk);
        if (P.x() < 0.0f) intersectBox.min().x() -= 1;
        else              intersectBox.max().x() += 1;
        if (P.y() < 0.0f) intersectBox.min().y() -= 1;
        else              intersectBox.max().y() += 1;
        if (P.z() < 0.0f) intersectBox.min().z() -= 1;
        else              intersectBox.max().z() += 1;
        assert(intersectBox.volume() == 8);

        intersectBox.intersect(bounds);
        if (intersectBox.empty()) return;

        auto* const data = this->buffer();
        const auto& mask = *(this->mask());

        const SourceValueT s(this->mSHandle->get(id));
        math::Vec3<RealT> centerw;

        const Coord& a(intersectBox.min());
        const Coord& b(intersectBox.max());
        for (Coord c = a; c.x() <= b.x(); ++c.x()) {
            const Index i = ((c.x() & (DIM-1u)) << 2*LOG2DIM); // unsigned bit shift mult
            const RealT x = static_cast<RealT>(c.x()-ijk.x()); // distance from ijk to c
            centerw[0] = value(P.x() - x); // center dist

            for (c.y() = a.y(); c.y() <= b.y(); ++c.y()) {
                const Index ij = i + ((c.y() & (DIM-1u)) << LOG2DIM);
                const RealT y = static_cast<RealT>(c.y()-ijk.y());
                centerw[1] = value(P.y() - y);

                for (c.z() = a.z(); c.z() <= b.z(); ++c.z()) {
                    assert(bounds.isInside(c));
                    const Index offset = ij + /*k*/(c.z() & (DIM-1u));
                    if (!mask.isOn(offset)) continue;
                    const RealT z = static_cast<RealT>(c.z()-ijk.z());
                    centerw[2] = value(P.z() - z);

                    assert(centerw[0] >= 0.0f && centerw[0] <= 1.0f);
                    assert(centerw[1] >= 0.0f && centerw[1] <= 1.0f);
                    assert(centerw[2] >= 0.0f && centerw[2] <= 1.0f);

                    const RealT weight = centerw.product();
                    data[offset] += s * weight;
                    this->mWeights[offset] += weight;
                }
            }
        }
    }

    inline bool finalize(const Coord&, const size_t)
    {
        auto* const data = this->buffer();
        const auto& mask = *(this->mask());

        for (auto iter = mask.beginOn(); iter; ++iter) {
            const Index offset = iter.pos();
            const auto& w = this->mWeights[offset];
            auto& v = data[offset];
            if (!math::isZero(w)) v /= w;
        }
        return true;
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
    using TransferT = typename std::conditional<Staggered,
            StaggeredTransfer<TargetTreeT, PositionCodecT, ValueT, CodecT>,
            CellCenteredTransfer<TargetTreeT, PositionCodecT, ValueT, CodecT>
        >::type;

    typename TargetTreeT::Ptr tree(new TargetTreeT);
    if (std::is_same<FilterT, NullFilter>::value) {
        tree->topologyUnion(points);
    }
    else {
        using MaskTreeT = typename PointDataTreeT::template ValueConverter<ValueMask>::Type;
        auto mask = convertPointsToMask<PointDataTreeT, MaskTreeT>(points, filter);
        tree->topologyUnion(*mask);
    }

    TransferT transfer(pidx, sidx, *tree);
    tools::dilateActiveValues(*tree, transfer.range(),
        tools::NN_FACE_EDGE_VERTEX, tools::EXPAND_TILES);

    rasterize<PointDataTreeT, TransferT>(points, transfer, filter);
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
    if (!iter) return typename TargetTreeT::Ptr(new TargetTreeT);

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
