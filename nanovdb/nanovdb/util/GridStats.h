// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridStats.h

    \author Ken Museth

    \date August 29, 2020

    \brief Re-computes min/max/avg/var/bbox information for each node in a
           pre-existing NanoVDB grid.
*/

#ifndef NANOVDB_GRIDSTATS_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRIDSTATS_H_HAS_BEEN_INCLUDED

#include "../NanoVDB.h"
#include "Range.h"
#include "ForEach.h"

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_reduce.h>
#endif

#include <atomic>
#include <iostream>

namespace nanovdb {

/// @brief Grid flags which indicate what extra information is present in the grid buffer
enum class StatsMode : uint32_t {
    Disable = 0,// disable the computation of any type of statistics (obviously the FASTEST!)
    BBox    = 1,// only compute the bbox of active values per node and total activeVoxelCount
    MinMax  = 2,// additionally compute extrema values
    All     = 3,// compute all of the statics, i.e. bbox, min/max, average and standard deviation
    Default = 3,// default computational mode for statistics
    End     = 4,
};

/// @brief Re-computes the min/max, stats and bbox information for an existing NanoVDB Grid
///
/// @param grid  Grid whose stats to update
/// @param mode  Mode of computation for the statistics.
template<typename BuildT>
void gridStats(NanoGrid<BuildT>& grid, StatsMode mode = StatsMode::Default);

//================================================================================================

template<typename ValueT, int Rank = TensorTraits<ValueT>::Rank>
class Extrema;

/// @brief Template specialization of Extrema on scalar value types, i.e. rank = 0
template<typename ValueT>
class Extrema<ValueT, 0>
{
protected:
    ValueT mMin, mMax;

public:
    using ValueType = ValueT;
    Extrema()
        : mMin(std::numeric_limits<ValueT>::max())
        , mMax(std::numeric_limits<ValueT>::lowest())
    {
    }
    Extrema(const ValueT& v)
        : mMin(v)
        , mMax(v)
    {
    }
    Extrema(const ValueT& a, const ValueT& b)
        : mMin(a)
        , mMax(b)
    {
    }
    Extrema& operator=(const Extrema&) = default;
    Extrema& min(const ValueT& v)
    {
        if (v < mMin) {
            mMin = v;
        }
        return *this;
    }
    Extrema& max(const ValueT& v)
    {
        if (v > mMax) {
            mMax = v;
        }
        return *this;
    }
    Extrema& add(const ValueT& v)
    {
        this->min(v);
        this->max(v);
        return *this;
    }
    Extrema& add(const ValueT& v, uint64_t) { return this->add(v); }
    Extrema& add(const Extrema& other)
    {
        this->min(other.mMin);
        this->max(other.mMax);
        return *this;
    }
    const ValueT& min() const { return mMin; }
    const ValueT& max() const { return mMax; }
    operator bool() const { return mMin <= mMax; }
    static constexpr bool hasMinMax() { return !std::is_same<bool, ValueT>::value; }
    static constexpr bool hasAverage() { return false; }
    static constexpr bool hasStdDeviation() { return false; }
    static constexpr size_t size() { return 0; }
}; // Extrema<T, 0>

/// @brief Template specialization of Extrema on vector value types, i.e. rank = 1
template<typename VecT>
class Extrema<VecT, 1>
{
protected:
    using Real = typename VecT::ValueType; // this works with both nanovdb and openvdb vectors
    struct Pair
    {
        Real scalar;
        VecT vector;

        Pair(Real s)// is only used by Extrema() default c-tor
            : scalar(s)
            , vector(s)
        {
        }
        Pair(const VecT& v)
            : scalar(v.lengthSqr())
            , vector(v)
        {
        }
        Pair& operator=(const Pair&) = default;
        bool  operator<(const Pair& rhs) const { return scalar < rhs.scalar; }
    } mMin, mMax;
    Extrema& add(const Pair& p)
    {
        if (p < mMin) {
            mMin = p;
        }
        if (mMax < p) {
            mMax = p;
        }
        return *this;
    }

public:
    using ValueType = VecT;
    Extrema()
        : mMin(std::numeric_limits<Real>::max())
        , mMax(std::numeric_limits<Real>::lowest())
    {
    }
    Extrema(const VecT& v)
        : mMin(v)
        , mMax(v)
    {
    }
    Extrema(const VecT& a, const VecT& b)
        : mMin(a)
        , mMax(b)
    {
    }
    Extrema& operator=(const Extrema&) = default;
    Extrema& min(const VecT& v)
    {
        Pair tmp(v);
        if (tmp < mMin) {
            mMin = tmp;
        }
        return *this;
    }
    Extrema& max(const VecT& v)
    {
        Pair tmp(v);
        if (mMax < tmp) {
            mMax = tmp;
        }
        return *this;
    }
    Extrema& add(const VecT& v) { return this->add(Pair(v)); }
    Extrema& add(const VecT& v, uint64_t) { return this->add(Pair(v)); }
    Extrema& add(const Extrema& other)
    {
        if (other.mMin < mMin) {
            mMin = other.mMin;
        }
        if (mMax < other.mMax) {
            mMax = other.mMax;
        }
        return *this;
    }
    const VecT& min() const { return mMin.vector; }
    const VecT& max() const { return mMax.vector; }
    operator bool() const { return !(mMax < mMin); }
    static constexpr bool hasMinMax() { return !std::is_same<bool, Real>::value; }
    static constexpr bool hasAverage() { return false; }
    static constexpr bool hasStdDeviation() { return false; }
    static constexpr size_t size() { return 0; }
}; // Extrema<T, 1>

//================================================================================================

template<typename ValueT, int Rank = TensorTraits<ValueT>::Rank>
class Stats;

/// @brief This class computes statistics (minimum value, maximum
/// value, mean, variance and standard deviation) of a population
/// of floating-point values.
///
/// @details variance = Mean[ (X-Mean[X])^2 ] = Mean[X^2] - Mean[X]^2,
///          standard deviation = sqrt(variance)
///
/// @note This class employs incremental computation and double precision.
template<typename ValueT>
class Stats<ValueT, 0> : public Extrema<ValueT, 0>
{
protected:
    using BaseT = Extrema<ValueT, 0>;
    using RealT = double; // for accuracy the internal precission must be 64 bit floats
    size_t mSize;
    double mAvg, mAux;

public:
    using ValueType = ValueT;
    Stats()
        : BaseT()
        , mSize(0)
        , mAvg(0.0)
        , mAux(0.0)
    {
    }
    Stats(const ValueT& val)
        : BaseT(val)
        , mSize(1)
        , mAvg(RealT(val))
        , mAux(0.0)
    {
    }
    /// @brief Add a single sample
    Stats& add(const ValueT& val)
    {
        BaseT::add(val);
        mSize += 1;
        const double delta = double(val) - mAvg;
        mAvg += delta / double(mSize);
        mAux += delta * (double(val) - mAvg);
        return *this;
    }
    /// @brief Add @a n samples with constant value @a val.
    Stats& add(const ValueT& val, uint64_t n)
    {
        const double denom = 1.0 / double(mSize + n);
        const double delta = double(val) - mAvg;
        mAvg += denom * delta * double(n);
        mAux += denom * delta * delta * double(mSize) * double(n);
        BaseT::add(val);
        mSize += n;
        return *this;
    }

    /// Add the samples from the other Stats instance.
    Stats& add(const Stats& other)
    {
        if (other.mSize > 0) {
            const double denom = 1.0 / double(mSize + other.mSize);
            const double delta = other.mAvg - mAvg;
            mAvg += denom * delta * double(other.mSize);
            mAux += other.mAux + denom * delta * delta * double(mSize) * double(other.mSize);
            BaseT::add(other);
            mSize += other.mSize;
        }
        return *this;
    }

    static constexpr bool hasMinMax() { return !std::is_same<bool, ValueT>::value; }
    static constexpr bool hasAverage() { return !std::is_same<bool, ValueT>::value; }
    static constexpr bool hasStdDeviation() { return !std::is_same<bool, ValueT>::value; }

    size_t size() const { return mSize; }

    //@{
    /// Return the  arithmetic mean, i.e. average, value.
    double avg() const { return mAvg; }
    double mean() const { return mAvg; }
    //@}

    //@{
    /// @brief Return the population variance.
    ///
    /// @note The unbiased sample variance = population variance * num/(num-1)
    double var() const { return mSize < 2 ? 0.0 : mAux / double(mSize); }
    double variance() const { return this->var(); }
    //@}

    //@{
    /// @brief Return the standard deviation (=Sqrt(variance)) as
    ///        defined from the (biased) population variance.
    double std() const { return sqrt(this->var()); }
    double stdDev() const { return this->std(); }
    //@}
}; // end Stats<T, 0>

/// @brief This class computes statistics (minimum value, maximum
/// value, mean, variance and standard deviation) of a population
/// of floating-point values.
///
/// @details variance = Mean[ (X-Mean[X])^2 ] = Mean[X^2] - Mean[X]^2,
///          standard deviation = sqrt(variance)
///
/// @note This class employs incremental computation and double precision.
template<typename ValueT>
class Stats<ValueT, 1> : public Extrema<ValueT, 1>
{
protected:
    using BaseT = Extrema<ValueT, 1>;
    using RealT = double; // for accuracy the internal precission must be 64 bit floats
    size_t mSize;
    double mAvg, mAux;

public:
    using ValueType = ValueT;
    Stats()
        : BaseT()
        , mSize(0)
        , mAvg(0.0)
        , mAux(0.0)
    {
    }
    /// @brief Add a single sample
    Stats& add(const ValueT& val)
    {
        typename BaseT::Pair tmp(val);
        BaseT::add(tmp);
        mSize += 1;
        const double delta = tmp.scalar - mAvg;
        mAvg += delta / double(mSize);
        mAux += delta * (tmp.scalar - mAvg);
        return *this;
    }
    /// @brief Add @a n samples with constant value @a val.
    Stats& add(const ValueT& val, uint64_t n)
    {
        typename BaseT::Pair tmp(val);
        const double         denom = 1.0 / double(mSize + n);
        const double         delta = tmp.scalar - mAvg;
        mAvg += denom * delta * double(n);
        mAux += denom * delta * delta * double(mSize) * double(n);
        BaseT::add(tmp);
        mSize += n;
        return *this;
    }

    /// Add the samples from the other Stats instance.
    Stats& add(const Stats& other)
    {
        if (other.mSize > 0) {
            const double denom = 1.0 / double(mSize + other.mSize);
            const double delta = other.mAvg - mAvg;
            mAvg += denom * delta * double(other.mSize);
            mAux += other.mAux + denom * delta * delta * double(mSize) * double(other.mSize);
            BaseT::add(other);
            mSize += other.mSize;
        }
        return *this;
    }

    static constexpr bool hasMinMax() { return !std::is_same<bool, ValueT>::value; }
    static constexpr bool hasAverage() { return !std::is_same<bool, ValueT>::value; }
    static constexpr bool hasStdDeviation() { return !std::is_same<bool, ValueT>::value; }

    size_t size() const { return mSize; }

    //@{
    /// Return the  arithmetic mean, i.e. average, value.
    double avg() const { return mAvg; }
    double mean() const { return mAvg; }
    //@}

    //@{
    /// @brief Return the population variance.
    ///
    /// @note The unbiased sample variance = population variance * num/(num-1)
    double var() const { return mSize < 2 ? 0.0 : mAux / double(mSize); }
    double variance() const { return this->var(); }
    //@}

    //@{
    /// @brief Return the standard deviation (=Sqrt(variance)) as
    ///        defined from the (biased) population variance.
    double std() const { return sqrt(this->var()); }
    double stdDev() const { return this->std(); }
    //@}
}; // end Stats<T, 1>

/// @brief No-op Stats class
template<typename ValueT>
struct NoopStats
{
    using ValueType = ValueT;
    NoopStats() {}
    NoopStats(const ValueT&) {}
    NoopStats& add(const ValueT&) { return *this; }
    NoopStats& add(const ValueT&, uint64_t) { return *this; }
    NoopStats& add(const NoopStats&) { return *this; }
    static constexpr size_t size() { return 0; }
    static constexpr bool hasMinMax() { return false; }
    static constexpr bool hasAverage() { return false; }
    static constexpr bool hasStdDeviation() { return false; }
}; // end NoopStats<T>

//================================================================================================

/// @brief Allows for the construction of NanoVDB grids without any dependecy
template<typename GridT, typename StatsT = Stats<typename GridT::ValueType>>
class GridStats
{
    struct NodeStats;
    using TreeT  = typename GridT::TreeType;
    using ValueT = typename TreeT::ValueType;
    using BuildT = typename TreeT::BuildType;
    using Node0  = typename TreeT::Node0; // leaf
    using Node1  = typename TreeT::Node1; // lower
    using Node2  = typename TreeT::Node2; // upper
    using RootT  = typename TreeT::Node3; // root
    static_assert(std::is_same<ValueT, typename StatsT::ValueType>::value, "Mismatching type");
    static constexpr bool DO_STATS = StatsT::hasMinMax() ||  StatsT::hasAverage() || StatsT::hasStdDeviation();

    ValueT mDelta; // skip node if: node.max < -mDelta || node.min > mDelta

    void process( GridT& );// process grid and all tree nodes
    void process( TreeT& );// process Tree, root node and child nodes
    void process( RootT& );// process root node and child nodes
    NodeStats process( Node0& );// process leaf node

    template<typename NodeT>
    NodeStats process( NodeT& );// process internal node and child nodes

    template<typename DataT, int Rank>
    void setStats(DataT*, const Extrema<ValueT, Rank>&);
    template<typename DataT, int Rank>
    void setStats(DataT*, const Stats<ValueT, Rank>&);
    template<typename DataT>
    void setStats(DataT*, const NoopStats<ValueT>&) {}

    template<typename T, typename FlagT>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    setFlag(const T&, const T&, FlagT& flag) const { flag &= ~FlagT(1); } // unset first bit

    template<typename T, typename FlagT>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    setFlag(const T& min, const T& max, FlagT& flag) const;

public:
    GridStats() = default;

    void operator()(GridT& grid, ValueT delta = ValueT(0));

}; // GridStats

template<typename GridT, typename StatsT>
struct GridStats<GridT, StatsT>::NodeStats
{
    StatsT    stats;
    //uint64_t  activeCount;
    CoordBBox bbox;

    NodeStats(): stats(), bbox() {}//activeCount(0), bbox() {};

    NodeStats& add(const NodeStats &other)
    {
        stats.add( other.stats );// no-op for NoopStats?!
        //activeCount += other.activeCount;
        bbox[0].minComponent(other.bbox[0]);
        bbox[1].maxComponent(other.bbox[1]);
        return *this;
    }
};// GridStats::NodeStats

//================================================================================================

template<typename GridT, typename StatsT>
void GridStats<GridT, StatsT>::operator()(GridT& grid, ValueT delta)
{
    mDelta = delta; // delta = voxel size for level sets, else 0
    this->process( grid );
}

//================================================================================================

template<typename GridT, typename StatsT>
template<typename DataT, int Rank>
inline void GridStats<GridT, StatsT>::
    setStats(DataT* data, const Extrema<ValueT, Rank>& e)
{
    data->setMin(e.min());
    data->setMax(e.max());
}

template<typename GridT, typename StatsT>
template<typename DataT, int Rank>
inline void GridStats<GridT, StatsT>::
    setStats(DataT* data, const Stats<ValueT, Rank>& s)
{
    data->setMin(s.min());
    data->setMax(s.max());
    data->setAvg(s.avg());
    data->setDev(s.std());
}

//================================================================================================

template<typename GridT, typename StatsT>
template<typename T, typename FlagT>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridStats<GridT, StatsT>::
    setFlag(const T& min, const T& max, FlagT& flag) const
{
    if (mDelta > 0 && (min > mDelta || max < -mDelta)) {
        flag |= FlagT(1); //  set 1st bit on to enable rendering
    } else {
        flag &= ~FlagT(1); // set 1st bit off to disable rendering
    }
}

//================================================================================================

template<typename GridT, typename StatsT>
void GridStats<GridT, StatsT>::process( GridT &grid )
{
    this->process( grid.tree() );// this processes tree, root and all nodes

    // set world space AABB
    auto& data = *grid.data();
    const auto& indexBBox = grid.tree().root().bbox();
    if (indexBBox.empty()) {
        data.mWorldBBox = BBox<Vec3R>();
        data.setBBoxOn(false);
    } else {
        // Note that below max is offset by one since CoordBBox.max is inclusive
        // while bbox<Vec3R>.max is exclusive. However, min is inclusive in both
        // CoordBBox and BBox<Vec3R>. This also guarantees that a grid with a single
        // active voxel, does not have an empty world bbox! E.g. if a grid with a
        // unit index-to-world transformation only contains the active voxel (0,0,0)
        // then indeBBox = (0,0,0) -> (0,0,0) and then worldBBox = (0.0, 0.0, 0.0)
        // -> (1.0, 1.0, 1.0). This is a consequence of the different definitions
        // of index and world bounding boxes inherited from OpenVDB!
        const Coord min = indexBBox[0];
        const Coord max = indexBBox[1] + Coord(1);

        auto& worldBBox = data.mWorldBBox;
        const auto& map = grid.map();
        worldBBox[0] = worldBBox[1] = map.applyMap(Vec3d(min[0], min[1], min[2]));
        worldBBox.expand(map.applyMap(Vec3d(min[0], min[1], max[2])));
        worldBBox.expand(map.applyMap(Vec3d(min[0], max[1], min[2])));
        worldBBox.expand(map.applyMap(Vec3d(max[0], min[1], min[2])));
        worldBBox.expand(map.applyMap(Vec3d(max[0], max[1], min[2])));
        worldBBox.expand(map.applyMap(Vec3d(max[0], min[1], max[2])));
        worldBBox.expand(map.applyMap(Vec3d(min[0], max[1], max[2])));
        worldBBox.expand(map.applyMap(Vec3d(max[0], max[1], max[2])));
        data.setBBoxOn(true);
    }

    // set bit flags
    data.setMinMaxOn(StatsT::hasMinMax());
    data.setAverageOn(StatsT::hasAverage());
    data.setStdDeviationOn(StatsT::hasStdDeviation());
} // GridStats::process( Grid )

//================================================================================================

template<typename GridT, typename StatsT>
inline void GridStats<GridT, StatsT>::process( typename GridT::TreeType &tree )
{
    this->process( tree.root() );
}

//================================================================================================

template<typename GridT, typename StatsT>
void GridStats<GridT, StatsT>::process(RootT &root)
{
    using ChildT = Node2;
    auto     &data = *root.data();
    if (data.mTableSize == 0) { // empty root node
        data.mMinimum = data.mMaximum = data.mBackground;
        data.mAverage = data.mStdDevi = 0;
        //data.mActiveVoxelCount = 0;
        data.mBBox = CoordBBox();
    } else {
        NodeStats total;
        for (uint32_t i = 0; i < data.mTableSize; ++i) {
            auto* tile = data.tile(i);
            if (tile->isChild()) { // process child node
                total.add( this->process( *data.getChild(tile) ) );
            } else if (tile->state) { // active tile
                //total.activeCount += ChildT::NUM_VALUES;
                const Coord ijk = tile->origin();
                total.bbox[0].minComponent(ijk);
                total.bbox[1].maxComponent(ijk + Coord(ChildT::DIM - 1));
                if (DO_STATS) { // resolved at compiletime
                    total.stats.add(tile->value, ChildT::NUM_VALUES);
                }
            }
        }
        this->setStats(&data, total.stats);
        if (total.bbox.empty()) {
            std::cerr << "\nWarning: input tree only contained inactive root tiles!"
                      << "\nWhile not strictly an error it's rather suspicious!\n";
        }
        //data.mActiveVoxelCount = total.activeCount;
        data.mBBox = total.bbox;
    }
} // GridStats::process( RootNode )

//================================================================================================

template<typename GridT, typename StatsT>
template<typename NodeT>
typename GridStats<GridT, StatsT>::NodeStats
GridStats<GridT, StatsT>::process(NodeT &node)
{
    static_assert(is_same<NodeT,Node1>::value || is_same<NodeT,Node2>::value, "Incorrect node type");
    using ChildT = typename NodeT::ChildNodeType;

    NodeStats total;
    auto* data = node.data();

    // Serial processing of active tiles
    if (const auto tileCount = data->mValueMask.countOn()) {
        //total.activeCount = tileCount * ChildT::NUM_VALUES; // active tiles
        for (auto it = data->mValueMask.beginOn(); it; ++it) {
            if (DO_STATS) { // resolved at compiletime
                total.stats.add( data->mTable[*it].value, ChildT::NUM_VALUES );
            }
            const Coord ijk = node.offsetToGlobalCoord(*it);
            total.bbox[0].minComponent(ijk);
            total.bbox[1].maxComponent(ijk + Coord(int32_t(ChildT::DIM) - 1));
        }
    }

    // Serial or parallel processing of child nodes
    if (const size_t childCount = data->mChildMask.countOn()) {
#ifndef NANOVDB_USE_TBB
        for (auto it = data->mChildMask.beginOn(); it; ++it) {
            total.add( this->process( *data->getChild(*it) ) );
        }
#else
        std::unique_ptr<ChildT*[]> childNodes(new ChildT*[childCount]);
        ChildT **ptr = childNodes.get();
        for (auto it = data->mChildMask.beginOn(); it; ++it) {
            *ptr++ = data->getChild( *it );
        }
        using RangeT = tbb::blocked_range<size_t>;
        total.add( tbb::parallel_reduce(RangeT(0, childCount), NodeStats(),
            [&](const RangeT &r, NodeStats local)->NodeStats {
                for(size_t i=r.begin(); i!=r.end(); ++i){
                    local.add( this->process( *childNodes[i] ) );
                }
                return local;},
            [](NodeStats a, const NodeStats &b)->NodeStats { return a.add( b ); }
        ));
#endif
    }

    data->mBBox = total.bbox;
    if (total.bbox.empty()) {
        data->mFlags &= ~uint32_t(1); // set 1st bit off since node to disable rendering of node
        data->mFlags &= ~uint32_t(2); // set 2nd bit off since node does not contain active values
    } else {
        data->mFlags |=  uint32_t(2); // set 2nd bit on since node contains active values
        if (DO_STATS) { // resolved at compiletime
            this->setStats(data, total.stats);
            this->setFlag(data->mMinimum, data->mMaximum, data->mFlags);
        }
    }
    return total;
} // GridStats::process( InternalNode )

//================================================================================================

template<typename GridT, typename StatsT>
typename GridStats<GridT, StatsT>::NodeStats
GridStats<GridT, StatsT>::process(Node0 &leaf)
{
    static_assert(Node0::SIZE == 512u, "Invalid size of leaf nodes");
    NodeStats local;
    auto *data = leaf.data();
    if (auto activeCount = data->mValueMask.countOn()) {
        data->mFlags |= uint8_t(2); // sets 2nd bit on since leaf contains active voxel
        //local.activeCount += activeCount;
        leaf.updateBBox(); // optionally update active bounding box
        local.bbox[0] = local.bbox[1] = data->mBBoxMin;
        local.bbox[1] += Coord(data->mBBoxDif[0], data->mBBoxDif[1], data->mBBoxDif[2]);
        if (DO_STATS) { // resolved at compiletime
            for (auto it = data->mValueMask.beginOn(); it; ++it) {
                local.stats.add(data->getValue(*it));
            }
            this->setStats(data, local.stats);
            this->setFlag(data->getMin(), data->getMax(), data->mFlags);
        }
    } else {
        data->mFlags &= ~uint8_t(2); // sets 2nd bit off since leaf does not contain active voxel
    }
    return local;
} // GridStats::process( LeafNode )

//================================================================================================

template<typename BuildT>
void gridStats(NanoGrid<BuildT>& grid, StatsMode mode)
{
    using GridT  = NanoGrid<BuildT>;
    using ValueT = typename GridT::ValueType;
    if (mode == StatsMode::Disable) {
        return;
    } else if (mode == StatsMode::BBox || std::is_same<bool, ValueT>::value) {
        GridStats<GridT, NoopStats<ValueT> > stats;
        stats(grid);
    } else if (mode == StatsMode::MinMax) {
        GridStats<GridT, Extrema<ValueT> > stats;
        stats(grid);
    } else if (mode == StatsMode::All) {
        GridStats<GridT, Stats<ValueT> > stats;
        stats(grid);
    } else {
        throw std::runtime_error("gridStats: Unsupported statistics mode.");
    }
}

//================================================================================================

namespace {

// returns a bitmask (of size 32^3 or 16^3) that marks all the entries
// in a node table which intersects with the specified bounding box.
template<typename NodeT>
Mask<NodeT::LOG2DIM> getBBoxMask(const CoordBBox &bbox, const NodeT* node)
{
    Mask<NodeT::LOG2DIM> mask;// typically 32^3 or 16^3 bit mask
    auto b = CoordBBox::createCube(node->origin(), node->dim());
    assert( bbox.hasOverlap(b) );
    if ( bbox.isInside(b) ) {
        mask.setOn();//node is completely inside the bbox so early out
    } else {
        b.intersect(bbox);// trim bounding box
        // transform bounding box from global to local coordinates
        b.min() &=  NodeT::DIM-1u;
        b.min() >>= NodeT::ChildNodeType::TOTAL;
        b.max() &=  NodeT::DIM-1u;
        b.max() >>= NodeT::ChildNodeType::TOTAL;
        assert( !b.empty() );
        auto it = b.begin();// iterates over all the child nodes or tiles that intersects bbox
        for (const Coord& ijk = *it; it; ++it) {
            mask.setOn(ijk[2] + (ijk[1] << NodeT::LOG2DIM) + (ijk[0] << 2*NodeT::LOG2DIM));
        }
    }
    return mask;
}
}// end of unnamed namespace

/// @brief return the extrema of all the values in a grid that
///        intersects the specified bounding box.
template<typename BuildT>
Extrema<typename NanoGrid<BuildT>::ValueType>
getExtrema(const NanoGrid<BuildT>& grid, const CoordBBox &bbox)
{
    using GridT  = NanoGrid<BuildT>;
    using ValueT = typename GridT::ValueType;
    using TreeT = typename GridTree<GridT>::type;
    using RootT = typename NodeTrait<TreeT, 3>::type;// root node
    using Node2 = typename NodeTrait<TreeT, 2>::type;// upper internal node
    using Node1 = typename NodeTrait<TreeT, 1>::type;// lower internal node
    using Node0 = typename NodeTrait<TreeT, 0>::type;// leaf node

    Extrema<ValueT> extrema;
    const RootT &root = grid.tree().root();
    const auto &bbox3 = root.bbox();
    if (bbox.isInside(bbox3)) {// bbox3 is contained inside bbox
        extrema.min(root.minimum());
        extrema.max(root.maximum());
        extrema.add(root.background());
    } else if (bbox.hasOverlap(bbox3)) {
        const auto *data3 = root.data();
        for (uint32_t i=0; i<data3->mTableSize; ++i) {
            const auto *tile = data3->tile(i);
            CoordBBox bbox2 = CoordBBox::createCube(tile->origin(), Node2::dim());
            if (!bbox.hasOverlap(bbox2)) continue;
            if (tile->isChild()) {
                const Node2 *node2 = data3->getChild(tile);
                if (bbox.isInside(bbox2)) {
                    extrema.min(node2->minimum());
                    extrema.max(node2->maximum());
                } else {// partial intersections at level 2
                    auto *data2 = node2->data();
                    const auto bboxMask2 = getBBoxMask(bbox, node2);
                    for (auto it2 = bboxMask2.beginOn(); it2; ++it2) {
                        if (data2->mChildMask.isOn(*it2)) {
                            const Node1* node1 = data2->getChild(*it2);
                            CoordBBox bbox1 = CoordBBox::createCube(node1->origin(), Node1::dim());
                            if (bbox.isInside(bbox1)) {
                                extrema.min(node1->minimum());
                                extrema.max(node1->maximum());
                            } else {// partial intersection at level 1
                                auto *data1 = node1->data();
                                const auto bboxMask1 = getBBoxMask(bbox, node1);
                                for (auto it1 = bboxMask1.beginOn(); it1; ++it1) {
                                    if (data1->mChildMask.isOn(*it1)) {
                                        const Node0* node0 = data1->getChild(*it1);
                                        CoordBBox bbox0 = CoordBBox::createCube(node0->origin(), Node0::dim());
                                        if (bbox.isInside(bbox0)) {
                                            extrema.min(node0->minimum());
                                            extrema.max(node0->maximum());
                                        } else {
                                            auto *data0 = node0->data();
                                            const auto bboxMask0 = getBBoxMask(bbox, node0);
                                            for (auto it0 = bboxMask0.beginOn(); it0; ++it0) {
                                                extrema.add(data0->getValue(*it0));
                                            }
                                        }
                                    } else {// tile at level 1
                                        extrema.add(data1->mTable[*it1].value);
                                    }
                                }
                            }// end of partial intersection at level 1
                        } else {// tile at level 2
                           extrema.add(data2->mTable[*it2].value);
                        }
                    }// loop over tiles and nodes at level 2
                }// end of partial intersection at level 1
            } else {// tile at root level
                extrema.add(tile->value);
            }
        }// loop over root table
    } else {// bbox does not overlap the grid
        extrema.add(root.background());
    }
    return extrema;
}// getExtrema


} // namespace nanovdb

#endif // NANOVDB_GRIDSTATS_H_HAS_BEEN_INCLUDED
