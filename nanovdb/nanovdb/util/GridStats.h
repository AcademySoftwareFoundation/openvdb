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

#include <atomic>

namespace nanovdb {

/// @brief Grid flags which indicate what extra information is present in the grid buffer.
enum class StatsMode : uint32_t {
    Disable = 0,// disable the computation of any type of statistics (obviously the FASTEST!)
    BBox = 1,// only compute the bbox of active values per node and total activeVoxelCount
    MinMax = 2,// additionally compute extrema values
    All = 3,// compute all of the statics, i.e. bbox, min/max, averager and standard deviation
    Default = 3,
    End = 4,
};

/// @brief Re-computes the min/max and bbox information for an existing NaoVDB Grid
///
/// @param grid  Grid whoes stats to update
/// @param mode  Mode of computation for the statistics.
template<typename ValueT>
void gridStats(NanoGrid<ValueT>& grid, StatsMode mode = StatsMode::Default);

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
    void min(const ValueT& v)
    {
        if (v < mMin) {
            mMin = v;
        }
    }
    void max(const ValueT& v)
    {
        if (v > mMax) {
            mMax = v;
        }
    }
    void add(const ValueT& v)
    {
        this->min(v);
        this->max(v);
    }
    void add(const ValueT& v, uint64_t) { this->add(v); }
    void add(const Extrema& other)
    {
        this->min(other.mMin);
        this->max(other.mMax);
    }
    const ValueT& min() const { return mMin; }
    const ValueT& max() const { return mMax; }
    operator bool() const { return mMin <= mMax; }
    static constexpr bool hasMinMax() { return true; }
    static constexpr bool hasAverage() { return false; }
    static constexpr bool hasStdDeviation() { return false; }
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
        Pair(Real s)
            : scalar(s)
            , vector(s)
        {
        } // is only used by Extrema() default c-tor
        Pair(const VecT& v)
            : scalar(v.lengthSqr())
            , vector(v)
        {
        }
        Pair& operator=(const Pair&) = default;
        bool  operator<(const Pair& rhs) const { return scalar < rhs.scalar; }
        bool  operator>(const Pair& rhs) const { return scalar > rhs.scalar; }
        bool  operator<=(const Pair& rhs) const { return scalar <= rhs.scalar; }
    } mMin, mMax;
    void add(const Pair& p)
    {
        if (p < mMin) {
            mMin = p;
        }
        if (p > mMax) {
            mMax = p;
        }
    }

public:
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
    void     min(const VecT& v)
    {
        Pair tmp(v);
        if (tmp < mMin) {
            mMin = tmp;
        }
    }
    void max(const VecT& v)
    {
        Pair tmp(v);
        if (tmp > mMax) {
            mMax = tmp;
        }
    }
    void add(const VecT& v) { this->add(Pair(v)); }
    void add(const VecT& v, uint64_t) { this->add(Pair(v)); }
    void add(const Extrema& other)
    {
        if (other.mMin < mMin)
            mMin = other.mMin;
        if (other.mMax > mMax)
            mMax = other.mMax;
    }
    const VecT& min() const { return mMin.vector; }
    const VecT& max() const { return mMax.vector; }
    operator bool() const { return mMin <= mMax; }
    static constexpr bool hasMinMax() { return true; }
    static constexpr bool hasAverage() { return false; }
    static constexpr bool hasStdDeviation() { return false; }
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
    void add(const ValueT& val)
    {
        BaseT::add(val);
        mSize += 1;
        const double delta = double(val) - mAvg;
        mAvg += delta / double(mSize);
        mAux += delta * (double(val) - mAvg);
    }
    /// @brief Add @a n samples with constant value @a val.
    void add(const ValueT& val, uint64_t n)
    {
        const double denom = 1.0 / double(mSize + n);
        const double delta = double(val) - mAvg;
        mAvg += denom * delta * double(n);
        mAux += denom * delta * delta * double(mSize) * double(n);
        BaseT::add(val);
        mSize += n;
    }

    /// Add the samples from the other Stats instance.
    void add(const Stats& other)
    {
        if (other.mSize > 0) {
            const double denom = 1.0 / double(mSize + other.mSize);
            const double delta = other.mAvg - mAvg;
            mAvg += denom * delta * double(other.mSize);
            mAux += other.mAux + denom * delta * delta * double(mSize) * double(other.mSize);
            BaseT::add(other);
            mSize += other.mSize;
        }
    }

    static constexpr bool hasMinMax() { return true; }
    static constexpr bool hasAverage() { return true; }
    static constexpr bool hasStdDeviation() { return true; }

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
    Stats()
        : BaseT()
        , mSize(0)
        , mAvg(0.0)
        , mAux(0.0)
    {
    }
    /// @brief Add a single sample
    void add(const ValueT& val)
    {
        typename BaseT::Pair tmp(val);
        BaseT::add(tmp);
        mSize += 1;
        const double delta = tmp.scalar - mAvg;
        mAvg += delta / double(mSize);
        mAux += delta * (tmp.scalar - mAvg);
    }
    /// @brief Add @a n samples with constant value @a val.
    void add(const ValueT& val, uint64_t n)
    {
        typename BaseT::Pair tmp(val);
        const double         denom = 1.0 / double(mSize + n);
        const double         delta = tmp.scalar - mAvg;
        mAvg += denom * delta * double(n);
        mAux += denom * delta * delta * double(mSize) * double(n);
        BaseT::add(tmp);
        mSize += n;
    }

    /// Add the samples from the other Stats instance.
    void add(const Stats& other)
    {
        if (other.mSize > 0) {
            const double denom = 1.0 / double(mSize + other.mSize);
            const double delta = other.mAvg - mAvg;
            mAvg += denom * delta * double(other.mSize);
            mAux += other.mAux + denom * delta * delta * double(mSize) * double(other.mSize);
            BaseT::add(other);
            mSize += other.mSize;
        }
    }

    static constexpr bool hasMinMax() { return true; }
    static constexpr bool hasAverage() { return true; }
    static constexpr bool hasStdDeviation() { return true; }

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

template<typename ValueT>
struct NoopStats
{
    int i;
    NoopStats() {}
    NoopStats(const ValueT&) {}
    /// @brief Add a single sample
    void add(const ValueT&) {}
    /// @brief Add @a n samples with constant value @a val.
    void add(const ValueT&, uint64_t) {}

    /// Add the samples from the other Stats instance.
    void add(const NoopStats&) {}

    static constexpr bool hasMinMax() { return false; }
    static constexpr bool hasAverage() { return false; }
    static constexpr bool hasStdDeviation() { return false; }

    static size_t size() { return 0; }

    //@{
    /// Return the  arithmetic mean, i.e. average, value.
    static double avg() { return 0.0; }
    static double mean() { return 0.0; }
    //@}

    //@{
    /// @brief Return the population variance.
    ///
    /// @note The unbiased sample variance = population variance * num/(num-1)
    static double var() { return 0.0; }
    static double variance() { return 0.0; }
    //@}

    //@{
    /// @brief Return the standard deviation (=Sqrt(variance)) as
    ///        defined from the (biased) population variance.
    static double std() { return 0.0; }
    static double stdDev() { return 0.0; }
    //@}
}; // end NoopStats<T>

//================================================================================================

/// @brief Allows for the construction of NanoVDB grids without any dependecy
template<typename ValueT, typename StatsT = Stats<ValueT>>
class GridStats
{
    using Node0 = LeafNode<ValueT>; // leaf
    using Node1 = InternalNode<Node0>; // lower
    using Node2 = InternalNode<Node1>; // upper
    using RootT = RootNode<Node2>;

    static constexpr bool DO_STATS = StatsT::hasMinMax() || StatsT::hasAverage() || StatsT::hasStdDeviation();

    NanoGrid<ValueT>*     mGrid;
    ValueT                mDelta; // skip node if: node.max < -mDelta || node.min > mDelta
    std::atomic<uint64_t> mActiveVoxelCount;

    // Below are private methods use to serialize nodes into NanoVDB
    void processLeafs(std::vector<StatsT>&);
    template<typename NodeT>
    void processNodes(std::vector<StatsT>&, std::vector<StatsT>&);
    void processRoot(std::vector<StatsT>&);
    void processGrid();

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
    GridStats()
        : mGrid(nullptr)
    {
    }

    void operator()(NanoGrid<ValueT>& grid, ValueT delta = ValueT(0));

}; // GridStats

//================================================================================================

template<typename ValueT, typename StatsT>
void GridStats<ValueT, StatsT>::operator()(NanoGrid<ValueT>& grid, ValueT delta)
{
    mGrid = &grid;
    mDelta = delta; // delta = voxel size for level sets, else 0
    mActiveVoxelCount = 0;

    std::vector<StatsT> stats0;
    std::vector<StatsT> stats1;

    if (DO_STATS) { // resolved at compiletime
        stats0.resize(mGrid->tree().nodeCount(0));
        stats1.resize(mGrid->tree().nodeCount(1));
    }

    this->processLeafs(stats0);

    this->template processNodes<Node1>(stats1, stats0);

    if (DO_STATS) { // resolved at compiletime
        stats0.resize(mGrid->tree().nodeCount(2));
    }
    this->template processNodes<Node2>(stats0, stats1);

    this->processRoot(stats0);

    this->processGrid();
}

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename DataT, int Rank>
inline void GridStats<ValueT, StatsT>::
    setStats(DataT* data, const Extrema<ValueT, Rank>& e)
{
    data->mMinimum = e.min();
    data->mMaximum = e.max();
}

template<typename ValueT, typename StatsT>
template<typename DataT, int Rank>
inline void GridStats<ValueT, StatsT>::
    setStats(DataT* data, const Stats<ValueT, Rank>& s)
{
    data->mMinimum = s.min();
    data->mMaximum = s.max();
    data->mAverage = s.avg();
    data->mStdDevi = s.std();
}

//================================================================================================

template<typename ValueT, typename StatsT>
template<typename T, typename FlagT>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridStats<ValueT, StatsT>::
    setFlag(const T& min, const T& max, FlagT& flag) const
{
    if (mDelta > 0 && (min > mDelta || max < -mDelta)) {
        flag |= FlagT(1); // set first bit
    } else {
        flag &= ~FlagT(1); // unset first bit
    }
}

//================================================================================================

template<typename ValueT, typename StatsT>
void GridStats<ValueT, StatsT>::
    processLeafs(std::vector<StatsT>& stats)
{
    auto& tree = mGrid->tree();
    auto  kernel = [&](const Range1D& r) {
        uint64_t sum = 0;
        for (auto i = r.begin(); i != r.end(); ++i) {
            Node0* leaf = tree.template getNode<Node0>(i);
            auto*  data = leaf->data();
            if (auto n = data->mValueMask.countOn()) {
                sum += n;
                leaf->updateBBox(); // optionally update active bounding box
                if (DO_STATS) { // resolved at compiletime
                    const ValueT* v = data->mValues;
                    StatsT&       s = stats[i];
                    for (auto it = data->mValueMask.beginOn(); it; ++it)
                        s.add(v[*it]);
                    this->setStats(data, s);
                    this->setFlag(data->mMinimum, data->mMaximum, data->mFlags);
                }
            } else {
                throw std::runtime_error("Expected at least one active voxel per leaf node.");
            }
        }
        mActiveVoxelCount += sum;
    };
    forEach(0, tree.nodeCount(0), 8, kernel);
} // GridStats::processLeafs

//================================================================================================
template<typename ValueT, typename StatsT>
template<typename NodeT>
void GridStats<ValueT, StatsT>::
    processNodes(std::vector<StatsT>& stats, std::vector<StatsT>& childStats)
{
    using ChildT = typename NodeT::ChildNodeType;
    auto& tree = mGrid->tree();
    auto  kernel = [&](const Range1D& r) {
        uint64_t sum = 0;
        for (auto i = r.begin(); i != r.end(); ++i) {
            NodeT* node = tree.template getNode<NodeT>(i);
            auto*  data = node->data();
            sum += ChildT::NUM_VALUES * data->mValueMask.countOn(); // active tiles
            CoordBBox bbox; // empty
            for (auto iter = data->mValueMask.beginOn(); iter; ++iter) {
                if (DO_STATS) { // resolved at compiletime
                    stats[i].add(data->mTable[*iter].value, ChildT::NUM_VALUES);
                }
                const Coord ijk = node->offsetToGlobalCoord(*iter);
                bbox[0].minComponent(ijk);
                bbox[1].maxComponent(ijk + Coord(int32_t(ChildT::DIM) - 1));
            }
            for (auto iter = data->mChildMask.beginOn(); iter; ++iter) {
                if (DO_STATS) { // resolved at compiletime
                    stats[i].add(childStats[data->mTable[*iter].childID]);
                }
                const auto& childBBox = data->child(*iter)->bbox();
                bbox[0].minComponent(childBBox[0]);
                bbox[1].maxComponent(childBBox[1]);
            }
            if (bbox.empty())
                throw std::runtime_error("Internal node with no children or active values! Hint: try pruneInactive.");
            data->mBBox = bbox;
            if (DO_STATS) { // resolved at compiletime
                this->setStats(data, stats[i]);
                this->setFlag(data->mMinimum, data->mMaximum, data->mFlags);
            }
        }
        mActiveVoxelCount += sum;
    };
    forEach(0, tree.template nodeCount<NodeT>(), 4, kernel);
} // GridStats::processNodes

//================================================================================================
template<typename ValueT, typename StatsT>
void GridStats<ValueT, StatsT>::
    processRoot(std::vector<StatsT>& childStats)
{
    using ChildT = Node2;
    RootT&    root = mGrid->tree().root();
    auto&     data = *root.data();
    CoordBBox bbox; // set to an empty bounding box
    if (data.mTileCount == 0) { // empty root node
        data.mMinimum = data.mMaximum = data.mBackground;
        data.mAverage = data.mStdDevi = 0;
        data.mActiveVoxelCount = 0;
    } else {
        StatsT s; // invalid
        for (uint32_t i = 0; i < data.mTileCount; ++i) {
            auto& tile = data.tile(i);
            if (tile.isChild()) { // process child node
                auto& childBBox = data.child(tile).bbox();
                bbox[0].minComponent(childBBox[0]);
                bbox[1].maxComponent(childBBox[1]);
                if (DO_STATS)
                    s.add(childStats[tile.childID]); // resolved at compiletime
            } else if (tile.state) { // active tile
                mActiveVoxelCount += ChildT::NUM_VALUES;
                const Coord ijk = tile.origin();
                bbox[0].minComponent(ijk);
                bbox[1].maxComponent(ijk + Coord(ChildT::DIM - 1));
                if (DO_STATS)
                    s.add(tile.value, ChildT::NUM_VALUES); // resolved at compiletime
            }
        }
        this->setStats(&data, s);
        data.mActiveVoxelCount = mActiveVoxelCount;
        if (bbox.empty()) {
            std::cerr << "\nWarning: input tree only contained inactive root tiles! While not strictly an error it's suspecious." << std::endl;
        }
    }
    data.mBBox = bbox;
} // GridStats::processRoot

//================================================================================================

template<typename ValueT, typename StatsT>
void GridStats<ValueT, StatsT>::
    processGrid()
{
    // set world space AABB
    auto&       data = *mGrid->data();
    const auto& indexBBox = mGrid->tree().root().bbox();
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
        const auto& map = mGrid->map();
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
} // GridStats::processGrid

//================================================================================================

template<typename ValueT>
void gridStats(NanoGrid<ValueT>& grid, StatsMode mode)
{
    if (mode == StatsMode::Disable) {
        return;
    } else if (mode == StatsMode::BBox) {
        GridStats<ValueT, NoopStats<ValueT> > stats;
        stats(grid);
    } else if (mode == StatsMode::MinMax) {
        GridStats<ValueT, Extrema<ValueT> > stats;
        stats(grid);
    } else if (mode == StatsMode::All) {
        GridStats<ValueT, Stats<ValueT> > stats;
        stats(grid); 
    } else {
        throw std::runtime_error("gridStats: Unsupported statistics mode.");
    }
}

} // namespace nanovdb

#endif // NANOVDB_GRIDSTATS_H_HAS_BEEN_INCLUDED
