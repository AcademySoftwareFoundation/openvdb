// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_AGGREGATE_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_AGGREGATE_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <openvdb/tools/GridTransformer.h>
#include <openvdb/math/Transform.h>

#include <vector>
#include <math.h>


/* openvdbmma::aggregate members

 struct SliceActiveVoxelCounts

 struct SliceActiveVoxelValueTotals

 struct GridSlice

 struct GridData

 struct ActiveTiles

 struct ActiveVoxelData

 activeVoxelSparseArray

 All are templated and intended for use with DynamicNodeManager.

*/


namespace openvdbmma {
namespace aggregate {

//////////// voxel value functors

template<typename TreeType>
struct SliceActiveVoxelCounts
{
    using LeafT = typename TreeType::LeafNodeType;

    explicit SliceActiveVoxelCounts(const int& zmin, const int& zmax)
    : mZmin(zmin), mZmax(zmax)
    {
        counts = std::vector<openvdb::Index64>(mZmax - mZmin + 1, 0);
    }

    SliceActiveVoxelCounts(const SliceActiveVoxelCounts& other, tbb::split)
    : mZmin(other.mZmin), mZmax(other.mZmax)
    {
        counts = std::vector<openvdb::Index64>(mZmax - mZmin + 1, 0);
    }

    // accumulate all voxels in active tile children bounded by the bbox
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        bool has_overlap = false;

        const CoordBBox bbox(node.getNodeBoundingBox());
        const long int nz1 = bbox.min().z(), nz2 = bbox.max().z();

        if (nz1 > mZmax || nz2 < mZmin)
            return false;

        // count any overlapping regions in active tiles
        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            const long int snz1 = iter.getCoord().z();
            const long int snz2 = snz1 + NodeT::ChildNodeType::DIM;

            if (snz1 > mZmax || snz2 < mZmin) {
                // box is completely outside the active tile
                continue;
            } else if (snz1 <= mZmin && mZmax <= snz2) {
                // bbox is completely inside the active tile
                has_overlap = true;
                aggregateTileSliceCounts(mZmin, mZmax, NodeT::ChildNodeType::DIM);
            } else if (mZmin <= snz1 && snz2 <= mZmax) {
                // active tile is completely inside bbox
                has_overlap = true;
                aggregateTileSliceCounts(snz1, snz2, NodeT::ChildNodeType::DIM);
            } else {
                // partial overlap between tile and bbox
                has_overlap = true;
                aggregateTileSliceCounts(snz1 >= mZmin ? snz1 : mZmin,
                    snz2 <= mZmax ? snz2 : mZmax, NodeT::ChildNodeType::DIM);
            }
        }

        // return true if any child nodes overlap with the bounding box
        if (has_overlap)
            return true;

        for (auto iter = node.cbeginChildOn(); iter; ++iter) {
            const CoordBBox cbbox(iter->getNodeBoundingBox());
            if (cbbox.max().z() >= mZmin && cbbox.min().z() <= mZmax)
                return true;
        }

        // otherwise return false to prevent recursion along this branch
        return false;
    }

    // accumulate all active voxels in the leaf bounded by the bbox
    inline bool operator()(const LeafT& leaf, size_t)
    {
        const CoordBBox bbox(leaf.getNodeBoundingBox());
        const long int nz1 = bbox.min().z(), nz2 = bbox.max().z();

        if (nz1 > mZmax || nz2 < mZmin) {
            // bbox is completely outside the leaf node
            return false;
        } else if (leaf.isDense()) {
            // partial overlap between dense leaf node and bbox
            aggregateTileSliceCounts(nz1 >= mZmin ? nz1 : mZmin,
                nz2 < mZmax ? nz2+1 : mZmax, nz2 - nz1 + 1);
        } else {
            // partial overlap between sparse leaf node and bbox
            aggregateLeafSliceCounts(leaf);
        }
        return false;
    }

    void join(const SliceActiveVoxelCounts& other)
    {
        std::transform(counts.begin(), counts.end(),
            other.counts.begin(), counts.begin(), std::plus<openvdb::Index64>());
    }

    std::vector<openvdb::Index64> counts;

private:

    inline void aggregateTileSliceCounts(long int z1, long int z2, long int dim)
    {
        const Index64 area = dim * dim;
        for(int i = z1 - mZmin; i < z2 - mZmin; i++)
            counts[i] += area;
    }

    inline void aggregateLeafSliceCounts(LeafT leaf)
    {
        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            const int zz = iter.getCoord().z();
            if (mZmin <= zz && zz <= mZmax)
                counts[zz - mZmin]++;
        }
    }

    //////////// private members

    long int mZmin, mZmax;
};


template<typename TreeType>
struct SliceActiveVoxelValueTotals
{
    using ValueT = typename TreeType::ValueType;
    using LeafT = typename TreeType::LeafNodeType;

    explicit SliceActiveVoxelValueTotals(const int& zmin, const int& zmax)
    : mZmin(zmin), mZmax(zmax)
    {
        totals = std::vector<ValueT>(mZmax - mZmin + 1, (ValueT)0);
    }

    SliceActiveVoxelValueTotals(const SliceActiveVoxelValueTotals& other, tbb::split)
    : mZmin(other.mZmin), mZmax(other.mZmax)
    {
        totals = std::vector<ValueT>(mZmax - mZmin + 1, (ValueT)0);
    }

    // accumulate all voxels in active tile children bounded by the bbox
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        bool has_overlap = false;

        const CoordBBox bbox(node.getNodeBoundingBox());
        const long int nz1 = bbox.min().z(), nz2 = bbox.max().z();

        if (nz1 > mZmax || nz2 < mZmin)
            return false;

        // count any overlapping regions in active tiles
        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            const long int snz1 = iter.getCoord().z();
            const long int snz2 = snz1 + NodeT::ChildNodeType::DIM;
            const ValueT val = iter.getValue();

            if (snz1 > mZmax || snz2 < mZmin) {
                // box is completely outside the active tile
                continue;
            } else if (snz1 <= mZmin && mZmax <= snz2) {
                // bbox is completely inside the active tile
                has_overlap = true;
                aggregateTileSliceTotals(val, mZmin, mZmax, NodeT::ChildNodeType::DIM);
            } else if (mZmin <= snz1 && snz2 <= mZmax) {
                // active tile is completely inside bbox
                has_overlap = true;
                aggregateTileSliceTotals(val, snz1, snz2, NodeT::ChildNodeType::DIM);
            } else {
                // partial overlap between tile and bbox
                has_overlap = true;
                aggregateTileSliceTotals(val, snz1 >= mZmin ? snz1 : mZmin,
                    snz2 <= mZmax ? snz2 : mZmax, NodeT::ChildNodeType::DIM);
            }
        }

        // return true if any child nodes overlap with the bounding box
        if (has_overlap)
            return true;

        for (auto iter = node.cbeginChildOn(); iter; ++iter) {
            const CoordBBox cbbox(iter->getNodeBoundingBox());
            if (cbbox.max().z() >= mZmin && cbbox.min().z() <= mZmax)
                return true;
        }

        // otherwise return false to prevent recursion along this branch
        return false;
    }

    // accumulate all active voxels in the leaf bounded by the bbox
    inline bool operator()(const LeafT& leaf, size_t)
    {
        const CoordBBox bbox(leaf.getNodeBoundingBox());
        const long int nz1 = bbox.min().z(), nz2 = bbox.max().z();

        if (nz1 > mZmax || nz2 < mZmin) {
            // bbox is completely outside the leaf node
            return false;
        } else {
            // partial overlap between sparse leaf node and bbox
            aggregateLeafSliceTotals(leaf);
        }
        return false;
    }

    void join(const SliceActiveVoxelValueTotals& other)
    {
        std::transform(totals.begin(), totals.end(),
            other.totals.begin(), totals.begin(), std::plus<ValueT>());
    }

    std::vector<ValueT> totals;

private:

    inline void aggregateTileSliceTotals(const ValueT val, long int z1, long int z2, long int dim)
    {
        const float area = dim * dim;
        for(int i = z1 - mZmin; i < z2 - mZmin; i++)
            totals[i] += val * area;
    }

    inline void aggregateLeafSliceTotals(LeafT leaf)
    {
        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            const int zz = iter.getCoord().z();
            if (mZmin <= zz && zz <= mZmax)
                totals[zz - mZmin] += iter.getValue();
        }
    }

    //////////// private members

    long int mZmin, mZmax;
};


template<typename TreeType, typename SliceType>
struct GridSlice
{
    using ValueT = typename TreeType::ValueType;
    using RootT  = typename TreeType::RootNodeType;
    using LeafT  = typename TreeType::LeafNodeType;

    explicit GridSlice(SliceType d, const int& slice,
        const int& xmn, const int& xmx, const int& ymn, const int& ymx,
        const bool& mirror)
    : data(d), mZ(slice), mXmin(xmn), mXmax(xmx), mYmin(ymn), mYmax(ymx), mMirror(mirror)
    {
        mXlen = mXmax - mXmin;
        mYlen = mYmax - mYmin;
    }

    GridSlice(const GridSlice& other, tbb::split)
    : data(other.data), mZ(other.mZ)
    , mXmin(other.mXmin), mXmax(other.mXmax), mYmin(other.mYmin), mYmax(other.mYmax)
    , mXlen(other.mXlen), mYlen(other.mYlen), mMirror(other.mMirror)
    {
    }

    void operator()(const RootT& node) {}

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if(noOverlap(node.getNodeBoundingBox()))
            return false;

        const int child_dim = NodeT::ChildNodeType::DIM;

        for (auto iter = node.cbeginValueAll(); iter; ++iter) {
            const Coord p = iter.getCoord();

            if (hasOverlap(p, child_dim)) {
                const ValueT ival = *iter;

                const int xstart = xLeft(p, child_dim), xend = xRight(p, child_dim);
                const int ystart = yLeft(p, child_dim), yend = yRight(p, child_dim);

                for(int j = ystart; j <= yend; j++)
                    for(int i = xstart; i <= xend; i++)
                        data.template setValue<ValueT>(j, i, ival);
            }
        }

        return true;
    }

    inline bool operator()(const LeafT& leaf, size_t)
    {
        if(noOverlap(leaf.getNodeBoundingBox()))
            return false;

        for (auto iter = leaf.cbeginValueAll(); iter; ++iter) {
            const Coord p = iter.getCoord();
            if (isInside(p))
                data.template setValue<ValueT>(yPos(p), xPos(p), *iter);
        }

        return false;
    }

    void join(const GridSlice& other) {}

    SliceType data;

private:

    inline bool hasOverlap(const Coord& p, const int& len) const
    {
        return p.z() <= mZ && p.x() <= mXmax && p.y() <= mYmax
            && mZ <= p.z() + len && mXmin <= p.x() + len && mYmin <= p.y() + len;
    }

    inline bool noOverlap(const CoordBBox& bbox) const
    {
        return bbox.min().z() > mZ || bbox.max().z() < mZ
            || bbox.min().x() > mXmax || bbox.max().x() < mXmin
            || bbox.min().y() > mYmax || bbox.max().y() < mYmin;
    }

    inline bool isInside(const Coord& p) const
    {
        return p.z() == mZ && mXmin <= p.x() && p.x() <= mXmax
            && mYmin <= p.y() && p.y() <= mYmax;
    }

    inline int xLeft (const Coord& p, const int& len) const
    {
        return mMirror ? math::Max(0, mXmax - p.x() - len) : math::Max(0, p.x() - mXmin);

    }
    inline int xRight(const Coord& p, const int& len) const
    {
        return mMirror ? math::Min(mXlen, mXmax - p.x()) : math::Min(mXlen, p.x() - mXmin + len);
    }

    inline int yLeft (const Coord& p, const int& len) const
    {
        return math::Max(0, mYmax - p.y() - len);
    }
    inline int yRight(const Coord& p, const int& len) const
    {
        return math::Min(mYlen, mYmax - p.y());
    }

    inline int xPos(const Coord& p) const { return mMirror ? mXmax - p.x() : p.x() - mXmin; }
    inline int yPos(const Coord& p) const { return mYmax - p.y(); }

    //////////// private members

    int mZ, mXmin, mXmax, mYmin, mYmax;
    int mXlen, mYlen;
    bool mMirror;
};


template<typename TreeType, typename CubeType>
struct GridData
{
    using ValueT = typename TreeType::ValueType;
    using RootT  = typename TreeType::RootNodeType;
    using LeafT  = typename TreeType::LeafNodeType;

    explicit GridData(CubeType d, const CoordBBox& bbox) : data(d), mBBox(bbox)
    {
        mXlen = bbox.max().x() - bbox.min().x();
        mYlen = bbox.max().y() - bbox.min().y();
        mZlen = bbox.max().z() - bbox.min().z();

        mOx = bbox.min().x();
        mOy = bbox.max().y();
        mOz = bbox.max().z();
    }

    GridData(const GridData& other, tbb::split)
    : data(other.data), mBBox(other.mBBox)
    , mOx(other.mOx), mOy(other.mOy), mOz(other.mOz)
    , mXlen(other.mXlen), mYlen(other.mYlen), mZlen(other.mZlen)
    {
    }

    void operator()(const RootT& node) {}

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if (!mBBox.hasOverlap(node.getNodeBoundingBox()))
            return false;

        for (auto iter = node.cbeginValueAll(); iter; ++iter) {
            const CoordBBox bbox(CoordBBox::createCube(iter.getCoord(), NodeT::ChildNodeType::DIM));

            if (bbox.hasOverlap(mBBox)) {
                const ValueT ival = *iter;

                const int xstart = xLeft(bbox), xend = xRight(bbox);
                const int ystart = yLeft(bbox), yend = yRight(bbox);
                const int zstart = zLeft(bbox), zend = zRight(bbox);

                for(int k = zstart; k <= zend; k++)
                    for(int j = ystart; j <= yend; j++)
                        for(int i = xstart; i <= xend; i++)
                            data.template setValue<ValueT>(k, j, i, ival);
            }
        }

        return true;
    }

    inline bool operator()(const LeafT& leaf, size_t)
    {
        if (!mBBox.hasOverlap(leaf.getNodeBoundingBox()))
            return false;

        for (auto iter = leaf.cbeginValueAll(); iter; ++iter) {
            const Coord p = iter.getCoord();
            if (mBBox.isInside(p))
                data.template setValue<ValueT>(mOz - p.z(), mOy - p.y(), p.x() - mOx, *iter);
        }

        return false;
    }

    void join(const GridData& other) {}

    CubeType data;

private:

    inline int xLeft (const CoordBBox& bbox) const { return math::Max(0, bbox.min().x() - mOx); }
    inline int xRight(const CoordBBox& bbox) const { return math::Min(mXlen, bbox.max().x() - mOx); }

    inline int yLeft (const CoordBBox& bbox) const { return math::Max(0, mOy - bbox.max().y()); }
    inline int yRight(const CoordBBox& bbox) const { return math::Min(mYlen, mOy - bbox.min().y()); }

    inline int zLeft (const CoordBBox& bbox) const { return math::Max(0, mOz - bbox.max().z()); }
    inline int zRight(const CoordBBox& bbox) const { return math::Min(mZlen, mOz - bbox.min().z()); }

    //////////// private members

    CoordBBox mBBox;
    int mOx, mOy, mOz;
    int mXlen, mYlen, mZlen;
};


template<typename TreeType>
struct ActiveTiles
{
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    explicit ActiveTiles(const CoordBBox& bbox, const bool& po)
    : mBBox(bbox), partial_overlap(po)
    {
    }

    ActiveTiles(const ActiveTiles& other, tbb::split)
    : mBBox(other.mBBox), partial_overlap(other.partial_overlap)
    {
    }

    void operator()(const RootT& node) {}

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if (!mBBox.hasOverlap(node.getNodeBoundingBox()))
            return false;

        for (auto iter = node.cbeginValueOn(); iter; ++iter) {
            const CoordBBox bbox(CoordBBox::createCube(iter.getCoord(), NodeT::ChildNodeType::DIM));

            if (mBBox.isInside(bbox) || (partial_overlap && mBBox.hasOverlap(bbox))) {
                tile_vec.insert(tile_vec.end(),
                  {bbox.min().x(), bbox.min().y(), bbox.min().z(),
                      bbox.max().x(), bbox.max().y(), bbox.max().z()});
            }
        }

        if (node.getLevel() == 1)
            return false;

        for (auto iter = node.cbeginChildOn(); iter; ++iter) {
            if (mBBox.hasOverlap(iter->getNodeBoundingBox()))
                return true;
        }

        return false;
    }

    inline bool operator()(const LeafT& leaf, size_t) { return false; }

    void join(const ActiveTiles& other)
    {
        tile_vec.insert(
            tile_vec.end(),
            std::make_move_iterator(other.tile_vec.begin()),
            std::make_move_iterator(other.tile_vec.end())
        );
    }

    std::vector<int> tile_vec;

private:

    CoordBBox mBBox;
    bool partial_overlap;
};


template<typename TreeType>
struct ActiveVoxelData
{
    using ValueT = typename TreeType::ValueType;
    using RootT = typename TreeType::RootNodeType;
    using LeafT = typename TreeType::LeafNodeType;

    explicit ActiveVoxelData(const CoordBBox& bbox, const bool ppos = true,
        const bool pvals = true) : mBBox(bbox), populate_pos(ppos), populate_vals(pvals) {}

    ActiveVoxelData(const ActiveVoxelData& other, tbb::split)
    : mBBox(other.mBBox), populate_pos(other.populate_pos), populate_vals(other.populate_vals) {}

    void operator()(const RootT& node) {}

    template<typename NodeT>
    bool operator()(const NodeT& node, size_t)
    {
        if (!mBBox.hasOverlap(node.getNodeBoundingBox()))
            return false;

        for (auto iter = node.cbeginChildOn(); iter; ++iter) {
            if (mBBox.hasOverlap(iter->getNodeBoundingBox()))
                return true;
        }

        return false;
    }

    inline bool operator()(const LeafT& leaf, size_t)
    {
        const CoordBBox bbox = leaf.getNodeBoundingBox();

        if(!bbox.hasOverlap(mBBox))
            return false;

        for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
            const Coord p = iter.getCoord();
            if(mBBox.isInside(p)) {
                if (populate_pos)
                    pos_vec.insert(pos_vec.end(), {p.x(), p.y(), p.z()});

                if (populate_vals)
                    val_vec.push_back(*iter);
            }
        }

        return false;
    }

    void join(const ActiveVoxelData& other)
    {
        if (populate_pos) {
            pos_vec.insert(
                pos_vec.end(),
                std::make_move_iterator(other.pos_vec.begin()),
                std::make_move_iterator(other.pos_vec.end())
            );
        }

        if (populate_vals) {
            val_vec.insert(
                val_vec.end(),
                std::make_move_iterator(other.val_vec.begin()),
                std::make_move_iterator(other.val_vec.end())
            );
        }
    }

    std::vector<int> pos_vec;
    std::vector<ValueT> val_vec;

private:

    CoordBBox mBBox;

    bool populate_pos, populate_vals;
};

template<typename TreeT, typename BaseT>
mma::SparseArrayRef<BaseT>
activeVoxelSparseArray(TreeT &tree, mma::IntBounds3DRef bds)
{
    using ValueT = typename TreeT::ValueType;

    static_assert(std::is_arithmetic<ValueT>::value,
        "activeVoxelSparseArray requires a numeric or Boolean tree type.");

    static_assert(std::is_arithmetic<BaseT>::value,
        "activeVoxelSparseArray requires a numeric base type.");

    static_assert((std::is_integral<ValueT>::value && std::is_integral<BaseT>::value) ||
        (std::is_floating_point<ValueT>::value && std::is_floating_point<BaseT>::value),
        "activeVoxelSparseArray requires consistent tree and base types.");

    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    const CoordBBox bbox(bds.toCoordBBox());

    ActiveVoxelData<TreeT> op(bbox);
    tree::DynamicNodeManager<const TreeT> nodeManager(tree);
    nodeManager.reduceTopDown(op, true);

    mma::check_abort();

    const mint cnt = op.val_vec.size();
    const Coord d = bbox.dim();

    mma::IntCoordinatesRef pos = mma::makeCoordinatesList<mint>(cnt);
    mma::VectorRef<BaseT> vals = mma::makeVector<BaseT>(cnt);
    mma::IntVectorRef dims = mma::makeVector<mint>({d.x(), d.y(), d.z()});

    const int xmin = bbox.min().x();
    const int ymin = bbox.min().y();
    const int zmin = bbox.min().z();

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, cnt),
        [&](tbb::blocked_range<mint> rng)
        {
            for(mint i = rng.begin(); i < rng.end(); ++i) {
                pos[3*i+0] = (op.pos_vec)[3*i+0] - xmin + 1;
                pos[3*i+1] = (op.pos_vec)[3*i+1] - ymin + 1;
                pos[3*i+2] = (op.pos_vec)[3*i+2] - zmin + 1;
                vals[i] = (op.val_vec)[i];
            }
        }
    );

    mma::SparseArrayRef<BaseT> sa = makeSparseArray(pos, vals, dims);

    pos.free();
    vals.free();
    dims.free();

    return sa;
}

} // namespace aggregate
} // namespace openvdbmma

#endif // OPENVDBLINK_UTILITIES_AGGREGATE_HAS_BEEN_INCLUDED
