// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_OPENVDBGRID_AGGREGATEDATA_HAS_BEEN_INCLUDED
#define OPENVDBLINK_OPENVDBGRID_AGGREGATEDATA_HAS_BEEN_INCLUDED

#include "../Utilities/Aggregate.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <vector>


/*  OpenVDBGrid public member function list

mma::IntTensorRef sliceVoxelCounts(const mint zmin, const mint zmax)

GlueVector sliceVoxelValueTotals(const mint zmin, const mint zmax)

mma::IntCubeRef activeTiles(mma::IntBounds3DRef bds, const bool partial_overlap = true)

mma::SparseArrayRef<mmaBaseValT> activeVoxels(mma::IntBounds3DRef bds)

mma::IntCoordinatesRef activeVoxelPositions(mma::IntBounds3DRef bds)

GlueVector activeVoxelValues(mma::IntBounds3DRef bds)

GlueMatrix gridSlice(const mint z, mma::IntBounds2DRef bds, const bool mirror_slice, const bool threaded)

GlueCube gridData(mma::IntBounds3DRef bds)

*/


//////////// OpenVDBGrid public member function definitions

template<typename V>
mma::IntTensorRef
openvdbmma::OpenVDBGrid<V>::sliceVoxelCounts(const mint zmin, const mint zmax) const
{
    openvdbmma::aggregate::SliceActiveVoxelCounts<wlTreeType> op(zmin, zmax);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, true);

    std::vector<openvdb::Index64> vcounts = op.counts;

    mma::IntTensorRef counts = mma::makeVector<mint>(vcounts.size());
    for(int i = 0; i < counts.size(); i++)
        counts[i] = vcounts[i];

    return counts;
}

template<typename V>
mma::TensorRef<typename openvdbmma::OpenVDBGrid<V>::mmaBaseValT>
openvdbmma::OpenVDBGrid<V>::sliceVoxelValueTotals(const mint zmin, const mint zmax) const
{
    openvdbmma::types::non_mask_type_assert<V>();
    openvdbmma::types::non_bool_type_assert<V>();

    openvdbmma::aggregate::SliceActiveVoxelValueTotals<wlTreeType> op(zmin, zmax);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, true);

    return GVector(op.totals).mmaData();
}

template<typename V>
mma::IntCubeRef
openvdbmma::OpenVDBGrid<V>::activeTiles(mma::IntBounds3DRef bds, const bool partial_overlap) const
{
    if(bds.isDegenerate())
        return mma::makeCube<mint>(0, 2, 3);

    openvdbmma::aggregate::ActiveTiles<wlTreeType> op(bds.toCoordBBox(), partial_overlap);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, true);

    mma::check_abort();

    const mint cnt = op.tile_vec.size();
    mma::IntCubeRef tiles = mma::makeCube<mint>(cnt/6, 2, 3);

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, cnt),
        [&](tbb::blocked_range<mint> rng)
        {
            for(mint i = rng.begin(); i < rng.end(); ++i)
                tiles[i] = (op.tile_vec)[i];
        }
    );

    return tiles;
}

template<typename V>
mma::SparseArrayRef<typename openvdbmma::OpenVDBGrid<V>::mmaBaseValT>
openvdbmma::OpenVDBGrid<V>::activeVoxels(mma::IntBounds3DRef bds) const
{
    using BaseT = typename OpenVDBGrid<V>::mmaBaseValT;

    return openvdbmma::aggregate::activeVoxelSparseArray<wlTreeType, BaseT>(grid()->tree(), bds);
}

template<typename V>
mma::IntCoordinatesRef
openvdbmma::OpenVDBGrid<V>::activeVoxelPositions(mma::IntBounds3DRef bds) const
{
    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    const CoordBBox bbox(bds.toCoordBBox());

    openvdbmma::aggregate::ActiveVoxelData<wlTreeType> op(bbox, true, false);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, true);

    mma::check_abort();

    const mint cnt = op.pos_vec.size();

    mma::IntCoordinatesRef pos = mma::makeCoordinatesList<mint>(cnt/3);

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, cnt),
        [&](tbb::blocked_range<mint> rng)
        {
            for(mint i = rng.begin(); i < rng.end(); ++i) {
                pos[i] = (op.pos_vec)[i];
            }
        }
    );

    return pos;
}

template<typename V>
mma::TensorRef<typename openvdbmma::OpenVDBGrid<V>::mmaBaseValT>
openvdbmma::OpenVDBGrid<V>::activeVoxelValues(mma::IntBounds3DRef bds) const
{
    openvdbmma::types::non_mask_type_assert<V>();

    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    const CoordBBox bbox(bds.toCoordBBox());

    openvdbmma::aggregate::ActiveVoxelData<wlTreeType> op(bbox, false, true);
    tree::DynamicNodeManager<const wlTreeType> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, true);

    mma::check_abort();

    return GVector(op.val_vec).mmaData();
}

template<typename V>
mma::TensorRef<typename openvdbmma::OpenVDBGrid<V>::mmaBaseValT>
openvdbmma::OpenVDBGrid<V>::gridSlice(const mint z, mma::IntBounds2DRef bds,
    const bool mirror_slice, const bool threaded) const
{
    openvdbmma::types::non_mask_type_assert<V>();

    using MatT = GMatrix;
    using TreeT = wlTreeType;

    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    // No built in 2D CoordBBox, so just pass in ordinates
    const int xmin = bds.xmin();
    const int xmax = bds.xmax();
    const int ymin = bds.ymin();
    const int ymax = bds.ymax();

    MatT data(ymax-ymin+1, xmax-xmin+1);

    openvdbmma::aggregate::GridSlice<TreeT, MatT> op(data, z, xmin, xmax, ymin, ymax, mirror_slice);
    tree::DynamicNodeManager<const TreeT> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, threaded);

    return data.mmaData();
}

template<typename V>
mma::TensorRef<typename openvdbmma::OpenVDBGrid<V>::mmaBaseValT>
openvdbmma::OpenVDBGrid<V>::gridData(mma::IntBounds3DRef bds) const
{
    openvdbmma::types::non_mask_type_assert<V>();

    using CubeT = GCube;
    using TreeT = wlTreeType;

    if(bds.isDegenerate())
        throw mma::LibraryError(LIBRARY_FUNCTION_ERROR);

    const CoordBBox bbox(bds.toCoordBBox());

    // {column count, row count, slice count}
    const Coord dims = bbox.dim();

    CubeT data(dims.z(), dims.y(), dims.x());

    openvdbmma::aggregate::GridData<TreeT, CubeT> op(data, bbox);
    tree::DynamicNodeManager<const TreeT> nodeManager(grid()->tree());
    nodeManager.reduceTopDown(op, true);

    return data.mmaData();
}

#endif // OPENVDBLINK_OPENVDBGRID_AGGREGATEDATA_HAS_BEEN_INCLUDED
