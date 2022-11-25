// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDBLINK_UTILITIES_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDBLINK_UTILITIES_LEVELSET_HAS_BEEN_INCLUDED

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/SignedFloodFill.h>

#include <openvdb/math/Transform.h>

/* openvdbmma::levelset members

 GridT::Ptr meshToLevelSet(pts, cells, spacing, halfWidth flags)

 GridT::Ptr offsetSurfaceLevelSet(pts, cells, offset, spacing, width, is_signed)

 void meshTensorsToVectors(pts, cells, &pvec, &cvec)

*/


namespace openvdbmma {
namespace levelset {

//////////// level set functors

template<typename TreeType>
struct ShiftDistanceValue
{
    using ValueT = typename TreeType::ValueType;

    ShiftDistanceValue(const ValueT o, const ValueT bg)
    : mOffset(o), mBackground(bg), mBackgroundneg(-1.0*bg)
    {
    }

    template <typename LeafNodeType>
    void operator()(LeafNodeType& leaf, size_t) const
    {
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            const ValueT distnew = *iter - mOffset;

            if (distnew < mBackgroundneg) {
                iter.setValue(mBackgroundneg);
                iter.setValueOff();
            } else if (distnew > mBackground) {
                iter.setValue(mBackground);
                iter.setValueOff();
            } else {
                iter.setValue(distnew);
            }
        }
    }

private:

    ValueT mOffset, mBackground, mBackgroundneg;
};

template<typename ValueT>
struct AbsOp {
    AbsOp() {}
    inline ValueT operator()(const ValueT& x) const {
        return math::Abs(x);
    }
};


//////////// tensor to vector conversion

void
meshTensorsToVectors(mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
    std::vector<Vec3s> &pvec, std::vector<Vec3I> &cvec)
{
    pvec.resize(pts.size());
    cvec.resize(tri_cells.rows());

    mma::check_abort();

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, pts.size()),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i)
                pvec[i] = Vec3s(pts[3*i], pts[3*i+1], pts[3*i+2]);
        }
    );

    mma::check_abort();

    tbb::parallel_for(
        tbb::blocked_range<mint>(0, tri_cells.rows()),
        [&](tbb::blocked_range<mint> rng)
        {
            for (mint i = rng.begin(); i < rng.end(); ++i)
                cvec[i] = Vec3I(tri_cells[3*i], tri_cells[3*i+1], tri_cells[3*i+2]);
        }
    );
}


//////////// internal level set functions

template<typename GridT>
typename GridT::Ptr
meshToLevelSet(std::vector<Vec3s> &pvec, std::vector<Vec3I> &cvec,
    double spacing, double halfWidth, const int flags)
{
    using MeshDataAdapter = QuadAndTriangleDataAdapter<Vec3s, Vec3I>;
    using InterruptT = mma::interrupt::LLInterrupter;

    const math::Transform xform(*(math::Transform::createLinearTransform(spacing)));

    const size_t numPoints = pvec.size();
    std::unique_ptr<Vec3s[]> indexSpacePoints{new Vec3s[numPoints]};

    mma::check_abort();

    tbb::parallel_for(tbb::blocked_range<size_t>(0, numPoints),
        mesh_to_volume_internal::TransformPoints<Vec3s>(&pvec[0], indexSpacePoints.get(), xform));

    mma::check_abort();

    MeshDataAdapter mesh(indexSpacePoints.get(), numPoints, &cvec[0], cvec.size());
    InterruptT interrupt;

    return meshToVolume<GridT, MeshDataAdapter, InterruptT>(
        interrupt, mesh, xform, halfWidth, halfWidth, flags);
}

template<typename GridT>
typename GridT::Ptr
meshToLevelSet(mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
    double spacing, double halfWidth, const int flags)
{
    std::vector<Vec3s> pvec;
    std::vector<Vec3I> cvec;
    meshTensorsToVectors(pts, tri_cells, pvec, cvec);

    return meshToLevelSet<GridT>(pvec, cvec, spacing, halfWidth, flags);
}

template<typename GridT>
typename GridT::Ptr
offsetSurfaceLevelSet(std::vector<Vec3s> &pvec, std::vector<Vec3I> &cvec,
    double offset, double spacing, double width, bool is_signed)
{
    using ValueT = typename GridT::ValueType;
    using GridPtr = typename GridT::Ptr;
    using TreeT = typename GridT::TreeType;

    if (offset <= 0)
        return meshToLevelSet<GridT>(pvec, cvec, spacing, width, is_signed);

    const ValueT halfWidth = width + offset/spacing;

    // ---------------- unsigned distance field of faces ----------------

    const int conversionFlags = UNSIGNED_DISTANCE_FIELD | DISABLE_INTERSECTING_VOXEL_REMOVAL
        | DISABLE_RENORMALIZATION | DISABLE_NARROW_BAND_TRIMMING;

    GridPtr grid = meshToLevelSet<GridT>(pvec, cvec, spacing, halfWidth, conversionFlags);

    // ---------------- signed distance field of offset faces ----------------

    mma::check_abort();

    ShiftDistanceValue<TreeT> op(offset, spacing * width);
    tree::LeafManager<TreeT> leafNodes(grid->tree());
    leafNodes.foreach(op);

    openvdb::tools::changeBackground(grid->tree(), spacing * width);

    // ---------------- assemble level set ----------------

    mma::check_abort();

    grid->setGridClass(GRID_LEVEL_SET);

    if (is_signed) {
        signedFloodFill(grid->tree());
    } else {
        transformActiveLeafValues<TreeT, AbsOp<ValueT>>(grid->tree(), AbsOp<ValueT>());
    }

    return grid;
}

template<typename GridT>
typename GridT::Ptr
offsetSurfaceLevelSet(mma::RealCoordinatesRef pts, mma::IntMatrixRef tri_cells,
    double offset, double spacing, double width, bool is_signed)
{
    std::vector<Vec3s> pvec;
    std::vector<Vec3I> cvec;
    meshTensorsToVectors(pts, tri_cells, pvec, cvec);

    return offsetSurfaceLevelSet<GridT>(pvec, cvec, offset, spacing, width, is_signed);
}

} // namespace levelset
} // namespace openvdbmma

#endif // OPENVDBLINK_UTILITIES_LEVELSET_HAS_BEEN_INCLUDED
