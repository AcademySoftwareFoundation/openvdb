// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth
///
/// @file tools/PolySoupToLevelSet.h
///
/// @brief Generates a LOD family of watertight shrink wrap surfaces from
///        a soup of polygons.
///
/// @details Details of this algorithm are given in an upcoming publication.

#ifndef OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/Assert.h>

#include "Composite.h" // for csgUnion
#include "GridTransformer.h" // for resampleToMatch
#include "MeshToVolume.h"// for meshToLevelSet
#include "VolumeToMesh.h"// for volumeToMesh
#include "LevelSetFilter.h"// for Filter
#include "LevelSetMeasure.h"/// for levelSetVolume

#include <iostream>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

struct PolySoup {
    std::vector<Vec3s> vtx;
    std::vector<Vec3I> tri;
    std::vector<Vec4I> quad;
    math::BBox<Vec3f>  bbox;
};

class ShrinkWrapLimit;// ShrinkWrapLimit

template <typename GridType>
class PolySoupToLevelSet;

/// @brief Convert a soup of polygons to a shrink wrapped level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         that shrink wraps the input polygons.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Unlike tools::meshToLevelSet this method works for any polygons,
///         and does not require a closed surface.
///
/// @param poly       struct with polygon soup, that will be destroyed (moved)
/// @param dim        optional dimension in voxel units (assuming voxelSize is invalid)
/// @param voxelSize  optional voxel size in world units (if invalid dim will be used instead)
/// @param D          functor mapping voxel size to maximum allowed surface deformation
///                   allowed by shrink wrapping as a function of the voxel size
/// @param halfWidth  half the width of the narrow band, in voxel units
/// @param progress   optional pointer to progress bar
template<typename GridType, class ShrinkWrapT = ShrinkWrapLimit, class ProgressT = void>
typename GridType::Ptr
polySoupToLevelSet(
    PolySoup &&poly,
    int dim = 256,
    float voxelSize = 0.0,// invalid so use dim instead
    const ShrinkWrapT &D = ShrinkWrapT(),
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    ProgressT *progress = nullptr);

/// @brief Convert a soup of polygons to a LOD sequence of shrink wrapped level set volumes.
///
/// @return a vector of grids of type @c GridType containing a narrow-band level set
///         at various resolution shrink wrapping the input polygon mesh. The first
///         element in this vector has the highest resolution.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Unlike tools::meshToLevelSet this method works for any polygons,
///         and does not require a closed surface.
///
/// @param dim          Largest voxel dimension of the finest output grid
/// @param bbox         bounding box of the vertices of the polygon mesh
/// @param vtx          vector of world space vertex positions
/// @param tri          vector of triangle indies
/// @param quads        vector of quad indices
/// @param D            functor mapping voxel size to maximum allowed surface deformation
///                     allowed by shrink wrapping as a function of the voxel size
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param progress   optional pointer to progress bar
template<typename GridType, class ShrinkWrapT = ShrinkWrapLimit, class ProgressT = void>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    int dim,
    const math::BBox<Vec3f> &bbox,
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D = ShrinkWrapT(),
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    ProgressT *progress = nullptr);

/// @brief Convert a soup of polygons to a LOD sequence of shrink wrapped level set volumes.
///
/// @return a vector of grids of type @c GridType containing a narrow-band level set
///         at various resolution shrink wrapping the input polygon mesh. The first
///         element in this vector has the highest resolution.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Unlike tools::meshToLevelSet this method works for any polygons,
///         and does not require a closed surface.
///
/// @param minVoxelSize finst/smallest voxel size of the output grids
/// @param bbox         bounding box of the vertices of the polygon mesh
/// @param vtx          vector of world space vertex positions
/// @param tri          vector of triangle indies
/// @param quads        vector of quad indices
/// @param D            functor mapping voxel size to maximum allowed surface deformation
///                     allowed by shrink wrapping as a function of the voxel size
/// @param halfWidth    half the width of the narrow band, in voxel units
/// @param progress   optional pointer to progress bar
template<typename GridType, class ShrinkWrapT = ShrinkWrapLimit, class ProgressT = void>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    float minVoxelSize,
    const math::BBox<Vec3f> &bbox,
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D = ShrinkWrapT(),
    float halfWidth = float(LEVEL_SET_HALF_WIDTH),
    ProgressT *progress = nullptr);

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType>
class PolySoupToLevelSet
{
public:

    PolySoupToLevelSet(PolySoup &&poly, int dim, float width = float(LEVEL_SET_HALF_WIDTH))
      : mPoly(poly), mHalfWidth(width)
    {
        if constexpr(!std::is_floating_point<typename GridType::ValueType>::value) {
            OPENVDB_THROW(TypeError, "polySoupToLevelSet: supported only for scalar floating-point grids");
        }
        if (!mPoly.bbox) mPoly.bbox = PolySoupToLevelSet::getBBox(mPoly.vtx);
        const float maxLength = mPoly.bbox.extents()[mPoly.bbox.maxExtent()];
        mMinVoxelSize = maxLength/(dim - 2.0f*(mHalfWidth + 1.0f));// +1 since final surface is dilated by dx
        mMaxVoxelSize = maxLength / 2.0f;
        OPENVDB_ASSERT(2*mMinVoxelSize <= mMaxVoxelSize);
    }
    PolySoupToLevelSet(PolySoup &&poly, float voxelSize, float width = float(LEVEL_SET_HALF_WIDTH))
      : mPoly(poly), mMinVoxelSize(voxelSize), mHalfWidth(width)
    {
        if constexpr(!std::is_floating_point<typename GridType::ValueType>::value) {
            OPENVDB_THROW(TypeError, "polySoupToLevelSet: supported only for scalar floating-point grids");
        }
        if (!mPoly.bbox) mPoly.bbox = PolySoupToLevelSet::getBBox(mPoly.vtx);
        const float maxLength = mPoly.bbox.extents()[mPoly.bbox.maxExtent()];
        mMaxVoxelSize = maxLength / 2.0f;
        OPENVDB_ASSERT(2*mMinVoxelSize <= mMaxVoxelSize);
    }

    template<class ShrinkWrapT, class ProgressT>
    void process(const ShrinkWrapT &D, ProgressT *progress);

    size_t gridCount() const {return mGrids.size();}

    typename GridType::Ptr grid(int n = 0) const {return mGrids[n];}

    std::vector<typename GridType::Ptr> grids() const {return mGrids;}

    const PolySoup& mesh(int n = 0, float adaptivity = 0.005f, float isoValue = 0.0f)
    {
        volumeToMesh(*mGrids[n], mPoly.vtx, mPoly.tri, mPoly.quad, isoValue, adaptivity);
        return mPoly;
    }

    static math::BBox<Vec3f> getBBox(const std::vector<Vec3s> &vtx)
    {
        using RangeT = tbb::blocked_range<std::vector<Vec3s>::const_iterator>;
        RangeT range(vtx.begin(), vtx.end(), 1024);
        struct BBoxOp {
            math::BBox<Vec3f> bbox;
            BBoxOp() : bbox() {}
            BBoxOp(BBoxOp& s, tbb::split) : bbox(s.bbox) {}
            void operator()(const RangeT& r) {for (auto p=r.begin(); p!=r.end(); ++p) bbox.expand(*p);}
            void join(BBoxOp& rhs) {bbox.expand(rhs.bbox);}
        } tmp;
#if 0
        tmp(range);// serial
#else
        tbb::parallel_reduce(range, tmp);// parallel
#endif
        return tmp.bbox;
    }

private:
    PolySoup mPoly;
    float mMinVoxelSize, mMaxVoxelSize, mHalfWidth;
    std::vector<typename GridType::Ptr> mGrids;// fine(0) -> coarse grids(size-1)    
    bool mIsGridSDF;

    auto offset(float dx){
        auto xform = math::Transform::createLinearTransform(dx);
        auto udf = meshToUnsignedDistanceField<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth);// mesh -> UDF
        volumeToMesh(*udf, mPoly.vtx, mPoly.tri, mPoly.quad, dx, 0.0);// UDF -> mesh (clears and re-allocates mesh)
        return meshToLevelSet<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth);// mesh -> SDF
    }

    auto upsample(const GridType &inGrid){
        auto outGrid = createLevelSet<GridType>(inGrid.voxelSize()[0]/2, mHalfWidth);
        resampleToMatch<BoxSampler>(inGrid, *outGrid);
        mIsGridSDF = true;
        return outGrid;
    }

    auto shrinkWrap(GridType &grid, const GridType &gridB, float &d){
        const float maxDist = 2.0f;
        LevelSetFilter<GridType> filter(grid);
        filter.setSpatialScheme(math::FIRST_BIAS);
        filter.setTemporalScheme(math::TVD_RK1);
        if (mIsGridSDF == false) filter.normalize();
        filter.offset(maxDist * grid.voxelSize()[0]);// erode by maxDist * dx
        mIsGridSDF = false;// the CSG operation messed up the SDF
        d += maxDist;
        return csgUnionCopy(grid, gridB);
    }
  
};// PolySoupToLevelSet<GridType>

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType>
template<class ShrinkWrapT, class ProgressT>
void PolySoupToLevelSet<GridType>::process(const ShrinkWrapT &D, ProgressT *progress)
{
    auto myProgress = [&](const std::string &s){if constexpr(!std::is_same<ProgressT,void>::value) if (progress) (*progress)(s);};

    if (progress) std::cerr << std::endl;

    // Fine to coarse offset generation
    for (float dx = mMinVoxelSize; dx <= mMaxVoxelSize; dx *= 2.0f) {
        myProgress("Offset: dx=" + std::to_string(dx)+", range: "+std::to_string(mMinVoxelSize)+" -> "+std::to_string(mMaxVoxelSize));
        mGrids.push_back(this->offset(dx));
    }

    // Coarse to fine shrink wrap algorithm
    float vol[2];
    auto grid = mGrids.back();// coarsest grid
    mGrids.pop_back();
    mIsGridSDF = true;
    for (auto iter = mGrids.rbegin(), end = mGrids.rend(); iter != end; ++iter) {// coarse -> fine
      grid = this->upsample(*grid);// grid(dx) -> grid(dx/2)
      for (float d = 0.0f, dx = grid->voxelSize()[0], Ddx = D(dx); d < Ddx; vol[0] = vol[1]) {
        myProgress("Shrink wrap d=" + std::to_string(d) + ", D("+std::to_string(dx) + ")=" + std::to_string(Ddx));
        grid = this->shrinkWrap(*grid, **iter, d);
        vol[1] = levelSetVolume(*grid);
        if (d>0.0f && math::isApproxZero(vol[0]-vol[1])) break;
      }
      *iter = grid;
    }// loop from coarse to fine voxel sizes

}// tools::PolySoupToLevelSet::process()

//////////////////////////////////////////////////////////////////////////

class ShrinkWrapLimit {
    const float mErode, mThres;
public:
    ShrinkWrapLimit(float erode = 8.0f, float thres = 0.0f) : mErode(erode), mThres(thres) {}
    float operator()(float dx) const {// if mThres == 0 this always returns mErode
        return dx>=2*mThres ? mErode : dx<=mThres ? 1.0f : 1.0f + (mErode-1.0f)*(dx-mThres)/mThres;
    }
};// ShrinkWrapLimit


/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType, class ShrinkWrapT, class ProgressT>
typename GridType::Ptr
polySoupToLevelSet(
    PolySoup &&poly,
    int dim,
    float voxelSize,
    const ShrinkWrapT &D,
    float halfWidth,
    ProgressT *progress)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "polySoupToLevelSet requires an SDF grid with floating-point values");
    using T = PolySoupToLevelSet<GridType>;
    auto ptr = voxelSize > 0.0f ? std::make_unique<T>(std::move(poly), voxelSize, halfWidth) :
                                  std::make_unique<T>(std::move(poly), dim, halfWidth);
    ptr->process(D, progress);
    return ptr->grid();
}

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType, class ShrinkWrapT, class ProgressT>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    int dim,
    const math::BBox<Vec3f> &bbox,
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D,
    float halfWidth,
    ProgressT *progress)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "polySoupToLevelSet requires an SDF grid with floating-point values");
    PolySoup poly{std::move(vtx), std::move(tri), std::move(quad), bbox};
    PolySoupToLevelSet<GridType> tmp(poly, dim, bbox, halfWidth);
    tmp.process(D, progress);
    return tmp.grids();
}

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType, class ShrinkWrapT, class ProgressT>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    float minVoxelSize,
    const math::BBox<Vec3f> &bbox,
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D,
    float halfWidth,
    ProgressT *progress)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "polySoupToLevelSet requires an SDF grid with floating-point values");
    PolySoup poly{std::move(vtx), std::move(tri), std::move(quad), bbox};
    PolySoupToLevelSet<GridType> tmp(poly, minVoxelSize, halfWidth);
    tmp.process(D, progress);
    return tmp.grids();
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED