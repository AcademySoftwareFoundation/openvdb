// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth
///
/// @file tools/PolySoupToLevelSet.h
///
/// @brief Generates a LOD family of watertight shrink wrap level set surfaces
///        (or meshes) from a soup of polygons.
///
/// @details Details of this algorithm are given in an upcoming publication.

#ifndef OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Math.h>
#include <openvdb/util/Assert.h>

#include "Composite.h" // for csgUnion
#include "ValueTransformer.h"// for tools::foreach
#include "GridTransformer.h" // for resampleToMatch
#include "MeshToVolume.h"// for meshToLevelSet
#include "VolumeToMesh.h"// for volumeToMesh
#include "LevelSetFilter.h"// for Filter
#include "LevelSetMeasure.h"// for levelSetVolume
#include "FastSweeping.h"// for fogToSdf

#include <iostream>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Simple structure for a polygon soup
/// @details bbox is allowed to be invalid, in which case it will be derived from vtx
struct PolySoup {
    std::vector<Vec3s> vtx;
    std::vector<Vec3I> tri;
    std::vector<Vec4I> quad;
    math::BBox<Vec3f>  bbox;
};

/// @brief Class used to define and controle the shrink wrap behaviour.
/// @note This class is required to have a member method with the signature:
///       float operator()(float dx) const.
/// @details See D(dx) in our paper for details on the closing threshold.
class ShrinkWrapLimit;

/// @brief Class that implements the actual shrink wrap algorithm
/// @tparam GridType Template parameter of the desired shrink wrap grids
template <typename GridType = FloatGrid>
class PolySoupToLevelSet;

/// @brief Convert a soup of polygons to a shrink wrapped level set volume. This version
///        takes a PolySoup struct and optional voxel dimension and/or voxel size. If the
///        the voxel size is invalue, i.e. not positive, the dimension and bbox of the PolySoup
///        is used to derive the voxel size.
///
/// @return A shared pointer to grid of type @c GridType containing a narrow-band level set
///         that shrink wraps the input polygons.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Unlike tools::meshToLevelSet this method works for any polygons,
///         and does not require a closed surface.
///
/// @param poly       Struct with polygon soup, that will be destroyed (moved)
/// @param dim        Optional dimension in voxel units (assuming voxelSize is invalid)
/// @param voxelSize  Optional voxel size in world units (if invalid dim will be used instead)
/// @param D          Functor mapping voxel size to maximum allowed surface deformation
///                   allowed by shrink wrapping as a function of the voxel size
/// @param halfWidth  Half the width of the narrow band, in voxel units
/// @param progress   Optional pointer to progress bar
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
/// @param dim        Largest voxel dimension of the finest output grid
/// @param bbox       Bounding box of the vertices of the polygon mesh
/// @param vtx        Vector of world space vertex positions
/// @param tri        Vector of triangle indies
/// @param quads      Vector of quad indices
/// @param D          Functor mapping voxel size to maximum allowed surface deformation
///                   allowed by shrink wrapping as a function of the voxel size
/// @param halfWidth  Half the width of the narrow band, in voxel units
/// @param progress   Optional pointer to progress bar
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

/// @brief This class implements our shrink wrap algorithm. Normally the free-standing
///.       function called above should we used instead of this class.
/// @tparam GridType Grid type of the generated level set surfaces (defaults to FloatGrid)
template<typename GridType>
class PolySoupToLevelSet
{
public:

    /// @brief Constructor from a desired voxel dimension.
    /// @param poly  Polygon soup that will be moved to this instance.
    /// @param dim   Desired voxel dimension of the output level set.
    /// @param width Half-width of the output narrow-band level set, in voxel units.
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

    /// @brief Constructor from a desired voxel size.
    /// @param poly      Polygon soup that will be moved to this instance.
    /// @param voxelSize Desired voxel size of the output level set in world units.
    /// @param width     Half-width of the output narrow-band level set, in voxel units.
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

    /// @brief Performs the actual processing to generate the shrink wrap surfaces.
    /// @tparam  ShrinkWrapT Optional template parameter of the functor controlling
    //           the number of constrained erosion steps (see our paper).
    /// @tparam  ProgressT Template parameter of the optional progress bar.
    /// @param D ShrinkWrapT Optional functor controlling the number of constrained
    ///          erosion steps, and the closing threshold (see our paper).
    /// @param progress Optional pointer to a progress bar.
    template<class ShrinkWrapT, class ProgressT>
    void process(const ShrinkWrapT &D, ProgressT *progress);

    /// @brief Number of shrink wrap grids generated, i.e. depth of the LOD hierarchy. 
    size_t gridCount() const {return mGrids.size();}

    /// @brief   Returns a shared point to a particular shrink wrap grid
    /// @param n Number of the shrink wrap grid, where n = 0 has the finest sampling and
    ///          n = gridCount-1 is the coarsest voxel sampling.
    typename GridType::Ptr grid(int n = 0) const {return mGrids[n];}

    /// @brief Vector with shared pointers to all the shrink wrap grids
    /// @note  The grids are arrange fine to coarse, grids()[0] is the finest.
    std::vector<typename GridType::Ptr> grids() const {return mGrids;}

    /// @brief Generate an adaptive polygon mesh from a particular shrink wrap SDF
    /// @param n Number of the shrink wrap grid, where n = 0 has the finest sampling and
    ///          n = gridCount-1 is the coarsest voxel sampling.
    /// @param adaptivity Optional adaptivity parameter used for meshing. a value of zero
    ///                   means adaptivity is disables, i.e. uniform quads are produced 
    /// @param isoValue Iso-value used for the mesh generation.
    /// @return 
    const PolySoup& mesh(int n = 0, float adaptivity = 0.005f, float isoValue = 0.0f)
    {
        volumeToMesh(*mGrids[n], mPoly.vtx, mPoly.tri, mPoly.quad, isoValue, adaptivity);
        return mPoly;
    }

    /// @brief Static method that computes the bounding box of a list of vertex coordinates.
    /// @param vtx Vector of vertex coordinates.
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

    /// @brief Private method that generates a dx offset level set surface from mPoly, while also updating mPoly
    auto offset(float dx){
        auto xform = math::Transform::createLinearTransform(dx);
#if 0// no mesh <-> VDB round trips!
        auto grid = meshToUnsignedDistanceField<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth + 1);// mesh -> UDF
        struct OffsetOp {
            const float dx;
            OffsetOp(float _dx) : dx(_dx) {}
            inline void operator()(const typename GridType::ValueOnIter& it) const {it.setValue(*it - dx);}
        } op(dx);
        tools::foreach(grid->beginValueOn(), op, true, true);
        tools::changeBackground(grid->tree(), mHalfWidth*dx);
        grid->setGridClass(GRID_LEVEL_SET);
        return grid;
#else
        auto udf = meshToUnsignedDistanceField<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth);// mesh -> UDF
        volumeToMesh(*udf, mPoly.vtx, mPoly.tri, mPoly.quad, dx, 0.0);// UDF -> mesh (clears and re-allocates mesh)
        return meshToLevelSet<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth);// mesh -> SDF
#endif
    }

    /// @brief Private method that resamples inGrid(dx) to outGrid(dx/2)
    auto upsample(const GridType &inGrid){
        auto outGrid = createLevelSet<GridType>(inGrid.voxelSize()[0]/2, mHalfWidth);
        resampleToMatch<BoxSampler>(inGrid, *outGrid);
        mIsGridSDF = true;
        return outGrid;
    }

    /// @brief Performs the shrink wrap operation as a constrained level set erosion
    auto shrinkWrap(GridType &grid, const GridType &gridB, float &d){
        const float maxDist = 2.0f;
        LevelSetFilter<GridType> filter(grid);
#if 1
        filter.setSpatialScheme(math::FIRST_BIAS);
        filter.setTemporalScheme(math::TVD_RK1);
#else
        filter.setSpatialScheme(math::HJWENO5_BIAS);
        filter.setTemporalScheme(math::TVD_RK3);
#endif
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
    auto grid = mGrids.back();// initiate grid with the coarsest offset
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
    PolySoupToLevelSet<GridType> tmp(std::move(poly), dim, halfWidth);
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
    PolySoupToLevelSet<GridType> tmp(std::move(poly), minVoxelSize, halfWidth);
    tmp.process(D, progress);
    return tmp.grids();
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED