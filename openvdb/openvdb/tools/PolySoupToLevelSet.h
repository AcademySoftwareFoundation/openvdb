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
#include "LevelSetDilatedMesh.h"// for createLevelSetDilatedMesh
#include "LevelSetFilter.h"// for Filter
#include "LevelSetMeasure.h"// for levelSetVolume
#include "FastSweeping.h"// for fogToSdf
#include "LevelSetUtil.h" // for distanceFieldToSDF

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

/// @brief Class used to define and control the shrink wrap behaviour.
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
///        voxel size is invalid, i.e. not positive, the dimension and bbox of the PolySoup
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
    ProgressT *progress = nullptr,
    int offset_mode = 0);

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
/// @param tri        Vector of triangle indices
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
    ProgressT *progress = nullptr,
    int offset_mode = 0);

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
/// @param minVoxelSize Finest/smallest voxel size of the output grids
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
    ProgressT *progress = nullptr,
    int offset_mode = 0);

/////////////////////////////////////////////////////////////////////////////////////

/// @brief This class implements our shrink wrap algorithm. Normally the free-standing
///        function called above should be used instead of this class.
/// @tparam GridType Grid type of the generated level set surfaces (defaults to FloatGrid)
template<typename GridType>
class PolySoupToLevelSet
{
public:

    /// @brief Constructor from a desired voxel dimension.
    /// @param poly  Polygon soup that will be moved to this instance.
    /// @param dim   Desired voxel dimension of the output level set.
    /// @param width Half-width of the output narrow-band level set, in voxel units.
    PolySoupToLevelSet(PolySoup &&poly, int dim, float width = float(LEVEL_SET_HALF_WIDTH));

    /// @brief Constructor from a desired voxel size.
    /// @param poly      Polygon soup that will be moved to this instance.
    /// @param voxelSize Desired voxel size of the output level set in world units.
    /// @param width     Half-width of the output narrow-band level set, in voxel units.
    PolySoupToLevelSet(PolySoup &&poly, float voxelSize, float width = float(LEVEL_SET_HALF_WIDTH));

    /// @brief Performs the actual processing to generate the shrink wrap surfaces.
    /// @tparam  ShrinkWrapT Optional template parameter of the functor controlling
    ///          the number of constrained erosion steps (see our paper).
    /// @tparam  ProgressT Template parameter of the optional progress bar.
    /// @param D Optional functor controlling the number of constrained
    ///          erosion steps and the closing threshold (see our paper).
    /// @param progress Optional pointer to a progress bar.
    template<class ShrinkWrapT, class ProgressT>
    void process(const ShrinkWrapT &D, ProgressT *progress, int mode = 0);

    /// @brief Number of shrink wrap grids generated, i.e. depth of the LOD hierarchy.
    size_t gridCount() const {return mGrids.size();}

    /// @brief   Returns a shared pointer to a particular shrink wrap grid
    /// @param n Number of the shrink wrap grid, where n = 0 has the finest sampling and
    ///          n = gridCount-1 is the coarsest voxel sampling.
    typename GridType::Ptr grid(int n = 0) const {return mGrids[n];}

    /// @brief Vector with shared pointers to all the shrink wrap grids
    /// @note  The grids are arranged fine to coarse, grids()[0] is the finest.
    std::vector<typename GridType::Ptr> grids() const {return mGrids;}

    /// @brief Generate an adaptive polygon mesh from a particular shrink wrap SDF
    /// @param n Number of the shrink wrap grid, where n = 0 has the finest sampling and
    ///          n = gridCount-1 is the coarsest voxel sampling.
    /// @param adaptivity Optional adaptivity parameter used for meshing. A value of zero
    ///                   means adaptivity is disabled, i.e. uniform quads are produced.
    /// @param isoValue Iso-value used for the mesh generation.
    /// @return Reference to the internal PolySoup populated with the generated mesh.
    const PolySoup& mesh(int n = 0, float adaptivity = 0.005f, float isoValue = 0.0f)
    {
        volumeToMesh(*mGrids[n], mPoly.vtx, mPoly.tri, mPoly.quad, isoValue, adaptivity);
        return mPoly;
    }

    /// @brief Static method that computes the bounding box of a list of vertex coordinates.
    /// @param vtx Vector of vertex coordinates.
    static math::BBox<Vec3f> getBBox(const std::vector<Vec3s> &vtx);

    /// @brief Generates a dx-offset level set surface from the internal polygon soup.
    /// @param dx Voxel size (world units) of the output level set.
    /// @param mode 0) old method (using mesh -> UDF -> mesh -> SDF), 1) Mihai's
    ///             signed-flood-fill and 2) Greg's createLevelSetDilatedMesh
    /// @return Shared pointer to the newly created level set grid.
    auto offset(float dx, int mode = 0);

    float minVoxelSize() const {return mMinVoxelSize;}
    float maxVoxelSize() const {return mMaxVoxelSize;}
    float halfWidth() const {return mHalfWidth;}

private:
    PolySoup mPoly;
    float mMinVoxelSize, mMaxVoxelSize, mHalfWidth;
    std::vector<typename GridType::Ptr> mGrids;// fine(0) -> coarse grids(size-1)
    bool mIsGridSDF;

    /// @brief Private method that resamples inGrid(dx) to outGrid(dx/2).
    auto upsample(const GridType &inGrid);

    /// @brief Performs the shrink wrap operation as a constrained level set erosion.
    auto shrinkWrap(GridType &grid, const GridType &gridB, float &d);

};// PolySoupToLevelSet<GridType>

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType>
PolySoupToLevelSet<GridType>::PolySoupToLevelSet(PolySoup &&poly, int dim, float width)
    : mPoly(poly), mHalfWidth(width)
{
    if constexpr(!std::is_floating_point<typename GridType::ValueType>::value) {
        OPENVDB_THROW(TypeError, "polySoupToLevelSet: supported only for scalar floating-point grids");
    }
    if (!mPoly.bbox) mPoly.bbox = PolySoupToLevelSet::getBBox(mPoly.vtx);
    const float maxLength = mPoly.bbox.extents()[mPoly.bbox.maxExtent()];
    mMinVoxelSize = maxLength/(float(dim) - 2.0f*(mHalfWidth + 1.0f));// +1 since final surface is dilated by dx
    mMaxVoxelSize = maxLength / 2.0f;
    OPENVDB_ASSERT(2*mMinVoxelSize <= mMaxVoxelSize);
}// tools::PolySoupToLevelSet::PolySoupToLevelSet()

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType>
PolySoupToLevelSet<GridType>::PolySoupToLevelSet(PolySoup &&poly, float voxelSize, float width)
    : mPoly(poly), mMinVoxelSize(voxelSize), mHalfWidth(width)
{
    if constexpr(!std::is_floating_point<typename GridType::ValueType>::value) {
        OPENVDB_THROW(TypeError, "polySoupToLevelSet: supported only for scalar floating-point grids");
    }
    if (!mPoly.bbox) mPoly.bbox = PolySoupToLevelSet::getBBox(mPoly.vtx);
    const float maxLength = mPoly.bbox.extents()[mPoly.bbox.maxExtent()];
    mMaxVoxelSize = maxLength / 2.0f;
    OPENVDB_ASSERT(2*mMinVoxelSize <= mMaxVoxelSize);
}// tools::PolySoupToLevelSet::PolySoupToLevelSet()

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType>
template<class ShrinkWrapT, class ProgressT>
void PolySoupToLevelSet<GridType>::process(const ShrinkWrapT &D, ProgressT *progress, int offset_mode)
{
    auto myProgress = [&](const std::string &s){if constexpr(!std::is_same<ProgressT,void>::value) if (progress) (*progress)(s);};

    if (progress) std::cerr << std::endl;

    // Fine to coarse offset generation
    for (float dx = mMinVoxelSize; dx <= mMaxVoxelSize; dx *= 2.0f) {
        myProgress("Offset: dx=" + std::to_string(dx)+", range: "+std::to_string(mMinVoxelSize)+" -> "+std::to_string(mMaxVoxelSize));
        mGrids.push_back(this->offset(dx, offset_mode));
    }

    // Coarse to fine shrink wrap algorithm
    double vol[2] = {0.0, 0.0};// levelSetVolume returns Real (double); keep full precision.
                               // Zero-init silences a GCC -Wmaybe-uninitialized false positive:
                               // vol[0] is only read when d>0, after the loop's increment has set it.
    auto grid = mGrids.back();// initiate grid with the coarsest offset
    mGrids.pop_back();
    mIsGridSDF = true;
    for (auto iter = mGrids.rbegin(), end = mGrids.rend(); iter != end; ++iter) {// coarse -> fine
      grid = this->upsample(*grid);// grid(dx) -> grid(dx/2)
      for (float d = 0.0f, dx = float(grid->voxelSize()[0]), Ddx = D(dx); d < Ddx; vol[0] = vol[1]) {
        myProgress("Shrink wrap d=" + std::to_string(d) + ", D("+std::to_string(dx) + ")=" + std::to_string(Ddx));
        grid = this->shrinkWrap(*grid, **iter, d);
        vol[1] = levelSetVolume(*grid);
        if (d>0.0f && math::isApproxZero(vol[0]-vol[1])) break;
      }
      *iter = grid;
    }// loop from coarse to fine voxel sizes

}// tools::PolySoupToLevelSet::process()

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
math::BBox<Vec3f> PolySoupToLevelSet<GridType>::getBBox(const std::vector<Vec3s> &vtx)
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
}// tools::PolySoupToLevelSet::getBBox

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
auto PolySoupToLevelSet<GridType>::offset(float dx, int mode)
{
    auto xform = math::Transform::createLinearTransform(dx);
    typename GridType::Ptr grid(nullptr);
    switch (mode) {
    case 0:// algorithm presented in the paper, using mesh<-> VDB round-trip
        grid = meshToUnsignedDistanceField<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth);// mesh -> UDF
        volumeToMesh(*grid, mPoly.vtx, mPoly.tri, mPoly.quad, /*iso*/dx, /*adapt*/0.0);// UDF -> mesh (clears and re-allocates mesh)
        grid = meshToLevelSet<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth);// mesh -> SDF
        break;
    case 1:// algorithm using Mihai's signed flood-fill algorithm
        grid = meshToUnsignedDistanceField<GridType>(*xform, mPoly.vtx, mPoly.tri, mPoly.quad, mHalfWidth + 1);// mesh -> UDF
        tools::foreach(grid->beginValueOn(), [dx](const typename GridType::ValueOnIter& it){it.setValue(*it - dx);}, /*threaded*/true, /*share functor*/true);
        //tools::changeBackground(grid->tree(), mHalfWidth*dx);
        tools::changeLevelSetBackground(grid->tree(), mHalfWidth);
        //grid->tree().root().setBackground(exteriorWidth, /*updateChildNodes=*/true);
        //tools::signedFloodFillWithValues(grid->tree(), exteriorWidth, interiorWidth);
        tools::distanceFieldToSDF(*grid, /*removeDisconnectedInterior*/true, /*rebuildNarrowBand*/true);
        break;
    case 2:// algorithm using Greg's polyOffset algorithm
        grid = tools::createLevelSetDilatedMesh<GridType, float>(mPoly.vtx, mPoly.tri, mPoly.quad, /*radius*/dx, /*voxel size*/dx, mHalfWidth);
        //tools::distanceFieldToSDF(*grid, /*removeDisconnectedInterior*/true, /*rebuildNarrowBand*/false);
        break;
    default:
        OPENVDB_THROW(TypeError, "polySoupToLevelSet::offset: invalid mode(" + std::to_string(mode) + ")");
        break;
    }// end of switch
    return grid;
}// tools::PolySoupToLevelSet<GridType>::offset

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
auto PolySoupToLevelSet<GridType>::upsample(const GridType &inGrid)
{
    auto outGrid = createLevelSet<GridType>(inGrid.voxelSize()[0]/2, mHalfWidth);
    resampleToMatch<BoxSampler>(inGrid, *outGrid);
    mIsGridSDF = true;
    return outGrid;
}// tools::PolySoupToLevelSet<GridType>::upsample

//////////////////////////////////////////////////////////////////////////

template<typename GridType>
auto PolySoupToLevelSet<GridType>::shrinkWrap(GridType &grid, const GridType &gridB, float &d)
{
    const float maxDist = 2.0f;
    LevelSetFilter<GridType> filter(grid);
    filter.setNormCount(3);// halfWidth
#if 1//first-order
    filter.setSpatialScheme(math::FIRST_BIAS);
    filter.setTemporalScheme(math::TVD_RK1);
#else// higher order
    filter.setSpatialScheme(math::HJWENO5_BIAS);
    filter.setTemporalScheme(math::TVD_RK3);
#endif
    if (mIsGridSDF == false) {
        filter.normalize();
        filter.prune();// is this needed?
    }
    filter.offset(static_cast<typename GridType::ValueType>(maxDist * grid.voxelSize()[0]));// erode by maxDist * dx
    mIsGridSDF = false;// the CSG operation messed up the SDF
    d += maxDist;
    return csgUnionCopy(grid, gridB);
}// tools::PolySoupToLevelSet<GridType>::shrinkWrap

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
    ProgressT *progress,
    int offset_mode)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "polySoupToLevelSet requires an SDF grid with floating-point values");
    using T = PolySoupToLevelSet<GridType>;
    auto ptr = voxelSize > 0.0f ? std::make_unique<T>(std::move(poly), voxelSize, halfWidth) :
                                  std::make_unique<T>(std::move(poly), dim, halfWidth);
    ptr->process(D, progress, offset_mode);
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
    ProgressT *progress,
    int offset_mode)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "polySoupToLevelSet requires an SDF grid with floating-point values");
    PolySoup poly{std::move(vtx), std::move(tri), std::move(quad), bbox};
    PolySoupToLevelSet<GridType> tmp(std::move(poly), dim, halfWidth);
    tmp.process(D, progress, offset_mode);
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
    ProgressT *progress,
    int offset_mode)
{
    static_assert(std::is_floating_point<typename GridType::ValueType>::value,
        "polySoupToLevelSet requires an SDF grid with floating-point values");
    PolySoup poly{std::move(vtx), std::move(tri), std::move(quad), bbox};
    PolySoupToLevelSet<GridType> tmp(std::move(poly), minVoxelSize, halfWidth);
    tmp.process(D, progress, offset_mode);
    return tmp.grids();
}// polySoupToLevelSet

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED