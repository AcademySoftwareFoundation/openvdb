// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth
///
/// @file tools/PolySoupToLevelSet.h
///
/// @brief Generates a LOD family of watertight shrink wrap surfaces from a
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

class ShrinkWrapLimit;// ShrinkWrapLimit

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
/// @param maxVoxelSize coarsest/largest voxel size of the output grids
/// @param vtx          vector of world space vertex positions
/// @param tri          vector of triangle indies
/// @param quads        vector of quad indices
/// @param D            functor mapping voxel size to maximum allowed surface deformation
///                     allowed by shrink wrapping as a function of the voxel size
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType, class ShrinkWrapT = ShrinkWrapLimit>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    float minVoxelSize,// output voxel size
    float maxVoxelSize,// typically bbox_max / 2
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D = ShrinkWrapT(),
    float halfWidth = float(LEVEL_SET_HALF_WIDTH));

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
template<typename GridType, class ShrinkWrapT = ShrinkWrapLimit>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    int dim,
    const math::BBox<Vec3f> &bbox,
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D = ShrinkWrapT(),
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))    
{
    const float maxLength = bbox.extents()[bbox.maxExtent()];
    const float minVoxelSize = maxLength/(dim - 2.0f*(halfWidth + 1.0f));// +1 since final surface is dilated by dx
    const float maxVoxelSize = maxLength / 2.0f;
    return polySoupToLevelSet(minVoxelSize, maxVoxelSize, vtx, tri, quad, D, halfWidth);
}

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
template<typename GridType, class ShrinkWrapT = ShrinkWrapLimit>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    float minVoxelSize,
    const math::BBox<Vec3f> &bbox,
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D = ShrinkWrapT(),
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))    
{
    const float maxLength = bbox.extents()[bbox.maxExtent()];
    const float maxVoxelSize = maxLength / 2.0f;
    return polySoupToLevelSet(minVoxelSize, maxVoxelSize, vtx, tri, quad, D, halfWidth);
}

/////////////////////////////////////////////////////////////////////////////////////

template<typename GridType, class ShrinkWrapT>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    float minVoxelSize,
    float maxVoxelSize,
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    const ShrinkWrapT &D,
    float halfWidth)    
{
    if constexpr(!std::is_floating_point<typename GridType::ValueType>::value) {
        OPENVDB_THROW(TypeError, "polySoupToLevelSet: supported only for scalar floating-point grids");
    }
    OPENVDB_ASSERT(2*minVoxelSize <= maxVoxelSize);
    bool isGridSDF = true;
    auto myUpsample = [&](const GridType &grid){
      auto outGrid = createLevelSet<GridType>(grid.voxelSize()[0]/2, halfWidth);
      resampleToMatch<BoxSampler>(grid, *outGrid);
      isGridSDF = true;
      return outGrid;
    };// myUpsample

    auto myOffset = [&](float dx){
      auto xform = math::Transform::createLinearTransform(dx);
      auto udf = meshToUnsignedDistanceField<GridType>(*xform, vtx, tri, quad, halfWidth);// mesh -> UDF
      volumeToMesh<GridType>(*udf, vtx, tri, quad, dx, 0.0);// updates the mesh
      return meshToLevelSet<GridType>(*xform, vtx, tri, quad, halfWidth);
    };// myOffset

    auto myShrinkWrap = [&](GridType &grid, const GridType &gridB, float &d){
      const float maxDist = 2.0f;
      LevelSetFilter<GridType> filter(grid);
      filter.setSpatialScheme(math::FIRST_BIAS);
      filter.setTemporalScheme(math::TVD_RK1);
      if (isGridSDF == false) filter.normalize();
      filter.offset(maxDist * grid.voxelSize()[0]);// erode by maxDist * dx
      isGridSDF = false;// the CSG operation messed up the SDF
      d += maxDist;
      return csgUnionCopy(grid, gridB);
    };// myShrinkWrap

    // Fine to coarse offset generation
    std::vector<typename GridType::Ptr> grids;// fine -> coarse grids
    for (float dx = minVoxelSize; dx <= maxVoxelSize; dx *= 2.0f) grids.push_back(myOffset(dx));

    // Coarse to fine shrink wrap algorithm
    float vol[2];
    auto g = grids.back();// coarsest grid
    grids.pop_back();
    for (auto iter = grids.rbegin(), end = grids.rend(); iter != end; ++iter) {// coarse -> fine
      g = myUpsample(*g);// g(dx) -> g(dx/2)
      for (float d = 0.0f, dx = g->voxelSize()[0], Ddx = D(dx); d < Ddx; vol[0] = vol[1]) {
        g = myShrinkWrap(*g, **iter, d);
        vol[1] = levelSetVolume(*g);
        if (d>0.0f && math::isApproxZero(vol[0]-vol[1])) break;
      }
      *iter = g;
    }// loop from coarse to fine voxel sizes

    return grids;
}// tools::polySoupToLevelSet

//////////////////////////////////////////////////////////////////////////

class ShrinkWrapLimit {
    const float mErode, mThres;
public:
    ShrinkWrapLimit(float erode = 8.0f, float thres = 0.0f) : mErode(erode), mThres(thres) {} 
    float operator()(float dx) const {// if mThres == 0 this always returns mErode
        return dx>=2*mThres ? mErode : dx<=mThres ? 1.0f : 1.0f + (mErode-1.0f)*(dx-mThres)/mThres;
    }
};// ShrinkWrapLimit

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED