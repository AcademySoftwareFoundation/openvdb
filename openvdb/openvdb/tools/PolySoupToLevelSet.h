// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @author Ken Museth
///
/// @file tools/PolySoupToLevelSet.h
///
/// @brief Generates a LOD family of watertight shrink wrap surfaces from a
///        a soup of polygons.
///
/// @details The details of this algorithm given in an upcoming publication

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

/// @brief Convert a triangle and quad mesh to a level set volume.
///
/// @return a grid of type @c GridType containing a narrow-band level set
///         representation of the input mesh.
///
/// @throw  TypeError if @c GridType is not scalar or not floating-point
///
/// @note   Requires a closed surface but not necessarily a manifold surface.
///         Supports surfaces with self intersections and degenerate faces
///         and is independent of mesh surface normals.
///
/// @param xform        transform for the output grid
/// @param points       list of world space vertex positions
/// @param triangles    triangle index list
/// @param quads        quad index list
/// @param halfWidth    half the width of the narrow band, in voxel units
template<typename GridType>
std::vector<typename GridType::Ptr>
polySoupToLevelSet(
    float minVoxelSize,// output voxel size
    float maxVoxelSize,// typically bbox_max / 2
    std::vector<Vec3s>& vtx,
    std::vector<Vec3I>& tri,
    std::vector<Vec4I>& quad,
    float nErode = 8.0f,
    float thres = 0.0f,
    float halfWidth = float(LEVEL_SET_HALF_WIDTH))
{
    auto D = [&](float dx)->float{
      if (dx >= 2*thres) return nErode;// if thres == 0 this is aways true
      if (dx <=   thres) return 1.0f;
      return 1.0f + (nErode-1.0f)*(dx-thres)/thres;
    };

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
    std::vector<typename GridType::Ptr> grids;// level-of-detail grids
    for (float dx = minVoxelSize; dx <= maxVoxelSize; dx *= 2.0f) grids.push_back(myOffset(dx));

    // Coarse to fine shrink wrap algorithm
    float vol[2];
    auto g = grids.back();// coarset grid
    grids.pop_back();
    for (auto iter = grids.rbegin(), end = grids.rend(); iter != end; ++iter) {// coarse -> fine
      g = myUpsample(*g);// g(dx) -> g(dx/2)
      for (float d = 0.0f, dx = g->voxelSize()[0], Ddx = D(dx); d < Ddx; vol[0] = vol[1]) {
        g = myShrinkWrap(*g, **iter, d);
        vol[1] = levelSetVolume(*g);
        if (d>0.0f && math::Abs(vol[0]-vol[1]) == 0.0f ) break;
      }
      *iter = g;
    }// loop from coarse to fine voxel sizes

    return grids;
}// tools::polySoupToLevelSet

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_POLYSOUP_TO_LEVELSET_HAS_BEEN_INCLUDED
