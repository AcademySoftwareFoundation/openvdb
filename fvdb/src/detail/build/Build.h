// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#ifndef FVDB_DETAIL_BUILD_BUILD_H
#define FVDB_DETAIL_BUILD_BUILD_H

#include <detail/GridBatchImpl.h>
#include <detail/VoxelCoordTransform.h>
#include <detail/utils/Utils.h>

#include <torch/all.h>

namespace fvdb {
namespace detail {
namespace build {

/// @brief Build an empty NanovVDB index grid or mutable index grid on the given device
/// @param device The device on which the grid will be allocated
/// @param isMutable Whether the grid should be mutable or not
/// @return A handle to the nanovdb grid
nanovdb::GridHandle<TorchDeviceBuffer> buildEmptyGrid(torch::Device device, bool isMutable);

/// @brief Build a NanoVDB grid encoding a dense 3D volume of voxels
/// @param device The device on which the grid will be allocated
/// @param isMutable Whether the grid should be mutable or not
/// @param batchSize The number of grids in the batch
/// @param size The size of the grid in voxels
/// @param ijkMin The coordinate of the bottom-back-left corner of the grid
/// @param mask An optional mask tensor that can be used to mask out some of the voxels (shape =
/// size)
/// @return A handle to the nanovdb grid
nanovdb::GridHandle<TorchDeviceBuffer> buildDenseGrid(torch::Device device, bool isMutable,
                                                      const uint32_t        batchSize,
                                                      const nanovdb::Coord &size,
                                                      const nanovdb::Coord &ijkMin,
                                                      const torch::optional<torch::Tensor> &mask);

/// @brief Build a NanoVDB grid representing the coarse grid of a given fine grid
/// @param isMutable Whether the grid should be mutable or not
/// @param fineGridHdl The handle to the fine grid
/// @param branchingFactor The coarsening factor from the fine grid to the coarse grid (i.e. N = [2,
/// 2, 2] for a 2x2x2 coarsening)
/// @return A handle to the nanovdb grid (the device will match fineGridHdl)
nanovdb::GridHandle<TorchDeviceBuffer>
buildCoarseGridFromFineGrid(bool isMutable, const GridBatchImpl &fineGridHdl,
                            const nanovdb::Coord branchingFactor);

/// @brief Build a NanoVDB grid representing the fine grid of a given coarse grid
/// @param isMutable Whether the grid should be mutable or not
/// @param coarseGridHdl The handle to the coarse grid
/// @param subdivMask An optional mask JaggedTensor that can be used to not refine certain voxels
/// (shape = [B, -1] matching number of coarse voxels)
/// @param subdivisionFactor The refinement factor from the coarse grid to the fine grid (i.e. (2,
/// 2, 2) for a 2x2x2 refinement)
/// @return A handle to the nanovdb grid (the device will match coarseGridHdl)
nanovdb::GridHandle<TorchDeviceBuffer>
buildFineGridFromCoarseGrid(bool isMutable, const GridBatchImpl &coarseGridHdl,
                            const torch::optional<JaggedTensor> &subdivMask,
                            const nanovdb::Coord                 subdivisionFactor);

nanovdb::GridHandle<TorchDeviceBuffer> buildConvGridFromGrid(bool                  isMutable,
                                                             const GridBatchImpl  &baseGridHdl,
                                                             const nanovdb::Coord &kernelSize,
                                                             const nanovdb::Coord &stride);

/// @brief Build a NanoVDB grid which is a padded version of the given grid
/// @param isMutable Whether the grid should be mutable or not
/// @param baseGridHdl The handle to the base grid
/// @param bmin The padding in the negative direction
/// @param bmax The padding in the positive direction
/// @param excludeBorder Whether to exclude the border voxels from padding
/// @return A handle to the padded nanovdb grid (the device will match baseGridHdl)
nanovdb::GridHandle<TorchDeviceBuffer> buildPaddedGridFromGrid(bool                 isMutable,
                                                               const GridBatchImpl &baseGridHdl,
                                                               int bmin, int bmax,
                                                               bool excludeBorder);

/// @brief Build a NanoVDB grid from a set of points and pad each voxel ijk which contains a point
/// from ijk - bmin to ijk + bmax
/// @param device The device on which the grid will be allocated
/// @param isMutable Whether the grid should be mutable or not
/// @param points The points to be encoded in the grid (JaggedTensor of shape = (B, -1, 3))
/// @param tx Transform from world to voxel coordinates
/// @param bmin The minimum padding (i.e. we pad ijk from ijk - bmin to ijk + bmax)
/// @param bmax The maximum padding (i.e. we pad ijk from ijk - bmin to ijk + bmax)
/// @return A handle to the nanovdb grid (the device will match points)
nanovdb::GridHandle<TorchDeviceBuffer>
buildPaddedGridFromPoints(bool isMutable, const JaggedTensor &points,
                          const std::vector<VoxelCoordTransform> &tx, const nanovdb::Coord &bmin,
                          const nanovdb::Coord &bmax);

/// @brief Build a NanoVDB grid from a set of points where the 8 nearest voxels to each point are
/// added to the grid
/// @param device The device on which the grid will be allocated
/// @param isMutable Whether the grid should be mutable or not
/// @param points The points to be encoded in the grid (JaggedTensor of shape = (B, -1, 3))
/// @param tx Transform from world to voxel coordinates
/// @return A handle to the nanovdb grid (the device will match points)
nanovdb::GridHandle<TorchDeviceBuffer>
buildNearestNeighborGridFromPoints(bool isMutable, const JaggedTensor &points,
                                   const std::vector<VoxelCoordTransform> &tx);

/// @brief Build a NanoVDB grid from a set of ijk coordinates pad each voxel from ijk - bmin to ijk
/// + bmax
/// @param device The device on which the grid will be allocated
/// @param isMutable Whether the grid should be mutable or not
/// @param coords The ijk coordinates to be encoded in the grid (JaggedTensor of shape = (B, -1, 3))
/// @param tx Transform from world to voxel coordinates
/// @param bmin The minimum padding (i.e. we pad ijk from ijk - bmin to ijk + bmax)
/// @param bmax The maximum padding (i.e. we pad ijk from ijk - bmin to ijk + bmax)
/// @return A handle to the nanovdb grid (the device will match coords)
nanovdb::GridHandle<TorchDeviceBuffer> buildPaddedGridFromCoords(bool                  isMutable,
                                                                 const JaggedTensor   &coords,
                                                                 const nanovdb::Coord &bmin,
                                                                 const nanovdb::Coord &bmax);

/// @brief Build a NanoVDB grid by voxelizing a mesh (i.e. each voxel in the ouput grid intersects
/// the mesh)
/// @param isMutable Whether the grid should be mutable or not
/// @param meshVertices A JaggedTensor of shape = (B, -1, 3) containing the vertices of each mesh to
/// voxelize
/// @param meshFaces A JaggedTensor of shape = (B, -1, 3) containing the face indexes of each mesh
/// to voxelize
/// @return A handle to the nanovdb grid (the device will match meshVertices and meshFaces)
nanovdb::GridHandle<TorchDeviceBuffer>
buildGridFromMesh(bool isMutable, const JaggedTensor meshVertices, const JaggedTensor meshFaces,
                  const std::vector<VoxelCoordTransform> &tx);

} // namespace build
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_BUILD_BUILD_H