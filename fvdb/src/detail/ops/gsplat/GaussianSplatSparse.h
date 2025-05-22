// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATSPARSE_H
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATSPARSE_H

#include <JaggedTensor.h>

#include <torch/torch.h>

#include <tuple>

namespace fvdb::detail::ops {

/// @brief Computes helper data structures for sparse gaussian splatting
///
/// @details This function computes four Tensors needed for sparse gaussian splatting:
///   For the following, let AP be the number of active pixels and AT be the number of active tiles.
///   1. active_tiles: An integer Tensor with shape [AT] indicating the tile_ids
///      corresponding to tiles which contain active pixels.
///   2. active_tile_mask: A Boolean Tensor of shape [C, TW, TH] indicating which tiles are active,
///      where C is the number of cameras, TW is the number of tiles along the width dimension, and
///      TH is the number of tiles along the height dimension.
///   3. tile_pixel_mask: An int64 Tensor of bitmasks with shape [AT, words_per_tile] where
///      words_per_tile is the number of int64_t words needed to make a bitmask for a tileSideLength
///      x tileSideLength tile. Bits are stored in raster order (top left to bottom right).
///   4. tile_pixel_cumsum: An int64 Tensor with shape [AT] encoding the cumulative sum of
///      active pixels in each active tile.
///   5. pixel_map: An int64 Tensor with shape [AP] specifying the write order of pixels within
///      tiles.
///        i.e. Suppose we're rendering the k^th active pixel in tile_id = active_tiles[t],
///             we write its rendered value to index pixel_map[tile_pixel_cumsum[tile_id-1] + k]
///             in the output
///
/// @param tileSideLength Number of pixels along one side of a square tile
/// @param numTilesW Number of tiles along width dimension
/// @param numTilesH Number of tiles along height dimension
/// @param pixelsToRender JaggedTensor containing pixel coordinates to render
///
/// @note The downstream rendering behavior is currently undefined for pixels
//       which are duplicates
/// @return Tuple of (active_tiles, active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
computeSparseInfo(const int32_t tileSideLength,
                  const int32_t numTilesW,
                  const int32_t numTilesH,
                  const fvdb::JaggedTensor &pixelsToRender);

/// @brief Computes the intersection between a set of 2D gaussians and a set of image tiles.
///        This function essentially "buckets" the gaussians into individual tiles, sorting them
///        by their depth values for downstream volume rendering.
/// @details The function returns two tensors:
///          1. tile_offsets: An integer tensor of shape [num_active_tiles + 1] where the i-th
///             element is the offset into intersection_values where first Gaussian in
///             active_tiles[i] begins.
///          2. intersection_values: An integer tensor of shape [num_intersections] where each
///             element is the index of a Gaussian which intersects a tile. Within a tile, these
///             Gaussians are sorted by their depth values.
///
/// @param means2d A float tensor of shape [C, N, 2] or [M, 2] where C is the number of cameras,
///                N is the number of Gaussians per camera, and M is the total number of Gaussians
///                if you're using a packed representation (i.e. variable Gaussians per camera).
/// @param radii A float tensor of shape [C, N] or [M] where C is the number of cameras, N is the
///              number of Gaussians per camera, and M is the total number of Gaussians if you're
///              using a packed representation.
/// @param depths A float tensor of shape [C, N] or [M] where C is the number of cameras, N is the
///               number of Gaussians per camera, and M is the total number of Gaussians if you're
///               using a packed representation.
/// @param tile_mask A boolean tensor of shape [C, H, W] where C is the number of cameras, H is the
///                  height of the image, and W is the width of the image.
/// @param active_tiles An integer tensor of shape [num_active_tiles] where each element is the
///                     tile_id of an active tile.
/// @param camera_ids An optional integer tensor of shape [M] where M is the total number of
///                   Gaussians. If provided, we assume that the Gaussians are packed and this
///                   tensor is used to index into the camera dimension of means2d, radii, and
///                   depths.
/// @param num_cameras The number of cameras in the scene.
/// @param tile_size The size of each tile in pixels.
/// @param num_tiles_h The number of tiles along the height (first) dimension.
/// @param num_tiles_w The number of tiles along the width (second) dimension.
///
/// @return Tuple of (tile_offsets, intersection_values)
template <c10::DeviceType>
std::tuple<torch::Tensor, torch::Tensor>
dispatchGaussianTileIntersectionSparse(const torch::Tensor &means2d,      // [C, N, 2] or [M, 2]
                                       const torch::Tensor &radii,        // [C, N] or [M]
                                       const torch::Tensor &depths,       // [C, N] or [M]
                                       const torch::Tensor &tile_mask,    // [C, H, W]
                                       const torch::Tensor &active_tiles, // [num_active_tiles]
                                       const at::optional<torch::Tensor> &camera_ids, // NULL or [M]
                                       const uint32_t num_cameras,
                                       const uint32_t tile_size,
                                       const uint32_t num_tiles_h,
                                       const uint32_t num_tiles_w);

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANSPLATSPARSE_H
