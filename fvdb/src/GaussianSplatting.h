// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_GAUSSIANSPLATTING_H
#define FVDB_GAUSSIANSPLATTING_H

#include <torch/all.h>
#include <unordered_map>

#include "JaggedTensor.h"

namespace fvdb {

torch::Tensor evaluateSphericalHarmonics(const torch::Tensor &directions,
                                         const torch::Tensor &sh_coeffs,
                                         const torch::Tensor  radii            = torch::Tensor(),
                                         const int            sh_degree_to_use = -1);

// Precompute values used to render images from a set of cameras
std::unordered_map<std::string, torch::Tensor>
precomputeGaussianRenderStateUnbatched(const torch::Tensor &means,     // [N1 + N2 + ..., 3]
                                       const torch::Tensor &quats,     // [N1 + N2 + ..., 4]
                                       const torch::Tensor &scales,    // [N1 + N2 + ..., 3]
                                       const torch::Tensor &opacities, // [N1 + N2 + ...]
                                       const torch::Tensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
                                       const torch::Tensor &viewmats,  // [C1 + C2 + ..., 4, 4]
                                       const torch::Tensor &Ks,        // [C1 + C2 + ..., 3, 3]
                                       const uint32_t image_width, const uint32_t image_height,
                                       const float near_plane = 0.01, const float far_plane = 1e10,
                                       const int sh_degree_to_use = -1, const int tile_size = 16,
                                       const float radius_clip = 0.0, const float eps2d = 0.3,
                                       bool antialias = false, bool render_depth_channel = false);

std::vector<torch::Tensor> projectGaussiansToImages(
    const torch::Tensor &means, const torch::Tensor &quats, const torch::Tensor &scales,
    const torch::Tensor &viewmats, const torch::Tensor &Ks, const uint32_t image_width,
    const uint32_t image_height, const float near_plane = 0.01, const float far_plane = 1e10,
    const float radius_clip = 0.0, const float eps2d = 0.3, bool antialias = false);

std::tuple<torch::Tensor, torch::Tensor> renderPixelsFromPrecomputedGaussianRenderStateUnbatched(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors, torch::Tensor opacities,
    uint32_t image_width, uint32_t image_height, uint32_t image_origin_w, uint32_t image_origin_h,
    uint32_t tile_size, torch::Tensor tile_offsets, torch::Tensor tile_gaussian_ids);

std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRender(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
               const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
               const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
               const JaggedTensor &opacities, // [N1 + N2 + ...]
               const JaggedTensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
               const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
               const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
               const uint32_t image_width, const uint32_t image_height,
               const float near_plane = 0.01, const float far_plane = 1e10,
               const int sh_degree_to_use = -1, const int tile_size = 16,
               const float radius_clip = 0.0, const float eps2d = 0.3, bool antialias = false,
               bool render_depth_channel = false, bool return_debug_info = false);

std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRenderDepth(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
                    const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
                    const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
                    const JaggedTensor &opacities, // [N1 + N2 + ...]
                    const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
                    const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
                    const uint32_t image_width, const uint32_t image_height,
                    const float near_plane = 0.01, const float far_plane = 1e10,
                    const int tile_size = 16, const float radius_clip = 0.0,
                    const float eps2d = 0.3, bool antialias = false,
                    bool return_debug_info = false);

} // namespace fvdb

#endif // FVDB_GAUSSIANSPLATTING_H