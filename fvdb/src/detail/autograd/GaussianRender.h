// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_AUTOGRAD_GAUSSIANRENDER_H
#define FVDB_DETAIL_AUTOGRAD_GAUSSIANRENDER_H

#include <torch/all.h>
#include <torch/autograd.h>

namespace fvdb {
namespace detail {
namespace autograd {

struct SphericalHarmonics : public torch::autograd::Function<SphericalHarmonics> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx, const int sh_degree_to_use,
                                 const torch::optional<Variable> dirs, // [N, 3] or empty for deg 0
                                 const Variable                 &sh_coeffs, // [K, N, 3]
                                 const Variable                 &radii      // [N,]
    );

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

struct GaussianFullyFusedProjection
    : public torch::autograd::Function<GaussianFullyFusedProjection> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const Variable  &means,    // [N, 3]
                                 const Variable  &quats,    // [N, 4]
                                 const Variable  &scales,   // [N, 3]
                                 const Variable  &viewmats, // [C, 4, 4]
                                 const Variable  &Ks,       // [C, 3, 3]
                                 const uint32_t image_width, const uint32_t image_height,
                                 const float eps2d, const float near_plane, const float far_plane,
                                 const float radius_clip, const bool calc_compensations,
                                 const bool ortho);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

struct GaussianRasterizeToPixels : public torch::autograd::Function<GaussianRasterizeToPixels> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const Variable  &means2d,   // [C, N, 2]
                                 const Variable  &conics,    // [C, N, 3]
                                 const Variable  &colors,    // [C, N, 3]
                                 const Variable  &opacities, // [N]
                                 const uint32_t image_width, const uint32_t image_height,
                                 const uint32_t image_origin_w, const uint32_t image_origin_h,
                                 const uint32_t  tile_size,
                                 const Variable &tile_offsets, // [C, tile_height, tile_width]
                                 const Variable &flatten_ids,  // [n_isects]
                                 const bool      absgrad);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

struct GaussianFullyFusedProjectionJagged
    : public torch::autograd::Function<GaussianFullyFusedProjectionJagged> {
    using variable_list   = torch::autograd::variable_list;
    using AutogradContext = torch::autograd::AutogradContext;
    using Variable        = torch::autograd::Variable;

    static variable_list forward(AutogradContext *ctx,
                                 const Variable  &g_sizes,  // [B] gaussian sizes
                                 const Variable  &means,    // [ggz, 3]
                                 const Variable  &quats,    // [ggz, 4] optional
                                 const Variable  &scales,   // [ggz, 3] optional
                                 const Variable  &c_sizes,  // [B] camera sizes
                                 const Variable  &viewmats, // [ccz, 4, 4]
                                 const Variable  &Ks,       // [ccz, 3, 3]
                                 const uint32_t image_width, const uint32_t image_height,
                                 const float eps2d, const float near_plane, const float far_plane,
                                 const float radius_clip, const bool ortho);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

} // namespace autograd
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_AUTOGRAD_GAUSSIANRENDER_H
