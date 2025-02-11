// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianRender.h"

#include <detail/ops/Ops.h>
#include <detail/utils/Utils.h>

namespace fvdb {
namespace detail {
namespace autograd {

SphericalHarmonics::variable_list
SphericalHarmonics::forward(
    SphericalHarmonics::AutogradContext *ctx, const int sh_degree_to_use,
    const torch::optional<SphericalHarmonics::Variable> maybe_dirs, // (C, N, 3) or (N, 3)
    const SphericalHarmonics::Variable                 &sh_coeffs,  // (C, M, K, D) or (N, K, D)
    const SphericalHarmonics::Variable                 &radii       // (C, N) or (N,) (optional)
) {
    torch::Tensor dirs;
    if (maybe_dirs.has_value()) {
        dirs = maybe_dirs.value();
    } else {
        dirs = sh_coeffs.dim() == 3 ? torch::empty({ 0, 3 }) : torch::empty({ 0, 0, 3 });
    }
    Variable colors = FVDB_DISPATCH_KERNEL_DEVICE(sh_coeffs.device(), [&]() {
        return ops::dispatchSphericalHarmonicsForward<DeviceTag>(sh_degree_to_use, dirs, sh_coeffs,
                                                                 radii);
    });
    ctx->save_for_backward({ dirs, sh_coeffs, radii });
    ctx->saved_data["sh_degree_to_use"] = (int64_t)sh_degree_to_use;
    return { colors };
}

SphericalHarmonics::variable_list
SphericalHarmonics::backward(SphericalHarmonics::AutogradContext *ctx,
                             SphericalHarmonics::variable_list    grad_output) {
    Variable v_colors = grad_output.at(0);

    // ensure the gradients are contiguous if they are not None
    if (v_colors.defined()) {
        v_colors = v_colors.contiguous();
    }

    variable_list saved     = ctx->get_saved_variables();
    Variable      dirs      = saved.at(0);
    Variable      sh_coeffs = saved.at(1);
    Variable      radii     = saved.at(2);

    const int  sh_degree_to_use = (int)ctx->saved_data["sh_degree_to_use"].toInt();
    const bool compute_v_dirs   = ctx->needs_input_grad(1);

    auto     variables   = FVDB_DISPATCH_KERNEL_DEVICE(sh_coeffs.device(), [&]() {
        return ops::dispatchSphericalHarmonicsBackward<DeviceTag>(sh_degree_to_use, dirs, sh_coeffs,
                                                                        v_colors, radii, compute_v_dirs);
    });
    Variable v_sh_coeffs = std::get<0>(variables);
    Variable v_dirs;
    if (compute_v_dirs) {
        v_dirs = std::get<1>(variables);
    } else {
        v_dirs = Variable();
    }

    return { Variable(), v_dirs, v_sh_coeffs, Variable() };
}

GaussianFullyFusedProjection::variable_list
GaussianFullyFusedProjection::forward(GaussianFullyFusedProjection::AutogradContext *ctx,
                                      const GaussianFullyFusedProjection::Variable  &means,
                                      const GaussianFullyFusedProjection::Variable  &quats,
                                      const GaussianFullyFusedProjection::Variable  &scales,
                                      const GaussianFullyFusedProjection::Variable  &viewmats,
                                      const GaussianFullyFusedProjection::Variable  &Ks,
                                      const uint32_t image_width, const uint32_t image_height,
                                      const float eps2d, const float near_plane,
                                      const float far_plane, const float radius_clip,
                                      const bool calc_compensations, const bool ortho) {
    TORCH_CHECK(means.dim() == 2, "means must have shape (N, 3)");
    TORCH_CHECK(viewmats.dim() == 3, "viewmats must have shape (C, 4, 4)");
    TORCH_CHECK(Ks.dim() == 3, "Ks must have shape (C, 3, 3)");

    auto     variables = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionForward<DeviceTag>(
            means, quats, scales, viewmats, Ks, image_width, image_height, eps2d, near_plane,
            far_plane, radius_clip, calc_compensations, ortho);
    });
    Variable radii     = std::get<0>(variables);
    Variable means2d   = std::get<1>(variables);
    Variable depths    = std::get<2>(variables);
    Variable conics    = std::get<3>(variables);

    ctx->saved_data["image_width"]        = (int64_t)image_width;
    ctx->saved_data["image_height"]       = (int64_t)image_height;
    ctx->saved_data["eps2d"]              = (double)eps2d;
    ctx->saved_data["calc_compensations"] = (bool)calc_compensations;
    ctx->saved_data["ortho"]              = (bool)ortho;

    if (calc_compensations) {
        Variable compensations = std::get<4>(variables);
        ctx->save_for_backward(
            { means, quats, scales, viewmats, Ks, radii, conics, compensations });
        return { radii, means2d, depths, conics, compensations };
    } else {
        ctx->save_for_backward({ means, quats, scales, viewmats, Ks, radii, conics });
        return { radii, means2d, depths, conics };
    }
}

GaussianFullyFusedProjection::variable_list
GaussianFullyFusedProjection::backward(GaussianFullyFusedProjection::AutogradContext *ctx,
                                       GaussianFullyFusedProjection::variable_list    grad_output) {
    Variable v_radii   = grad_output.at(0);
    Variable v_means2d = grad_output.at(1);
    Variable v_depths  = grad_output.at(2);
    Variable v_conics  = grad_output.at(3);

    // ensure the gradients are contiguous if they are not None
    if (v_radii.defined()) {
        v_radii = v_radii.contiguous();
    }
    if (v_means2d.defined()) {
        v_means2d = v_means2d.contiguous();
    }
    if (v_depths.defined()) {
        v_depths = v_depths.contiguous();
    }
    if (v_conics.defined()) {
        v_conics = v_conics.contiguous();
    }

    variable_list saved    = ctx->get_saved_variables();
    Variable      means    = saved.at(0);
    Variable      quats    = saved.at(1);
    Variable      scales   = saved.at(2);
    Variable      viewmats = saved.at(3);
    Variable      Ks       = saved.at(4);
    Variable      radii    = saved.at(5);
    Variable      conics   = saved.at(6);

    const bool calc_compensations = ctx->saved_data["calc_compensations"].toBool();

    at::optional<Variable> compensations, v_compensations;
    if (calc_compensations) {
        Variable vcomp = grad_output.at(4);
        if (vcomp.defined()) {
            vcomp = vcomp.contiguous();
        }
        v_compensations = vcomp;
        compensations   = saved.at(7);
    }

    const int   image_width  = (int)ctx->saved_data["image_width"].toInt();
    const int   image_height = (int)ctx->saved_data["image_height"].toInt();
    const float eps2d        = (float)ctx->saved_data["eps2d"].toDouble();
    const bool  ortho        = (bool)ctx->saved_data["ortho"].toBool();

    auto     variables = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionBackward<DeviceTag>(
            means, quats, scales, viewmats, Ks, compensations, image_width, image_height, eps2d,
            radii, conics, v_means2d, v_depths, v_conics, v_compensations, ctx->needs_input_grad(4),
            ortho);
    });
    Variable v_means   = std::get<0>(variables);
    // Variable v_covars = std::get<1>(variables);
    Variable v_quats    = std::get<2>(variables);
    Variable v_scales   = std::get<3>(variables);
    Variable v_viewmats = std::get<4>(variables);

    return { v_means,    v_quats,    v_scales,   v_viewmats, Variable(), Variable(), Variable(),
             Variable(), Variable(), Variable(), Variable(), Variable(), Variable() };
}

GaussianRasterizeToPixels::variable_list
GaussianRasterizeToPixels::forward(
    GaussianRasterizeToPixels::AutogradContext *ctx,
    const GaussianRasterizeToPixels::Variable  &means2d,          // [C, N, 2]
    const GaussianRasterizeToPixels::Variable  &conics,           // [C, N, 3]
    const GaussianRasterizeToPixels::Variable  &colors,           // [C, N, 3]
    const GaussianRasterizeToPixels::Variable  &opacities,        // [N]
    const uint32_t image_width, const uint32_t image_height, const uint32_t image_origin_w,
    const uint32_t image_origin_h, const uint32_t tile_size,
    const GaussianRasterizeToPixels::Variable &tile_offsets,      // [C, tile_height, tile_width]
    const GaussianRasterizeToPixels::Variable &tile_gaussian_ids, // [n_isects]
    const bool                                 absgrad) {
    // const int C = means2d.size(0);
    // const int N = means2d.size(1);

    auto     variables     = FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeForward<DeviceTag>(
            means2d, conics, colors, opacities, image_width, image_height, image_origin_w,
            image_origin_h, tile_size, tile_offsets, tile_gaussian_ids);
    });
    Variable render_colors = std::get<0>(variables);
    Variable render_alphas = std::get<1>(variables);
    Variable last_ids      = std::get<2>(variables);

    ctx->save_for_backward({ means2d, conics, colors, opacities, tile_offsets, tile_gaussian_ids,
                             render_alphas, last_ids });
    ctx->saved_data["image_width"]    = (int64_t)image_width;
    ctx->saved_data["image_height"]   = (int64_t)image_height;
    ctx->saved_data["tile_size"]      = (int64_t)tile_size;
    ctx->saved_data["image_origin_w"] = (int64_t)image_origin_w;
    ctx->saved_data["image_origin_h"] = (int64_t)image_origin_h;
    ctx->saved_data["absgrad"]        = absgrad;

    return { render_colors, render_alphas };
}

GaussianRasterizeToPixels::variable_list
GaussianRasterizeToPixels::backward(GaussianRasterizeToPixels::AutogradContext *ctx,
                                    GaussianRasterizeToPixels::variable_list    grad_output) {
    Variable v_render_colors = grad_output.at(0);
    Variable v_render_alphas = grad_output.at(1);

    // ensure the gradients are contiguous if they are not None
    if (v_render_colors.defined())
        v_render_colors = v_render_colors.contiguous();
    if (v_render_alphas.defined())
        v_render_alphas = v_render_alphas.contiguous();

    variable_list saved             = ctx->get_saved_variables();
    Variable      means2d           = saved.at(0);
    Variable      conics            = saved.at(1);
    Variable      colors            = saved.at(2);
    Variable      opacities         = saved.at(3);
    Variable      tile_offsets      = saved.at(4);
    Variable      tile_gaussian_ids = saved.at(5);
    Variable      render_alphas     = saved.at(6);
    Variable      last_ids          = saved.at(7);

    const int  image_width    = (int)ctx->saved_data["image_width"].toInt();
    const int  image_height   = (int)ctx->saved_data["image_height"].toInt();
    const int  tile_size      = (int)ctx->saved_data["tile_size"].toInt();
    const int  image_origin_w = (int)ctx->saved_data["image_origin_w"].toInt();
    const int  image_origin_h = (int)ctx->saved_data["image_origin_h"].toInt();
    const bool absgrad        = ctx->saved_data["absgrad"].toBool();

    auto     variables = FVDB_DISPATCH_KERNEL_DEVICE(means2d.device(), [&]() {
        return ops::dispatchGaussianRasterizeBackward<DeviceTag>(
            means2d, conics, colors, opacities, image_width, image_height, image_origin_w,
            image_origin_h, tile_size, tile_offsets, tile_gaussian_ids, render_alphas, last_ids,
            v_render_colors, v_render_alphas, absgrad);
    });
    Variable v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = std::get<0>(variables);
        // means2d.absgrad = v_means2d_abs;
    } else {
        v_means2d_abs = Variable();
    }
    Variable v_means2d   = std::get<1>(variables);
    Variable v_conics    = std::get<2>(variables);
    Variable v_colors    = std::get<3>(variables);
    Variable v_opacities = std::get<4>(variables);

    return {
        v_means2d,  v_conics,   v_colors,   v_opacities, Variable(), Variable(),
        Variable(), Variable(), Variable(), Variable(),  Variable(), Variable(),
    };
}

GaussianFullyFusedProjectionJagged::variable_list
GaussianFullyFusedProjectionJagged::forward(
    GaussianFullyFusedProjectionJagged::AutogradContext *ctx,
    const GaussianFullyFusedProjectionJagged::Variable  &g_sizes,  // [B] gaussian sizes
    const GaussianFullyFusedProjectionJagged::Variable  &means,    // [ggz, 3]
    const GaussianFullyFusedProjectionJagged::Variable  &quats,    // [ggz, 4] optional
    const GaussianFullyFusedProjectionJagged::Variable  &scales,   // [ggz, 3] optional
    const GaussianFullyFusedProjectionJagged::Variable  &c_sizes,  // [B] camera sizes
    const GaussianFullyFusedProjectionJagged::Variable  &viewmats, // [ccz, 4, 4]
    const GaussianFullyFusedProjectionJagged::Variable  &Ks,       // [ccz, 3, 3]
    const uint32_t image_width, const uint32_t image_height, const float eps2d,
    const float near_plane, const float far_plane, const float radius_clip, const bool ortho) {
    auto     variables = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionJaggedForward<DeviceTag>(
            g_sizes, means, quats, scales, c_sizes, viewmats, Ks, image_width, image_height, eps2d,
            near_plane, far_plane, radius_clip, ortho);
    });
    Variable radii     = std::get<0>(variables);
    Variable means2d   = std::get<1>(variables);
    Variable depths    = std::get<2>(variables);
    Variable conics    = std::get<3>(variables);

    ctx->save_for_backward({ g_sizes, means, quats, scales, c_sizes, viewmats, Ks, radii, conics });
    ctx->saved_data["image_width"]  = (int64_t)image_width;
    ctx->saved_data["image_height"] = (int64_t)image_height;
    ctx->saved_data["eps2d"]        = (double)eps2d;
    ctx->saved_data["ortho"]        = (bool)ortho;

    return { radii, means2d, depths, conics };
}

GaussianFullyFusedProjectionJagged::variable_list
GaussianFullyFusedProjectionJagged::backward(
    GaussianFullyFusedProjectionJagged::AutogradContext *ctx,
    GaussianFullyFusedProjectionJagged::variable_list    grad_output) {
    Variable v_radii   = grad_output.at(0);
    Variable v_means2d = grad_output.at(1);
    Variable v_depths  = grad_output.at(2);
    Variable v_conics  = grad_output.at(3);

    // ensure the gradients are contiguous if they are not None
    if (v_radii.defined())
        v_radii = v_radii.contiguous();
    if (v_means2d.defined())
        v_means2d = v_means2d.contiguous();
    if (v_depths.defined())
        v_depths = v_depths.contiguous();
    if (v_conics.defined())
        v_conics = v_conics.contiguous();

    variable_list saved    = ctx->get_saved_variables();
    Variable      g_sizes  = saved.at(0);
    Variable      means    = saved.at(1);
    Variable      quats    = saved.at(2);
    Variable      scales   = saved.at(3);
    Variable      c_sizes  = saved.at(4);
    Variable      viewmats = saved.at(5);
    Variable      Ks       = saved.at(6);
    Variable      radii    = saved.at(7);
    Variable      conics   = saved.at(8);

    const int   image_width  = (int)ctx->saved_data["image_width"].toInt();
    const int   image_height = (int)ctx->saved_data["image_height"].toInt();
    const float eps2d        = (float)ctx->saved_data["eps2d"].toDouble();
    const bool  ortho        = (bool)ctx->saved_data["ortho"].toBool();

    auto     variables = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return ops::dispatchGaussianProjectionJaggedBackward<DeviceTag>(
            g_sizes, means, quats, scales, c_sizes, viewmats, Ks, image_width, image_height, eps2d,
            radii, conics, v_means2d, v_depths, v_conics, ctx->needs_input_grad(6), ortho);
    });
    Variable v_means   = std::get<0>(variables);
    // Variable v_covars = std::get<1>(variables);
    Variable v_quats    = std::get<2>(variables);
    Variable v_scales   = std::get<3>(variables);
    Variable v_viewmats = std::get<4>(variables);

    return { Variable(), v_means,    v_quats,    v_scales,   Variable(), v_viewmats, Variable(),
             Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable() };
}

} // namespace autograd
} // namespace detail
} // namespace fvdb
