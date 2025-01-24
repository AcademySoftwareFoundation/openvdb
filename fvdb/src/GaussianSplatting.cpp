// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianSplatting.h"

#include <detail/autograd/Autograd.h>
#include <detail/ops/Ops.h>

namespace fvdb {

namespace {

torch::Tensor
evaluateSphericalHarmonics(const torch::optional<torch::Tensor> directions,
                           const torch::Tensor                 &sh_coeffs,
                           const torch::Tensor                  radii            = torch::Tensor(),
                           const int                            sh_degree_to_use = -1) {
    const int K                = sh_coeffs.size(0); // number of SH bases
    const int actual_sh_degree = sh_degree_to_use < 0 ? (std::sqrt(K) - 1) : sh_degree_to_use;
    TORCH_CHECK(K >= (actual_sh_degree + 1) * (actual_sh_degree + 1),
                "K must be at least (sh_degree_to_use + 1)^2");
    auto sh_results =
        detail::autograd::SphericalHarmonics::apply(actual_sh_degree, directions, sh_coeffs, radii);

    return sh_results[0];
}

std::array<torch::Tensor, 7>
computeGaussianRenderStateUnbatched(const torch::Tensor &means, const torch::Tensor &quats,
                                    const torch::Tensor &scales, const torch::Tensor &opacities,
                                    const torch::Tensor &sh_coeffs, const torch::Tensor &viewmats,
                                    const torch::Tensor &Ks, const uint32_t image_width,
                                    const uint32_t image_height, const float near_plane,
                                    const float far_plane, const int sh_degree_to_use,
                                    const int tile_size, const float radius_clip, const float eps2d,
                                    const bool antialias, const bool render_depth_channel,
                                    const bool render_depth_only, const bool ortho) {
    const int C = viewmats.size(0);                           // number of cameras
    const int N = means.size(0);                              // number of gaussians
    const int K = render_depth_only ? 1 : sh_coeffs.size(0);  // number of SH bases
    const int D = render_depth_only ? 1 : sh_coeffs.size(-1); // Dimension of output

    TORCH_CHECK(means.sizes() == torch::IntArrayRef({ N, 3 }), "means must have shape (N, 3)");
    TORCH_CHECK(quats.sizes() == torch::IntArrayRef({ N, 4 }), "quats must have shape (N, 4)");
    TORCH_CHECK(scales.sizes() == torch::IntArrayRef({ N, 3 }), "scales must have shape (N, 3)");
    TORCH_CHECK(opacities.sizes() == torch::IntArrayRef({ N }), "opacities must have shape (N)");
    TORCH_CHECK(viewmats.sizes() == torch::IntArrayRef({ C, 4, 4 }),
                "viewmats must have shape (C, 4, 4)");
    TORCH_CHECK(Ks.sizes() == torch::IntArrayRef({ C, 3, 3 }), "Ks must have shape (C, 3, 3)");
    TORCH_CHECK(render_depth_only || sh_coeffs.sizes() == torch::IntArrayRef({ K, N, D }),
                "sh_coeffs must have shape (K, N, D)");

    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(quats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(opacities.is_contiguous(), "opacities must be contiguous");
    TORCH_CHECK(viewmats.is_contiguous(), "viewmats must be contiguous");
    TORCH_CHECK(Ks.is_contiguous(), "Ks must be contiguous");

    // Project to image plane [differentiable]
    const auto projection_results = detail::autograd::GaussianFullyFusedProjection::apply(
        means, quats, scales, viewmats, Ks, image_width, image_height, eps2d, near_plane, far_plane,
        radius_clip, antialias, ortho);
    const torch::Tensor radii                 = projection_results[0];
    const torch::Tensor means2d               = projection_results[1];
    const torch::Tensor depths                = projection_results[2];
    const torch::Tensor conics                = projection_results[3];
    torch::Tensor       opacities_compensated = opacities.repeat({ C, 1 });
    if (antialias) {
        opacities_compensated *= projection_results[4];
        // FIXME (Francis): The contiguity requirement is dumb and should be
        // removed by using accessors in the kernel
        opacities_compensated = opacities_compensated.contiguous();
    }

    torch::Tensor colors;
    if (render_depth_only) {
        colors = depths.unsqueeze(-1); // [C, N, 1]
    } else {
        if (K == 1 || sh_degree_to_use == 0) {
            // Handle the case where we only have degree zero spherical harmonics, which just
            // represent diffuse colors. This means that each Gaussian receives the same color
            // regardless of which camera sees it, and we can just expand the colors to the correct
            // shape (without reallocating memory). i.e. the color tensor has shape [C, N, D] but
            // only allocates NxD floats in memory.
            // This is useful for rendering e.g. high dimensional diffuse features.
            colors = evaluateSphericalHarmonics(torch::nullopt, sh_coeffs.unsqueeze(1), radii,
                                                sh_degree_to_use);
            colors = colors.expand({ C, -1, -1 });

        } else {
            // FIXME (Francis): Do this in the kernel instead of materializing a large
            //                  tensor here. It's a bit annoying because we'll have to update
            //                  the current backward pass
            const torch::Tensor camtoworlds = torch::inverse(viewmats);
            const torch::Tensor dirs =
                means.index(
                    { torch::indexing::None, torch::indexing::Slice(), torch::indexing::Slice() }) -
                camtoworlds.index({ torch::indexing::Slice(), torch::indexing::None,
                                    torch::indexing::Slice(0, 3), 3 }); // [1, N, 3] - [C, 1, 3]
            colors = evaluateSphericalHarmonics(
                dirs, sh_coeffs.unsqueeze(1).expand({ -1, C, -1, -1 }), radii, sh_degree_to_use);
        }

        if (render_depth_channel) {
            colors = torch::cat({ colors, depths.unsqueeze(-1) }, -1); // [C, N, D + 1]
        }
    }

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int num_tiles_w = std::ceil(image_width / static_cast<float>(tile_size));
    const int num_tiles_h = std::ceil(image_height / static_cast<float>(tile_size));
    const int num_cameras = viewmats.size(0);
    const std::tuple<torch::Tensor, torch::Tensor> tile_intersections =
        FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
            return detail::ops::dispatchGaussianTileIntersection<DeviceTag>(
                means2d, radii, depths, at::nullopt, num_cameras, tile_size, num_tiles_h,
                num_tiles_w);
        });
    const torch::Tensor tile_offsets      = std::get<0>(tile_intersections); // [C, TH, TW]
    const torch::Tensor tile_gaussian_ids = std::get<1>(tile_intersections); // [M]
    return {
        means2d, conics, opacities_compensated, radii, colors, tile_offsets, tile_gaussian_ids
    };
}

// Gaussian render for a single torch Tensor
std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRenderUnbatchedInternal(const torch::Tensor &means, const torch::Tensor &quats,
                                const torch::Tensor &scales, const torch::Tensor &opacities,
                                const torch::Tensor &sh_coeffs, const torch::Tensor &viewmats,
                                const torch::Tensor &Ks, const uint32_t image_width,
                                const uint32_t image_height, const float near_plane,
                                const float far_plane, const int sh_degree_to_use,
                                const int tile_size, const float radius_clip, const float eps2d,
                                const bool antialias, const bool render_depth_channel,
                                const bool return_debug_info, const bool render_depth_only,
                                const bool ortho) {
    std::array<torch::Tensor, 7> renderState = computeGaussianRenderStateUnbatched(
        means, quats, scales, opacities, sh_coeffs, viewmats, Ks, image_width, image_height,
        near_plane, far_plane, sh_degree_to_use, tile_size, radius_clip, eps2d, antialias,
        render_depth_channel, render_depth_only, ortho);

    torch::Tensor means2d               = renderState[0];
    torch::Tensor conics                = renderState[1];
    torch::Tensor opacities_compensated = renderState[2];
    torch::Tensor radii                 = renderState[3];
    torch::Tensor colors                = renderState[4];
    torch::Tensor tile_offsets          = renderState[5];
    torch::Tensor tile_gaussian_ids     = renderState[6];

    std::unordered_map<std::string, torch::Tensor> info;
    if (return_debug_info) {
        info["means2d"]           = means2d;
        info["conics"]            = conics;
        info["opacities"]         = opacities;
        info["radii"]             = radii;
        info["colors"]            = colors;
        info["tile_offsets"]      = tile_offsets;
        info["tile_gaussian_ids"] = tile_gaussian_ids;
    }

    // Rasterize projected Gaussians to pixels [differentiable]
    auto outputs = detail::autograd::GaussianRasterizeToPixels::apply(
        means2d, conics, colors, opacities_compensated, image_width, image_height, 0, 0, tile_size,
        tile_offsets, tile_gaussian_ids, false);
    torch::Tensor render_colors = outputs[0];
    torch::Tensor render_alphas = outputs[1];

    return { render_colors, render_alphas, info };
}

std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRenderInternal(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
                       const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
                       const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
                       const JaggedTensor &opacities, // [N1 + N2 + ...]
                       const JaggedTensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
                       const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
                       const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
                       const uint32_t image_width, const uint32_t image_height,
                       const float near_plane, const float far_plane, const int sh_degree_to_use,
                       const int tile_size, const float radius_clip, const float eps2d,
                       const bool antialias, const bool render_depth_channel,
                       const bool return_debug_info, const bool render_depth_only,
                       const bool ortho) {
    const int ccz = viewmats.rsize(0);                           // number of cameras
    const int ggz = means.rsize(0);                              // number of gaussians
    const int D   = render_depth_only ? 1 : sh_coeffs.rsize(-1); // Dimension of output

    using namespace torch::indexing;                             // For the Slice operation

    TORCH_CHECK(means.rsizes() == torch::IntArrayRef({ ggz, 3 }), "means must have shape (ggz, 3)");
    TORCH_CHECK(quats.rsizes() == torch::IntArrayRef({ ggz, 4 }), "quats must have shape (ggz, 4)");
    TORCH_CHECK(scales.rsizes() == torch::IntArrayRef({ ggz, 3 }),
                "scales must have shape (ggz, 3)");
    TORCH_CHECK(opacities.rsizes() == torch::IntArrayRef({ ggz }),
                "opacities must have shape (ggz)");
    TORCH_CHECK(viewmats.rsizes() == torch::IntArrayRef({ ccz, 4, 4 }),
                "viewmats must have shape (C, 4, 4)");
    TORCH_CHECK(Ks.rsizes() == torch::IntArrayRef({ ccz, 3, 3 }), "Ks must have shape (ccz, 3, 3)");

    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(quats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(opacities.is_contiguous(), "opacities must be contiguous");
    TORCH_CHECK(viewmats.is_contiguous(), "viewmats must be contiguous");
    TORCH_CHECK(Ks.is_contiguous(), "Ks must be contiguous");

    if (means.num_tensors() == 1) {
        return gaussianRenderUnbatchedInternal(
            means.jdata(), quats.jdata(), scales.jdata(), opacities.jdata(), sh_coeffs.jdata(),
            viewmats.jdata(), Ks.jdata(), image_width, image_height, near_plane, far_plane,
            sh_degree_to_use, tile_size, radius_clip, eps2d, antialias, render_depth_channel,
            return_debug_info, render_depth_only, ortho);
    }

    TORCH_CHECK_NOT_IMPLEMENTED(!ortho, "orthographic projection not implemented")
    // Check after we dispatch the unbatched version since the unbatched version accepts a
    // [K, N, D] tensor for sh_coeffs while the batched version accepts a [ggz, K, D] tensor,
    // which gets permuted later on.
    const int K = render_depth_only ? 1 : sh_coeffs.rsize(-2); // number of SH bases
    TORCH_CHECK(render_depth_only || sh_coeffs.rsizes() == torch::IntArrayRef({ ggz, K, D }),
                "sh_coeffs must have shape (ggz, K, D)");

    // TODO: this part is very convoluted. But I don't have a better way of coding it without
    // customized CUDA kernels. The idea is that given Gaussians with shape [\sum(N_i), ...] and
    // cameras with shape [\sum(C_i), ...], we would calculate the intersection of each Gaussian
    // with each camera, which result in a JaggedTensor with shape
    // [\sum(C_i * N_i), ...]. And I need to keep track of the camera and Gaussian IDs (the index in
    // the jagged tensor) for each intersection:
    // - camera_ids: Shape of [\sum(C_i * N_i), ...], with each value \in [0, \sum(C_i))
    // - gaussian_ids: Shape of [\sum(C_i * N_i), ...], with each value \in [0, \sum(N_i))

    // g_sizes is [N1, N2, ...]
    torch::Tensor g_sizes =
        means.joffsets().index({ Slice(1, None) }) - means.joffsets().index({ Slice(0, -1) });
    // c_sizes is [C1, C2, ...]
    torch::Tensor c_sizes =
        Ks.joffsets().index({ Slice(1, None) }) - Ks.joffsets().index({ Slice(0, -1) });
    // camera_ids is [0, 0, ..., 1, 1, ...]
    torch::Tensor tt = g_sizes.repeat_interleave(c_sizes);
    torch::Tensor camera_ids =
        torch::arange(viewmats.rsize(0), means.options().dtype(torch::kInt32))
            .repeat_interleave(tt, 0);
    // gaussian_ids is [0, 1, ..., 0, 1, ...]
    torch::Tensor dd0    = means.joffsets().index({ Slice(0, -1) }).repeat_interleave(c_sizes, 0);
    torch::Tensor dd1    = means.joffsets().index({ Slice(1, None) }).repeat_interleave(c_sizes, 0);
    torch::Tensor shifts = dd0.index({ Slice(1, None) }) - dd1.index({ Slice(0, -1) });
    shifts               = torch::cat({ torch::tensor({ 0 }, means.device()), shifts });
    torch::Tensor shifts_cumsum = shifts.cumsum(0);
    torch::Tensor gaussian_ids =
        torch::arange(camera_ids.size(0), means.options().dtype(torch::kInt32));
    gaussian_ids += shifts_cumsum.repeat_interleave(tt, 0);

    // Project to image plane [differentiable]
    auto projection_results = detail::autograd::GaussianFullyFusedProjectionJagged::apply(
        g_sizes, means.jdata(), quats.jdata(), scales.jdata(), c_sizes, viewmats.jdata(),
        Ks.jdata(), image_width, image_height, eps2d, near_plane, far_plane, radius_clip);
    torch::Tensor radii   = projection_results[0];
    torch::Tensor means2d = projection_results[1];
    torch::Tensor depths  = projection_results[2];
    torch::Tensor conics  = projection_results[3];

    // Turn [N1 + N2 + N3 + ..., ...] into [C1*N1 + C2*N2 + ..., ...]
    torch::Tensor opacities_batched = opacities.jdata().index({ gaussian_ids }); // [nnz]
    if (antialias) {
        opacities_batched *= projection_results[4];
    }

    std::unordered_map<std::string, torch::Tensor> debug_info;
    if (return_debug_info) {
        debug_info["camera_ids"]   = camera_ids;
        debug_info["gaussian_ids"] = gaussian_ids;
        debug_info["radii"]        = radii;
        debug_info["means2d"]      = means2d;
        debug_info["depths"]       = depths;
        debug_info["conics"]       = conics;
        debug_info["opacities"]    = opacities_batched;
    }

    torch::Tensor colors;
    if (render_depth_only) {
        colors = depths.index({ gaussian_ids }).unsqueeze(-1); // [nnz, 1]
    } else {
        // Colors from SH coefficients [differentiable]
        const torch::Tensor sh_coeffs_batched =
            sh_coeffs.jdata()
                .permute({ 1, 0, 2 })
                .index({ Slice(), gaussian_ids, Slice() });                 // [K, nnz, 3]

        const torch::Tensor camtoworlds = torch::inverse(viewmats.jdata()); // [ccz, 4, 4]
        const torch::Tensor dirs        = means.jdata().index({ gaussian_ids, Slice() }) -
                                   camtoworlds.index({ camera_ids, Slice(None, 3), 3 });
        colors = evaluateSphericalHarmonics(dirs, sh_coeffs_batched, radii, sh_degree_to_use);

        if (render_depth_channel) {
            colors = torch::cat({ colors, depths.index({ gaussian_ids }).unsqueeze(-1) }, -1);
        }
    }

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int num_tiles_w = std::ceil(image_width / static_cast<float>(tile_size));
    const int num_tiles_h = std::ceil(image_height / static_cast<float>(tile_size));
    std::tuple<torch::Tensor, torch::Tensor> tile_intersections =
        FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
            return detail::ops::dispatchGaussianTileIntersection<DeviceTag>(
                means2d, radii, depths, camera_ids, ccz, tile_size, num_tiles_h, num_tiles_w);
        });
    torch::Tensor tile_offsets      = std::get<0>(tile_intersections);
    torch::Tensor tile_gaussian_ids = std::get<1>(tile_intersections);
    if (return_debug_info) {
        debug_info["tile_offsets"]      = tile_offsets;
        debug_info["tile_gaussian_ids"] = tile_gaussian_ids;
    }

    // Rasterize projected Gaussians to pixels [differentiable]
    auto outputs = detail::autograd::GaussianRasterizeToPixels::apply(
        means2d, conics, colors, opacities_batched.contiguous(), image_width, image_height, 0, 0,
        tile_size, tile_offsets, tile_gaussian_ids, false);
    torch::Tensor render_colors = outputs[0];
    torch::Tensor render_alphas = outputs[1];

    return { render_colors, render_alphas, debug_info };
}

} // namespace

std::vector<torch::Tensor>
projectGaussiansToImages(const torch::Tensor &means, const torch::Tensor &quats,
                         const torch::Tensor &scales, const torch::Tensor &viewmats,
                         const torch::Tensor &Ks, const uint32_t image_width,
                         const uint32_t image_height, const float near_plane, const float far_plane,
                         const float radius_clip, const float eps2d, const bool antialias,
                         const bool ortho) {
    return detail::autograd::GaussianFullyFusedProjection::apply(
        means, quats, scales, viewmats, Ks, image_width, image_height, eps2d, near_plane, far_plane,
        radius_clip, antialias, ortho);
}

// Gaussian render for a single torch Tensor
std::unordered_map<std::string, torch::Tensor>
precomputeGaussianRenderStateUnbatched(
    const torch::Tensor &means, const torch::Tensor &quats, const torch::Tensor &scales,
    const torch::Tensor &opacities, const torch::Tensor &sh_coeffs, const torch::Tensor &viewmats,
    const torch::Tensor &Ks, const uint32_t image_width, const uint32_t image_height,
    const float near_plane, const float far_plane, const int sh_degree_to_use, const int tile_size,
    const float radius_clip, const float eps2d, const bool antialias,
    const bool render_depth_channel, const bool ortho) {
    const bool                   render_depth_only = false;
    std::array<torch::Tensor, 7> renderState       = computeGaussianRenderStateUnbatched(
        means, quats, scales, opacities, sh_coeffs, viewmats, Ks, image_width, image_height,
        near_plane, far_plane, sh_degree_to_use, tile_size, radius_clip, eps2d, antialias,
        render_depth_channel, render_depth_only, ortho);

    std::unordered_map<std::string, torch::Tensor> info;
    info["means2d"]           = renderState[0];
    info["conics"]            = renderState[1];
    info["opacities"]         = renderState[2];
    info["radii"]             = renderState[3];
    info["colors"]            = renderState[4];
    info["tile_offsets"]      = renderState[5];
    info["tile_gaussian_ids"] = renderState[6];
    return info;
}

std::tuple<torch::Tensor, torch::Tensor>
renderPixelsFromPrecomputedGaussianRenderStateUnbatched(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors, torch::Tensor opacities,
    uint32_t image_width, uint32_t image_height, uint32_t image_origin_w, uint32_t image_origin_h,
    uint32_t tile_size, torch::Tensor tile_offsets, torch::Tensor tile_gaussian_ids) {
    // Rasterize projected Gaussians to pixels [differentiable]
    auto outputs = detail::autograd::GaussianRasterizeToPixels::apply(
        means2d, conics, colors, opacities, image_width, image_height, image_origin_w,
        image_origin_h, tile_size, tile_offsets, tile_gaussian_ids, false);
    torch::Tensor render_colors = outputs[0];
    torch::Tensor render_alphas = outputs[1];

    return { render_colors, render_alphas };
}

std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRender(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
               const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
               const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
               const JaggedTensor &opacities, // [N1 + N2 + ...]
               const JaggedTensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
               const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
               const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
               const uint32_t image_width, const uint32_t image_height, const float near_plane,
               const float far_plane, const int sh_degree_to_use, const int tile_size,
               const float radius_clip, const float eps2d, const bool antialias,
               const bool render_depth_channel, const bool return_debug_info,
               const torch::optional<JaggedTensor> pixels_to_render, const bool ortho) {
    return gaussianRenderInternal(
        means, quats, scales, opacities, sh_coeffs, viewmats, Ks, image_width, image_height,
        near_plane, far_plane, sh_degree_to_use, tile_size, radius_clip, eps2d, antialias,
        render_depth_channel, return_debug_info, false /* render_depth_only*/, ortho);
}

std::tuple<torch::Tensor, torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
gaussianRenderDepth(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
                    const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
                    const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
                    const JaggedTensor &opacities, // [N1 + N2 + ...]
                    const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
                    const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
                    const uint32_t image_width, const uint32_t image_height, const float near_plane,
                    const float far_plane, const int tile_size, const float radius_clip,
                    const float eps2d, const bool antialias, const bool return_debug_info,
                    const torch::optional<JaggedTensor> pixels_to_render, const bool ortho) {
    fvdb::JaggedTensor dummy_coeffs;
    return gaussianRenderInternal(
        means, quats, scales, opacities, dummy_coeffs, viewmats, Ks, image_width, image_height,
        near_plane, far_plane, -1 /* sh_degree_to_use */, tile_size, radius_clip, eps2d, antialias,
        false /* render_depth_channel */, return_debug_info, true, ortho);
}

} // namespace fvdb
