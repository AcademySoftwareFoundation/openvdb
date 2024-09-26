// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
#include "FVDB.h"

#include "detail/autograd/Autograd.h"
#include "detail/io/IO.h"
#include "detail/ops/jagged/JaggedOps.h"

#include <ATen/cuda/CUDAContext.h>

#include <execution>

namespace fvdb {

namespace {
// Gaussian render for a single torch Tensor
std::vector<torch::Tensor>
gaussianRender(const torch::Tensor &means, const torch::Tensor &quats, const torch::Tensor &scales,
               const torch::Tensor &opacities, const torch::Tensor &sh_coeffs,
               const torch::Tensor &viewmats, const torch::Tensor &Ks, const uint32_t image_width,
               const uint32_t image_height, const float eps2d, const float near_plane,
               const float far_plane, const float radius_clip, const int sh_degree_to_use,
               const int tile_size) {
    const int C = viewmats.size(0);  // number of cameras
    const int N = means.size(0);     // number of gaussians
    const int K = sh_coeffs.size(1); // number of SH bases

    TORCH_CHECK(means.sizes() == torch::IntArrayRef({ N, 3 }), "means must have shape (N, 3)");
    TORCH_CHECK(quats.sizes() == torch::IntArrayRef({ N, 4 }), "quats must have shape (N, 4)");
    TORCH_CHECK(scales.sizes() == torch::IntArrayRef({ N, 3 }), "scales must have shape (N, 3)");
    TORCH_CHECK(opacities.sizes() == torch::IntArrayRef({ N }), "opacities must have shape (N)");
    TORCH_CHECK(sh_coeffs.sizes() == torch::IntArrayRef({ N, K, 3 }),
                "sh_coeffs must have shape (N, K, 3)");
    TORCH_CHECK(K >= (sh_degree_to_use + 1) * (sh_degree_to_use + 1),
                "K must be at least (sh_degree_to_use + 1)^2");
    TORCH_CHECK(viewmats.sizes() == torch::IntArrayRef({ C, 4, 4 }),
                "viewmats must have shape (C, 4, 4)");
    TORCH_CHECK(Ks.sizes() == torch::IntArrayRef({ C, 3, 3 }), "Ks must have shape (C, 3, 3)");

    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(quats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(opacities.is_contiguous(), "opacities must be contiguous");
    TORCH_CHECK(sh_coeffs.is_contiguous(), "sh_coeffs must be contiguous");
    TORCH_CHECK(viewmats.is_contiguous(), "viewmats must be contiguous");
    TORCH_CHECK(Ks.is_contiguous(), "Ks must be contiguous");

    // Project to image plane [differentiable]
    auto projection_results = detail::autograd::GaussianFullyFusedProjection::apply(
        means, quats, scales, viewmats, Ks, image_width, image_height, eps2d, near_plane, far_plane,
        radius_clip);
    torch::Tensor radii   = projection_results[0];
    torch::Tensor means2d = projection_results[1];
    torch::Tensor depths  = projection_results[2];
    torch::Tensor conics  = projection_results[3];

    // Colors from SH coefficients [differentiable]
    torch::Tensor camtoworlds = torch::inverse(viewmats);
    torch::Tensor dirs =
        means.index({ torch::indexing::None, torch::indexing::Slice(), torch::indexing::Slice() }) -
        camtoworlds.index(
            { torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(0, 3), 3 });
    auto sh_results = detail::autograd::SphericalHarmonics::apply(
        sh_degree_to_use, dirs.contiguous(),
        sh_coeffs.unsqueeze(0).expand({ C, -1, -1, -1 }).contiguous());
    torch::Tensor colors = sh_results[0];
    colors               = torch::clamp_min(colors, colors + 0.5f); // [C, N, 3]

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int tile_width  = std::ceil(image_width / static_cast<float>(tile_size));
    const int tile_height = std::ceil(image_height / static_cast<float>(tile_size));

    auto          variables       = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return detail::ops::dispatchIsectTiles<DeviceTag>(
            means2d, radii, depths, at::nullopt /*camera_ids*/, at::nullopt /*gaussian_ids*/, C,
            tile_size, tile_width, tile_height, true, true);
    });
    torch::Tensor tiles_per_gauss = std::get<0>(variables).contiguous();
    torch::Tensor isect_ids       = std::get<1>(variables).contiguous();
    torch::Tensor flatten_ids     = std::get<2>(variables).contiguous();

    torch::Tensor isect_offsets = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return detail::ops::dispatchIsectOffsetEncode<DeviceTag>(isect_ids, C, tile_width,
                                                                 tile_height);
    });
    isect_offsets               = isect_offsets.contiguous();

    // Rasterize projected Gaussians to pixels [differentiable]
    auto outputs = detail::autograd::GaussianRasterizeToPixels::apply(
        means2d, conics, colors, opacities.repeat({ C, 1 }).contiguous(), image_width, image_height,
        tile_size, isect_offsets, flatten_ids, false);
    torch::Tensor render_colors = outputs[0];
    torch::Tensor render_alphas = outputs[1];

    return { render_colors, render_alphas, means2d };
}
} // namespace

std::vector<torch::Tensor>
volumeRender(const torch::Tensor &sigmas, const torch::Tensor &rgbs, const torch::Tensor &deltaTs,
             const torch::Tensor &ts, const torch::Tensor &jOffsets, double transmittanceThresh) {
    return detail::autograd::VolumeRender::apply(sigmas, rgbs, deltaTs, ts, jOffsets,
                                                 transmittanceThresh);
}

std::vector<torch::Tensor>
gaussianFullyFusedProjection(const torch::Tensor &means, const torch::Tensor &quats,
                             const torch::Tensor &scales, const torch::Tensor &viewmats,
                             const torch::Tensor &Ks, const uint32_t image_width,
                             const uint32_t image_height, const float eps2d, const float near_plane,
                             const float far_plane, const float radius_clip) {
    return detail::autograd::GaussianFullyFusedProjection::apply(
        means, quats, scales, viewmats, Ks, image_width, image_height, eps2d, near_plane, far_plane,
        radius_clip);
}

std::vector<torch::Tensor>
gaussianRender(const JaggedTensor &means,     // [N1 + N2 + ..., 3]
               const JaggedTensor &quats,     // [N1 + N2 + ..., 4]
               const JaggedTensor &scales,    // [N1 + N2 + ..., 3]
               const JaggedTensor &opacities, // [N1 + N2 + ...]
               const JaggedTensor &sh_coeffs, // [N1 + N2 + ..., K, 3]
               const JaggedTensor &viewmats,  // [C1 + C2 + ..., 4, 4]
               const JaggedTensor &Ks,        // [C1 + C2 + ..., 3, 3]
               const uint32_t image_width, const uint32_t image_height, const float eps2d,
               const float near_plane, const float far_plane, const float radius_clip,
               const int sh_degree_to_use, const int tile_size) {
    const int ccz = viewmats.rsize(0);   // number of cameras
    const int ggz = means.rsize(0);      // number of gaussians
    const int K   = sh_coeffs.rsize(-2); // number of SH bases

    using namespace torch::indexing;     // For the Slice operation

    TORCH_CHECK(means.rsizes() == torch::IntArrayRef({ ggz, 3 }), "means must have shape (ggz, 3)");
    TORCH_CHECK(quats.rsizes() == torch::IntArrayRef({ ggz, 4 }), "quats must have shape (ggz, 4)");
    TORCH_CHECK(scales.rsizes() == torch::IntArrayRef({ ggz, 3 }),
                "scales must have shape (ggz, 3)");
    TORCH_CHECK(opacities.rsizes() == torch::IntArrayRef({ ggz }),
                "opacities must have shape (ggz)");
    TORCH_CHECK(sh_coeffs.rsizes() == torch::IntArrayRef({ ggz, K, 3 }),
                "sh_coeffs must have shape (ggz, K, 3)");
    TORCH_CHECK(K >= (sh_degree_to_use + 1) * (sh_degree_to_use + 1),
                "K must be at least (sh_degree_to_use + 1)^2");
    TORCH_CHECK(viewmats.rsizes() == torch::IntArrayRef({ ccz, 4, 4 }),
                "viewmats must have shape (C, 4, 4)");
    TORCH_CHECK(Ks.rsizes() == torch::IntArrayRef({ ccz, 3, 3 }), "Ks must have shape (ccz, 3, 3)");

    TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
    TORCH_CHECK(quats.is_contiguous(), "quats must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(opacities.is_contiguous(), "opacities must be contiguous");
    TORCH_CHECK(sh_coeffs.is_contiguous(), "sh_coeffs must be contiguous");
    TORCH_CHECK(viewmats.is_contiguous(), "viewmats must be contiguous");
    TORCH_CHECK(Ks.is_contiguous(), "Ks must be contiguous");

    if (means.num_tensors() == 1) {
        return gaussianRender(means.jdata(), quats.jdata(), scales.jdata(), opacities.jdata(),
                              sh_coeffs.jdata(), viewmats.jdata(), Ks.jdata(), image_width,
                              image_height, eps2d, near_plane, far_plane, radius_clip,
                              sh_degree_to_use, tile_size);
    }

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
        torch::arange(viewmats.rsize(0), means.device()).repeat_interleave(tt, 0);
    // gaussian_ids is [0, 1, ..., 0, 1, ...]
    torch::Tensor dd0    = means.joffsets().index({ Slice(0, -1) }).repeat_interleave(c_sizes, 0);
    torch::Tensor dd1    = means.joffsets().index({ Slice(1, None) }).repeat_interleave(c_sizes, 0);
    torch::Tensor shifts = dd0.index({ Slice(1, None) }) - dd1.index({ Slice(0, -1) });
    shifts               = torch::cat({ torch::tensor({ 0 }, means.device()), shifts });
    torch::Tensor shifts_cumsum = shifts.cumsum(0);
    torch::Tensor gaussian_ids  = torch::arange(camera_ids.size(0), means.device());
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
    torch::Tensor sh_coeffs_batched =
        sh_coeffs.jdata().index({ gaussian_ids, Slice(), Slice() });             // [nnz, K, 3]

    // Colors from SH coefficients [differentiable]
    torch::Tensor camtoworlds = torch::inverse(viewmats.jdata()); // [ccz, 4, 4]
    torch::Tensor dirs        = means.jdata().index({ gaussian_ids, Slice() }) -
                         camtoworlds.index({ camera_ids, Slice(None, 3), 3 });
    auto sh_results = detail::autograd::SphericalHarmonics::apply(
        sh_degree_to_use, dirs.contiguous(), sh_coeffs_batched.contiguous());
    torch::Tensor colors = sh_results[0];
    colors               = torch::clamp_min(colors, colors + 0.5f); // [ggz, 3]

    // Intersect projected Gaussians with image tiles [non-differentiable]
    const int tile_width  = std::ceil(image_width / static_cast<float>(tile_size));
    const int tile_height = std::ceil(image_height / static_cast<float>(tile_size));

    auto variables = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return detail::ops::dispatchIsectTiles<DeviceTag>(means2d, radii, depths, camera_ids,
                                                          gaussian_ids, ccz, tile_size, tile_width,
                                                          tile_height, true, true);
    });

    torch::Tensor tiles_per_gauss = std::get<0>(variables).contiguous();
    torch::Tensor isect_ids       = std::get<1>(variables).contiguous();
    torch::Tensor flatten_ids     = std::get<2>(variables).contiguous();

    torch::Tensor isect_offsets = FVDB_DISPATCH_KERNEL_DEVICE(means.device(), [&]() {
        return detail::ops::dispatchIsectOffsetEncode<DeviceTag>(isect_ids, ccz, tile_width,
                                                                 tile_height);
    });
    isect_offsets               = isect_offsets.contiguous();

    // Rasterize projected Gaussians to pixels [differentiable]
    auto outputs = detail::autograd::GaussianRasterizeToPixels::apply(
        means2d, conics, colors, opacities_batched.contiguous(), image_width, image_height,
        tile_size, isect_offsets, flatten_ids, false);
    torch::Tensor render_colors = outputs[0];
    torch::Tensor render_alphas = outputs[1];

    return { render_colors, render_alphas, means2d, gaussian_ids, camera_ids };
}

JaggedTensor
scaledDotProductAttention(const JaggedTensor &query, const JaggedTensor &key,
                          const JaggedTensor &value, float scale) {
    cudaDeviceProp *p                 = at::cuda::getDeviceProperties(query.device().index());
    const int       computeCapability = p->major * 10 + p->minor;

    if (computeCapability < 90) {
        // https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        // - query: (N, ..., L, E)
        // - key: (N, ..., S, E)
        // - value: (N, ..., S, V)
        std::vector<torch::Tensor> outList;
        torch::Tensor              qOffsets  = query.joffsets().cpu();
        torch::Tensor              kvOffsets = key.joffsets().cpu();

        for (int64_t b = 0; b < query.num_tensors(); ++b) {
            int64_t qStart  = qOffsets[b].item<int64_t>();
            int64_t qEnd    = qOffsets[b + 1].item<int64_t>();
            int64_t kvStart = kvOffsets[b].item<int64_t>();
            int64_t kvEnd   = kvOffsets[b + 1].item<int64_t>();

            torch::Tensor q =
                query.jdata().index({ torch::indexing::Slice(qStart, qEnd) }).permute({ 1, 0, 2 });
            torch::Tensor k =
                key.jdata().index({ torch::indexing::Slice(kvStart, kvEnd) }).permute({ 1, 0, 2 });
            torch::Tensor v = value.jdata()
                                  .index({ torch::indexing::Slice(kvStart, kvEnd) })
                                  .permute({ 1, 0, 2 });

            torch::Tensor out =
                at::native::scaled_dot_product_attention(q, k, v, {}, 0.0, false, scale);
            outList.push_back(out.permute({ 1, 0, 2 }));
        }

        return JaggedTensor(outList);
    }

    // Custom implementation with CUDNN is only available for Hopper.
    torch::Tensor qLengths =
        query.joffsets().index({ torch::indexing::Slice(1, query.num_tensors()) });
    torch::Tensor kvLengths =
        key.joffsets().index({ torch::indexing::Slice(1, query.num_tensors()) });
    torch::Tensor res = detail::autograd::Attention::apply(
        query.jdata(), key.jdata(), value.jdata(), qLengths, kvLengths, scale)[0];
    return query.jagged_like(res);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
from_nanovdb(nanovdb::GridHandle<nanovdb::HostBuffer> &handle) {
    return detail::io::fromNVDB(handle);
}

nanovdb::GridHandle<nanovdb::HostBuffer>
to_nanovdb(const GridBatch &gridBatch, const torch::optional<JaggedTensor> maybeData,
           const torch::optional<StringOrListOfStrings> maybeNames) {
    return detail::io::toNVDB(gridBatch, maybeData, maybeNames);
}

GridBatch
jcat(const std::vector<GridBatch> &vec) {
    std::vector<c10::intrusive_ptr<detail::GridBatchImpl>> vecHdls;
    std::transform(vec.begin(), vec.end(), std::back_inserter(vecHdls),
                   [](const GridBatch &grid) { return grid.impl(); });
    return GridBatch(detail::GridBatchImpl::concatenate(vecHdls));
}

JaggedTensor
jcat(const std::vector<JaggedTensor> &vec, torch::optional<int64_t> dim) {
    return JaggedTensor::jcat(vec, dim);
}

void
save(const std::string &path, const GridBatch &gridBatch,
     const torch::optional<JaggedTensor>          maybeData,
     const torch::optional<StringOrListOfStrings> maybeNames, bool compressed, bool verbose) {
    detail::io::saveNVDB(path, gridBatch, maybeData, maybeNames, compressed, verbose);
}

std::tuple<GridBatch, JaggedTensor, std::vector<std::string>>
load(const std::string &path, NanoVDBFileGridIdentifier gridIdentifier, TorchDeviceOrString device,
     bool verbose) {
    return detail::io::loadNVDB(path, gridIdentifier, device, verbose);
}

GridBatch
gridbatch_from_points(const JaggedTensor &points, const Vec3i &pad_min, const Vec3i &pad_max,
                      const Vec3dBatchOrScalar &voxel_sizes, const Vec3dBatch &origins,
                      bool is_mutable) {
    auto ret = GridBatch(points.device(), is_mutable);
    ret.set_from_points(points, pad_min, pad_max, voxel_sizes, origins);
    return ret;
}

GridBatch
gridbatch_from_ijk(const JaggedTensor &ijk, const Vec3i &pad_min, const Vec3i &pad_max,
                   const Vec3dBatchOrScalar &voxel_sizes, const Vec3dBatch &origins,
                   bool is_mutable) {
    auto ret = GridBatch(ijk.device(), is_mutable);
    ret.set_from_ijk(ijk, pad_min, pad_max, voxel_sizes, origins);
    return ret;
}

GridBatch
gridbatch_from_nearest_voxels_to_points(const JaggedTensor       &points,
                                        const Vec3dBatchOrScalar &voxel_sizes,
                                        const Vec3dBatch &origins, bool is_mutable) {
    auto ret = GridBatch(points.device(), is_mutable);
    ret.set_from_nearest_voxels_to_points(points, voxel_sizes, origins);
    return ret;
}

GridBatch
gridbatch_from_dense(const int64_t numGrids, const Vec3i &denseDims, const Vec3i &ijkMin,
                     const Vec3dBatchOrScalar &voxel_sizes, const Vec3dBatch &origins,
                     torch::optional<torch::Tensor> mask, TorchDeviceOrString device,
                     bool is_mutable) {
    auto ret = GridBatch(device, is_mutable);
    ret.set_from_dense_grid(numGrids, denseDims, ijkMin, voxel_sizes, origins, mask);
    return ret;
}

GridBatch
gridbatch_from_mesh(const JaggedTensor &vertices, const JaggedTensor &faces,
                    const Vec3dBatchOrScalar &voxel_sizes, const Vec3dBatch &origins,
                    bool is_mutable) {
    auto ret = GridBatch(vertices.device(), is_mutable);
    ret.set_from_mesh(vertices, faces, voxel_sizes, origins);
    return ret;
}

std::vector<int64_t>
jdataShape1(const std::vector<int64_t> &lsizes, const std::vector<int64_t> &rsizes) {
    const int64_t totalElements = std::reduce(std::execution::par, lsizes.begin(), lsizes.end());
    std::vector<int64_t> shape;
    shape.reserve(rsizes.size() + 1);
    shape.push_back(totalElements);
    shape.insert(shape.end(), rsizes.begin(), rsizes.end());
    return shape;
}

std::tuple<int64_t, std::vector<int64_t>>
jdataShape2(const std::vector<std::vector<int64_t>> &lsizes, const std::vector<int64_t> &rsizes) {
    std::vector<int64_t> elementCountsPerList;
    std::vector<int64_t> tensorCountsPerList;
    elementCountsPerList.reserve(lsizes.size());
    tensorCountsPerList.reserve(lsizes.size());
    for (const auto &l: lsizes) {
        elementCountsPerList.push_back(std::reduce(std::execution::par, l.begin(), l.end()));
        tensorCountsPerList.push_back(l.size());
    }
    const int64_t totalSize =
        std::reduce(std::execution::par, elementCountsPerList.begin(), elementCountsPerList.end());
    const int64_t totalTensors =
        std::reduce(std::execution::par, tensorCountsPerList.begin(), tensorCountsPerList.end());
    std::vector<int64_t> shape;
    shape.reserve(rsizes.size() + 1);
    shape.push_back(totalSize);
    shape.insert(shape.end(), rsizes.begin(), rsizes.end());

    return std::make_tuple(totalTensors, shape);
}

#define __FVDB__BUILDER(FNAME, JFNAME)                                                         \
    JaggedTensor JFNAME(const std::vector<int64_t> &lsizes, const std::vector<int64_t> rsizes, \
                        at::TensorOptions options) {                                           \
        auto shape = jdataShape1(lsizes, rsizes);                                              \
        return JaggedTensor(lsizes, FNAME(shape, options));                                    \
    }                                                                                          \
                                                                                               \
    JaggedTensor JFNAME(const std::vector<std::vector<int64_t>> &lsizes,                       \
                        const std::vector<int64_t> rsizes, at::TensorOptions options) {        \
        auto shape = jdataShape2(lsizes, rsizes);                                              \
        return JaggedTensor(lsizes, std::get<0>(shape), FNAME(std::get<1>(shape), options));   \
    }

__FVDB__BUILDER(torch::rand, jrand)
__FVDB__BUILDER(torch::randn, jrandn)
__FVDB__BUILDER(torch::zeros, jzeros)
__FVDB__BUILDER(torch::ones, jones)
__FVDB__BUILDER(torch::empty, jempty)

#undef __FVDB__BUILDER

} // namespace fvdb