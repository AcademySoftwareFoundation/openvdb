// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GsplatUtils.cuh"
#include <detail/ops/Ops.h>

#include <ATen/cuda/Atomic.cuh>

constexpr int NUM_THREADS = 256;

namespace fvdb {
namespace detail {
namespace ops {

template <typename T, bool Ortho>
__global__ void
jaggedProjectionForwardKernel(const uint32_t B, const int64_t nnz,
                              const int64_t *__restrict__ g_sizes,  // [B]
                              const int64_t *__restrict__ c_sizes,  // [B]
                              const int64_t *__restrict__ g_indptr, // [B] start indices
                              const int64_t *__restrict__ c_indptr, // [B] start indices
                              const int64_t *__restrict__ n_indptr, // [B] start indices
                              const T *__restrict__ means,          // [ggz, 3]
                              const T *__restrict__ covars,         // [ggz, 6] optional
                              const T *__restrict__ quats,          // [ggz, 4] optional
                              const T *__restrict__ scales,         // [ggz, 3] optional
                              const T *__restrict__ viewmats,       // [ccz, 4, 4]
                              const T *__restrict__ Ks,             // [ccz, 3, 3]
                              const int32_t image_width, const int32_t image_height, const T eps2d,
                              const T near_plane, const T far_plane, const T radius_clip,
                              // outputs
                              int32_t *__restrict__ radii,     // [nnz]
                              T *__restrict__ means2d,         // [nnz, 2]
                              T *__restrict__ depths,          // [nnz]
                              T *__restrict__ conics,          // [nnz, 3]
                              T *__restrict__ compensations) { // [nnz] optional
    // parallelize over nnz.
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) {
        return;
    }

    // TODO: too many global memory accesses.
    int64_t bid       = bin_search(n_indptr, B, static_cast<int64_t>(idx)); // batch id
    int64_t idx_local = idx - n_indptr[bid];       // local elem idx within Ci * Ni
    int64_t cid_local = idx_local / g_sizes[bid];  // local camera id within Ci
    int64_t gid_local = idx_local % g_sizes[bid];  // local gaussian id within Ni
    int64_t cid       = cid_local + c_indptr[bid]; // camera id
    int64_t gid       = gid_local + g_indptr[bid]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3<T> R = mat3<T>(viewmats[0], viewmats[4],
                        viewmats[8], // 1st column
                        viewmats[1], viewmats[5],
                        viewmats[9], // 2nd column
                        viewmats[2], viewmats[6],
                        viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3<T> covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = mat3<T>(covars[0], covars[1],
                        covars[2], // 1st column
                        covars[1], covars[3],
                        covars[4], // 2nd column
                        covars[2], covars[4],
                        covars[5]  // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci<T>(glm::make_vec4(quats), glm::make_vec3(scales), &covar,
                                     nullptr);
    }
    mat3<T> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // camera projection
    const T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    auto [covar2d, mean2d] = [&]() {
        if constexpr (Ortho) {
            return ortho_proj<T>(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height);
        } else {
            return persp_proj<T>(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height);
        }
    }();

    T compensation;
    T det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2<T> covar2d_inv;
    inverse(covar2d, covar2d_inv);

    // take 3 sigma as the radius (non differentiable)
    T b      = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    T v1     = b + sqrt(max(0.01f, b * b - det));
    T radius = ceil(3.f * sqrt(v1));
    // T v2 = b - sqrt(max(0.1f, b * b - det));
    // T radius = ceil(3.f * sqrt(max(v1, v2)));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width || mean2d.y + radius <= 0 ||
        mean2d.y - radius >= image_height) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx]           = (int32_t)radius;
    means2d[idx * 2]     = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx]          = mean_c.z;
    conics[idx * 3]      = covar2d_inv[0][0];
    conics[idx * 3 + 1]  = covar2d_inv[0][1];
    conics[idx * 3 + 2]  = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedForward<torch::kCUDA>(
    const torch::Tensor &g_sizes,  // [B] gaussian sizes
    const torch::Tensor &means,    // [ggz, 3]
    const torch::Tensor &quats,    // [ggz, 4] optional
    const torch::Tensor &scales,   // [ggz, 3] optional
    const torch::Tensor &c_sizes,  // [B] camera sizes
    const torch::Tensor &viewmats, // [ccz, 4, 4]
    const torch::Tensor &Ks,       // [ccz, 3, 3]
    const uint32_t image_width, const uint32_t image_height, const float eps2d,
    const float near_plane, const float far_plane, const float radius_clip, const bool ortho) {
    // These are supported by the underlying kernel, but they are not exposed
    const at::optional<torch::Tensor> &covars             = std::nullopt;
    constexpr bool                     calc_compensations = false;

    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(g_sizes);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        GSPLAT_CHECK_INPUT(quats);
        GSPLAT_CHECK_INPUT(scales);
    }
    GSPLAT_CHECK_INPUT(c_sizes);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);

    // TODO: use inclusive sum
    uint32_t      B        = g_sizes.size(0);
    torch::Tensor c_indptr = torch::cumsum(c_sizes, 0, torch::kInt64) - c_sizes;
    torch::Tensor g_indptr = torch::cumsum(g_sizes, 0, torch::kInt64) - g_sizes;
    torch::Tensor n_sizes  = c_sizes * g_sizes;            // element size = Ci * Ni
    torch::Tensor n_indptr = torch::cumsum(n_sizes, 0, torch::kInt64);
    int64_t       nnz      = n_indptr[-1].item<int64_t>(); // total number of elements
    n_indptr               = n_indptr - n_sizes;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor radii   = torch::empty({ nnz }, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({ nnz, 2 }, means.options());
    torch::Tensor depths  = torch::empty({ nnz }, means.options());
    torch::Tensor conics  = torch::empty({ nnz, 3 }, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({ nnz }, means.options());
    }
    if (nnz) {
        if (ortho) {
            jaggedProjectionForwardKernel<float, true>
                <<<(nnz + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    B, nnz, g_sizes.data_ptr<int64_t>(), c_sizes.data_ptr<int64_t>(),
                    g_indptr.data_ptr<int64_t>(), c_indptr.data_ptr<int64_t>(),
                    n_indptr.data_ptr<int64_t>(), means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
                    eps2d, near_plane, far_plane, radius_clip, radii.data_ptr<int32_t>(),
                    means2d.data_ptr<float>(), depths.data_ptr<float>(), conics.data_ptr<float>(),
                    calc_compensations ? compensations.data_ptr<float>() : nullptr);
        } else {
            jaggedProjectionForwardKernel<float, false>
                <<<(nnz + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    B, nnz, g_sizes.data_ptr<int64_t>(), c_sizes.data_ptr<int64_t>(),
                    g_indptr.data_ptr<int64_t>(), c_indptr.data_ptr<int64_t>(),
                    n_indptr.data_ptr<int64_t>(), means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
                    eps2d, near_plane, far_plane, radius_clip, radii.data_ptr<int32_t>(),
                    means2d.data_ptr<float>(), depths.data_ptr<float>(), conics.data_ptr<float>(),
                    calc_compensations ? compensations.data_ptr<float>() : nullptr);
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedForward<torch::kCPU>(
    const torch::Tensor &g_sizes,  // [B] gaussian sizes
    const torch::Tensor &means,    // [ggz, 3]
    const torch::Tensor &quats,    // [ggz, 4] optional
    const torch::Tensor &scales,   // [ggz, 3] optional
    const torch::Tensor &c_sizes,  // [B] camera sizes
    const torch::Tensor &viewmats, // [ccz, 4, 4]
    const torch::Tensor &Ks,       // [ccz, 3, 3]
    const uint32_t image_width, const uint32_t image_height, const float eps2d,
    const float near_plane, const float far_plane, const float radius_clip, const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
