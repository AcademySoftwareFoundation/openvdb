// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>
#include <detail/utils/cuda/Utils.cuh>

namespace fvdb {
namespace detail {
namespace ops {

template <typename T>
__global__ void
__launch_bounds__(256) computeNanInfMaskKernel(fvdb::TorchRAcc64<T, 2>    means,     // [N, 3]
                                               fvdb::TorchRAcc64<T, 2>    quats,     // [N, 4]
                                               fvdb::TorchRAcc64<T, 2>    scales,    // [N, 3]
                                               fvdb::TorchRAcc64<T, 1>    opacities, // [N,]
                                               fvdb::TorchRAcc64<T, 3>    sh,        // [N, K, 3]
                                               fvdb::TorchRAcc64<bool, 1> outValid   // [N,]
) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < means.size(0);
         x += blockDim.x * gridDim.x) {
        bool valid = true;
        for (auto i = 0; i < means.size(1); i += 1) {
            if (std::isnan(means[x][i]) || std::isinf(means[x][i])) {
                valid = false;
            }
        }

        for (auto i = 0; i < quats.size(1); i += 1) {
            if (std::isnan(quats[x][i]) || std::isinf(quats[x][i])) {
                valid = false;
            }
        }

        for (auto i = 0; i < scales.size(1); i += 1) {
            if (std::isnan(scales[x][i]) || std::isinf(scales[x][i])) {
                valid = false;
            }
        }

        if (std::isnan(opacities[x]) || std::isinf(opacities[x])) {
            valid = false;
        }

        for (auto i = 0; i < sh.size(1); i += 1) {
            for (auto j = 0; j < sh.size(2); j += 1) {
                if (std::isnan(sh[x][i][j]) || std::isinf(sh[x][i][j])) {
                    valid = false;
                }
            }
        }

        outValid[x] = valid;
    }
}

template <>
fvdb::JaggedTensor
dispatchGaussianNanInfMask<torch::kCUDA>(const fvdb::JaggedTensor &means,
                                         const fvdb::JaggedTensor &quats,
                                         const fvdb::JaggedTensor &scales,
                                         const fvdb::JaggedTensor &opacities,
                                         const fvdb::JaggedTensor &sh_coeffs) {
    TORCH_CHECK_VALUE(means.rsize(0) == quats.rsize(0),
                      "All inputs must have the same number of gaussians");
    TORCH_CHECK_VALUE(means.rsize(0) == scales.rsize(0),
                      "All inputs must have the same number of gaussians");
    TORCH_CHECK_VALUE(means.rsize(0) == opacities.rsize(0),
                      "All inputs must have the same number of gaussians");
    TORCH_CHECK_VALUE(means.rsize(0) == sh_coeffs.rsize(0),
                      "All inputs must have the same number of gaussians");

    TORCH_CHECK_VALUE(means.rsize(1) == 3, "Means must have 3 components");
    TORCH_CHECK_VALUE(quats.rsize(1) == 4, "Quaternions must have 4 components");
    TORCH_CHECK_VALUE(scales.rsize(1) == 3, "Scales must have 3 components");
    TORCH_CHECK_VALUE(opacities.rdim() == 1, "Opacities must have 1 component (rshape [N,])");
    TORCH_CHECK_VALUE(sh_coeffs.rdim() == 3, "SH coefficients must have rshape [N, K, D]");

    if (means.rsize(0) == 0) {
        return means.jagged_like(
            torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kBool).device(means.device())));
    }
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means.jdata()));
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    const auto N = means.rsize(0);

    auto outValid =
        torch::empty({ N }, torch::TensorOptions().dtype(torch::kBool).device(means.device()));

    const size_t NUM_THREADS = 256;
    const size_t NUM_BLOCKS  = (N + NUM_THREADS - 1) / NUM_THREADS;

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(), "computeNanInfMaskKernel", ([&] {
            computeNanInfMaskKernel<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                means.jdata().packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                quats.jdata().packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                scales.jdata().packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                opacities.jdata().packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                sh_coeffs.jdata().packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                outValid.packed_accessor64<bool, 1, torch::RestrictPtrTraits>());
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return means.jagged_like(outValid);
}

template <>
fvdb::JaggedTensor
dispatchGaussianNanInfMask<torch::kCPU>(const fvdb::JaggedTensor &means,
                                        const fvdb::JaggedTensor &quats,
                                        const fvdb::JaggedTensor &scales,
                                        const fvdb::JaggedTensor &opacities,
                                        const fvdb::JaggedTensor &sh_coeffs) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "dispatchGaussianNanInfMask not implemented on the CPU");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
