// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <detail/ops/Ops.h>
#include <detail/utils/AccessorHelpers.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename T>
__global__ void
__launch_bounds__(256) computeNanInfMaskKernel(fvdb::TorchRAcc64<T, 2>    means,          // [N, 3]
                                               fvdb::TorchRAcc64<T, 2>    quats,          // [N, 4]
                                               fvdb::TorchRAcc64<T, 2>    logScales,      // [N, 3]
                                               fvdb::TorchRAcc64<T, 1>    logitOpacities, // [N,]
                                               fvdb::TorchRAcc64<T, 3>    sh0,     // [1, N, D]
                                               fvdb::TorchRAcc64<T, 3>    shN,     // [K-1, N, D]
                                               fvdb::TorchRAcc64<bool, 1> outValid // [N,]
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

        for (auto i = 0; i < logScales.size(1); i += 1) {
            if (std::isnan(logScales[x][i]) || std::isinf(logScales[x][i])) {
                valid = false;
            }
        }

        if (std::isnan(logitOpacities[x]) || std::isinf(logitOpacities[x])) {
            valid = false;
        }

        for (auto i = 0; i < sh0.size(2); i += 1) {
            if (std::isnan(sh0[x][0][i]) || std::isinf(sh0[x][0][i])) {
                valid = false;
            }
        }

        for (auto i = 0; i < shN.size(1); i += 1) {
            for (auto j = 0; j < shN.size(2); j += 1) {
                if (std::isnan(shN[x][i][j]) || std::isinf(shN[x][i][j])) {
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
                                         const fvdb::JaggedTensor &logScales,
                                         const fvdb::JaggedTensor &logitOpacities,
                                         const fvdb::JaggedTensor &sh0,
                                         const fvdb::JaggedTensor &shN) {
    TORCH_CHECK_VALUE(means.rsize(0) == quats.rsize(0),
                      "All inputs must have the same number of gaussians");
    TORCH_CHECK_VALUE(means.rsize(0) == logScales.rsize(0),
                      "All inputs must have the same number of gaussians");
    TORCH_CHECK_VALUE(means.rsize(0) == logitOpacities.rsize(0),
                      "All inputs must have the same number of gaussians");
    TORCH_CHECK_VALUE(means.rsize(0) == sh0.rsize(0),
                      "All inputs must have the same number of gaussians");
    TORCH_CHECK_VALUE(means.rsize(0) == shN.rsize(0),
                      "All inputs must have the same number of gaussians");

    TORCH_CHECK_VALUE(means.rsize(1) == 3, "Means must have 3 components (shape [N, 3])");
    TORCH_CHECK_VALUE(quats.rsize(1) == 4, "Quaternions must have 4 components (shape [N, 4])");
    TORCH_CHECK_VALUE(logScales.rsize(1) == 3, "logScales must have 3 components (shape [N, 3])");
    TORCH_CHECK_VALUE(logitOpacities.rdim() == 1,
                      "logit_opacities must have 1 component (shape [N,])");
    TORCH_CHECK_VALUE(sh0.rdim() == 3, "sh0 coefficients must have shape [N, 1, D]");
    TORCH_CHECK_VALUE(shN.rdim() == 3, "shN coefficients must have shape [N, K-1, D]");

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

    AT_DISPATCH_V2(
        means.scalar_type(), "computeNanInfMaskKernel", AT_WRAP([&] {
            computeNanInfMaskKernel<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                means.jdata().packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                quats.jdata().packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                logScales.jdata().packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                logitOpacities.jdata().packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                sh0.jdata().packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                shN.jdata().packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                outValid.packed_accessor64<bool, 1, torch::RestrictPtrTraits>());
        }),
        AT_EXPAND(AT_FLOATING_TYPES));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return means.jagged_like(outValid);
}

template <>
fvdb::JaggedTensor
dispatchGaussianNanInfMask<torch::kCPU>(const fvdb::JaggedTensor &means,
                                        const fvdb::JaggedTensor &quats,
                                        const fvdb::JaggedTensor &logScales,
                                        const fvdb::JaggedTensor &logitOpacities,
                                        const fvdb::JaggedTensor &sh0,
                                        const fvdb::JaggedTensor &shN) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "dispatchGaussianNanInfMask not implemented on the CPU");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
