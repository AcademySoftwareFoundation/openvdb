// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <detail/utils/AccessorHelpers.cuh>
#include <detail/utils/cuda/Utils.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace fvdb {
namespace detail {
namespace ops {

template <typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
volumeRenderFwdCallback(
    const TensorAccessor<scalar_t, 1> sigmas, const TensorAccessor<scalar_t, 2> rgbs,
    const TensorAccessor<scalar_t, 1> deltas, const TensorAccessor<scalar_t, 1> ts,
    const TensorAccessor<JOffsetsType, 1> jOffsets, scalar_t tsmtThreshold, int32_t numChannels,
    int32_t rayIdx, TensorAccessor<int64_t, 1> outTotalSamples,
    TensorAccessor<scalar_t, 1> outOpacity, TensorAccessor<scalar_t, 1> outDepth,
    TensorAccessor<scalar_t, 2> outRGB, TensorAccessor<scalar_t, 1> outWs) {
    const JOffsetsType sampleStartIdx = jOffsets[rayIdx];
    const JOffsetsType numRaySamples  = jOffsets[rayIdx + 1] - sampleStartIdx;

    // front to back compositing
    JOffsetsType numSamples = 0;
    scalar_t     T          = static_cast<scalar_t>(1.0);

    while (numSamples < numRaySamples) {
        const JOffsetsType s = sampleStartIdx + numSamples;
        const scalar_t     a =
            static_cast<scalar_t>(1.0) - c10::cuda::compat::exp(-sigmas[s] * deltas[s]);
        const scalar_t w = a * T; // weight of the sample point

        // Forward pass works for arbitrary number of channels
        for (int c = 0; c < numChannels; ++c) {
            outRGB[rayIdx][c] += w * rgbs[s][c];
        }
        outDepth[rayIdx] += w * ts[s];
        // outDepthSq[rayIdx] += w*ts[s]*ts[s];
        outOpacity[rayIdx] += w;
        outWs[s] = w;
        T *= static_cast<scalar_t>(1.0) - a;

        // ray has enough opacity
        if (T <= tsmtThreshold) {
            break;
        }
        numSamples += 1;
    }
    outTotalSamples[rayIdx] = numSamples;
}

template <c10::DeviceType device, typename scalar_t,
          template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void
volumeRenderBwdCallback(const TensorAccessor<scalar_t, 1>     dLdOpacity,     // [B*R]
                        const TensorAccessor<scalar_t, 1>     dLdDepth,       // [B*R]
                        const TensorAccessor<scalar_t, 2>     dLdRgb,         // [B*R, C]
                        const TensorAccessor<scalar_t, 1>     dLdWs,          // [B*R*S]
                        const TensorAccessor<scalar_t, 1>     dLdWs_times_ws, // [B*R*S]
                        const TensorAccessor<scalar_t, 1>     sigmas,         // [B*R*S]
                        const TensorAccessor<scalar_t, 2>     rgbs,           // [B*R*S, C]
                        const TensorAccessor<scalar_t, 1>     deltas,         // [B*R*S]
                        const TensorAccessor<scalar_t, 1>     ts,             // [B*R*S]
                        const TensorAccessor<JOffsetsType, 1> jOffsets,       // [B*R, 2]
                        const TensorAccessor<scalar_t, 1>     opacity,        // [B*R]
                        const TensorAccessor<scalar_t, 1>     depth,          // [B*R]
                        const TensorAccessor<scalar_t, 2>     rgb,            // [B*R, C]
                        const scalar_t                        tsmtThreshold,  // scalar
                        const int32_t rayIdx, TensorAccessor<scalar_t, 1> out_dL_dsigmas, // [B*R*S]
                        TensorAccessor<scalar_t, 2> out_dLdRgbs) { // [B*R*S, C]

    const JOffsetsType sampleStartIdx = jOffsets[rayIdx];
    const JOffsetsType numRaySamples  = jOffsets[rayIdx + 1] - sampleStartIdx;

    // front to back compositing
    JOffsetsType numSamples = 0;
    scalar_t     R = rgb[rayIdx][0], G = rgb[rayIdx][1], B = rgb[rayIdx][2];
    scalar_t     O = opacity[rayIdx], D = depth[rayIdx]; //, Dsq = depthSq[rayIdx];
    scalar_t     T = static_cast<scalar_t>(1.0), r = static_cast<scalar_t>(0.0),
             g = static_cast<scalar_t>(0.0), b = static_cast<scalar_t>(0.0),
             d = static_cast<scalar_t>(0.0);             //, dsq = static_cast<scalar_t>(0.0);
    // compute prefix sum of dLdWs * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    if constexpr (device == torch::kCUDA) {
        thrust::inclusive_scan(thrust::device, dLdWs_times_ws.data() + sampleStartIdx,
                               dLdWs_times_ws.data() + sampleStartIdx + numRaySamples,
                               dLdWs_times_ws.data() + sampleStartIdx);
    } else {
        thrust::inclusive_scan(thrust::seq, dLdWs_times_ws.data() + sampleStartIdx,
                               dLdWs_times_ws.data() + sampleStartIdx + numRaySamples,
                               dLdWs_times_ws.data() + sampleStartIdx);
    }
    scalar_t dLdWs_times_ws_sum = dLdWs_times_ws[sampleStartIdx + numRaySamples - 1];

    while (numSamples < numRaySamples) {
        const JOffsetsType s = sampleStartIdx + numSamples;
        const scalar_t     a =
            static_cast<scalar_t>(1.0) - c10::cuda::compat::exp(-sigmas[s] * deltas[s]);
        const scalar_t w = a * T;

        r += w * rgbs[s][0];
        g += w * rgbs[s][1];
        b += w * rgbs[s][2];
        d += w * ts[s]; // dsq += w*ts[s]*ts[s];
        T *= static_cast<scalar_t>(1.0) - a;

        // compute gradients by math...
        out_dLdRgbs[s][0] = dLdRgb[rayIdx][0] * w;
        out_dLdRgbs[s][1] = dLdRgb[rayIdx][1] * w;
        out_dLdRgbs[s][2] = dLdRgb[rayIdx][2] * w;

        out_dL_dsigmas[s] =
            deltas[s] * (dLdRgb[rayIdx][0] * (rgbs[s][0] * T - (R - r)) +
                         dLdRgb[rayIdx][1] * (rgbs[s][1] * T - (G - g)) +
                         dLdRgb[rayIdx][2] * (rgbs[s][2] * T - (B - b)) + // gradients from rgb
                         dLdOpacity[rayIdx] * (1 - O) +                   // gradient from opacity
                         dLdDepth[rayIdx] * (ts[s] * T - (D - d)) +       // gradient from depth
                         // dLdDepthSq[rayIdx]*(ts[s]*ts[s]*T-(Dsq-dsq)) +
                         T * dLdWs[s] - (dLdWs_times_ws_sum - dLdWs_times_ws[s]) // gradient from ws
                        );

        if (T <= tsmtThreshold)
            break; // ray has enough opacity
        numSamples++;
    }
}

template <typename scalar_t>
void
volumeRenderCPU(const TorchAcc<scalar_t, 1>     sigmas,          // [B*R*S]
                const TorchAcc<scalar_t, 2>     rgbs,            // [B*R*S, C]
                const TorchAcc<scalar_t, 1>     deltas,          // [B*R*S]
                const TorchAcc<scalar_t, 1>     ts,              // [B*R*S]
                const TorchAcc<JOffsetsType, 1> jOffsets,        // [B*R, 2]
                const scalar_t                  tsmtThreshold,   // scalar
                TorchAcc<int64_t, 1>            outTotalSamples, // [B*R]
                TorchAcc<scalar_t, 1>           outOpacity,      // [B*R]
                TorchAcc<scalar_t, 1>           outDepth,        // [B*R]
                TorchAcc<scalar_t, 2>           outRGB,          // [B*R, C]
                TorchAcc<scalar_t, 1>           outWs) {                   // [B*R*S]

    const int numChannels = rgbs.size(1);

    for (int rayIdx = 0; rayIdx < (jOffsets.size(0) - 1); rayIdx += 1) {
        volumeRenderFwdCallback<scalar_t, TorchAcc>(
            sigmas, rgbs, deltas, ts, jOffsets, tsmtThreshold, numChannels, rayIdx, outTotalSamples,
            outOpacity, outDepth, outRGB, outWs);
    }
}

template <typename scalar_t>
void
volumeRenderBackwardCPU(const TorchAcc<scalar_t, 1>     dLdOpacity,     // [B*R]
                        const TorchAcc<scalar_t, 1>     dLdDepth,       // [B*R]
                        const TorchAcc<scalar_t, 2>     dLdRgb,         // [B*R, C]
                        const TorchAcc<scalar_t, 1>     dLdWs,          // [B*R*S]
                        const TorchAcc<scalar_t, 1>     dLdWs_times_ws, // [B*R*S]
                        const TorchAcc<scalar_t, 1>     sigmas,         // [B*R*S]
                        const TorchAcc<scalar_t, 2>     rgbs,           // [B*R*S, C]
                        const TorchAcc<scalar_t, 1>     deltas,         // [B*R*S]
                        const TorchAcc<scalar_t, 1>     ts,             // [B*R*S]
                        const TorchAcc<JOffsetsType, 1> jOffsets,       // [B*R, 2]
                        const TorchAcc<scalar_t, 1>     opacity,        // [B*R]
                        const TorchAcc<scalar_t, 1>     depth,          // [B*R]
                        const TorchAcc<scalar_t, 2>     rgb,            // [B*R, C]
                        const scalar_t                  tsmtThreshold,  // scalar
                        TorchAcc<scalar_t, 1>           out_dL_dsigmas, // [B*R*S]
                        TorchAcc<scalar_t, 2>           out_dLdRgbs) {            // [B*R*S, C]

    for (int rayIdx = 0; rayIdx < (jOffsets.size(0) - 1); rayIdx += 1) {
        volumeRenderBwdCallback<torch::kCPU, scalar_t, TorchAcc>(
            dLdOpacity, dLdDepth, dLdRgb, dLdWs, dLdWs_times_ws, sigmas, rgbs, deltas, ts, jOffsets,
            opacity, depth, rgb, tsmtThreshold, rayIdx, out_dL_dsigmas, out_dLdRgbs);
    }
}

template <typename scalar_t>
__global__ void
volumeRender(const TorchRAcc32<scalar_t, 1> sigmas, const TorchRAcc32<scalar_t, 2> rgbs,
             const TorchRAcc32<scalar_t, 1> deltas, const TorchRAcc32<scalar_t, 1> ts,
             const TorchRAcc32<JOffsetsType, 1> jOffsets, const scalar_t tsmtThreshold,
             TorchRAcc32<int64_t, 1> outTotalSamples, TorchRAcc32<scalar_t, 1> outOpacity,
             TorchRAcc32<scalar_t, 1> outDepth, TorchRAcc32<scalar_t, 2> outRGB,
             TorchRAcc32<scalar_t, 1> outWs) {
    const int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= outOpacity.size(0)) {
        return;
    }
    const int numChannels = rgbs.size(1);
    volumeRenderFwdCallback<scalar_t, TorchRAcc32>(
        sigmas, rgbs, deltas, ts, jOffsets, tsmtThreshold, numChannels, rayIdx, outTotalSamples,
        outOpacity, outDepth, outRGB, outWs);
}

template <typename scalar_t>
__global__ void
volumeRenderBackward(const TorchRAcc32<scalar_t, 1> dLdOpacity,
                     const TorchRAcc32<scalar_t, 1> dLdDepth, const TorchRAcc32<scalar_t, 2> dLdRgb,
                     const TorchRAcc32<scalar_t, 1> dLdWs,
                     const TorchRAcc32<scalar_t, 1> dLdWs_times_ws,
                     const TorchRAcc32<scalar_t, 1> sigmas, const TorchRAcc32<scalar_t, 2> rgbs,
                     const TorchRAcc32<scalar_t, 1> deltas, const TorchRAcc32<scalar_t, 1> ts,
                     const TorchRAcc32<JOffsetsType, 1> jOffsets,
                     const TorchRAcc32<scalar_t, 1> opacity, const TorchRAcc32<scalar_t, 1> depth,
                     const TorchRAcc32<scalar_t, 2> rgb, const scalar_t tsmtThreshold,
                     TorchRAcc32<scalar_t, 1> out_dL_dsigmas,
                     TorchRAcc32<scalar_t, 2> out_dLdRgbs) {
    const int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= opacity.size(0)) {
        return;
    }
    volumeRenderBwdCallback<torch::kCUDA, scalar_t, TorchRAcc32>(
        dLdOpacity, dLdDepth, dLdRgb, dLdWs, dLdWs_times_ws, sigmas, rgbs, deltas, ts, jOffsets,
        opacity, depth, rgb, tsmtThreshold, rayIdx, out_dL_dsigmas, out_dLdRgbs);
}

template <>
void
dispatchVolumeRender<torch::kCUDA>(
    const torch::Tensor sigmas,       // [B*R*S]
    const torch::Tensor rgbs,         // [B*R*S, C]
    const torch::Tensor deltas,       // [B*R*S]
    const torch::Tensor ts,           // [B*R*S]
    const torch::Tensor jOffsets,     // JaggedTensor joffsets for sigmas, rgbs, deltas, ts [B*R, 2]
    const float         tsmtThreshold,
    torch::Tensor      &outOpacity,   // [B*R]
    torch::Tensor      &outDepth,     // [B*R]
    torch::Tensor      &outRgb,       // [B*R, C]
    torch::Tensor      &outWs,        // [B*R*S]
    torch::Tensor      &outTotalSamples) { // [B*R]
    const int64_t numRays = jOffsets.size(0) - 1;
    const int64_t N       = sigmas.size(0);

    TORCH_CHECK(sigmas.device().is_cuda(), "sigmas must be a CUDA tensor");
    TORCH_CHECK(sigmas.device().has_index(), "sigmas must have CUDA index");
    TORCH_CHECK(sigmas.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == deltas.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == jOffsets.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outOpacity.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outDepth.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outRgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outWs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outTotalSamples.device(),
                "All tensors must be on the same device");

    c10::cuda::CUDAGuard deviceGuard(sigmas.device());

    // auto opacity = torch::zeros({numRays}, sigmas.options());
    // auto depth = torch::zeros({numRays}, sigmas.options());
    // auto depthSq = torch::zeros({numRays}, sigmas.options());
    // auto rgb = torch::zeros({numRays, 3}, sigmas.options());
    // auto ws = torch::zeros({N}, sigmas.options());
    // auto total_samples = torch::zeros({numRays},
    // torch::dtype(torch::kLong).device(sigmas.device()));

    const int64_t NUM_THREADS = 1024;
    const int64_t NUM_BLOCKS  = GET_BLOCKS(numRays, NUM_THREADS);

    AT_DISPATCH_V2(
        sigmas.scalar_type(), "volumeRender", AT_WRAP([&] {
            volumeRender<scalar_t><<<NUM_BLOCKS, NUM_THREADS>>>(
                sigmas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                rgbs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                deltas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                ts.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                jOffsets.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>(),
                tsmtThreshold,
                outTotalSamples.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
                outOpacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                outDepth.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                // outDepthSq.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                outRgb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                outWs.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }),
        AT_EXPAND(AT_FLOATING_TYPES), c10::kHalf);

    // return {total_samples, opacity, depth, depthSq, rgb, ws};
}

template <>
void
dispatchVolumeRenderBackward<torch::kCUDA>(const torch::Tensor dLdOpacity,
                                           const torch::Tensor dLdDepth,
                                           // const torch::Tensor dLdDepthSq,
                                           const torch::Tensor dLdRgb, const torch::Tensor dLdWs,
                                           const torch::Tensor sigmas, const torch::Tensor rgbs,
                                           const torch::Tensor ws, const torch::Tensor deltas,
                                           const torch::Tensor ts, const torch::Tensor jOffsets,
                                           const torch::Tensor opacity, const torch::Tensor depth,
                                           // const torch::Tensor depthSq,
                                           const torch::Tensor rgb, const float tsmtThreshold,
                                           torch::Tensor &outDLdSigmas, torch::Tensor &outDLdRbgs) {
    TORCH_CHECK(dLdOpacity.device().is_cuda(), "dLdOpacity must be a CUDA tensor");
    TORCH_CHECK(dLdOpacity.device().has_index(), "dLdOpacity must have CUDA index");
    TORCH_CHECK(dLdOpacity.device() == dLdDepth.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == dLdRgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == dLdWs.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == ws.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == deltas.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == jOffsets.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == opacity.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == depth.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == rgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == outDLdSigmas.device(),
                "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == outDLdRbgs.device(),
                "All tensors must be on the same device");

    c10::cuda::CUDAGuard deviceGuard(dLdOpacity.device());

    const int64_t N       = sigmas.size(0);
    const int64_t numRays = jOffsets.size(0) - 1;

    // auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    // auto dLdRgbs = torch::zeros({N, 3}, sigmas.options());

    torch::Tensor dLdWs_times_ws = (dLdWs * ws); // auxiliary input

    const int64_t NUM_THREADS = 1024;
    const int64_t NUM_BLOCKS  = GET_BLOCKS(numRays, NUM_THREADS);

    AT_DISPATCH_V2(
        sigmas.scalar_type(), "volumeRenderBackward", AT_WRAP([&] {
            volumeRenderBackward<scalar_t><<<NUM_BLOCKS, NUM_THREADS>>>(
                dLdOpacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                dLdDepth.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                // dLdDepthSq.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                dLdRgb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                dLdWs.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                dLdWs_times_ws.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                sigmas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                rgbs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                deltas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                ts.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                jOffsets.packed_accessor32<JOffsetsType, 1, torch::RestrictPtrTraits>(),
                opacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                depth.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                // depthSq.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                rgb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), tsmtThreshold,
                outDLdSigmas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                outDLdRbgs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }),
        AT_EXPAND(AT_FLOATING_TYPES), c10::kHalf);

    // return {dL_dsigmas, dLdRgbs};
}

template <>
void
dispatchVolumeRender<torch::kCPU>(const torch::Tensor sigmas, const torch::Tensor rgbs,
                                  const torch::Tensor deltas, const torch::Tensor ts,
                                  const torch::Tensor jOffsets, const float tsmtThreshold,
                                  torch::Tensor &outOpacity, torch::Tensor &outDepth,
                                  torch::Tensor &outRgb, torch::Tensor &outWs,
                                  torch::Tensor &outTotalSamples) {
    AT_DISPATCH_V2(sigmas.scalar_type(), "volumeRender", AT_WRAP([&] {
                       volumeRenderCPU<scalar_t>(
                           sigmas.accessor<scalar_t, 1>(), rgbs.accessor<scalar_t, 2>(),
                           deltas.accessor<scalar_t, 1>(), ts.accessor<scalar_t, 1>(),
                           jOffsets.accessor<JOffsetsType, 1>(), tsmtThreshold,
                           outTotalSamples.accessor<int64_t, 1>(),
                           outOpacity.accessor<scalar_t, 1>(), outDepth.accessor<scalar_t, 1>(),
                           outRgb.accessor<scalar_t, 2>(), outWs.accessor<scalar_t, 1>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), c10::kHalf);
}

template <>
void
dispatchVolumeRenderBackward<torch::kCPU>(const torch::Tensor dLdOpacity,
                                          const torch::Tensor dLdDepth, const torch::Tensor dLdRgb,
                                          const torch::Tensor dLdWs, const torch::Tensor sigmas,
                                          const torch::Tensor rgbs, const torch::Tensor ws,
                                          const torch::Tensor deltas, const torch::Tensor ts,
                                          const torch::Tensor jOffsets, const torch::Tensor opacity,
                                          const torch::Tensor depth, const torch::Tensor rgb,
                                          const float tsmtThreshold, torch::Tensor &outDLdSigmas,
                                          torch::Tensor &outDLdRbgs) {
    torch::Tensor dLdWs_times_ws = (dLdWs * ws); // auxiliary input

    AT_DISPATCH_V2(sigmas.scalar_type(), "volumeRenderBackward", AT_WRAP([&] {
                       volumeRenderBackwardCPU<scalar_t>(
                           dLdOpacity.accessor<scalar_t, 1>(), dLdDepth.accessor<scalar_t, 1>(),
                           dLdRgb.accessor<scalar_t, 2>(), dLdWs.accessor<scalar_t, 1>(),
                           dLdWs_times_ws.accessor<scalar_t, 1>(), sigmas.accessor<scalar_t, 1>(),
                           rgbs.accessor<scalar_t, 2>(), deltas.accessor<scalar_t, 1>(),
                           ts.accessor<scalar_t, 1>(), jOffsets.accessor<JOffsetsType, 1>(),
                           opacity.accessor<scalar_t, 1>(), depth.accessor<scalar_t, 1>(),
                           rgb.accessor<scalar_t, 2>(), tsmtThreshold,
                           outDLdSigmas.accessor<scalar_t, 1>(),
                           outDLdRbgs.accessor<scalar_t, 2>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), c10::kHalf);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
