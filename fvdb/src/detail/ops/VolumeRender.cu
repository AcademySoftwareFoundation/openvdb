#include "detail/utils/cuda/Utils.cuh"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>


namespace fvdb {
namespace detail {
namespace ops {

template <typename scalar_t>
void volumeRenderCPU(const torch::TensorAccessor<scalar_t, 1> sigmas,   // [B*R*S]
                     const torch::TensorAccessor<scalar_t, 2> rgbs,     // [B*R*S, C]
                     const torch::TensorAccessor<scalar_t, 1> deltas,   // [B*R*S]
                     const torch::TensorAccessor<scalar_t, 1> ts,       // [B*R*S]
                     const torch::TensorAccessor<int32_t, 2> packInfo,  // [B*R, 2]
                     const scalar_t tsmtThreshold,                      // scalar
                     torch::TensorAccessor<int64_t, 1> outTotalSamples, // [B*R]
                     torch::TensorAccessor<scalar_t, 1> outOpacity,     // [B*R]
                     torch::TensorAccessor<scalar_t, 1> outDepth,       // [B*R]
                     torch::TensorAccessor<scalar_t, 2> outRGB,         // [B*R, C]
                     torch::TensorAccessor<scalar_t, 1> outWs) {        // [B*R*S]

    const int numChannels = rgbs.size(1);

    for (int rayIdx = 0; rayIdx < packInfo.size(0); rayIdx += 1) {
        const int sampleStartIdx = packInfo[rayIdx][0];
        const int numRaySamples = packInfo[rayIdx][1];

        // front to back compositing
        int numSamples = 0;
        scalar_t T = static_cast<scalar_t>(1.0);

        while (numSamples < numRaySamples) {
            const int s = sampleStartIdx + numSamples;
            const scalar_t a = static_cast<scalar_t>(1.0)- exp(-sigmas[s]*deltas[s]);
            const scalar_t w = a * T; // weight of the sample point

            // Forward pass works for arbitrary number of channels
            for (int c = 0; c < numChannels; ++c) {
                outRGB[rayIdx][c] += w*rgbs[s][c];
            }
            outDepth[rayIdx] += w*ts[s];
            outOpacity[rayIdx] += w;
            outWs[s] = w;
            T *= static_cast<scalar_t>(1.0)-a;

            // ray has enough opacity
            if (T <= tsmtThreshold) {
                break;
            }
            numSamples += 1;
        }
        outTotalSamples[rayIdx] = numSamples;
    }
}


template <typename scalar_t>
void volumeRenderBackwardCPU(const torch::TensorAccessor<scalar_t, 1> dLdOpacity,      // [B*R]
                             const torch::TensorAccessor<scalar_t, 1> dLdDepth,        // [B*R]
                             const torch::TensorAccessor<scalar_t, 2> dLdRgb,          // [B*R, C]
                             const torch::TensorAccessor<scalar_t, 1> dLdWs,           // [B*R*S]
                             const torch::TensorAccessor<scalar_t, 1> dLdWs_times_ws,  // [B*R*S]
                             const torch::TensorAccessor<scalar_t, 1> sigmas,          // [B*R*S]
                             const torch::TensorAccessor<scalar_t, 2> rgbs,            // [B*R*S, C]
                             const torch::TensorAccessor<scalar_t, 1> deltas,          // [B*R*S]
                             const torch::TensorAccessor<scalar_t, 1> ts,              // [B*R*S]
                             const torch::TensorAccessor<int32_t, 2> packInfo,         // [B*R, 2]
                             const torch::TensorAccessor<scalar_t, 1> opacity,         // [B*R]
                             const torch::TensorAccessor<scalar_t, 1> depth,           // [B*R]
                             const torch::TensorAccessor<scalar_t, 2> rgb,             // [B*R, C]
                             const scalar_t tsmtThreshold,                             // scalar
                             torch::TensorAccessor<scalar_t, 1> out_dL_dsigmas,        // [B*R*S]
                             torch::TensorAccessor<scalar_t, 2> out_dLdRgbs) {           // [B*R*S, C]

    for (int rayIdx = 0; rayIdx < packInfo.size(0); rayIdx += 1) {
        const int sampleStartIdx = packInfo[rayIdx][0];
        const int numRaySamples = packInfo[rayIdx][1];
        const int numChannels = rgbs.size(1);

        // front to back compositing
        int numSamples = 0;
        scalar_t R = rgb[rayIdx][0], G = rgb[rayIdx][1], B = rgb[rayIdx][2];
        scalar_t O = opacity[rayIdx], D = depth[rayIdx]; //, Dsq = depthSq[rayIdx];
        scalar_t T = static_cast<scalar_t>(1.0), r = 0.0f, g = 0.0f, b = 0.0f, d = 0.0f;//, dsq = 0.0f;
        // compute prefix sum of dLdWs * ws
        // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
        thrust::inclusive_scan(dLdWs_times_ws.data()+sampleStartIdx,
                                dLdWs_times_ws.data()+sampleStartIdx+numRaySamples,
                                dLdWs_times_ws.data()+sampleStartIdx);
        scalar_t dLdWs_times_ws_sum = dLdWs_times_ws[sampleStartIdx+numRaySamples-1];

        while (numSamples < numRaySamples) {
            const int s = sampleStartIdx + numSamples;
            const scalar_t a = static_cast<scalar_t>(1.0) - exp(-sigmas[s]*deltas[s]);
            const scalar_t w = a * T;

            r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
            d += w*ts[s]; //dsq += w*ts[s]*ts[s];
            T *= static_cast<scalar_t>(1.0)-a;

            for (int i = 0; i < numChannels; i++) {
                out_dLdRgbs[s][i] = dLdRgb[rayIdx][i]*w;
            }

            // FIXME: Arbitrary number of channels
            out_dL_dsigmas[s] = deltas[s] * (
                dLdRgb[rayIdx][0]*(rgbs[s][0]*T-(R-r)) +
                dLdRgb[rayIdx][1]*(rgbs[s][1]*T-(G-g)) +
                dLdRgb[rayIdx][2]*(rgbs[s][2]*T-(B-b)) + // gradients from rgb
                dLdOpacity[rayIdx]*(1-O) + // gradient from opacity
                dLdDepth[rayIdx]*(ts[s]*T-(D-d)) + // gradient from depth
                // dLdDepthSq[rayIdx]*(ts[s]*ts[s]*T-(Dsq-dsq)) +
                T*dLdWs[s]-(dLdWs_times_ws_sum-dLdWs_times_ws[s]) // gradient from ws
            );

            if (T <= tsmtThreshold) break; // ray has enough opacity
            numSamples++;
        }
    }
}


template <typename scalar_t>
__global__ void volumeRender(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigmas,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rgbs,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> deltas,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ts,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> packInfo,
    const scalar_t tsmtThreshold,
    torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> outTotalSamples,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> outOpacity,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> outDepth,
    // torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> outDepthSq,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> outRGB,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> outWs)
{
    // const int n = blockIdx.x * blockDim.x + threadIdx.x;
    // if (n >= outOpacity.size(0)) {
    //     return;
    // }
    // const int rayIdx = packInfo[n][0], sampleStartIdx = packInfo[n][1], numRaySamples = packInfo[n][2];

    const int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= outOpacity.size(0)) {
        return;
    }
    const int sampleStartIdx = packInfo[rayIdx][0];
    const int numRaySamples = packInfo[rayIdx][1];

    // front to back compositing
    int numSamples = 0;
    scalar_t T = 1.0f;

    while (numSamples < numRaySamples) {
        const int s = sampleStartIdx + numSamples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T; // weight of the sample point

        outRGB[rayIdx][0] += w*rgbs[s][0];
        outRGB[rayIdx][1] += w*rgbs[s][1];
        outRGB[rayIdx][2] += w*rgbs[s][2];
        outDepth[rayIdx] += w*ts[s];
        // outDepthSq[rayIdx] += w*ts[s]*ts[s];
        outOpacity[rayIdx] += w;
        outWs[s] = w;
        T *= 1.0f-a;

        // ray has enough opacity
        if (T <= tsmtThreshold) {
            break;
        }
        numSamples += 1;
    }
    outTotalSamples[rayIdx] = numSamples;
}


template <typename scalar_t>
__global__ void volumeRenderBackward(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dLdOpacity,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dLdDepth,
    // const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dLdDepthSq,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dLdRgb,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dLdWs,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dLdWs_times_ws,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> sigmas,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rgbs,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> deltas,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> ts,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> packInfo,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> opacity,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> depth,
    // const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> depthSq,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rgb,
    const scalar_t tsmtThreshold,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> out_dL_dsigmas,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out_dLdRgbs
){
    // const int n = blockIdx.x * blockDim.x + threadIdx.x;
    // if (n >= opacity.size(0)) {
    //     return;
    // }
    // const int rayIdx = packInfo[n][0], sampleStartIdx = packInfo[n][1], numRaySamples = packInfo[n][2];

    const int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rayIdx >= opacity.size(0)) {
        return;
    }
    const int sampleStartIdx = packInfo[rayIdx][0];
    const int numRaySamples = packInfo[rayIdx][1];

    // front to back compositing
    int numSamples = 0;
    scalar_t R = rgb[rayIdx][0], G = rgb[rayIdx][1], B = rgb[rayIdx][2];
    scalar_t O = opacity[rayIdx], D = depth[rayIdx]; //, Dsq = depthSq[rayIdx];
    scalar_t T = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f, d = 0.0f;//, dsq = 0.0f;
    // compute prefix sum of dLdWs * ws
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           dLdWs_times_ws.data()+sampleStartIdx,
                           dLdWs_times_ws.data()+sampleStartIdx+numRaySamples,
                           dLdWs_times_ws.data()+sampleStartIdx);
    scalar_t dLdWs_times_ws_sum = dLdWs_times_ws[sampleStartIdx+numRaySamples-1];

    while (numSamples < numRaySamples) {
        const int s = sampleStartIdx + numSamples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
        d += w*ts[s]; //dsq += w*ts[s]*ts[s];
        T *= 1.0f-a;

        // compute gradients by math...
        out_dLdRgbs[s][0] = dLdRgb[rayIdx][0]*w;
        out_dLdRgbs[s][1] = dLdRgb[rayIdx][1]*w;
        out_dLdRgbs[s][2] = dLdRgb[rayIdx][2]*w;

        out_dL_dsigmas[s] = deltas[s] * (
            dLdRgb[rayIdx][0]*(rgbs[s][0]*T-(R-r)) +
            dLdRgb[rayIdx][1]*(rgbs[s][1]*T-(G-g)) +
            dLdRgb[rayIdx][2]*(rgbs[s][2]*T-(B-b)) + // gradients from rgb
            dLdOpacity[rayIdx]*(1-O) + // gradient from opacity
            dLdDepth[rayIdx]*(ts[s]*T-(D-d)) + // gradient from depth
            // dLdDepthSq[rayIdx]*(ts[s]*ts[s]*T-(Dsq-dsq)) +
            T*dLdWs[s]-(dLdWs_times_ws_sum-dLdWs_times_ws[s]) // gradient from ws
        );

        if (T <= tsmtThreshold) break; // ray has enough opacity
        numSamples++;
    }
}





template <>
void dispatchVolumeRender<torch::kCUDA>(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor packInfo,
    const float tsmtThreshold,
    torch::Tensor& outOpacity,
    torch::Tensor& outDepth,
    // torch::Tensor& outDepthSq,
    torch::Tensor& outRgb,
    torch::Tensor& outWs,
    torch::Tensor& outTotalSamples) {
    const int numRays = packInfo.size(0), N = sigmas.size(0);

    TORCH_CHECK(sigmas.device().is_cuda(), "sigmas must be a CUDA tensor");
    TORCH_CHECK(sigmas.device().has_index(), "sigmas must have CUDA index");
    TORCH_CHECK(sigmas.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == deltas.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == packInfo.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outOpacity.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outDepth.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outRgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outWs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == outTotalSamples.device(), "All tensors must be on the same device");

    c10::cuda::CUDAGuard deviceGuard(sigmas.device());

    // auto opacity = torch::zeros({numRays}, sigmas.options());
    // auto depth = torch::zeros({numRays}, sigmas.options());
    // auto depthSq = torch::zeros({numRays}, sigmas.options());
    // auto rgb = torch::zeros({numRays, 3}, sigmas.options());
    // auto ws = torch::zeros({N}, sigmas.options());
    // auto total_samples = torch::zeros({numRays}, torch::dtype(torch::kLong).device(sigmas.device()));

    const int threads = 1024, blocks = (numRays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "volumeRender",
    ([&] {
        volumeRender<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            rgbs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            deltas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            ts.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            packInfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            tsmtThreshold,
            outTotalSamples.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            outOpacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            outDepth.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            // outDepthSq.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            outRgb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            outWs.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    // return {total_samples, opacity, depth, depthSq, rgb, ws};
}

template <>
void dispatchVolumeRenderBackward<torch::kCUDA>(
    const torch::Tensor dLdOpacity,
    const torch::Tensor dLdDepth,
    // const torch::Tensor dLdDepthSq,
    const torch::Tensor dLdRgb,
    const torch::Tensor dLdWs,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor packInfo,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    // const torch::Tensor depthSq,
    const torch::Tensor rgb,
    const float tsmtThreshold,
    torch::Tensor& outDLdSigmas,
    torch::Tensor& outDLdRbgs
) {
    TORCH_CHECK(dLdOpacity.device().is_cuda(), "dLdOpacity must be a CUDA tensor");
    TORCH_CHECK(dLdOpacity.device().has_index(), "dLdOpacity must have CUDA index");
    TORCH_CHECK(dLdOpacity.device() == dLdDepth.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == dLdRgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == dLdWs.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == ws.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == deltas.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == packInfo.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == opacity.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == depth.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == rgb.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == outDLdSigmas.device(), "All tensors must be on the same device");
    TORCH_CHECK(dLdOpacity.device() == outDLdRbgs.device(), "All tensors must be on the same device");

    c10::cuda::CUDAGuard deviceGuard(dLdOpacity.device());

    const int N = sigmas.size(0), numRays = packInfo.size(0);

    // auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    // auto dLdRgbs = torch::zeros({N, 3}, sigmas.options());

    torch::Tensor dLdWs_times_ws = (dLdWs * ws); // auxiliary input

    const int threads = 1024, blocks = (numRays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "volumeRenderBackward",
    ([&] {
        volumeRenderBackward<scalar_t><<<blocks, threads>>>(
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
            packInfo.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            depth.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            // depthSq.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            tsmtThreshold,
            outDLdSigmas.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            outDLdRbgs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }));

    // return {dL_dsigmas, dLdRgbs};
}


template <>
void dispatchVolumeRender<torch::kCPU>(const torch::Tensor sigmas,
                                     const torch::Tensor rgbs,
                                     const torch::Tensor deltas,
                                     const torch::Tensor ts,
                                     const torch::Tensor packInfo,
                                     const float tsmtThreshold,
                                     torch::Tensor& outOpacity,
                                     torch::Tensor& outDepth,
                                     torch::Tensor& outRgb,
                                     torch::Tensor& outWs,
                                     torch::Tensor& outTotalSamples) {

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "volumeRender",
    ([&] {
        volumeRenderCPU<scalar_t>(
            sigmas.accessor<scalar_t, 1>(),
            rgbs.accessor<scalar_t, 2>(),
            deltas.accessor<scalar_t, 1>(),
            ts.accessor<scalar_t, 1>(),
            packInfo.accessor<int32_t, 2>(),
            tsmtThreshold,
            outTotalSamples.accessor<int64_t, 1>(),
            outOpacity.accessor<scalar_t, 1>(),
            outDepth.accessor<scalar_t, 1>(),
            outRgb.accessor<scalar_t, 2>(),
            outWs.accessor<scalar_t, 1>()
        );
    }));
}

template <>
void dispatchVolumeRenderBackward<torch::kCPU>(const torch::Tensor dLdOpacity,
                                             const torch::Tensor dLdDepth,
                                             const torch::Tensor dLdRgb,
                                             const torch::Tensor dLdWs,
                                             const torch::Tensor sigmas,
                                             const torch::Tensor rgbs,
                                             const torch::Tensor ws,
                                             const torch::Tensor deltas,
                                             const torch::Tensor ts,
                                             const torch::Tensor packInfo,
                                             const torch::Tensor opacity,
                                             const torch::Tensor depth,
                                             const torch::Tensor rgb,
                                             const float tsmtThreshold,
                                             torch::Tensor& outDLdSigmas,
                                             torch::Tensor& outDLdRbgs) {
    torch::Tensor dLdWs_times_ws = (dLdWs * ws); // auxiliary input

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.scalar_type(), "volumeRenderBackward",
    ([&] {
        volumeRenderBackwardCPU<scalar_t>(
            dLdOpacity.accessor<scalar_t, 1>(),
            dLdDepth.accessor<scalar_t, 1>(),
            dLdRgb.accessor<scalar_t, 2>(),
            dLdWs.accessor<scalar_t, 1>(),
            dLdWs_times_ws.accessor<scalar_t, 1>(),
            sigmas.accessor<scalar_t, 1>(),
            rgbs.accessor<scalar_t, 2>(),
            deltas.accessor<scalar_t, 1>(),
            ts.accessor<scalar_t, 1>(),
            packInfo.accessor<int32_t, 2>(),
            opacity.accessor<scalar_t, 1>(),
            depth.accessor<scalar_t, 1>(),
            rgb.accessor<scalar_t, 2>(),
            tsmtThreshold,
            outDLdSigmas.accessor<scalar_t, 1>(),
            outDLdRbgs.accessor<scalar_t, 2>()
        );
    }));
}


}  // namespace ops
}  // namespace detail
}  // namespace fvdb
