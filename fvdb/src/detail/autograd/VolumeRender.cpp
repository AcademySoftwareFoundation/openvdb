#include "VolumeRender.h"

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"

namespace fvdb {
namespace detail {
namespace autograd {

VolumeRender::variable_list VolumeRender::forward(VolumeRender::AutogradContext *ctx,
                                                  const VolumeRender::Variable& sigmas,
                                                  const VolumeRender::Variable& rgbs,
                                                  const VolumeRender::Variable& deltaTs,
                                                  const VolumeRender::Variable& ts,
                                                  const VolumeRender::Variable& raysAcc,
                                                  double tsmtThreshold) {
    const int numRays = raysAcc.size(0), N = sigmas.size(0);

    TORCH_CHECK(raysAcc.dim() == 2, "raysAcc must have shape (nRays, 2)");
    TORCH_CHECK(sigmas.dim() == 1, "sigmas must have shape (nRays*nSamplesPerRay,)");
    TORCH_CHECK(rgbs.dim() == 2, "rgbs must have shape (nRays*nSamplesPerRay, 3)");
    TORCH_CHECK(deltaTs.dim() == 1, "deltaTs must have shape (nRays*nSamplesPerRay,)");
    TORCH_CHECK(ts.dim() == 1, "ts must have shape (N,)");

    TORCH_CHECK(sigmas.device() == rgbs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == deltaTs.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == ts.device(), "All tensors must be on the same device");
    TORCH_CHECK(sigmas.device() == raysAcc.device(), "All tensors must be on the same device");

    TORCH_CHECK(sigmas.dtype() == rgbs.dtype(), "All floating point tensors must be on the same dtype");
    TORCH_CHECK(sigmas.dtype() == deltaTs.dtype(), "All floating point tensors must be on the same dtype");
    TORCH_CHECK(sigmas.dtype() == ts.dtype(),"All floating point tensors must be on the same dtype");
    TORCH_CHECK(raysAcc.dtype() == torch::dtype(torch::kInt32).dtype(), "raysAcc must be of type torch.int32");

    TORCH_CHECK(sigmas.size(0) == rgbs.size(0), "sigmas and rgbs must have the same number of elements");
    TORCH_CHECK(sigmas.size(0) == deltaTs.size(0), "sigmas and deltaTs must have the same number of elements");
    TORCH_CHECK(sigmas.size(0) == ts.size(0), "sigmas and ts must have the same number of elements");
    torch::Tensor outOpacity = torch::zeros({numRays}, sigmas.options());
    torch::Tensor outDepth = torch::zeros({numRays}, sigmas.options());
    // torch::Tensor outDepthSq = torch::zeros({numRays}, sigmas.options());
    torch::Tensor outRgb = torch::zeros({numRays, 3}, sigmas.options());
    torch::Tensor outWs = torch::zeros({N}, sigmas.options());
    torch::Tensor outTotalSamples = torch::zeros({numRays}, torch::dtype(torch::kLong).device(sigmas.device()));

    FVDB_DISPATCH_KERNEL_DEVICE(sigmas.device(), [&]() {
        ops::dispatchVolumeRender<DeviceTag>(
            sigmas, rgbs, deltaTs, ts, raysAcc, tsmtThreshold,
            outOpacity, outDepth, outRgb, outWs, outTotalSamples);
    });

    ctx->saved_data["tsmtThreshold"] = tsmtThreshold;

    ctx->save_for_backward({
        sigmas, rgbs, deltaTs, ts, raysAcc,
        outOpacity, outDepth, outRgb, outWs
    });

    return { outRgb, outDepth, outOpacity, outWs, outTotalSamples };
}

VolumeRender::variable_list VolumeRender::backward(VolumeRender::AutogradContext *ctx,
                                                   VolumeRender::variable_list grad_output) {
    Variable dLdRgb = grad_output.at(0);
    Variable dLdDepth = grad_output.at(1);
    Variable dLdOpacity = grad_output.at(2);
    Variable dLdWs = grad_output.at(3);
    // Variable dLdDepthSq = grad_output.at(3);

    variable_list saved = ctx->get_saved_variables();
    Variable sigmas = saved.at(0);
    Variable rgbs = saved.at(1);
    Variable deltaTs = saved.at(2);
    Variable ts = saved.at(3);
    Variable raysAcc = saved.at(4);

    Variable outOpacity = saved.at(5);
    Variable outDepth = saved.at(6);
    // Variable outDepthSq = ctx->saved_data["outDepthSq"].toTensor();
    Variable outRgb = saved.at(7);
    Variable outWs = saved.at(8);
    const double tsmtThreshold = ctx->saved_data["tsmtThreshold"].toDouble();

    const int N = sigmas.size(0);

    Variable dLdSigmas = torch::zeros({N}, sigmas.options());
    Variable dLdRgbs = torch::zeros({N, 3}, sigmas.options());

    FVDB_DISPATCH_KERNEL_DEVICE(sigmas.device(), [&]() {
        ops::dispatchVolumeRenderBackward<DeviceTag>(
            dLdOpacity, dLdDepth, /*dLdDepthSq,*/ dLdRgb, dLdWs,
            sigmas, rgbs, outWs, deltaTs, ts, raysAcc,
            outOpacity, outDepth, /*outDepthSq,*/ outRgb, tsmtThreshold,
            dLdSigmas, dLdRgbs);
    });

    return { dLdSigmas, dLdRgbs, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor() };
}


} // namespace autograd
} // namespace detail
} // namespace fvdb
