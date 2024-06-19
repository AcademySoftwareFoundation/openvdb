#include "Attention.h"

#include "detail/ops/Ops.h"
#include "detail/utils/Utils.h"

namespace fvdb {
namespace detail {
namespace autograd {

Attention::variable_list Attention::forward(Attention::AutogradContext *ctx,
                                            const Attention::Variable& query,
                                            const Attention::Variable& key,
                                            const Attention::Variable& value,
                                            const Attention::Variable& qLengths,
                                            const Attention::Variable& kvLengths,
                                            float scale) {
    torch::Tensor out = FVDB_DISPATCH_KERNEL_DEVICE(query.device(), [&]() {
        return ops::dispatchScaledDotProductAttention<DeviceTag>(
            query, key, value, qLengths, kvLengths, true, scale);
    });

    // ctx->saved_data["tsmtThreshold"] = tsmtThreshold;

    // ctx->save_for_backward({
    //     sigmas, rgbs, deltaTs, ts, raysAcc,
    //     outOpacity, outDepth, outRgb, outWs
    // });

    return { out};
}

Attention::variable_list Attention::backward(Attention::AutogradContext *ctx,
                                                   Attention::variable_list grad_output) {
    TORCH_CHECK(false, "Not implemented");
}


} // namespace autograd
} // namespace detail
} // namespace fvdb
