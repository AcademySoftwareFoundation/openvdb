#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAException.h>
#include <ATen/cudnn/Handle.h>

#define JSON_HAS_RANGES 0
#include <cudnn_frontend.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/BezierInterpolationIterator.h"


namespace fvdb {
namespace detail {
namespace ops {

static torch::Tensor exclusivePrefixSum(torch::Tensor input) {
    return torch::cat({torch::zeros(1, input.options()), input.cumsum(0)});
}

template <>
torch::Tensor dispatchScaledDotProductAttention<torch::kCUDA>(const torch::Tensor& query,
                                                              const torch::Tensor& key,
                                                              const torch::Tensor& value,
                                                              const torch::Tensor& qLengths,
                                                              const torch::Tensor& kvLengths,
                                                              bool training,
                                                              float scale) {

    // TODO: Cache built execution graph and plans!
    // Get dimensions: query (B*Sq, H, D), key (B*Skv, H, D), value (B*Skv, H, T)
    // https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md

    int64_t num_batch = qLengths.size(0);
    int64_t num_heads = query.size(1);
    int64_t num_qk_feat = query.size(2);
    int64_t num_v_feat = value.size(2);
    int64_t num_sq = qLengths.max().item<int64_t>();
    int64_t num_skv = kvLengths.max().item<int64_t>();

    TORCH_CHECK(query.is_contiguous(), "query tensor must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key tensor must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value tensor must be contiguous");
    TORCH_CHECK_EQ(num_qk_feat, num_v_feat) << "Different query and key dimensions seem not to be supported";

    // Init cuDNN execution graph
    namespace fe = cudnn_frontend;
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_compute_data_type(fe::DataType_t::FLOAT);
    graph->set_intermediate_data_type(fe::DataType_t::FLOAT);
    if (query.scalar_type() == torch::kBFloat16) {
        graph->set_io_data_type(fe::DataType_t::BFLOAT16);
    } else {
        graph->set_io_data_type(fe::DataType_t::FLOAT);
    }

    // Create input tensor nodes (cudnn_frontend::graph::Tensor_attributes)
    // (although storage is BSHD, MHA needs dimension to be BHSD)
    auto qNode = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
        .set_dim({num_batch, num_heads, num_sq, num_qk_feat})
        .set_stride({num_sq * num_heads * num_qk_feat, num_qk_feat, num_heads * num_qk_feat, 1}));
    auto kNode = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
        .set_dim({num_batch, num_heads, num_skv, num_qk_feat})
        .set_stride({num_skv * num_heads * num_qk_feat, num_qk_feat, num_heads * num_qk_feat, 1}));
    auto vNode = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
        .set_dim({num_batch, num_heads, num_skv, num_v_feat})
        .set_stride({num_skv * num_heads * num_v_feat, num_v_feat, num_heads * num_v_feat, 1}));
    auto qLenNode = graph->tensor(fe::graph::Tensor_attributes().set_name("seq_len_q")
        .set_dim({num_batch, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));
    auto kvLenNode = graph->tensor(fe::graph::Tensor_attributes().set_name("seq_len_kv")
        .set_dim({num_batch, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));
    auto qOffsetNode = graph->tensor(fe::graph::Tensor_attributes().set_name("offset_q")
        .set_dim({num_batch + 1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));
    auto kOffsetNode = graph->tensor(fe::graph::Tensor_attributes().set_name("offset_k")
        .set_dim({num_batch + 1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));
    auto vOffsetNode = graph->tensor(fe::graph::Tensor_attributes().set_name("offset_v")
        .set_dim({num_batch + 1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));
    qNode->set_ragged_offset(qOffsetNode);
    kNode->set_ragged_offset(kOffsetNode);
    vNode->set_ragged_offset(vOffsetNode);

    // Build and link SPDA options
    auto sdpa_options = fe::graph::SDPA_attributes().set_name("SDPA").set_is_inference(!training);
    sdpa_options.set_attn_scale(scale);
    sdpa_options.set_padding_mask(true);
    sdpa_options.set_seq_len_q(qLenNode);
    sdpa_options.set_seq_len_kv(kvLenNode);

    // Create SPDA and output node
    auto [O, stats] = graph->sdpa(qNode, kNode, vNode, sdpa_options);
    O->set_output(true).set_dim({num_batch, num_heads, num_sq, num_v_feat})
        .set_stride({num_sq * num_heads * num_v_feat, num_v_feat, num_heads * num_v_feat, 1});
    auto oOffsetNode = graph->tensor(fe::graph::Tensor_attributes().set_name("offset_o")
        .set_dim({num_batch + 1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::INT32));
    O->set_ragged_offset(oOffsetNode);

    if (training) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    // Validate graph
    auto validate_status = graph->validate();
    TORCH_CHECK(validate_status.is_good(), std::string("Graph validation failed: ") + validate_status.get_message());

    // Build execution engine (pytorch <= 2.2.1 use cudnn 8930 that does not support ragged offset)
    // cudnnHandle_t handle = at::native::getCudnnHandle();
    // We use our own linked cudnn 9.0
    cudnnHandle_t handle;
    auto create_status = cudnnCreate(&handle);
    TORCH_CHECK(create_status == CUDNN_STATUS_SUCCESS, std::string("CUDNN handle creation failed: ") + std::to_string(create_status));

    std::cout << "CUDNN Version: " << cudnnGetVersion() << std::endl;

    auto build_status = graph->build_operation_graph(handle);
    TORCH_CHECK(build_status.is_good(), std::string("Graph build failed: ") + build_status.get_message());

    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    TORCH_CHECK(graph->check_support(handle).is_good(), "Graph support check failed");
    TORCH_CHECK(graph->build_plans(handle).is_good(), "Graph plan build failed");

    // Build output tensor
    torch::Tensor output = torch::empty(
        {num_batch, num_heads, num_sq, num_v_feat},
        torch::TensorOptions().dtype(query.dtype()).device(query.device()));
    torch::Tensor statsTensor;
    if (training) {
        statsTensor = torch::empty(
            {num_batch, num_heads, num_sq, 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
    }

    // Build variant pack
    torch::Tensor seqLenQ = qLengths.to(torch::kInt32);
    torch::Tensor seqLenKV = kvLengths.to(torch::kInt32);
    torch::Tensor offsetQ = exclusivePrefixSum(seqLenQ * num_heads * num_qk_feat).to(torch::kInt32);
    torch::Tensor offsetK = exclusivePrefixSum(seqLenKV * num_heads * num_qk_feat).to(torch::kInt32);
    torch::Tensor offsetV = exclusivePrefixSum(seqLenKV * num_heads * num_v_feat).to(torch::kInt32);
    torch::Tensor offsetO = exclusivePrefixSum(seqLenQ * num_heads * num_v_feat).to(torch::kInt32);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {qNode, (char*) query.data_ptr()},
        {kNode, (char*) key.data_ptr()},
        {vNode, (char*) value.data_ptr()},
        {qLenNode, (char*) seqLenQ.data_ptr()},
        {kvLenNode, (char*) seqLenKV.data_ptr()},
        {qOffsetNode, (char*) offsetQ.data_ptr()},
        {kOffsetNode, (char*) offsetK.data_ptr()},
        {vOffsetNode, (char*) offsetV.data_ptr()},
        {oOffsetNode, (char*) offsetO.data_ptr()},
        {O, (char*) output.data_ptr()}
    };
    if (training) {
        variant_pack[stats] = (char*) statsTensor.data_ptr();
    }

    // Build workspace
    auto workspace_size = graph->get_workspace_size();
    auto workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
    TORCH_CHECK(graph->execute(handle, variant_pack, workspace_ptr.get()).is_good());

    // Destory handle
    cudnnDestroy(handle);

    return output;
}


template <>
torch::Tensor dispatchScaledDotProductAttention<torch::kCPU>(const torch::Tensor& query,
                                                              const torch::Tensor& key,
                                                              const torch::Tensor& value,
                                                              const torch::Tensor& qLengths,
                                                              const torch::Tensor& kvLengths,
                                                              bool training,
                                                              float scale) {
    TORCH_CHECK(false, "CPU implementation not available");
}


} // namespace ops
} // namespace detail
} // namespace fvdb

