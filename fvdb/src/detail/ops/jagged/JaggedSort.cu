#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/ops/jagged/JaggedOps.h"


namespace fvdb {
namespace detail {
namespace ops {

template<typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
inline __hostdev__ int64_t qsortPartition(TensorAccessor<scalar_t, 1> data,
                                          TensorAccessor<int64_t, 1> idx,
                                          int64_t l, int64_t h) {
    // Index of smaller element
    int64_t i = l - 1;
    scalar_t pivot = data[h];

    for (uint32_t j = l; j < h; ++j) {
        // If current element is smaller than or equal to pivot
        if (data[j] <= pivot) {
            // Increment index of smaller element
            i++;
            { scalar_t tmp = data[j]; data[j] = data[i]; data[i] = tmp; }
            { int64_t tmp = idx[j]; idx[j] = idx[i]; idx[i] = tmp; }
        }
    }
    { scalar_t tmp = data[i+1]; data[i+1] = data[h]; data[h] = tmp; }
    { int64_t tmp = idx[i+1]; idx[i+1] = idx[h]; idx[h] = tmp; }

    return i + 1;
}

template <typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void qsortCallback(int32_t tidx,
                               const TensorAccessor<int64_t, 2> offsets,
                               TensorAccessor<scalar_t, 1> data,
                               TensorAccessor<int64_t, 1> idx,
                               TensorAccessor<int64_t, 1> stack) {
    int64_t begin = offsets[tidx][0];
    int64_t num = offsets[tidx][1] - begin;

    int64_t l = 0, h = num - 1;
    int64_t top = -1;

    stack[begin + (++top)] = l;
    stack[begin + (++top)] = h;

    while (top >= 0) {
        h = stack[begin + (top--)];
        l = stack[begin + (top--)];

        int64_t p = qsortPartition<scalar_t, TensorAccessor>(
            data, idx, begin + l, begin + h) - begin;

        if (p-1 > l) {
            stack[begin + (++top)] = l;
            stack[begin + (++top)] = p-1;
        }

        if (p+1 < h) {
            stack[begin + (++top)] = p + 1;
            stack[begin + (++top)] = h;
        }
    }
}

template <c10::DeviceType DeviceTag>
torch::Tensor JaggedArgsort(const JaggedTensor& jt) {
    torch::Tensor data = jt.jdata();
    torch::Tensor offsets = jt.joffsets();

    if (data.ndimension() != 1) {
        TORCH_CHECK(data.ndimension() == 2 && data.size(1) == 1, "data must be 1D");
        data = data.squeeze(1);
    }

    // Algorithm will modify data in-place, so we need to clone it
    data = data.clone();

    auto longOption = torch::TensorOptions().dtype(torch::kInt64).device(data.device());

    torch::Tensor idx = torch::arange(data.size(0), longOption);
    torch::Tensor stack = torch::zeros({data.size(0)}, longOption);

    auto idxAccessor = tensorAccessor<DeviceTag, int64_t, 1>(idx);
    auto stackAccessor = tensorAccessor<DeviceTag, int64_t, 1>(stack);

    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, data.scalar_type(), "JaggedArgsort", [&]() {
        auto dataAcc = tensorAccessor<DeviceTag, scalar_t, 1>(data);
        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t ridx, int32_t cidx, TorchRAcc32<int64_t, 2> offsetAcc) {
                qsortCallback<scalar_t, TorchRAcc32>(ridx, offsetAcc, dataAcc, idxAccessor, stackAccessor);
            };
            forEachTensorElementChannelCUDA<int64_t, 2>(256, 1, offsets, cb);
        } else {
            auto cb = [=] (int32_t ridx, int32_t cidx, TorchAcc<int64_t, 2> offsetAcc) {
                qsortCallback<scalar_t, TorchAcc>(ridx, offsetAcc, dataAcc, idxAccessor, stackAccessor);
            };
            forEachTensorElementChannelCPU<int64_t, 2>(1, offsets, cb);
        }
    });

    return idx;
}

template <>
torch::Tensor dispatchJaggedArgsort<torch::kCPU>(const JaggedTensor& jt) {
    TORCH_CHECK(jt.is_cpu(), "jagged tensor must be on CPU");
    return JaggedArgsort<torch::kCPU>(jt);
}

template <>
torch::Tensor dispatchJaggedArgsort<torch::kCUDA>(const JaggedTensor& jt) {
    TORCH_CHECK(jt.is_cuda(), "jagged tensor must be on CUDA");
    return JaggedArgsort<torch::kCUDA>(jt);
}

} // namespace ops
} // namespace detail
} // namespace fvdb
