#include <c10/cuda/CUDAException.h>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/utils/cuda/Atomics.cuh"
#include "detail/ops/jagged/JaggedOps.h"


namespace fvdb {
namespace detail {
namespace ops {

enum ReductionType { SUM, MAX, MIN };

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
    // 1. For each reduced slot, init.
    // 2. Atomic write values and use separate kernels to compute arg.

    static inline scalar_t init() {
        if constexpr (REDUCE == ReductionType::SUM) {
            return 0;
        } else if constexpr (REDUCE == ReductionType::MIN) {
            return std::numeric_limits<scalar_t>::max();
        } else if constexpr (REDUCE == ReductionType::MAX) {
            return std::numeric_limits<scalar_t>::lowest();
        } else {
            return 0;
        }
    }

    static inline __device__ void atomicWriteCUDA(scalar_t* value, scalar_t new_value) {
        if constexpr (REDUCE == ReductionType::SUM) {
            atomAdd(value, new_value);
        } else if constexpr (REDUCE == ReductionType::MIN) {
            atomMin(value, new_value);
        } else if constexpr (REDUCE == ReductionType::MAX) {
            atomMax(value, new_value);
        }
    }

    static inline void atomicWriteCPU(scalar_t* value, scalar_t new_value) {
        if constexpr (REDUCE == ReductionType::SUM) {
            *value += new_value;
        } else if constexpr (REDUCE == ReductionType::MIN) {
            *value = std::min(*value, new_value);
        } else if constexpr (REDUCE == ReductionType::MAX) {
            *value = std::max(*value, new_value);
        }
    }
};

template <typename scalar_t, template <typename T, int32_t D> typename TensorAccessor, c10::DeviceType DeviceTag, ReductionType REDUCE>
__hostdev__ void jaggedReduceCallback(int32_t tidx, int32_t eidx,
                                      const TensorAccessor<scalar_t, 2> data,
                                      const TensorAccessor<int16_t, 1> idx,
                                      TensorAccessor<scalar_t, 2> out) {
    using RCD = Reducer<scalar_t, REDUCE>;

    if constexpr (DeviceTag == torch::kCUDA) {
        RCD::atomicWriteCUDA(&out[idx[tidx]][eidx], data[tidx][eidx]);
    } else {
        RCD::atomicWriteCPU(&out[idx[tidx]][eidx], data[tidx][eidx]);
    }
}

template <typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
__hostdev__ void jaggedArgReduceCallback(int32_t tidx, int32_t eidx,
                                         const TensorAccessor<scalar_t, 2> data,
                                         const TensorAccessor<int16_t, 1> idx,
                                         const TensorAccessor<scalar_t, 2> out,
                                         TensorAccessor<int64_t, 2> argOut) {
    if (data[tidx][eidx] == out[idx[tidx]][eidx]) {
        argOut[idx[tidx]][eidx] = tidx;
    }
}

template <c10::DeviceType DeviceTag, ReductionType REDUCE>
std::tuple<torch::Tensor, torch::optional<torch::Tensor>> JaggedReduce(
        const torch::Tensor& jdataRaw, const torch::Tensor& jidx, int64_t dimSize) {

    torch::Tensor jdata = featureCoalescedView(jdataRaw);

    static constexpr bool isMinMax = (REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX);

    torch::Tensor out = torch::empty({dimSize, jdata.size(1)}, jdata.options());
    torch::optional<torch::Tensor> argOut = torch::nullopt;
    if constexpr (isMinMax) {
        argOut = torch::empty({dimSize, jdata.size(1)}, jidx.options().dtype(torch::kLong));
    }

    auto jidxAccessor = tensorAccessor<DeviceTag, int16_t, 1>(jidx);

    AT_DISPATCH_FLOATING_TYPES(jdata.scalar_type(), "JaggedReduce", [&]() {
        out.fill_(Reducer<scalar_t, REDUCE>::init());
        auto outAccessor = tensorAccessor<DeviceTag, scalar_t, 2>(out);

        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t ridx, int32_t cidx, TorchRAcc32<scalar_t, 2> dataAcc) {
                jaggedReduceCallback<scalar_t, TorchRAcc32, DeviceTag, REDUCE>(
                    ridx, cidx, dataAcc, jidxAccessor, outAccessor);
            };
            forEachTensorElementChannelCUDA<scalar_t, 2>(256, jdata.size(1), jdata, cb);
        } else {
            auto cb = [=] (int32_t ridx, int32_t cidx, TorchAcc<scalar_t, 2> dataAcc) {
                jaggedReduceCallback<scalar_t, TorchAcc, DeviceTag, REDUCE>(
                    ridx, cidx, dataAcc, jidxAccessor, outAccessor);
            };
            forEachTensorElementChannelCPU<scalar_t, 2>(jdata.size(1), jdata, cb);
        }

        // Fill empty slots with 0 instead of Inf/-Inf, and compute arguments.
        if constexpr (isMinMax) {
            out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t) 0);
            argOut->fill_(-1);

            auto argOutAccessor = tensorAccessor<DeviceTag, int64_t, 2>(argOut.value());
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t ridx, int32_t cidx, TorchRAcc32<scalar_t, 2> dataAcc) {
                    jaggedArgReduceCallback<scalar_t, TorchRAcc32>(
                        ridx, cidx, dataAcc, jidxAccessor, outAccessor, argOutAccessor);
                };
                forEachTensorElementChannelCUDA<scalar_t, 2>(256, jdata.size(1), jdata, cb);
            } else {
                auto cb = [=] (int32_t ridx, int32_t cidx, TorchAcc<scalar_t, 2> dataAcc) {
                    jaggedArgReduceCallback<scalar_t, TorchAcc>(
                        ridx, cidx, dataAcc, jidxAccessor, outAccessor, argOutAccessor);
                };
                forEachTensorElementChannelCPU<scalar_t, 2>(jdata.size(1), jdata, cb);
            }

        }
    });

    torch::Tensor rOut = out.reshape(spliceShape({out.size(0)}, jdataRaw));
    torch::optional<torch::Tensor> rArgOut = torch::nullopt;
    if constexpr (isMinMax) {
        rArgOut = argOut.value().reshape(spliceShape({out.size(0)}, jdataRaw));
    }

    return std::make_tuple(rOut, rArgOut);
}

template <>
torch::Tensor dispatchJaggedSum<torch::kCPU>(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize) {
    TORCH_CHECK(jdata.is_cpu(), "jagged tensor must be on CPU");
    return std::get<0>(JaggedReduce<torch::kCPU, ReductionType::SUM>(jdata, jidx, dimSize));
}

template <>
torch::Tensor dispatchJaggedSum<torch::kCUDA>(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize) {
    TORCH_CHECK(jdata.is_cuda(), "jagged tensor must be on CUDA");
    return std::get<0>(JaggedReduce<torch::kCUDA, ReductionType::SUM>(jdata, jidx, dimSize));
}

template <>
std::vector<torch::Tensor> dispatchJaggedMin<torch::kCPU>(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize) {
    TORCH_CHECK(jdata.is_cpu(), "jagged tensor must be on CPU");
    auto res = JaggedReduce<torch::kCPU, ReductionType::MIN>(jdata, jidx, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

template <>
std::vector<torch::Tensor> dispatchJaggedMin<torch::kCUDA>(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize) {
    TORCH_CHECK(jdata.is_cuda(), "jagged tensor must be on CUDA");
    auto res = JaggedReduce<torch::kCUDA, ReductionType::MIN>(jdata, jidx, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

template <>
std::vector<torch::Tensor> dispatchJaggedMax<torch::kCPU>(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize) {
    TORCH_CHECK(jdata.is_cpu(), "jagged tensor must be on CPU");
    auto res = JaggedReduce<torch::kCPU, ReductionType::MAX>(jdata, jidx, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

template <>
std::vector<torch::Tensor> dispatchJaggedMax<torch::kCUDA>(const torch::Tensor& jdata, const torch::Tensor& jidx, int64_t dimSize) {
    TORCH_CHECK(jdata.is_cuda(), "jagged tensor must be on CUDA");
    auto res = JaggedReduce<torch::kCUDA, ReductionType::MAX>(jdata, jidx, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
