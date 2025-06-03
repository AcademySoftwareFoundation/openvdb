// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/jagged/JaggedOps.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/cuda/Atomics.cuh>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>

#include <c10/cuda/CUDAException.h>

namespace fvdb {
namespace detail {
namespace ops {

enum ReductionType { SUM, MAX, MIN };

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
    // 1. For each reduced slot, init.
    // 2. Atomic write values and use separate kernels to compute arg.

    static inline scalar_t
    init() {
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

    static inline __device__ void
    atomicWriteCUDA(scalar_t *value, scalar_t new_value) {
        if constexpr (REDUCE == ReductionType::SUM) {
            atomAdd(value, new_value);
        } else if constexpr (REDUCE == ReductionType::MIN) {
            atomMin(value, new_value);
        } else if constexpr (REDUCE == ReductionType::MAX) {
            atomMax(value, new_value);
        }
    }

    static inline void
    atomicWriteCPU(scalar_t *value, scalar_t new_value) {
        if constexpr (REDUCE == ReductionType::SUM) {
            *value += new_value;
        } else if constexpr (REDUCE == ReductionType::MIN) {
            *value = std::min(*value, new_value);
        } else if constexpr (REDUCE == ReductionType::MAX) {
            *value = std::max(*value, new_value);
        }
    }
};

template <typename scalar_t,
          template <typename T, int32_t D>
          typename TensorAccessor,
          ReductionType REDUCE>
void
jaggedReduceHostCallback(int32_t tidx,
                         int32_t eidx,
                         const TensorAccessor<scalar_t, 2> data,
                         const TensorAccessor<fvdb::JIdxType, 1> jidx,
                         TensorAccessor<scalar_t, 2> out) {
    Reducer<scalar_t, REDUCE>::atomicWriteCPU(&out[jidx[tidx]][eidx], data[tidx][eidx]);
}

template <typename scalar_t,
          template <typename T, int32_t D>
          typename TensorAccessor,
          ReductionType REDUCE>
__device__ void
jaggedReduceDeviceCallback(int32_t tidx,
                           int32_t eidx,
                           const TensorAccessor<scalar_t, 2> data,
                           const TensorAccessor<fvdb::JIdxType, 1> jidx,
                           TensorAccessor<scalar_t, 2> out) {
    Reducer<scalar_t, REDUCE>::atomicWriteCUDA(&out[jidx[tidx]][eidx], data[tidx][eidx]);
}

template <typename scalar_t,
          template <typename T, int32_t D>
          typename TensorAccessor,
          c10::DeviceType DeviceTag>
__hostdev__ void
jaggedArgReduceCallback(int32_t tidx,
                        int32_t eidx,
                        const TensorAccessor<scalar_t, 2> data,
                        const TensorAccessor<fvdb::JIdxType, 1> jidx,
                        const TensorAccessor<fvdb::JOffsetsType, 1> joffsets,
                        const TensorAccessor<scalar_t, 2> out,
                        TensorAccessor<int64_t, 2> argOut) {
    const int64_t jidxVal    = jidx[tidx];
    const int64_t baseOffset = joffsets[jidxVal];
    const int64_t index      = tidx;
    const int64_t localIndex = index - baseOffset;
    if (data[tidx][eidx] == out[jidxVal][eidx]) {
        if constexpr (DeviceTag == torch::kCUDA) {
            atomMax(&argOut[jidxVal][eidx], localIndex);
        } else {
            argOut[jidxVal][eidx] = localIndex;
        }
    }
}

template <c10::DeviceType DeviceTag, ReductionType REDUCE>
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
JaggedReduce(const torch::Tensor &jdataRaw,
             const torch::Tensor &jidx,
             const torch::Tensor &joffsets,
             int64_t dimSize) {
    torch::Tensor jdata = featureCoalescedView(jdataRaw);

    static constexpr bool isMinMax = (REDUCE == ReductionType::MIN || REDUCE == ReductionType::MAX);

    torch::Tensor out                   = torch::empty({dimSize, jdata.size(1)}, jdata.options());
    std::optional<torch::Tensor> argOut = std::nullopt;
    if constexpr (isMinMax) {
        argOut = torch::empty({dimSize, jdata.size(1)}, jidx.options().dtype(torch::kLong));
    }

    auto jidxAccessor = tensorAccessor<DeviceTag, fvdb::JIdxType, 1>(jidx);

    auto joffsetsAccessor = tensorAccessor<DeviceTag, fvdb::JOffsetsType, 1>(joffsets);

    AT_DISPATCH_V2(
        jdata.scalar_type(),
        "JaggedReduce",
        AT_WRAP([&]() {
            out.fill_(Reducer<scalar_t, REDUCE>::init());
            auto outAccessor = tensorAccessor<DeviceTag, scalar_t, 2>(out);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb =
                    [=] __device__(int32_t ridx, int32_t cidx, TorchRAcc32<scalar_t, 2> dataAcc) {
                        jaggedReduceDeviceCallback<scalar_t, TorchRAcc32, REDUCE>(
                            ridx, cidx, dataAcc, jidxAccessor, outAccessor);
                    };
                forEachTensorElementChannelCUDA<scalar_t, 2>(256, jdata.size(1), jdata, cb);
            } else {
                auto cb = [=](int32_t ridx, int32_t cidx, TorchAcc<scalar_t, 2> dataAcc) {
                    jaggedReduceHostCallback<scalar_t, TorchAcc, REDUCE>(
                        ridx, cidx, dataAcc, jidxAccessor, outAccessor);
                };
                forEachTensorElementChannelCPU<scalar_t, 2>(jdata.size(1), jdata, cb);
            }

            // Fill empty slots with 0 instead of Inf/-Inf, and compute arguments.
            if constexpr (isMinMax) {
                out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
                argOut->fill_(-1);

                auto argOutAccessor = tensorAccessor<DeviceTag, int64_t, 2>(argOut.value());
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__(
                                  int32_t ridx, int32_t cidx, TorchRAcc32<scalar_t, 2> dataAcc) {
                        jaggedArgReduceCallback<scalar_t, TorchRAcc32, DeviceTag>(ridx,
                                                                                  cidx,
                                                                                  dataAcc,
                                                                                  jidxAccessor,
                                                                                  joffsetsAccessor,
                                                                                  outAccessor,
                                                                                  argOutAccessor);
                    };
                    forEachTensorElementChannelCUDA<scalar_t, 2>(256, jdata.size(1), jdata, cb);
                } else {
                    auto cb = [=](int32_t ridx, int32_t cidx, TorchAcc<scalar_t, 2> dataAcc) {
                        jaggedArgReduceCallback<scalar_t, TorchAcc, DeviceTag>(ridx,
                                                                               cidx,
                                                                               dataAcc,
                                                                               jidxAccessor,
                                                                               joffsetsAccessor,
                                                                               outAccessor,
                                                                               argOutAccessor);
                    };
                    forEachTensorElementChannelCPU<scalar_t, 2>(jdata.size(1), jdata, cb);
                }
            }
        }),
        AT_EXPAND(AT_ALL_TYPES),
        c10::kHalf);

    torch::Tensor rOut                   = out.reshape(spliceShape({out.size(0)}, jdataRaw));
    std::optional<torch::Tensor> rArgOut = std::nullopt;
    if constexpr (isMinMax) {
        rArgOut = argOut.value().reshape(spliceShape({out.size(0)}, jdataRaw));
    }

    return std::make_tuple(rOut, rArgOut);
}

template <>
torch::Tensor
dispatchJaggedSum<torch::kCPU>(const torch::Tensor &jdata,
                               const torch::Tensor &jidx,
                               const torch::Tensor &joffsets,
                               int64_t dimSize) {
    TORCH_CHECK(jdata.is_cpu(), "jagged tensor must be on CPU");
    return std::get<0>(
        JaggedReduce<torch::kCPU, ReductionType::SUM>(jdata, jidx, joffsets, dimSize));
}

template <>
torch::Tensor
dispatchJaggedSum<torch::kCUDA>(const torch::Tensor &jdata,
                                const torch::Tensor &jidx,
                                const torch::Tensor &joffsets,
                                int64_t dimSize) {
    TORCH_CHECK(jdata.is_cuda(), "jagged tensor must be on CUDA");
    return std::get<0>(
        JaggedReduce<torch::kCUDA, ReductionType::SUM>(jdata, jidx, joffsets, dimSize));
}

template <>
std::vector<torch::Tensor>
dispatchJaggedMin<torch::kCPU>(const torch::Tensor &jdata,
                               const torch::Tensor &jidx,
                               const torch::Tensor &joffsets,
                               int64_t dimSize) {
    TORCH_CHECK(jdata.is_cpu(), "jagged tensor must be on CPU");
    auto res = JaggedReduce<torch::kCPU, ReductionType::MIN>(jdata, jidx, joffsets, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

template <>
std::vector<torch::Tensor>
dispatchJaggedMin<torch::kCUDA>(const torch::Tensor &jdata,
                                const torch::Tensor &jidx,
                                const torch::Tensor &joffsets,
                                int64_t dimSize) {
    TORCH_CHECK(jdata.is_cuda(), "jagged tensor must be on CUDA");
    auto res = JaggedReduce<torch::kCUDA, ReductionType::MIN>(jdata, jidx, joffsets, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

template <>
std::vector<torch::Tensor>
dispatchJaggedMax<torch::kCPU>(const torch::Tensor &jdata,
                               const torch::Tensor &jidx,
                               const torch::Tensor &joffsets,
                               int64_t dimSize) {
    TORCH_CHECK(jdata.is_cpu(), "jagged tensor must be on CPU");
    auto res = JaggedReduce<torch::kCPU, ReductionType::MAX>(jdata, jidx, joffsets, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

template <>
std::vector<torch::Tensor>
dispatchJaggedMax<torch::kCUDA>(const torch::Tensor &jdata,
                                const torch::Tensor &jidx,
                                const torch::Tensor &joffsets,
                                int64_t dimSize) {
    TORCH_CHECK(jdata.is_cuda(), "jagged tensor must be on CUDA");
    auto res = JaggedReduce<torch::kCUDA, ReductionType::MAX>(jdata, jidx, joffsets, dimSize);
    return {std::get<0>(res), std::get<1>(res).value()};
}

} // namespace ops
} // namespace detail
} // namespace fvdb
