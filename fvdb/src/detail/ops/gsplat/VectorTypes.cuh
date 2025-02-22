// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_VECTORTYPES_CUH
#define FVDB_DETAIL_OPS_GSPLAT_VECTORTYPES_CUH

#include <ATen/native/Math.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.hpp>
#include <cuda_fp16.hpp>
#include <cuda_runtime.h>

#include <cstdint>

/*
Wrap 2D vector types for different scalar types
*/
template <typename Scalar> struct Vec2Type {};

template <> struct Vec2Type<int8_t> {
    using type = char2;
};

template <> struct Vec2Type<uint8_t> {
    using type = uchar2;
};

template <> struct Vec2Type<int16_t> {
    using type = short2;
};

template <> struct Vec2Type<uint16_t> {
    using type = ushort2;
};

template <> struct Vec2Type<int32_t> {
    using type = int2;
};

template <> struct Vec2Type<uint32_t> {
    using type = uint2;
};

template <> struct Vec2Type<int64_t> {
    using type = long2;
};

template <> struct Vec2Type<uint64_t> {
    using type = ulong2;
};

template <> struct Vec2Type<float> {
    using type = float2;
};

template <> struct Vec2Type<double> {
    using type = double2;
};

/*
Wrap 3D vector types for different scalar types
*/
template <typename Scalar> struct Vec3Type {};

template <> struct Vec3Type<int8_t> {
    using type = char3;
};

template <> struct Vec3Type<uint8_t> {
    using type = uchar3;
};

template <> struct Vec3Type<int16_t> {
    using type = short3;
};

template <> struct Vec3Type<uint16_t> {
    using type = ushort3;
};

template <> struct Vec3Type<int32_t> {
    using type = int3;
};

template <> struct Vec3Type<uint32_t> {
    using type = uint3;
};

template <> struct Vec3Type<int64_t> {
    using type = long3;
};

template <> struct Vec3Type<uint64_t> {
    using type = ulong3;
};

template <> struct Vec3Type<float> {
    using type = float3;
};

template <> struct Vec3Type<double> {
    using type = double3;
};

/*
Wrap scalar types, and upcast half precision to float32
*/
template <typename T> struct OpType {
    typedef T type;
};

template <> struct OpType<__nv_bfloat16> {
    typedef float type;
};

template <> struct OpType<__half> {
    typedef float type;
};

template <> struct OpType<c10::Half> {
    typedef float type;
};

template <> struct OpType<c10::BFloat16> {
    typedef float type;
};

namespace fvdb {

template <typename WarpT, typename ScalarT>
inline __device__ ScalarT
warpMax(const ScalarT &val, WarpT &warp) {
    return cooperative_groups::reduce(warp, val, cooperative_groups::greater<ScalarT>());
}

template <typename WarpT, typename ScalarT>
inline __device__ ScalarT
warpSum(const ScalarT &val, WarpT &warp) {
    return cooperative_groups::reduce(warp, val, cooperative_groups::plus<ScalarT>());
}

template <typename WarpT, typename ScalarT>
inline __device__ void
warpSumMut(ScalarT &val, WarpT &warp) {
    val = cooperative_groups::reduce(warp, val, cooperative_groups::plus<ScalarT>());
}

template <typename WarpT, typename ScalarT>
inline __device__ void
warpSumMut(typename Vec2Type<ScalarT>::type &val, WarpT &warp) {
    val.x = cooperative_groups::reduce(warp, val.x, cooperative_groups::plus<ScalarT>());
    val.y = cooperative_groups::reduce(warp, val.y, cooperative_groups::plus<ScalarT>());
}

template <typename WarpT, typename ScalarT>
inline __device__ void
warpSumMut(typename Vec3Type<ScalarT>::type &val, WarpT &warp) {
    val.x = cooperative_groups::reduce(warp, val.x, cooperative_groups::plus<ScalarT>());
    val.y = cooperative_groups::reduce(warp, val.y, cooperative_groups::plus<ScalarT>());
    val.z = cooperative_groups::reduce(warp, val.z, cooperative_groups::plus<ScalarT>());
}

template <uint32_t DIM, typename WarpT, typename ScalarT>
inline __device__ void
warpSumMut(ScalarT *val, WarpT &warp) {
#pragma unroll DIM
    for (uint32_t i = 0; i < DIM; i++) {
        warpSumMut<WarpT, ScalarT>(val[i], warp);
    }
}

} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_VECTORTYPES_CUH
