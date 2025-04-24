// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTYPES_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTYPES_CUH

#include "GaussianMacros.cuh"

#include <ATen/native/Math.h>

#include <cooperative_groups/reduce.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace fvdb {
namespace detail {
namespace ops {

// ----------------------------------------------------------------
// --- TODO: replace these with types from nvdb -------------------
// ----------------------------------------------------------------
template <typename T> using vec2   = glm::vec<2, T>;
template <typename T> using vec3   = glm::vec<3, T>;
template <typename T> using vec4   = glm::vec<4, T>;
template <typename T> using mat2   = glm::mat<2, 2, T>;
template <typename T> using mat3   = glm::mat<3, 3, T>;
template <typename T> using mat3x2 = glm::mat<3, 2, T>;

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

template <uint32_t DIM, class T, class WarpT>
inline __device__ void
warpSum(T *val, WarpT &warp) {
#pragma unroll DIM
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cooperative_groups::reduce(warp, val[i], cooperative_groups::plus<T>());
    }
}

template <class T, class WarpT>
inline __device__ void
warpSum(T *val, size_t dim, WarpT &warp) {
    for (uint32_t i = 0; i < dim; i++) {
        val[i] = cooperative_groups::reduce(warp, val[i], cooperative_groups::plus<T>());
    }
}

template <class WarpT, class ScalarT>
inline __device__ void
warpSum(ScalarT &val, WarpT &warp) {
    val = cooperative_groups::reduce(warp, val, cooperative_groups::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void
warpSum(vec4<ScalarT> &val, WarpT &warp) {
    val.x = cooperative_groups::reduce(warp, val.x, cooperative_groups::plus<ScalarT>());
    val.y = cooperative_groups::reduce(warp, val.y, cooperative_groups::plus<ScalarT>());
    val.z = cooperative_groups::reduce(warp, val.z, cooperative_groups::plus<ScalarT>());
    val.w = cooperative_groups::reduce(warp, val.w, cooperative_groups::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void
warpSum(vec3<ScalarT> &val, WarpT &warp) {
    val.x = cooperative_groups::reduce(warp, val.x, cooperative_groups::plus<ScalarT>());
    val.y = cooperative_groups::reduce(warp, val.y, cooperative_groups::plus<ScalarT>());
    val.z = cooperative_groups::reduce(warp, val.z, cooperative_groups::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void
warpSum(vec2<ScalarT> &val, WarpT &warp) {
    val.x = cooperative_groups::reduce(warp, val.x, cooperative_groups::plus<ScalarT>());
    val.y = cooperative_groups::reduce(warp, val.y, cooperative_groups::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void
warpSum(mat3<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void
warpSum(mat2<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANTYPES_CUH
