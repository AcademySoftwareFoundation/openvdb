// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIANWARPUTILS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIANWARPUTILS_CUH

#include "GaussianVectorTypes.cuh"

#include <nanovdb/math/Math.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace fvdb {
namespace detail {
namespace ops {

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
warpSum(nanovdb::math::Mat3<ScalarT> &val, WarpT &warp) {
    warpSum<3>(val[0], warp);
    warpSum<3>(val[1], warp);
    warpSum<3>(val[2], warp);
}

template <typename WarpT, typename ScalarT>
inline __device__ ScalarT
warpMax(const ScalarT &val, WarpT &warp) {
    return cooperative_groups::reduce(warp, val, cooperative_groups::greater<ScalarT>());
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

template <size_t DIM, typename WarpT, typename ScalarT>
inline __device__ void
warpSumMut(ScalarT *val, WarpT &warp) {
#pragma unroll DIM
    for (uint32_t i = 0; i < DIM; i++) {
        warpSumMut<WarpT, ScalarT>(val[i], warp);
    }
}

template <typename WarpT, typename ScalarT>
inline __device__ void
warpSumMut(ScalarT *val, size_t nDims, WarpT &warp) {
    for (uint32_t i = 0; i < nDims; i++) {
        warpSumMut<WarpT, ScalarT>(val[i], warp);
    }
}

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIANWARPUTILS_CUH
