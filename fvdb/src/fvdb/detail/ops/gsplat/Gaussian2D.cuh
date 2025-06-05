// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_GAUSSIAN2D_CUH
#define FVDB_DETAIL_OPS_GSPLAT_GAUSSIAN2D_CUH

#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>

namespace fvdb::detail::ops {

template <typename ScalarType> struct alignas(32) Gaussian2D { // 28 bytes
    using vec2t = typename Vec2Type<ScalarType>::type;
    using vec3t = typename Vec3Type<ScalarType>::type;

    int32_t id;         // 4 bytes
    vec2t xy;           // 8 bytes
    ScalarType opacity; // 4 bytes
    vec3t conic;        // 12 bytes

    inline __device__ vec2t
    delta(const ScalarType px, const ScalarType py) const {
        return {xy.x - px, xy.y - py};
    }

    inline __device__ ScalarType
    sigma(const vec2t delta) const {
        return ScalarType{0.5} * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
               conic.y * delta.x * delta.y;
    }
};

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_OPS_GSPLAT_GAUSSIAN2D_CUH
